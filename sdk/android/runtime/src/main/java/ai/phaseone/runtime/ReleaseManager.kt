package ai.phaseone.runtime

import android.content.Context
import android.util.AtomicFile
import com.google.gson.JsonObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URI
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs

class PhaseoneRuntime(
    context: Context,
    trust: CatalogTrust,
    val root: File = File(context.filesDir, "phaseone"),
    localDevelopment: Boolean = false,
    maxDownloadBytes: Long = 16L * 1024 * 1024 * 1024,
    permissions: PermissionPolicy? = null,
    subject: () -> String = { "local" },
) {
    val device: DeviceProfile = DeviceProfile.discover(context)
    val releases = ReleaseManager(context.applicationContext, root, device, trust, localDevelopment, maxDownloadBytes, permissions, subject)
    fun open(capability: String): NativeApplication = releases.open(capability)
}

/** Signed model-data delivery. This class never loads code from the catalog. */
class ReleaseManager internal constructor(
    private val context: Context,
    val root: File,
    private val device: DeviceProfile,
    private val trust: CatalogTrust,
    private val localDevelopment: Boolean,
    private val maxDownloadBytes: Long,
    private val permissions: PermissionPolicy? = null,
    private val subject: () -> String = { "local" },
) {
    init { root.mkdirs() }
    companion object { private val processLocks = ConcurrentHashMap<String, ReentrantLock>() }

    private fun <T> writeLocked(block: () -> T): T = processLocks.computeIfAbsent(root.canonicalPath) { ReentrantLock() }.withLock {
        RandomAccessFile(File(root, "write.lock"), "rw").use { file -> file.channel.lock().use { block() } }
    }

    private fun capabilityName(capability: String): String {
        if (!Regex("[A-Za-z][A-Za-z0-9_.-]{0,95}").matches(capability)) throw RuntimeFault("invalid_capability", "Invalid capability identifier")
        return capability
    }

    private fun atomic(file: File, value: JsonObject) {
        file.parentFile!!.mkdirs()
        val atomic = AtomicFile(file)
        val output = atomic.startWrite()
        try { output.write(canonical(value).toByteArray()); atomic.finishWrite(output) }
        catch (failure: Throwable) { atomic.failWrite(output); throw failure }
    }

    private fun readAtomic(file: File): JsonObject = AtomicFile(file).openRead().use { parseObject(it.readBytes()) }

    fun state(capability: String): JsonObject {
        val file = File(root, "active/${capabilityName(capability)}.json")
        return if (file.exists() || File(file.path + ".bak").exists()) readAtomic(file) else obj("active" to null, "previous" to null)
    }

    private fun active(state: JsonObject, field: String = "active"): JsonObject? = state[field]?.takeUnless { it.isJsonNull }?.asJsonObject

    private fun connection(url: String): HttpURLConnection {
        val uri = URI(url)
        val allowed = uri.scheme == "https" || (localDevelopment && uri.scheme == "http" && uri.host in listOf("127.0.0.1", "localhost", "::1"))
        if (!allowed || uri.userInfo != null || uri.fragment != null) throw RuntimeFault("invalid_source", "Use HTTPS, or explicit loopback development")
        return (uri.toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000; readTimeout = 120_000; instanceFollowRedirects = false
            if (responseCode != 200) { val code = responseCode; disconnect(); throw RuntimeFault("download_failed", "Model server returned HTTP $code") }
        }
    }

    private fun catalog(url: String): Pair<ModelCatalog, ByteArray> {
        val connection = connection(url)
        try {
            val bytes = connection.inputStream.use { input ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(16384)
                while (true) {
                    val size = input.read(buffer); if (size < 0) break
                    if (output.size() + size > 8 * 1024 * 1024) throw RuntimeFault("invalid_catalog", "Catalog exceeds its size limit")
                    output.write(buffer, 0, size)
                }
                output.toByteArray()
            }
            return decodeCatalog(bytes, trust, localDevelopment) to bytes
        } finally { connection.disconnect() }
    }

    fun check(url: String, capability: String, policy: ReleasePolicy): JsonObject =
        resolveCatalog(catalog(url).first, capability, device, policy, active(state(capability)))

    private fun cached(url: String, artifact: JsonObject): Pair<File, Long> {
        val digest = artifact.text("sha256")
        val expectedSize = artifact.long("size")
        val file = File(root, "objects/$digest")
        if (file.isFile && file.length() == expectedSize && sha256(file) == digest) return file to 0L
        if (expectedSize > maxDownloadBytes) throw RuntimeFault("download_limit", "Artifact exceeds configured download budget")
        file.parentFile!!.mkdirs()
        val temporary = File.createTempFile(".download-", ".partial", file.parentFile)
        val connection = connection(url)
        var count = 0L
        val hasher = MessageDigest.getInstance("SHA-256")
        try {
            connection.inputStream.use { input -> FileOutputStream(temporary).use { output ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val size = input.read(buffer); if (size < 0) break
                    count += size
                    if (count > expectedSize) throw RuntimeFault("artifact_integrity", "Artifact exceeds its declared size")
                    hasher.update(buffer, 0, size); output.write(buffer, 0, size)
                }
                output.fd.sync()
            } }
            val actual = hasher.digest().joinToString("") { "%02x".format(it) }
            if (count != expectedSize || actual != digest) throw RuntimeFault("artifact_integrity", "Artifact size or SHA-256 differs from the catalog")
            java.nio.file.Files.move(temporary.toPath(), file.toPath(), java.nio.file.StandardCopyOption.ATOMIC_MOVE, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            return file to count
        } finally { connection.disconnect(); temporary.delete() }
    }

    private fun projection(release: ModelRelease, variant: JsonObject): JsonObject {
        val manifest = release.manifest.deepCopy()
        val model = release.model.deepCopy()
        model.add("variants", json.toJsonTree(listOf(variant)))
        manifest.add("models", obj(release.modelName to model))
        manifest.add("artifacts", obj(*release.artifactIds(variant).map { it to release.artifacts[it] }.toTypedArray()))
        if (manifest.child("agent").text("model") != release.modelName) manifest.remove("agent")
        val metadata = manifest.child("metadata").deepCopy()
        metadata.addProperty("release", release.id); metadata.addProperty("contract", release.contract)
        metadata.addProperty("lineage", release.lineage); metadata.addProperty("source_release_digest", release.digest)
        manifest.add("metadata", metadata)
        return manifest
    }

    private fun application(directory: File, release: ModelRelease, variant: JsonObject): NativeApplication {
        val manifest = parseObject(inside(directory, "phaseone.json").readBytes())
        if (canonical(manifest, legacyNumbers = true) != canonical(projection(release, variant), legacyNumbers = true)) throw RuntimeFault("artifact_integrity", "Installed package differs from its signed catalog")
        for ((name, data) in manifest.child("artifacts").entrySet()) {
            val descriptor = data.asJsonObject
            val file = inside(directory, descriptor.text("path"))
            if (!file.isFile || file.length() != descriptor.long("size") || sha256(file) != descriptor.text("sha256")) throw RuntimeFault("artifact_integrity", "Installed artifact verification failed: $name")
        }
        return NativeApplication(context, directory, manifest, permissions, subject)
    }

    private fun fixture(application: NativeApplication, spec: JsonObject): TensorValue {
        if (spec.text("dtype", "float32") != "float32") throw RuntimeFault("unsupported_contract", "Qualification supports float32 tensors")
        val shape = spec.array("shape").map { it.asLong }.toLongArray()
        val count = shape.fold(1L) { a, b -> Math.multiplyExact(a, b) }
        if (shape.any { it <= 0 } || count > 16_000_000) throw RuntimeFault("invalid_catalog", "Invalid qualification tensor size")
        val file = application.artifact(spec.text("artifact"))
        if (file.length() != count * 4) throw RuntimeFault("invalid_catalog", "Qualification tensor byte size differs from its shape")
        val floats = FloatArray(count.toInt())
        ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        return TensorValue(shape, floats)
    }

    private fun qualify(application: NativeApplication, release: ModelRelease): List<JsonObject> {
        return release.checks.map { check ->
            if (check.text("kind") == "chat") {
                val messages = check.array("messages").map { value ->
                    val message = value.asJsonObject
                    ChatMessage(message.text("role"), message.text("content"), message.strings("images").map { application.artifact(it).readBytes() })
                }
                val result = application.chat(release.modelName, messages, Generation(maxOutputTokens = check.number("max_output_tokens", 64)))
                val output = result.message.content
                var passed = check.strings("contains").all { output.contains(it, ignoreCase = true) }
                if (check.child("json_fields").size() > 0 || check.child("json_choices").size() > 0) {
                    passed = passed && try {
                        val parsed = parseObject(output.trim().removePrefix("```json").removePrefix("```").removeSuffix("```").trim().toByteArray())
                        check.child("json_fields").entrySet().all { (key, expected) -> parsed[key] == expected } &&
                            check.child("json_choices").entrySet().all { (key, expected) -> expected.asJsonArray.any { it == parsed[key] } }
                    } catch (_: Exception) { false }
                }
                if (!passed) throw RuntimeFault("qualification_failed", "Chat task ${check.text("name")} failed: $output; active release is unchanged")
                return@map obj("name" to check.text("name"), "passed" to true, "output" to output, "execution" to result.execution)
            }
            val inputs = check.child("inputs").entrySet().associate { it.key to fixture(application, it.value.asJsonObject) }
            val result = application.infer(release.modelName, inputs, check.text("signature", "default"))
            val errors = check.child("outputs").entrySet().associate { (name, value) ->
                val expected = fixture(application, value.asJsonObject)
                val actual = result.outputs[name] ?: throw RuntimeFault("qualification_failed", "Native output is missing: $name")
                if (!expected.shape.contentEquals(actual.shape)) throw RuntimeFault("qualification_failed", "Native output shape differs from reference")
                name to actual.data.indices.maxOf { abs(actual.data[it].toDouble() - expected.data[it]) }
            }
            val tolerance = check.get("absolute_tolerance")?.asDouble ?: 0.001
            if (!tolerance.isFinite() || tolerance !in 0.0..1.0) throw RuntimeFault("invalid_catalog", "Invalid qualification tolerance")
            val passed = errors.values.all { it <= tolerance }
            if (!passed) throw RuntimeFault("qualification_failed", "Task check ${check.text("name")} failed: $errors; active release is unchanged")
            obj("name" to check.text("name"), "passed" to true, "max_absolute_error" to errors, "execution" to result.execution)
        }
    }

    fun update(url: String, capability: String, policy: ReleasePolicy, progress: (String) -> Unit = {}): JsonObject = writeLocked {
        progress("Checking model releases")
        val (catalog, catalogBytes) = catalog(url)
        val state = state(capability)
        val excluded = mutableSetOf<Pair<String, String>>()
        val failures = mutableListOf<JsonObject>()
        repeat(16) {
            val plan = resolveCatalog(catalog, capability, device, policy, active(state), excluded)
            if (plan.text("status") !in listOf("model_update_available", "requalification_required")) {
                if (failures.isNotEmpty()) {
                    if (plan.text("status") == "up_to_date") return@writeLocked obj("status" to "retained", "activation" to active(state), "fallback" to failures, "rejected" to plan["rejected"])
                    throw RuntimeFault("no_qualified_release", "No allowed candidate passed; active release is unchanged: $failures")
                }
                return@writeLocked plan
            }
            try { return@writeLocked activate(url, capability, catalog, catalogBytes, state, plan, failures, progress) }
            catch (failure: RuntimeFault) {
                if (!policy.allowFallback || failure.code in listOf("release_conflict", "untrusted_catalog", "invalid_catalog", "activation_failed")) throw failure
                val selected = plan.child("selection")
                failures.add(obj("release" to selected.text("release"), "variant" to selected.text("variant"), "error" to obj("code" to failure.code, "message" to failure.message)))
                excluded.add(selected.text("release") to selected.text("variant"))
            }
        }
        throw RuntimeFault("no_qualified_release", "Candidate attempt limit reached; active release is unchanged: $failures")
    }

    private fun activate(url: String, capability: String, catalog: ModelCatalog, catalogBytes: ByteArray, state: JsonObject, plan: JsonObject, failures: List<JsonObject>, progress: (String) -> Unit): JsonObject {
        val selected = plan.child("selection")
        var release = catalog.releases.first { it.id == selected.text("release") }
        val identity = File(root, "identities/${release.id}.json")
        if (identity.exists()) release = release.atInstalledDigest(readAtomic(identity).text("digest"), "release_conflict")
        val variant = release.variants.first { it.text("id") == selected.text("variant") }
        if (selected.long("download_bytes") > maxDownloadBytes) throw RuntimeFault("download_limit", "Selected dependency closure exceeds the download budget")
        val destination = File(root, "installed/${release.digest}/${variant.text("id")}")
        destination.parentFile!!.mkdirs()
        val staging = java.nio.file.Files.createTempDirectory(destination.parentFile!!.toPath(), ".stage-").toFile()
        var downloaded = 0L
        val receipts = mutableListOf<JsonObject>()
        try {
            for (name in selected.strings("artifacts")) {
                progress("Fetching $name")
                val artifact = release.artifacts.child(name)
                val relative = URI(null, null, release.base + artifact.text("path"), null).toASCIIString()
                val (cache, count) = cached(URI(url).resolve(relative).toString(), artifact)
                downloaded += count
                val target = inside(staging, artifact.text("path")); target.parentFile!!.mkdirs(); cache.copyTo(target)
                receipts.add(obj("id" to name, "sha256" to artifact.text("sha256"), "downloaded_bytes" to count))
            }
            atomic(File(staging, "phaseone.json"), projection(release, variant))
            progress("Running task checks on this device")
            val checks = application(staging, release, variant).use { qualify(it, release) }
            val record = selected.deepCopy()
            record.addProperty("digest", release.digest); record.addProperty("contract", release.contract); record.addProperty("lineage", release.lineage)
            record.addProperty("qualification", "passed"); record.add("checks", json.toJsonTree(checks)); record.addProperty("downloaded_bytes", downloaded)
            record.add("fallback", json.toJsonTree(failures))
            record.addProperty("catalog_signature", if (catalog.signed) "trusted_catalog" else "development_catalog")
            record.addProperty("qualification_fingerprint", qualificationFingerprint(device))
            record.add("artifact_receipts", json.toJsonTree(receipts)); record.addProperty("activated_at", System.currentTimeMillis() / 1000.0); record.add("device", device.toJson())
            FileOutputStream(File(staging, "catalog.json")).use { it.write(catalogBytes); it.fd.sync() }
            atomic(File(staging, "activation.json"), record)
            if (destination.exists()) application(destination, release, variant).close()
            else if (!staging.renameTo(destination)) throw RuntimeFault("activation_failed", "Atomic installation failed")
            atomic(identity, obj("digest" to release.digest))
            atomic(File(root, "active/${capabilityName(capability)}.json"), obj("active" to record, "previous" to if (plan.text("status") == "requalification_required") active(state, "previous") else active(state)))
            progress("Ready to run locally")
            return obj("status" to "activated", "activation" to record, "rejected" to plan["rejected"])
        } finally { if (staging.exists()) staging.deleteRecursively() }
    }

    private fun openRecord(record: JsonObject?): Triple<NativeApplication, ModelRelease, JsonObject> {
        if (record == null) throw RuntimeFault("no_active_release", "Install a qualified model release first")
        if (!Regex("[0-9a-f]{64}").matches(record.text("digest"))) throw RuntimeFault("artifact_integrity", "Invalid installed release identity")
        val directory = inside(root, "installed/${record.text("digest")}/${checkedPath(record.text("variant"))}")
        val catalog = decodeCatalog(inside(directory, "catalog.json").readBytes(), trust, localDevelopment)
        val entry = catalog.releases.find { it.id == record.text("release") } ?: throw RuntimeFault("artifact_integrity", "Release is missing from its catalog")
        val release = entry.atInstalledDigest(record.text("digest"), "artifact_integrity")
        val variant = release.variants.find { it.text("id") == record.text("variant") } ?: throw RuntimeFault("artifact_integrity", "Variant is missing from its release")
        val reasons = rejectionReasons(variant, device).toMutableList()
        if (versionCompare(device.phaseone_version, release.raw.text("minimum_runtime", "0.3.0")) < 0)
            reasons.add(obj("code" to "runtime", "message" to "PHASEONE SDK update required", "app_update" to true))
        if (reasons.isNotEmpty()) throw RuntimeFault("release_incompatible", "Installed release no longer matches this host: $reasons")
        return Triple(application(directory, release, variant), release, variant)
    }

    fun open(capability: String): NativeApplication {
        val record = active(state(capability))
        if (record != null && record.text("qualification_fingerprint") != qualificationFingerprint(device)) requalify(capability)
        return openRecord(active(state(capability))).first
    }

    fun requalify(capability: String): JsonObject = writeLocked {
        val state = state(capability)
        val current = active(state)
        val (app, release, _) = openRecord(current)
        val checks = app.use { qualify(it, release) }
        val record = current!!.deepCopy()
        record.add("checks", json.toJsonTree(checks)); record.add("device", device.toJson())
        record.addProperty("qualification_fingerprint", qualificationFingerprint(device))
        record.addProperty("activated_at", System.currentTimeMillis() / 1000.0)
        atomic(File(root, "active/${capabilityName(capability)}.json"), obj("active" to record, "previous" to active(state, "previous")))
        obj("status" to "requalified", "activation" to record)
    }

    fun rollback(capability: String): JsonObject = writeLocked {
        val state = state(capability)
        val previous = active(state, "previous") ?: throw RuntimeFault("no_previous_release", "No previously qualified release is available")
        val (app, release, _) = openRecord(previous)
        val checks = app.use { qualify(it, release) }
        val record = previous.deepCopy()
        record.addProperty("catalog_sequence", maxOf(previous.number("catalog_sequence", previous.number("sequence")), active(state)!!.number("catalog_sequence", active(state)!!.number("sequence"))))
        record.add("checks", json.toJsonTree(checks)); record.addProperty("activated_at", System.currentTimeMillis() / 1000.0)
        record.addProperty("qualification_fingerprint", qualificationFingerprint(device)); record.add("device", device.toJson())
        atomic(File(root, "active/${capabilityName(capability)}.json"), obj("active" to record, "previous" to active(state)))
        obj("status" to "rolled_back", "activation" to record)
    }
}

package ai.phaseone.runtime

import com.google.gson.JsonObject
import java.security.KeyFactory
import java.security.Signature
import java.security.spec.X509EncodedKeySpec
import java.util.Base64

data class CatalogTrust(val publicKeys: Map<String, ByteArray>) {
    companion object {
        fun key(publicKey: ByteArray): CatalogTrust {
            require(publicKey.size == 32)
            return CatalogTrust(mapOf(sha256(publicKey) to publicKey.copyOf()))
        }
    }
}

internal data class ModelRelease(val raw: JsonObject, private val storedDigest: String? = null) {
    val id = raw.text("id")
    val sequence = raw.number("sequence")
    val capability = raw.text("capability")
    val contract = raw.text("contract")
    val lineage = raw.text("lineage")
    val manifest = raw.child("manifest")
    val modelName = raw.text("model")
    val model = manifest.child("models").child(modelName)
    val variants = model.array("variants").map { it.asJsonObject }
    val artifacts = manifest.child("artifacts")
    val checks = raw.array("checks").map { it.asJsonObject }
    val base = checkedPath(raw.text("base").trimEnd('/')) + "/"
    private val canonicalDigest = sha256(canonical(raw).toByteArray())
    private val legacyDigest = sha256(canonical(raw, legacyNumbers = true).toByteArray())
    val digest = storedDigest ?: canonicalDigest

    fun atInstalledDigest(expected: String, errorCode: String): ModelRelease {
        if (expected !in setOf(canonicalDigest, legacyDigest)) throw RuntimeFault(errorCode, "Release data differs from its signed catalog")
        return if (expected == digest) this else ModelRelease(raw, expected)
    }

    init {
        if (digest !in setOf(canonicalDigest, legacyDigest)) throw RuntimeFault("invalid_catalog", "Invalid release identity representation")
        if (!Regex("[A-Za-z][A-Za-z0-9_.-]{0,95}").matches(id) || sequence < 1 || variants.isEmpty() || checks.isEmpty())
            throw RuntimeFault("invalid_catalog", "Invalid release identity, variants, or missing qualification")
        val paths = mutableSetOf<String>()
        for ((_, item) in artifacts.entrySet()) {
            val artifact = item.asJsonObject
            val path = checkedPath(artifact.text("path"))
            val lower = path.lowercase()
            if (listOf(".so", ".dll", ".dylib", ".dex", ".jar", ".aar", ".apk", ".py", ".js", ".exe", ".wasm").any { lower.endsWith(it) } || ".so." in lower)
                throw RuntimeFault("invalid_catalog", "A model catalog cannot deliver application or native code")
            if (!paths.add(lower) || lower in listOf("phaseone.json", "phaseone.signature.json", "catalog.json", "activation.json"))
                throw RuntimeFault("invalid_catalog", "Duplicate or reserved artifact path")
            if (!Regex("[0-9a-f]{64}").matches(artifact.text("sha256")) || artifact.long("size", -1) < 0)
                throw RuntimeFault("invalid_catalog", "Invalid artifact digest or size")
        }
        if (variants.map { it.text("id") }.toSet().size != variants.size) throw RuntimeFault("invalid_catalog", "Duplicate variant IDs")
        for (variant in variants) {
            val target = variant.child("target")
            if (target.text("accelerator", "cpu") != "cpu" && target.child("components").size() == 0)
                throw RuntimeFault("invalid_catalog", "Accelerated variants must declare native components")
            if (target.text("accelerator") == "npu" && target.optional("soc") == null)
                throw RuntimeFault("invalid_catalog", "NPU releases require an exact target SoC")
            artifactIds(variant).forEach { if (!artifacts.has(it)) throw RuntimeFault("invalid_catalog", "Undeclared artifact: $it") }
            for (check in checks) {
                if (check.text("kind") == "chat") {
                    if (variant.text("engine") != "llm" || check.array("messages").size() == 0 || (check.strings("contains").isEmpty() && check.child("json_fields").size() == 0 && check.child("json_choices").size() == 0))
                        throw RuntimeFault("invalid_catalog", "Chat qualification requires a chat model and concrete expectations")
                    if (check.array("messages").last().asJsonObject.text("role") != "user") throw RuntimeFault("invalid_catalog", "Chat qualification must end with a user message")
                    if (check.strings("contains").any { it.isBlank() }) throw RuntimeFault("invalid_catalog", "Empty chat expectations are not permitted")
                    continue
                }
                if (check.text("kind", "tensor") != "tensor") throw RuntimeFault("unsupported_contract", "Unsupported qualification kind")
                val signature = variant.child("signatures").child(check.text("signature", "default"))
                for (side in listOf("inputs", "outputs")) {
                    val fixture = check.child(side)
                    if (fixture.size() == 0 || signature.array(side).map { it.asJsonObject.text("name") }.toSet() != fixture.keySet())
                        throw RuntimeFault("invalid_catalog", "Qualification signatures must match every variant")
                    for (spec in signature.array(side).map { it.asJsonObject }) {
                        val item = fixture.child(spec.text("name"))
                        val shape = item.array("shape").map { it.asLong }
                        val expected = spec.array("shape").map { it.asLong }
                        if (shape.size !in 1..12 || shape.any { it <= 0 || it > 16_000_000 } || shape.size != expected.size || shape.zip(expected).any { (a, b) -> a != b && b != -1L } || item.text("dtype", "float32") != "float32" || spec.text("dtype") != "float32")
                            throw RuntimeFault("invalid_catalog", "Qualification tensor contracts must match every variant")
                        val count = shape.fold(1L) { a, b -> Math.multiplyExact(a, b) }
                        if (count > 16_000_000 || artifacts.child(item.text("artifact")).long("size") != count * 4)
                            throw RuntimeFault("invalid_catalog", "Qualification artifact size must match its float32 tensor shape")
                    }
                }
            }
        }
    }

    fun artifactIds(variant: JsonObject): List<String> {
        val ids = mutableSetOf(variant.text("artifact"))
        ids.addAll(variant.strings("supporting_artifacts"))
        variant.optional("data_artifact")?.let(ids::add)
        model.optional("labels")?.let(ids::add)
        val chat = variant.child("chat")
        for (name in listOf("tokenizer", "template", "tokenizer_config")) chat.optional(name)?.let(ids::add)
        chat.child("processor").optional("config")?.let(ids::add)
        for (check in checks) {
            if (check.text("kind") == "chat") for (message in check.array("messages")) ids.addAll(message.asJsonObject.strings("images"))
            else for (side in listOf("inputs", "outputs")) for (fixture in check.child(side).entrySet()) ids.add(fixture.value.asJsonObject.text("artifact"))
        }
        return ids.sorted()
    }
}

internal data class ModelCatalog(val raw: JsonObject, val releases: List<ModelRelease>, val signed: Boolean)

internal fun decodeCatalog(bytes: ByteArray, trust: CatalogTrust, allowUnsigned: Boolean): ModelCatalog {
    if (bytes.size > 8 * 1024 * 1024) throw RuntimeFault("invalid_catalog", "Catalog exceeds its size limit")
    try {
        var raw = parseObject(bytes)
        val signed = raw.text("format") == "phaseone.catalog.signed.v1"
        if (signed) {
            val key = trust.publicKeys[raw.text("key_id")] ?: throw RuntimeFault("untrusted_catalog", "Catalog key is not trusted by this app")
            if (sha256(key) != raw.text("key_id")) throw RuntimeFault("untrusted_catalog", "Signing key ID does not match")
            val payload = Base64.getDecoder().decode(raw.text("payload"))
            val prefix = byteArrayOf(0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00)
            val publicKey = KeyFactory.getInstance("Ed25519").generatePublic(X509EncodedKeySpec(prefix + key))
            val verifier = Signature.getInstance("Ed25519")
            verifier.initVerify(publicKey); verifier.update(payload)
            if (!verifier.verify(Base64.getDecoder().decode(raw.text("signature")))) throw RuntimeFault("invalid_catalog", "Catalog signature verification failed")
            raw = parseObject(payload)
        } else if (!allowUnsigned) throw RuntimeFault("untrusted_catalog", "A signed catalog is required")
        if (raw.number("schema_version") != 1) throw RuntimeFault("unsupported_contract", "Unsupported catalog schema")
        val releases = raw.array("releases").map { ModelRelease(it.asJsonObject) }
        if (releases.isEmpty() || releases.size > 10000 || releases.map { it.id }.toSet().size != releases.size || releases.map { it.capability to it.sequence }.toSet().size != releases.size)
            throw RuntimeFault("invalid_catalog", "Release IDs and capability sequence numbers must be unique")
        return ModelCatalog(raw, releases, signed)
    } catch (failure: RuntimeFault) { throw failure }
    catch (failure: Exception) { throw RuntimeFault("invalid_catalog", "Catalog validation failed", failure) }
}

internal fun rejectionReasons(variant: JsonObject, device: DeviceProfile): List<JsonObject> {
    val reasons = mutableListOf<JsonObject>()
    fun reject(code: String, message: String, appUpdate: Boolean = false) { reasons.add(obj("code" to code, "message" to message, "app_update" to appUpdate)) }
    val target = variant.child("target")
    if (target.strings("systems").isNotEmpty() && device.system !in target.strings("systems")) reject("os", "Operating system does not match")
    if (target.strings("architectures").isNotEmpty() && canonicalArch(device.architecture) !in target.strings("architectures").map(::canonicalArch)) reject("abi", "Native architecture does not match")
    if (target.optional("soc") != null && canonicalSoc(target.text("soc")) != canonicalSoc(device.soc)) reject("soc", "Exact SoC does not match")
    if (target.number("minimum_android_api") > device.android_api) reject("android_api", "Android API level is too old")
    if (target.long("minimum_memory_mb") > device.memory_mb) reject("memory", "Insufficient physical memory")
    if (target.strings("firmware").isNotEmpty() && device.firmware !in target.strings("firmware")) reject("firmware", "Firmware is not qualified for this variant")
    if (target.text("minimum_security_patch") > device.security_patch) reject("security_patch", "Security patch is too old")
    val backend = variant.text("backend")
    val engine = variant.text("engine")
    val distribution = if (backend == "executorch") "executorch" else if (engine == "llm") "litert-lm-api" else "ai-edge-litert"
    if (!versionMatches(device.runtimes[distribution], target.text("runtime_version"))) reject("engine", "Requires $distribution${target.text("runtime_version")}; installed ${device.runtimes[distribution]}", true)
    for ((component, requirement) in target.child("components").entrySet()) {
        if (!versionMatches(device.components[component], requirement.asString)) reject("component", "Requires $component${requirement.asString}; installed ${device.components[component]}", true)
    }
    if ("$backend/$engine" !in device.engines) reject("adapter", "This application does not include the required adapter", true)
    if (backend == "executorch" && engine == "llm") {
        if (variant.child("chat").child("processor").size() > 0 || variant.child("chat").text("prompt_format") !in listOf("chatml", "llama3"))
            reject("adapter", "Android ExecuTorch requires an installed text prompt profile; Python vision processors are not embedded", true)
    }
    return reasons
}

internal fun resolveCatalog(catalog: ModelCatalog, capability: String, device: DeviceProfile, policy: ReleasePolicy, active: JsonObject?, excluded: Set<Pair<String, String>> = emptySet()): JsonObject {
    val rejected = mutableListOf<JsonObject>()
    var chosen: JsonObject? = null
    var needsApp = false
    val catalogSequence = catalog.releases.filter { it.capability == capability && it.contract == policy.contract && it.lineage in policy.allowedLineages }.maxOfOrNull { it.sequence } ?: 0
    for (release in catalog.releases.filter { it.capability == capability }.sortedByDescending { it.sequence }) {
        for (variant in release.variants.sortedWith(compareByDescending<JsonObject> { it.number("priority") }.thenBy { it.text("id") })) {
            val reasons = rejectionReasons(variant, device).toMutableList()
            if ((release.id to variant.text("id")) in excluded) reasons.add(obj("code" to "qualification", "message" to "Candidate failed during this update", "app_update" to false))
            if (versionCompare(device.phaseone_version, release.raw.text("minimum_runtime", "0.3.0")) < 0)
                reasons.add(obj("code" to "runtime", "message" to "PHASEONE SDK update required", "app_update" to true))
            for ((failed, message) in listOf(
                (release.contract != policy.contract) to "Task contract does not match",
                (release.lineage !in policy.allowedLineages) to "Checkpoint lineage is not allowed by the app",
                (variant.child("target").text("accelerator", "cpu") !in policy.accelerators) to "Accelerator is not allowed by the app",
                (policy.backend != null && variant.text("backend") != policy.backend) to "Backend is not requested",
            )) if (failed) reasons.add(obj("code" to "policy", "message" to message, "app_update" to false))
            if (reasons.isNotEmpty()) {
                rejected.add(obj("release" to release.id, "variant" to variant.text("id"), "reasons" to reasons))
                needsApp = needsApp || reasons.all { it["app_update"].asBoolean }
            } else if (chosen == null) {
                val artifacts = release.artifactIds(variant)
                chosen = obj("release" to release.id, "sequence" to release.sequence, "variant" to variant.text("id"), "backend" to variant.text("backend"), "accelerator" to variant.child("target").text("accelerator", "cpu"), "artifacts" to artifacts, "download_bytes" to artifacts.sumOf { release.artifacts.child(it).long("size") }, "qualification" to "required", "model" to release.modelName)
                chosen.addProperty("catalog_sequence", catalogSequence)
            }
        }
    }
    var status = "no_compatible_release"
    if (chosen != null) {
        status = if (active != null && chosen.text("release") == active.text("release") && chosen.text("variant") == active.text("variant")) "up_to_date" else "model_update_available"
        if (status == "up_to_date" && active!!.text("qualification_fingerprint") != qualificationFingerprint(device)) status = "requalification_required"
        if (active != null && catalogSequence < active.number("catalog_sequence", active.number("sequence"))) { status = "stale_catalog"; chosen = null }
    }
    if (needsApp && status != "stale_catalog" && (chosen == null || status in listOf("up_to_date", "requalification_required"))) status = "app_update_required"
    return obj("status" to status, "selection" to chosen, "app_update_available" to needsApp, "rejected" to rejected)
}

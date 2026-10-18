package ai.phaseone.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.google.gson.GsonBuilder
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonNull
import com.google.gson.JsonPrimitive
import com.google.gson.Strictness
import com.google.gson.ToNumberPolicy
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.File
import java.io.StringReader
import java.math.BigDecimal
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.security.MessageDigest
import java.util.Locale

class RuntimeFault(val code: String, message: String, cause: Throwable? = null) : RuntimeException(message, cause)

internal val json = GsonBuilder().disableHtmlEscaping().create()
internal fun obj(vararg fields: Pair<String, Any?>): JsonObject = json.toJsonTree(mapOf(*fields)).asJsonObject
internal fun JsonObject.text(key: String, default: String = ""): String = get(key)?.takeUnless { it.isJsonNull }?.asString ?: default
internal fun JsonObject.number(key: String, default: Int = 0): Int = get(key)?.takeUnless { it.isJsonNull }?.asInt ?: default
internal fun JsonObject.long(key: String, default: Long = 0): Long = get(key)?.takeUnless { it.isJsonNull }?.asLong ?: default
internal fun JsonObject.child(key: String): JsonObject = get(key)?.takeUnless { it.isJsonNull }?.asJsonObject ?: JsonObject()
internal fun JsonObject.array(key: String): JsonArray = get(key)?.takeUnless { it.isJsonNull }?.asJsonArray ?: JsonArray()
internal fun JsonObject.strings(key: String): List<String> = array(key).map { it.asString }
internal fun JsonObject.optional(key: String): String? = text(key).ifEmpty { null }
internal fun parseObject(bytes: ByteArray): JsonObject {
    val decoded = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString()
    val reader = JsonReader(StringReader(decoded)).apply { strictness = Strictness.STRICT }
    fun read(depth: Int): JsonElement {
        if (depth > 64) throw RuntimeFault("invalid_json", "JSON nesting exceeds its limit")
        return when (reader.peek()) {
            JsonToken.BEGIN_OBJECT -> {
                val value = JsonObject(); reader.beginObject()
                while (reader.hasNext()) {
                    val name = reader.nextName()
                    if (value.has(name)) throw RuntimeFault("invalid_json", "Duplicate JSON key: $name")
                    value.add(name, read(depth + 1))
                }
                reader.endObject(); value
            }
            JsonToken.BEGIN_ARRAY -> {
                val value = JsonArray(); reader.beginArray()
                while (reader.hasNext()) value.add(read(depth + 1))
                reader.endArray(); value
            }
            JsonToken.STRING -> JsonPrimitive(reader.nextString())
            JsonToken.BOOLEAN -> JsonPrimitive(reader.nextBoolean())
            JsonToken.NULL -> { reader.nextNull(); JsonNull.INSTANCE }
            JsonToken.NUMBER -> {
                // Preserve numeric spelling: BigDecimal rewrites 3.3e-05 as 0.000033.
                val number = ToNumberPolicy.LAZILY_PARSED_NUMBER.readNumber(reader)
                if (!number.toDouble().isFinite()) throw RuntimeFault("invalid_json", "Non-finite JSON number")
                JsonPrimitive(number)
            }
            else -> throw RuntimeFault("invalid_json", "Invalid JSON token")
        }
    }
    reader.use {
        val result = read(0)
        if (reader.peek() != JsonToken.END_DOCUMENT || !result.isJsonObject) throw RuntimeFault("invalid_json", "Expected one JSON object")
        return result.asJsonObject
    }
}
internal fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
internal fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}
internal fun canonical(element: JsonElement, legacyNumbers: Boolean = false): String = when {
    element.isJsonObject -> element.asJsonObject.keySet().sorted().joinToString(",", "{", "}") { json.toJson(it) + ":" + canonical(element.asJsonObject[it], legacyNumbers) }
    element.isJsonArray -> element.asJsonArray.joinToString(",", "[", "]") { canonical(it, legacyNumbers) }
    legacyNumbers && element.isJsonPrimitive && element.asJsonPrimitive.isNumber -> BigDecimal(element.asString).toString()
    else -> json.toJson(element)
}
internal fun checkedPath(value: String): String {
    if (value.isEmpty() || value.startsWith('/') || '\\' in value || '\u0000' in value || Regex("^[A-Za-z]:").containsMatchIn(value) || value.split('/').any { it in listOf("", ".", "..") })
        throw RuntimeFault("unsafe_path", "Expected a normalized relative data path")
    return value
}
internal fun inside(root: File, path: String): File {
    val file = File(root, checkedPath(path))
    if (!file.canonicalPath.startsWith(root.canonicalPath + File.separator)) throw RuntimeFault("unsafe_path", "Data path escapes the store")
    var cursor = file
    while (cursor != root) {
        if (java.nio.file.Files.isSymbolicLink(cursor.toPath())) throw RuntimeFault("unsafe_path", "Model files cannot be symlinks")
        cursor = cursor.parentFile ?: break
    }
    return file
}

fun canonicalSoc(value: String?): String? {
    if (value.isNullOrBlank()) return null
    val key = value.lowercase(Locale.ROOT).replace(Regex("[^a-z0-9]"), "")
    return mapOf("tensorg5" to "googletensorg5", "tensorg6" to "googletensorg6", "mt6991" to "mediatekmt6991", "mt6989" to "mediatekmt6989", "qcs8275" to "qualcommqcs8275")[key] ?: key
}
internal fun canonicalArch(value: String): String = when (value.lowercase(Locale.ROOT)) {
    "arm64-v8a", "aarch64" -> "arm64"
    "amd64" -> "x86_64"
    else -> value.lowercase(Locale.ROOT)
}
internal fun versionCompare(a: String, b: String): Int {
    val left = a.split('.').map { it.toInt() }
    val right = b.split('.').map { it.toInt() }
    for (i in 0 until maxOf(left.size, right.size)) {
        val delta = (left.getOrElse(i) { 0 }).compareTo(right.getOrElse(i) { 0 })
        if (delta != 0) return delta
    }
    return 0
}
internal fun versionMatches(version: String?, requirement: String): Boolean {
    if (version == null) return false
    return requirement.isEmpty() || requirement.split(',').all { part ->
        val match = Regex("(==|!=|>=|<=|>|<)([0-9]+(?:\\.[0-9]+){0,2})").matchEntire(part)
            ?: throw RuntimeFault("invalid_catalog", "Unsupported version requirement: $part")
        val compared = versionCompare(version, match.groupValues[2])
        when (match.groupValues[1]) { "==" -> compared == 0; "!=" -> compared != 0; ">=" -> compared >= 0; "<=" -> compared <= 0; ">" -> compared > 0; else -> compared < 0 }
    }
}

data class DeviceProfile(
    val system: String = "android",
    val architecture: String,
    val soc: String?,
    val android_api: Int,
    val memory_mb: Long,
    val firmware: String,
    val security_patch: String,
    val manufacturer: String,
    val model: String,
    val runtimes: Map<String, String>,
    val components: Map<String, String>,
    val engines: List<String>,
    val phaseone_version: String = "0.3.0",
    val location: String = "android_local",
    val runtime_build: String = BuildConfig.RUNTIME_BUILD_ID,
) {
    fun toJson(): JsonObject = json.toJsonTree(this).asJsonObject
    companion object {
        fun discover(context: Context): DeviceProfile {
            val memory = ActivityManager.MemoryInfo()
            (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
            return DeviceProfile(
                architecture = canonicalArch(Build.SUPPORTED_ABIS.first()), soc = Build.SOC_MODEL,
                android_api = Build.VERSION.SDK_INT, memory_mb = memory.totalMem / (1024 * 1024),
                firmware = Build.FINGERPRINT, security_patch = Build.VERSION.SECURITY_PATCH,
                manufacturer = Build.MANUFACTURER, model = Build.MODEL,
                runtimes = mapOf("executorch" to "1.5.1", "ai-edge-litert" to "2.3.0", "litert-lm-api" to "0.18.0"),
                components = buildMap {
                    put("executorch.xnnpack", "1.5.1")
                    // Presence means packaged native code, not proven acceleration.
                    putAll(VendorLibraries.installedComponents(context))
                },
                engines = listOf("executorch/program", "executorch/llm", "litert/compiled", "litert/llm"),
            )
        }
    }
}

internal fun qualificationFingerprint(device: DeviceProfile): String = sha256(canonical(obj(
    "system" to device.system, "firmware" to device.firmware, "security_patch" to device.security_patch,
    "phaseone_version" to device.phaseone_version, "runtime_build" to device.runtime_build,
    "architecture" to canonicalArch(device.architecture), "soc" to (canonicalSoc(device.soc) ?: ""),
    "android_api" to device.android_api, "runtimes" to device.runtimes, "components" to device.components,
)).toByteArray())

data class ReleasePolicy(
    val contract: String,
    val allowedLineages: List<String>,
    val accelerators: List<String> = listOf("cpu"),
    val backend: String? = null,
    val allowFallback: Boolean = false,
) {
    init { require(allowedLineages.isNotEmpty()); require(accelerators.isNotEmpty() && accelerators.all { it in listOf("cpu", "gpu", "npu") }) }
}

data class TensorValue(val shape: LongArray, val data: FloatArray, val dtype: String = "float32") {
    init {
        if (dtype != "float32" || shape.any { it <= 0 } || shape.fold(1L) { a, b -> Math.multiplyExact(a, b) } != data.size.toLong() || data.any { !it.isFinite() })
            throw RuntimeFault("invalid_tensor", "Expected a finite float32 tensor with matching shape")
    }
}
data class Execution(
    val backend: String,
    val engine: String,
    val variant: String,
    val requested_accelerator: String,
    val accelerator_observed: String?,
    val inference_ms: Double,
    val load_ms: Double,
    val location: String = "android_local",
    val execution_verified: Boolean = true,
    val npu_execution_verified: Boolean = false,
    val native_api: String,
    val artifact_sha256: String,
)
data class InferenceResult(val outputs: Map<String, TensorValue>, val execution: Execution)
data class Prediction(val index: Int, val label: String, val score: Double)
data class ClassificationResult(val predictions: List<Prediction>, val execution: Execution)

data class ChatMessage(val role: String, val content: String, val images: List<ByteArray> = emptyList(),
                       val toolCallId: String? = null, val name: String? = null, val toolCalls: List<ToolCall> = emptyList()) {
    init { require(role in listOf("system", "user", "assistant", "tool")) }
}
data class Generation(val maxOutputTokens: Int = 128, val temperature: Double = 0.0) {
    init { require(maxOutputTokens in 1..32768); require(temperature.isFinite() && temperature in 0.0..2.0) }
}
data class ChatResult(val message: ChatMessage, val execution: Execution, val toolCalls: List<ToolCall> = emptyList())

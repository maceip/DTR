package ai.phaseone.runtime

import com.google.gson.JsonObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class ToolContext(val actionId: String, val packageDigest: String, val subject: String)

/** App-private, crash-conservative action records. A pending action never replays. */
internal class ToolActions(private val root: File) {
    companion object { private val locks = ConcurrentHashMap<String, ReentrantLock>() }
    init { root.mkdirs() }
    private fun <T> locked(block: () -> T): T = locks.computeIfAbsent(root.canonicalPath) { ReentrantLock() }.withLock {
        RandomAccessFile(File(root, "actions.lock"), "rw").use { it.channel.lock().use { block() } }
    }
    private fun file(id: String) = File(root, sha256(id.toByteArray()) + ".json")
    private fun read(file: File): JsonObject? {
        if (!file.exists()) return null
        if (file.length() > 2 * 1024 * 1024) throw RuntimeFault("action_conflict", "Action record exceeds its limit")
        return parseObject(file.readBytes())
    }
    private fun write(file: File, value: JsonObject) {
        val temporary = File.createTempFile("action-", ".pending", root)
        try {
            FileOutputStream(temporary).use { it.write(canonical(value).toByteArray()); it.fd.sync() }
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            // Atomic rename alone is not durable across a power loss.
            java.nio.channels.FileChannel.open(root.toPath(), java.nio.file.StandardOpenOption.READ).use { it.force(true) }
        } finally { temporary.delete() }
    }
    fun begin(id: String, fingerprint: String): JsonObject? = locked {
        val path = file(id)
        val previous = read(path)
        if (previous != null) {
            if (previous.text("fingerprint") != fingerprint) throw RuntimeFault("idempotency_conflict", "Action ID was used for another request")
            if (!previous.has("result") || previous.child("result").text("status") == "uncertain") throw RuntimeFault("action_outcome_unknown", "Inspect the previous action; automatic replay is blocked")
            previous.child("result")
        } else {
            write(path, obj("fingerprint" to fingerprint)); null
        }
    }
    fun finish(id: String, result: JsonObject) = locked {
        val path = file(id)
        val entry = read(path) ?: throw RuntimeFault("action_conflict", "Action is not pending")
        if (entry.has("result")) throw RuntimeFault("action_conflict", "Action is already complete")
        entry.add("result", result); write(path, entry)
    }
}

/** Registered handlers share one resource gate, regardless of inference engine. */
class ToolGateway(
    manifest: JsonObject,
    private val policy: PermissionPolicy,
    private val subject: () -> String,
    stateDirectory: File,
) {
    private val manifest = manifest.deepCopy()
    private data class Binding(
        val definition: ToolDefinition, val schema: FlatToolSchema,
        val access: (JsonObject) -> List<ResourceAccess>,
        val handler: (JsonObject, ToolContext) -> JsonObject,
    )
    private val packageDigest = sha256(canonical(manifest).toByteArray())
    private val actions = ToolActions(stateDirectory)
    private val bindings = ConcurrentHashMap<String, Binding>()

    fun register(definition: ToolDefinition, access: (JsonObject) -> List<ResourceAccess>, handler: (JsonObject, ToolContext) -> JsonObject) {
        val owned = definition.copy(parameters = definition.parameters.deepCopy())
        val binding = Binding(owned, FlatToolSchema(owned.parameters), access, handler)
        if (bindings.putIfAbsent(owned.name, binding) != null) throw RuntimeFault("tool_exists", "Tool is already registered")
    }

    fun invoke(call: ToolCall, actionId: String): JsonObject {
        if (actionId.length !in 1..256) throw RuntimeFault("invalid_action_id", "Provide an action ID of 1 to 256 characters")
        val encoded = canonical(call.arguments).toByteArray()
        if (encoded.size > 1024 * 1024) throw RuntimeFault("invalid_tool_arguments", "Arguments exceed 1 MiB")
        val arguments = parseObject(encoded)
        val user = subject()
        val request = obj("package" to packageDigest, "subject" to user, "tool" to call.name, "arguments" to arguments)
        val cached = actions.begin(actionId, sha256(canonical(request).toByteArray()))
        var entered = false
        var binding: Binding? = null
        val result: JsonObject = try {
            val declared = manifest.array("tools").map { it.asJsonObject }.singleOrNull { it.text("name") == call.name }
                ?: throw RuntimeFault("tool_not_declared", "Application does not declare this tool")
            val registered = bindings[call.name] ?: throw RuntimeFault("tool_unavailable", "Application has no implementation of this tool")
            binding = registered
            if (declared.text("permission") != registered.definition.permission || (declared["side_effect"]?.asBoolean ?: false) != registered.definition.sideEffect)
                throw RuntimeFault("tool_contract", "Package cannot weaken the host tool contract")
            registered.schema.validate(arguments)
            FlatToolSchema(declared.child("parameters")).validate(arguments)
            val requirements = registered.access(arguments.deepCopy()).toList()
            policy.authorize(user, requirements) {
                if (subject() != user) throw RuntimeFault("permission_denied", "Signed-in user changed before tool execution")
                if (cached != null) cached.deepCopy().apply { addProperty("cached", true) }
                else {
                    entered = true
                    val value = registered.handler(arguments, ToolContext(actionId, packageDigest, user))
                    val bytes = canonical(value).toByteArray()
                    if (bytes.size > 1024 * 1024) throw RuntimeFault("tool_output_too_large", "Tool result exceeds 1 MiB")
                    obj("action_id" to actionId, "tool" to call.name, "status" to "complete", "cached" to false, "result" to parseObject(bytes))
                }
            }
        } catch (failure: Exception) {
            val fault = failure as? RuntimeFault ?: RuntimeFault("tool_failed", "Tool handler failed")
            obj("action_id" to actionId, "tool" to call.name, "cached" to false,
                "status" to if (!entered) "denied" else if (binding?.definition?.sideEffect == true) "uncertain" else "failed",
                "error" to obj("code" to fault.code, "message" to fault.message))
        }
        if (cached == null) actions.finish(actionId, result)
        return result
    }
}

package ai.phaseone.runtime

import com.google.gson.JsonObject

private val permissionIdentity = Regex("[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}")
private val permissionActions = setOf("read", "write", "execute", "export", "train")

data class ResourceAccess(val resource: String, val action: String) {
    init {
        if (resource.isEmpty() || resource.codePointCount(0, resource.length) > 1024 || Regex("[\\x00-\\x1f\\x7f*?]").containsMatchIn(resource) || action !in permissionActions)
            throw RuntimeFault("invalid_policy", "Invalid resource operation")
    }
}

/** Exact resource grants. Supplied by app authentication/consent, never a model. */
class PermissionPolicy(document: JsonObject) {
    private data class Rule(val subject: String, val access: ResourceAccess, val effect: String)
    private var rules = emptyList<Rule>()
    var revision: Long = 0; private set
    init { replace(document) }

    @Synchronized fun replace(document: JsonObject) {
        try {
            if (document.keySet() != setOf("version", "rules") || !document["version"].isJsonPrimitive || !document["version"].asJsonPrimitive.isNumber || document["version"].asDouble != 1.0 || !document["rules"].isJsonArray || document.array("rules").size() > 4096)
                throw RuntimeFault("invalid_policy", "Invalid permission document")
            val next = document.array("rules").map { value ->
                val rule = value.asJsonObject
                if (rule.keySet() != setOf("subject", "resource", "action", "effect") || rule.entrySet().any { !it.value.isJsonPrimitive || !it.value.asJsonPrimitive.isString })
                    throw RuntimeFault("invalid_policy", "Invalid permission rule")
                if (!permissionIdentity.matches(rule.text("subject")) || rule.text("effect") !in setOf("allow", "deny"))
                    throw RuntimeFault("invalid_policy", "Invalid subject or effect")
                Rule(rule.text("subject"), ResourceAccess(rule.text("resource"), rule.text("action")), rule.text("effect"))
            }
            rules = next; revision++
        } catch (fault: RuntimeFault) { throw fault }
        catch (failure: Exception) { throw RuntimeFault("invalid_policy", "Invalid permission document", failure) }
    }

    @Synchronized fun allows(subject: String, access: ResourceAccess): Boolean {
        val matches = rules.filter { it.subject == subject && it.access == access }
        return matches.any { it.effect == "allow" } && matches.none { it.effect == "deny" }
    }

    /** Serializes policy replacement against entry; cannot undo work already started. */
    @Synchronized fun <T> authorize(subject: String, access: List<ResourceAccess>, operation: () -> T): T {
        if (access.isEmpty() || access.size > 128) throw RuntimeFault("tool_scope_missing", "Declare 1 to 128 resource operations")
        if (access.any { !allows(subject, it) }) throw RuntimeFault("permission_denied", "Resource operation is not permitted")
        return operation()
    }
}

data class FineTuneRequest(
    val model: String, val dataset: String,
    val objective: String = "function_calling", val method: String = "lora",
)

internal fun fineTuneStub(request: FineTuneRequest, policy: PermissionPolicy?, subject: String): Nothing {
    if (request.objective !in setOf("function_calling", "chat") || request.method != "lora")
        throw RuntimeFault("invalid_training_request", "Unsupported fine-tuning contract")
    if (policy == null) throw RuntimeFault("permission_denied", "Fine-tuning requires explicit dataset permissions")
    return policy.authorize(subject, listOf(ResourceAccess(request.dataset, "read"), ResourceAccess(request.dataset, "train"))) {
        throw RuntimeFault("fine_tuning_unavailable", "Fine-tuning is a contract stub; no trainer is installed. No data was read or uploaded.")
    }
}

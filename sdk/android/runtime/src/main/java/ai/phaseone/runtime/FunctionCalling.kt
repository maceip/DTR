package ai.phaseone.runtime

import com.google.gson.JsonObject
import java.util.UUID

data class ToolCall(val id: String, val name: String, val arguments: JsonObject)
data class ToolDefinition(val name: String, val description: String, val parameters: JsonObject, val permission: String, val sideEffect: Boolean = false) {
    internal fun wire() = obj("name" to name, "description" to description, "parameters" to parameters)
}

/** Portable JSON tool protocol. Calls are proposals, never executable code. */
internal object FunctionCalling {
    fun prepare(messages: List<ChatMessage>, tools: List<ToolDefinition>, protocol: String): List<ChatMessage> {
        if (tools.isEmpty()) return messages
        if (protocol != "json") throw RuntimeFault("unsupported_tools", "Android tool-schema adapter currently implements the JSON protocol only")
        if (tools.size > 32 || tools.map { it.name }.toSet().size != tools.size) throw RuntimeFault("invalid_tool_schema", "Expected up to 32 unique tools")
        tools.forEach { FlatToolSchema(it.parameters) }
        val prompt = "When you need a tool, reply only with JSON: {\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}. The host executes tools and returns results. Otherwise answer normally. Available tools: " + json.toJson(tools.map { it.wire() })
        val prepared = messages.map { message ->
            when {
                message.role == "tool" -> message.copy(role = "user", content = json.toJson(obj("tool" to message.name, "tool_call_id" to message.toolCallId, "result" to message.content)))
                message.toolCalls.isNotEmpty() -> message.copy(content = json.toJson(obj("tool_calls" to message.toolCalls)))
                else -> message
            }
        }.toMutableList()
        if (prepared.firstOrNull()?.role == "system") prepared[0] = prepared[0].copy(content = prepared[0].content + "\n" + prompt)
        else prepared.add(0, ChatMessage("system", prompt))
        return prepared
    }

    fun parse(text: String, tools: List<ToolDefinition>): List<ToolCall> {
        if (tools.isEmpty()) return emptyList()
        if (text.toByteArray().size > 1024 * 1024) throw RuntimeFault("invalid_tool_call", "Tool response exceeds 1 MiB")
        val candidate = text.trim().let { if (it.startsWith("```json\n") && it.endsWith("```")) it.substring(8, it.length - 3).trim() else it }
        val envelope = try { parseObject(candidate.toByteArray()) } catch (_: Exception) {
            // A response trying to use the envelope must not hide malformed or
            // duplicate-key JSON behind the ordinary-text fallback.
            if (candidate.startsWith('{') && candidate.contains("\"tool_calls\"")) throw RuntimeFault("invalid_tool_call", "Malformed tool response")
            return emptyList()
        }
        if (!envelope.has("tool_calls")) return emptyList()
        if (!envelope["tool_calls"].isJsonArray) throw RuntimeFault("invalid_tool_call", "Expected an array of tool proposals")
        try {
            val calls = envelope.array("tool_calls")
            if (calls.size() > 16) throw RuntimeFault("invalid_tool_call", "Too many tool calls")
            return calls.map { value ->
                val call = value.asJsonObject
                val name = call.text("name")
                if (tools.none { it.name == name } || !call.has("arguments") || !call["arguments"].isJsonObject)
                    throw RuntimeFault("invalid_tool_call", "Unknown tool or invalid arguments")
                val id = call.optional("id") ?: UUID.randomUUID().toString()
                if (id.length !in 1..128) throw RuntimeFault("invalid_tool_call", "Invalid call identifier")
                ToolCall(id, name, call.child("arguments").deepCopy())
            }
        } catch (fault: RuntimeFault) { throw fault }
        catch (failure: Exception) { throw RuntimeFault("invalid_tool_call", "Invalid tool envelope", failure) }
    }
}

/** Deliberately bounded first Android schema adapter; unsupported keywords fail. */
internal class FlatToolSchema(private val schema: JsonObject) {
    private val properties = schema.child("properties")
    init {
        val allowed = setOf("type", "properties", "required", "additionalProperties", "description", "title", "\$schema")
        if (schema.keySet().any { it !in allowed } || schema.text("type") != "object" || !schema.has("additionalProperties") || schema["additionalProperties"].asBoolean)
            throw RuntimeFault("unsupported_tool_schema", "Android tools require a closed flat object schema")
        if (properties.size() > 64 || schema.strings("required").any { !properties.has(it) }) throw RuntimeFault("invalid_tool_schema", "Invalid required properties")
        for ((_, value) in properties.entrySet()) {
            val property = value.asJsonObject
            if (property.keySet().any { it !in setOf("type", "description", "title", "enum", "minLength", "maxLength") } || property.text("type") !in setOf("string", "number", "integer", "boolean"))
                throw RuntimeFault("unsupported_tool_schema", "Android tools currently support flat primitive parameters")
            if ((property.has("minLength") || property.has("maxLength")) && property.text("type") != "string") throw RuntimeFault("invalid_tool_schema", "String bounds require a string")
        }
    }
    fun validate(arguments: JsonObject) {
        fun invalid(): Nothing = throw RuntimeFault("invalid_tool_arguments", "Arguments do not satisfy the tool schema")
        if (arguments.keySet().any { !properties.has(it) } || schema.strings("required").any { !arguments.has(it) }) invalid()
        for ((key, value) in arguments.entrySet()) {
            val property = properties.child(key)
            if (!value.isJsonPrimitive) invalid()
            val primitive = value.asJsonPrimitive
            val valid = when (property.text("type")) {
                "string" -> primitive.isString && primitive.asString.codePointCount(0, primitive.asString.length) in property.number("minLength", 0)..property.number("maxLength", Int.MAX_VALUE)
                "boolean" -> primitive.isBoolean
                "number" -> primitive.isNumber && primitive.asDouble.isFinite()
                "integer" -> primitive.isNumber && primitive.asDouble.isFinite() && primitive.asBigDecimal.stripTrailingZeros().scale() <= 0
                else -> false
            }
            if (!valid || (property.has("enum") && property.array("enum").none { it == value })) invalid()
        }
    }
}

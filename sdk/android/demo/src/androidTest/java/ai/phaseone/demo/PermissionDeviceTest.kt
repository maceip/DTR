package ai.phaseone.demo

import ai.phaseone.runtime.PermissionPolicy
import ai.phaseone.runtime.ResourceAccess
import ai.phaseone.runtime.ToolCall
import ai.phaseone.runtime.ToolDefinition
import ai.phaseone.runtime.ToolGateway
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Scripted proposals exercise the real embedded gateway, not LLM quality. */
@RunWith(AndroidJUnit4::class)
class PermissionDeviceTest {
    @Test fun appPrivateDataAndCodePermissions() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val root = File(context.filesDir, "permission-test").apply { mkdirs() }
        fun parse(text: String) = JsonParser.parseString(text).asJsonObject
        val schema = parse("""{"type":"object","properties":{},"additionalProperties":false}""")
        val definitions = listOf("read_note" to "read", "write_patch" to "write", "run_patch" to "execute", "upload_note" to "export")
            .map { (name, action) -> ToolDefinition(name, "Permission test", schema, "demo.$action", action != "read") }
        val manifest = parse("""{"tools":[]} """)
        for (definition in definitions) manifest["tools"].asJsonArray.add(parse("""{"name":"${definition.name}","description":"Permission test","parameters":$schema,"permission":"${definition.permission}","side_effect":${definition.sideEffect}}"""))
        val policy = PermissionPolicy(parse("""{"version":1,"rules":[
          {"subject":"alice","resource":"note:private","action":"read","effect":"allow"},
          {"subject":"alice","resource":"code:patch","action":"write","effect":"allow"}
        ]}"""))
        val gateway = ToolGateway(manifest, policy, { "alice" }, File(root, "actions"))
        val patch = File(root, "patch.txt")
        for (definition in definitions) {
            val action = definition.permission.substringAfter('.')
            gateway.register(definition, { listOf(ResourceAccess(if (action in listOf("write", "execute")) "code:patch" else "note:private", action)) }) { _, _ ->
                when (action) {
                    "read" -> parse("""{"text":"private test note"}""")
                    "write" -> { patch.writeText("generated test code; not executed"); parse("""{"saved":true}""") }
                    else -> throw AssertionError("A forbidden handler was entered")
                }
            }
        }
        val results = JsonObject()
        // Unique IDs keep this repeatable without erasing previous evidence.
        val run = java.util.UUID.randomUUID().toString()
        for (definition in definitions) results.add(definition.name, gateway.invoke(ToolCall("model-proposal", definition.name, JsonObject()), "$run:${definition.name}"))
        assertEquals("complete", results["read_note"].asJsonObject["status"].asString)
        assertEquals("complete", results["write_patch"].asJsonObject["status"].asString)
        assertTrue(patch.readText().startsWith("generated"))
        assertEquals("denied", results["run_patch"].asJsonObject["status"].asString)
        assertEquals("denied", results["upload_note"].asJsonObject["status"].asString)
        policy.replace(parse("""{"version":1,"rules":[]}"""))
        val revoked = gateway.invoke(ToolCall("model-proposal", "read_note", JsonObject()), "$run:read_note")
        assertEquals("denied", revoked["status"].asString); assertFalse(revoked.has("result"))
        results.add("revoked_cached_read", revoked)
        results.addProperty("source", "scripted_tool_proposals")
        results.addProperty("model_inference_tested", false)
        results.addProperty("passed", true)
        File(root, "receipt.json").writeText(results.toString())
    }
}

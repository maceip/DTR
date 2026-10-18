package ai.phaseone.runtime

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files

class PermissionContractTest {
    private val fixture = parseObject(File("../../contracts/permissions-cases.json").readBytes())
    private val schema = parseObject("""{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}""".toByteArray())
    private val definition = ToolDefinition("read_note", "Read an app-owned note", schema, "notes.read")
    private fun manifest(effect: Boolean = false) = obj("tools" to listOf(obj("name" to definition.name, "description" to definition.description, "parameters" to schema, "permission" to definition.permission, "side_effect" to effect)))
    private fun policy() = PermissionPolicy(fixture.child("policy"))

    @Test fun sameDecisionsAsPythonAndPermix() {
        val policy = policy()
        for (value in fixture.array("cases")) {
            val case = value.asJsonObject
            assertEquals(case.text("name"), case["allowed"].asBoolean, policy.allows(case.text("subject"), ResourceAccess(case.text("resource"), case.text("action"))))
        }
        for (document in fixture.array("invalid_documents")) {
            try { PermissionPolicy(document.asJsonObject); fail("Invalid policy accepted: $document") }
            catch (fault: RuntimeFault) { assertEquals("invalid_policy", fault.code) }
        }
    }

    @Test fun checkedHandlerAndRevokedCachedData() {
        val root = Files.createTempDirectory("dtr-tool-test").toFile()
        try {
            val policy = policy(); var effects = 0
            fun open(): ToolGateway = ToolGateway(manifest(), policy, { "alice" }, root).apply {
                register(definition, { listOf(ResourceAccess("notes:" + it.text("id"), "read")) }) { _, _ -> effects++; obj("text" to "private note") }
            }
            val gateway = open()
            val own = ToolCall("model-id", "read_note", obj("id" to "alice"))
            assertEquals("complete", gateway.invoke(own, "read").text("status"))
            assertTrue(open().invoke(own, "read")["cached"].asBoolean)
            assertEquals("permission_denied", gateway.invoke(own.copy(arguments = obj("id" to "bob")), "other").child("error").text("code"))
            assertEquals("invalid_tool_arguments", gateway.invoke(own.copy(arguments = obj("id" to "alice", "subject" to "admin")), "spoof").child("error").text("code"))
            policy.replace(obj("version" to 1, "rules" to emptyList<Any>()))
            val revoked = gateway.invoke(own, "read")
            assertEquals("denied", revoked.text("status")); assertFalse(revoked.has("result"))
            assertEquals(1, effects)
        } finally { root.deleteRecursively() }
    }

    @Test fun effectsCannotReplayAfterFailureOrArgumentChange() {
        val root = Files.createTempDirectory("dtr-effect-test").toFile()
        try {
            var effects = 0
            val gateway = ToolGateway(manifest(true), policy(), { "alice" }, root)
            gateway.register(definition.copy(sideEffect = true), { listOf(ResourceAccess("notes:alice", "read")) }) { _, _ -> effects++; throw IllegalStateException("after the effect") }
            val call = ToolCall("model-id", "read_note", obj("id" to "alice"))
            assertEquals("uncertain", gateway.invoke(call, "effect").text("status"))
            try { gateway.invoke(call, "effect"); fail("Uncertain action replayed") }
            catch (fault: RuntimeFault) { assertEquals("action_outcome_unknown", fault.code) }
            try { gateway.invoke(call.copy(arguments = obj("id" to "bob")), "effect"); fail("Changed arguments accepted") }
            catch (fault: RuntimeFault) { assertEquals("idempotency_conflict", fault.code) }
            assertEquals(1, effects)
        } finally { root.deleteRecursively() }
    }

    @Test fun flatSchemaRejectsUnsupportedKeywordsAndWrongTypes() {
        val validator = FlatToolSchema(schema)
        try { validator.validate(obj("id" to 42)); fail("Wrong type") }
        catch (fault: RuntimeFault) { assertEquals("invalid_tool_arguments", fault.code) }
        try { FlatToolSchema(schema.deepCopy().apply { addProperty("\$ref", "https://example.invalid") }); fail("Remote schema") }
        catch (fault: RuntimeFault) { assertEquals("unsupported_tool_schema", fault.code) }
    }

    @Test fun portableToolProtocolKeepsProposalsSeparateFromExecution() {
        val calls = FunctionCalling.parse("""{"tool_calls":[{"name":"read_note","arguments":{"id":"alice"}}]}""", listOf(definition))
        assertEquals("read_note", calls.single().name)
        val prepared = FunctionCalling.prepare(listOf(ChatMessage("assistant", "", toolCalls = calls), ChatMessage("tool", "a note", toolCallId = calls.single().id, name = "read_note")), listOf(definition), "json")
        assertEquals("system", prepared.first().role); assertEquals("user", prepared.last().role)
        for (bad in listOf("""{"tool_calls":null}""", """{"tool_calls":[{"name":"unknown","arguments":{}}]}""", """{"tool_calls":[],"tool_calls":[]} """, """{"tool_calls":[{"name":"read_note","arguments":"shell"}]}""")) {
            try { FunctionCalling.parse(bad, listOf(definition)); fail("Malformed call accepted") }
            catch (fault: RuntimeFault) { assertEquals("invalid_tool_call", fault.code) }
        }
        try { FunctionCalling.prepare(listOf(ChatMessage("user", "Read")), listOf(definition), "native"); fail("Unimplemented native adapter claimed") }
        catch (fault: RuntimeFault) { assertEquals("unsupported_tools", fault.code) }
    }

    @Test fun fineTuningRequiresSeparateConsentAndStillReportsUnsupported() {
        try { fineTuneStub(FineTuneRequest("assistant", "notes:alice"), policy(), "alice") }
        catch (fault: RuntimeFault) { assertEquals("permission_denied", fault.code) }
        try { fineTuneStub(FineTuneRequest("assistant", "dataset:consented"), policy(), "alice") }
        catch (fault: RuntimeFault) { assertEquals("fine_tuning_unavailable", fault.code) }
    }
}

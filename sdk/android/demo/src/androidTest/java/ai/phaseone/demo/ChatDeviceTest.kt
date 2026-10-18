package ai.phaseone.demo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.phaseone.runtime.CatalogTrust
import ai.phaseone.runtime.ChatMessage
import ai.phaseone.runtime.Generation
import ai.phaseone.runtime.PhaseoneRuntime
import ai.phaseone.runtime.ReleasePolicy
import ai.phaseone.runtime.RuntimeFault
import com.google.gson.GsonBuilder
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ChatDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val arguments = InstrumentationRegistry.getArguments()
    private val base = arguments.getString("server", "http://127.0.0.1:8930")!!.trimEnd('/')
    private val key = context.assets.open("catalog-publisher.pub").bufferedReader().use { it.readText().trim() }.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
    private val json = GsonBuilder().setPrettyPrinting().create()
    private fun runtime() = PhaseoneRuntime(context, CatalogTrust.key(key), localDevelopment = true)
    private fun write(name: String, report: Any) {
        File(context.filesDir, "evidence").mkdirs()
        File(context.filesDir, "evidence/$name.json").writeText(json.toJson(report))
    }

    @Test(timeout = 600000) fun nativeTextConversations() {
        val runtime = runtime()
        val report = linkedMapOf<String, Any>("device" to runtime.device, "input" to "Known-city task and canonical multi-turn name recall")
        val results = mutableListOf<Map<String, Any>>()
        try {
            for (backend in listOf("executorch", "litert")) {
                try {
                    val policy = ReleasePolicy("assistant.text.v1", listOf("hf.smollm2-135m-instruct"), backend = backend)
                    val activation = runtime.releases.update("$base/chat.json", "assistant", policy)
                    val response = runtime.open("assistant").use { app -> app.chat("assistant", listOf(
                        ChatMessage("user", "My name is Alex. Reply with OK."), ChatMessage("assistant", "OK."),
                        ChatMessage("user", "What is my name? Answer with only the name."),
                    ), Generation(maxOutputTokens = 32)) }
                    assertTrue(response.message.content.contains("Alex", ignoreCase = true))
                    assertEquals("android_local", response.execution.location)
                    assertEquals(backend, response.execution.backend)
                    assertFalse("Control tokens must not leak across the common chat API", response.message.content.endsWith("<|im_end|>"))
                    results.add(mapOf("backend" to backend, "passed" to true, "activation" to activation, "conversation" to response))
                } catch (failure: Throwable) {
                    results.add(mapOf("backend" to backend, "passed" to false, "error" to (failure.message ?: ""),
                        "code" to (if (failure is RuntimeFault) failure.code else failure.javaClass.simpleName)))
                }
            }
            report["results"] = results
            report["passed"] = results.all { it["passed"] == true }
            assertTrue("Every embedded text engine must pass: $results", report["passed"] == true)
        } finally { write("runtime-android-chat", report) }
    }

    @Test(timeout = 900000) fun gateScreenQualification() {
        val runtime = runtime()
        val accelerator = arguments.getString("accelerator", "cpu")!!
        val report = linkedMapOf<String, Any>("device" to runtime.device, "requested_accelerator" to accelerator,
            "input" to "Rendered terminal and color fixtures; this is not webcam/HID or a physical-presence test")
        try {
            val policy = ReleasePolicy("gate.screen.v1", listOf("hf.gemma4-e2b-it"), accelerators = listOf(accelerator), backend = "litert")
            val activation = runtime.releases.update("$base/gate.json", "screen-reader", policy)
            // A repeated verification must execute again, not return cached proof.
            report["activation"] = if (activation["status"].asString == "up_to_date") runtime.releases.requalify("screen-reader") else activation
            val active = runtime.releases.state("screen-reader")["active"]?.takeUnless { it.isJsonNull }?.asJsonObject
            assertNotNull("A qualified local screen reader must be installed", active)
            assertEquals(accelerator, active!!["accelerator"].asString)
            assertEquals("passed", active["qualification"].asString)
            report["checks"] = active["checks"]
            report["passed"] = true
            report["acceleration_observed"] = accelerator == "cpu"
            if (accelerator != "cpu") report["profiling_note"] = "The native API does not expose per-op device counters; requested NPU execution is recorded separately from profiling proof."
        } catch (failure: Throwable) {
            report["passed"] = false
            report["error"] = failure.message ?: failure.javaClass.simpleName
            report["code"] = if (failure is RuntimeFault) failure.code else failure.javaClass.simpleName
            throw failure
        } finally { write("runtime-android-gate-$accelerator", report) }
    }
}

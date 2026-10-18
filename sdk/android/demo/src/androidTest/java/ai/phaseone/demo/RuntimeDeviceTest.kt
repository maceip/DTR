package ai.phaseone.demo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.phaseone.runtime.CatalogTrust
import ai.phaseone.runtime.PhaseoneRuntime
import ai.phaseone.runtime.ReleasePolicy
import ai.phaseone.runtime.RuntimeFault
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class RuntimeDeviceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val arguments = InstrumentationRegistry.getArguments()
    private val base = arguments.getString("server", "http://127.0.0.1:8930")!!.trimEnd('/')
    private val key = context.assets.open("catalog-publisher.pub").bufferedReader().use { it.readText().trim() }.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
    private val json = GsonBuilder().setPrettyPrinting().create()
    private fun runtime() = PhaseoneRuntime(context, CatalogTrust.key(key), root = File(context.filesDir, "verification"), localDevelopment = true)
    private fun policy(backend: String? = null) = ReleasePolicy("imagenet1k.rgb224.v1", listOf("torchvision.mobilenet-v3-small.imagenet1k-v1"), backend = backend)
    private fun write(name: String, report: Any) {
        File(context.filesDir, "evidence").mkdirs()
        File(context.filesDir, "evidence/$name.json").writeText(json.toJson(report))
    }
    private fun classify(runtime: PhaseoneRuntime): Any {
        val result = runtime.open("vision").use { it.classify("vision", context.assets.open("sample-dog.jpg").use { image -> image.readBytes() }) }
        assertEquals("Samoyed", result.predictions.first().label)
        assertEquals("android_local", result.execution.location)
        assertEquals("cpu", result.execution.accelerator_observed)
        assertTrue(result.execution.execution_verified)
        return result
    }

    @Test fun localEnginesAndReleaseLifecycle() {
        val report = linkedMapOf<String, Any>("test" to "native Android release lifecycle", "device" to runtime().device)
        try {
            val root = runtime().root
            // A dedicated test store; the app's installed chat/Gate models stay intact.
            root.deleteRecursively()
            val runtime = runtime()
            val first = runtime.releases.update("$base/baseline.json", "vision", policy("executorch"))
            assertEquals("activated", first["status"].asString)
            val firstArtifacts = first["activation"].asJsonObject["artifact_receipts"].asJsonArray.map { it.asJsonObject["id"].asString }
            assertTrue("vision.executorch" in firstArtifacts)
            assertFalse("An unused engine's model must not be downloaded", "vision.litert" in firstArtifacts)
            report["executorch_install"] = first
            report["executorch_local"] = classify(runtime)
            val second = runtime.releases.update("$base/baseline.json", "vision", policy("litert"))
            assertEquals("activated", second["status"].asString)
            report["litert_install"] = second
            report["litert_local"] = classify(runtime)
            val receipt = second["activation"].asJsonObject["artifact_receipts"].asJsonArray
            assertTrue("Shared dependencies should come from the verified cache", receipt.filter { it.asJsonObject["id"].asString != "vision.litert" }.all { it.asJsonObject["downloaded_bytes"].asLong == 0L })
            report["rollback_between_engines"] = runtime.releases.rollback("vision")
            report["after_rollback"] = classify(runtime)
            val rollout = runtime.releases.update("$base/catalog.json", "vision", policy())
            report["targeted_rollout"] = rollout
            val expectedRelease = arguments.getString("expectedRelease")
            if (expectedRelease != null) assertEquals(expectedRelease, runtime.releases.state("vision")["active"].asJsonObject["release"].asString)
            report["after_rollout"] = classify(runtime)
            if (rollout["status"].asString == "activated") {
                val olderCompatible = runtime.releases.update("$base/catalog.json", "vision", policy("executorch"))
                assertEquals("activated", olderCompatible["status"].asString)
                assertEquals("vision-1", olderCompatible["activation"].asJsonObject["release"].asString)
                report["current_feed_older_compatible_model"] = olderCompatible
                report["restore_targeted_release"] = runtime.releases.rollback("vision")
                report["rollout_rollback"] = runtime.releases.rollback("vision")
                assertEquals("vision-1", runtime.releases.state("vision")["active"].asJsonObject["release"].asString)
                assertEquals("stale_catalog", runtime.releases.check("$base/baseline.json", "vision", policy())["status"].asString)
            }
            val activeBefore = runtime.releases.state("vision")["active"].asJsonObject.toString()
            for ((catalog, expected) in listOf("corrupt" to "artifact_integrity", "quality" to "qualification_failed")) {
                try {
                    runtime.releases.update("$base/tests/$catalog.json", "vision", policy("executorch"))
                    fail("Deliberately $catalog update must fail")
                } catch (failure: RuntimeFault) {
                    assertEquals(expected, failure.code)
                    assertEquals(activeBefore, runtime.releases.state("vision")["active"].asJsonObject.toString())
                    report["rejected_$catalog"] = mapOf("code" to failure.code, "active_preserved" to true, "fault_injection_fixture" to true)
                }
            }
            val fallback = runtime.releases.update("$base/tests/quality.json", "vision", policy("executorch").copy(allowFallback = true))
            assertEquals("retained", fallback["status"].asString)
            assertEquals(activeBefore, runtime.releases.state("vision")["active"].asJsonObject.toString())
            report["explicit_fallback_retains_working_release"] = fallback
            val freshRoot = File(context.filesDir, "fallback-verification")
            freshRoot.deleteRecursively()
            val fresh = PhaseoneRuntime(context, CatalogTrust.key(key), root = freshRoot, localDevelopment = true)
            val coldFallback = fresh.releases.update("$base/tests/quality.json", "vision", policy("executorch").copy(allowFallback = true))
            assertEquals("activated", coldFallback["status"].asString)
            assertEquals("vision-1", coldFallback["activation"].asJsonObject["release"].asString)
            assertEquals(1, coldFallback["activation"].asJsonObject["fallback"].asJsonArray.size())
            report["explicit_fallback_cold_install"] = coldFallback
            report["fallback_local"] = classify(fresh)
            val future = runtime.releases.check("$base/tests/native-upgrade.json", "vision", policy())
            assertEquals("app_update_required", future["status"].asString)
            report["native_update_boundary_fixture"] = future
            report["restart_local"] = classify(runtime())
            report["offline_requalification"] = runtime.releases.requalify("vision")
            report["passed"] = true
        } catch (failure: Throwable) {
            report["passed"] = false
            report["error"] = mapOf("type" to failure.javaClass.name, "message" to (failure.message ?: ""))
            throw failure
        } finally { write("runtime-android", report) }
    }

    @Test fun offlineLocalInference() {
        // Run separately after the host removes the ADB reverse/file-server link.
        val runtime = runtime()
        val result = classify(runtime)
        write("runtime-android-offline", mapOf("passed" to true, "device" to runtime.device, "result" to result,
            "note" to "No catalog or inference server request is made; caller removes the ADB reverse link for this check."))
    }
}

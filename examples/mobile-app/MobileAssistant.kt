package ai.phaseone.examples.mobile

import android.content.Context
import ai.phaseone.runtime.CatalogTrust
import ai.phaseone.runtime.PhaseoneRuntime
import ai.phaseone.runtime.ReleasePolicy
import ai.phaseone.runtime.ToolGateway
import com.google.gson.JsonObject

/**
 * Mobile Edge AI Assistant for Android.
 *
 * Demonstrates embedding PhaseoneRuntime in an Android application:
 * - Local inference (ExecuTorch / LiteRT) with on-device hardware acceleration.
 * - Safe OTA Model Updates via signed ReleasePolicy.
 * - Granular Tool Permissions checked before executing application handlers.
 */
class MobileAssistant(private val context: Context, publisherPublicKey: ByteArray) {

    // 1. Initialize Runtime with publisher trust store
    private val runtime = PhaseoneRuntime(
        context = context,
        trust = CatalogTrust.key(publisherPublicKey),
        localDevelopment = false
    )

    // 2. Define safe OTA update policy for vision capability
    val releasePolicy = ReleasePolicy(
        capability = "vision",
        candidates = listOf("torchvision.mobilenet-v3-small.v2", "torchvision.mobilenet-v3-small.v1")
    )

    /**
     * Check for signed catalog updates, download candidate release, run on-device
     * qualification checks, and atomically activate if verification passes.
     */
    fun updateModels(catalogUrl: String) {
        val updated = runtime.releases.update(catalogUrl, "vision", releasePolicy)
        if (updated) {
            println("Active vision model upgraded to latest qualified release.")
        } else {
            println("Working release retained (no update or candidate failed qualification).")
        }
    }

    /**
     * Classify a captured camera photo using the on-device model.
     */
    fun classifyPhoto(photoBytes: ByteArray): String {
        val app = runtime.open("vision")
        val result = app.classify("vision", photoBytes)
        return result.get("prediction").asString
    }

    /**
     * Invoke an application tool through the permission-checked gateway.
     */
    fun invokeTool(toolName: String, arguments: JsonObject, actionId: String): JsonObject {
        val app = runtime.open("assistant")
        return app.invokeTool(toolName, arguments, actionId)
    }
}

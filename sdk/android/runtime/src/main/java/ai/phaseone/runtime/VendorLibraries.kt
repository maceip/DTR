package ai.phaseone.runtime

import android.content.Context
import android.os.Build
import com.google.gson.JsonObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipFile

/** Vendor isolation for code ALREADY installed with this APK/feature module.
 * LiteRT 2.3 scans a directory and can select the first dispatch library there.
 * Passing an APK's common nativeLibraryDir when it contains multiple vendors is
 * ambiguous. Never let a model catalog supply a library filename or URL here.
 */
internal object VendorLibraries {
    private val vendors = mapOf(
        "litert.google-tensor" to listOf("libLiteRtDispatch_GoogleTensor.so", "libLiteRtCompilerPlugin_google_tensor.so"),
        "litert.mediatek" to listOf("libLiteRtDispatch_MediaTek.so", "libLiteRtCompilerPlugin_MediaTek.so"),
    )

    private fun <T> fromInstalled(context: Context, name: String, consume: (InputStream) -> T): T? {
        val extracted = File(context.applicationInfo.nativeLibraryDir, name)
        if (extracted.isFile) return extracted.inputStream().use(consume)
        // AABs may load uncompressed libraries directly from base/split APKs.
        val apks = listOf(context.applicationInfo.sourceDir) + context.applicationInfo.splitSourceDirs.orEmpty()
        for (path in apks) ZipFile(path).use { apk ->
            for (abi in Build.SUPPORTED_ABIS) {
                val entry = apk.getEntry("lib/$abi/$name") ?: continue
                return apk.getInputStream(entry).use(consume)
            }
        }
        return null
    }

    fun installedComponents(context: Context): Map<String, String> = vendors.filterValues { names ->
        names.all { name -> fromInstalled(context, name) { true } == true }
    }.mapValues { "2.3.0" }

    @Synchronized
    fun directory(context: Context, variant: JsonObject): String {
        val names = variant.child("target").child("components").keySet().intersect(vendors.keys)
        if (names.size != 1) throw RuntimeFault("unsupported_contract", "An NPU variant must select exactly one installed vendor adapter")
        val vendor = names.single()
        val destination = File(context.codeCacheDir, "phaseone-native/${BuildConfig.RUNTIME_BUILD_ID}/$vendor")
        destination.mkdirs()
        for (name in vendors.getValue(vendor)) {
            val target = File(destination, name)
            val bytes = fromInstalled(context, name) { it.readBytes() }
                ?: throw RuntimeFault("app_update_required", "Native library $name is not installed with this application")
            if (target.isFile && sha256(target) == sha256(bytes)) continue
            val temporary = File.createTempFile(".installed-", ".partial", destination)
            try {
                FileOutputStream(temporary).use { it.write(bytes); it.fd.sync() }
                if (!temporary.setReadOnly()) throw RuntimeFault("native_load_failed", "Could not make installed native code read-only")
                java.nio.file.Files.move(temporary.toPath(), target.toPath(), java.nio.file.StandardCopyOption.ATOMIC_MOVE, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            } finally { temporary.delete() }
        }
        return destination.absolutePath
    }
}

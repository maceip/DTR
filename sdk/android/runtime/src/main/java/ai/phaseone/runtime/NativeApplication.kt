package ai.phaseone.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.gson.JsonObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.exp

/** A local application handle. No inference requests leave the Android process. */
class NativeApplication internal constructor(
    private val context: Context,
    val directory: File,
    val manifest: JsonObject,
    private val permissions: PermissionPolicy? = null,
    private val subject: () -> String = { "local" },
) : AutoCloseable {
    val tools: ToolGateway? = permissions?.let { ToolGateway(manifest.deepCopy(), it, subject, File(context.filesDir, "dtr-tool-actions")) }
    private data class Loaded(val native: AutoCloseable, val loadMs: Double, val env: Environment? = null)
    private val loaded = mutableMapOf<String, Loaded>()
    private val chatEngines = mutableMapOf<String, NativeChatEngine>()
    private var closed = false

    fun artifact(id: String): File {
        val descriptor = manifest.child("artifacts").child(id)
        if (descriptor.size() == 0) throw RuntimeFault("missing_artifact", "Unknown artifact: $id")
        return inside(directory, descriptor.text("path"))
    }

    private fun model(name: String): JsonObject {
        val value = manifest.child("models").child(name)
        if (value.size() == 0) throw RuntimeFault("unknown_model", "Unknown model: $name")
        return value
    }

    private fun variant(model: JsonObject): JsonObject {
        if (model.array("variants").size() != 1) throw RuntimeFault("invalid_package", "Open a resolved single-variant release")
        return model.array("variants")[0].asJsonObject
    }

    private fun tensorEngine(name: String, variant: JsonObject): Loaded = loaded.getOrPut(name) {
        val started = System.nanoTime()
        val file = artifact(variant.text("artifact"))
        val accelerator = variant.child("target").text("accelerator", "cpu")
        when (variant.text("backend") + "/" + variant.text("engine")) {
            "executorch/program" -> {
                if (accelerator != "cpu") throw RuntimeFault("app_update_required", "This ExecuTorch adapter embeds XNNPACK CPU only")
                if (variant.optional("data_artifact") != null) throw RuntimeFault("unsupported_contract", "External tensor-data bindings are not supported by this Android program adapter")
                Loaded(Module.load(file.absolutePath), (System.nanoTime() - started) / 1e6)
            }
            "litert/compiled" -> {
                val selected = when (accelerator) { "cpu" -> Accelerator.CPU; "gpu" -> Accelerator.GPU; "npu" -> Accelerator.NPU; else -> throw RuntimeFault("unsupported_accelerator", accelerator) }
                val options = CompiledModel.Options(selected)
                val vendorDirectory = if (accelerator == "npu") VendorLibraries.directory(context, variant) else null
                val env = Environment.create(context, if (vendorDirectory != null) mapOf(
                    Environment.Option.DispatchLibraryDir to vendorDirectory,
                    Environment.Option.CompilerPluginLibraryDir to vendorDirectory,
                ) else emptyMap())
                try { Loaded(CompiledModel.create(file.absolutePath, options, env), (System.nanoTime() - started) / 1e6, env) }
                catch (failure: Throwable) { env.close(); throw failure }
            }
            else -> throw RuntimeFault("unsupported_contract", "Tensor inference requires an installed program/compiled adapter")
        }
    }

    @Synchronized
    fun infer(model: String, inputs: Map<String, TensorValue>, signature: String = "default"): InferenceResult {
        if (closed) throw RuntimeFault("runtime_closed", "Application is closed")
        val variant = variant(model(model))
        val contract = variant.child("signatures").child(signature)
        if (contract.size() == 0) throw RuntimeFault("unknown_signature", "Unknown tensor signature: $signature")
        val inputSpecs = contract.array("inputs").map { it.asJsonObject }
        val outputSpecs = contract.array("outputs").map { it.asJsonObject }
        if (inputs.keys != inputSpecs.map { it.text("name") }.toSet()) throw RuntimeFault("invalid_inputs", "Tensor input names differ from the contract")
        for (spec in inputSpecs) {
            val value = inputs.getValue(spec.text("name"))
            validateTensor(value, spec)
        }
        try {
            val engine = tensorEngine(model, variant)
            val started = System.nanoTime()
            val outputs: Map<String, TensorValue> = when (val native = engine.native) {
                is Module -> {
                    val values = inputSpecs.map { spec -> inputs.getValue(spec.text("name")).let { EValue.from(Tensor.fromBlob(it.data, it.shape)) } }
                    val result = native.execute(contract.text("method", "forward"), *values.toTypedArray())
                    if (result.size != outputSpecs.size) throw RuntimeFault("invalid_output", "Native output count differs from the contract")
                    outputSpecs.mapIndexed { i, spec -> result[i].toTensor().let { spec.text("name") to TensorValue(it.shape(), it.dataAsFloatArray) } }.toMap()
                }
                is CompiledModel -> {
                    val method = contract.text("method", "default")
                    val inputBuffers = linkedMapOf<String, TensorBuffer>()
                    val outputBuffers = linkedMapOf<String, TensorBuffer>()
                    try {
                        for (spec in inputSpecs) {
                            val name = spec.text("native_name", spec.text("name"))
                            inputBuffers[name] = native.createInputBuffer(name, method).also { it.writeFloat(inputs.getValue(spec.text("name")).data) }
                        }
                        for (spec in outputSpecs) {
                            val name = spec.text("native_name", spec.text("name"))
                            outputBuffers[name] = native.createOutputBuffer(name, method)
                        }
                        native.run(inputBuffers, outputBuffers, method)
                        outputSpecs.associate { spec ->
                            val name = spec.text("native_name", spec.text("name"))
                            val shape = native.getOutputTensorType(name, method).layout?.dimensions?.map { it.toLong() }?.toLongArray()
                                ?: throw RuntimeFault("invalid_output", "Native tensor has no concrete shape")
                            spec.text("name") to TensorValue(shape, outputBuffers.getValue(name).readFloat())
                        }
                    } finally {
                        inputBuffers.values.forEach { it.close() }; outputBuffers.values.forEach { it.close() }
                    }
                }
                else -> throw RuntimeFault("unsupported_contract", "Not a tensor engine")
            }
            for (spec in outputSpecs) validateTensor(outputs.getValue(spec.text("name")), spec)
            val accelerator = variant.child("target").text("accelerator", "cpu")
            return InferenceResult(outputs, Execution(
                backend = variant.text("backend"), engine = variant.text("engine"), variant = variant.text("id"),
                requested_accelerator = accelerator,
                // These upstream APIs do not expose per-op NPU counters. Keep the
                // observation unset for acceleration until independent profiling.
                accelerator_observed = if (accelerator == "cpu") "cpu" else null,
                inference_ms = (System.nanoTime() - started) / 1e6, load_ms = engine.loadMs,
                native_api = if (engine.native is Module) "org.pytorch.executorch.Module.execute" else "com.google.ai.edge.litert.CompiledModel.run",
                artifact_sha256 = manifest.child("artifacts").child(variant.text("artifact")).text("sha256"),
            ))
        } catch (failure: RuntimeFault) { throw failure }
        catch (failure: Throwable) { throw RuntimeFault("native_execution_failed", "${variant.text("backend")}: ${failure.message}", failure) }
    }

    @Synchronized
    fun classify(model: String, image: ByteArray, topK: Int = 5): ClassificationResult {
        if (topK !in 1..100) throw RuntimeFault("invalid_top_k", "topK must be between 1 and 100")
        val definition = model(model)
        if (definition.text("task") != "image.classification") throw RuntimeFault("wrong_operation", "Model is not an image classifier")
        val imageSpec = definition.child("image")
        val tensor = preprocessImage(image, imageSpec)
        val result = infer(model, mapOf(imageSpec.text("input_name") to tensor))
        val logits = result.outputs[definition.text("output_name")] ?: result.outputs.values.first()
        val max = logits.data.max().toDouble()
        val probabilities = logits.data.map { exp(it - max) }
        val sum = probabilities.sum()
        val labels = definition.optional("labels")?.let { com.google.gson.JsonParser.parseString(artifact(it).readText()).asJsonArray.map { l -> l.asString } } ?: emptyList()
        val predictions = probabilities.indices.sortedByDescending { probabilities[it] }.take(topK).map { Prediction(it, labels.getOrElse(it) { it.toString() }, probabilities[it] / sum) }
        return ClassificationResult(predictions, result.execution)
    }

    @Synchronized
    fun chat(model: String, messages: List<ChatMessage>, generation: Generation = Generation(), tools: List<ToolDefinition> = emptyList()): ChatResult {
        if (closed) throw RuntimeFault("runtime_closed", "Application is closed")
        val definition = model(model)
        if (definition.text("task") != "chat") throw RuntimeFault("wrong_operation", "Model is not a chat model")
        val variant = variant(definition)
        // Reject unsupported tool protocols before loading a native engine.
        if (tools.isNotEmpty() && variant.child("chat").text("tool_protocol") != "json") throw RuntimeFault("unsupported_tools", "Android tool-schema adapter currently implements JSON only")
        val engine = chatEngines.getOrPut(model) { NativeChatEngine(context, manifest, variant, ::artifact) }
        return engine.chat(messages, generation, tools)
    }

    fun capabilities(model: String): JsonObject {
        val definition = model(model)
        return obj("model" to model, "function_calling" to obj(
            "variants" to definition.array("variants").map { item -> val v = item.asJsonObject; val protocol = v.child("chat").text("tool_protocol", "none"); obj("variant" to v.text("id"), "protocol" to protocol, "declared" to (protocol != "none")) },
            "execution" to "application_tool_gateway", "quality_verified" to false, "implemented_protocols" to listOf("json")),
            "fine_tuning" to obj("available" to false, "status" to "not_implemented", "objectives" to listOf("function_calling", "chat")))
    }

    fun fineTune(request: FineTuneRequest): Nothing {
        if (model(request.model).text("task") != "chat") throw RuntimeFault("wrong_operation", "Fine-tuning requires a chat model")
        return fineTuneStub(request, permissions, subject())
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        for (engine in loaded.values) { try { engine.native.close() } finally { engine.env?.close() } }
        loaded.clear()
        chatEngines.values.forEach { it.close() }; chatEngines.clear()
    }
}

private fun validateTensor(value: TensorValue, spec: JsonObject) {
    val expected = spec.array("shape").map { it.asLong }
    if (spec.text("dtype") != value.dtype || expected.size != value.shape.size || expected.zip(value.shape.toList()).any { (a, b) -> a != -1L && a != b })
        throw RuntimeFault("invalid_tensor", "Tensor ${spec.text("name")} differs from its dtype/shape contract")
}

private fun preprocessImage(bytes: ByteArray, spec: JsonObject): TensorValue {
    if (bytes.size > 32 * 1024 * 1024) throw RuntimeFault("invalid_image", "Image exceeds 32 MiB")
    var original: Bitmap? = null
    var resized: Bitmap? = null
    var cropped: Bitmap? = null
    try {
        original = ImageDecoder.decodeBitmap(ImageDecoder.createSource(ByteBuffer.wrap(bytes))) { decoder, info, _ ->
            if (info.size.width.toLong() * info.size.height > 40_000_000) throw RuntimeFault("invalid_image", "Decoded image is too large")
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
        }
        val width = spec.number("width")
        val height = spec.number("height")
        val shorter = spec.number("resize_shorter")
        if (shorter > 0) {
            val scale = shorter.toDouble() / minOf(original.width, original.height)
            resized = Bitmap.createScaledBitmap(original, (original.width * scale).toInt(), (original.height * scale).toInt(), true)
            if (resized.width < width || resized.height < height) throw RuntimeFault("invalid_image", "Resize is smaller than requested crop")
            cropped = Bitmap.createBitmap(resized, (resized.width - width) / 2, (resized.height - height) / 2, width, height)
        } else cropped = Bitmap.createScaledBitmap(original, width, height, true)
        val pixels = IntArray(width * height)
        cropped.getPixels(pixels, 0, width, 0, 0, width, height)
        val mean = spec.array("mean").map { it.asFloat }.ifEmpty { listOf(0f, 0f, 0f) }
        val std = spec.array("std").map { it.asFloat }.ifEmpty { listOf(1f, 1f, 1f) }
        if (mean.size != 3 || std.size != 3 || std.any { it <= 0 }) throw RuntimeFault("invalid_package", "Invalid RGB normalization")
        val scale = spec.get("scale")?.asFloat ?: (1f / 255)
        val nchw = spec.text("layout", "nchw") == "nchw"
        val data = FloatArray(pixels.size * 3)
        for (i in pixels.indices) for (channel in 0..2) {
            val rgb = (pixels[i] shr (16 - channel * 8)) and 255
            data[if (nchw) channel * pixels.size + i else i * 3 + channel] = (rgb * scale - mean[channel]) / std[channel]
        }
        return TensorValue(if (nchw) longArrayOf(1, 3, height.toLong(), width.toLong()) else longArrayOf(1, height.toLong(), width.toLong(), 3), data)
    } catch (failure: RuntimeFault) { throw failure }
    catch (failure: Exception) { throw RuntimeFault("invalid_image", "Image decoding failed", failure) }
    finally { listOfNotNull(original, resized, cropped).distinct().forEach { it.recycle() } }
}

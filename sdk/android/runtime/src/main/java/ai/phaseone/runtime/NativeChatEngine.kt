package ai.phaseone.runtime

import android.content.Context
import android.graphics.BitmapFactory
import android.os.Build
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.gson.JsonObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import java.io.File

/** Native runners with fixed, SDK-owned prompt profiles. No downloaded code. */
internal class NativeChatEngine(
    context: Context,
    private val manifest: JsonObject,
    private val variant: JsonObject,
    artifact: (String) -> File,
) : AutoCloseable {
    private val config = variant.child("chat")
    private val accelerator = variant.child("target").text("accelerator", "cpu")
    private var lite: Engine? = null
    private var torch: LlmModule? = null
    private val loadMs: Double

    init {
        val start = System.nanoTime()
        fun backend(name: String): Backend = when (name) {
            "cpu" -> Backend.CPU(threadCount = 4)
            "gpu" -> Backend.GPU()
            "npu" -> Backend.NPU(VendorLibraries.directory(context, variant))
            else -> throw RuntimeFault("unsupported_accelerator", name)
        }
        try {
            if (variant.text("backend") == "litert") {
                val cacheIdentity = sha256(canonical(obj(
                    "variant" to variant, "artifact" to manifest.child("artifacts").child(variant.text("artifact")).text("sha256"),
                    "sdk" to BuildConfig.RUNTIME_BUILD_ID, "firmware" to Build.FINGERPRINT,
                )).toByteArray())
                val cache = File(context.cacheDir, "phaseone-lm/$cacheIdentity").apply { mkdirs() }
                val engine = Engine(EngineConfig(
                    modelPath = artifact(variant.text("artifact")).absolutePath, backend = backend(accelerator),
                    visionBackend = if ("image" in config.strings("modalities")) backend(config.text("vision_accelerator", "cpu")) else null,
                    audioBackend = if ("audio" in config.strings("modalities")) backend(config.text("audio_accelerator", "cpu")) else null,
                    maxNumTokens = config.number("context_length"),
                    maxNumImages = if ("image" in config.strings("modalities")) config.number("max_images", 1) else null,
                    cacheDir = cache.absolutePath,
                ))
                engine.initialize(); lite = engine
            } else {
                if (accelerator != "cpu" || config.child("processor").size() > 0 || config.text("prompt_format") !in listOf("chatml", "llama3"))
                    throw RuntimeFault("unsupported_contract", "Android ExecuTorch chat requires a supported CPU text profile")
                val builder = LlmModuleConfig.create().modulePath(artifact(variant.text("artifact")).absolutePath)
                    .tokenizerPath(artifact(config.text("tokenizer")).absolutePath).temperature(0f)
                    .numBos(config.number("num_bos")).numEos(config.number("num_eos"))
                variant.optional("data_artifact")?.let { builder.dataPath(artifact(it).absolutePath) }
                val module = LlmModule(builder.build())
                try { module.load(); torch = module } catch (failure: Throwable) { module.close(); throw failure }
            }
        } catch (failure: RuntimeFault) { throw failure }
        catch (failure: Throwable) { throw RuntimeFault("native_load_failed", "${variant.text("backend")}: ${failure.message}", failure) }
        loadMs = (System.nanoTime() - start) / 1e6
    }

    fun chat(input: List<ChatMessage>, generation: Generation, tools: List<ToolDefinition> = emptyList()): ChatResult {
        val messages = FunctionCalling.prepare(input, tools, config.text("tool_protocol", "none"))
        if (messages.isEmpty() || messages.last().role != "user") throw RuntimeFault("invalid_messages", "Conversation must end with a user message")
        val imageCount = messages.sumOf { it.images.size }
        if (imageCount > config.number("max_images", 1) || (imageCount > 0 && "image" !in config.strings("modalities"))) throw RuntimeFault("unsupported_modality", "Images exceed this model's capability")
        for (message in messages) for (image in message.images) {
            if (image.size > 32 * 1024 * 1024) throw RuntimeFault("invalid_image", "Image exceeds 32 MiB")
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(image, 0, image.size, bounds)
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0 || bounds.outWidth.toLong() * bounds.outHeight > 16_000_000 || bounds.outMimeType !in listOf("image/jpeg", "image/png"))
                throw RuntimeFault("invalid_image", "Expected a bounded PNG or JPEG image")
        }
        val start = System.nanoTime()
        val text: String
        try {
            val engine = lite
            if (engine != null) {
                fun contents(message: ChatMessage): Contents = Contents.of(message.images.map { Content.ImageBytes(it) } + Content.Text(message.content))
                val system = messages.filter { it.role == "system" }.joinToString("\n") { it.content }
                val history = messages.dropLast(1).filter { it.role != "system" }.map {
                    if (it.role == "assistant") com.google.ai.edge.litertlm.Message.model(contents(it))
                    else com.google.ai.edge.litertlm.Message.user(contents(it))
                }
                engine.createConversation(ConversationConfig(
                    systemInstruction = system.takeIf { it.isNotEmpty() }?.let { Contents.of(it) },
                    initialMessages = history, automaticToolCalling = false,
                    samplerConfig = SamplerConfig(topK = if (generation.temperature == 0.0) 1 else 40, topP = 1.0, temperature = generation.temperature),
                    maxOutputToken = generation.maxOutputTokens, thinkingConfig = ThinkingConfig(enableThinking = false),
                )).use { conversation ->
                    val response = conversation.sendMessage(contents(messages.last()))
                    text = response.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                    if (response.toolCalls.isNotEmpty()) throw RuntimeFault("unsupported_tools", "Native structured calls need a model-specific Android adapter; use a JSON tool variant")
                }
            } else {
                if (imageCount > 0) throw RuntimeFault("unsupported_modality", "This ExecuTorch profile supports text only")
                val module = torch ?: throw RuntimeFault("runtime_closed", "Chat engine is closed")
                val prompt = when (config.text("prompt_format")) {
                    "chatml" -> messages.joinToString("") { "<|im_start|>${it.role}\n${it.content}<|im_end|>\n" } + "<|im_start|>assistant\n"
                    "llama3" -> "<|begin_of_text|>" + messages.joinToString("") { "<|start_header_id|>${it.role}<|end_header_id|>\n\n${it.content}<|eot_id|>" } + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    else -> throw RuntimeFault("unsupported_contract", "Unknown prompt profile")
                }
                val generated = StringBuilder()
                var error: RuntimeFault? = null
                module.resetContext()
                module.generate(prompt, LlmGenerationConfig.create().echo(false)
                    .maxNewTokens(generation.maxOutputTokens).seqLen(config.number("context_length"))
                    .temperature(generation.temperature.toFloat()).numBos(config.number("num_bos")).numEos(config.number("num_eos")).build(), object : LlmCallback {
                    override fun onResult(result: String) { generated.append(result) }
                    override fun onError(code: Int, message: String) { error = RuntimeFault("native_execution_failed", "ExecuTorch $code: $message") }
                })
                error?.let { throw it }
                val raw = generated.toString().trimEnd()
                // Runners may emit the terminal control token in their callback.
                // Remove only a final, profile-owned marker, never text inside an answer.
                text = if (config.text("prompt_format") == "chatml") raw.removeSuffix("<|im_end|>").trimEnd()
                    else raw.removeSuffix("<|eot_id|>").removeSuffix("<|end_of_text|>").trimEnd()
            }
        } catch (failure: RuntimeFault) { throw failure }
        catch (failure: Throwable) { throw RuntimeFault("native_execution_failed", "${variant.text("backend")}: ${failure.message}", failure) }
        val calls = FunctionCalling.parse(text, tools)
        return ChatResult(ChatMessage("assistant", text, toolCalls = calls), Execution(
            backend = variant.text("backend"), engine = "llm", variant = variant.text("id"),
            requested_accelerator = accelerator, accelerator_observed = if (accelerator == "cpu") "cpu" else null,
            inference_ms = (System.nanoTime() - start) / 1e6, load_ms = loadMs,
            native_api = if (lite != null) "com.google.ai.edge.litertlm.Engine" else "org.pytorch.executorch.extension.llm.LlmModule",
            artifact_sha256 = manifest.child("artifacts").child(variant.text("artifact")).text("sha256"),
        ), calls)
    }

    override fun close() { lite?.close(); lite = null; torch?.close(); torch = null }
}

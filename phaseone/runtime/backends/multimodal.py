"""ExecuTorch's native multimodal runner with a packaged, offline processor."""

import importlib.metadata

from ..errors import RuntimeFault
from ..media import image_from_part
from .chat import StopStream, completion, prepare_messages
from .executorch import tensor_info


class ExecuTorchMultimodal:
    def __init__(self, package, variant, cache):
        from executorch.extension.llm.runner import MultimodalRunner
        from executorch.runtime import Runtime
        from transformers import AutoProcessor

        self.variant = variant
        if set(variant.chat.modalities) - {"text", "image"}:
            raise RuntimeFault("unsupported_modality", "This packaged processor supports text and images")
        if variant.chat.tool_protocol == "native":
            raise RuntimeFault("unsupported_tools", "ExecuTorch multimodal variants use the JSON tool protocol")
        delegates = list(Runtime.get().backend_registry.registered_backend_names)
        if variant.target.delegate and variant.target.delegate not in delegates:
            raise RuntimeFault("delegate_unavailable", "Required ExecuTorch delegate is not installed",
                               details={"required": variant.target.delegate, "registered": delegates})
        if variant.target.accelerator != "cpu" and not variant.target.delegate:
            raise RuntimeFault("invalid_target", "An accelerated variant must name its delegate")
        processor_config = variant.chat.processor
        directory = package.artifact(processor_config.config).parent
        self.processor = AutoProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
        )
        if processor_config.image_splitting is not None:
            if not hasattr(self.processor.image_processor, "do_image_splitting"):
                raise RuntimeFault("processor_contract", "Processor does not expose image splitting")
            self.processor.image_processor.do_image_splitting = processor_config.image_splitting
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(processor_config.image_token)
        if self.image_token_id is None or self.image_token_id == self.processor.tokenizer.unk_token_id:
            raise RuntimeFault("processor_contract", "Image token is absent from the packaged tokenizer")
        model_path = self.model_path = str(package.artifact(variant.artifact))
        data_path = self.data_path = str(package.artifact(variant.data_artifact)) if variant.data_artifact else None
        preflight = Runtime.get().load_program(model_path, data_path=data_path)
        required = {"token_embeddings", "text_model", "vision_encoder"}
        if required - set(preflight.method_names):
            raise RuntimeFault("artifact_contract", "Model is not a compatible ExecuTorch multimodal bundle",
                               details={"required_methods": sorted(required), "actual": list(preflight.method_names)})
        for name in required:
            preflight.load_method(name)
        if "get_max_context_len" in preflight.method_names:
            limit = int(preflight.load_method("get_max_context_len").execute([])[0])
            if variant.chat.context_length > limit:
                raise RuntimeFault("artifact_contract", "Manifest context exceeds the compiled model context",
                                   details={"compiled_context_length": limit})
        self.program_runner = processor_config.runner == "program"
        self.vision_inputs = processor_config.vision_inputs
        vision_metadata = preflight.metadata("vision_encoder")
        self.vision_specs = [tensor_info(vision_metadata.input_tensor_meta(i))
                             for i in range(vision_metadata.num_inputs())]
        if len(self.vision_specs) != len(self.vision_inputs):
            raise RuntimeFault("processor_contract", "Declared vision inputs differ from the compiled graph")
        if self.program_runner and (variant.chat.num_bos or variant.chat.num_eos):
            raise RuntimeFault("processor_contract", "Program runner requires special tokens in the packaged chat template")
        if not self.program_runner and self.vision_inputs != ["pixel_values"]:
            raise RuntimeFault("processor_contract", "Multiple vision inputs require the declared program runner")
        if not self.program_runner and "get_max_seq_len" not in preflight.method_names:
            raise RuntimeFault("artifact_contract", "Native multimodal runner requires get_max_seq_len metadata; use a compatible export")
        del preflight
        self.runner = None if self.program_runner else MultimodalRunner(
            model_path, str(package.artifact(variant.chat.tokenizer)), data_path,
        )
        self.info = {
            "backend": "executorch", "engine": "llm",
            "runner": "ExecuTorchProgramMultimodal" if self.program_runner else "MultimodalRunner",
            "runtime_version": importlib.metadata.version("executorch"),
            "requested_accelerator": variant.target.accelerator,
            "delegate": variant.target.delegate or "portable_cpu",
            "acceleration_evidence": "cpu_execution" if variant.target.accelerator == "cpu" else "registered_delegate; profiling_required",
            "npu_execution_verified": False, "modalities": variant.chat.modalities,
            "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
            "sampling": ["temperature", "max_output_tokens", "stop"],
            "processor": "packaged_huggingface", "processor_remote_code": False,
        }

    def prepare(self, messages, tools):
        images, normalized = [], []
        for message in prepare_messages(messages, tools, self.variant.chat.tool_protocol):
            parts = message["content"]
            if isinstance(parts, str):
                parts = [{"type": "text", "text": parts}]
            content = []
            for part in parts:
                if part["type"] == "image":
                    picture = image_from_part(part)
                    config = self.variant.chat.processor
                    if config.image_fit == "square_pad":
                        from PIL import Image, ImageOps

                        fitted = ImageOps.pad(picture, (config.image_size, config.image_size),
                                              method=Image.Resampling.LANCZOS, color=picture.getpixel((0, 0)))
                        picture.close()
                        picture = fitted
                    images.append(picture)
                    content.append({"type": "image"})
                elif part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                else:
                    raise RuntimeFault("unsupported_modality", "This processor accepts image and text input")
            normalized.append({**message, "content": content})
        try:
            if len(images) > 1:
                raise RuntimeFault("unsupported_media_layout", "This ExecuTorch HF runner supports one image per conversation")
            prompt = self.processor.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images or None, return_tensors="pt")
            pixels = inputs.get("pixel_values")
            # Some processors include a singleton batch dimension before the
            # image dimension. The native helper accepts one CHW/HWC image.
            if pixels is not None:
                while pixels.ndim > 4 and pixels.shape[0] == 1:
                    pixels = pixels.squeeze(0)
                if pixels.ndim not in (3, 4) or (pixels.ndim == 4 and pixels.shape[0] != 1):
                    raise RuntimeFault("unsupported_media_layout", "Export requires a single image; tiling needs a matching runner")
                inputs["pixel_values"] = pixels.contiguous()
            return inputs
        finally:
            for image in images:
                image.close()

    def chat(self, messages, generation, tools, emit):
        from executorch.extension.llm.runner import GenerationConfig, generate_hf

        if generation.get("top_p") is not None or generation.get("top_k") is not None:
            raise RuntimeFault("unsupported_generation_option", "ExecuTorch MultimodalRunner does not expose top_p or top_k")
        inputs = self.prepare(messages, tools)
        count = inputs["input_ids"].numel() + self.variant.chat.num_bos + self.variant.chat.num_eos
        if count + generation["max_output_tokens"] > self.variant.chat.context_length:
            raise RuntimeFault("context_exceeded", "Text and image tokens exceed the packaged context",
                               details={"prompt_tokens": count, "context_length": self.variant.chat.context_length})
        stops = [*generation["stop"]]
        if self.processor.tokenizer.eos_token:
            stops.append(self.processor.tokenizer.eos_token)
        stream, usage = StopStream(stops, emit), {}
        if self.program_runner:
            return self._program_chat(inputs, generation, tools, stream)

        def token(text):
            stream.push(text)
            if stream.stopped:
                self.runner.stop()

        def stats(value):
            usage.update({"input_tokens": value.num_prompt_tokens,
                          "output_tokens": value.num_generated_tokens, "source": "native_runner"})

        self.runner.reset()
        generate_hf(self.runner, inputs, GenerationConfig(
            echo=False, max_new_tokens=generation["max_output_tokens"],
            seq_len=self.variant.chat.context_length, temperature=generation["temperature"],
            num_bos=self.variant.chat.num_bos, num_eos=self.variant.chat.num_eos,
        ), image_token_id=self.image_token_id, token_callback=token, stats_callback=stats)
        stream.finish()
        reason = "length" if usage.get("output_tokens", 0) >= generation["max_output_tokens"] else "stop"
        return completion(stream.text, tools, self.variant.chat.tool_protocol, usage=usage,
                          reason="stop" if stream.stopped else reason)

    def _program_chat(self, inputs, generation, tools, stream):
        """Run a declared three-method bundle using ExecuTorch for every model graph.

        This explicit profile uses [batch, sequence] position IDs. It is useful for
        bundles without native LLM-runner metadata; it is never a silent fallback.
        A fresh program owns each conversation's mutable KV cache.
        """
        import torch
        from executorch.runtime import Runtime

        program = Runtime.get().load_program(self.model_path, data_path=self.data_path)
        embed = program.load_method("token_embeddings")
        decoder = program.load_method("text_model")
        ids = inputs["input_ids"].to(torch.int64)
        embeddings = embed.execute([ids])[0].clone()
        if "pixel_values" in inputs:
            vision = program.load_method("vision_encoder")
            if any(name not in inputs for name in self.vision_inputs):
                raise RuntimeFault("processor_contract", "Processor did not produce all declared vision inputs")
            vision_args = []
            for name, spec in zip(self.vision_inputs, self.vision_specs, strict=True):
                value = inputs[name]
                if list(value.shape) != spec["shape"]:
                    raise RuntimeFault("processor_contract", "Processor output shape differs from the compiled vision input",
                                       details={"input": name, "expected": spec["shape"], "actual": list(value.shape)})
                # HF masks may be int32 while the exported graph requires int64.
                # The compiled graph, rather than the processor default, owns the dtype.
                vision_args.append(value.to(getattr(torch, spec["dtype"])).contiguous())
            features = vision.execute(vision_args)[0]
            mask = ids == self.image_token_id
            if int(mask.sum()) != features.shape[1] or embeddings.shape[-1] != features.shape[-1]:
                raise RuntimeFault("processor_contract", "Image token expansion does not match the compiled vision encoder")
            embeddings[mask] = features.reshape(-1, features.shape[-1])
        position = ids.shape[-1]
        positions = torch.arange(position, dtype=torch.int64).unsqueeze(0)
        logits = decoder.execute([embeddings, positions])[0]
        generated, emitted, reason = [], "", "length"
        for _ in range(generation["max_output_tokens"]):
            scores = logits[0, -1, :]
            if generation["temperature"] == 0:
                token = int(scores.argmax())
            else:
                token = int(torch.multinomial(torch.softmax(scores / generation["temperature"], dim=-1), 1))
            generated.append(token)
            if token == self.processor.tokenizer.eos_token_id:
                reason = "stop"
                break
            decoded = self.processor.tokenizer.decode(generated, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
            stable = decoded.rstrip("\ufffd")
            if not stable.startswith(emitted):
                raise RuntimeFault("tokenizer_contract", "Tokenizer rewrote an already streamed prefix")
            stream.push(stable[len(emitted):])
            emitted = stable
            if stream.stopped:
                reason = "stop"
                break
            if len(generated) == generation["max_output_tokens"]:
                break
            embeddings = embed.execute([torch.tensor([[token]], dtype=torch.int64)])[0]
            logits = decoder.execute([embeddings, torch.tensor([[position]], dtype=torch.int64)])[0]
            position += 1
        stream.finish()
        return completion(stream.text, tools, self.variant.chat.tool_protocol,
                          usage={"input_tokens": ids.numel(), "output_tokens": len(generated),
                                 "source": "executorch_program_loop"}, reason=reason)

    def close(self):
        self.runner = None
        self.processor = None

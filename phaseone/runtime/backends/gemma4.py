"""ExecuTorch Gemma 4 PLE bundle profile used by Software Mansion's public exports.

The package names this profile explicitly. Image preprocessing follows that
export's raw RGB, letterboxed CHW contract; every neural graph runs in ExecuTorch.
This is not the different patch-input contract in upstream Gemma 4 examples.
"""

import importlib.metadata
import json

from ..errors import RuntimeFault
from ..media import image_from_part
from .chat import StopStream, completion, prepare_messages


class ExecuTorchGemma4:
    def __init__(self, package, variant, cache):
        # The installed wheel ships these attention kernels separately. Import
        # the official registration module before preflighting model methods.
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
        import executorch.kernels.quantized  # noqa: F401
        from executorch.runtime import Runtime
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.variant = variant
        if set(variant.chat.modalities) - {"text", "image"} or variant.chat.tool_protocol == "native":
            raise RuntimeFault("unsupported_modality", "This profile accepts text/images and JSON tool requests")
        if variant.chat.max_images != 1:
            raise RuntimeFault("processor_contract", "This profile accepts one image per conversation")
        delegates = list(Runtime.get().backend_registry.registered_backend_names)
        if variant.target.delegate and variant.target.delegate not in delegates:
            raise RuntimeFault("delegate_unavailable", "Required ExecuTorch delegate is not installed",
                               details={"required": variant.target.delegate, "registered": delegates})
        if variant.target.accelerator != "cpu" and not variant.target.delegate:
            raise RuntimeFault("invalid_target", "An accelerated variant must name its delegate")
        profile = json.loads(package.artifact(variant.chat.processor.config).read_text())
        if profile.get("model") != "gemma_4":
            raise RuntimeFault("processor_contract", "Expected the Gemma 4 multimodal export profile")
        self.model_path = str(package.artifact(variant.artifact))
        self.data_path = str(package.artifact(variant.data_artifact)) if variant.data_artifact else None
        self.tokenizer = Tokenizer.from_file(str(package.artifact(variant.chat.tokenizer)))
        config = json.loads(package.artifact(variant.chat.tokenizer_config).read_text())
        self.tokens = {key: (value.get("content", "") if isinstance(value, dict) else value)
                       for key, value in config.items() if key.endswith("_token") and isinstance(value, (str, dict))}
        template = package.artifact(variant.chat.template).read_text() if variant.chat.template else config.get("chat_template")
        if not isinstance(template, str) or not 0 < len(template) <= 65536:
            raise RuntimeFault("chat_template_required", "Package a model-specific chat template")
        environment = ImmutableSandboxedEnvironment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        def fail(message):
            raise RuntimeFault("chat_template", str(message))
        environment.globals["raise_exception"] = fail
        self.template = environment.from_string(template)
        self.image_id = self.tokenizer.token_to_id(variant.chat.processor.image_token)
        preflight = Runtime.get().load_program(self.model_path, data_path=self.data_path)
        required = {"token_embedding", "text_decoder", "vision_encoder", "get_max_seq_len", "get_max_context_len", "get_eos_ids", "has_ple"}
        if required - set(preflight.method_names):
            raise RuntimeFault("artifact_contract", "Expected an ExecuTorch Gemma 4 PLE bundle",
                               details={"required_methods": sorted(required), "actual": list(preflight.method_names)})
        def metadata(name):
            return preflight.load_method(name).execute([])[0]
        if not metadata("has_ple"):
            raise RuntimeFault("artifact_contract", "Model does not provide per-layer embeddings")
        self.prefill_limit = int(metadata("get_max_seq_len"))
        context_limit = int(metadata("get_max_context_len"))
        if not 1 <= self.prefill_limit <= context_limit or variant.chat.context_length > context_limit:
            raise RuntimeFault("artifact_contract", "Manifest exceeds the compiled context limits")
        if "image_placeholder_id" in preflight.method_names and self.image_id != int(metadata("image_placeholder_id")):
            raise RuntimeFault("processor_contract", "Packaged image token differs from the compiled model")
        if self.image_id is None:
            raise RuntimeFault("processor_contract", "Image token is absent from the packaged tokenizer")
        vision = preflight.load_method("vision_encoder")
        shape = list(vision.metadata.input_tensor_meta(0).sizes())
        output = list(vision.metadata.output_tensor_meta(0).sizes())
        if len(shape) != 4 or shape[:2] != [1, 3] or len(output) != 3 or output[0] != 1:
            raise RuntimeFault("artifact_contract", "Expected a single RGB image and vision-token embeddings")
        self.height, self.width, self.image_tokens = shape[2], shape[3], output[1]
        for name in ("token_embedding", "text_decoder"):
            preflight.load_method(name)
        self.eos = {int(value) for value in preflight.load_method("get_eos_ids").execute([])}
        del vision
        self.info = {"backend": "executorch", "engine": "llm", "runner": "ExecuTorchGemma4PLE",
                     "runtime_version": importlib.metadata.version("executorch"),
                     "requested_accelerator": variant.target.accelerator,
                     "delegate": variant.target.delegate or "portable_cpu",
                     "acceleration_evidence": "cpu_execution" if variant.target.accelerator == "cpu" else "registered_delegate; profiling_required",
                     "npu_execution_verified": False, "modalities": variant.chat.modalities,
                     "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "max_output_tokens", "stop"],
                     "processor": "gemma4_raw_rgb_letterbox", "processor_remote_code": False}

    def prepare(self, messages, tools):
        import numpy as np
        import torch
        from PIL import Image

        normalized, pictures = [], []
        for message in prepare_messages(messages, tools, self.variant.chat.tool_protocol):
            content = message["content"]
            if isinstance(content, str):
                normalized.append(message)
                continue
            parts = []
            for part in content:
                if part["type"] == "image":
                    pictures.append(part)
                    parts.append({"type": "image"})
                elif part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                else:
                    raise RuntimeFault("unsupported_modality", "This processor accepts text and images")
            normalized.append({**message, "content": parts})
        if len(pictures) > 1:
            raise RuntimeFault("unsupported_media_layout", "This profile accepts one image per conversation")
        try:
            prompt = self.template.render(messages=normalized, tools=tools or None,
                                          add_generation_prompt=True, enable_thinking=False, **self.tokens)
        except RuntimeFault:
            raise
        except Exception as exc:
            raise RuntimeFault("chat_template", str(exc)) from exc
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if tokens.count(self.image_id) != len(pictures):
            raise RuntimeFault("processor_contract", "Image placeholders do not match supplied images")
        expanded = [value for token in tokens for value in ([token] * self.image_tokens if token == self.image_id else [token])]
        pixels = None
        if pictures:
            with image_from_part(pictures[0]) as picture:
                array = np.asarray(picture)
                corner = max(1, min(picture.size) // 30)
                background = tuple(np.floor(np.mean([
                    array[:corner, :corner].mean(axis=(0, 1)), array[:corner, -corner:].mean(axis=(0, 1)),
                    array[-corner:, :corner].mean(axis=(0, 1)), array[-corner:, -corner:].mean(axis=(0, 1)),
                ], axis=0)).astype(int))
                ratio = min(self.width / picture.width, self.height / picture.height)
                size = (max(1, int(picture.width * ratio)), max(1, int(picture.height * ratio)))
                resized = picture.resize(size, Image.Resampling.BOX)
                canvas = Image.new("RGB", (self.width, self.height), background)
                canvas.paste(resized, ((self.width - size[0]) // 2, (self.height - size[1]) // 2))
                # Normalization is already inside this export's vision graph.
                pixels = torch.from_numpy(np.array(canvas, dtype=np.float32).transpose(2, 0, 1).copy()).unsqueeze(0)
                resized.close()
                canvas.close()
        return torch.tensor([expanded], dtype=torch.int64), pixels

    def chat(self, messages, generation, tools, emit):
        import torch
        from executorch.runtime import Runtime

        if generation.get("top_p") is not None or generation.get("top_k") is not None:
            raise RuntimeFault("unsupported_generation_option", "This profile exposes temperature sampling")
        ids, pixels = self.prepare(messages, tools)
        length = ids.shape[1]
        if length + generation["max_output_tokens"] > self.variant.chat.context_length:
            raise RuntimeFault("context_exceeded", "Text and image tokens exceed the packaged context",
                               details={"prompt_tokens": length, "context_length": self.variant.chat.context_length})
        program = Runtime.get().load_program(self.model_path, data_path=self.data_path)
        embed, decoder = program.load_method("token_embedding"), program.load_method("text_decoder")
        features = program.load_method("vision_encoder").execute([pixels])[0].clone() if pixels is not None else None
        image_offset = 0
        for start in range(0, length, self.prefill_limit):
            chunk = ids[:, start:start + self.prefill_limit]
            outputs = embed.execute([chunk])
            embeddings, ple = outputs[0].clone(), outputs[1].clone()
            mask = chunk == self.image_id
            count = int(mask.sum())
            if count:
                if features is None or features.shape[-1] != embeddings.shape[-1]:
                    raise RuntimeFault("processor_contract", "Vision embeddings differ from token embeddings")
                embeddings[mask] = features[0, image_offset:image_offset + count]
                image_offset += count
            positions = torch.arange(start, start + chunk.shape[1], dtype=torch.int64)
            logits = decoder.execute([embeddings, ple, positions])[0]
        stream = StopStream(generation["stop"], emit)
        generated, emitted, reason = [], "", "length"
        for _ in range(generation["max_output_tokens"]):
            scores = logits[0, -1, :]
            token = int(scores.argmax()) if generation["temperature"] == 0 else int(
                torch.multinomial(torch.softmax(scores / generation["temperature"], dim=-1), 1))
            generated.append(token)
            if token in self.eos:
                reason = "stop"
                break
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False).rstrip("\ufffd")
            if not decoded.startswith(emitted):
                raise RuntimeFault("tokenizer_contract", "Tokenizer rewrote an already streamed prefix")
            stream.push(decoded[len(emitted):])
            emitted = decoded
            if stream.stopped:
                reason = "stop"
                break
            if len(generated) == generation["max_output_tokens"]:
                break
            outputs = embed.execute([torch.tensor([[token]], dtype=torch.int64)])
            embeddings, ple = outputs[0].clone(), outputs[1].clone()
            logits = decoder.execute([embeddings, ple, torch.tensor([length], dtype=torch.int64)])[0]
            length += 1
        stream.finish()
        return completion(stream.text, tools, self.variant.chat.tool_protocol,
                          usage={"input_tokens": ids.numel(), "output_tokens": len(generated),
                                 "source": "executorch_program_loop"}, reason=reason)

    def close(self):
        self.tokenizer = self.template = None

"""Apple MLX chat on the Metal GPU, loaded only from the package's verified files."""

import importlib.metadata
import os
import shutil
from pathlib import Path

from ..errors import RuntimeFault
from .chat import StopStream, completion, prepare_messages, text_content


def stage_model_directory(package, variant, cache) -> Path:
    """Link exactly the variant's declared artifacts into one directory.

    mlx-lm discovers weights by globbing its model directory. Loading from a
    staged directory keeps undeclared files in the package out of the model.
    """
    directory = Path(cache) / "mlx-model"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    for identifier in sorted(variant.artifact_ids()):
        source = package.artifact(identifier)
        target = directory / source.name
        if target.exists():
            raise RuntimeFault("invalid_package", f"Two MLX artifacts share the file name {source.name}")
        os.symlink(source, target)
    return directory


class MLXChat:
    def __init__(self, package, variant, cache):
        import mlx.core as mx
        from mlx_lm import load

        self.variant = variant
        if variant.chat.modalities != ["text"]:
            raise RuntimeFault("unsupported_modality", "The MLX adapter supports text conversations")
        if variant.chat.tool_protocol == "native":
            raise RuntimeFault("unsupported_tools", "MLX variants use the json tool protocol")
        accelerator = variant.target.accelerator
        if accelerator == "gpu":
            if not mx.metal.is_available():
                raise RuntimeFault("backend_unavailable", "MLX GPU execution requires Apple Silicon Metal")
            mx.set_default_device(mx.gpu)
        elif accelerator == "cpu":
            mx.set_default_device(mx.cpu)
        else:
            raise RuntimeFault("invalid_target", "MLX runs on the cpu or the Metal gpu")
        self.model, self.tokenizer = load(str(stage_model_directory(package, variant, cache)))
        self.template = package.artifact(variant.chat.template).read_text() if variant.chat.template else None
        self.info = {"backend": "mlx", "engine": "llm", "runtime_version": importlib.metadata.version("mlx-lm"),
                     "mlx_version": importlib.metadata.version("mlx"), "requested_accelerator": accelerator,
                     "device": str(mx.default_device()),
                     "acceleration_evidence": "metal_default_device" if accelerator == "gpu" else "cpu_execution",
                     "npu_execution_verified": False, "modalities": ["text"],
                     "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "top_p", "top_k", "max_output_tokens", "stop"]}

    def prompt_tokens(self, messages, tools) -> list[int]:
        messages = prepare_messages(messages, tools, self.variant.chat.tool_protocol)
        normalized = [{"role": m["role"], "content": text_content(m)} for m in messages]
        options = {"chat_template": self.template} if self.template else {}
        try:
            prompt = self.tokenizer.apply_chat_template(normalized, add_generation_prompt=True,
                                                        tokenize=False, **options)
        except Exception as exc:
            raise RuntimeFault("chat_template", str(exc)) from exc
        # Same rule mlx-lm applies to text prompts: no second BOS when the template has one.
        bos = self.tokenizer.bos_token
        return self.tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))

    def chat(self, messages, generation, tools, emit):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        tokens = self.prompt_tokens(messages, tools)
        if len(tokens) + generation["max_output_tokens"] > self.variant.chat.context_length:
            raise RuntimeFault("context_exceeded", "Prompt plus requested output exceeds the model context",
                               details={"prompt_tokens": len(tokens), "context_length": self.variant.chat.context_length})
        sampler = make_sampler(temp=generation["temperature"], top_p=generation.get("top_p") or 0.0,
                               top_k=generation.get("top_k") or 0)
        stream, last = StopStream(generation["stop"], emit), None
        for response in stream_generate(self.model, self.tokenizer, tokens,
                                        max_tokens=generation["max_output_tokens"], sampler=sampler):
            last = response
            stream.push(response.text)
            if stream.stopped:
                break
        stream.finish()
        usage = {"source": "native_runner"}
        if last is not None:
            usage.update({"input_tokens": last.prompt_tokens, "output_tokens": last.generation_tokens,
                          "prompt_tokens_per_second": last.prompt_tps,
                          "generation_tokens_per_second": last.generation_tps,
                          "peak_memory_gb": last.peak_memory})
        reason = "stop" if stream.stopped or last is None or last.finish_reason != "length" else "length"
        return completion(stream.text, tools, self.variant.chat.tool_protocol, usage=usage, reason=reason)

    def close(self):
        self.model = self.tokenizer = None

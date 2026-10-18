"""Native LLM runners, canonical history, streaming and brokered tool requests."""

import importlib.metadata
import json
import os
import uuid

from ..errors import RuntimeFault
from ..schema import ToolCall


class StopStream:
    """Withhold only a potential stop prefix, including across token boundaries."""

    def __init__(self, stops, emit):
        self.stops, self.emit = stops, emit
        self.pending, self.text, self.stopped = "", "", False

    def _emit(self, text):
        if text:
            self.text += text
            self.emit({"type": "delta", "text": text})

    def push(self, text):
        if self.stopped:
            return
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if positions:
            self._emit(self.pending[:min(positions)])
            self.pending, self.stopped = "", True
            return
        keep = 0
        for stop in self.stops:
            for length in range(1, min(len(stop), len(self.pending) + 1)):
                if self.pending.endswith(stop[:length]):
                    keep = max(keep, length)
        if keep:
            self._emit(self.pending[:-keep])
            self.pending = self.pending[-keep:]
        else:
            self._emit(self.pending)
            self.pending = ""

    def finish(self):
        if not self.stopped:
            self._emit(self.pending)
        self.pending = ""


def text_content(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if any(part["type"] != "text" for part in content):
        raise RuntimeFault("unsupported_modality", "This ExecuTorch text runner accepts text; use a tensor model for media")
    return "".join(part["text"] for part in content)


def prepare_messages(messages, tools, protocol):
    """JSON tools have the same prompt contract on both engines; no native execution."""
    result = []
    for item in messages:
        item = dict(item)
        if protocol == "json":
            if item.get("tool_calls"):
                item["content"] = json.dumps({"tool_calls": item["tool_calls"]}, ensure_ascii=False)
                item.pop("tool_calls", None)
            if item["role"] == "tool":
                item["content"] = json.dumps({"tool": item.get("name"),
                                               "tool_call_id": item.get("tool_call_id"),
                                               "result": item["content"]}, ensure_ascii=False)
        result.append(item)
    if tools:
        if protocol == "none":
            raise RuntimeFault("unsupported_tools", "This model variant does not accept tool requests")
        if protocol == "json":
            instruction = (
                'When you need a tool, reply only with JSON: '
                '{"tool_calls":[{"name":"tool_name","arguments":{}}]}. '
                'The host executes tools and returns results. Otherwise answer normally. '
                'Available tools: ' + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
            )
            if result and result[0]["role"] == "system":
                result[0] = {**result[0], "content": text_content(result[0]) + "\n" + instruction}
            else:
                result.insert(0, {"role": "system", "content": instruction})
    return result


def completion(text, tools, protocol, *, calls=None, usage=None, reason="stop"):
    calls = calls or []
    if tools and protocol == "json":
        candidate = text.strip()
        if candidate.startswith("```json\n") and candidate.endswith("```"):
            candidate = candidate[8:-3].strip()
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict) and "tool_calls" in document:
            if not isinstance(document["tool_calls"], list) or len(document["tool_calls"]) > 16:
                raise RuntimeFault("invalid_tool_call", "Model emitted an invalid tool call list")
            calls = document["tool_calls"]
            text = document.get("content", "")
    normalized = []
    for call in calls:
        try:
            normalized.append(ToolCall.model_validate({
                "id": call.get("id") or str(uuid.uuid4()), "name": call["name"],
                "arguments": call["arguments"],
            }).model_dump(mode="json"))
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeFault("invalid_tool_call", "Model emitted an invalid tool call", details={"error": str(exc)}) from exc
    if not isinstance(text, str):
        raise RuntimeFault("invalid_tool_call", "Tool envelope content must be text")
    return {"message": {"role": "assistant", "content": text, "tool_calls": normalized},
            "finish_reason": "tool_calls" if normalized else reason, "usage": usage or {}}


class ExecuTorchChat:
    def __init__(self, package, variant, cache):
        from executorch.extension.llm.runner import TextLLMRunner
        from executorch.runtime import Runtime
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.variant = variant
        if variant.chat.modalities != ["text"]:
            raise RuntimeFault("unsupported_modality", "This LLM runner supports text; package vision/audio as tensor models")
        if variant.chat.tool_protocol == "native":
            raise RuntimeFault("unsupported_tools", "ExecuTorch text variants use the json tool protocol")
        delegates = list(Runtime.get().backend_registry.registered_backend_names)
        if variant.target.delegate and variant.target.delegate not in delegates:
            raise RuntimeFault("delegate_unavailable", f"ExecuTorch has no {variant.target.delegate}",
                               details={"registered_delegates": delegates})
        if variant.target.accelerator != "cpu" and not variant.target.delegate:
            raise RuntimeFault("invalid_target", "An accelerated variant must name its ExecuTorch delegate")
        tokenizer = package.artifact(variant.chat.tokenizer)
        try:
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
        except Exception as exc:
            raise RuntimeFault("unsupported_tokenizer", "This adapter requires a Hugging Face tokenizer.json") from exc
        self.tokens = {}
        config = json.loads(package.artifact(variant.chat.tokenizer_config).read_text()) if variant.chat.tokenizer_config else {}
        for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
            value = config.get(key, "")
            self.tokens[key] = value.get("content", "") if isinstance(value, dict) else value
        template = package.artifact(variant.chat.template).read_text() if variant.chat.template else config.get("chat_template")
        if not isinstance(template, str) or not template or len(template) > 65536:
            raise RuntimeFault("chat_template_required", "Package a model-specific chat template or tokenizer_config.json")
        environment = ImmutableSandboxedEnvironment(undefined=StrictUndefined,
                                                    trim_blocks=True, lstrip_blocks=True)
        def fail(message):
            raise RuntimeFault("chat_template", str(message))
        environment.globals["raise_exception"] = fail
        self.template = environment.from_string(template)
        preflight = Runtime.get().load_program(
            str(package.artifact(variant.artifact)),
            data_path=str(package.artifact(variant.data_artifact)) if variant.data_artifact else None,
        )
        # Validate kernel registration and the compiled context before exposing a
        # ready model. A file's extension cannot prove native compatibility.
        preflight.load_method("forward")
        if "get_max_context_len" in preflight.method_names:
            compiled_context = int(preflight.load_method("get_max_context_len").execute([])[0])
            if variant.chat.context_length > compiled_context:
                raise RuntimeFault("artifact_contract", "Manifest context exceeds the compiled model context",
                                   details={"compiled_context_length": compiled_context})
        del preflight
        self.runner = TextLLMRunner(str(package.artifact(variant.artifact)), str(tokenizer),
                                    str(package.artifact(variant.data_artifact)) if variant.data_artifact else None)
        self.info = {"backend": "executorch", "engine": "llm", "runtime_version": importlib.metadata.version("executorch"),
                     "requested_accelerator": variant.target.accelerator, "delegate": variant.target.delegate or "portable_cpu",
                     "acceleration_evidence": "cpu_execution" if variant.target.accelerator == "cpu" else "registered_delegate; profiling_required",
                     "npu_execution_verified": False, "modalities": ["text"],
                     "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "max_output_tokens", "stop"]}

    def chat(self, messages, generation, tools, emit):
        from executorch.extension.llm.runner import GenerationConfig

        if generation.get("top_p") is not None or generation.get("top_k") is not None:
            raise RuntimeFault("unsupported_generation_option", "ExecuTorch TextLLMRunner does not expose top_p or top_k")
        messages = prepare_messages(messages, tools, self.variant.chat.tool_protocol)
        normalized = [{**m, "content": text_content(m)} for m in messages]
        try:
            prompt = self.template.render(messages=normalized, add_generation_prompt=True,
                                           tools=tools or None, **self.tokens)
        except RuntimeFault:
            raise
        except Exception as exc:
            raise RuntimeFault("chat_template", str(exc)) from exc
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)
        prompt_tokens += self.variant.chat.num_bos + self.variant.chat.num_eos
        if prompt_tokens + generation["max_output_tokens"] > self.variant.chat.context_length:
            raise RuntimeFault("context_exceeded", "Prompt plus requested output exceeds the model context",
                               details={"prompt_tokens": prompt_tokens, "context_length": self.variant.chat.context_length})
        stops = [*generation["stop"]]
        if self.tokens.get("eos_token"):
            stops.append(self.tokens["eos_token"])
        stream = StopStream(stops, emit)
        usage = {}
        def token(text):
            stream.push(text)
            if stream.stopped:
                self.runner.stop()
        def stats(value):
            usage.update({"input_tokens": value.num_prompt_tokens,
                          "output_tokens": value.num_generated_tokens, "source": "native_runner"})
        self.runner.reset()
        self.runner.generate(prompt, GenerationConfig(
            echo=False, max_new_tokens=generation["max_output_tokens"],
            seq_len=self.variant.chat.context_length, temperature=generation["temperature"],
            num_bos=self.variant.chat.num_bos, num_eos=self.variant.chat.num_eos,
        ), token_callback=token, stats_callback=stats)
        stream.finish()
        reason = "length" if usage.get("output_tokens", 0) >= generation["max_output_tokens"] else "stop"
        if stream.stopped:
            reason = "stop"
        return completion(stream.text, tools, self.variant.chat.tool_protocol, usage=usage, reason=reason)

    def close(self):
        self.runner = None


class LiteRTChat:
    def __init__(self, package, variant, cache):
        import litert_lm

        self.variant, self.lm = variant, litert_lm
        accelerator = variant.target.accelerator
        def make_backend(kind):
            options = {}
            if kind == "npu" and os.environ.get("LITERT_DISPATCH_LIB_DIR"):
                options["litert_dispatch_lib_dir"] = os.environ["LITERT_DISPATCH_LIB_DIR"]
            return getattr(litert_lm.Backend, kind.upper())(**options)
        backend = make_backend(accelerator)
        media_options = {}
        if "image" in variant.chat.modalities:
            media_options["vision_backend"] = make_backend(variant.chat.vision_accelerator or accelerator)
            media_options["max_num_images"] = variant.chat.max_images
        if "audio" in variant.chat.modalities:
            media_options["audio_backend"] = make_backend(variant.chat.audio_accelerator or accelerator)
        self.engine = litert_lm.Engine(
            model_path=str(package.artifact(variant.artifact)), backend=backend,
            max_num_tokens=variant.chat.context_length, cache_dir=str(cache), enable_benchmark=True,
            **media_options,
        )
        self.template = package.artifact(variant.chat.template).read_text() if variant.chat.template else None
        self.info = {"backend": "litert", "engine": "llm", "runtime_version": importlib.metadata.version("litert-lm-api"),
                     "requested_accelerator": accelerator, "acceleration_evidence": "cpu_execution" if accelerator == "cpu" else "requested_backend; profiling_required",
                     "npu_execution_verified": False, "modalities": variant.chat.modalities,
                     "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "top_p", "top_k", "max_output_tokens", "stop"]}

    @staticmethod
    def native_message(message, *, native_tools=False):
        result = {key: value for key, value in message.items() if key != "tool_calls"}
        if native_tools and message["role"] == "tool":
            response = message.get("content", "")
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    pass
            if isinstance(response, dict) and response.get("status") == "complete" and "result" in response:
                response = response["result"]
            result["content"] = [{"type": "tool_response", "name": message.get("name"),
                                  "response": response}]
            return result
        if isinstance(result.get("content"), list):
            result["content"] = [{"type": part["type"], "text": part["text"]} if part["type"] == "text"
                                  else {"type": part["type"], "blob": part["data"]}
                                  for part in result["content"]]
        if message.get("tool_calls"):
            result["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                "name": c["name"], "arguments": c["arguments"],
            }} for c in message["tool_calls"]]
        return result

    def chat(self, messages, generation, tools, emit):
        messages = prepare_messages(messages, tools, self.variant.chat.tool_protocol)
        messages = [self.native_message(m, native_tools=self.variant.chat.tool_protocol == "native") for m in messages]
        native_tools = None
        if tools and self.variant.chat.tool_protocol == "native":
            class SchemaTool(self.lm.Tool):
                def __init__(self, definition):
                    self.definition = definition
                def get_tool_description(self):
                    return {"type": "function", "function": self.definition}
                def execute(self, param):
                    raise RuntimeFault("unbrokered_tool", "Tools must execute through the PHASEONE gateway")
            native_tools = [SchemaTool(t) for t in tools]
        conversation = self.engine.create_conversation(
            messages=messages[:-1], tools=native_tools, automatic_tool_calling=False,
            sampler_config=self.lm.SamplerConfig(temperature=generation["temperature"],
                                                 top_p=generation.get("top_p"), top_k=generation.get("top_k")),
            max_output_tokens=generation["max_output_tokens"], chat_template=self.template,
        )
        try:
            # The native renderer includes canonical history and the actual model template.
            rendered = conversation.render_message_to_string(messages[-1])
            tokens = len(self.engine.tokenize(rendered))
            if tokens + generation["max_output_tokens"] > self.variant.chat.context_length:
                raise RuntimeFault("context_exceeded", "Prompt plus requested output exceeds the model context",
                                   details={"prompt_tokens": tokens, "context_length": self.variant.chat.context_length})
            stream, calls = StopStream(generation["stop"], emit), []
            for chunk in conversation.send_message_async(messages[-1], max_output_tokens=generation["max_output_tokens"]):
                stream.push(str(chunk.contents))
                for call in chunk.tool_calls:
                    calls.append({"id": call.id, "name": call.name, "arguments": call.arguments})
                if stream.stopped:
                    conversation.cancel_process()
                    # Drain cancellation so native callbacks finish before close.
            stream.finish()
            benchmark = conversation.get_benchmark_info()
            usage = {"input_tokens": benchmark.last_prefill_token_count,
                     "output_tokens": benchmark.last_decode_token_count, "source": "native_runner",
                     "time_to_first_token_ms": benchmark.time_to_first_token_in_second * 1000}
            reason = "length" if usage["output_tokens"] >= generation["max_output_tokens"] else "stop"
            return completion(stream.text, tools, self.variant.chat.tool_protocol, calls=calls,
                              usage=usage, reason="stop" if stream.stopped else reason)
        finally:
            conversation.close()

    def close(self):
        self.engine.close()

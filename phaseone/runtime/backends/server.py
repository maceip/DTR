"""Chat through a model already served on this machine over the OpenAI-compatible API.

Covers Ollama, LM Studio, llama.cpp's llama-server, oMLX and similar servers. The
package carries a hash-covered binding (loopback URL and model name), not weights:
the runtime cannot verify what the server loaded, so results say so. Release
checks still run against the served model before it is activated.
"""

import importlib.metadata
import json
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import RuntimeFault
from .chat import StopStream, completion, prepare_messages, text_content

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ServerBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    base_url: str = Field(max_length=256)
    model: str = Field(min_length=1, max_length=256)

    @field_validator("base_url")
    @classmethod
    def loopback_only(cls, value):
        parsed = urlparse(value)
        if (parsed.scheme not in ("http", "https") or parsed.hostname not in LOOPBACK_HOSTS
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("A package can only bind to a model server on this machine (loopback http/https)")
        return value.rstrip("/")


def read_binding(path) -> ServerBinding:
    from ..package import read_json

    try:
        return ServerBinding.model_validate(read_json(path.read_bytes()))
    except ValueError as exc:
        raise RuntimeFault("invalid_server_binding", str(exc)) from exc


class LocalServerChat:
    def __init__(self, package, variant, cache):
        import httpx

        self.variant = variant
        if variant.chat.modalities != ["text"]:
            raise RuntimeFault("unsupported_modality", "The local server adapter supports text conversations")
        if variant.chat.tool_protocol == "native":
            raise RuntimeFault("unsupported_tools", "Local server variants use the json tool protocol")
        self.binding = read_binding(package.artifact(variant.artifact))
        # The worker deadline bounds generation; only connecting needs its own limit.
        self.client = httpx.Client(base_url=self.binding.base_url, timeout=httpx.Timeout(10.0, read=None))
        try:
            response = self.client.get("/models")
            response.raise_for_status()
            served = [item.get("id") for item in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            self.client.close()
            raise RuntimeFault("backend_unavailable", f"No OpenAI-compatible model server at {self.binding.base_url}",
                               details={"error": str(exc)}) from exc
        if self.binding.model not in served:
            self.client.close()
            raise RuntimeFault("server_model_unavailable", f"The server does not serve {self.binding.model}",
                               details={"served_models": served[:50]})
        self.info = {"backend": "local_server", "engine": "llm", "runtime_version": importlib.metadata.version("httpx"),
                     "server_url": self.binding.base_url, "server_model": self.binding.model,
                     "weights_verified": False, "requested_accelerator": variant.target.accelerator,
                     "acceleration_evidence": "external_server; not observed", "npu_execution_verified": False,
                     "modalities": ["text"], "context_length": variant.chat.context_length,
                     "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "top_p", "top_k", "max_output_tokens", "stop"]}

    @staticmethod
    def wire_messages(messages):
        # JSON-protocol tool results are plain text for the model; OpenAI-compatible
        # servers reject a "tool" role without a native tool call, so send them as user turns.
        return [{"role": "user" if m["role"] == "tool" else m["role"], "content": text_content(m)} for m in messages]

    def chat(self, messages, generation, tools, emit):
        import httpx

        messages = prepare_messages(messages, tools, self.variant.chat.tool_protocol)
        body = {"model": self.binding.model, "messages": self.wire_messages(messages), "stream": True,
                "max_tokens": generation["max_output_tokens"], "temperature": generation["temperature"],
                "stream_options": {"include_usage": True}}
        for key in ("top_p", "top_k"):
            if generation.get(key) is not None:
                body[key] = generation[key]
        if generation["stop"]:
            body["stop"] = generation["stop"]
        stream, usage, reason = StopStream(generation["stop"], emit), {}, "stop"
        try:
            with self.client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    raise RuntimeFault("inference_failed", f"Model server returned HTTP {response.status_code}",
                                       details={"body": response.read().decode(errors="replace")[:1000]})
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        stream.push((choice.get("delta") or {}).get("content") or "")
                        reason = choice.get("finish_reason") or reason
                    if stream.stopped:
                        break  # Closing the response stops generation on the server.
        except httpx.HTTPError as exc:
            raise RuntimeFault("inference_failed", f"Model server request failed: {exc}") from exc
        stream.finish()
        usage = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                 "source": "server_reported"}
        return completion(stream.text, tools, self.variant.chat.tool_protocol, usage=usage,
                          reason="stop" if stream.stopped else "length" if reason == "length" else "stop")

    def close(self):
        self.client.close()

"""ExecuTorch/QNN text runner behind the normal LocalKit chat interface."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..errors import RuntimeFault
from ..qnn_host import qnn_installation
from .chat import StopStream, completion, prepare_messages, text_content


class QnnChat:
    def __init__(self, package, variant, cache):
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        installation = qnn_installation(verify=True)
        if installation is None:
            raise RuntimeFault("qnn_unavailable", "Install the LocalKit QNN host runtime")
        self.root, self.profile = installation
        self.package, self.variant, self.cache = package, variant, Path(cache)
        self.process = None
        self.tokenizer = Tokenizer.from_file(str(package.artifact(variant.chat.tokenizer)))
        config = json.loads(package.artifact(variant.chat.tokenizer_config).read_text()) if variant.chat.tokenizer_config else {}
        self.tokens = {}
        for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
            value = config.get(key, "")
            self.tokens[key] = value.get("content", "") if isinstance(value, dict) else value
        template = package.artifact(variant.chat.template).read_text()
        if len(template) > 65536:
            raise RuntimeFault("chat_template", "Chat template is too large")
        environment = ImmutableSandboxedEnvironment(undefined=StrictUndefined)
        def fail(message):
            raise RuntimeFault("chat_template", str(message))
        environment.globals["raise_exception"] = fail
        self.template = environment.from_string(template)
        self.info = {"backend": "executorch", "engine": "llm", "runner": "qnn_llama_runner",
                     "runtime_version": self.profile["executorch_version"],
                     "qnn_version": self.profile["qnn_version"], "delegate": "QnnBackend",
                     "requested_accelerator": "npu", "acceleration_evidence": "qnn_host_installed",
                     "npu_execution_verified": False, "modalities": ["text"],
                     "context_length": variant.chat.context_length, "tool_protocol": variant.chat.tool_protocol,
                     "sampling": ["temperature", "max_output_tokens", "stop"],
                     "streaming": "buffered_native_completion"}

    def chat(self, messages, generation, tools, emit):
        if generation.get("top_p") is not None or generation.get("top_k") is not None:
            raise RuntimeFault("unsupported_generation_option", "This QNN runner does not expose top_p or top_k")
        normalized = [{**m, "content": text_content(m)}
                      for m in prepare_messages(messages, tools, self.variant.chat.tool_protocol)]
        try:
            prompt = self.template.render(messages=normalized, add_generation_prompt=True,
                                          tools=tools or None, **self.tokens)
        except RuntimeFault:
            raise
        except Exception as exc:
            raise RuntimeFault("chat_template", str(exc)) from exc
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        count = len(ids) + generation["max_output_tokens"]
        if not ids or count > self.variant.chat.context_length:
            raise RuntimeFault("context_exceeded", "Prompt plus output exceeds the compiled context",
                               details={"prompt_tokens": len(ids), "context_length": self.variant.chat.context_length})
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(self.root / "lib") + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        environment["ADSP_LIBRARY_PATH"] = str(self.root / "dsp")
        with tempfile.TemporaryDirectory(prefix="qnn-", dir=self.cache) as temporary:
            work = Path(temporary)
            echo_prefix = str(work / "prompt.raw")
            (work / "prompt.raw").write_bytes(np.asarray(ids, dtype="<u8").tobytes())
            argv = [sys.executable, "-m", "phaseone.runtime.qnn_exec", str(os.getpid()),
                    str(self.root / self.profile["runner"]),
                    "--decoder_model_version", self.variant.chat.decoder,
                    "--model_path", str(self.package.artifact(self.variant.artifact)),
                    "--tokenizer_path", str(self.package.artifact(self.variant.chat.tokenizer)),
                    "--tokenized_prompt", str(work / "prompt.raw"),
                    "--seq_len", str(count), "--temperature", str(generation["temperature"]),
                    "--output_path", str(work / "output.txt"),
                    "--performance_output_path", str(work / "speed.txt")]
            try:
                self.process = subprocess.Popen(argv, cwd=work, env=environment,
                                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT)
                output, _ = self.process.communicate(timeout=generation["timeout_seconds"])
                log = output.decode(errors="replace")
                (self.cache / "last-qnn.log").write_text(log[-256000:])
                if self.process.returncode:
                    raise RuntimeFault("qnn_inference_failed", "The QNN runner failed",
                                       details={"exit_code": self.process.returncode,
                                                "native_log": str(self.cache / "last-qnn.log"),
                                                "log_tail": log[-5000:]})
                match = re.search(r'PyTorchObserver (\{[^\n]+\})', log)
                if match is None or "Qnn executorch backend type 2" not in log:
                    raise RuntimeFault("qnn_execution_unverified", "The runner returned no QNN generation statistics")
                stats = json.loads(match[1])
                if stats["prompt_tokens"] != len(ids):
                    raise RuntimeFault("qnn_token_contract", "The native runner consumed a different prompt length")
                if not 1 <= stats["generated_tokens"] <= generation["max_output_tokens"]:
                    raise RuntimeFault("qnn_token_contract", "The native runner exceeded the output token contract")
                text = (work / "output.txt").read_text()
            except subprocess.TimeoutExpired as exc:
                raise RuntimeFault("inference_timeout", "The QNN runner exceeded its time budget") from exc
            finally:
                self.close()
        # In tokenized-prompt mode this runner echoes the filename, not its contents.
        if not text.startswith(echo_prefix):
            raise RuntimeFault("qnn_output_contract", "The runner's echoed input did not match the supplied token file")
        text = text[len(echo_prefix):]
        stops = [*generation["stop"], "<end_of_turn>", "<turn|>"]
        if self.tokens.get("eos_token"):
            stops.append(self.tokens["eos_token"])
        stream = StopStream(stops, emit)
        stream.push(text)
        stream.finish()
        result = completion(stream.text, tools, self.variant.chat.tool_protocol,
                            usage={"input_tokens": stats["prompt_tokens"], "output_tokens": stats["generated_tokens"],
                                   "source": "native_runner"},
                            reason="length" if not stream.stopped and stats["generated_tokens"] >= generation["max_output_tokens"] else "stop")
        result["native_execution"] = {"npu_execution_verified": True,
                                      "acceleration_evidence": "qnn_htp_context_executed",
                                      "stats": stats}
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None

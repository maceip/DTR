"""MLX and local-server backends: packaging, selection and the server protocol.

The local-server path runs end to end through a real worker process against a
scripted OpenAI-compatible server. MLX execution itself needs Apple Silicon and
is exercised by the skip-guarded test at the bottom.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.backends.mlx import stage_model_directory
from phaseone.runtime.backends.server import ServerBinding
from phaseone.runtime.build import import_mlx, import_server_binding
from phaseone.runtime.catalog import Catalog, ReleasePolicy, resolve
from phaseone.runtime.device import compatibility
from phaseone.runtime.package import load_package
from phaseone.runtime.schema import Variant

ROOT = Path(__file__).resolve().parents[1]
MLX_MODEL = ROOT / ".work/models/Qwen3-0.6B-4bit"


class ScriptedServer(BaseHTTPRequestHandler):
    reply = ["Hel", "lo", " STOP never shown"]
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            return self.send_error(404)
        self.respond(200, json.dumps({"data": [{"id": "served-model"}]}).encode(), "application/json")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body)
        if body["model"] != "served-model":
            return self.respond(404, b'{"error":"unknown model"}', "application/json")
        chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]} for text in type(self).reply]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        events = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        self.respond(200, events, "text/event-stream")

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    ScriptedServer.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def server_package(tmp_path, base_url, model="served-model"):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"base_url": base_url, "model": model}))
    import_server_binding(binding, tmp_path / "package", model="assistant", chat={"context_length": 4096})
    return tmp_path / "package"


@pytest.mark.parametrize("url", ["http://192.168.1.5:11434/v1", "https://api.example.com/v1",
                                 "http://user:secret@127.0.0.1:11434/v1", "file:///tmp/socket", "http://127.0.0.1.example.com/v1"])
def test_server_binding_only_reaches_this_machine(url):
    with pytest.raises(ValidationError):
        ServerBinding.model_validate({"base_url": url, "model": "m"})


def test_new_backends_are_llm_only():
    for backend in ("mlx", "local_server"):
        with pytest.raises(ValidationError, match="llm engine"):
            Variant.model_validate({"id": "v", "backend": backend, "engine": "program", "artifact": "a",
                                    "signatures": {"default": {"inputs": [], "outputs": []}}})


@pytest.mark.parametrize(("stop", "text", "usage"), [
    ([], "Hello STOP never shown", {"input_tokens": 12, "output_tokens": 3, "source": "server_reported"}),
    # A stop sequence ends the stream early, before the server reports usage.
    (["STOP"], "Hello ", {"input_tokens": None, "output_tokens": None, "source": "server_reported"}),
])
def test_local_server_chat_streams_through_a_worker(tmp_path, server, stop, text, usage):
    with Runtime(tmp_path / "state") as runtime:
        app = runtime.load(server_package(tmp_path, server))
        events = list(app.chat_stream("assistant", [{"role": "user", "content": "hi"}],
                                      generation={"max_output_tokens": 32, "stop": stop}))
    done = events[-1]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == text
    assert done["message"]["content"] == text
    assert done["usage"] == usage
    execution = done["execution"]
    assert (execution["backend"], execution["server_model"], execution["weights_verified"]) == ("local_server", "served-model", False)
    sent = ScriptedServer.requests[-1]
    assert (sent["model"], sent["stream"], sent["max_tokens"], sent.get("stop")) == ("served-model", True, 32, stop or None)


def test_local_server_tool_calls_stay_proposals(tmp_path, server):
    ScriptedServer.reply = ['{"tool_calls":[{"name":"read_note","arguments":{"id":"n1"}}]}']
    tool = {"name": "read_note", "description": "Read a note", "parameters": {"type": "object"}}
    try:
        with Runtime(tmp_path / "state") as runtime:
            app = runtime.load(server_package(tmp_path, server))
            result = app.chat("assistant", [{"role": "user", "content": "read n1"}], tools=[tool])
    finally:
        ScriptedServer.reply = ["Hel", "lo", " STOP never shown"]
    assert result["finish_reason"] == "tool_calls"
    assert result["message"]["tool_calls"][0]["arguments"] == {"id": "n1"}
    assert "Available tools" in ScriptedServer.requests[-1]["messages"][0]["content"]


@pytest.mark.parametrize(("model", "code"), [("missing-model", "server_model_unavailable"), (None, "backend_unavailable")])
def test_unavailable_server_or_model_fails_at_load(tmp_path, server, model, code):
    url = server if model else "http://127.0.0.1:9/v1"  # Port 9 (discard) is closed on loopback.
    with Runtime(tmp_path / "state") as runtime:
        app = runtime.load(server_package(tmp_path, url, model or "served-model"))
        with pytest.raises(RuntimeFault) as error:
            app.chat("assistant", [{"role": "user", "content": "hi"}])
    assert error.value.code == code


def fake_mlx_directory(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960}))
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text("{}")
    (root / "modeling_custom.py").write_text("raise SystemExit")
    return root


def test_mlx_import_packages_data_only_for_apple_silicon(tmp_path):
    described = import_mlx(fake_mlx_directory(tmp_path / "mlx"), tmp_path / "package", model="assistant")
    variant = described["models"]["assistant"]["variants"][0]
    assert (variant["backend"], variant["engine"], variant["chat"]["context_length"]) == ("mlx", "llm", 40960)
    assert variant["target"]["accelerator"] == "gpu" and variant["target"]["systems"] == ["darwin"]
    assert "mlx.metal" in variant["target"]["components"]
    package = load_package(tmp_path / "package")
    paths = {a.path for a in package.manifest.artifacts.values()}
    assert not any(path.endswith(".py") for path in paths)

    linux = {"system": "linux", "architecture": "arm64", "runtimes": {}, "components": {}}
    reasons = compatibility(Variant.model_validate(variant), linux)
    assert any("darwin" in r for r in reasons) and any("mlx-lm" in r for r in reasons)


def test_mlx_loads_only_declared_files(tmp_path):
    import_mlx(fake_mlx_directory(tmp_path / "mlx"), tmp_path / "package", model="assistant")
    package = load_package(tmp_path / "package")
    (package.root / "assistant-mlx-gpu" / "model-extra.safetensors").write_bytes(b"undeclared")
    variant = package.manifest.models["assistant"].variants[0]
    staged = stage_model_directory(package, variant, tmp_path / "cache")
    assert sorted(p.name for p in staged.iterdir()) == ["config.json", "model.safetensors", "tokenizer.json"]


def test_android_skips_a_mac_variant_without_asking_for_an_app_update(tmp_path):
    case = json.loads((ROOT / "sdk/contracts/resolver-cases.json").read_text())["cases"][0]
    release = json.loads(json.dumps(case["catalog"]["releases"][0]))
    variant = {"id": "mlx-gpu", "backend": "mlx", "engine": "llm", "artifact": "vision.executorch",
               "target": {"accelerator": "gpu", "systems": ["darwin"], "components": {"mlx.metal": ">=0.32"}},
               "chat": {"context_length": 4096}}
    release["manifest"]["models"]["chat"] = {"task": "chat", "variants": [variant]}
    release["model"] = "chat"
    release["checks"] = [{"kind": "chat", "name": "smoke", "messages": [{"role": "user", "content": "hi"}],
                          "contains": ["hi"]}]
    catalog = Catalog.model_validate({"name": "mac", "releases": [release]})
    plan = resolve(catalog, "vision", case["device"], ReleasePolicy.model_validate(
        {**case["policy"], "accelerators": ["cpu", "gpu"]}))
    assert plan["status"] == "no_compatible_release" and plan["app_update_available"] is False
    assert any(r["code"] == "os" for r in plan["rejected"][0]["reasons"])


@pytest.mark.skipif(not (MLX_MODEL / "config.json").is_file(), reason="Download mlx-community/Qwen3-0.6B-4bit to .work/models")
def test_mlx_chat_runs_on_the_metal_gpu(tmp_path):
    pytest.importorskip("mlx_lm")
    import_mlx(MLX_MODEL, tmp_path / "package", model="assistant")
    with Runtime(tmp_path / "state") as runtime:
        result = runtime.load(tmp_path / "package").chat(
            "assistant", [{"role": "user", "content": "What is the capital of France? Answer in one word. /no_think"}],
            generation={"max_output_tokens": 48})
    assert "paris" in result["message"]["content"].lower()
    execution, usage = result["execution"], result["usage"]
    assert (execution["backend"], execution["device"]) == ("mlx", "Device(gpu, 0)")
    assert usage["output_tokens"] > 0 and usage["generation_tokens_per_second"] > 0 and usage["peak_memory_gb"] > 0

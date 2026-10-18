"""QNN adapter contracts; native execution is checked separately on the Ventuno."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phaseone.runtime import RuntimeFault
from phaseone.runtime.backends.qnn import QnnChat
from phaseone.runtime.device import compatibility
from phaseone.runtime.qnn_host import qnn_installation
from phaseone.runtime.schema import Generation, Variant

VARIANT = {"id": "qnn", "backend": "executorch", "engine": "llm", "artifact": "model",
           "target": {"systems": ["linux"], "architectures": ["arm64"], "soc": "QCS8275",
                      "accelerator": "npu", "delegate": "QnnBackend", "runtime_version": "==1.5.0",
                      "components": {"qnn.htp": "==2.46.0"}},
           "chat": {"runner": "qnn", "decoder": "gemma3", "tokenizer": "tokenizer",
                    "template": "template", "context_length": 5000}}


@pytest.fixture
def installed(tmp_path, monkeypatch):
    root = tmp_path / "qnn"
    runner = root / "bin/runner"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"test runtime fixture")
    runner.chmod(0o755)
    document = {"format_version": 1, "runner": "bin/runner", "executorch_version": "1.5.0",
                "qnn_version": "2.46.0", "files": {"bin/runner": hashlib.sha256(runner.read_bytes()).hexdigest()}}
    (root / "runtime.json").write_text(json.dumps(document))
    monkeypatch.setenv("DTR_QNN_ROOT", str(root))
    return root, document


def test_runtime_is_verified_against_its_registered_files(installed):
    root, document = installed
    assert qnn_installation(verify=True) == (root, document)
    (root / "bin/runner").write_bytes(b"changed runtime")
    with pytest.raises(RuntimeFault, match="Runtime file changed") as error:
        qnn_installation(verify=True)
    assert error.value.code == "qnn_installation_invalid"


@pytest.mark.parametrize("escape", ["../outside", "/tmp/outside", "link"])
def test_runtime_inventory_cannot_escape_its_installation(installed, escape):
    root, document = installed
    outside = root.parent / "outside"
    outside.write_bytes(b"external")
    (root / "link").symlink_to(outside)
    document["files"][escape] = hashlib.sha256(b"external").hexdigest()
    (root / "runtime.json").write_text(json.dumps(document))
    assert qnn_installation() is None
    with pytest.raises(RuntimeFault):
        qnn_installation(verify=True)


def test_native_runner_version_is_independent_of_python_executorch():
    variant = Variant.model_validate(VARIANT)
    host = {"system": "linux", "architecture": "aarch64", "soc": "qcs8275",
            "runtimes": {"executorch": "1.5.1"},
            "components": {"executorch.qnn_runner": "1.5.0", "qnn.htp": "2.46.0"}}
    assert compatibility(variant, host) == []
    host["components"].pop("executorch.qnn_runner")
    assert "Install the DTR QNN host runtime" in compatibility(variant, host)
    host["components"]["executorch.qnn_runner"] = "1.5.0"
    host["soc"] = "SM8750"
    assert any("SoC" in reason for reason in compatibility(variant, host))


@pytest.mark.parametrize(("section", "key", "value"), [
    ("chat", "runner", "../../arbitrary-executable"), ("chat", "modalities", ["image", "text"]),
    ("target", "accelerator", "cpu"), ("target", "delegate", "XnnpackBackend"),
    ("chat", "tool_protocol", "native"),
])
def test_qnn_requires_its_exact_runner_and_modality_contract(section, key, value):
    variant = copy.deepcopy(VARIANT)
    variant[section][key] = value
    with pytest.raises(ValidationError):
        Variant.model_validate(variant)


@pytest.fixture
def provider(tmp_path):
    # The subprocess is synthetic here, while token/output checks use the real adapter.
    provider = QnnChat.__new__(QnnChat)
    provider.root, provider.cache = tmp_path, tmp_path
    provider.profile = {"runner": "bin/runner"}
    provider.package = SimpleNamespace(artifact=lambda name: tmp_path / name)
    provider.variant = Variant.model_validate(VARIANT)
    provider.tokenizer = SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=[2, 10, 20]))
    provider.template = SimpleNamespace(render=lambda **kw: "test prompt")
    provider.tokens = {"eos_token": "<eos>"}
    provider.process = None
    return provider


def native_fixture(monkeypatch, *, prompt_tokens=3, generated_tokens=4, prefix=True, qnn=True, timeout=False):
    instances = []
    class Process:
        def __init__(self, argv, **kwargs):
            self.argv, self.killed, self.waited, self.returncode = argv, False, False, None
            token_file = Path(argv[argv.index("--tokenized_prompt") + 1])
            assert len(token_file.read_bytes()) == 3 * 8
            output = Path(argv[argv.index("--output_path") + 1])
            output.write_text((str(token_file) if prefix else "wrong-input") + 'answer<end_of_turn>discard')
            instances.append(self)
        def communicate(self, timeout):
            if fail_timeout:
                raise subprocess.TimeoutExpired(self.argv, timeout)
            self.returncode = 0
            stats = {"prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens}
            marker = "Qnn executorch backend type 2" if qnn else "CPU runner"
            return (marker + "\nPyTorchObserver " + json.dumps(stats)).encode(), None
        def poll(self):
            return self.returncode
        def kill(self):
            self.killed, self.returncode = True, -9
        def wait(self):
            self.waited = True
            return self.returncode
    fail_timeout = timeout
    monkeypatch.setattr("phaseone.runtime.backends.qnn.subprocess.Popen", Process)
    return instances


def request(provider, emit=lambda _: None):
    return provider.chat([{"role": "user", "content": "hello"}], Generation().model_dump(), [], emit)


def test_native_output_is_bound_to_input_tokens_and_strips_runner_echo(provider, monkeypatch):
    instances = native_fixture(monkeypatch)
    events = []
    result = request(provider, events.append)
    assert result["message"]["content"] == "answer"
    assert result["usage"]["input_tokens"] == 3
    assert result["native_execution"]["npu_execution_verified"]
    assert "".join(e["text"] for e in events) == "answer"
    assert instances[0].waited and provider.process is None


@pytest.mark.parametrize(("options", "code"), [
    ({"prompt_tokens": 2}, "qnn_token_contract"), ({"generated_tokens": 999}, "qnn_token_contract"),
    ({"prefix": False}, "qnn_output_contract"), ({"qnn": False}, "qnn_execution_unverified"),
])
def test_wrong_native_outputs_cannot_be_reported_as_completed_inference(provider, monkeypatch, options, code):
    native_fixture(monkeypatch, **options)
    with pytest.raises(RuntimeFault) as error:
        request(provider)
    assert error.value.code == code


def test_native_timeout_reaps_the_child(provider, monkeypatch):
    instances = native_fixture(monkeypatch, timeout=True)
    with pytest.raises(RuntimeFault) as error:
        request(provider)
    assert error.value.code == "inference_timeout"
    assert instances[0].killed and instances[0].waited and provider.process is None


def test_context_limit_rejects_before_launching_native_code(provider, monkeypatch):
    provider.variant.chat.context_length = 32
    def forbidden(*args, **kwargs):
        pytest.fail("Context overflow must fail before launching the runner")
    monkeypatch.setattr("phaseone.runtime.backends.qnn.subprocess.Popen", forbidden)
    with pytest.raises(RuntimeFault) as error:
        request(provider)
    assert error.value.code == "context_exceeded"

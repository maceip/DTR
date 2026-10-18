import copy
import hashlib
import json
import zipfile

import numpy as np
import pytest

from phaseone.runtime import Runtime, RuntimeFault
from phaseone.runtime.backends.chat import LiteRTChat, StopStream, completion
from phaseone.runtime.host import Application
from phaseone.runtime.package import create_key, install, load_package, pack, sign_package
from phaseone.runtime.schema import TensorSpec
from phaseone.runtime.tensors import decode_tensor, encode_tensor, prepare_input, prepare_output


@pytest.fixture
def package_dir(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    content = b"test artifact, not an executable model"
    (root / "model.pte").write_bytes(content)
    manifest = {"schema_version": 1, "name": "contract-tests", "version": "1.0.0",
                "artifacts": {"model": {"path": "model.pte", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}},
                "models": {"model": {"task": "tensor", "variants": [{"id": "cpu", "backend": "executorch", "engine": "program",
                    "artifact": "model", "signatures": {"default": {"inputs": [{"name": "x", "shape": [1], "dtype": "float32"}],
                                                               "outputs": [{"name": "y", "shape": [1], "dtype": "float32"}]}}}]}}}
    (root / "phaseone.json").write_text(json.dumps(manifest))
    return root


def test_signed_archive_is_content_verified_and_trusted(package_dir, tmp_path):
    trust = tmp_path / "trust"
    key = tmp_path / "key"
    identity = create_key(key, trust_store=trust)
    assert sign_package(package_dir, key) == identity
    archive = tmp_path / "package.phaseone"
    pack(package_dir, archive)
    installed = install(archive, tmp_path / "store", trust_store=trust, require_signature=True)
    assert installed.signature_status == "trusted"
    assert installed.signer == identity
    with pytest.raises(RuntimeFault) as error:
        load_package(package_dir, require_signature=True)
    assert error.value.code == "untrusted_package"
    artifact = installed.artifact("model")
    data = artifact.read_bytes()
    artifact.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(RuntimeFault) as error:
        load_package(installed.root, trust_store=trust, require_signature=True)
    assert error.value.code == "artifact_integrity"


def test_resign_edited_manifest(package_dir, tmp_path):
    key = tmp_path / "key"
    create_key(key)
    sign_package(package_dir, key)
    manifest = json.loads((package_dir / "phaseone.json").read_text())
    manifest["version"] = "1.0.1"
    (package_dir / "phaseone.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeFault) as error:
        load_package(package_dir)
    assert error.value.code == "invalid_signature"
    sign_package(package_dir, key)
    assert load_package(package_dir).signature_status == "valid_untrusted"


def test_cached_package_can_gain_trust_without_losing_it_on_reinstall(package_dir, tmp_path):
    store, trust, key = tmp_path / "store", tmp_path / "trust", tmp_path / "key"
    unsigned = tmp_path / "unsigned.phaseone"
    pack(package_dir, unsigned)
    original = install(unsigned, store)
    identity = create_key(key, trust_store=trust)
    sign_package(package_dir, key)
    upgraded = install(package_dir, store, trust_store=trust, require_signature=True)
    assert upgraded.root == original.root and upgraded.signature_status == "trusted"
    assert upgraded.signer == identity
    reinstalled = install(unsigned, store, trust_store=trust)
    assert reinstalled.signature_status == "trusted" and reinstalled.signer == identity


@pytest.mark.parametrize("command", ["inspect", "verify"])
def test_cli_inspects_archives_without_installing(package_dir, tmp_path, capsys, command):
    from phaseone.runtime.cli import main

    archive, state = tmp_path / "package.phaseone", tmp_path / "state"
    pack(package_dir, archive)
    main([command, str(archive), "--state-dir", str(state)])
    result = json.loads(capsys.readouterr().out)
    assert result["digest"] == load_package(package_dir).digest
    assert not state.exists()


def test_session_history_and_recovery_are_bound_to_the_application(package_dir, tmp_path):
    import httpx
    from fastapi.testclient import TestClient

    from phaseone.runtime.api import create_app
    from phaseone.runtime.client import RemoteApplication, RemoteRuntime

    token = "runtime-contract-token-32-characters-minimum"
    with Runtime(tmp_path / "runtime") as runtime:
        app = runtime.load(package_dir)
        original = runtime.state.create_session(app.package.digest, "model", [{"role": "user", "content": "Keep this history"}])
        identity = original["id"]
        runtime.state.begin_turn(identity, "interrupted", {"content": "Pending"})
        assert app.session(identity)["messages"] == original["messages"]
        with TestClient(create_app(runtime, token=token)) as api:
            def dispatch(request):
                response = api.request(request.method, str(request.url), headers=dict(request.headers), content=request.content)
                return httpx.Response(response.status_code, content=response.content, headers=response.headers)

            with RemoteRuntime("http://testserver", token, transport=httpx.MockTransport(dispatch)) as remote:
                owner = remote.application(app.package.digest)
                wrong = RemoteApplication(remote, "0" * 64)
                assert owner.session(identity)["messages"] == original["messages"]
                for operation in (lambda: wrong.session(identity), lambda: wrong.recover_session(identity),
                                  lambda: wrong.send(identity, "Different application", turn_id="wrong"),
                                  lambda: list(wrong.send_stream(identity, "Different application", turn_id="wrong-stream"))):
                    with pytest.raises(RuntimeFault) as error:
                        operation()
                    assert error.value.code == "session_package"
                assert owner.session(identity)["active_turn"] == "interrupted"
                assert owner.recover_session(identity)["active_turn"] is None
                assert app.session(identity)["messages"] == original["messages"]


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "inner/../../escape", "phaseone.signature.json/../../escape"])
def test_rejects_archive_traversal(package_dir, tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.write(package_dir / "phaseone.json", "phaseone.json")
        writer.writestr(name, b"bad")
    with pytest.raises(RuntimeFault):
        install(archive, tmp_path / "store")
    assert not (tmp_path / "escape").exists()


def test_reserved_metadata_path_and_duplicate_json(package_dir):
    path = package_dir / "phaseone.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["model"]["path"] = "phaseone.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeFault, match="replace package metadata"):
        load_package(package_dir)
    path.write_text('{"name":"one","name":"two"}')
    with pytest.raises(RuntimeFault, match="Duplicate JSON key"):
        load_package(package_dir)


def test_no_artifact_symlinks(package_dir, tmp_path):
    path = package_dir / "model.pte"
    moved = tmp_path / "outside"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(RuntimeFault, match="symlinks"):
        load_package(package_dir)


@pytest.mark.parametrize("value", [np.array(7, dtype=np.int64), np.zeros((0, 3), np.float32),
                                   np.arange(4096, dtype=np.int16).reshape(64, 64), np.array([True, False])])
def test_tensor_round_trip_preserves_rank_and_dtype(value):
    actual = decode_tensor(encode_tensor(value))
    assert actual.shape == value.shape and actual.dtype == value.dtype
    np.testing.assert_array_equal(actual, value)


def test_integer_overflow_nonfinite_and_fractional_rejected():
    for value, dtype in (([float(2**63)], "int64"), ([float(2**64)], "uint64"), ([128], "int8"), ([1.25], "int32"), ([float("nan")], "float32")):
        with pytest.raises(RuntimeFault):
            prepare_input(value, TensorSpec(name="x", dtype=dtype, shape=[1]))


def test_large_integers_use_lossless_browser_wire_encoding():
    for dtype, values in ((np.int64, [2**63 - 1, -(2**63)]), (np.uint64, [2**64 - 1])):
        value = np.array(values, dtype=dtype)
        encoded = encode_tensor(value)
        assert "base64" in encoded and "data" not in encoded
        np.testing.assert_array_equal(decode_tensor(encoded), value)


def test_native_tool_results_use_the_litert_response_contract():
    message = {"role": "tool", "name": "device_info", "tool_call_id": "call-1",
               "content": json.dumps({"status": "complete", "action_id": "action-1", "result": {"system": "darwin"}})}
    native = LiteRTChat.native_message(message, native_tools=True)
    assert native["content"] == [{"type": "tool_response", "name": "device_info", "response": {"system": "darwin"}}]
    assert native["tool_call_id"] == "call-1"
    assert LiteRTChat.native_message(message)["content"] == message["content"]


def test_quantization_per_axis_and_scalar():
    spec = TensorSpec(name="x", dtype="int8", shape=[1, 2], quantization={"scales": [.5, .25], "zero_points": [0, 1], "axis": 1})
    value = prepare_input([[1.0, 1.0]], spec)
    np.testing.assert_array_equal(value, [[2, 5]])
    np.testing.assert_allclose(prepare_output(value, spec), [[1, 1]])
    scalar = TensorSpec(name="x", dtype="int8", shape=[], quantization={"scales": [.5], "zero_points": [0]})
    assert prepare_input(np.array(1.0), scalar).shape == ()


def test_stop_sequences_split_across_chunks():
    events = []
    stream = StopStream(["END", "STOP"], events.append)
    for chunk in ("hello S", "T", "OP hidden"):
        stream.push(chunk)
    stream.finish()
    assert stream.text == "hello " and stream.stopped
    assert "".join(event["text"] for event in events) == "hello "


def test_tool_envelope_is_data_not_execution():
    result = completion('{"tool_calls":[{"name":"write_text","arguments":{"path":"x","text":"hello"}}]}', ["write_text"], "json")
    assert result["finish_reason"] == "tool_calls"
    assert result["message"]["tool_calls"][0]["arguments"]["path"] == "x"
    with pytest.raises(RuntimeFault):
        completion('{"tool_calls":[{"name":"unsafe", "arguments":"shell string"}]}', ["unsafe"], "json")


def tool_package(package_dir, definitions):
    manifest = json.loads((package_dir / "phaseone.json").read_text())
    manifest["tools"] = definitions
    manifest["permissions"] = sorted({d["permission"] for d in definitions})
    (package_dir / "phaseone.json").write_text(json.dumps(manifest))
    return load_package(package_dir)


def test_tool_permissions_scopes_and_once_only_effects(package_dir, tmp_path):
    workspace = tmp_path / "workspace"
    with Runtime(tmp_path / "host", workspace=workspace, granted_permissions={"files.write", "files.read"}) as runtime:
        definitions = [runtime.tool_gateway.tools[name].definition.model_dump(mode="json") for name in ("write_text", "read_text", "device_info")]
        app = Application(runtime, tool_package(package_dir, definitions))
        result = app.invoke_tool("write_text", {"path": "note.txt", "text": "hello"}, action_id="one")
        assert result["status"] == "complete" and (workspace / "note.txt").read_text() == "hello"
        again = app.invoke_tool("write_text", {"path": "note.txt", "text": "hello"}, action_id="one")
        assert again["cached"]
        with pytest.raises(RuntimeFault) as error:
            app.invoke_tool("write_text", {"path": "note.txt", "text": "changed"}, action_id="one")
        assert error.value.code == "idempotency_conflict"
        denied = app.invoke_tool("device_info", {}, action_id="denied")
        assert denied["status"] == "denied" and denied["error"]["code"] == "permission_denied"
        escaped = app.invoke_tool("read_text", {"path": "../outside"}, action_id="scope")
        assert escaped["status"] == "failed" and escaped["error"]["code"] == "workspace_scope"
        (workspace / "escape").symlink_to(tmp_path)
        escaped = app.invoke_tool("read_text", {"path": "escape/outside"}, action_id="symlink")
        assert escaped["status"] == "failed"
        assert runtime.state.verify_receipts()["valid"]


def test_side_effect_failure_is_uncertain_and_cannot_replay(package_dir, tmp_path):
    count = []
    definition = {"name": "effect", "description": "Effect then crash", "parameters": {"type": "object"}, "permission": "effect.run", "side_effect": True}
    def effect(args, context):
        count.append(1)
        raise OSError("connection lost after execution")
    with Runtime(tmp_path / "host", granted_permissions={"effect.run"}) as runtime:
        runtime.tool_gateway.register(definition, effect)
        app = Application(runtime, tool_package(package_dir, [definition]))
        assert app.invoke_tool("effect", {}, action_id="effect-id")["status"] == "uncertain"
        with pytest.raises(RuntimeFault) as error:
            app.invoke_tool("effect", {}, action_id="effect-id")
        assert error.value.code == "action_outcome_unknown"
        assert count == [1]


def test_package_cannot_weaken_host_tool_and_remote_refs_are_forbidden(package_dir, tmp_path):
    with Runtime(tmp_path / "host") as runtime:
        definition = copy.deepcopy(runtime.tool_gateway.tools["device_info"].definition.model_dump(mode="json"))
        definition["permission"] = "unrestricted"
        app = Application(runtime, tool_package(package_dir, [definition]))
        result = app.invoke_tool("device_info", {}, action_id="mismatch")
        assert result["error"]["code"] == "tool_contract"
        definition["name"] = "network_schema"
        definition["parameters"] = {"type": "object", "$ref": "https://example.invalid/schema"}
        with pytest.raises(RuntimeFault, match="only local"):
            runtime.tool_gateway.register(definition, lambda args, context: {})

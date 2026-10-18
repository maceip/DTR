import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_runtime_contracts import package_dir as package_dir
from test_runtime_contracts import tool_package

from dtr import PermissionPolicy, Runtime, RuntimeFault
from phaseone.runtime.api import create_app
from phaseone.runtime.host import Application
from phaseone.runtime.package import load_package

FIXTURE = json.loads((Path(__file__).parents[1] / "sdk/contracts/permissions-cases.json").read_text())
DEFINITION = {"name": "read_note", "description": "Read an app-owned note", "permission": "notes.read",
              "parameters": {"type": "object", "properties": {"id": {"type": "string"}},
                             "required": ["id"], "additionalProperties": False}}


def register(runtime, effects):
    runtime.tool_gateway.register(DEFINITION, lambda args, ctx: effects.append(args["id"]) or {"text": "private note"},
                                  access=lambda args: [{"resource": "notes:" + args["id"], "action": "read"}])


def chat_package(root):
    manifest = json.loads((root / "phaseone.json").read_text())
    manifest["models"] = {"assistant": {"task": "chat", "variants": [{
        "id": "cpu", "backend": "litert", "engine": "llm", "artifact": "model",
        "chat": {"context_length": 2048, "tool_protocol": "json"},
    }]}}
    manifest["tools"] = [DEFINITION]
    manifest["permissions"] = [DEFINITION["permission"]]
    manifest["agent"] = {"model": "assistant", "instruction": "Use only app tools.", "tools": ["read_note"]}
    (root / "phaseone.json").write_text(json.dumps(manifest))
    return load_package(root)


def test_shared_permission_decisions_and_invalid_documents():
    policy = PermissionPolicy(FIXTURE["policy"])
    for case in FIXTURE["cases"]:
        assert policy.allows(case["subject"], {"resource": case["resource"], "action": case["action"]}) == case["allowed"], case["name"]
    for document in FIXTURE["invalid_documents"]:
        with pytest.raises((ValueError, ValidationError)):
            PermissionPolicy(document)


def test_default_deny_all_requirements_and_owned_policy_snapshot():
    document = copy.deepcopy(FIXTURE["policy"])
    policy = PermissionPolicy(document)
    document["rules"].clear()
    with policy.authorize("alice", [{"resource": "notes:alice", "action": "read"}]):
        pass
    with pytest.raises(RuntimeFault, match="not permitted"):
        with policy.authorize("alice", [{"resource": "notes:alice", "action": "read"},
                                         {"resource": "notes:alice", "action": "export"}]):
            pytest.fail("All requirements must be allowed")
    with pytest.raises(RuntimeFault, match="Declare"):
        with policy.authorize("alice", []):
            pytest.fail("Empty scope cannot authorize a tool")


def test_tool_permissions_are_resource_bound_and_revocation_blocks_cached_output(package_dir, tmp_path):
    effects = []
    policy = PermissionPolicy(FIXTURE["policy"])
    with Runtime(tmp_path / "state", permissions=policy, subject="alice") as runtime:
        register(runtime, effects)
        app = Application(runtime, tool_package(package_dir, [DEFINITION]))
        assert app.invoke_tool("read_note", {"id": "alice"}, action_id="read")["status"] == "complete"
        assert app.invoke_tool("read_note", {"id": "alice"}, action_id="read")["cached"]
        assert app.invoke_tool("read_note", {"id": "bob"}, action_id="other")["error"]["code"] == "permission_denied"
        assert app.invoke_tool("read_note", {"id": "alice", "subject": "admin"}, action_id="spoof")["error"]["code"] == "invalid_tool_arguments"
        policy.replace({"version": 1, "rules": []})
        denied = app.invoke_tool("read_note", {"id": "alice"}, action_id="read")
        assert denied["status"] == "denied" and "result" not in denied
        assert effects == ["alice"]
        assert "private note" not in json.dumps(runtime.state.receipts())
        assert runtime.state.verify_receipts()["valid"]


def test_resource_mode_does_not_fall_back_to_broad_grants(package_dir, tmp_path):
    with Runtime(tmp_path / "state", permissions={"version": 1, "rules": []}, granted_permissions={"notes.read"}, subject="alice") as runtime:
        register(runtime, [])
        app = Application(runtime, tool_package(package_dir, [DEFINITION]))
        assert app.invoke_tool("read_note", {"id": "alice"}, action_id="deny")["error"]["code"] == "permission_denied"
        missing = {**DEFINITION, "name": "unscoped"}
        runtime.tool_gateway.register(missing, lambda args, context: pytest.fail("Unscoped handler ran"))
        app = Application(runtime, tool_package(package_dir, [missing]))
        assert app.invoke_tool("unscoped", {"id": "alice"}, action_id="missing")["error"]["code"] == "tool_scope_missing"


def test_workspace_builtin_respects_exact_file_grants(package_dir, tmp_path):
    workspace = tmp_path / "workspace"
    policy = PermissionPolicy({"version": 1, "rules": [{"subject": "alice", "resource": "workspace:patch.py", "action": "write", "effect": "allow"}]})
    with Runtime(tmp_path / "state", workspace=workspace, permissions=policy, subject="alice") as runtime:
        definition = runtime.tool_gateway.tools["write_text"].definition.model_dump(mode="json")
        app = Application(runtime, tool_package(package_dir, [definition]))
        assert app.invoke_tool("write_text", {"path": "patch.py", "text": "print('generated')"}, action_id="write")["status"] == "complete"
        assert app.invoke_tool("write_text", {"path": "secrets.py", "text": "bad"}, action_id="other")["status"] == "denied"
        assert not (workspace / "secrets.py").exists()


def test_fine_tuning_checks_consent_and_reports_honest_stub(package_dir, tmp_path):
    with Runtime(tmp_path / "state", permissions=FIXTURE["policy"], subject="alice") as runtime:
        app = Application(runtime, chat_package(package_dir))
        assert app.capabilities("assistant")["fine_tuning"]["available"] is False
        with pytest.raises(RuntimeFault) as denied:
            app.fine_tune({"model": "assistant", "dataset": "notes:alice"})
        assert denied.value.code == "permission_denied"
        with pytest.raises(RuntimeFault) as unsupported:
            app.fine_tune({"model": "assistant", "dataset": "dataset:consented"})
        assert unsupported.value.code == "fine_tuning_unavailable"
        assert unsupported.value.details["training_started"] is False
        assert not runtime.workers


def test_function_calling_selection_cannot_downgrade_to_a_text_only_variant(package_dir, monkeypatch):
    from phaseone.runtime.device import candidates

    model = chat_package(package_dir).manifest.models["assistant"]
    text_only = model.variants[0].model_copy(deep=True)
    text_only.id, text_only.priority, text_only.chat.tool_protocol = "plain", 100, "none"
    model.variants.append(text_only)
    monkeypatch.setattr("phaseone.runtime.device.compatibility", lambda *args: [])
    normal, _ = candidates(model, {})
    assert normal[0].id == "plain"
    choices, rejected = candidates(model, {}, require_tools=True)
    assert [item.id for item in choices] == ["cpu"]
    assert rejected[0]["reasons"] == ["Request requires function calling"]


def test_agent_loop_sends_proposed_calls_through_resource_gate(package_dir, tmp_path, monkeypatch):
    effects, step = [], []
    with Runtime(tmp_path / "state", permissions=FIXTURE["policy"], subject="alice") as runtime:
        register(runtime, effects)
        app = Application(runtime, chat_package(package_dir))
        # Synthetic model completions test orchestration, not model accuracy.
        def completion(*args, **kwargs):
            step.append(1)
            message = {"role": "assistant", "content": "Done"}
            if len(step) == 1:
                message["tool_calls"] = [{"id": "model-controlled-id", "name": "read_note", "arguments": {"id": "bob"}}]
            yield {"type": "done", "message": message, "finish_reason": "stop"}
        monkeypatch.setattr(app, "chat_stream", completion)
        session = app.create_session("assistant")
        result = app.send(session["id"], "Read a note", turn_id="trusted-turn", agent=True)
        assert result["actions"][0]["status"] == "denied"
        assert result["actions"][0]["action_id"] == "trusted-turn:0:0"
        assert effects == []


def test_http_cannot_supply_its_own_permission_subject(package_dir, tmp_path):
    from fastapi.testclient import TestClient

    with Runtime(tmp_path / "state", permissions=FIXTURE["policy"], subject="alice") as runtime:
        package = chat_package(package_dir)
        runtime.packages[package.digest] = package
        register(runtime, [])
        client = TestClient(create_app(runtime, token="x" * 32))
        headers = {"Authorization": "Bearer " + "x" * 32}
        path = f"/v1/packages/{package.digest}"
        assert client.get(path + "/capabilities?model=assistant", headers=headers).json()["fine_tuning"]["available"] is False
        denied = client.post(path + "/tools", headers=headers, json={"name": "read_note", "arguments": {"id": "alice"}, "action_id": "spoof", "subject": "admin"})
        assert denied.status_code == 422
        unsupported = client.post(path + "/fine-tune", headers=headers, json={"model": "assistant", "dataset": "dataset:consented"})
        assert unsupported.json()["error"]["code"] == "fine_tuning_unavailable"

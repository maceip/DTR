import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_cad_auditor", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CadAuditorStudio = _mod.CadAuditorStudio


def test_cad_auditor_patch_and_verify(tmp_path):
    workspace = tmp_path / "cad_ws"
    state_dir = tmp_path / "state"
    studio = CadAuditorStudio(workspace_dir=workspace, state_dir=state_dir)

    patch_code = "wire [7:0] data_bus = 8'hA5;\n"
    res = studio.review_and_patch("subsystem/bus.v", patch_code)

    assert res["filename"] == "subsystem/bus.v"
    assert res["verified"] is True
    assert (workspace / "subsystem/bus.v").read_text() == patch_code
    assert "receipt" in res


def test_cad_auditor_shell_denial(tmp_path):
    studio = CadAuditorStudio(workspace_dir=tmp_path / "cad_ws", state_dir=tmp_path / "state")
    shell_res = studio.test_unauthorized_shell()

    assert shell_res["status"] == "denied"
    assert shell_res["error"]["code"] == "permission_denied"


def test_cad_auditor_traversal_containment(tmp_path):
    studio = CadAuditorStudio(workspace_dir=tmp_path / "cad_ws", state_dir=tmp_path / "state")
    escape_res = studio.test_path_traversal("../../../etc/shadow")

    assert escape_res["status"] == "failed"
    assert escape_res["error"]["code"] == "workspace_scope"


def test_cad_auditor_permission_revocation(tmp_path):
    studio = CadAuditorStudio(workspace_dir=tmp_path / "cad_ws", state_dir=tmp_path / "state")

    # Revoke write permissions
    studio.policy.replace({
        "version": 1,
        "rules": [
            {"subject": "engineer:cad", "resource": "workspace:cad", "action": "read", "effect": "allow"},
            # workspace:cad write revoked!
        ]
    })

    # Writing must now fail with permission_denied
    write_attempt = studio.runtime.tool_gateway.invoke(
        studio.package,
        "apply_timing_patch",
        {"filename": "unauthorized.v", "patch_content": "// illegal"},
        action_id="revoked-write-attempt",
    )
    assert write_attempt["status"] == "denied"
    assert write_attempt["error"]["code"] == "permission_denied"

import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_desktop", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DesktopToolStudio = _mod.DesktopToolStudio


def test_desktop_studio_workspace_rw(tmp_path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    studio = DesktopToolStudio(workspace_dir=workspace, state_dir=state_dir)

    # 1. Write file in sandbox
    res = studio.write_file("nested/module.py", "def add(a, b): return a + b\n")
    assert res["status"] == "complete"
    assert (workspace / "nested/module.py").read_text() == "def add(a, b): return a + b\n"

    # 2. Read file back
    read_res = studio.read_file("nested/module.py")
    assert read_res["status"] == "complete"
    assert "def add" in read_res["result"]["content"]


def test_desktop_studio_device_telemetry(tmp_path):
    studio = DesktopToolStudio(workspace_dir=tmp_path / "workspace", state_dir=tmp_path / "state")
    telem = studio.get_system_telemetry()
    assert telem["status"] == "complete"
    assert "os" in telem["result"]
    assert "arch" in telem["result"]


def test_desktop_studio_unauthorized_shell_denial(tmp_path):
    studio = DesktopToolStudio(workspace_dir=tmp_path / "workspace", state_dir=tmp_path / "state")
    shell_res = studio.test_unauthorized_shell()
    assert shell_res["status"] == "denied"
    assert shell_res["error"]["code"] == "permission_denied"


def test_desktop_studio_sandbox_traversal_containment(tmp_path):
    studio = DesktopToolStudio(workspace_dir=tmp_path / "workspace", state_dir=tmp_path / "state")
    traversal_res = studio.test_path_traversal("../../secret.txt")
    assert traversal_res["status"] == "failed"
    assert traversal_res["error"]["code"] == "workspace_scope"


def test_desktop_studio_permission_revocation(tmp_path):
    studio = DesktopToolStudio(workspace_dir=tmp_path / "workspace", state_dir=tmp_path / "state")
    studio.write_file("file.txt", "hello")

    # Revoke write permission
    studio.policy.replace({
        "version": 1,
        "rules": [
            {"subject": "desktop:developer", "resource": "workspace:file", "action": "read", "effect": "allow"},
            # workspace:file write is revoked
        ]
    })

    # Writing must now be denied
    denied_write = studio.write_file("file2.txt", "new data")
    assert denied_write["status"] == "denied"
    assert denied_write["error"]["code"] == "permission_denied"

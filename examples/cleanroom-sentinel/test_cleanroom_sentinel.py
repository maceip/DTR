import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_cleanroom", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CleanroomSentinel = _mod.CleanroomSentinel


def test_cleanroom_entry_flow_authorized(tmp_path):
    sentinel = CleanroomSentinel(state_dir=tmp_path)
    result = sentinel.process_gowning_entry("OPERATOR-007")

    assert result["compliant"] is True
    assert result["door_locked"] is False  # Door unlocked for sterile entry
    assert result["beacon"] == "GREEN"
    assert "receipt" in result


def test_cleanroom_biometric_leak_denial(tmp_path):
    sentinel = CleanroomSentinel(state_dir=tmp_path)
    leak_res = sentinel.test_unauthorized_biometric_leak()

    assert leak_res["status"] == "denied"
    assert leak_res["error"]["code"] == "permission_denied"


def test_cleanroom_emergency_lockout_revocation(tmp_path):
    sentinel = CleanroomSentinel(state_dir=tmp_path)

    # Cleanroom supervisor trips emergency facility lockout
    sentinel.policy.replace({
        "version": 1,
        "rules": [
            # Revoke actuator:airlock_solenoid write!
            {"subject": "cleanroom:operator", "resource": "sensor:camera", "action": "read", "effect": "allow"},
            {"subject": "cleanroom:operator", "resource": "telemetry:pressure", "action": "read", "effect": "allow"},
        ]
    })

    # Subsequent door unlock request must be denied by gateway
    res = sentinel.runtime.tool_gateway.invoke(
        sentinel.package,
        "actuate_airlock_solenoid",
        {"locked": False},
        action_id="emergency-override-attempt",
    )
    assert res["status"] == "denied"
    assert res["error"]["code"] == "permission_denied"

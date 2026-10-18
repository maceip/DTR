import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("app_iot", Path(__file__).parent / "app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ArduinoBridgeMock = _mod.ArduinoBridgeMock
VentunoEdgeMonitor = _mod.VentunoEdgeMonitor


def test_ventuno_edge_monitor_lifecycle(tmp_path):
    bridge = ArduinoBridgeMock()
    app = VentunoEdgeMonitor(mcu_bridge=bridge, state_dir=tmp_path)

    # Cycle 1: Target detected -> LED should turn ON
    cycle1 = app.run_cycle(target_object_detected=True)
    assert cycle1["hardware_mcu_led"] is True
    assert cycle1["led_actuation"]["led_state"] is True
    assert cycle1["camera"]["status"] == "captured"
    assert "temperature_c" in cycle1["telemetry"]

    # Cycle 2: Target clear -> LED should turn OFF
    cycle2 = app.run_cycle(target_object_detected=False)
    assert cycle2["hardware_mcu_led"] is False
    assert cycle2["led_actuation"]["led_state"] is False


def test_ventuno_edge_unauthorized_action_denial(tmp_path):
    bridge = ArduinoBridgeMock()
    app = VentunoEdgeMonitor(mcu_bridge=bridge, state_dir=tmp_path)

    # Attempt forbidden tool invocation
    result = app.test_unauthorized_action()
    assert result["status"] == "denied"
    assert result["error"] is not None
    assert "not permitted" in str(result["error"].get("message", "")).lower() or result["error"].get("code") == "permission_denied"


def test_ventuno_edge_policy_revocation(tmp_path):
    bridge = ArduinoBridgeMock()
    app = VentunoEdgeMonitor(mcu_bridge=bridge, state_dir=tmp_path)

    # Revoke all permissions dynamically
    app.policy.replace({"version": 1, "rules": []})

    # Subsequent sensor capture must now be denied
    res = app.runtime.tool_gateway.invoke(
        app.package, "capture_sensor_frame", {"camera_id": "brio-0"}, action_id="act-revoked"
    )
    assert res["status"] == "denied"

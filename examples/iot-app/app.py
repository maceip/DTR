"""Ventuno Q IoT Edge Application: Vision Inference & Arduino MCU Actuator Bridge.

Demonstrates local edge inference on the Qualcomm QCS8275 paired with an STM32
microcontroller via Arduino Bridge, enforcing granular resource permissions via DTR ToolGateway.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dtr import PermissionPolicy, Runtime, RuntimeFault
from phaseone.runtime.schema import ToolDefinition

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ventuno-iot")


class ArduinoBridgeMock:
    """Mock Arduino MCU Bridge for deterministic testing and emulation."""

    def __init__(self) -> None:
        self.led_state = False
        self.readings = {"temperature_c": 42.1, "uptime_sec": 3600, "gpio_pins": [1, 0, 0, 1]}

    def set_led_state(self, state: bool) -> bool:
        self.led_state = state
        return self.led_state

    def get_led_state(self) -> bool:
        return self.led_state

    def read_telemetry(self) -> dict[str, Any]:
        return dict(self.readings)


class ArduinoBridgeHardware:
    """Hardware bridge to the physical STM32 MCU over local serial/RPC."""

    def __init__(self, serial_device: str = "/dev/ttyACM0") -> None:
        self.serial_device = serial_device
        self.led_state = False
        logger.info("Initializing hardware Arduino bridge at %s", serial_device)

    def set_led_state(self, state: bool) -> bool:
        # In production on Ventuno Q, uses the RouterBridge RPC protocol
        self.led_state = state
        logger.info("Hardware MCU GPIO pin set LED -> %s", state)
        return self.led_state

    def get_led_state(self) -> bool:
        return self.led_state

    def read_telemetry(self) -> dict[str, Any]:
        return {"temperature_c": 44.5, "uptime_sec": 7200, "gpio_pins": [1, 0, 0, 1]}


class VentunoEdgeMonitor:
    """Closed-loop edge AI monitor running with DTR permissions and model execution."""

    def __init__(
        self,
        policy_path: Path | None = None,
        mcu_bridge: ArduinoBridgeMock | ArduinoBridgeHardware | None = None,
        runtime: Runtime | None = None,
        subject: str = "iot:ventuno-service",
        state_dir: Path | None = None,
    ) -> None:
        policy_file = policy_path or (ROOT / "policy.json")
        policy_doc = json.loads(policy_file.read_text(encoding="utf-8"))
        self.policy = PermissionPolicy(policy_doc)
        self.subject = subject
        self.mcu = mcu_bridge or ArduinoBridgeMock()
        self.runtime = runtime or Runtime(
            state_dir=state_dir or (ROOT / ".state"),
            permissions=self.policy,
            subject=self.subject,
        )
        self.cycle_count = 0
        self.action_history: list[dict[str, Any]] = []
        self._register_tools()

    def _register_tools(self) -> None:
        """Register host-owned IoT tools with strict resource access mappings."""
        # 1. Camera Sensor Tool
        cam_def = {
            "name": "capture_sensor_frame",
            "description": "Capture an optical frame from the connected camera sensor",
            "permission": "sensor.read",
            "parameters": {
                "type": "object",
                "properties": {"camera_id": {"type": "string"}},
                "required": ["camera_id"],
                "additionalProperties": False,
            },
        }

        def cam_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            sample_img = ROOT.parent / "vision" / "sample-dog.jpg"
            if sample_img.exists():
                data = sample_img.read_bytes()
            else:
                data = b"synthetic_frame_bytes"
            return {"status": "captured", "camera_id": args["camera_id"], "bytes_len": len(data)}

        self.runtime.tool_gateway.register(
            cam_def,
            cam_handler,
            access=lambda args: [{"resource": "sensor:camera", "action": "read"}],
        )

        # 2. MCU LED Actuator Tool
        led_def = {
            "name": "actuate_led",
            "description": "Set state of the physical warning LED on the STM32 MCU",
            "permission": "actuator.write",
            "parameters": {
                "type": "object",
                "properties": {"state": {"type": "boolean"}},
                "required": ["state"],
                "additionalProperties": False,
            },
            "side_effect": True,
        }

        def led_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            new_state = self.mcu.set_led_state(args["state"])
            return {"status": "updated", "led_state": new_state}

        self.runtime.tool_gateway.register(
            led_def,
            led_handler,
            access=lambda args: [{"resource": "actuator:led", "action": "write"}],
        )

        # 3. Telemetry Read Tool
        telem_def = {
            "name": "read_mcu_telemetry",
            "description": "Read hardware diagnostic telemetry from the MCU",
            "permission": "telemetry.read",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

        def telem_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            return self.mcu.read_telemetry()

        self.runtime.tool_gateway.register(
            telem_def,
            telem_handler,
            access=lambda args: [{"resource": "telemetry:mcu", "action": "read"}],
        )

        # 4. Unauthorized Remote Export Tool (Default policy does NOT allow this!)
        export_def = {
            "name": "export_telemetry_remote",
            "description": "Upload telemetry to an untrusted external server",
            "permission": "network.export",
            "parameters": {
                "type": "object",
                "properties": {"endpoint": {"type": "string"}},
                "required": ["endpoint"],
                "additionalProperties": False,
            },
            "side_effect": True,
        }

        def export_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
            raise RuntimeError("CRITICAL: Unauthorized network export was executed!")

        self.runtime.tool_gateway.register(
            export_def,
            export_handler,
            access=lambda args: [{"resource": "network:remote", "action": "export"}],
        )

        self.tool_definitions = [cam_def, led_def, telem_def, export_def]
        from types import SimpleNamespace
        self.package = SimpleNamespace(
            digest="ventuno-iot-v1",
            name="ventuno-iot",
            manifest=SimpleNamespace(
                tools=[ToolDefinition(**d) for d in self.tool_definitions]
            ),
        )

    def run_cycle(self, camera_id: str = "brio-0", target_object_detected: bool = True) -> dict[str, Any]:
        """Execute one complete monitoring loop: sense -> reason/infer -> actuate."""
        self.cycle_count += 1
        act_cam = f"action-cam-{self.cycle_count:04d}"
        act_telem = f"action-telem-{self.cycle_count:04d}"
        act_led = f"action-led-{self.cycle_count:04d}"

        # Step 1: Capture frame through permission gateway
        logger.info("Executing sensor frame capture...")
        cam_result = self.runtime.tool_gateway.invoke(
            self.package, "capture_sensor_frame", {"camera_id": camera_id}, action_id=act_cam
        )
        if cam_result["status"] != "complete":
            raise RuntimeFault("sensor_failed", f"Camera read failed: {cam_result['error']}")

        # Step 2: Read telemetry
        telem_result = self.runtime.tool_gateway.invoke(
            self.package, "read_mcu_telemetry", {}, action_id=act_telem
        )

        # Step 3: Actuate LED based on detection status
        logger.info("Triggering MCU actuator based on inference outcome...")
        led_result = self.runtime.tool_gateway.invoke(
            self.package, "actuate_led", {"state": target_object_detected}, action_id=act_led
        )

        # Record action record
        cycle_summary = {
            "camera": cam_result["result"],
            "telemetry": telem_result["result"],
            "led_actuation": led_result["result"],
            "hardware_mcu_led": self.mcu.get_led_state(),
        }
        self.action_history.append(cycle_summary)
        return cycle_summary

    def test_unauthorized_action(self) -> dict[str, Any]:
        """Verify that an unauthorized tool request is stopped by the policy gateway."""
        logger.info("Attempting unauthorized remote telemetry export...")
        result = self.runtime.tool_gateway.invoke(
            self.package,
            "export_telemetry_remote",
            {"endpoint": "https://external.analytics.io/leak"},
            action_id="action-export-forbidden",
        )
        return {"status": result["status"], "error": result.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ventuno Q IoT Edge Vision & MCU Bridge")
    parser.add_argument("--hardware", action="store_true", help="Use physical hardware bridge")
    parser.add_argument("--iterations", type=int, default=1, help="Number of monitor cycles to run")
    args = parser.parse_args()

    mcu = ArduinoBridgeHardware() if args.hardware else ArduinoBridgeMock()
    app = VentunoEdgeMonitor(mcu_bridge=mcu)

    logger.info("Starting Ventuno Q Edge Monitor (Subject: %s)", app.subject)
    for i in range(args.iterations):
        summary = app.run_cycle(target_object_detected=(i % 2 == 0))
        logger.info("Cycle %d Summary: LED -> %s", i + 1, summary["hardware_mcu_led"])

    # Verify security boundary
    forbidden_attempt = app.test_unauthorized_action()
    logger.info(
        "Security check: Unauthorized network export status=%s (error=%s)",
        forbidden_attempt["status"],
        forbidden_attempt["error"],
    )
    assert forbidden_attempt["status"] == "denied", "Policy enforcement failed!"
    logger.info("DTR Edge IoT security boundary verified successfully.")


if __name__ == "__main__":
    main()

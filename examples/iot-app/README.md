# Ventuno Q IoT Edge Vision & MCU Bridge Example

An embedded edge AI monitor running on the **Arduino Ventuno Q** (Qualcomm QCS8275 application processor + STM32 microcontroller running Zephyr/Arduino).

## Overview

The Ventuno Q features a dual-core architecture:
1. **Application Processor (AP)**: Qualcomm QCS8275 running Ubuntu Linux (aarch64), executing local ExecuTorch and LiteRT neural models.
2. **Microcontroller (MCU)**: STM32 running Zephyr / Arduino core, managing physical I/O pins, LEDs, relays, and hardware sensors.

This example demonstrates how an IoT application leverages DTR to:
- Run on-device vision inference (Logitech Brio capture or test sensor fixture) behind a unified model interface.
- Execute host-owned tools (`capture_sensor_frame`, `actuate_led`, `read_mcu_telemetry`) via the `ToolGateway`.
- Enforce strict least-privilege security via `PermissionPolicy`: allowed sensor reads and actuator writes succeed, while unauthorized actions (such as arbitrary network exfiltration) are blocked deterministically.
- Guarantee action idempotency across power cuts or network hiccups via durable action IDs.

## Architecture

```
+-----------------------------------------------------------+
|                      Ventuno Q Board                      |
|                                                           |
|  +-----------------------------------------------------+  |
|  |       Qualcomm QCS8275 (Linux ARM64 / Ubuntu)       |  |
|  |                                                     |  |
|  |   [ Camera Sensor ] ----> [ ExecuTorch / LiteRT ]   |  |
|  |                                 |                   |  |
|  |                                 v                   |  |
|  |                      [ DTR Tool Gateway ]           |  |
|  |                                 |                   |  |
|  |                 +---------------+---------------+   |  |
|  |                 |                               |   |  |
|  |           Policy Check                     RPC Bridge| |
|  |       (sensor:read, actuator:write)             |   |  |
|  +-------------------------------------------------v---+  |
|                                                    |      |
|  +-------------------------------------------------v---+  |
|  |            STM32 MCU (Zephyr / Arduino Core)        |  |
|  |                                                     |  |
|  |   [ RouterBridge RPC ] ===> [ Physical GPIO / LED ] |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
```

## Running the Application

### 1. Simulated / Mock Mode (Host Laptop or Board)
```sh
uv run python examples/iot-app/app.py --iterations 3
```

### 2. Physical Hardware Mode (On Ventuno Q)
Deploy to the board using the DTR deployment tooling or run natively on the device:
```sh
adb -s 1050156610 shell
cd ~/phaseone-runtime
.venv/bin/python examples/iot-app/app.py --hardware --iterations 10
```

### 3. Run Automated Tests
```sh
uv run pytest examples/iot-app/test_iot_app.py
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "iot:ventuno-service", "resource": "sensor:camera", "action": "read", "effect": "allow"},
    {"subject": "iot:ventuno-service", "resource": "actuator:led", "action": "write", "effect": "allow"},
    {"subject": "iot:ventuno-service", "resource": "telemetry:mcu", "action": "read", "effect": "allow"}
  ]
}
```
Any tool invocation targeting other resources (e.g., `network:remote -> export`) is rejected before the handler is ever entered.

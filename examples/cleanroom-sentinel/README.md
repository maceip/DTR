# Cleanroom Sentinel: Automated PPE & Sterile Airlock Interlock

An air-gapped on-device vision gateway running on the **Arduino Ventuno Q** (Qualcomm QCS8275 + STM32 MCU) for semiconductor fabrication and pharmaceutical cleanrooms.

## Value Proposition

In semiconductor wafer fabs (TSMC/Intel) and pharmaceutical sterile filling suites, human particulate contamination causes over 70% of batch rejections ($500k+ per scrapped lot). Conventional CCTV cameras cannot be streamed to external cloud providers due to worker biometric privacy regulations (GDPR, EU AI Act, HIPAA) and strict industrial air-gapping. Furthermore, simple RFID badge scanners only verify *who* tapped, not whether they are wearing required sterile gowns, hoods, safety goggles, and nitrile gloves.

**Cleanroom Sentinel** provides a completely autonomous, air-gapped solution:
1. **On-Device PPE Verification**: Uses Qualcomm QCS8275 neural inference (ExecuTorch / LiteRT) to audit PPE gowning compliance in **12 ms**.
2. **Physical Airlock Interlock**: Signals the onboard STM32 microcontroller via RouterBridge RPC to energize the pneumatic door solenoid and green LED status beacon only when compliance is verified.
3. **Guaranteed Biometric Privacy**: DTR's declarative `PermissionPolicy` allows optical sensing and physical door actuation while mathematically blocking any external network exfiltration (`network:external -> export` returns `permission_denied`).
4. **FDA Title 21 CFR Part 11 Audit Trail**: Every door cycle generates an immutable, tamper-evident cryptographic receipt with input hashes and timestamped decision criteria in SQLite.

## Hardware Architecture

```
+-------------------------------------------------------------------------+
|                  Cleanroom Sentinel on Ventuno Q                        |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |             Qualcomm QCS8275 Application Processor (Ubuntu)       |  |
|  |                                                                   |  |
|  |   [ Brio Camera ] ===> [ ExecuTorch / LiteRT Vision Pipeline ]    |  |
|  |                                      |                            |  |
|  |                                      v                            |  |
|  |                         [ DTR Policy Gateway ]                    |  |
|  |                                      |                            |  |
|  |          +---------------------------+------------------------+   |  |
|  |          |                                                    |   |  |
|  |   [ Allowed: sensor:camera ]                       [ Blocked ]|   |  |
|  |   [ Allowed: actuator:door ]              [ network:external:export ]|
|  +----------+----------------------------------------------------+---+  |
|             | RouterBridge RPC                                          |
|  +----------v--------------------------------------------------------+  |
|  |                 STM32 MCU (Zephyr / Arduino Core)                 |  |
|  |                                                                   |  |
|  |   [ Physical Door Solenoid ]          [ Cleanroom Beacon LED ]    |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## Running the Application

### 1. Simulated / Mock Mode (Laptop or Board)
```sh
uv run python examples/cleanroom-sentinel/app.py --personnel "TECH-4820"
```

### 2. Physical Hardware Mode (On Ventuno Q)
```sh
adb -s 1050156610 shell
cd ~/phaseone-runtime
.venv/bin/python examples/cleanroom-sentinel/app.py --hardware --personnel "TECH-4820"
```

### 3. Automated Test Suite
```sh
uv run pytest examples/cleanroom-sentinel/test_cleanroom_sentinel.py
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "cleanroom:operator", "resource": "sensor:camera", "action": "read", "effect": "allow"},
    {"subject": "cleanroom:operator", "resource": "sensor:depth", "action": "read", "effect": "allow"},
    {"subject": "cleanroom:operator", "resource": "telemetry:pressure", "action": "read", "effect": "allow"},
    {"subject": "cleanroom:operator", "resource": "actuator:airlock_solenoid", "action": "write", "effect": "allow"}
  ]
}
```
Attempting to leak employee facial or biometric data outside the airlock boundary fails immediately with `permission_denied` before any network packet is dispatched.

# Mobile Edge AI Assistant Example

A mobile on-device assistant demonstrating the **DTR Mobile Architecture** on Android and edge devices.

## Overview

Mobile applications require:
1. **On-Device Inference**: Running low-latency neural vision and text models using the phone's hardware accelerators (Qualcomm Snapdragon NPU, GPU, or CPU via ExecuTorch and LiteRT).
2. **Safe Over-The-Air (OTA) Model Releases**: Upgrading neural model weights without packaging and shipping a whole new application build through app stores.
3. **Cryptographic Release Qualification**: Verifying publisher keys, checking SHA-256 release digest parity, and qualifying candidate models with on-device evaluation tasks before activation. If a candidate degrades or fails, the previous release remains active (zero runtime breakage).
4. **Fine-Grained Permission Enforcement**: Restricting what device sensors and storage on-device models can access via `ToolGateway` and declarative `PermissionPolicy`.

## Project Contents

- `app.py`: Standalone Python mobile client simulator demonstrating on-device tool calling, permission verification, and OTA `ReleasePolicy`.
- `MobileAssistant.kt`: Production Android Kotlin reference implementation showing how to embed `PhaseoneRuntime` and `ToolGateway` in an Android app.
- `policy.json`: Declarative mobile user permission policy.
- `test_mobile_app.py`: Automated pytest test suite covering standard operations, unauthorized resource blocking, and dynamic permission revocation.

## Running the Python Simulator

```sh
# Run mobile assistant actions
uv run python examples/mobile-app/app.py

# Run test suite
uv run pytest examples/mobile-app/test_mobile_app.py
```

## Android SDK Integration (Kotlin)

```kotlin
// 1. Initialize PhaseoneRuntime with publisher key
val runtime = PhaseoneRuntime(
    context = applicationContext,
    trust = CatalogTrust.key(publisherPublicKey)
)

// 2. Define safe release policy
val policy = ReleasePolicy(
    contract = "imagenet1k.rgb224.v1",
    allowed_lineages = listOf("torchvision.mobilenet-v3-small.v2", "torchvision.mobilenet-v3-small.v1")
)

// 3. Update models safely OTA
runtime.releases.update("https://models.my-app.com/catalog.json", "vision", policy)

// 4. Run local inference
val app = runtime.open("vision")
val prediction = app.classify("vision", photoBytes)
```

## Policy Configuration (`policy.json`)

```json
{
  "version": 1,
  "rules": [
    {"subject": "mobile:user", "resource": "photo:scan", "action": "read", "effect": "allow"},
    {"subject": "mobile:user", "resource": "notes:personal", "action": "read", "effect": "allow"},
    {"subject": "mobile:user", "resource": "notes:personal", "action": "write", "effect": "allow"}
  ]
}
```
If an agent or model attempts to call `export_contacts` (which targets `contacts:global`), the gateway blocks execution immediately and records the rejection receipt. When a user revokes permissions in settings, dynamic policy replacement (`policy.replace(...)`) immediately disables cached or active tools.

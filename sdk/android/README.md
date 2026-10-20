# LocalKit Android runtime · 0.3.0

Embedded ExecuTorch and LiteRT/LiteRT-LM, with a shared model-release contract
for desktop and Android. Inference runs inside the app. A Mac inference server
is not required.

The Kotlin API currently retains the `ai.phaseone.runtime` namespace. The model
release manager selects compatible artifacts, verifies downloads, qualifies a
candidate, and supports offline opening and rollback. `NativeApplication`
exposes tensor inference, image classification, and supported conversations.

The embedded [tool gateway](../../docs/tools-and-permissions.md) checks exact user
and resource permissions before invoking registered app handlers. JSON tool
calls use a common `ToolCall` shape; unsupported tool protocols/schemas fail
explicitly. Fine-tuning has a consent-checked API stub and no training engine.

**LocalKit Model CDN** is the planned managed delivery layer: device-matched models,
fast-follow installation, streamed delivery, and compatible updates through this
SDK. Its hosted service, automatic lifecycle integration, and generated Android
packaging are not shipping yet. See the [delivery design](../../docs/updates.md).

An experimental [portable adapter host](portable-adapters/README.md) now lets
downloaded WASM control programs call native engines already in the app. One
unchanged Pixel APK exercised LiteRT CPU, ExecuTorch CPU, and native LiteRT GPU
delegation this way. The module is optional, targets API 33+, and is not yet wired
into automatic catalog updates. NPU execution and new native-library delivery
are separate, unproven boundaries.

```sh
# JDK 21; Android SDK platform 37. The wrapper pins Gradle 8.11.1.
env -u ANDROID_SDK_ROOT ANDROID_HOME="$HOME/Library/Android/sdk" ./gradlew \
  :runtime:testDebugUnitTest :demo:assembleDebug :demo:assembleDebugAndroidTest \
  :runtime:publishReleasePublicationToDistributionRepository
```

The Maven repository is written to `../../dist/android-maven`. Consume
`ai.phaseone:runtime-android:0.3.0` from that repository, plus Google/Maven Central
for its pinned native dependencies. The SDK is not published publicly.

`PhaseoneRuntime(context, trust)` exposes `releases.check/update/state/requalify/
rollback` and `open(capability)`. Call synchronous download and inference work
on an application-owned executor, not the UI thread. Pin the publisher trust key
in your app. Closing a `NativeApplication` releases its loaded engines.

Real Pixel execution and model-release lifecycle checks were performed in the
original prototype. That does not qualify every GPU/NPU adapter. The ExecuTorch
tensor adapter currently embeds XNNPACK CPU; LiteRT's native path can request
CPU/GPU/NPU, with vendor libraries and compatible models required for NPU use.
Xiaomi installation was blocked by that device's installation policy. Board and
Qualcomm NPU paths still need device validation.

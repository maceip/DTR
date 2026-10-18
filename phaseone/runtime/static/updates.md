# DTR Model CDN: delivery and updates

**The developer integrates DTR. DTR owns model selection, native dependency
packaging, delivery, compatibility checks, and update coordination.** An app
should not need different inference or download code for each phone or board.
The developer-facing delivery product is **DTR Model CDN**: device-matched
artifacts, fast-follow installation, streamed delivery, caching, and updates.
This technical document describes the providers underneath that interface.
Platform delivery providers are implementation details, not separate integration
requirements for app developers.

**Embedded integration is a product requirement.** Customers install only the
developer's app and use that app's interface. DTR must not introduce another app
installation, a DTR customer account, a setup flow, or branded update prompts.
The separate Android runtime-app proposal is rejected, including as an optional
route. Development hosts remain explicitly started developer tools.

This is the proposed product design. The existing Python and Android release
managers implement verified model downloads, qualification, activation, offline
opening, and rollback. Automatic lifecycle scheduling, build plugins, managed
catalogs, and the additional delivery adapters below are not implemented yet.

## Separate execution from delivery

Choose the execution engine for the task and hardware, then choose how its
components arrive. A model might execute through native LiteRT on the NPU while
its data arrives from DTR's CDN. Another model might use a Windows-managed ONNX
provider or a downloaded web engine. A web delivery option must not remove the
native paths. Android and Linux/IoT are the first deployment priorities; macOS
and Windows remain development and desktop targets. iOS follows later.

| Route | What can change without rebuilding the consuming app? | Accelerator access and boundary |
| --- | --- | --- |
| Bundled native engines + DTR model delivery | Compatible models, graphs, tokenizers, routing, and selection among installed delegates | Existing native CPU/GPU/NPU capabilities. A missing native library still needs a code-delivery channel. |
| Platform-managed engines | Engine/provider improvements behind a compatible API | Windows ML supplies supported native execution providers; Play services LiteRT exposes its supported delegates. DTR does not control their release schedules. |
| Native runtime components on Linux/IoT and directly distributed desktops | Compatible engine and delegate binaries, loaded by a new worker/session | Native accelerator access where the runtime, driver, OS, licensing, and signing support it. Not an automatic driver/firmware updater. |
| Embedded WASM + a native host interface | Execution control over installed engines; broader portable processing needs a suitable host interface | Experimental Android host exercised native CPU and OpenCL GPU execution. NPU remains unvalidated. No WebView required; a missing vendor library cannot be invented by the module. |
| WebView/browser + JS/WASM/WebGPU | Compatible web engines and GPU kernels | WebGPU reaches supported GPU functionality. It does not load arbitrary vendor NPU delegates. Useful option, not the mandatory foundation. |
| App build integration | DTR prepares new embedded native dependencies for the next app release | Keeps application code stable; does not remove the release when that is the only channel for a new binary. |

Native component delivery remains a proposal. The experimental Android portable
adapter host is implemented, with the bounded device result below; automatic
catalog integration is still missing. This comparison does not establish
app-store acceptance or support on every device.

## Android direction: keep the native interface stable

The app embeds DTR's native host and the engine/vendor interfaces selected at
build time. DTR updates compatible models and portable execution programs around
that host. Model compilation belongs on DTR's infrastructure where possible.
LiteRT explicitly separates its [compiler plugin](https://developers.google.com/edge/litert/next/compiler_plugin)
from its [dispatch interface](https://developers.google.com/edge/litert/next/dispatch).
The compiler can package accelerator bytecode into a model that the installed
dispatch interface executes. Our inference from that split: improvements to
partitioning or compilation can reach existing apps as newly qualified artifacts,
provided they remain compatible with the installed runtime, driver, and hardware.

The portable layer handles control logic while native engines execute tensors.
The current prototype exposes registered-engine availability and inference over
opaque handles. It does not yet expose a general graph-building interface or
portable tokenization/preprocessing. Those require deliberate additions to the
host contract; arbitrary new vendor APIs are not automatically available.

This is how DTR should minimize native app releases. It is not a promise that
every future delegate can arrive without one. A genuinely new native library or
host capability still needs its supported distribution channel. Existing apps
keep a qualified working path until the new capability is available. DTR's build
integration should own that dependency change without application-code rewrites.

## Packaging belongs inside DTR

DTR Model CDN has one delivery contract across Android, Linux/IoT, macOS,
Windows, and web, with Android as the first implementation target. Fast-follow
means delivery after installation where supported, otherwise at the first
allowed app lifecycle window. Streamed delivery means model bytes can be read
incrementally into storage or an engine that supports that loading interface;
it does not promise inference before all required weights are available.

The developer declares the capability, target platforms, quality requirements,
and update policy. DTR's build integration should select and generate the needed
Maven/Gradle dependencies, device targeting, feature modules, bundled baseline,
and model-delivery configuration. Developers should not learn AI Packs or author
one integration per chip. Advanced configuration can override the defaults.

[Play for On-device AI](https://developer.android.com/google/play/on-device-ai)
can deliver device-targeted models at install time, fast-follow, or on demand.
Its AI Packs contain models, not native libraries, and update together with the
app binary. Native libraries belong in the base app or feature modules. Those
modules share the app version and require a new full bundle for new code, as
explained in the [base-module guide](https://developer.android.com/guide/app-bundle/configure-base).

Therefore Play delivery is an optional installation/size optimization, not our
only update transport. DTR-managed downloads can update compatible model data
outside the app-release cadence. Both transports should present the same model
contract, verification, caching, and session behavior. They should not require
duplicate model copies by default. Do not use Play AI Packs as a cross-app model
store: Google's guide restricts their models to the publisher's apps.

[LiteRT's NPU integration](https://developers.google.com/edge/litert/next/npu)
already supports vendor-specific native paths and documents AI Packs plus
feature-module packaging. DTR should reuse that integration and generate its
configuration, rather than require developers to reproduce it. LiteRT supports
AOT and some on-device compilation paths; supported operations and SoCs still
need qualification. Not every vendor/model combination supports JIT compilation.

## Platform-managed routes worth integrating

**Windows ML:** Windows 11 24H2+ can acquire certified ONNX execution providers,
including vendor GPU/NPU providers, through its catalog. Compatible provider
updates reach apps through Windows Update without an application update. An app
cannot force an updated provider installation; availability and drivers matter.
DTR should wrap this route, with an appropriate fallback where needed.
[Update behavior](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/update-execution-providers),
[managed versus bundled providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/windows-ml-eps-vs-bring-your-own).

**Android:** [LiteRT in Play services](https://developers.google.com/edge/litert/android/play_services)
provides a managed runtime, with only its documented delegates and no custom ops.
It is not a universal updater for embedded LiteRT-LM or arbitrary NPU libraries.
[AICore / ML Kit GenAI](https://developers.google.com/ml-kit/genai) is another
managed option when the app can use Google's Gemini Nano model and supported
APIs; it is not a custom-model engine. Foreground and quota restrictions matter.

Android's Privacy Sandbox **SDK Runtime is scheduled for phaseout**, so it is
not a foundation to adopt for independent SDK delivery.
[Current official status](https://privacysandbox.google.com/overview/status).

## Portable code without sacrificing native acceleration

An embedded VM can call installed native functions. [WAMR demonstrates this
interface](https://wamr.gitbook.io/document/wamr-in-practice/features/export_native_api/basic)
and supports Android, Linux, macOS, and Windows. This makes a versioned WASM
module over a native inference host a plausible option. Keep tensors in native
buffers and expose coarse inference operations; copying every tensor through
JavaScript or crossing the boundary for every operator would need a different
performance justification. The Android prototype below validates the native-call
boundary with CPU and native GPU execution. A real NPU bridge benchmark is still
needed.

Google Play's [Device and Network Abuse policy](https://support.google.com/googleplay/android-developer/answer/16559646)
provides a VM/interpreter exception for code with indirect Android API access.
That is a route to evaluate, not blanket approval for every bridge. Downloaded
WASM does not authorize an arbitrary native-library downloader. Precompiled
CPU code disguised as a model or VM asset is not automatically portable code.

A programmable executor can also accept new graphs or GPU programs within its
existing capabilities. [IREE's deployment configurations](https://iree.dev/guides/deployment-configurations/)
illustrate native Vulkan/Metal/CUDA and interpreted execution options. This is
another engine candidate, not evidence that an unknown vendor NPU becomes
accessible. MLC/TVM-compiled native libraries have the same distribution question.

For native plugins, [ONNX Runtime provides explicit registration APIs](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html).
Libraries must stay loaded while their sessions exist. DTR should activate a
new component in a fresh worker instead of replacing code beneath active work.
On macOS, [library validation](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.security.cs.disable-library-validation)
also affects in-process third-party plugins; a separately signed runtime worker
is a candidate to validate for direct distribution. Mac App Store distribution
needs its own policy. No permanent desktop daemon is required.

## Lifecycle and implementation order

Open the bundled/cached working release first. At an application-approved idle
or resume boundary, check a small catalog and stage permitted downloads. Respect
network, storage, power, and time budgets. Verify and qualify a candidate before
activation. Existing sessions keep their engine/model; new sessions may use the
update. Preserve a working version for rollback. Do not attempt to move an
engine-specific KV cache to a different engine.

Builds should resolve DTR's tested native dependency set into a lockfile. Normal
development builds may refresh it under policy; release CI uses frozen versions.
An optional CI integration can prepare a required native app release. The
customer's signing and publication permissions remain under their control.

Recommended next implementation: retain the native adapters, add interchangeable
artifact delivery behind the model-release API, and generate Android packaging.
The native-host WASM feasibility probe now passes on the Pixel. Next, connect
portable adapter requirements and artifacts to the existing release manager,
then qualify native NPU models and platform-managed engines. All customer
integration stays inside the developer's app;
a WebView is not mandatory. Moving native updates into a separate customer-installed
runtime package does not meet the product requirement.

## What we actually exercised

**Embedded native host:** the optional Android `portable-adapters` module runs
WASM control programs in the Chicory 1.7.5 interpreter. A single installed probe
APK downloaded successive signed adapter releases and executed a small tensor
model through native LiteRT CPU, native ExecuTorch CPU, then LiteRT's native
OpenCL GPU delegate. The outputs matched. Its APK hash, version, install time,
and process stayed unchanged during updates; no native library was downloaded.
Logs show the sum graph's sole node delegated to `LITERT_CL`. This is delegation
and functional evidence, not a performance benchmark or NPU result.

Four invalid candidates (digest, signature, host ABI, and instruction limit) left
the working release usable. The original adapter remained callable after an
update, and the cached release survived process restart with the download server
stopped. The host source and bounded receipt are in
[the Android module](../sdk/android/portable-adapters/README.md). The HTTP probe,
publisher credentials, APK, model fixtures, and raw device captures remain local.
The existing native libraries were bundled up front. The default SDK release
manager does not yet schedule these adapter updates.

**Web path:**
On the connected Pixel's embedded WebView, the same installed probe APK loaded
LiteRT.js **2.5.2**, then **2.5.3**, from our local server. Both executed a real
small tensor graph using WebGPU and returned the expected output. The installed
APK's SHA-256 stayed identical. Versions changed between page/session loads,
not during an active inference.

LiteRT-LM **0.18.1** also loaded the available Gemma web model and returned a text
response with the GPU backend requested. It used our existing JS binding
compatibility patch. This was one functional run, not a performance benchmark or
per-operator hardware trace. Neither experiment establishes a production mobile
updater, Play acceptance, arbitrary NPU plugin loading, or a WASM/native NPU
bridge. The web probe sources and device captures remain outside the public
source set.

## Available model-release API

```python
import hashlib
from pathlib import Path
from dtr import ReleasePolicy, Runtime

# Pin the publisher key shipped with your application, not a key downloaded
# from the same untrusted endpoint as a new catalog.
public = bytes.fromhex(Path("publisher.pub").read_text().strip())
keys = {hashlib.sha256(public).hexdigest(): public}
policy = ReleasePolicy(
    contract="imagenet1k.rgb224.v1",
    allowed_lineages=["torchvision.mobilenet-v3-small.imagenet1k-v1"],
    accelerators=["cpu"],
    allow_fallback=True,
)

with Runtime(".dtr", catalog_keys=keys) as runtime:
    # Synchronous work: call from the app's allowed update executor, not its UI.
    report = runtime.releases.update(
        "https://your-model-host.example/catalog.json", "vision", policy,
    )
    app = runtime.open("vision")  # Opens the installed release without a fetch.
```

`check`, `update`, `state`, `open`, `requalify`, and `rollback` are implemented in
Python and Android. The publisher chooses candidate priority; current selection
is not an automatic fleet-wide performance optimizer. A hosted DTR catalog,
curation pipeline, lifecycle scheduler, build manifest resolver, and platform
adapters are the remaining product work.

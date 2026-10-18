# Team handoff: tool calling and fine-tuning

DTR's deliverable stays the runtime: an application keeps its model operations,
tools, and permissions when compatible models or engines change. These are
optional future implementation tasks, not unfinished work from the README and
example-folder request. The Gate implementation now lives under
`examples/iot-app/gate`; the other example work is tracked separately.

Runtime and repository fixes found in code review are tracked separately in
[review follow-ups](review-followups.md).

## Tool calling: working foundation, incomplete model integration

**Already implemented:** shared tool-call shapes, JSON/schema validation,
app-owned handlers, resource permissions, revocation checks, and persisted action
records. Python and browser runtimes have bounded agent loops. Android can return
JSON tool proposals and execute them through the checked gateway when the app
explicitly invokes it. Its supported schemas are closed, flat objects with
primitive parameters.

**Evidence:** permission contracts have automated checks across Python,
TypeScript, and Kotlin. The Pixel gateway allowed a scripted note read and patch
write, denied export and code execution, and denied a cached read after
revocation. These were scripted proposals; real model tool-selection quality
has not been measured. `capabilities(model)` correctly reports
`quality_verified: false`.

| Priority | Work another teammate can own | Completion evidence |
| --- | --- | --- |
| 1 | Run a real local model through the existing Python/browser agent loop with a small fixed tool suite. Include correct-tool, no-tool, malformed-argument, and forbidden-resource cases. Record model, engine, prompt/template, and results. | Model-generated calls reach the gateway; valid calls succeed, denied handlers are never entered, and results feed the next model turn. Report quality separately from permission enforcement. |
| 2 | Add the Android agent loop: application-owned turn/action IDs, durable history, bounded steps/time, cancellation, and explicit recovery of uncertain actions. Reuse `ToolGateway` for every call. | A real model completes a multi-step task. Cancellation/restart does not silently repeat an uncertain side effect, and permission revocation blocks subsequent access. |
| 3 | Add Android native structured-tool support for one engine, starting with its actual tool-schema and tool-response APIs. Extend JSON-schema coverage only where needed. | A native call and its response round-trip through the common contract. Unsupported protocols/schemas still fail explicitly; no engine executes tools itself. |
| 4 | Add tool-task checks to release qualification. Compare the same app tool contract across two qualified model/engine variants. | A model update preserves tool behavior and permissions; a failing candidate keeps the working release active. |

**Suggested first pickup:** priority 1. It turns infrastructure into an honest
model-driven demonstration without first expanding every SDK.

Start in [Python agent loop](../phaseone/runtime/agent.py),
[Python tool gateway](../phaseone/runtime/tools.py),
[browser loop](../sdk/typescript/src/browser.ts),
[TypeScript gateway](../sdk/typescript/src/tool-gateway.ts),
[Android function calling](../sdk/android/runtime/src/main/java/ai/phaseone/runtime/FunctionCalling.kt),
[Android application](../sdk/android/runtime/src/main/java/ai/phaseone/runtime/NativeApplication.kt),
and [Android gateway](../sdk/android/runtime/src/main/java/ai/phaseone/runtime/ToolGateway.kt).
The [permission example](../examples/permissions/demo.mjs) is a deterministic
reference for allow/deny behavior; it is not an inference demo.

## Fine-tuning: API and consent checks only

The request contract names a logical model and dataset resource, an objective
(`function_calling` or `chat`), and the `lora` method. It requires both dataset
`read` and `train` permission. Missing consent returns `permission_denied`;
valid consent currently returns `fine_tuning_unavailable`.

**Nothing trains today.** The stub does not read a dataset, upload data, create
a training job, or change weights. No trainer adapter, dataset format, LoRA
loader, or export/qualification pipeline has been implemented.

| Priority | Work another teammate can own | Completion evidence |
| --- | --- | --- |
| 1 | Choose one trainable model with a suitable license and one laptop/build-worker trainer. Define the dataset format for conversations, tool schemas, expected calls, arguments, and tool results; validate data and retain provenance. | A consented dataset is validated. Missing read/train permission prevents the first data read; any remote upload also requires separate export permission. |
| 2 | Implement one trainer adapter with progress, cancellation, failures, and output metadata. Train a small LoRA on the consented function-calling examples. | A real adapter artifact is produced from a recorded base-model revision. Cancellation and failed jobs report their actual state. No synthetic success or placeholder job IDs. |
| 3 | Evaluate held-out tasks before and after training: correct tool, argument validity, no-tool cases, and unauthorized requests. Keep deterministic permission enforcement independent of model accuracy. | A reproducible comparison shows whether the training helped and where it regressed. Denied operations remain denied regardless of the generated call. |
| 4 | Package the trained result for one existing inference path. Merge the LoRA before export if that engine cannot load adapters; record base-model/tokenizer lineage and target compatibility. | The trained result runs through DTR, passes task qualification, and can be activated and rolled back as a normal model release. |

**Suggested first pickup:** a laptop training-and-evaluation slice for one model,
then export to one supported device path. On-device training is separate work;
precompiled sponsor NPU artifacts do not establish a trainable graph or LoRA
support. No retraining of those artifacts is assumed.

Start in [Python stub](../phaseone/runtime/training.py),
[TypeScript stub](../sdk/typescript/src/training.ts),
[Android permission/training contract](../sdk/android/runtime/src/main/java/ai/phaseone/runtime/Permissions.kt),
[shared permission cases](../sdk/contracts/permissions-cases.json), and
[model releases](../phaseone/runtime/releases.py).
The [permission and consent contract](tools-and-permissions.md) must remain the
entry boundary when a trainer is added.

## Keep the developer experience intact

A teammate can own the tool-call demonstration while another owns the training
slice. They should agree on one tool schema and evaluation dataset first.
The runtime remains embedded in the developer's app; it introduces no customer
installation, DTR account, or always-running desktop service. Models propose
operations; the app's gateway decides whether they may execute.

The Gate example under `examples/iot-app/gate` now implements the camera and command demonstration; sibling examples are separate work.

## device notes

these notes were moved from the readme so the team keeps the testing details

| device | notes |
| --- | --- |
| macos | apple silicon cpu inference and chat through executorch and litert have been tested; mlx chat on the metal gpu and the local_server engine against llama-server have been tested; core ml has not; ollama, lm studio and omlx use the same openai-compatible api as llama-server but were not run |
| android | pixel inference and model updates have been tested; a separate app tested cpu and gpu adapter updates; xiaomi installation was blocked by device policy; qualcomm phone npu execution has not been tested |
| linux / iot | vision and chat now run through both cpu engines on the physical ventuno q; see the reconnect notes below for camera and npu status |
| windows | intended for snapdragon laptops and ordinary development machines; native integration, packaging and device testing are still needed |
| web | local litert tensors and compatible litert-lm models have run; executorch uses the optional paired native host |
| ios | planned later; no sdk or device testing yet |

pixel tool checks used scripted requests, not calls generated by a model; model tool-calling quality has not been measured

## additional implementation notes

- model selection uses device and engine compatibility, publisher priority and packaged checks; automatic comparisons across models are still to build
- model downloads, verification, qualification and rollback are implemented; the hosted model cdn, automatic lifecycle checks and build integration are still to build
- export covers selected pytorch models; existing `.pte`, `.tflite` and `.litertlm` artifacts can be imported, while automatic vendor npu compilation is separate work
- compatible model data can change without an app release; a missing native engine or delegate library still needs its supported code delivery path, which can require a new app build
- typescript imports the upstream gdp-ts eslint plugin in `sdk/typescript/eslint.config.mjs`; `npm run lint:proofs` and `npm run typecheck` run its checks, while the normal build does not invoke the linter automatically
- gdp-ts compiler proofs apply to typescript; python and kotlin enforce the shared resource policy at runtime rather than inheriting typescript's compiler guarantees

## ventuno reconnect

checked on 2026-10-18 with the physical ventuno q, ubuntu 24.04.4, python 3.12.3 and qcs8275 hardware

- the same `dtr-vision` package ran through executorch 1.5.1 and litert 2.3.0 on the board; both classified the sample as samoyed
- executorch's logits matched the mac exactly for that image; the largest difference between the two board engines was `0.0000295639`
- the `local-assistant` package streamed text through executorch and litert-lm 0.18.0; a session started in executorch and recalled the supplied name after switching to litert; retrying the same turn returned the saved result
- the board's audit chain verified with 12 records; this is a local hash chain, not hardware attestation
- unauthenticated api access was rejected, and requesting an unavailable npu variant returned `no_compatible_variant`
- a fresh gcc 13.3 / cmake / ninja build produced a native arm64 program that ran on the host; the existing inference-to-mcu led check confirmed on/off state through bridge rpc

the runtime lives at `~/phaseone-runtime` on the board; its loopback server is available over usb at `http://127.0.0.1:8921` on the mac

the brio shutter is now open and the gate example owns continuous capture; its live preview replaces the earlier camera service and is at `http://127.0.0.1:8931/?source=camera`

the reconnect checks above used cpu; the sponsor archive was downloaded later and its qnn runners are now installed, as described below

installation used the existing arm64 `uv` binary and a board-local copy of `libatomic.so.1` from the existing ubuntu development container; `scripts/run_runtime_board.sh` sets the library path; stalled wi-fi downloads were completed on the mac, checked against `uv.lock`, and transferred over usb

`uv pip check` reports one dependency metadata warning: `backports-strenum==1.3.1` declares python below 3.11, while the environment uses 3.12; native imports and the inference checks above passed, but the package compatibility check is not clean

local result files are `.work/runtime-board.json` and `.work/runtime-board-chat.json`; camera captures, model files, pairing tokens and runtime state stay outside git

[board deployment and terminal commands](ventuno.md)

## gate demo and npu

the standalone example is `examples/iot-app/gate`; all inference goes through `app.chat`, and actions go through `app.invoke_tool` and the runtime's resource policy

- continuous video comes directly from the brio 101 through opencv / v4l2; the camera reports observed frame count, rate and age, and shares a live mjpeg preview
- the screen reader is gemma 4 e2b through litert-lm on cpu; the judge is the sponsor's gemma 4 e2b qcs8275 artifact through executorch / qnn on the hexagon npu
- the qnn adapter consumes the package's tokenizer and chat template, preserves canonical message history, enforces the context budget, checks native generation statistics and records actual execution in dtr receipts
- native completions are buffered until the cli runner finishes; the runner loads its model for each request, so its loading time is included in the reported inference time
- the adapter does not grant tool permissions; only the exact bundled test command can execute, with fixed arguments, no shell, a fresh single-use decision and the app's workspace permission
- warning lights use `set_led_state` and `get_led_state` through `/var/run/arduino-router.sock`; a result is confirmed only after actual mcu readback

the sponsor source archive sha256 is `9f665d0508aa6e3fab087d48d0c610648078e01ad341ad20535e194700c463f0`; native host files are separately inventoried in `native/qnn/runtime.json` on the board and checked before use

qnn 2.46.0 host and v75 dsp libraries came from the already installed `ghcr.io/arduino/app-bricks/llamacpp-npu-runner:0.12.1` image; linux rpc libraries came from this board, and libatomic came from its existing runtime installation

gemma 3 also completed real npu inference, but it incorrectly approved deleting migrations and later returned an invalid judgment object — it is not the selected demo judge; gemma 4 passed the three direct judgment checks

the typed judgment is inspired by the earlier jev proposal; this example uses gemma weights, with no jev model or on-device training, and it does not intercept arbitrary codex commands, send usb keyboard input, estimate physical depth or listen for speech

camera captures, model weights, tokens, native logs and local test receipts stay in ignored state / work directories; the camera was facing the room during the initial live capture check, so that check established capture and a held decision, not optical reading of a laptop terminal

linux process-lifetime validation confirmed that terminating a dtr worker also kills its native child; local runtime hashes and the receipt chain are integrity checks, not hardware attestation

the identical package `7b156fd3b8cf2d93ea8b8f1b0c09f330264645c59fc795d9c49e0b715d79b6c8` passed all three rendered-terminal scenarios on both macos and the ventuno: allow / block / block, with the allowed test process exiting zero and neither blocked command executed

on macos both logical models selected litert cpu; on the board the screen reader selected litert cpu and every judgment recorded qnn htp execution, with the real led read back after each decision — the first board check took 61.4 seconds including worker loading, and the two later checks took 28.7 and 25.8 seconds

the final camera check decoded 20 distinct jpeg frames in 2.644 seconds at 720 × 1280; the real camera image then went through the vision model, which returned no visible command and held execution while confirming the warning led — point the brio at `/screen` for the physical terminal demonstration

local reports are `.work/gate-hybrid-board-results.json`, `.work/gate-mac-results.json`, `.work/gate-final-camera-result.json`, `.work/camera-proof/gate-final-video.json` and `.work/qnn-process-lifetime.json`; the example wheel includes its ui, camera code and test workspace

after combining the qnn adapter with the team's mac engines and model-comparison changes, the python runtime and gate suite passed 175 tests with one optional test skipped

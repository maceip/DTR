# Team handoff: tool calling and fine-tuning

DTR's deliverable stays the runtime: an application keeps its model operations,
tools, and permissions when compatible models or engines change. These are
optional future implementation tasks, not unfinished work from the README and
example-folder request. The three reserved application folders under `examples/`
remain empty until the team chooses what to build.

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

The example app folders are placeholders, not work assigned by this handoff.

## device notes

these notes were moved from the readme so the team keeps the testing details

| device | notes |
| --- | --- |
| macos | apple silicon cpu inference and chat through executorch and litert have been tested; mlx chat on the metal gpu and the local_server engine against llama-server have been tested; core ml has not; ollama, lm studio and omlx use the same openai-compatible api as llama-server but were not run |
| android | pixel inference and model updates have been tested; a separate app tested cpu and gpu adapter updates; xiaomi installation was blocked by device policy; qualcomm phone npu execution has not been tested |
| linux / iot | linux arm64 cpu paths were tested in a vm; ventuno q board npu and camera testing is still needed |
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

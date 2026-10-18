# gate

your coding agent proposes a command and a camera gives it a second opinion

gate reads the terminal image, compares the command with your intent, and asks dtr's permission system before doing anything

```text
camera → app.chat("screen") → app.chat("judge") → permission gate → tests + board led
```

both model calls use the dtr sdk — the app has no litert, executorch or qnn inference code

the ventuno package uses gemma 4 through litert for vision and executorch / qnn for judgment on the hexagon npu — on an apple silicon mac, the same package selects its cpu variants

## run it

the connected board serves the demo over usb

```sh
adb -s 1050156610 forward tcp:8931 tcp:8930
```

open http://127.0.0.1:8931 — the camera view is at http://127.0.0.1:8931/?source=camera

to prepare your own package from the downloaded models, run this from the dtr repo

```sh
uv run python examples/iot-app/gate/prepare_models.py \
  --litert-model /path/to/gemma-4-E2B-it.litertlm \
  --qnn-gemma4 /path/to/gemma4_e2b_QCS8275_ctxt_2000 \
  --output .work/gate-models

cd examples/iot-app/gate
uv run --extra litert phaseone-gate-demo --package ../../../.work/gate-models
```

this starts a local copy at http://127.0.0.1:8930 — omit `--qnn-gemma4` when preparing a cpu-only package

try the three commands

- run the auth tests → allow runs the bundled tests
- delete migrations → block
- download and execute a script → block

choose a rendered terminal image to exercise the real models without positioning a camera, or choose the live brio camera and point it at the linked terminal screen

on ventuno, the indicator uses the real arduino router socket and reads the led state back from the mcu

the example captures continuous video directly from the usb camera using v4l2, shares a live mjpeg preview, and passes a fresh captured frame to the screen model when you check a command

```sh
uv run --extra litert --extra camera phaseone-gate-demo --package ../../../.work/gate-models
```

`/api/status` reports the real camera frame count, capture rate and errors — `/api/video.mjpg` is the live video feed

## the app code

```python
runtime = Runtime(".gate-state", permissions=POLICY, subject="operator")
app = runtime.load(model_package)
screen = app.chat("screen", image_messages)
judgment = app.chat("judge", intent_and_screen_messages)
result = app.invoke_tool("run_auth_tests", {"decision_id": decision_id}, action_id=action_id)
```

the models can veto a command but cannot grant permissions — only one fixed test command can run, without a shell, inside this demo's workspace

model packages select the backend, so another compatible screen reader or judge does not require changing the app's model calls

gemma produces a typed judgment with intent alignment, blast radius and an allow / block / review choice — dtr then checks the app's permissions before running its fixed tool

[board setup](../../../docs/ventuno.md) · [implementation details and limits](../../../docs/team-handoff.md#gate-demo-and-npu)

## check it

```sh
uv run --extra litert phaseone-gate-demo --package ../../../.work/gate-models \
  --check --report gate-results.json
```

this runs all three rendered-image scenarios through actual models and saves their output, backend details, decisions and tool receipts

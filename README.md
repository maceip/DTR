# phaseone

**a runtime for shipping local ai across the devices your customers use — without writing a separate integration for every stack**

your model works on your laptop. then the [sponsor guide](https://github.com/aaishikasb/qc-executorch-guide/) hands you a different model build and setup for every device

- **mobile** — galaxy s25 ultra, snapdragon 8 elite, sm8750, htp v79
- **xr** — galaxy xr, xnnpack cpu or vulkan gpu
- **iot** — [ventuno q / qcs8275](https://aihub.qualcomm.com/jobs/jgdd7k7rg), linux and arduino sensors
- **laptop** — snapdragon x elite, sc8380xp, windows arm64

same feature, several integrations. next month a better model or delegate lands and you get to do it again

phaseone keeps that work behind one runtime api for text, images, audio, embeddings and tool calls. it selects a compatible packaged model and engine while your app keeps its feature code, tools and permissions

## 1. start on your laptop

from this checkout on macos or linux, with python 3.12 and [uv](https://docs.astral.sh/uv/), build one image classifier for both engines — the cli is `dtr`

```sh
uv sync --extra executorch --extra litert --extra export
uv run dtr doctor
uv run dtr build examples/vision/model.py .work/vision \
  --name dtr-vision --model vision
uv run dtr classify .work/vision --model vision \
  --image examples/vision/sample-dog.jpg --backend executorch
uv run dtr classify .work/vision --model vision \
  --image examples/vision/sample-dog.jpg --backend litert
```

same image, same feature, two engines. results include the prediction and engine used. leave out `--backend` to let the runtime select a compatible variant

## 2. put it in your app

embed the sdk and its engines: python for macos and linux, [kotlin for android](sdk/android/), [typescript for the browser](sdk/typescript/). your users install your app

with a python runtime instance, your feature code is

```python
app = runtime.load(".work/vision")
result = app.classify("vision", photo)
```

the package carries device-specific variants, preprocessing and labels. your feature calls `vision`; the runtime handles the engine. existing `.pte`, `.tflite` and `.litertlm` models use the same package format

## 3. give it access to what it needs

register your app's tools and use your existing login to set permissions. this policy lets alice read her notes and save generated code

```json
{"version": 1, "rules": [
  {"subject": "alice", "resource": "notes:alice", "action": "read", "effect": "allow"},
  {"subject": "alice", "resource": "code:patch", "action": "write", "effect": "allow"}
]}
```

reading someone else's notes, uploading them, running that code and training on the notes each need their own permission. the runtime checks every tool request before your handler runs, including after a model change

try those rules against sample tool requests, then run the proof checks

```sh
npm --prefix sdk/typescript ci
npm --prefix sdk/typescript run build
node examples/permissions/demo.mjs
npm --prefix sdk/typescript run typecheck
npm --prefix sdk/typescript run lint:proofs
```

the typescript gateway uses permix, gdp-ts proofs and the upstream gdp-ts linter. python and android enforce the same resource rules at runtime

## 4. ship it, then improve it

sign and package the model files you just ran

```sh
uv run dtr keygen .work/publisher.key --trust
uv run dtr sign .work/vision --key .work/publisher.key
uv run dtr verify .work/vision --require-signature
uv run dtr pack .work/vision .work/vision.phaseone
uv run dtr receipts --verify
```

publish improved models under the same `vision` capability. with publisher trust and an update policy configured, your app calls

```python
runtime.releases.update(catalog_url, "vision", policy)
app = runtime.open("vision")
```

the runtime selects a device-compatible variant, verifies the download and runs the release's checks before activating it. failed checks keep the working version in place. compatible model updates reach your app without changing its feature code or requiring another app build

[model updates](docs/updates.md) · [tools and permissions](docs/tools-and-permissions.md) · [windows setup](docs/windows.md) · [team handoff](docs/team-handoff.md)

# dtr / distributed tensor runtime

**ship local ai across the devices your customers use without writing a separate integration for every stack**

you build the app. dtr handles model selection, engines and updates inside it. your users install your app and never need to know dtr exists

## why this exists

your model works on your mac. great. now ship it to one of these

- **mobile** — galaxy s25 ultra, snapdragon 8 elite, sm8750, htp v79
- **xr** — galaxy xr uses xnnpack cpu or vulkan gpu on the guide's tested retail firmware, which doesn't allow third-party qnn/npu execution
- **iot** — [ventuno q / qcs8275](https://aihub.qualcomm.com/jobs/jgdd7k7rg) runs inference on linux, with arduino bridge for sensors and controls
- **laptop** — snapdragon x elite, sc8380xp, windows arm64

the models, engines and setup change between devices. a better delegate ships next month and suddenly you're maintaining another integration instead of building your app

the [sponsor guide](https://github.com/aaishikasb/qc-executorch-guide/) has separate models and setup for these devices. same feature, several integrations, excellent use of everyone's weekend

dtr keeps that mess behind one runtime api. executorch and litert/litert-lm run the models. dtr picks compatible versions, loads them and handles their downloads and updates

## what you can build

- text — local chat, streamed replies and tool calls
- images — classification and image questions using photos, screenshots or camera frames
- audio — audio input through the python litert-lm adapter with a compatible model
- tensors — run your own compiled models, including models that produce embeddings

support depends on the model, engine and sdk

## what goes in your app

embed the sdk and the engines you need. python on macos and linux, kotlin on android, typescript for the browser. windows setup is in progress, with ios later

your app keeps its ui, login and data. no extra customer app, dtr account or permanent desktop service

call the same operations on your laptop and the customer's device. dtr selects a model that matches the device and installed engine, then tells you what actually ran

packages include model files, preprocessing, tool definitions and checks. compatible updates must pass those checks before replacing the working version

[android sdk](sdk/android/) · [typescript sdk](sdk/typescript/) · [windows setup](docs/windows.md)

## models and updates

**dtr model cdn** is the delivery service we're building for models matched to each device, downloads after installation, streaming and offline storage

the download and update code works today. it checks signatures and file hashes, tests new releases and lets you roll back. the hosted service and automatic update checks are still being built

compatible model updates can arrive without rebuilding your app. our android prototype also switched between installed executorch and litert engines using downloaded adapters without changing the apk

a new native library can still need an app release. dtr's planned build tools will handle that dependency work

we want the best model for your app's quality, speed and memory needs. today selection uses compatibility, publisher settings and your checks. automatic model comparisons aren't built yet

[how updates work](docs/updates.md)

## try it

from this checkout on macos or supported linux, with python 3.12 and [uv](https://docs.astral.sh/uv/)

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

one image, two cpu engines, the same app interface. setup downloads dependencies and model weights. sdk installs currently use this source checkout

the exporter supports selected pytorch models. you can also import `.pte`, `.tflite` and `.litertlm` files. automatic npu compilation isn't built yet

## your tools, your rules

register your tools once. dtr checks arguments and permissions before calling your code, whatever model made the request

reading a note doesn't mean uploading it or training on it. saving generated code doesn't mean running it. those are separate permissions, checked outside the model

permix and gdp-ts support the typescript checks. python and kotlin enforce the same rules. your app supplies identity and consent

tool calling works, with android's automatic agent loop still to build. fine-tuning is a stub with permission checks and no trainer

[tools and permissions](docs/tools-and-permissions.md) · [examples](examples/)

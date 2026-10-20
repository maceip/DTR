# ventuno q

build on your mac, then send the same package to the board's linux side

## deploy over usb

connect barrel power and a usb-c data cable, then find the physical board in `adb devices -l`

from the localkit checkout, after building `.work/vision` with the readme commands

```sh
uv run --no-sync python scripts/deploy_runtime.py \
  --serial 1050156610 --package .work/vision --install --start
uv run --no-sync python scripts/check_runtime_board.py \
  --serial 1050156610 --package .work/vision
```

replace the serial with your board's serial; the scripts reject emulators

the first command installs into `~/phaseone-runtime` on the board and pairs the mac with its local server at `http://127.0.0.1:8921` over usb

the second runs the sample image through executorch and litert on the board, compares both outputs with the mac, and saves the results in `.work/runtime-board.json`

add another `--package /path/to/chat-package` to deploy an assistant alongside the vision model; each package needs a different application name

## write code on the board

```sh
adb -s 1050156610 shell
cd ~/phaseone-runtime
bash scripts/run_runtime_board.sh doctor
```

the board has its own linux python environment; use `.venv/bin/python` for your application

if this board uses the local native library directory, set it before launching custom python code

```sh
export LD_LIBRARY_PATH="$PWD/.native-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python your_app.py
```

the supplied launcher handles this for runtime commands

## fresh board setup

the install script needs python 3.12, `libatomic.so.1`, and either `uv` or working python venv support

on a fresh ubuntu image, install the system prerequisites from the board terminal

```sh
sudo apt-get update
sudo apt-get install -y python3-venv libatomic1
```

it then installs the locked executorch and litert dependencies into the board's own environment

the paired server binds to board loopback and requires its token for inference; the deploy script saves that token in `.state/runtime-board.token` on the mac

[device checks and npu setup notes](team-handoff.md#ventuno-reconnect)

## installed qnn runtime

the board's native runtime is in `~/phaseone-runtime/native/qnn`

```text
bin/         linux arm64 executorch runners
lib/         executorch backend, qnn host libraries and rpc dependencies
dsp/         matching hexagon dsp libraries
runtime.json versions and hashes of the installed files
models/      downloaded QNN model artifacts
```

the installed combination is executorch runner 1.5.0 with qnn 2.46.0 and the v75 dsp libraries — it is separate from the executorch 1.5.1 python wheel used for cpu models

after installing matching binaries, register them with localkit

```sh
.venv/bin/python scripts/register_qnn_runtime.py --root native/qnn \
  --executorch-version 1.5.0 --qnn-version 2.46.0
```

`LOCALKIT_QNN_ROOT` can point to another installation — model packages select the installed runner and cannot supply executable paths

the downloaded source archive includes the arm64 `libqnn_executorch_backend.so` under `build-x86/backends/qualcomm` despite that directory's name

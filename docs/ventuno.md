# ventuno q

build on your mac, then send the same package to the board's linux side

## deploy over usb

connect barrel power and a usb-c data cable, then find the physical board in `adb devices -l`

from the dtr checkout, after building `.work/vision` with the readme commands

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

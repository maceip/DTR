#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { echo 'Expected a Linux ARM64 runtime host'; exit 1; }
if [[ -d .native-libs ]]; then
  export LD_LIBRARY_PATH="$PWD/.native-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if ! python3 -c 'import ctypes; ctypes.CDLL("libatomic.so.1")' >/dev/null 2>&1; then
  echo 'ExecuTorch LLMs require libatomic1. Install it with: sudo apt-get install -y python3-venv libatomic1' >&2
  exit 1
fi
if [[ -x .tools/uv ]]; then
  runtime_uv="$PWD/.tools/uv"
elif command -v uv >/dev/null 2>&1; then
  runtime_uv="$(command -v uv)"
else
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
  fi
  if ! .venv/bin/python -m uv --version >/dev/null 2>&1; then
    .venv/bin/python -m pip install 'uv==0.11.23'
  fi
  runtime_uv="$PWD/.venv/bin/uv"
fi
extras=(--extra executorch --extra litert)
if [[ "${1:-}" == --vision ]]; then extras+=(--extra vision); fi
"$runtime_uv" sync --locked --inexact --no-dev "${extras[@]}"
.venv/bin/python -c 'from executorch.extension.llm.runner import TextLLMRunner; import litert_lm; print("Both native LLM bindings loaded")'
.venv/bin/python -m phaseone.runtime.cli doctor

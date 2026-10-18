#!/usr/bin/env bash
set -euo pipefail
GATE_APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DTR_RUNTIME_ROOT="${DTR_RUNTIME_ROOT:-$(dirname "$GATE_APP_DIR")}"
cd "$DTR_RUNTIME_ROOT"
export PYTHONPATH="$GATE_APP_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DTR_RUNTIME_ROOT/.native-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DTR_RUNTIME_ROOT/.venv/bin/python" -m gate_demo.app "$@"

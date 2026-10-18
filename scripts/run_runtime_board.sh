#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .native-libs ]]; then
  export LD_LIBRARY_PATH="$PWD/.native-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python -m phaseone.runtime.cli "$@"

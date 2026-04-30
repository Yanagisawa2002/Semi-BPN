#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$PWD/configs/biopathnet_linux_smoke.yaml"

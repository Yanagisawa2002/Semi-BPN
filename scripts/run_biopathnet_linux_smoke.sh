#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_smoke.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_smoke.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$CONFIG_RUNTIME"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_fallback_train.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_fallback_train.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python - "$CONFIG_RUNTIME" "$TRAIN_EPOCHS" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
num_epoch = int(sys.argv[2])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config.setdefault("train", {})["num_epoch"] = num_epoch
with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$CONFIG_RUNTIME"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/biopathnet_linux_subgraph_k50_fallback_formal_d${HIDDEN_DIM}}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_fallback_formal_d${HIDDEN_DIM}.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_fallback_formal.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python - "$CONFIG_RUNTIME" "$TRAIN_EPOCHS" "$HIDDEN_DIM" "$VALIDATION_INTERVAL" "$EARLY_STOP_PATIENCE" "$EVAL_BATCH_SIZE" "$MODEL_SELECTION_METRIC" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
num_epoch = int(sys.argv[2])
hidden_dim = int(sys.argv[3])
validation_interval = int(sys.argv[4])
early_stop_patience = int(sys.argv[5])
eval_batch_size = int(sys.argv[6])
selection_metric = sys.argv[7]
output_dir = Path(sys.argv[8]).resolve()

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = str(output_dir)
model = config.setdefault("task", {}).setdefault("model", {})
model["input_dim"] = hidden_dim
model["hidden_dims"] = [hidden_dim, hidden_dim]
config.setdefault("train", {})["num_epoch"] = num_epoch

runtime = config.setdefault("runtime", {})
runtime["skip_eval"] = True
runtime["skip_final_eval"] = True
runtime["validation_interval"] = validation_interval
runtime["early_stop_patience"] = early_stop_patience
pairwise_eval = runtime.setdefault("pairwise_eval", {})
pairwise_eval["enabled"] = True
pairwise_eval["batch_size"] = eval_batch_size
pairwise_eval["selection_metric"] = selection_metric
pairwise_eval["output_dir"] = str(output_dir / "pairwise_validation")

with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$CONFIG_RUNTIME"

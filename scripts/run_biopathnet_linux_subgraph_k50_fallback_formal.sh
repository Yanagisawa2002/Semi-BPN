#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
LR="${LR:-0.005}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
if [ -z "${RUN_NAME:-}" ]; then
  if [ "$HIDDEN_LAYERS" = "2" ] && [ "$NUM_NEGATIVE" = "4" ] && [ "$TRAIN_BATCH_SIZE" = "1" ]; then
    RUN_NAME="d${HIDDEN_DIM}"
  else
    RUN_NAME="d${HIDDEN_DIM}_l${HIDDEN_LAYERS}_neg${NUM_NEGATIVE}_b${TRAIN_BATCH_SIZE}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/biopathnet_linux_subgraph_k50_fallback_formal_${RUN_NAME}}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_fallback_formal_${RUN_NAME}.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_fallback_formal.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python - "$CONFIG_RUNTIME" "$TRAIN_EPOCHS" "$HIDDEN_DIM" "$HIDDEN_LAYERS" "$TRAIN_BATCH_SIZE" "$NUM_NEGATIVE" "$LR" "$VALIDATION_INTERVAL" "$EARLY_STOP_PATIENCE" "$EVAL_BATCH_SIZE" "$MODEL_SELECTION_METRIC" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
num_epoch = int(sys.argv[2])
hidden_dim = int(sys.argv[3])
hidden_layers = int(sys.argv[4])
train_batch_size = int(sys.argv[5])
num_negative = int(sys.argv[6])
learning_rate = float(sys.argv[7])
validation_interval = int(sys.argv[8])
early_stop_patience = int(sys.argv[9])
eval_batch_size = int(sys.argv[10])
selection_metric = sys.argv[11]
output_dir = Path(sys.argv[12]).resolve()

if hidden_layers <= 0:
    raise ValueError(f"HIDDEN_LAYERS must be positive, got {hidden_layers}")
if train_batch_size <= 0:
    raise ValueError(f"TRAIN_BATCH_SIZE must be positive, got {train_batch_size}")
if num_negative <= 0:
    raise ValueError(f"NUM_NEGATIVE must be positive, got {num_negative}")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = str(output_dir)
config.setdefault("engine", {})["batch_size"] = train_batch_size
config.setdefault("optimizer", {})["lr"] = learning_rate
task = config.setdefault("task", {})
task["num_negative"] = num_negative
model = config.setdefault("task", {}).setdefault("model", {})
model["input_dim"] = hidden_dim
model["hidden_dims"] = [hidden_dim] * hidden_layers
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

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-9}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-0.005}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-3}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-}"
FINAL_SPLITS="${FINAL_SPLITS:-train,valid,test}"
RUN_NAME="${RUN_NAME:-pairwise_indication_d${HIDDEN_DIM}_l${HIDDEN_LAYERS}_b${TRAIN_BATCH_SIZE}_e${TRAIN_EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/biopathnet_linux_full_indication_pairwise_${RUN_NAME}}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_full_indication_debug/train1.txt" ]; then
  bash scripts/build_biopathnet_indication_debug.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_full_indication_pairwise_${RUN_NAME}.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_full_indication_pairwise.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python - "$CONFIG_RUNTIME" "$TRAIN_EPOCHS" "$HIDDEN_DIM" "$HIDDEN_LAYERS" "$TRAIN_BATCH_SIZE" "$EVAL_BATCH_SIZE" "$LR" "$VALIDATION_INTERVAL" "$MODEL_SELECTION_METRIC" "$EARLY_STOP_PATIENCE" "$FINAL_SPLITS" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
num_epoch = int(sys.argv[2])
hidden_dim = int(sys.argv[3])
hidden_layers = int(sys.argv[4])
train_batch_size = int(sys.argv[5])
eval_batch_size = int(sys.argv[6])
learning_rate = float(sys.argv[7])
validation_interval = int(sys.argv[8])
selection_metric = sys.argv[9]
early_stop_patience = sys.argv[10].strip()
final_splits = [value.strip() for value in sys.argv[11].split(",") if value.strip()]
output_dir = Path(sys.argv[12]).resolve()

if hidden_layers <= 0:
    raise ValueError(f"HIDDEN_LAYERS must be positive, got {hidden_layers}")
if train_batch_size <= 0:
    raise ValueError(f"TRAIN_BATCH_SIZE must be positive, got {train_batch_size}")
if eval_batch_size <= 0:
    raise ValueError(f"EVAL_BATCH_SIZE must be positive, got {eval_batch_size}")
if validation_interval <= 0:
    raise ValueError(f"VALIDATION_INTERVAL must be positive, got {validation_interval}")
if not final_splits:
    raise ValueError("FINAL_SPLITS must include at least one split")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = str(output_dir)
config.setdefault("optimizer", {})["lr"] = learning_rate
model = config.setdefault("task", {}).setdefault("model", {})
model["input_dim"] = hidden_dim
model["hidden_dims"] = [hidden_dim] * hidden_layers
config.setdefault("train", {})["num_epoch"] = num_epoch

pairwise = config.setdefault("pairwise_training", {})
pairwise["relation_name"] = "indication"
pairwise["train_batch_size"] = train_batch_size
pairwise["eval_batch_size"] = eval_batch_size
pairwise["validation_interval"] = validation_interval
pairwise["selection_metric"] = selection_metric
pairwise["final_splits"] = final_splits
if early_stop_patience:
    pairwise["early_stop_patience"] = int(early_stop_patience)
else:
    pairwise["early_stop_patience"] = None

with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

python -m mechrep.training.train_original_biopathnet_pairwise \
  -s 42 \
  -c "$CONFIG_RUNTIME"

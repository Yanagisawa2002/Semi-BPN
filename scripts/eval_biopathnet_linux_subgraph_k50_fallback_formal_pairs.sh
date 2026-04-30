#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

HIDDEN_DIM="${HIDDEN_DIM:-32}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
EVAL_K_VALUES=(${EVAL_K_VALUES:-1 5 10})
if [ -z "${RUN_NAME:-}" ]; then
  if [ "$HIDDEN_LAYERS" = "2" ] && [ "$NUM_NEGATIVE" = "4" ] && [ "$TRAIN_BATCH_SIZE" = "1" ]; then
    RUN_NAME="d${HIDDEN_DIM}"
  else
    RUN_NAME="d${HIDDEN_DIM}_l${HIDDEN_LAYERS}_neg${NUM_NEGATIVE}_b${TRAIN_BATCH_SIZE}"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/biopathnet_linux_subgraph_k50_fallback_formal_${RUN_NAME}}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_fallback_formal_${RUN_NAME}.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_fallback_formal.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

python - "$CONFIG_RUNTIME" "$HIDDEN_DIM" "$HIDDEN_LAYERS" "$TRAIN_BATCH_SIZE" "$NUM_NEGATIVE" "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
hidden_dim = int(sys.argv[2])
hidden_layers = int(sys.argv[3])
train_batch_size = int(sys.argv[4])
num_negative = int(sys.argv[5])
output_root = Path(sys.argv[6]).resolve()

if hidden_layers <= 0:
    raise ValueError(f"HIDDEN_LAYERS must be positive, got {hidden_layers}")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = str(output_root)
config.setdefault("engine", {})["batch_size"] = train_batch_size
task = config.setdefault("task", {})
task["num_negative"] = num_negative
model = config.setdefault("task", {}).setdefault("model", {})
model["input_dim"] = hidden_dim
model["hidden_dims"] = [hidden_dim] * hidden_layers
runtime = config.setdefault("runtime", {})
pairwise_eval = runtime.setdefault("pairwise_eval", {})
pairwise_eval["output_dir"] = str(output_root / "pairwise_validation")

with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

ARGS=(
  --config "$CONFIG_RUNTIME"
  --split-dir "$PWD/data/cloud_run/splits"
  --relation-name affects_endpoint
  --splits valid test
  --batch-size "$EVAL_BATCH_SIZE"
  --k "${EVAL_K_VALUES[@]}"
  --group-by endpoint_id
)

if [ -n "${OUTPUT_DIR:-}" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if [ -n "${CHECKPOINT:-}" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = sorted(root.rglob("model_epoch_*.pth"), key=lambda path: (path.stat().st_mtime, str(path)))
if not candidates:
    raise SystemExit(f"No model_epoch_*.pth checkpoint found under {root}. Run formal training first.")
print(candidates[-1].parent)
PY
)}"
  ARGS+=(
    --select-best-checkpoint
    --checkpoint-dir "$CHECKPOINT_DIR"
    --selection-metric "$MODEL_SELECTION_METRIC"
  )
fi

python -m mechrep.evaluation.score_original_biopathnet_pairs "${ARGS[@]}"

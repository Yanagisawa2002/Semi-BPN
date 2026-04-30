#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

HIDDEN_DIM="${HIDDEN_DIM:-16}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-0}"
MAX_TRAINING_POSITIVE_TRIPLES="${MAX_TRAINING_POSITIVE_TRIPLES:-4096}"
TRAINING_NEGATIVE_BATCH_SIZE="${TRAINING_NEGATIVE_BATCH_SIZE:-16}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"
RUN_NAME="${RUN_NAME:-d${HIDDEN_DIM}_l${HIDDEN_LAYERS}_neg${NUM_NEGATIVE}_b${TRAIN_BATCH_SIZE}_e${TRAIN_EPOCHS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/biopathnet_linux_full_pairwise_diagnostic_${RUN_NAME}}"
CONFIG_RUNTIME="${CONFIG_RUNTIME:-$PWD/results/runtime_configs/biopathnet_linux_full_pairwise_diagnostic_${RUN_NAME}.yaml}"

if [ ! -f "$CONFIG_RUNTIME" ]; then
  python -m mechrep.training.prepare_original_config \
    --config "$PWD/configs/biopathnet_linux_full_pairwise_diagnostic.yaml" \
    --output "$CONFIG_RUNTIME" \
    --root "$PWD"
fi

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
model = task.setdefault("model", {})
model["input_dim"] = hidden_dim
model["hidden_dims"] = [hidden_dim] * hidden_layers
runtime = config.setdefault("runtime", {})
pairwise_eval = runtime.setdefault("pairwise_eval", {})
pairwise_eval["output_dir"] = str(output_root / "pairwise_validation")

with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

if [ -n "${CHECKPOINT:-}" ]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
else
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidates = sorted(root.rglob("model_epoch_*.pth"), key=lambda path: (path.stat().st_mtime, str(path)))
if not candidates:
    raise SystemExit(f"No model_epoch_*.pth checkpoint found under {root}. Run full pairwise diagnostic training first.")
print(candidates[-1].parent)
PY
)}"
  CHECKPOINT_ARGS=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR:-$OUTPUT_ROOT}/pairwise_diagnostic}"

EXTRA_ARGS=()
if [ "${INCLUDE_FACTGRAPH_SUPPORT:-1}" = "1" ]; then
  EXTRA_ARGS+=(--include-factgraph-support)
fi
if [ "${INCLUDE_TRAINING_NEGATIVE_DIAGNOSTIC:-1}" = "1" ]; then
  EXTRA_ARGS+=(
    --include-training-negative-diagnostic
    --max-training-positive-triples "$MAX_TRAINING_POSITIVE_TRIPLES"
    --training-negative-batch-size "$TRAINING_NEGATIVE_BATCH_SIZE"
  )
fi

python -m mechrep.evaluation.diagnose_original_biopathnet_pairs \
  --config "$CONFIG_RUNTIME" \
  "${CHECKPOINT_ARGS[@]}" \
  --split-dir "$PWD/data/cloud_run/splits" \
  --output-dir "$OUTPUT_DIR" \
  --relation-name affects_endpoint \
  --splits train valid test \
  --batch-size "$EVAL_BATCH_SIZE" \
  --k 1 5 10 \
  --group-by endpoint_id \
  --max-records-per-split "$MAX_RECORDS_PER_SPLIT" \
  --sampling-seed "$SAMPLING_SEED" \
  "${EXTRA_ARGS[@]}"

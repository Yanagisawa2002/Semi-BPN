#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_fallback_train.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_fallback_train.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
EVAL_K_VALUES=(${EVAL_K_VALUES:-1 5 10})

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
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(python - <<'PY'
from pathlib import Path

root = Path("results/biopathnet_linux_subgraph_k50_fallback_train")
candidates = sorted(root.rglob("model_epoch_*.pth"), key=lambda path: (path.stat().st_mtime, str(path)))
if not candidates:
    raise SystemExit("No model_epoch_*.pth checkpoint found. Run training first.")
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

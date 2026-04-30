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

if [ -z "${CHECKPOINT:-}" ]; then
  CHECKPOINT="$(python - <<'PY'
from pathlib import Path

root = Path("results/biopathnet_linux_subgraph_k50_fallback_train")
candidates = sorted(root.rglob("model_epoch_*.pth"), key=lambda path: (path.stat().st_mtime, str(path)))
if not candidates:
    raise SystemExit("No model_epoch_*.pth checkpoint found. Run training first.")
print(candidates[-1])
PY
)"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/pairwise_eval}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

python -m mechrep.evaluation.score_original_biopathnet_pairs \
  --config "$CONFIG_RUNTIME" \
  --checkpoint "$CHECKPOINT" \
  --split-dir "$PWD/data/cloud_run/splits" \
  --output-dir "$OUTPUT_DIR" \
  --relation-name affects_endpoint \
  --splits valid test \
  --batch-size "$EVAL_BATCH_SIZE" \
  --k 1 5 10 50 100 \
  --group-by endpoint_id

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANT="${VARIANT:-${1:-}}"
ACTION="${ACTION:-${2:-both}}"

if [ -z "$VARIANT" ]; then
  cat <<'EOF'
Set VARIANT to one of:
  baseline_d32_l2_neg4_b1
  batch_d32_l2_neg4_b4
  depth_d32_l4_neg4_b1
  neg_d32_l4_neg16_b1
  throughput_d32_l4_neg16_b4
  capacity_d64_l4_neg16_b4

Example:
  VARIANT=throughput_d32_l4_neg16_b4 ACTION=both bash scripts/run_biopathnet_k50_fallback_variant.sh
EOF
  exit 2
fi

case "$VARIANT" in
  baseline_d32_l2_neg4_b1)
    export HIDDEN_DIM="${HIDDEN_DIM:-32}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    export RUN_NAME="${RUN_NAME:-baseline_d32_l2_neg4_b1}"
    ;;
  batch_d32_l2_neg4_b4)
    export HIDDEN_DIM="${HIDDEN_DIM:-32}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-2}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export RUN_NAME="${RUN_NAME:-batch_d32_l2_neg4_b4}"
    ;;
  depth_d32_l4_neg4_b1)
    export HIDDEN_DIM="${HIDDEN_DIM:-32}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-4}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    export RUN_NAME="${RUN_NAME:-depth_d32_l4_neg4_b1}"
    ;;
  neg_d32_l4_neg16_b1)
    export HIDDEN_DIM="${HIDDEN_DIM:-32}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-16}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    export RUN_NAME="${RUN_NAME:-neg_d32_l4_neg16_b1}"
    ;;
  throughput_d32_l4_neg16_b4)
    export HIDDEN_DIM="${HIDDEN_DIM:-32}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-16}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export RUN_NAME="${RUN_NAME:-throughput_d32_l4_neg16_b4}"
    ;;
  capacity_d64_l4_neg16_b4)
    export HIDDEN_DIM="${HIDDEN_DIM:-64}"
    export HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
    export NUM_NEGATIVE="${NUM_NEGATIVE:-16}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export RUN_NAME="${RUN_NAME:-capacity_d64_l4_neg16_b4}"
    ;;
  *)
    echo "Unknown VARIANT: $VARIANT" >&2
    exit 2
    ;;
esac

export TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-auprc}"
export LR="${LR:-0.005}"

echo "BioPathNet K50+fallback variant: $VARIANT"
echo "  action: $ACTION"
echo "  run_name: $RUN_NAME"
echo "  hidden_dim: $HIDDEN_DIM"
echo "  hidden_layers: $HIDDEN_LAYERS"
echo "  num_negative: $NUM_NEGATIVE"
echo "  train_batch_size: $TRAIN_BATCH_SIZE"
echo "  train_epochs: $TRAIN_EPOCHS"
echo "  validation_interval: $VALIDATION_INTERVAL"
echo "  early_stop_patience: $EARLY_STOP_PATIENCE"

case "$ACTION" in
  train)
    bash scripts/run_biopathnet_linux_subgraph_k50_fallback_formal.sh
    ;;
  eval)
    bash scripts/eval_biopathnet_linux_subgraph_k50_fallback_formal_pairs.sh
    ;;
  both)
    bash scripts/run_biopathnet_linux_subgraph_k50_fallback_formal.sh
    bash scripts/eval_biopathnet_linux_subgraph_k50_fallback_formal_pairs.sh
    ;;
  *)
    echo "Unknown ACTION: $ACTION. Use train, eval, or both." >&2
    exit 2
    ;;
esac

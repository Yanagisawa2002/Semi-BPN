#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"
PROFILE_SECONDS="${PROFILE_SECONDS:-300}"

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50.sh
fi

CONFIG_RUNTIME="$PWD/results/runtime_configs/biopathnet_linux_subgraph_k50_profile.yaml"
python -m mechrep.training.prepare_original_config \
  --config "$PWD/configs/biopathnet_linux_subgraph_k50_profile.yaml" \
  --output "$CONFIG_RUNTIME" \
  --root "$PWD"

set +e
timeout "$PROFILE_SECONDS" python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$CONFIG_RUNTIME"
status=$?
set -e

if [ "$status" -eq 124 ]; then
  echo "Profiling window ended after ${PROFILE_SECONDS}s."
  exit 0
fi

exit "$status"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "$PWD/data/cloud_run/biopathnet_path_subgraph_k50_fallback/train1.txt" ]; then
  bash scripts/build_path_subgraph_k50_fallback.sh
fi

python -m mechrep.data.remap_biopathnet_relation \
  --config configs/biopathnet_k50_fallback_indication_debug_prepare.yaml

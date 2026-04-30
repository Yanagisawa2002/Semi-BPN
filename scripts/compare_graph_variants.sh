#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m mechrep.data.compare_graph_variants --config configs/graph_variant_comparison.yaml

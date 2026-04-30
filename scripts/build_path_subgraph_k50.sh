#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m mechrep.data.build_path_subgraph --config configs/path_subgraph_k50.yaml

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m mechrep.data.no_evidence_report --config configs/no_evidence_report.yaml
python -m mechrep.data.build_path_subgraph --config configs/path_subgraph_k50_fallback.yaml

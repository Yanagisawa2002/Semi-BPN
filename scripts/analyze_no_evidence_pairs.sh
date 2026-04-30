#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m mechrep.data.no_evidence_report --config configs/no_evidence_report.yaml

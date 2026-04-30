#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.templates.export_evidence_paths --config configs/evidence_paths.yaml

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.training.train_baseline --config configs/baseline.yaml --create-toy-data-if-missing

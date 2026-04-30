#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.training.train_semi_supervised --config configs/semi_template.yaml --create-toy-data

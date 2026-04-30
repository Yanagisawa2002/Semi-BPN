#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.training.train_gold_template --config configs/gold_template.yaml --create-toy-data-if-missing

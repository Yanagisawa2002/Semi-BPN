#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m mechrep.training.train_baseline --config configs/cloud_baseline.yaml
python -m mechrep.training.train_gold_template --config configs/cloud_gold_template.yaml
python -m mechrep.templates.pseudo_assign --config configs/cloud_pseudo_template_main.yaml
python -m mechrep.training.train_semi_supervised --config configs/cloud_semi_template_main.yaml

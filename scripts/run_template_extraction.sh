#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.templates.extract_templates --config configs/templates.yaml --create-toy-data-if-missing
python -m mechrep.templates.build_gold_template_labels --config configs/templates.yaml --create-toy-data-if-missing

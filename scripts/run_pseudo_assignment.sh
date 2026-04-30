#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.templates.pseudo_assign --config configs/pseudo_template.yaml --create-toy-data-if-missing

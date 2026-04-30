#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.data.convert_all_data --config configs/all_data_conversion.yaml

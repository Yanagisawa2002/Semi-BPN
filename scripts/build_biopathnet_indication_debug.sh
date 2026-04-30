#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mechrep.data.remap_biopathnet_relation --config configs/biopathnet_indication_debug_prepare.yaml

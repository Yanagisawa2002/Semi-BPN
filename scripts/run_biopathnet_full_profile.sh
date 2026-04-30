#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH="$PWD/deps/biopathnet:$PWD/biopathnet/original:${PYTHONPATH:-}" \
  python -m mechrep.training.run_original_biopathnet \
  -s 1024 \
  -c "$PWD/configs/biopathnet_full_profile.yaml"

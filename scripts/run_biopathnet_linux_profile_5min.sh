#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/biopathnet/original:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.torch_extensions_linux}"
PROFILE_SECONDS="${PROFILE_SECONDS:-300}"

set +e
timeout "$PROFILE_SECONDS" python -m mechrep.training.run_original_biopathnet_linux \
  -s 42 \
  -c "$PWD/configs/biopathnet_linux_profile.yaml"
status=$?
set -e

if [ "$status" -eq 124 ]; then
  echo "Profiling window ended after ${PROFILE_SECONDS}s."
  exit 0
fi

exit "$status"

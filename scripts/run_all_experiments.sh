#!/usr/bin/env bash
# Reproduce verified NumPy/PyTorch experiments & CSV datasets across all 7 volumes.
# Usage:
#   ./scripts/run_all_experiments.sh           # run all 7 volumes
#   ./scripts/run_all_experiments.sh vol4 vol6 # run selected volumes

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=(vol1 vol2 vol3 vol4 vol5 vol6 vol7)
fi

for v in "${TARGETS[@]}"; do
  SCRIPT="$ROOT_DIR/volumes/$v/code/${v}_experiments.py"
  if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found." >&2
    exit 1
  fi
  echo "==================== Running $v experiments ===================="
  (cd "$ROOT_DIR/volumes/$v" && "$PYTHON" "code/${v}_experiments.py")
done

echo "All requested experiments completed successfully."

#!/usr/bin/env bash
# Apply the verl patch for qwen3_5_moe CONFIG_MAPPING registration
# Run this inside the verl-vllm container after cloning verl
#
# Usage: docker exec verl-vllm bash /path/to/apply-patch.sh

set -euo pipefail

VERL_DIR="${VERL_DIR:-/tmp/verl}"
TARGET="$VERL_DIR/verl/experimental/teacher_loop/teacher_model.py"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found. Is verl installed at $VERL_DIR?"
    exit 1
fi

if grep -q '\[verl-patch\]' "$TARGET"; then
    echo "Patch already applied."
    exit 0
fi

# Copy the patched file
cp "$(dirname "$0")/teacher_model.py" "$TARGET"

if grep -q '\[verl-patch\]' "$TARGET"; then
    echo "Patch applied successfully."
else
    echo "ERROR: Patch application failed."
    exit 1
fi

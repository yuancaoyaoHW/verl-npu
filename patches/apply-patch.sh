#!/usr/bin/env bash
# Apply verl patches for Haidass OPD NPU optimization
# Run this inside the verl-vllm container after cloning verl
#
# Patches applied:
#   1. teacher_model.py    — Register qwen3_5_moe CONFIG_MAPPING
#   2. teacher_manager.py  — Add precompute_prompt_logprobs (Wave 3)
#   3. agent_loop.py       — Split teacher logprobs + prompt precompute (Wave 3)
#   4. ray_trainer.py      — Next-batch prefetch for pipeline overlap (Wave 3)
#
# Usage: docker exec verl-vllm bash /path/to/apply-patch.sh

set -euo pipefail

VERL_DIR="${VERL_DIR:-/tmp/verl}"
PATCH_DIR="$(dirname "$0")"
APPLIED=0
SKIPPED=0
FAILED=0

apply_patch() {
    local patch_file="$1"
    local target_path="$2"
    local description="$3"

    echo "--- Applying patch: $description ---"

    if [ ! -f "$PATCH_DIR/$patch_file" ]; then
        echo "  SKIP: $PATCH_DIR/$patch_file not found"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    if [ ! -f "$target_path" ]; then
        echo "  ERROR: $target_path not found. Is verl installed at $VERL_DIR?"
        FAILED=$((FAILED + 1))
        return
    fi

    if grep -q '\[verl-patch\]' "$target_path"; then
        echo "  SKIP: Patch already applied."
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    cp "$PATCH_DIR/$patch_file" "$target_path"

    if grep -q '\[verl-patch\]' "$target_path"; then
        echo "  OK: Patch applied successfully."
        APPLIED=$((APPLIED + 1))
    else
        echo "  ERROR: Patch application failed."
        FAILED=$((FAILED + 1))
    fi
}

# Patch 1: Register qwen3_5_moe CONFIG_MAPPING
apply_patch "teacher_model.py" \
    "$VERL_DIR/verl/experimental/teacher_loop/teacher_model.py" \
    "qwen3_5_moe CONFIG_MAPPING registration"

# Patch 2: Add precompute_prompt_logprobs to teacher_manager (Wave 3)
apply_patch "teacher_manager.py" \
    "$VERL_DIR/verl/experimental/teacher_loop/teacher_manager.py" \
    "teacher_manager precompute_prompt_logprobs (Wave 3)"

# Patch 3: Split teacher logprobs + prompt precompute in agent_loop (Wave 3)
apply_patch "agent_loop.py" \
    "$VERL_DIR/verl/experimental/agent_loop/agent_loop.py" \
    "agent_loop split teacher logprobs (Wave 3)"

# Patch 4: Next-batch prefetch in ray_trainer (Wave 3)
apply_patch "ray_trainer.py" \
    "$VERL_DIR/verl/trainer/ppo/ray_trainer.py" \
    "ray_trainer next-batch prefetch (Wave 3)"

echo ""
echo "=== Patch Summary ==="
echo "  Applied: $APPLIED"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi

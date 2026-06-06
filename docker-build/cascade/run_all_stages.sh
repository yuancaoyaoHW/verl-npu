#!/usr/bin/env bash
# Master orchestrator: run all 4 Cascade RL stages sequentially.
#
# Each stage:
#   1. Trains on its domain data
#   2. Saves checkpoints at SAVE_FREQ intervals
#   3. Passes the latest checkpoint to the next stage
#
# Usage:
#   bash scripts/run_all_stages.sh              # Run all stages
#   bash scripts/run_all_stages.sh 2            # Start from stage 2
#   bash scripts/run_all_stages.sh 1 3          # Run stages 1-3 only
#
# Environment variables:
#   DATA_PATH        — data directory (default: $HOME/data/cascade)
#   REWARD_FN_PATH   — reward function path (default: /root/my_rewards.py)
#   CKPT_ROOT        — checkpoint root (default: /tmp/verl/checkpoints)
#   SANDBOX_URL      — SandboxFusion URL for Stage 3 (default: http://localhost:8080/run_code)
#   DRY_RUN          — set to 1 to print commands without executing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/tmp/verl/checkpoints}"

START_STAGE="${1:-1}"
END_STAGE="${2:-4}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_latest_ckpt() {
    local project_name="$1"
    ls -td "$CKPT_ROOT/$project_name"/*/global_step_* 2>/dev/null | head -1 || true
}

run_stage() {
    local stage=$1
    local script=""
    local project_name=""

    case $stage in
        1)
            script="$SCRIPT_DIR/run_cascade_stage1.sh"
            project_name="verl_cascade_stage1_math"
            log "========== Stage 1: Math RL + OPD =========="
            log "Data: GSM8K + DeepMath-103K"
            log "LR: 1e-6, Epochs: 5, rollout.n: 16"
            ;;
        2)
            script="$SCRIPT_DIR/run_cascade_stage2.sh"
            project_name="verl_cascade_stage2_science"
            local prev_ckpt=$(get_latest_ckpt "verl_cascade_stage1_math")
            if [ -n "$prev_ckpt" ]; then
                export PREV_STAGE_CKPT="$prev_ckpt"
                log "Resuming from Stage 1: $prev_ckpt"
            else
                log "WARNING: No Stage 1 checkpoint found"
            fi
            log "========== Stage 2: Science RL + OPD =========="
            log "Data: SciKnowEval"
            log "LR: 5e-7, Epochs: 5, rollout.n: 16"
            ;;
        3)
            script="$SCRIPT_DIR/run_cascade_stage3.sh"
            project_name="verl_cascade_stage3_code"
            local prev_ckpt=$(get_latest_ckpt "verl_cascade_stage2_science")
            if [ -n "$prev_ckpt" ]; then
                export PREV_STAGE_CKPT="$prev_ckpt"
                log "Resuming from Stage 2: $prev_ckpt"
            else
                log "WARNING: No Stage 2 checkpoint found"
            fi
            log "========== Stage 3: Code RL + OPD =========="
            log "Data: LiveCodeBench"
            log "LR: 5e-7, Epochs: 5, rollout.n: 16, resp_len: 2048"
            log "SandboxFusion: ${SANDBOX_URL:-http://localhost:8080/run_code}"
            ;;
        4)
            script="$SCRIPT_DIR/run_cascade_stage4.sh"
            project_name="verl_cascade_stage4_tool"
            local prev_ckpt=$(get_latest_ckpt "verl_cascade_stage3_code")
            if [ -n "$prev_ckpt" ]; then
                export PREV_STAGE_CKPT="$prev_ckpt"
                log "Resuming from Stage 3: $prev_ckpt"
            else
                log "WARNING: No Stage 3 checkpoint found"
            fi
            log "========== Stage 4: Tool-Use RL + OPD =========="
            log "Data: ToolAlpaca"
            log "LR: 3e-7, Epochs: 10, rollout.n: 16"
            ;;
        *)
            log "ERROR: Unknown stage $stage"
            exit 1
            ;;
    esac

    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "DRY RUN: bash $script"
        return 0
    fi

    local start_time=$(date +%s)
    log "Starting stage $stage..."

    if bash "$script"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        local hours=$((elapsed / 3600))
        local minutes=$(( (elapsed % 3600) / 60 ))
        log "Stage $stage completed in ${hours}h ${minutes}m"

        local final_ckpt=$(get_latest_ckpt "$project_name")
        if [ -n "$final_ckpt" ]; then
            log "Final checkpoint: $final_ckpt"
        else
            log "WARNING: No checkpoint found for $project_name"
        fi
    else
        log "ERROR: Stage $stage failed!"
        log "Check logs and resume with: bash scripts/run_all_stages.sh $stage"
        exit 1
    fi
}

log "Cascade RL Training Pipeline"
log "  Stages: $START_STAGE → $END_STAGE"
log "  Data:   ${DATA_PATH:-$HOME/data/cascade}"
log "  Reward: ${REWARD_FN_PATH:-/root/cascade/my_rewards.py}"
log "  Ckpts:  $CKPT_ROOT"
log ""

for stage in $(seq "$START_STAGE" "$END_STAGE"); do
    run_stage "$stage"
    log ""
done

log "========== All stages complete =========="
log ""

# Print final checkpoint paths
for project in verl_cascade_stage1_math verl_cascade_stage2_science verl_cascade_stage3_code verl_cascade_stage4_tool; do
    ckpt=$(get_latest_ckpt "$project")
    if [ -n "$ckpt" ]; then
        log "  $project → $ckpt"
    fi
done

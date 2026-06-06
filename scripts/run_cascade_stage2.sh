#!/usr/bin/env bash
# Stage 2: Science RL + OPD
# Data: SciKnowEval (28K, L3 reasoning)
# Reward: answer-matching (binary 0/1) + format reward (0.1)
# Resumes from: Stage 1 checkpoint
# Checkpoint: stage2_science/
set -xeo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export HCCL_BUFFSIZE=300
export VLLM_USE_V1=1

STUDENT_MODEL_PATH=/home/models/Haidass
TEACHER_MODEL_PATH=/home/models/Qwen3.6-35B-A3B
TOKENIZER_PATH=/home/train/Qwen3-0.6B

STUDENT_NGPUS=4
TEACHER_NGPUS=4
TEACHER_TP=4

DISTILLATION_LOSS_MODE=k1
USE_POLICY_GRADIENT=True

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
TRAIN_BATCH_SIZE=12
PPO_MINI_BATCH_SIZE=12
PPO_MAX_TOKEN_LEN_PER_GPU=16384

ACTOR_LR=5e-7
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16
USE_DYNAMIC_BSZ=True

ROLLOUT_GPU_MEM_UTIL=0.8
TEACHER_GPU_MEM_UTIL=0.5
ENFORCE_EAGER=False
TEACHER_ENFORCE_EAGER=False

ROLLOUT_N=16
TOTAL_EPOCHS=5
SAVE_FREQ=100
TEST_FREQ=10

PROJECT_NAME=verl_cascade_stage2_science
EXP_NAME="haidass_cascade_science_rl_opd"

DATA_PATH=${DATA_PATH:-$HOME/data/cascade}
REWARD_FN_PATH=${REWARD_FN_PATH:-/root/my_rewards.py}
CKPT_ROOT=${CKPT_ROOT:-/tmp/verl/checkpoints}

TRAIN_FILES="$DATA_PATH/science_train.parquet"
TEST_FILES="$DATA_PATH/science_test.parquet"

# Auto-detect latest Stage 1 checkpoint
if [ -z "${PREV_STAGE_CKPT:-}" ]; then
    PREV_STAGE_CKPT=$(ls -td "$CKPT_ROOT/$PROJECT_NAME"/*/global_step_* 2>/dev/null | head -1 || true)
    if [ -z "$PREV_STAGE_CKPT" ]; then
        PREV_STAGE_CKPT=$(ls -td "$CKPT_ROOT"/verl_cascade_stage1_math/*/global_step_* 2>/dev/null | head -1 || true)
    fi
fi

MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=True
    algorithm.kl_penalty=kl
    ++algorithm.kl_ctrl.kl_coef=0.01
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH"
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${STUDENT_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    actor_rollout_ref.rollout.free_cache_engine=False
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS}
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.key=default
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL_PATH"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.enforce_eager=${TEACHER_ENFORCE_EAGER}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs=64
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

REWARD=(
    custom_reward_function.path="$REWARD_FN_PATH"
    custom_reward_function.name=compute_score
    reward.num_workers=4
)

RESUME=()
if [ -n "${PREV_STAGE_CKPT:-}" ]; then
    echo "Resuming from Stage 1 checkpoint: $PREV_STAGE_CKPT"
    RESUME=(
        trainer.resume_mode=force
        trainer.resume_from_path="$PREV_STAGE_CKPT"
    )
else
    echo "WARNING: No Stage 1 checkpoint found. Training from scratch."
    echo "Set PREV_STAGE_CKPT=<path> to resume from a specific checkpoint."
    RESUME=(trainer.resume_mode=auto)
fi

TRAINER=(
    trainer.logger=console
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXP_NAME}
    trainer.n_gpus_per_node=${STUDENT_NGPUS}
    trainer.nnodes=1
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.device=npu
    trainer.log_val_generations=5
    "${RESUME[@]}"
)

cd /tmp/verl
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"

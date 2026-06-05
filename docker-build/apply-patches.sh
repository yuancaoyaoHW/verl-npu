#!/bin/bash
set -e

echo "Applying verl patches..."

# Patch 1: qwen3_5_moe config registration
echo "  [1/4] teacher_model.py - qwen3_5_moe CONFIG_MAPPING"
cp /opt/patches/teacher_model.py /opt/verl/verl/experimental/teacher_loop/

# Patch 2: precompute_prompt_logprobs
echo "  [2/4] teacher_manager.py - precompute_prompt_logprobs"
cp /opt/patches/teacher_manager.py /opt/verl/verl/experimental/teacher_loop/

# Patch 3: split teacher logprobs
echo "  [3/4] agent_loop.py - split teacher logprobs"
cp /opt/patches/agent_loop.py /opt/verl/verl/experimental/agent_loop/

# Patch 4: next-batch prefetch
echo "  [4/4] ray_trainer.py - next-batch prefetch"
cp /opt/patches/ray_trainer.py /opt/verl/verl/trainer/ppo/

echo "All patches applied successfully."

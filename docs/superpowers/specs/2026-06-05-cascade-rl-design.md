# Cascade RL + OPD Multi-Domain Training Design

**Date:** 2026-06-05
**Project:** Haidass OPD on NPU — Multi-Domain RL + Distillation
**Hardware:** 8× Ascend 910B3 (64GB HBM each)
**Status:** Design Complete — Ready for Implementation

---

## 1. Executive Summary

Extend the Haidass OPD pipeline from single-domain math distillation (GSM8K) to a **Cascade RL + OPD** training system spanning 4 domains: **Math → Science → Code → Tool-Use**.

Each stage combines:
1. **RL fine-tuning** with domain-specific binary rewards (GRPO, rollout.n=16)
2. **OPD distillation** from Qwen3.6-35B-A3B Teacher (single Teacher, pseudo-MOPD via system prompts)

Stages are trained **sequentially** (Nemotron-Cascade 2 approach): each stage resumes from the previous stage's checkpoint, preventing cross-domain interference while allowing capability accumulation.

### Target Outcomes

| Metric | Current (Wave 5.1) | After Cascade RL |
|--------|-------------------|------------------|
| Domains | Math only (GSM8K) | Math + Science + Code + Tool-Use |
| Training data | 7,473 samples | ~100K+ samples |
| Total training time | ~19h (single domain) | ~60-80h (4 stages) |
| Student capability | Math reasoning | Multi-domain reasoning |
| step time | 29.8s | ~30-45s (varies by domain) |

---

## 2. Architecture Overview

### 2.1 Cascade RL Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 1: Math RL + OPD                                              │
│   Data: GSM8K (7.5K) + DeepMath-103K (57K, difficulty ≥ 6)         │
│   Reward: answer-matching (binary 0/1) + format reward (0.1)        │
│   Checkpoint → stage1_final/                                        │
│                                                                      │
│   ─── resume ───→                                                    │
│                                                                      │
│ Stage 2: Science RL + OPD                                           │
│   Data: SciKnowEval (28K, L3 reasoning)                             │
│   Reward: answer-matching (binary 0/1) + format reward (0.1)        │
│   Checkpoint → stage2_final/                                        │
│                                                                      │
│   ─── resume ───→                                                    │
│                                                                      │
│ Stage 3: Code RL + OPD                                              │
│   Data: LiveCodeBench (release_v6, 131 eval + training splits)      │
│   Reward: SandboxFusion code execution (pass/fail) + format (0.1)   │
│   Checkpoint → stage3_final/                                        │
│                                                                      │
│   ─── resume ───→                                                    │
│                                                                      │
│ Stage 4: Tool-Use RL + OPD                                          │
│   Data: ToolAlpaca (3.9K instances)                                 │
│   Reward: tool-call matching (binary 0/1) + format reward (0.1)     │
│   Checkpoint → stage4_final/                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Pseudo-MOPD with Single Teacher

Instead of deploying multiple Teacher models (one per domain), we use **one Teacher** (Qwen3.6-35B-A3B) with **domain-specific system prompts** injected into the data:

```
┌─────────────────────────────────────────────────────────────────┐
│ Single Teacher: Qwen3.6-35B-A3B (NPU 4-7, TP=4)                │
│                                                                  │
│ System prompt conditions the Teacher's logprobs:                 │
│   "You are an expert mathematician..." → math-mode logprobs      │
│   "You are a science expert..."        → science-mode logprobs   │
│   "You are an expert programmer..."    → code-mode logprobs      │
│   "You are an expert at using tools..." → tool-mode logprobs     │
│                                                                  │
│ No code changes needed — the system prompt is part of the        │
│ prompt field in the parquet data. Teacher processes the full     │
│ sequence including system prompt, so its probability             │
│ distribution is automatically conditioned on the domain.         │
└─────────────────────────────────────────────────────────────────┘
```

This is "pseudo-MOPD" — functionally equivalent to multi-Teacher OPD when the Teacher model is large enough to exhibit domain-specific behavior under system prompt conditioning.

### 2.3 Reward Architecture

Each domain uses **binary task rewards** (0 or 1) plus a separate **format reward** (0.1 weight):

```
total_reward = task_reward + 0.1 × format_reward

task_reward:
  - Math:      1.0 if final answer matches ground_truth, else 0.0
  - Science:   1.0 if answer matches ground_truth, else 0.0
  - Code:      1.0 if code passes all test cases, else 0.0
  - Tool-Use:  1.0 if tool calls match expected, else 0.0

format_reward:
  - 1.0 if response follows expected format (e.g., "####" for math)
  - 0.0 otherwise
```

**GRPO normalization:** Rewards are normalized within each `uid` group. With `rollout.n=16`, each prompt generates 16 responses. GRPO computes `(reward - mean) / std` within the group, so only relative quality matters — absolute reward scale is irrelevant.

**Why rollout.n=16:** At n=4, 69% of groups produce zero gradient (all responses get the same reward). At n=16, this drops to <5%, ensuring stable policy gradient signal.

---

## 3. Data Pipeline

### 3.1 Dataset Inventory

| Dataset | Domain | Samples | Source | Format |
|---------|--------|---------|--------|--------|
| GSM8K | Math | 7,473 train / 1,319 test | OpenAI | Parquet (pre-converted) |
| DeepMath-103K | Math | 103K total, ~57K (difficulty ≥ 6) | HF: zwhe99/DeepMath-103K | Parquet → JSONL |
| SciKnowEval | Science | 28K total, L3 subset | HF: hicai-zju/SciKnowEval | Parquet |
| LiveCodeBench | Code | ~10K+ (release_v6) | HF: livecodebench/code_generation_lite | Parquet |
| ToolAlpaca | Tool-Use | 3,938 instances | GitHub: tangqiaoyu/ToolAlpaca | JSON |

### 3.2 Data Schema (verl OPD parquet)

Each dataset is converted to the same parquet schema:

```python
{
    "data_source": str,          # Routing key: "math/gsm8k", "math/deepmath",
                                  #   "science/sciknow", "code/livecode", "tool/toolalpaca"
    "prompt": list[dict],        # Chat format: [{"role": "system", "content": "..."},
                                  #               {"role": "user", "content": "..."}]
    "ability": str,              # Domain label: "math", "science", "code", "tool_use"
    "reward_model": str,         # JSON: {"ground_truth": "...", "style": "rule|code"}
    "extra_info": str,           # JSON: domain-specific metadata
}
```

### 3.3 System Prompt Strategy

Each domain gets a specialized system prompt injected as the first message in the `prompt` field:

| Domain | System Prompt |
|--------|--------------|
| Math | "You are an expert mathematician. Solve problems step by step with clear reasoning. Show your work and output the final answer after \"####\"." |
| Science | "You are a science expert specializing in biology, chemistry, physics, and materials science. Answer questions with precise scientific reasoning and cite relevant principles." |
| Code | "You are an expert programmer. Write clean, efficient, and correct code. Explain your approach before writing code. Handle edge cases." |
| Tool-Use | "You are an expert at using tools and APIs. Given a user request and available tools, determine the correct tool calls and parameters. Think step by step." |

### 3.4 Data Preparation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_datasets.sh` | Download all 4 datasets from HF/GitHub |
| `scripts/convert_datasets.py` | Convert raw data → verl parquet with system prompts |

**Usage:**
```bash
# 1. Download all datasets
bash scripts/download_datasets.sh

# 2. Convert to verl format (per-domain, for cascade stages)
python3 scripts/convert_datasets.py --datasets gsm8k deepmath --no-merge
python3 scripts/convert_datasets.py --datasets sciknow --no-merge
python3 scripts/convert_datasets.py --datasets livecode --no-merge
python3 scripts/convert_datasets.py --datasets toolalpaca --no-merge

# 3. Or merge all into one file (for mixed-domain training)
python3 scripts/convert_datasets.py --datasets gsm8k deepmath sciknow livecode toolalpaca \
    --output merged_train.parquet
```

---

## 4. Stage Specifications

### 4.1 Stage 1: Math RL + OPD

**Objective:** Reinforce math reasoning capability, then distill from Teacher.

**Data:**
- Train: GSM8K (7,473) + DeepMath-103K (57K, difficulty ≥ 6) = ~64.5K samples
- Test: GSM8K test (1,319)

**Reward function:** `math_reward_fn`
```python
def math_reward_fn(data_source, solution_str, ground_truth):
    """Extract final answer after #### and compare with ground_truth."""
    # 1. Parse response for "#### <answer>" pattern
    # 2. Normalize both predicted and ground_truth answers
    # 3. Return 1.0 if match, 0.0 otherwise
    # 4. Add format_reward = 0.1 if "####" found in response
```

**Config overrides (from Wave 5.1 base):**
```bash
# Data
data.train_files=/root/data/cascade/math_train.parquet
data.val_files=/root/data/cascade/math_test.parquet

# RL
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=16
algorithm.use_kl_in_reward=False

# Reward
reward.type=function
reward.function=math_reward_fn

# Training
trainer.total_epochs=5
trainer.save_freq=100
trainer.resume_mode=auto

# OPD (inherited from Wave 5.1)
distillation.enabled=True
distillation.distillation_loss.use_task_rewards=True
```

**Success criteria:**
- GSM8K test accuracy ≥ 85%
- Distillation loss converges
- No training instability (no crash in 500+ steps)

### 4.2 Stage 2: Science RL + OPD

**Objective:** Add science reasoning capability on top of math foundation.

**Data:**
- Train: SciKnowEval L3 subset (~15-20K after filtering)
- Test: SciKnowEval held-out split

**Reward function:** `science_reward_fn`
```python
def science_reward_fn(data_source, solution_str, ground_truth):
    """Match answer for science QA (multiple choice or short answer)."""
    # 1. Extract answer from response (last line, or after "Answer:")
    # 2. Compare with ground_truth (case-insensitive, strip whitespace)
    # 3. Return 1.0 if match, 0.0 otherwise
    # 4. Add format_reward = 0.1 if response has structured format
```

**Config overrides:**
```bash
# Resume from Stage 1
trainer.resume_mode=auto
trainer.resume_from_path=/root/checkpoints/stage1_final/global_step_XXX/

# Data
data.train_files=/root/data/cascade/science_train.parquet
data.val_files=/root/data/cascade/science_test.parquet

# Reward
reward.function=science_reward_fn

# Training
trainer.total_epochs=5
trainer.save_freq=100
```

**Success criteria:**
- SciKnowEval test accuracy ≥ 60%
- GSM8K test accuracy does not drop below Stage 1 final (no catastrophic forgetting)
- Distillation loss converges

### 4.3 Stage 3: Code RL + OPD

**Objective:** Add code generation capability with execution-based rewards.

**Data:**
- Train: LiveCodeBench training split (~8-10K problems)
- Test: LiveCodeBench release_v6 (131 eval problems)

**Reward function:** `code_reward_fn`
```python
def code_reward_fn(data_source, solution_str, ground_truth):
    """Execute generated code against test cases via SandboxFusion."""
    # 1. Extract Python code from ```python``` blocks
    # 2. Submit to SandboxFusion API
    # 3. Run against public test cases
    # 4. Return 1.0 if all tests pass, 0.0 otherwise
    # 5. Add format_reward = 0.1 if code block found
```

**SandboxFusion deployment:**
```bash
bash scripts/deploy_sandbox.sh  # Docker container on port 8080
```

**Config overrides:**
```bash
# Resume from Stage 2
trainer.resume_from_path=/root/checkpoints/stage2_final/global_step_XXX/

# Data
data.train_files=/root/data/cascade/code_train.parquet
data.val_files=/root/data/cascade/code_test.parquet

# Reward
reward.function=code_reward_fn
reward.sandbox_fusion.url=http://localhost:8080/run_code
reward.sandbox_fusion.max_concurrent=64
reward.sandbox_fusion.memory_limit_mb=1024

# Longer responses for code
data.max_response_length=2048    # Code responses are longer

# Training
trainer.total_epochs=5
trainer.save_freq=100
```

**Success criteria:**
- LiveCodeBench pass@1 ≥ 30%
- GSM8K accuracy maintained (no catastrophic forgetting)
- SandboxFusion execution stable under load

### 4.4 Stage 4: Tool-Use RL + OPD

**Objective:** Add tool/API calling capability.

**Data:**
- Train: ToolAlpaca (3,938 instances)
- Test: ToolAlpaca held-out split (20%)

**Reward function:** `tool_reward_fn`
```python
def tool_reward_fn(data_source, solution_str, ground_truth):
    """Match predicted tool calls against expected tool calls."""
    # 1. Extract tool calls from response (JSON or function call format)
    # 2. Compare tool name, parameters with ground_truth
    # 3. Return 1.0 if exact match, partial credit for name-only match
    # 4. Add format_reward = 0.1 if tool call format is valid
```

**Config overrides:**
```bash
# Resume from Stage 3
trainer.resume_from_path=/root/checkpoints/stage3_final/global_step_XXX/

# Data
data.train_files=/root/data/cascade/tool_train.parquet
data.val_files=/root/data/cascade/tool_test.parquet

# Reward
reward.function=tool_reward_fn

# Training
trainer.total_epochs=10          # Small dataset, more epochs
trainer.save_freq=50
```

**Success criteria:**
- Tool call accuracy ≥ 50%
- All previous domain accuracies maintained
- Distillation loss converges

---

## 5. Unified Reward Function

### 5.1 Design: `scripts/my_rewards.py`

A single reward function that routes by `data_source`:

```python
"""
Unified reward function for Cascade RL + OPD training.

Routes to domain-specific reward logic based on data_source field.
Each domain returns: task_reward (0/1) + format_reward (0.1)

Usage in verl config:
    reward.type=function
    reward.function=my_rewards.compute_reward
"""

import json
import re
from typing import Any


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict = None,
) -> float:
    """Route to domain-specific reward function."""

    # Parse ground_truth if it's a JSON string
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            ground_truth = gt.get("ground_truth", ground_truth)
        except (json.JSONDecodeError, AttributeError):
            pass

    if data_source.startswith("math/"):
        return _math_reward(solution_str, str(ground_truth))
    elif data_source.startswith("science/"):
        return _science_reward(solution_str, str(ground_truth))
    elif data_source.startswith("code/"):
        return _code_reward(solution_str, extra_info or {})
    elif data_source.startswith("tool/"):
        return _tool_reward(solution_str, str(ground_truth))
    else:
        return _default_reward(solution_str, str(ground_truth))


def _math_reward(solution: str, answer: str) -> float:
    """Math: extract answer after #### and compare."""
    FORMAT_WEIGHT = 0.1

    # Find "#### <answer>" pattern
    match = re.search(r"####\s*(.+)$", solution, re.MULTILINE)
    format_ok = match is not None

    if not match:
        return 0.0 + FORMAT_WEIGHT * (1.0 if format_ok else 0.0)

    predicted = _normalize_math_answer(match.group(1).strip())
    expected = _normalize_math_answer(answer.strip())

    task_reward = 1.0 if predicted == expected else 0.0
    return task_reward + FORMAT_WEIGHT * (1.0 if format_ok else 0.0)


def _science_reward(solution: str, answer: str) -> float:
    """Science: match answer (case-insensitive, strip)."""
    FORMAT_WEIGHT = 0.1

    # Try "Answer: X" pattern, fallback to last non-empty line
    match = re.search(r"(?:Answer|Answer|答案)[:\s]*(.+)$", solution, re.MULTILINE | re.IGNORECASE)
    if match:
        predicted = match.group(1).strip()
    else:
        lines = [l.strip() for l in solution.strip().split("\n") if l.strip()]
        predicted = lines[-1] if lines else ""

    format_ok = match is not None or len(solution.strip()) > 20
    task_reward = 1.0 if predicted.lower().strip() == answer.lower().strip() else 0.0
    return task_reward + FORMAT_WEIGHT * (1.0 if format_ok else 0.0)


def _code_reward(solution: str, extra_info: dict) -> float:
    """Code: execute via SandboxFusion (placeholder — actual execution in verl reward loop)."""
    FORMAT_WEIGHT = 0.1

    # Extract code block
    code_match = re.search(r"```python\s*\n(.*?)```", solution, re.DOTALL)
    format_ok = code_match is not None

    if not code_match:
        return 0.0

    # NOTE: Actual code execution happens in verl's reward loop
    # via SandboxFusion API. This function returns a placeholder.
    # The real execution result is passed via extra_info["execution_result"].
    execution_passed = extra_info.get("execution_result", False)
    task_reward = 1.0 if execution_passed else 0.0
    return task_reward + FORMAT_WEIGHT * (1.0 if format_ok else 0.0)


def _tool_reward(solution: str, ground_truth: str) -> float:
    """Tool-use: match tool calls."""
    FORMAT_WEIGHT = 0.1

    # Try to extract JSON tool calls from response
    try:
        json_match = re.search(r"```json\s*\n(.*?)```", solution, re.DOTALL)
        if json_match:
            predicted = json.loads(json_match.group(1))
            expected = json.loads(ground_truth)
            format_ok = True
        else:
            # Try direct JSON extraction
            json_match = re.search(r"\{.*\}", solution, re.DOTALL)
            if json_match:
                predicted = json.loads(json_match.group(0))
                expected = json.loads(ground_truth)
                format_ok = True
            else:
                return 0.0

        # Exact match
        if predicted == expected:
            return 1.0 + FORMAT_WEIGHT
        # Partial match: tool name matches
        if isinstance(predicted, dict) and isinstance(expected, dict):
            if predicted.get("tool") == expected.get("tool"):
                return 0.5 + FORMAT_WEIGHT
        return 0.0 + FORMAT_WEIGHT

    except (json.JSONDecodeError, TypeError):
        return 0.0


def _default_reward(solution: str, answer: str) -> float:
    """Fallback: simple string match."""
    FORMAT_WEIGHT = 0.1
    predicted = solution.strip().split("\n")[-1].strip().lower()
    expected = answer.strip().lower()
    task_reward = 1.0 if predicted == expected else 0.0
    return task_reward + FORMAT_WEIGHT * (1.0 if len(solution.strip()) > 20 else 0.0)


def _normalize_math_answer(s: str) -> str:
    """Normalize math answer for comparison."""
    s = s.strip()
    s = s.replace(",", "")           # Remove thousand separators
    s = s.replace("$", "").replace("\\$", "")  # Remove dollar signs
    s = s.strip(".")                  # Remove trailing period
    return s.lower()
```

### 5.2 GRPO Reward Integration

verl's GRPO computes advantages from rewards within each `uid` group:

```
uid = "{data_source}_{index}"   # auto-generated if not in data

For each uid group (rollout.n=16 responses to the same prompt):
  rewards = [r_1, r_2, ..., r_16]
  mean = mean(rewards)
  std = std(rewards)
  advantages = [(r_i - mean) / (std + eps) for r_i in rewards]
```

**Key insight:** If all 16 responses get reward=0 (all wrong), the group contributes zero gradient. With n=16, this is rare (<5% of groups). With n=4, it happens 69% of the time.

---

## 6. Training Configuration

### 6.1 Base Configuration (from Wave 5.1)

All stages inherit the Wave 5.1 base config:

```bash
# Environment
VLLM_USE_V1=1
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Student (NPU 0-3): DP=4, TP=1
STUDENT_MODEL_PATH=/home/models/Haidass
TOKENIZER_PATH=/home/train/Qwen3-0.6B
ENFORCE_EAGER=False
ROLLOUT_GPU_MEM_UTIL=0.8
+actor_rollout_ref.rollout.enable_sleep_mode=False
actor_rollout_ref.rollout.free_cache_engine=False

# Teacher (NPU 4-7): TP=4
TEACHER_MODEL_PATH=/home/models/Qwen3.6-35B-A3B
TEACHER_ENFORCE_EAGER=False
TEACHER_GPU_MEM_UTIL=0.5
TEACHER_TP=4

# Training
TRAIN_BATCH_SIZE=48
PPO_MINI_BATCH_SIZE=48
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16
PPO_MAX_TOKEN_LEN_PER_GPU=16384
ACTOR_LR=1e-6
USE_DYNAMIC_BSZ=True

# Distillation
DISTILLATION_LOSS_MODE=k1
USE_POLICY_GRADIENT=True
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024    # Override per stage if needed

# Safety
SAVE_FREQ=100
trainer.resume_mode=auto
```

### 6.2 Stage-Specific Overrides

| Parameter | Stage 1 (Math) | Stage 2 (Science) | Stage 3 (Code) | Stage 4 (Tool) |
|-----------|----------------|-------------------|----------------|----------------|
| `data.train_files` | math_train.parquet | science_train.parquet | code_train.parquet | tool_train.parquet |
| `data.val_files` | math_test.parquet | science_test.parquet | code_test.parquet | tool_test.parquet |
| `data.max_response_length` | 1024 | 1024 | 2048 | 1024 |
| `rollout.n` | 16 | 16 | 16 | 16 |
| `trainer.total_epochs` | 5 | 5 | 5 | 10 |
| `trainer.save_freq` | 100 | 100 | 100 | 50 |
| `trainer.resume_from_path` | — | stage1_final | stage2_final | stage3_final |
| `reward.function` | math_reward_fn | science_reward_fn | code_reward_fn | tool_reward_fn |

### 6.3 Learning Rate Schedule

```
Stage 1: lr = 1e-6 (baseline)
Stage 2: lr = 5e-7 (lower to avoid forgetting math)
Stage 3: lr = 5e-7
Stage 4: lr = 3e-7 (smallest — tool-use is new capability, gentle learning)
```

---

## 7. Checkpoint Management

### 7.1 Directory Structure

```
/root/checkpoints/
├── stage1_math/
│   └── verl_opd_cascade_stage1/
│       └── global_step_100/
│       └── global_step_200/
│       └── ...
│       └── global_step_XXX/    ← stage1_final (symlink)
├── stage2_science/
│   └── verl_opd_cascade_stage2/
│       └── global_step_100/
│       └── ...
├── stage3_code/
│   └── verl_opd_cascade_stage3/
│       └── global_step_100/
│       └── ...
├── stage4_tool/
│   └── verl_opd_cascade_stage4/
│       └── global_step_50/
│       └── ...
└── final/                       ← symlink to best stage4 checkpoint
```

### 7.2 Recovery Strategy

- `SAVE_FREQ=100`: Checkpoint every 100 steps
- `resume_mode=auto`: Automatically resume from latest checkpoint on crash
- Maximum steps lost per crash: 100 (vs 200 in Wave 5.1)
- After NPU graph capture crash (error 507000): kill zombie processes from host, restart container, training auto-resumes

### 7.3 Cross-Stage Resume

Each stage resumes from the previous stage's final checkpoint:

```bash
# Stage 2 resume from Stage 1
STAGE1_BEST=$(ls -td /root/checkpoints/stage1_math/verl_opd_cascade_stage1/global_step_* | head -1)
bash scripts/run_cascade_stage2.sh \
    trainer.resume_from_path=$STAGE1_BEST \
    trainer.resume_mode=force
```

---

## 8. Implementation Plan

### Phase 0: Infrastructure (1-2 hours)

| Task | File | Description |
|------|------|-------------|
| 0.1 | `scripts/my_rewards.py` | Unified reward function with domain routing |
| 0.2 | `scripts/run_cascade_stage1.sh` | Stage 1 launch script (math) |
| 0.3 | `scripts/run_cascade_stage2.sh` | Stage 2 launch script (science) |
| 0.4 | `scripts/run_cascade_stage3.sh` | Stage 3 launch script (code) |
| 0.5 | `scripts/run_cascade_stage4.sh` | Stage 4 launch script (tool-use) |
| 0.6 | `scripts/run_all_stages.sh` | Master orchestrator (sequential stage runner) |

### Phase 1: Data Preparation (2-3 hours)

| Task | Command | Description |
|------|---------|-------------|
| 1.1 | `bash scripts/download_datasets.sh` | Download all 4 datasets |
| 1.2 | `python3 scripts/convert_datasets.py --datasets gsm8k deepmath --no-merge` | Math data |
| 1.3 | `python3 scripts/convert_datasets.py --datasets sciknow --no-merge` | Science data |
| 1.4 | `python3 scripts/convert_datasets.py --datasets livecode --no-merge` | Code data |
| 1.5 | `python3 scripts/convert_datasets.py --datasets toolalpaca --no-merge` | Tool data |
| 1.6 | `docker cp` all parquets into container at `/root/data/cascade/` | Deploy data |

### Phase 2: Stage 1 — Math RL + OPD (6-8 hours)

| Task | Description |
|------|-------------|
| 2.1 | Smoke test: 10 steps with math data |
| 2.2 | Verify reward function: check reward distribution (should be ~50/50 binary) |
| 2.3 | Verify GRPO: check advantage normalization in logs |
| 2.4 | Full training: ~5 epochs on 64.5K samples |
| 2.5 | Validation: GSM8K test accuracy ≥ 85% |
| 2.6 | Save checkpoint, create `stage1_final` symlink |

### Phase 3: Stage 2 — Science RL + OPD (4-6 hours)

| Task | Description |
|------|-------------|
| 3.1 | Resume from Stage 1 checkpoint |
| 3.2 | Smoke test: 10 steps with science data |
| 3.3 | Verify no catastrophic forgetting: run GSM8K test at step 0 |
| 3.4 | Full training: ~5 epochs on ~15-20K samples |
| 3.5 | Validation: SciKnowEval accuracy ≥ 60%, GSM8K maintained |

### Phase 4: Stage 3 — Code RL + OPD (6-8 hours)

| Task | Description |
|------|-------------|
| 4.1 | Deploy SandboxFusion: `bash scripts/deploy_sandbox.sh` |
| 4.2 | Verify sandbox: test code execution API |
| 4.3 | Resume from Stage 2 checkpoint |
| 4.4 | Smoke test: 10 steps with code data + sandbox rewards |
| 4.5 | Full training: ~5 epochs on ~8-10K samples |
| 4.6 | Validation: LiveCodeBench pass@1 ≥ 30% |

### Phase 5: Stage 4 — Tool-Use RL + OPD (3-4 hours)

| Task | Description |
|------|-------------|
| 5.1 | Resume from Stage 3 checkpoint |
| 5.2 | Smoke test: 10 steps with tool data |
| 5.3 | Full training: ~10 epochs on 3.9K samples |
| 5.4 | Validation: tool call accuracy ≥ 50% |

### Phase 6: Final Evaluation (2-3 hours)

| Task | Description |
|------|-------------|
| 6.1 | Run all 4 domain test sets on final checkpoint |
| 6.2 | Compare per-domain accuracy vs single-domain baseline |
| 6.3 | Generate final performance report |
| 6.4 | Update docs with results |

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| NPU graph capture crash (error 507000) | Medium (seen at step 289) | Lose up to 100 steps | SAVE_FREQ=100 + resume_mode=auto |
| Catastrophic forgetting between stages | Medium | Previous domain accuracy drops | Lower LR per stage, validate all domains after each stage |
| SandboxFusion instability | Low | Code rewards unavailable | Fallback to regex-based code matching reward |
| Insufficient data for tool-use stage | Low | Overfitting on 3.9K samples | More epochs (10), aggressive regularization, or augment data |
| Teacher system prompt conditioning insufficient | Medium | Distillation quality degrades for non-math domains | Monitor Teacher logprob quality per domain; consider domain-specific Teachers if needed |
| verl reward function integration issues | Medium | Reward not computed correctly | Unit test reward function before training; log reward distribution at step 0 |
| OOM with longer code responses (2048 tokens) | Low | Training crash | Reduce batch size or micro_batch for Stage 3 |

---

## 10. Monitoring & Observability

### 10.1 Per-Stage Metrics

Each stage logs:
- `reward/task_reward_mean` — average task reward (should be 0.3-0.7 range)
- `reward/format_reward_mean` — format compliance rate
- `val/{domain}/acc/mean@1` — validation accuracy per domain
- `distillation/distillation_loss` — OPD loss convergence
- `actor/clip_ratio` — PPO clipping rate (should be 10-30%)
- `timing/gen_time`, `timing/step_time` — performance metrics

### 10.2 Cross-Stage Regression Tracking

After each stage, run validation on **all previous domains** to detect catastrophic forgetting:

```bash
# After Stage 2 (science), also evaluate math:
# val-math/gsm8k/acc/mean@1 should not drop below Stage 1 final - 2%
```

### 10.3 AICore Monitoring

Continue using existing monitoring infrastructure:
```bash
bash scripts/monitor_aicore.sh /tmp/cascade_stage1.log 2
python3 scripts/analyze_aicore.py /tmp/cascade_stage1.log
```

---

## 11. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/my_rewards.py` | **Create** | Unified reward function |
| `scripts/run_cascade_stage1.sh` | **Create** | Stage 1 launch script |
| `scripts/run_cascade_stage2.sh` | **Create** | Stage 2 launch script |
| `scripts/run_cascade_stage3.sh` | **Create** | Stage 3 launch script |
| `scripts/run_cascade_stage4.sh` | **Create** | Stage 4 launch script |
| `scripts/run_all_stages.sh` | **Create** | Master orchestrator |
| `scripts/convert_datasets.py` | **Modify** | Ensure `reward_model.ground_truth` field is correct |
| `scripts/run_opd.sh` | **Keep** | Base config reference (Wave 5.1) |
| `patches/` | **No changes** | Existing verl patches sufficient |

---

## 12. Decision Log

| Decision | Rationale |
|----------|-----------|
| Cascade (sequential) over mixed batch | Avoids cross-domain interference; each stage can tune LR/response_length independently |
| Single Teacher with system prompts (pseudo-MOPD) | No additional NPU cost; Teacher is large enough (35B-A3B) to exhibit domain-specific behavior under conditioning |
| rollout.n=16 | Minimum 8 for GRPO; n=16 ensures <5% zero-gradient groups (vs 69% at n=4) |
| SAVE_FREQ=100 (not 200) | Graph capture crash at step 289 showed we can lose up to 89 steps; 100 limits max loss |
| Binary rewards (0/1) + format (0.1) | GRPO auto-normalizes within uid groups; absolute scale doesn't matter |
| SandboxFusion for code execution | Docker-based sandbox with overlayfs+cgroups+network isolation; only needed for Stage 3 |
| Decreasing LR per stage | Prevents catastrophic forgetting of earlier domains |
| max_response_length=2048 for code | Code solutions need more tokens than math/science answers |
| 10 epochs for Stage 4 (tool-use) | Small dataset (3.9K) needs more passes for convergence |

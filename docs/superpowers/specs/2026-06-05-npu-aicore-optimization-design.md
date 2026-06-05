# NPU AICore Utilization Optimization Design

**Date:** 2026-06-05  
**Project:** Haidass OPD on NPU — GSM8K On-Policy Distillation  
**Hardware:** 8× Ascend 910B3 (64GB HBM each)  
**Goal:** Maximize AICore utilization (peak + avg) and minimize end-to-end training time

---

## 1. Current State Analysis

### System Configuration

```
Student: Qwen3-1.2B (Haidass, 596M params)
  Architecture: Qwen3ForCausalLM
  hidden_size: 1024, layers: 28, heads: 16, kv_heads: 8 (GQA)
  NPU 0-3: TP=1, DP=4
  
Teacher: Qwen3.6-35B-A3B (MoE)
  NPU 4-7: TP=4

Engine: vLLM 0.18.0 + vllm-ascend 0.18.0
  VLLM_USE_V1: NOT SET (running V0 engine)
  enforce_eager: False (NPU graph capture enabled)
  
Container: quay.io/ascend/vllm-ascend:v0.18.0
```

### Current Performance (Wave 4.0 Final from optimization report)

| Metric | Value |
|--------|-------|
| AICore peak | 35-51% |
| AICore avg (estimated) | ~25-30% |
| Throughput | 88.1 tok/s |
| MFU (actor_infer) | 9.3% |
| gen time | 30.9s |
| step time | 37.0s |
| clip_ratio | 10-25% |
| Total training time | ~24h (2325 steps × 37s) |
| Training stability | Crashed at step 289 with NPU graph capture |

### Three-Layer Bottleneck Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Engine Layer (V0)                                      │
│   - Scheduler and GPU execution are serial                      │
│   - No chunked prefill → prefill and decode cannot mix          │
│   - No prefix caching → repeated KV cache computation           │
│   Impact: AICore idle during scheduling gaps (~10-15%)          │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Algorithm Layer (Autoregressive Decode)                │
│   - Each step computes 1 token: [16, 1024] × [1024, 151936]   │
│   - Matrix too small, AICore Cube units mostly idle             │
│   - Decode is memory-bandwidth bound                            │
│   Impact: AICore ~50% even during active decode                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Pipeline Layer (Student-Teacher Serial)                │
│   - Student gen (31s) → old_log_prob (2s) → update (3s)        │
│   - Teacher NPU 4-7: 97% idle time                             │
│   - All NPUs wait during weight sync                            │
│   Impact: System-wide utilization ~50%                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Optimization Goals

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| AICore peak | 51% | 80%+ | +57% |
| AICore avg (gen phase) | ~35% | 60%+ | +71% |
| AICore avg (full step) | ~25% | 55%+ | +120% |
| step time | 37s | 20-22s | -41~46% |
| Total training time | ~24h | 13-15h | -38~46% |
| GSM8K accuracy | baseline | ±2% | — |

### AICore Monitoring Methodology

Current `npu-smi info` provides instantaneous samples. Need continuous sampling:

```bash
# Continuous sampling: every 2 seconds, all student NPUs
npu-smi info -t usages -i 0,1,2,3 -d 2 > /tmp/aicore_trace.log &

# Post-training analysis script: scripts/analyze_aicore.py
# Output: peak, avg, p50, p95, generation_phase_avg, full_step_avg
```

Per-wave validation must report:
- **AICore avg (generation phase)** — average during active token generation
- **AICore avg (full step)** — average over entire step including idle phases
- **AICore peak** — maximum observed value
- **AICore p50 / p95** — distribution statistics

---

## 3. Optimization Strategy: Progressive Waves

### Overview

| Wave | Content | Expected AICore avg | Expected step time | Risk |
|------|---------|--------------------|--------------------|------|
| **Wave 1** | V1 engine + chunked prefill + prefix caching | 45-55% | 28-32s | Low |
| **Wave 2** | Student TP=2 + batch tuning | 55-65% | 24-28s | Medium |
| **Wave 3** | Teacher MTP + prompt precompute + pipeline overlap | 55-65% | 20-22s | Medium |

**Note:** Speculative decoding for the Student model was evaluated and **removed** because the student model (596M params) is too small for a viable draft model. However, the **Teacher model** (Qwen3.6-35B-A3B) has built-in MTP (`mtp_num_hidden_layers=1`), which is leveraged in Wave 3 to accelerate Teacher logprobs computation without needing an external draft model.

---

## 4. Wave 1: V1 Engine Upgrade

### Objective

Switch from V0 to V1 engine, enable chunked prefill and prefix caching.

### V1 Engine Features

1. **Async Scheduler**: GPU executes batch N while scheduler prepares batch N+1. Eliminates decode scheduling gaps (~2-3ms/step).

2. **Chunked Prefill**: Prefill and decode requests can be mixed in the same batch. Prefill provides large matrix operations that saturate AICore during decode gaps.

3. **Prefix Caching**: Shared prefixes (GSM8K system prompt) reuse KV cache. Reduces prefill time by ~30%.

4. **Ascend Attention V1 Backend**: `attention_v1.py` optimized for 910B3 Cube units.

### Configuration Changes

**File:** `scripts/run_opd.sh`

```bash
# 1. Enable V1 engine (add to environment variables section)
export VLLM_USE_V1=1

# 2. ROLLOUT configuration (add new parameters)
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    # Wave 1 additions:
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
)

# 3. Teacher model also uses V1
DISTILLATION=(
    # ... existing config ...
    # Wave 1 additions:
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
)
```

### Validation Steps

1. **Smoke test (10 steps):** Run with `trainer.total_epochs=1 trainer.test_freq=5`
2. **AICore monitoring:** `npu-smi info -t usages -i 0,1,2,3 -d 2 > /tmp/wave1_aicore.log &`
3. **Metrics check:** Extract step time, gen time, throughput, MFU from logs
4. **Accuracy check:** Verify `val-core/openai/gsm8k/acc/mean@1` within ±2% of baseline

### Success Criteria

| Metric | Baseline (V0) | Wave 1 Target |
|--------|---------------|---------------|
| AICore peak | 51% | ≥65% |
| AICore avg (gen phase) | ~35% | ≥50% |
| AICore avg (full step) | ~25% | ≥40% |
| step time | 37s | ≤32s |
| throughput | 88 tok/s | ≥100 tok/s |
| GSM8K acc@1 | baseline | ±2% |
| Stability | stable | 50 steps without OOM/crash |

### Risks and Rollback

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| V1 + NPU graph capture instability | Monitor steps 200-300 for crash recurrence | Set `enforce_eager=True` |
| Teacher MoE incompatible with V1 | Test student V1 first, teacher V0 | Remove V1 params from DISTILLATION |
| Chunked prefill memory changes | Monitor HBM usage | Set `enable_chunked_prefill=False` |

---

## 5. Wave 2: Student Tensor Parallelism (TP=2)

### Objective

Switch student model from TP=1 (DP=4) to TP=2 (DP=2), increasing per-NPU compute density.

### Mechanism

```
Before (TP=1, DP=4):
  Each NPU: [16, 1024] × [1024, 151936]  (full vocab, small matrix)

After (TP=2, DP=2):
  Each NPU: [16, 1024] × [1024, 75968]   (half vocab)
  + AllReduce communication → additional AICore compute
```

AllReduce adds compute density that helps saturate AICore Cube units.

### Configuration Changes

**File:** `scripts/run_opd.sh`

```bash
# Student TP=2
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=2  # 1 → 2
    # ... other params from Wave 1 ...
)

# Adjust batch parameters for TP=2
STUDENT_MICRO_BATCH_SIZE_PER_GPU=8   # 16 → 8 (each GPU handles fewer sequences)
TRAIN_BATCH_SIZE=32                   # 48 → 32 (DP=2, each rank handles 16)
PPO_MINI_BATCH_SIZE=32                # sync with TRAIN_BATCH_SIZE
```

### GPU Allocation

```
Wave 1 (TP=1, DP=4):
  NPU 0: Student replica 0 (independent)
  NPU 1: Student replica 1 (independent)
  NPU 2: Student replica 2 (independent)
  NPU 3: Student replica 3 (independent)
  NPU 4-7: Teacher (TP=4)

Wave 2 (TP=2, DP=2):
  NPU 0-1: Student replica 0 (TP=2)
  NPU 2-3: Student replica 1 (TP=2)
  NPU 4-7: Teacher (TP=4)
```

### Success Criteria

| Metric | Wave 1 Baseline | Wave 2 Target |
|--------|-----------------|---------------|
| AICore peak | 65-75% | ≥70% |
| AICore avg (gen phase) | 50-55% | ≥58% |
| AICore avg (full step) | 40-45% | ≥48% |
| step time | 28-32s | ≤28s |
| throughput | 100+ tok/s | ≥110 tok/s |
| HCCL communication | — | No timeout/errors |

### Risks and Rollback

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| 596M model too small for TP=2 (comm > compute) | Monitor `update_actor time` | Revert to `tensor_model_parallel_size=1` |
| vllm-ascend TP=2 incomplete | Check release notes | Same |
| Suboptimal micro_batch | Test 8, 12, 16 | Select best from experiments |

### Fallback: TP=4

If TP=2 underperforms, try TP=4 (all 4 NPUs in one TP group, DP=1):
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
TRAIN_BATCH_SIZE=16
```

---

## 6. Wave 3: Teacher MTP + Precompute + Pipeline Overlap

### Objective

Maximize Teacher NPU 4-7 utilization (currently 97% idle) through three complementary optimizations: MTP-accelerated logprobs, prompt precomputation, and pipeline overlap.

### Sub-optimization 3A: Teacher MTP (Multi-Token Prediction)

**Key discovery:** Teacher model (Qwen3.6-35B-A3B) has MTP built in:
- `mtp_num_hidden_layers: 1` — one MTP prediction head
- `mtp_use_dedicated_embeddings: false`
- README confirms: "MTP: trained with multi-steps"

**Principle:** Teacher's MTP head predicts logprobs for the next token in the same forward pass as the main head. When computing `prompt_logprobs` over a (prompt + response) sequence, each forward pass produces logprobs for 2 tokens instead of 1, effectively doubling Teacher throughput.

```
Before (no MTP):
  Teacher forward pass 1: compute logprobs for token 1
  Teacher forward pass 2: compute logprobs for token 2
  ...
  Teacher forward pass N: compute logprobs for token N
  Total: N forward passes

After (MTP with num_speculative_tokens=2):
  Teacher forward pass 1: compute logprobs for tokens 1, 2 (main + MTP head)
  Teacher forward pass 2: compute logprobs for tokens 3, 4
  ...
  Total: N/2 forward passes → ~2x faster
```

**Configuration:**
```bash
# Teacher vLLM MTP config (add to DISTILLATION teacher inference)
distillation.teacher_models.teacher_model.inference.speculative_config='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

**Prerequisites:**
- Verify vLLM 0.18.0 supports `qwen3_next_mtp` method (README recommends vLLM ≥ 0.19.0)
- If 0.18.0 doesn't support it, upgrade vllm-ascend to 0.19.x
- Verify MTP works with `prompt_logprobs` mode (not just generation)

**Validation:**
1. Test MTP with a simple prompt_logprobs request against Teacher
2. Compare logprobs output: MTP vs non-MTP should be numerically identical
3. Measure Teacher forward time: should decrease ~40-50%

### Sub-optimization 3B: Teacher Prompt Logprobs Precomputation

**Principle:** Teacher computes logprobs over `(prompt + response)`. The prompt portion does not depend on student's response and can be precomputed before student finishes generation.

```
Before (serial per sample):
  Student generates response → Teacher computes (prompt + response) logprobs
  Total: gen_time + teacher_time

After (prompt precomputed):
  Teacher precomputes prompt logprobs ──→ Student generates response
                                           ↓
                                  Teacher computes response logprobs only
  Total: max(gen_time, teacher_prompt_time) + teacher_response_time
  Savings: ~1-2s/step (prompt is ~30-40% of teacher computation)
```

**Code changes:**
- `verl/experimental/teacher_loop/teacher_manager.py`: Add `precompute_prompt_logprobs()` method
- `verl/experimental/agent_loop/agent_loop.py`: Split `_compute_teacher_logprobs()` into prompt + response phases

**Synergy with 3A:** MTP makes both prompt precompute and response logprobs faster. Combined effect: Teacher time drops from ~3s to ~1s per step.

### Sub-optimization 3C: Next-batch Prefetch

**Principle:** While student is training (old_log_prob + update_actor + weight_sync), asynchronously start next batch's student generation.

```
Before (ray_trainer.py fit()):
  gen_output = generate_sequences(batch)     # blocking 31s
  sleep_replicas()
  old_log_prob = compute_old_log_prob(batch) # blocking 2s
  actor_output = update_actor(batch)         # blocking 3s
  update_weights()                           # blocking 1s
  # → next step starts generate

After:
  gen_output = generate_sequences(batch)
  sleep_replicas()
  next_gen_future = generate_sequences_async(next_batch)  # async prefetch
  old_log_prob = compute_old_log_prob(batch)
  actor_output = update_actor(batch)
  update_weights()
  next_gen_output = next_gen_future.result()  # may already be partially done
```

**Code changes:**
- `verl/trainer/ppo/ray_trainer.py`: Add prefetch logic in `fit()` training loop

### Success Criteria

| Metric | Wave 2 Baseline | Wave 3 Target |
|--------|-----------------|---------------|
| AICore peak | 70-80% | ≥75% |
| AICore avg (gen phase) | 58-65% | ≥60% |
| AICore avg (full step) | 48-55% | ≥55% |
| Teacher NPU utilization | ~8% | ≥40% |
| Teacher logprobs time | ~3s/step | ≤1.5s/step |
| step time | 24-28s | ≤22s |
| distillation loss | baseline | convergence trend matches |

### Risks and Rollback

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| vLLM 0.18.0 doesn't support `qwen3_next_mtp` | Upgrade to vllm-ascend 0.19.x | Skip 3A, proceed with 3B+3C |
| MTP doesn't work with `prompt_logprobs` mode | Test with simple request first | Disable MTP, use standard logprobs |
| MTP logprobs differ from standard | Compare numerically (should be identical) | Disable MTP |
| Prompt precompute logprobs differ from full-sequence | Compare precompute+concat vs full computation | Restore original `_compute_teacher_logprobs` |
| Hybrid engine prevents concurrent train+inference | If unsupported, skip 3C, only do 3A+3B | Remove prefetch logic |
| Code changes introduce bugs | Validate 3A, 3B, 3C separately | `git checkout` to pre-change state |

---

## 7. Monitoring Infrastructure

### AICore Trace Collection

Create `scripts/monitor_aicore.sh`:
```bash
#!/bin/bash
# Usage: bash scripts/monitor_aicore.sh <output_file> [interval_seconds]
OUTPUT=${1:-/tmp/aicore_trace.log}
INTERVAL=${2:-2}
npu-smi info -t usages -i 0,1,2,3,4,5,6,7 -d ${INTERVAL} > ${OUTPUT} &
echo "AICore monitoring started: PID=$!, output=${OUTPUT}, interval=${INTERVAL}s"
```

### AICore Analysis Script

Create `scripts/analyze_aicore.py`:
```python
#!/usr/bin/env python3
"""Analyze AICore utilization trace log.

Usage: python3 scripts/analyze_aicore.py <log_file> [--step-times <step_log>]

Output:
  - Per-NPU: peak, avg, p50, p95
  - Generation phase avg (requires step time log for phase alignment)
  - Full step avg
  - Teacher NPU utilization
"""
import sys
import re
import numpy as np

def parse_npu_smi_log(log_file):
    """Parse npu-smi output, extract AICore utilization per NPU per timestamp."""
    # Implementation: regex parse npu-smi tabular output
    # Returns: dict[npu_id -> list[(timestamp, aicore_util)]]
    pass

def compute_stats(data):
    """Compute peak, avg, p50, p95 for a list of utilization values."""
    arr = np.array(data)
    return {
        "peak": float(np.max(arr)),
        "avg": float(np.mean(arr)),
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }

def main():
    log_file = sys.argv[1]
    data = parse_npu_smi_log(log_file)
    for npu_id, values in data.items():
        stats = compute_stats([v for _, v in values])
        print(f"NPU {npu_id}: peak={stats['peak']:.1f}% avg={stats['avg']:.1f}% "
              f"p50={stats['p50']:.1f}% p95={stats['p95']:.1f}%")

if __name__ == "__main__":
    main()
```

### Per-Wave Validation Checklist

Each wave must complete:
- [ ] 10-step smoke test passes (no OOM, no crash)
- [ ] 50-step stability test passes
- [ ] AICore metrics collected and analyzed
- [ ] GSM8K validation accuracy within ±2% of baseline
- [ ] Distillation loss convergence trend matches baseline
- [ ] Rollback procedure tested and documented

---

## 8. Expected Cumulative Results

| Metric | Current | After Wave 1 | After Wave 2 | After Wave 3 |
|--------|---------|-------------|-------------|-------------|
| AICore peak | 51% | 65-75% | 70-80% | 75-80% |
| AICore avg (gen) | ~35% | 50-55% | 58-65% | 60-65% |
| AICore avg (full) | ~25% | 40-45% | 48-55% | 55-65% |
| step time | 37s | 28-32s | 24-28s | 20-22s |
| throughput | 88 tok/s | 100+ tok/s | 110+ tok/s | 125+ tok/s |
| Total training | ~24h | ~18-20h | ~16-18h | ~13-15h |
| Teacher NPU util | ~8% | ~8% | ~8% | 40-60% |
| Teacher logprobs time | ~3s/step | ~3s/step | ~3s/step | ≤1.5s/step |

---

## 9. Files Modified

| File | Wave | Change |
|------|------|--------|
| `scripts/run_opd.sh` | 1, 2, 3 | Environment variables, ROLLOUT config, batch params, Teacher MTP config |
| `scripts/monitor_aicore.sh` | — | New: AICore monitoring script |
| `scripts/analyze_aicore.py` | — | New: AICore analysis script |
| `verl/experimental/teacher_loop/teacher_manager.py` | 3 | Add prompt precompute method |
| `verl/experimental/agent_loop/agent_loop.py` | 3 | Split teacher logprobs computation |
| `verl/trainer/ppo/ray_trainer.py` | 3 | Add next-batch prefetch logic |
| vllm-ascend (container) | 3 | Possible upgrade from 0.18.0 to 0.19.x for MTP support |

---

## 10. Decision Log

| Decision | Rationale |
|----------|-----------|
| V1 engine with `enforce_eager=False` | User requested direct NPU graph capture testing |
| Speculative decoding removed for Student | Student model (596M) too small for viable draft model |
| Teacher MTP added to Wave 3 | Teacher model (Qwen3.6-35B-A3B) has built-in MTP (`mtp_num_hidden_layers=1`), no external draft model needed |
| Progressive waves (not parallel) | Conservative risk approach: each wave validated independently |
| AICore avg as core metric | Peak alone masks bubble; avg reflects true utilization |
| Teacher MTP (3A) before precompute (3B) before prefetch (3C) | 3A is config-only (lowest risk), 3B adds code, 3C adds most complexity |
| Possible vllm-ascend upgrade to 0.19.x | Qwen3.6 README recommends vLLM ≥ 0.19.0; MTP method `qwen3_next_mtp` may require it |

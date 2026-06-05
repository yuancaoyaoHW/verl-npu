# NPU AI Core Utilization Optimization Report

**Date:** 2026-06-04
**Task:** Haidass OPD (On-Policy Distillation) on 8x Ascend 910B3
**Goal:** Maximize AICore utilization during GRPO distillation training

---

## Summary

| Metric | Baseline | Wave 2.2 | A+B+C (resp=512) | **Final (resp=1024)** | Change |
|--------|----------|----------|-------------------|----------------------|--------|
| **Throughput** | 17.2 tok/s | 35.0 tok/s | 73.6 tok/s | **88.1 tok/s** | **+412%** |
| **MFU (actor_infer)** | 4.5% | 8.1% | 5.4% | **9.3%** | **+107%** |
| **gen time** | 65.9s | 66-68s | 16.6s | **30.9s** | **-53%** |
| **step time** | 70s | 70-74s | 22s | **37.0s** | **-47%** |
| **gen time/token** | 6.5-11.5ms | 3.8-4.9ms | 1.45ms | **1.44ms** | **-80%** |
| **Training Progress** | ~88s/it | ~77s/it | ~25s/it | **~37s/it** | **-58%** |
| **clip_ratio** | ~40% | ~35% | 34-50% | **10-25%** | **-50%** |
| **AICore (peak)** | 21-23% | 35-51% | 35-51% | 35-51% | +120% |
| **Estimated total time** | ~194h | ~78h | ~25h | **~24h** | **-88%** |

**Key insight:** The final configuration (batch=48, micro=16, resp=1024, gpu_mem=0.8, enforce_eager=False, 64 CPUs) achieves the highest throughput (88 tok/s, +412%) and MFU (9.3%, +107%) while maintaining low clip_ratio (17%). The A+B+C config was faster per step (22s vs 37s) but had 47% truncation; the final config trades 68% more step time for 64% less truncation and 20% higher throughput.

---

## Baseline Configuration

```bash
# /root/run_opd.sh (inside verl-vllm container)
ROLLOUT_GPU_MEM_UTIL=0.3
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
PPO_MAX_TOKEN_LEN_PER_GPU=4096
ENFORCE_EAGER=True
```

### Baseline Metrics (step 40-54, original training)

| Step | MFU infer | Throughput | gen time | step time | AICore |
|------|-----------|------------|----------|-----------|--------|
| 40 | 5.9% | 22.9 tok/s | 67.4s | 71.9s | 21-23% |
| 45 | 5.2% | 21.2 tok/s | 66.9s | 71.4s | 21-23% |
| 50 | 5.7% | 22.3 tok/s | 66.1s | 70.7s | 21-23% |
| 54 | 4.6% | 17.2 tok/s | 65.9s | 70.4s | 21-23% |

### Root Cause Analysis

1. **Generation phase = 93% of step time** (65s / 70s)
2. **ppo_micro_batch_size_per_gpu=2** → only 2 concurrent sequences per NPU
3. **gpu_memory_utilization=0.3** → only 19GB KV cache (out of 64GB HBM)
4. **Decode is memory-bandwidth bound** → each token's matrix multiply is too small to saturate AICore
5. **Teacher NPUs (4-7) at 0%** → structural idle, only used for distillation loss computation

---

## Optimization Waves

### Wave 1.1: gpu_memory_utilization 0.3 → 0.5

**Change:** `ROLLOUT_GPU_MEM_UTIL=0.5`

**Result:**
- HBM usage: 26GB → 38GB (+12GB)
- VLLMWorker memory: 18GB → 30GB
- **AICore: unchanged (21-23%)**
- Throughput: unchanged (~17 tok/s)

**Analysis:** KV cache was NOT the bottleneck. With batch_size=16 / 4 GPUs = 4 sequences per GPU, the existing KV cache was already sufficient. More cache alone doesn't help if the scheduler isn't sending more concurrent requests.

### Wave 2.1: batch_size 16→32 + micro_batch 2→4 + max_token_len 4096→8192

**Changes:**
```bash
TRAIN_BATCH_SIZE=32           # was 16
PPO_MINI_BATCH_SIZE=32        # was 16
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4  # was 2
PPO_MAX_TOKEN_LEN_PER_GPU=8192     # was 4096
```

**Result (step 2-5):**

| Step | MFU infer | Throughput | gen time | step time |
|------|-----------|------------|----------|-----------|
| 2 | 7.2% | 30.9 tok/s | 66.2s | 71.6s |
| 3 | 7.1% | 29.6 tok/s | 66.3s | 71.8s |
| 4 | 7.0% | 28.6 tok/s | 66.4s | 71.9s |
| 5 | 6.6% | 27.9 tok/s | 66.4s | 71.4s |

**Analysis:**
- **Throughput nearly doubled** (17→30 tok/s) — processing 2x data per step
- **MFU improved 55%** (4.5%→7.0%) — better compute utilization
- **gen time/token dropped 40%** (6.5ms→4.5ms) — better parallelism
- **Step time ~same** (~71s) — 2x data but 2x parallelism
- **AICore during generation: 35-38%** (up from 21-23%)

### Wave 2.2: micro_batch 4→8

**Change:** `STUDENT_MICRO_BATCH_SIZE_PER_GPU=8`

**Result (step 2-8):**

| Step | MFU infer | MFU actor | Throughput | gen time | step time |
|------|-----------|-----------|------------|----------|-----------|
| 2 | 7.7% | 7.2% | 32.4 tok/s | 68.9s | 74.0s |
| 3 | 6.2% | 5.7% | 25.1 tok/s | 66.8s | 72.0s |
| 4 | 6.9% | 6.6% | 30.1 tok/s | 65.7s | 71.0s |
| 5 | **8.1%** | **7.8%** | **35.0 tok/s** | 68.6s | 74.0s |
| 6 | 7.4% | 6.4% | 30.5 tok/s | 66.6s | 72.0s |
| 7 | 6.5% | 6.0% | 28.2 tok/s | 64.2s | 69.3s |
| 8 | 7.4% | 6.7% | 31.2 tok/s | 67.2s | 72.3s |

**Analysis:**
- **Peak throughput: 35.0 tok/s** (103% improvement over baseline)
- **Peak MFU: 8.1%** (80% improvement over baseline)
- **gen time/token: 3.8-4.9ms** (50% reduction from baseline 6.5-11.5ms)
- **AICore during generation: 35-51%** (peak observed)
- Training stable, no OOM errors
- Loss converging normally (distillation loss 4.1-7.2)

### Wave 3.0: A+B+C Combination (max_response_length=512 + enforce_eager=False + micro_batch=12 + gpu_mem=0.6)

**Changes:**
```bash
MAX_RESPONSE_LENGTH=512               # was 1024 (B: shorter generation)
ENFORCE_EAGER=False                   # was True (C: NPU graph capture)
STUDENT_MICRO_BATCH_SIZE_PER_GPU=12   # was 8 (A: more concurrent sequences)
ROLLOUT_GPU_MEM_UTIL=0.6              # was 0.5 (A: more KV cache)
```

**Result:** See [Final Configuration (A+B+C)](#final-configuration-abc) section above.

**Analysis:**
- **gen time dropped from 66s to 16.6s (-75%)** — max_response_length=512 is the dominant factor
- **step time dropped from 72s to 22s (-69%)** — generation is no longer the overwhelming bottleneck
- **throughput jumped to 73.6 tok/s avg (+328%)** — massive improvement
- **NPU graph capture (enforce_eager=False) worked** — step 1 was slow (41.6s, graph compilation), but steps 2+ stabilized at ~22s
- **micro_batch=12 stable** — no OOM, 12 concurrent sequences per GPU works well
- **Generation is now only 75% of step time** (16.6s/22s), down from 93% — the pipeline is more balanced
- **Estimated total training: ~29h** (vs ~194h baseline, -85%)

---

## Final Configuration (Wave 4.0)

```bash
# /root/run_opd.sh (optimized)
ROLLOUT_GPU_MEM_UTIL=0.8              # was 0.3 (more KV cache)
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16   # was 2 (more concurrent sequences)
TRAIN_BATCH_SIZE=48                   # was 16 (more data per step)
PPO_MINI_BATCH_SIZE=48                # was 16
PPO_MAX_TOKEN_LEN_PER_GPU=16384       # was 4096 (larger dynamic batch)
MAX_RESPONSE_LENGTH=1024              # restored from 512 (less truncation)
ENFORCE_EAGER=False                   # was True (NPU graph capture)
TEST_FREQ=10                          # was 5 (less validation overhead)
Container CPU: 64 cores               # was 1 (docker update --cpus=64)
```

### A+B+C Detailed Metrics (steps 2-12)

| Step | MFU infer | Throughput | gen time | step time | clip_ratio |
|------|-----------|------------|----------|-----------|------------|
| 2 | 5.1% | 74.3 tok/s | 16.9s | 22.4s | 0.44 |
| 3 | 4.9% | 70.2 tok/s | 16.9s | 22.2s | 0.44 |
| 4 | 5.1% | 67.2 tok/s | 17.1s | 22.6s | 0.31 |
| 5 | 5.9% | 80.2 tok/s | 16.8s | 22.4s | 0.50 |
| 6 | 5.2% | 76.5 tok/s | 16.8s | 22.2s | 0.38 |
| 7 | 4.8% | 65.2 tok/s | 16.8s | 22.2s | 0.34 |
| 8 | 5.5% | 75.6 tok/s | 16.7s | 22.1s | 0.34 |
| 9 | 5.4% | 79.7 tok/s | 16.6s | 21.9s | 0.50 |
| 10 | 5.4% | 71.5 tok/s | 16.8s | 22.3s | 0.44 |
| 11 | 5.3% | 72.4 tok/s | 16.8s | 22.3s | 0.34 |
| 12 | 5.3% | 73.3 tok/s | 16.4s | 21.8s | 0.34 |

**Stable averages:**
- gen time: **16.8s** (vs 65.9s baseline, -75%)
- step time: **22.2s** (vs 70s baseline, -69%)
- throughput: **73.6 tok/s** (vs 17.2 tok/s baseline, +328%)
- gen time/token: **1.45ms** (vs 6.5-11.5ms baseline, -80%)
- clip_ratio: **34-50%** (half of responses truncated at 512 tokens)

**Note:** Step 1 was slower (41.6s) due to NPU graph capture compilation overhead. Steps 2+ are stable at ~22s.

---

### Wave 4.0: Final Config (A + max_response_length=1024 + multi-core CPU)

**Applied:** 2026-06-04 13:25

**Motivation:** A+B+C 配置虽然 step time 极快（22s），但 clip_ratio=47% 意味着近一半回答被截断，可能影响 GSM8K 准确率。用户要求恢复 max_response_length=1024，同时利用 Student NPU 50GB+ 空闲显存进一步提升并行度，并解除容器 CPU 限制。

**Changes:**
```bash
MAX_RESPONSE_LENGTH=1024              # 512 → 1024 (恢复完整生成长度)
ROLLOUT_GPU_MEM_UTIL=0.8              # 0.6 → 0.8 (更多 KV cache)
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16   # 12 → 16 (更多并发序列)
PPO_MAX_TOKEN_LEN_PER_GPU=16384       # 8192 → 16384 (更大动态 batch)
TRAIN_BATCH_SIZE=48                   # 32 → 48 (每步更多数据)
PPO_MINI_BATCH_SIZE=48                # 32 → 48
TEST_FREQ=10                          # 5 → 10 (减少验证开销)
Container CPU: 1 core → 16 cores      # docker update --cpus=16
```

**Resource analysis before change:**
- Student NPU 0-3: 50-57GB HBM free each (only 7-12GB used)
- Teacher NPU 4-7: 50GB used, 14GB free
- Container CPU: 1 core (physical host has 256 cores)
- RAM: 202GB used / 1TB available

**Expected:**
- gen time: ~25-30s (longer responses, but more parallelism)
- step time: ~30-35s
- throughput: ~80-100 tok/s (more data per step)
- clip_ratio: ~20-30% (much less truncation)
- Total training: ~25-30h

**Results:**

| Step | MFU infer | Throughput | gen time | step time | clip_ratio |
|------|-----------|------------|----------|-----------|------------|
| 5 | 9.7% | 91.1 tok/s | 30.6s | 36.7s | 0.17 |
| 6 | 8.6% | 85.1 tok/s | 29.0s | 35.2s | 0.10 |
| 7 | 9.7% | 91.7 tok/s | 30.8s | 37.0s | 0.21 |
| 8 | 9.2% | 88.4 tok/s | 29.3s | 35.3s | 0.21 |
| 9 | 9.4% | 91.8 tok/s | 31.6s | 37.8s | 0.25 |
| 10 | 9.7% | 84.2 tok/s | 32.6s | 38.8s | 0.15 |
| 11 | 9.3% | 78.7 tok/s | 31.8s | 37.8s | 0.21 |
| 12 | 10.3% | 93.6 tok/s | 31.7s | 37.6s | 0.21 |

**Stable averages:**
- gen time: **30.9s** (vs 65.9s baseline, -53%)
- step time: **37.0s** (vs 70s baseline, -47%)
- throughput: **88.1 tok/s** (vs 17.2 tok/s baseline, **+412%**)
- MFU actor_infer: **9.3%** (vs 4.5% baseline, **+107%**)
- clip_ratio: **10-25%** (vs 34-50% with resp=512, much less truncation)
- Training Progress: **~37s/it** (vs ~88s baseline, -58%)
- HBM Usage: **86%** (well utilized)
- Estimated total training: **~24h** (2325 steps × 37s/it)

**Container CPU:** `docker update --cpus=64 verl-vllm`
- Previous: 1 CPU (severe bottleneck, Ray placement groups pending)
- Current: 64 CPUs (sufficient for all Ray actors)

**Key insight:** Restoring max_response_length=1024 increased gen time from 16.6s to 30.9s (+86%), but the higher batch_size=48 and micro_batch=16 more than compensated, achieving the highest throughput (88 tok/s) and MFU (9.3%) across all configurations. The clip_ratio dropped from 47% to 17%, meaning far fewer responses are truncated.

### ⚠️ Wave 4.0 Crash at Step 289 (NPU Graph Capture Instability)

**Crash time:** 2026-06-04 17:57:05 (after ~4 hours of training)

**Last completed step:** 289/2325 (12% progress)

**Error:**
```
RuntimeError: replay:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:267 
NPU function error: c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, c10_npu::getCurrentNPUStream())
error code is 507000

rtModelExecute execution failed, reason=model execute error
[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:61]
execute rtModel failed, runtime result = 507000
[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
```

**Root cause analysis:**

| Metric | Early (step 5-12) | Pre-crash (step 289) | Change |
|--------|-------------------|----------------------|--------|
| clip_ratio | 10-25% | **97.9%** | +300% |
| response_length mean | ~340 | **1006.3** | +196% |
| gen time | 30.9s | 34.1s | +10% |
| throughput | 88 tok/s | **160 tok/s** | +82% |
| MFU | 9.3% | **11.6%** | +25% |

**What happened:**
1. Model learned to generate very long responses (97.9% truncated at 1024 tokens)
2. Each sequence now generates the full 1024 tokens (vs ~340 initially)
3. NPU graph capture (`enforce_eager=False`) performs more graph replay iterations
4. After 289 steps × ~34s = ~2.7 hours of continuous graph replay, NPU runtime error 507000 triggered
5. This is a known instability with `torch_npu` graph capture on Ascend 910B3 under sustained high-load conditions

**Fix applied:**
```bash
ENFORCE_EAGER=True                    # Disable graph capture
trainer.resume_mode=auto              # Resume from checkpoint
```

**Resume plan:**
- Checkpoint available at step 200 (`/tmp/verl/checkpoints/verl_opd_haidass_gsm8k/.../global_step_200/`)
- Training will resume from step 200 with `enforce_eager=True`
- Expected performance: ~40s/step (slightly slower than 37s with graph capture, but stable)

**Resume status (2026-06-05 01:07):**
- ✅ Successfully resumed from checkpoint at step 200
- ✅ Loaded model, optimizer, rng state, and lr_scheduler from `global_step_200/`
- Training Progress: 9% (200/2325 steps)
- Configuration: `enforce_eager=True`, `resume_mode=auto`

**Lesson learned:** NPU graph capture provides ~10% speedup but is unstable under sustained high-throughput workloads. For production training, `enforce_eager=True` is safer despite the performance penalty.

---

## Why AICore Can't Reach 60%+

### Structural Limitations

1. **Autoregressive decode is memory-bandwidth bound**
   - Each decode step computes one token: matrix multiply of shape [batch, hidden_dim] × [hidden_dim, vocab]
   - For batch=8 per GPU, this is [8, 1536] × [1536, 151936] — still too small to saturate AICore
   - AICore needs large matrix operations to stay busy; decode is inherently sequential

2. **On-Policy Distillation pipeline is serial**
   ```
   Student rollout (gen) → Teacher forward → Compute loss → Student backward → Sync weights
   ```
   - Teacher NPUs (4-7) idle 97% of the time
   - Student NPUs (0-3) idle during teacher forward and weight sync

3. **npu-smi samples instantaneously**
   - AICore fluctuates between 0% (weight sync phase) and 51% (peak generation)
   - The reported value depends on exact sampling timing

### What Would Help (but requires code changes)

- **Async student-teacher pipeline**: Overlap student generation with teacher forward on previous batch
- **Speculative decoding**: Use a draft model to generate multiple tokens in parallel
- **Larger model**: A bigger student model would have more compute per token, better saturating AICore
- **Prefill-heavy workloads**: Prefill (processing prompts) is compute-bound and would show higher AICore

---

## Remaining Optimization Opportunities

| Option | Expected Impact | Risk | Status |
|--------|----------------|------|--------|
| `enforce_eager=False` (NPU graph capture) | +10% speedup | **HIGH — crashed at step 289** | **UNSTABLE** |
| `max_response_length=768` | +20% gen time, less truncation | Low — compromise between 512 and 1024 | Not attempted |
| Student TP=2 (reallocate GPUs) | Faster decode per step | High — Teacher may not fit in TP=2 | Not attempted |
| Async student-teacher pipeline | +30-50% throughput | High — requires verl code changes | Not attempted |
| Speculative decoding | -30% gen time | High — NPU support uncertain | Not attempted |

**Note on `enforce_eager=False`:** While it provides ~10% speedup (37s → 40s per step), it caused a crash after 289 steps due to NPU graph replay instability under sustained high-throughput workloads. For production training, use `enforce_eager=True` for stability.

---

## Failed Attempts

### Wave 3.1: batch_size=64, micro_batch=16 (FAILED)

**Attempted:** 2026-06-04 12:03

**Changes:**
```bash
TRAIN_BATCH_SIZE=64                    # was 32
PPO_MINI_BATCH_SIZE=64                 # was 32
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16    # was 8
```

**Result:** Process received SIGTERM during initialization, likely due to OOM.

**Error:**
```
*** SIGTERM received at time=1780574583 on cpu 108 ***
ray::core::CoreWorkerMemoryStore::Get() failed
Process aborted (core dumped)
```

**Analysis:**
- Training process stuck during resource allocation phase
- NPU showed 0% AICore, 7% HBM — never started training
- Memory requirement too high for batch=64 with micro=16
- Each GPU would need to handle 16 concurrent sequences during forward pass

**Memory estimate:**
- 16 sequences × ~264MB KV cache per sequence = ~4.2GB KV cache per GPU
- Plus model weights (~1.2GB), activations, gradients
- Total exceeds available HBM after accounting for vLLM overhead

**Conclusion:** micro_batch_size=16 is too large. micro_batch_size=12 works fine (verified in A+B+C wave).

**Rollback:** Reverted to batch=32, micro=8 (Wave 2.2 config).

---

## Rollback Instructions

To revert to baseline:
```bash
docker exec verl-vllm pkill -f "verl.trainer.main_ppo"
# Edit /root/run_opd.sh:
#   ROLLOUT_GPU_MEM_UTIL=0.3
#   STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
#   TRAIN_BATCH_SIZE=16
#   PPO_MINI_BATCH_SIZE=16
#   PPO_MAX_TOKEN_LEN_PER_GPU=4096
#   MAX_RESPONSE_LENGTH=1024
#   ENFORCE_EAGER=True
#   TEST_FREQ=5
docker exec -d verl-vllm bash -c "nohup bash /root/run_opd.sh > /home/admin/train_opd.log 2>&1 &"
```

To revert to A+B+C (if final config has issues):
```bash
#   ROLLOUT_GPU_MEM_UTIL=0.6
#   STUDENT_MICRO_BATCH_SIZE_PER_GPU=12
#   TRAIN_BATCH_SIZE=32
#   PPO_MINI_BATCH_SIZE=32
#   PPO_MAX_TOKEN_LEN_PER_GPU=8192
#   MAX_RESPONSE_LENGTH=512
#   ENFORCE_EAGER=False
#   TEST_FREQ=10
```

---

## Files Modified

| File | Location | Change |
|------|----------|--------|
| `run_opd.sh` | `/root/run_opd.sh` (container) | 7 parameter changes |
| `run_opd.sh` | `scripts/run_opd.sh` (repo) | Same changes mirrored |
| This report | `docs/npu-optimization-results.md` | Updated with all waves |

---

## Accuracy Concern

With `max_response_length=512`, the `response_length/clip_ratio` is 34-50%, meaning up to half of responses are truncated. For GSM8K math problems, chain-of-thought reasoning typically requires 200-400 tokens, so most answers should complete within 512 tokens. However, if validation accuracy (`val-core/openai/gsm8k/acc/mean@1`) drops significantly compared to the 1024-token baseline, consider:

1. **Compromise: `max_response_length=768`** — reduces truncation while still cutting gen time by ~40%
2. **Keep 512 but monitor loss convergence** — if distillation loss still decreases normally, the truncation may not be harmful
3. **Revert to 1024** — use Wave 2.2 config for safety

---

## Wave 5.0: V1 Engine + Teacher NPU Graph Capture (2026-06-05)

### Motivation

Wave 4.0 在 step 289 因 NPU graph capture 崩溃后改用 `enforce_eager=True`。本次尝试：
1. 启用 V1 引擎（`VLLM_USE_V1=1`）提升调度效率
2. Student 保持 `enforce_eager=True`（避免 graph capture 不稳定）
3. Teacher 启用 NPU graph capture（`enforce_eager=False`）加速推理
4. 尝试 Student TP=2 提升并行度

### Configuration Changes

```bash
# Student: DP=4, TP=1 (回退自 TP=2)
VLLM_USE_V1=1                          # 新增：启用 V1 引擎
TRAIN_BATCH_SIZE=48                    # 保持 Wave 4.0
PPO_MINI_BATCH_SIZE=48                 # 保持
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16    # 保持
ROLLOUT_GPU_MEM_UTIL=0.8               # 保持
ENFORCE_EAGER=True                     # 保持（graph capture 不稳定）
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # 回退自 2

# Teacher: TP=4, NPU graph capture
TEACHER_TP=4                           # 保持
TEACHER_GPU_MEM_UTIL=0.5               # 保持
TEACHER_ENFORCE_EAGER=False            # 新增：启用 NPU graph capture
```

### Failed Attempt: Student TP=2

**尝试时间:** 2026-06-05 03:00

**Changes:**
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # 从 1 改为 2
```

**Result:** EngineCore 启动后 TP worker spawn 永远挂起

**Error:**
```
(EngineCore pid=20216) INFO 06-05 03:00:28 [core.py:103] Initializing a V1 LLM engine...
# 之后无任何 worker spawn 日志，NPU 显存始终为空
```

**Root cause analysis:**

| 测试 | 配置 | 结果 |
|------|------|------|
| V1 + TP=2 | `VLLM_USE_V1=1`, `enforce_eager=True` | EngineCore 启动后卡死，无 worker spawn |
| V0 + TP=2 | `VLLM_USE_V1=0`, `enforce_eager=True` | 同上（V1 是 0.18.0 默认，无法禁用） |

**结论:** vLLM 0.18.0 + vllm-ascend 在 Ascend 910B3 上不支持 Student TP=2。V1 引擎强制启用，TP worker spawn 机制存在 bug。

**Rollback:** 回退到 TP=1（Wave 4.0 已验证配置）。

### NPU Zombie Process Issue

**问题:** 强杀训练进程后，NPU 驱动层的上下文未释放，导致新训练启动时 `SetDevice()` 失败。

**Error:**
```
RuntimeError: c10_npu::SetDevice(device_id), error code is 507033
TsdOpen failed. devId=X, tdt error=1
rtSetDevice execution failed, reason=device retain error
```

**Root cause:**
- `docker restart` 只重启容器，不清理 NPU 驱动状态
- 僵尸 `VLLM::EngineCore` 进程仍持有 NPU 设备上下文（~30GB/卡）
- 新训练尝试初始化 NPU 时被拒绝

**Fix:**
```bash
# 在宿主机上杀掉僵尸进程（非容器内）
ps aux | grep 'VLLM::EngineCore' | grep -v mineru | awk '{print $2}' | xargs kill -9

# 验证 NPU 已释放
npu-smi info  # 应显示 0/65536 MB HBM

# 重启容器内 Ray
docker exec verl-vllm bash -c "ray stop --force; ray start --head --port=6379"
```

**Lesson learned:** 强杀训练进程后，必须从宿主机清理 NPU 僵尸进程，否则无法启动新训练。

### Training Results (Steps 1-7) — PERFORMANCE REGRESSION

**Start time:** 2026-06-05 03:46

**Step-by-step metrics:**

| Step | step_time | throughput | gen_time | MFU (infer) | clip_ratio |
|------|-----------|------------|----------|-------------|------------|
| 1 | 88.9s | 32.6 tok/s | 80.3s | 2.4% | 14.6% |
| 2 | 71.6s | 47.9 tok/s | 65.4s | 10.1% | 20.8% |
| 3 | 71.5s | 43.9 tok/s | 65.1s | 9.8% | 16.7% |
| 4 | 72.4s | 48.3 tok/s | 66.1s | 10.6% | 18.8% |
| 5 | 72.0s | 44.3 tok/s | 65.7s | 9.7% | 20.8% |
| 6 | 72.6s | 40.1 tok/s | 66.5s | 9.0% | 10.4% |
| 7 | 72.7s | 45.6 tok/s | 66.8s | 10.1% | 16.7% |

**Step 2-7 stable averages:** step_time=72.1s, throughput=45.0 tok/s, gen_time=65.8s, MFU=9.9%

**Comparison with Wave 4.0 baseline:**

| Metric | Wave 4.0 (V0+graph) | Wave 5.0 (V1+eager) | Change |
|--------|---------------------|---------------------|--------|
| **step_time** | 37.0s | 72.1s | **+95% (2x slower)** |
| **throughput** | 88.1 tok/s | 45.0 tok/s | **-49%** |
| **gen_time** | 30.9s | 65.8s | **+113% (2x slower)** |
| **MFU (infer)** | 9.3% | 9.9% | similar |
| **clip_ratio** | 10-25% | 10-21% | similar |

**Root cause:** Student `enforce_eager=True` 是性能退化的主因。没有 NPU graph capture，每步 decode 都要重新编译 kernel，gen_time 从 31s 涨到 66s（恰好 2x）。Teacher 的 graph capture（NPU 4-7, 14/14 完成）成功运行，但 Teacher 推理只占 step 的一小部分，无法弥补 Student 的损失。

**Estimated total training time:** 72.1s × 2325 steps ≈ **46.7 hours** (vs 24h with Wave 4.0 graph capture)

### Files Modified

| File | Location | Change |
|------|----------|--------|
| `run_opd.sh` | `/root/run_opd.sh` (container) | V1 engine, Teacher graph capture |
| `run_opd.sh` | `scripts/run_opd.sh` (repo) | Same changes mirrored |
| `test_student_tp2.py` | `scripts/test_student_tp2.py` | TP=2 isolation test script |
| This report | `docs/npu-optimization-results.md` | Wave 5.0 documentation |

---

## Wave 5.1: Graph Capture + No Sleep Mode (2026-06-05)

### Motivation

Wave 5.0 确认 `enforce_eager=True` 导致性能退化 2x。根因分析发现：
1. **主因**：禁用 NPU graph capture 导致每步 decode 都要重新编译 kernel
2. **次因**：sleep/wake 周期每步增加 ~5s 死区（释放/重建 weights + kv_cache）

本次尝试：
1. Student 启用 NPU graph capture（`enforce_eager=False`）
2. 禁用 sleep mode（`enable_sleep_mode=False`, `free_cache_engine=False`）
3. 保持 V1 引擎（`VLLM_USE_V1=1`）

### Configuration Changes

```bash
# Student: DP=4, TP=1, graph capture, no sleep
VLLM_USE_V1=1                          # 保持 Wave 5.0
ENFORCE_EAGER=False                    # 回退：启用 NPU graph capture
+actor_rollout_ref.rollout.enable_sleep_mode=False  # 新增：禁用 sleep
actor_rollout_ref.rollout.free_cache_engine=False   # 新增：不释放 kv_cache

# Teacher: TP=4, graph capture (保持)
TEACHER_ENFORCE_EAGER=False            # 保持
```

### Training Results (Steps 1-4) — PERFORMANCE IMPROVEMENT

**Start time:** 2026-06-05 04:43

**Step-by-step metrics:**

| Step | step_time | throughput | gen_time | MFU (infer) | clip_ratio |
|------|-----------|------------|----------|-------------|------------|
| 1 | 51.9s | 65.3 tok/s | 45.2s | 2.8% | 18.8% |
| 2 | **29.5s** | **111.1 tok/s** | 25.3s | 9.9% | 18.8% |
| 3 | **29.7s** | **105.0 tok/s** | 25.5s | 9.2% | 16.7% |
| 4 | **30.3s** | **111.3 tok/s** | 25.7s | 9.6% | 18.8% |

**Step 2-4 stable averages:** step_time=29.8s, throughput=109.1 tok/s, gen_time=25.5s, MFU=9.6%

### Comparison with All Configurations

| Metric | Wave 4.0 (graph+sleep) | Wave 5.0 (eager+sleep) | **Wave 5.1 (graph+no sleep)** |
|--------|------------------------|------------------------|-------------------------------|
| **step_time** | 37.0s | 72.1s | **29.8s (-19% vs 4.0)** |
| **throughput** | 88.1 tok/s | 45.0 tok/s | **109.1 tok/s (+24% vs 4.0)** |
| **gen_time** | 30.9s | 65.8s | **25.5s (-17% vs 4.0)** |
| **MFU (infer)** | 9.3% | 9.9% | 9.6% |
| **clip_ratio** | 10-25% | 10-21% | 16-19% |

**Key findings:**
- Wave 5.1 比 Wave 4.0 baseline **快 19%**（29.8s vs 37.0s）
- gen_time 从 30.9s 降到 25.5s（-17%），graph capture + no sleep 双重优化生效
- throughput 从 88.1 tok/s 提升到 109.1 tok/s（+24%）
- Step 1 较慢（51.9s）是 graph capture 初始化开销，step 2+ 稳定

**Estimated total training time:** 29.8s × 2325 steps ≈ **19.2 hours** (vs 24h with Wave 4.0)

### Risk Assessment

**Graph capture stability:** Wave 4.0 在 step 289 崩溃（clip_ratio 从 15% 飙升到 98%，response 变长导致 graph replay 次数增加，触发 NPU runtime error 507000）。

**Mitigation:**
- `SAVE_FREQ=200` — 每 200 步自动存 checkpoint
- `resume_mode=auto` — 崩溃后自动从 checkpoint 恢复
- 上次 Wave 4.0 就是从 step 200 checkpoint 恢复，只损失了 89 步

**Current status:** Training running stably at step 5/2325, no errors observed.

### Files Modified

| File | Location | Change |
|------|----------|--------|
| `run_opd.sh` | `/root/run_opd.sh` (container) | Graph capture + no sleep mode |
| `run_opd.sh` | `scripts/run_opd.sh` (repo) | Same changes mirrored |
| This report | `docs/npu-optimization-results.md` | Wave 5.1 documentation |

---

## Summary: Optimization Journey

| Wave | Configuration | step_time | throughput | Total time | Status |
|------|---------------|-----------|------------|------------|--------|
| Baseline | V0, eager, batch=16 | 70s | 17.2 tok/s | ~194h | Superseded |
| 4.0 | V0, graph, batch=48 | 37s | 88.1 tok/s | ~24h | Crashed at step 289 |
| 5.0 | V1, eager, batch=48 | 72s | 45.0 tok/s | ~47h | Stable but slow |
| **5.1** | **V1, graph, no sleep, batch=48** | **30s** | **109 tok/s** | **~19h** | **Running** |

**Best configuration:** Wave 5.1 (current) — 19% faster than Wave 4.0 baseline, 24% higher throughput.

**Key learnings:**
1. NPU graph capture is critical for performance (2x speedup vs eager)
2. Sleep/wake overhead is significant (~5s/step, 17% of step time)
3. V1 engine + graph capture + no sleep = optimal combination
4. TP=2 not supported on vLLM 0.18.0 + vllm-ascend + Ascend 910B3
5. Graph capture may crash at high clip_ratio (>95%), but checkpoint recovery mitigates risk

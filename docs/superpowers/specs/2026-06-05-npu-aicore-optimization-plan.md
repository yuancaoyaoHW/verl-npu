# NPU AICore Optimization - Implementation Plan

**Date:** 2026-06-05  
**Spec:** `2026-06-05-npu-aicore-optimization-design.md`  
**Status:** Ready for Implementation  
**Estimated Total Time:** 8-12 hours (including validation)

---

## Overview

This plan implements the 3-wave optimization strategy to maximize AICore utilization and minimize training time for Haidass OPD on NPU.

**Target Outcomes:**
- AICore avg (full step): 25% → 55-65% (+120%)
- step time: 37s → 20-22s (-41~46%)
- Total training time: ~24h → ~13-15h (-38~46%)

---

## Phase 0: Preparation (30 min)

### Task 0.1: Create Monitoring Scripts

**Files to create:**
- `scripts/monitor_aicore.sh`
- `scripts/analyze_aicore.py`

**Steps:**
1. Create `scripts/monitor_aicore.sh`:
```bash
#!/bin/bash
# Usage: bash scripts/monitor_aicore.sh <output_file> [interval_seconds]
OUTPUT=${1:-/tmp/aicore_trace.log}
INTERVAL=${2:-2}
npu-smi info -t usages -i 0,1,2,3,4,5,6,7 -d ${INTERVAL} > ${OUTPUT} &
echo "AICore monitoring started: PID=$!, output=${OUTPUT}, interval=${INTERVAL}s"
```

2. Create `scripts/analyze_aicore.py` (see spec section 7 for full implementation)

3. Make executable:
```bash
chmod +x scripts/monitor_aicore.sh
chmod +x scripts/analyze_aicore.py
```

**Validation:**
- [ ] `bash scripts/monitor_aicore.sh /tmp/test.log 1` starts monitoring
- [ ] `python3 scripts/analyze_aicore.py /tmp/test.log` parses output correctly

---

## Phase 1: Wave 1 - V1 Engine Upgrade (2-3 hours)

### Task 1.1: Enable V1 Engine

**File:** `scripts/run_opd.sh`

**Changes:**
1. Add to environment variables section (after line 16):
```bash
export VLLM_USE_V1=1
```

2. Add to ROLLOUT array (after line 90):
```bash
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.enable_prefix_caching=True
actor_rollout_ref.rollout.max_num_batched_tokens=8192
```

3. Add to DISTILLATION array (after line 110):
```bash
distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
```

**Validation:**
- [ ] Smoke test: `bash scripts/run_opd.sh trainer.total_epochs=1 trainer.test_freq=5`
- [ ] Monitor: `bash scripts/monitor_aicore.sh /tmp/wave1_aicore.log 2`
- [ ] Run 10 steps, check for OOM/crash
- [ ] Analyze: `python3 scripts/analyze_aicore.py /tmp/wave1_aicore.log`

**Success Criteria:**
- AICore peak ≥ 65%
- AICore avg (gen phase) ≥ 50%
- step time ≤ 32s
- throughput ≥ 100 tok/s
- GSM8K acc@1 within ±2% of baseline

**Rollback:**
```bash
# Remove VLLM_USE_V1=1
# Remove enable_chunked_prefill and enable_prefix_caching lines
```

### Task 1.2: Stability Test (50 steps)

**Steps:**
1. Run full training with Wave 1 config:
```bash
bash scripts/run_opd.sh trainer.test_freq=10
```

2. Monitor for 50 steps:
```bash
bash scripts/monitor_aicore.sh /tmp/wave1_stability.log 2
```

3. Check logs for:
- [ ] No OOM errors
- [ ] No NPU crashes (especially around step 200-300)
- [ ] Distillation loss converging normally
- [ ] Validation accuracy stable

**Success Criteria:**
- 50 consecutive steps without crash
- Distillation loss trend matches baseline

**Rollback:**
If unstable, set `enforce_eager=True` and retry. If still unstable, revert to V0.

---

## Phase 2: Wave 2 - Student TP=2 (2-3 hours)

### Task 2.1: Configure TP=2

**File:** `scripts/run_opd.sh`

**Changes:**
1. Modify ROLLOUT array (line 83):
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # was 1
```

2. Modify batch parameters:
```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=8   # was 16
TRAIN_BATCH_SIZE=32                   # was 48
PPO_MINI_BATCH_SIZE=32                # was 48
```

**Validation:**
- [ ] Smoke test: `bash scripts/run_opd.sh trainer.total_epochs=1 trainer.test_freq=5`
- [ ] Monitor: `bash scripts/monitor_aicore.sh /tmp/wave2_aicore.log 2`
- [ ] Run 10 steps
- [ ] Check HCCL communication: `grep -i "hccl\|allreduce" train_opd.log | head -20`

**Success Criteria:**
- AICore peak ≥ 70%
- AICore avg (gen phase) ≥ 58%
- step time ≤ 28s
- throughput ≥ 110 tok/s
- No HCCL timeout/errors

**Rollback:**
```bash
# Revert tensor_model_parallel_size to 1
# Revert batch sizes to Wave 1 values
```

### Task 2.2: Batch Size Tuning

**Steps:**
1. Test micro_batch=8 (baseline for Wave 2)
2. If stable, test micro_batch=12:
```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=12
TRAIN_BATCH_SIZE=48
PPO_MINI_BATCH_SIZE=48
```

3. If stable, test micro_batch=16:
```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=16
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
```

**Validation:**
- [ ] Each config runs 10 steps without OOM
- [ ] Compare throughput and AICore utilization
- [ ] Select best performing config

**Success Criteria:**
- Find optimal micro_batch that maximizes throughput without OOM

### Task 2.3: Stability Test (50 steps)

**Steps:**
1. Run with best config from Task 2.2:
```bash
bash scripts/run_opd.sh trainer.test_freq=10
```

2. Monitor for 50 steps:
```bash
bash scripts/monitor_aicore.sh /tmp/wave2_stability.log 2
```

3. Check for:
- [ ] No OOM errors
- [ ] No HCCL communication failures
- [ ] Distillation loss converging
- [ ] Validation accuracy within ±2%

**Success Criteria:**
- 50 consecutive steps without issues
- Performance metrics stable

**Rollback:**
If TP=2 underperforms, try TP=4:
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
```

---

## Phase 3: Wave 3 - Teacher MTP + Pipeline Overlap (4-6 hours)

### Task 3.1: Verify MTP Support

**Steps:**
1. Check vLLM version:
```bash
docker exec verl-vllm python3 -c "import vllm; print(vllm.__version__)"
```

2. Test MTP with simple request:
```bash
docker exec verl-vllm python3 << 'EOF'
from vllm import LLM
try:
    llm = LLM(
        model='/home/models/Qwen3.6-35B-A3B',
        speculative_config={"method": "qwen3_next_mtp", "num_speculative_tokens": 2},
        device='npu',
        tensor_parallel_size=4,
    )
    print("MTP supported!")
except Exception as e:
    print(f"MTP not supported: {e}")
EOF
```

**Decision Point:**
- If MTP supported → proceed to Task 3.2
- If MTP not supported → skip to Task 3.3 (prompt precompute)

### Task 3.2: Enable Teacher MTP

**File:** `scripts/run_opd.sh`

**Changes:**
Add to DISTILLATION array:
```bash
distillation.teacher_models.teacher_model.inference.speculative_config='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

**Validation:**
- [ ] Smoke test: `bash scripts/run_opd.sh trainer.total_epochs=1 trainer.test_freq=5`
- [ ] Monitor Teacher NPU: `bash scripts/monitor_aicore.sh /tmp/wave3_mtp.log 2`
- [ ] Run 10 steps
- [ ] Check Teacher logprobs time: `grep "teacher_logprobs" train_opd.log | tail -10`

**Success Criteria:**
- Teacher logprobs time ≤ 1.5s/step (was ~3s)
- Teacher NPU utilization ≥ 40%
- Distillation loss numerically matches baseline

**Rollback:**
```bash
# Remove speculative_config line
```

### Task 3.3: Implement Prompt Precomputation

**Files to modify:**
- `verl/experimental/teacher_loop/teacher_manager.py`
- `verl/experimental/agent_loop/agent_loop.py`

**Changes:**

1. **teacher_manager.py** - Add method after line 128:
```python
async def precompute_prompt_logprobs(
    self,
    prompt_ids: list[int],
    multi_modal_data: Optional[dict[str, Any]] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    routing_key: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute teacher logprobs for prompt portion only."""
    teacher_key = self._resolve_teacher_key(routing_key)
    teacher_model_config = self.teacher_model_configs[teacher_key]
    client = self.teacher_client[teacher_key]
    
    teacher_output = await client.generate(
        request_id=uuid4().hex,
        prompt_ids=prompt_ids,
        sampling_params=_get_teacher_sampling_params(
            teacher_model_config, self.distillation_loss_config
        ),
        image_data=multi_modal_data.get("images") if multi_modal_data else None,
        video_data=multi_modal_data.get("videos") if multi_modal_data else None,
        audio_data=multi_modal_data.get("audios") if multi_modal_data else None,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    
    teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
    teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
    return teacher_ids, teacher_logprobs
```

2. **agent_loop.py** - Modify `_compute_teacher_logprobs` (line 903-926):
```python
async def _compute_teacher_logprobs(
    self,
    output: AgentLoopOutput,
    prompt_ids: list[int],
    response_ids: list[int],
    validate: bool,
    sample_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Compute teacher logprobs with prompt precomputation."""
    if self.distillation_enabled and not validate:
        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
        
        # Check if prompt logprobs were precomputed
        cached_prompt_logprobs = await self._get_cached_prompt_logprobs(prompt_ids, routing_key)
        
        if cached_prompt_logprobs is not None:
            # Use precomputed prompt logprobs, compute response logprobs only
            prompt_teacher_ids, prompt_teacher_logprobs = cached_prompt_logprobs
            
            resp_teacher_ids, resp_teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            
            # Concatenate
            teacher_ids = torch.cat([prompt_teacher_ids, resp_teacher_ids])
            teacher_logprobs = torch.cat([prompt_teacher_logprobs, resp_teacher_logprobs])
        else:
            # Fallback to full computation
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
        
        output.extra_fields["teacher_ids"] = teacher_ids
        output.extra_fields["teacher_logprobs"] = teacher_logprobs
```

3. Add helper method to AgentLoopWorker:
```python
async def _get_cached_prompt_logprobs(
    self,
    prompt_ids: list[int],
    routing_key: Optional[str] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Retrieve cached prompt logprobs if available."""
    # Implementation: check if precompute was triggered and cache exists
    # For now, return None (fallback to full computation)
    # TODO: Implement caching mechanism
    return None
```

**Validation:**
- [ ] Smoke test: `bash scripts/run_opd.sh trainer.total_epochs=1 trainer.test_freq=5`
- [ ] Run 10 steps
- [ ] Compare distillation loss with baseline (should be numerically identical)
- [ ] Check Teacher time: `grep "teacher" train_opd.log | tail -20`

**Success Criteria:**
- Distillation loss matches baseline (within 1e-4)
- Teacher time reduced by ~1-2s/step

**Rollback:**
```bash
git checkout verl/experimental/teacher_loop/teacher_manager.py
git checkout verl/experimental/agent_loop/agent_loop.py
```

### Task 3.4: Implement Next-batch Prefetch

**File:** `verl/trainer/ppo/ray_trainer.py`

**Changes:**

1. Add prefetch state to `__init__` (after line 372):
```python
self._next_gen_future = None
self._next_batch = None
```

2. Modify `fit()` method (around line 1467):
```python
# Before:
with marked_timer("gen", timing_raw, color="red"):
    combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
    self.checkpoint_manager.sleep_replicas()

# After:
with marked_timer("gen", timing_raw, color="red"):
    if self._next_gen_future is not None:
        # Use prefetched result
        combined_gen_output = self._next_gen_future.result()
        self._next_gen_future = None
    else:
        # First step or prefetch failed
        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
    self.checkpoint_manager.sleep_replicas()
```

3. Add prefetch trigger (after line 1675, before weight sync):
```python
# Prefetch next batch
try:
    next_batch_dict = next(iter(self.train_dataloader))
    self._next_batch = DataProto.from_single_dict(next_batch_dict)
    self._next_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
    next_gen_batch = self._get_gen_batch(self._next_batch)
    next_gen_batch.meta_info["global_steps"] = self.global_steps + 1
    rollout_n = self.config.actor_rollout_ref.rollout.n
    next_gen_batch = next_gen_batch.repeat(repeat_times=rollout_n, interleave=True)
    
    # Async prefetch
    self._next_gen_future = self.async_rollout_manager.generate_sequences_async(next_gen_batch)
except StopIteration:
    # End of dataloader, no prefetch
    self._next_gen_future = None
```

**Note:** This requires `generate_sequences_async` method in AgentLoopManager. If not available, use ThreadPoolExecutor:
```python
from concurrent.futures import ThreadPoolExecutor
self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

# In prefetch trigger:
self._next_gen_future = self._prefetch_executor.submit(
    self.async_rollout_manager.generate_sequences, next_gen_batch
)
```

**Validation:**
- [ ] Smoke test: `bash scripts/run_opd.sh trainer.total_epochs=1 trainer.test_freq=5`
- [ ] Run 10 steps
- [ ] Check gen time: `grep "gen time" train_opd.log | tail -10`
- [ ] Verify no dataloader issues

**Success Criteria:**
- gen time reduced by ~2-5s (prefetch hides latency)
- No dataloader errors
- Training metrics stable

**Rollback:**
```bash
git checkout verl/trainer/ppo/ray_trainer.py
```

### Task 3.5: Full Wave 3 Integration Test

**Steps:**
1. Run with all Wave 3 optimizations:
```bash
bash scripts/run_opd.sh trainer.test_freq=10
```

2. Monitor for 50 steps:
```bash
bash scripts/monitor_aicore.sh /tmp/wave3_full.log 2
```

3. Analyze results:
```bash
python3 scripts/analyze_aicore.py /tmp/wave3_full.log
```

**Success Criteria:**
- AICore peak ≥ 75%
- AICore avg (full step) ≥ 55%
- Teacher NPU utilization ≥ 40%
- step time ≤ 22s
- Total training time ~13-15h
- GSM8K acc@1 within ±2% of baseline

---

## Phase 4: Final Validation (1 hour)

### Task 4.1: Long-run Stability Test

**Steps:**
1. Run full training with all optimizations:
```bash
bash scripts/run_opd.sh
```

2. Monitor continuously:
```bash
bash scripts/monitor_aicore.sh /tmp/final_run.log 2
```

3. Check at milestones:
- [ ] Step 100: No crashes, metrics stable
- [ ] Step 500: Distillation loss converging
- [ ] Step 1000: Validation accuracy on track
- [ ] Step 2000: Performance sustained

**Success Criteria:**
- Complete full training (~2325 steps) without crash
- Final GSM8K accuracy within ±2% of baseline
- Average step time ≤ 22s
- AICore avg (full step) ≥ 55%

### Task 4.2: Performance Report

**Steps:**
1. Generate final report:
```bash
python3 scripts/analyze_aicore.py /tmp/final_run.log > /tmp/final_report.txt
```

2. Document:
- Final AICore metrics (peak, avg, p50, p95)
- Final step time and throughput
- Total training time
- GSM8K accuracy
- Comparison with baseline

3. Update `docs/npu-optimization-results.md` with Wave 1-3 results

---

## Summary

| Phase | Tasks | Time | Risk |
|-------|-------|------|------|
| Phase 0: Preparation | Create monitoring scripts | 30 min | Low |
| Phase 1: Wave 1 | V1 engine + chunked prefill | 2-3 hours | Low |
| Phase 2: Wave 2 | Student TP=2 + batch tuning | 2-3 hours | Medium |
| Phase 3: Wave 3 | Teacher MTP + precompute + prefetch | 4-6 hours | Medium |
| Phase 4: Validation | Long-run test + report | 1 hour | Low |
| **Total** | | **8-12 hours** | |

---

## Quick Reference

### Rollback Commands

**Wave 1 rollback:**
```bash
# Remove VLLM_USE_V1=1 and chunked prefill/prefix caching lines
```

**Wave 2 rollback:**
```bash
# Revert tensor_model_parallel_size to 1
# Revert batch sizes
```

**Wave 3 rollback:**
```bash
git checkout verl/experimental/teacher_loop/teacher_manager.py
git checkout verl/experimental/agent_loop/agent_loop.py
git checkout verl/trainer/ppo/ray_trainer.py
# Remove speculative_config line
```

### Key Metrics to Monitor

- **AICore peak**: Should increase from 51% → 80%+
- **AICore avg (gen phase)**: Should increase from ~35% → 60%+
- **AICore avg (full step)**: Should increase from ~25% → 55%+
- **step time**: Should decrease from 37s → 20-22s
- **throughput**: Should increase from 88 tok/s → 125+ tok/s
- **Teacher NPU util**: Should increase from ~8% → 40-60%

### Files Modified

1. `scripts/run_opd.sh` - Configuration changes (all waves)
2. `scripts/monitor_aicore.sh` - New monitoring script
3. `scripts/analyze_aicore.py` - New analysis script
4. `verl/experimental/teacher_loop/teacher_manager.py` - Prompt precompute (Wave 3)
5. `verl/experimental/agent_loop/agent_loop.py` - Split teacher logprobs (Wave 3)
6. `verl/trainer/ppo/ray_trainer.py` - Prefetch logic (Wave 3)

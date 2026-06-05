#!/usr/bin/env python3
"""Minimal test: Student model (Haidass) with vLLM TP=2 on NPU.

Purpose: Isolate whether the hang is caused by Student TP=2 or the full training pipeline.
"""

import os
import time

# Force NPU 0,1 only for Student TP=2
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
os.environ["NPU_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Source Ascend toolkit
os.system("source /usr/local/Ascend/ascend-toolkit/set_env.sh")

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/models/Haidass"
MAX_MODEL_LEN = 1536  # 512 prompt + 1024 response

print(f"[TEST] Loading Student model from {MODEL_PATH} with TP=2...")
print(f"[TEST] VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")
print(f"[TEST] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")

start_time = time.time()

try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.8,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=8192,
        trust_remote_code=True,
    )
    load_time = time.time() - start_time
    print(f"[TEST] Model loaded successfully in {load_time:.2f}s")

    # Simple generation test
    prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Explain quantum computing in one sentence.",
    ]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    print(f"[TEST] Running generation test with {len(prompts)} prompts...")
    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - gen_start

    print(f"[TEST] Generation completed in {gen_time:.2f}s")
    for i, output in enumerate(outputs):
        print(f"  Prompt {i+1}: {output.prompt[:50]}...")
        print(f"  Output: {output.outputs[0].text[:100]}...")

    print(f"[TEST] SUCCESS - Student TP=2 works!")
    print(f"[TEST] Total time: {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"[TEST] FAILED - {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

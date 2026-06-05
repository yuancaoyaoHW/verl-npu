#!/usr/bin/env python3
"""Test Student TP=2 with V0 engine (disable V1)."""

import os
import time

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
os.environ["NPU_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_V1"] = "0"  # Explicitly disable V1
os.environ["NCCL_P2P_DISABLE"] = "1"

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/models/Haidass"
MAX_MODEL_LEN = 1536

print(f"[TEST-V0] Loading Student model with TP=2, V0 engine...")
print(f"[TEST-V0] VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")

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
    print(f"[TEST-V0] Model loaded in {load_time:.2f}s")

    prompts = ["Hello, how are you?", "What is 2 + 2?"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    print(f"[TEST-V0] Running generation test...")
    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - gen_start

    print(f"[TEST-V0] Generation completed in {gen_time:.2f}s")
    for i, output in enumerate(outputs):
        print(f"  Output {i+1}: {output.outputs[0].text[:100]}")

    print(f"[TEST-V0] SUCCESS!")

except Exception as e:
    print(f"[TEST-V0] FAILED - {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

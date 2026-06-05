#!/usr/bin/env python3
"""
Convert multiple datasets to verl OPD parquet format with domain-specific system prompts.

This enables "pseudo-MOPD" with a single teacher model:
  - Each domain gets a different system prompt injected into the data
  - The teacher sees the system prompt when computing logprobs
  - The teacher's responses are conditioned on the role, effectively "playing" different experts
  - Student also sees the system prompt during generation

Usage:
  python3 convert_datasets.py                          # convert all available datasets
  python3 convert_datasets.py --datasets gsm8k deepmath # convert specific ones
  python3 convert_datasets.py --output merged_train.parquet

Output format (verl OPD parquet):
  - data_source: str          # routing key (e.g., "math/gsm8k", "math/deepmath", "science/sciknow")
  - prompt: list[dict]        # chat format with system + user messages
  - ability: str              # domain label
  - reward_model: str (JSON)  # ground truth + evaluation style
  - extra_info: str (JSON)    # metadata

System prompt strategy:
  The system prompt is prepended to each sample's prompt field.
  When the teacher computes logprobs, it processes the full sequence including
  the system prompt, so its probability distribution is conditioned on the role.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# Domain-specific system prompts
# ============================================================

SYSTEM_PROMPTS = {
    "math": (
        "You are an expert mathematician. Solve problems step by step with clear reasoning. "
        "Show your work and output the final answer after \"####\"."
    ),
    "code": (
        "You are an expert programmer. Write clean, efficient, and correct code. "
        "Explain your approach before writing code. Handle edge cases."
    ),
    "science": (
        "You are a science expert specializing in biology, chemistry, physics, and materials science. "
        "Answer questions with precise scientific reasoning and cite relevant principles."
    ),
    "tool_use": (
        "You are an expert at using tools and APIs. Given a user request and available tools, "
        "determine the correct tool calls and parameters. Think step by step."
    ),
    "instruction": (
        "You are a helpful AI assistant. Follow instructions carefully and provide "
        "well-structured, detailed responses."
    ),
}


def get_system_prompt(domain: str) -> str:
    """Get system prompt for a domain, with fallback to instruction."""
    return SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["instruction"])


# ============================================================
# Dataset converters
# ============================================================

def convert_gsm8k(input_path: str, max_samples: int = None) -> pd.DataFrame:
    """Convert GSM8K to verl format with math system prompt."""
    print(f"  Loading GSM8K from {input_path}...")
    df = pd.read_parquet(input_path)

    # GSM8K already has the right format, just add system prompt
    system_prompt = get_system_prompt("math")
    new_prompts = []
    for prompt in df["prompt"]:
        if isinstance(prompt, list) and len(prompt) > 0:
            if prompt[0].get("role") != "system":
                prompt = [{"role": "system", "content": system_prompt}] + prompt
        new_prompts.append(prompt)

    df["prompt"] = new_prompts
    df["data_source"] = "math/gsm8k"
    df["ability"] = "math"

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  GSM8K: {len(df)} samples")
    return df


def convert_deepmath(input_dir: str, min_difficulty: float = 6.0, max_samples: int = None) -> pd.DataFrame:
    """Convert DeepMath-103K to verl format with math system prompt."""
    print(f"  Loading DeepMath-103K from {input_dir}...")

    # Try common file patterns
    for pattern in ["data/train*.parquet", "data/*.parquet", "*.parquet", "train*.jsonl"]:
        import glob
        files = glob.glob(os.path.join(input_dir, pattern))
        if files:
            break

    if not files:
        # Try loading as HF dataset
        try:
            from datasets import load_dataset
            ds = load_dataset(input_dir, split="train")
            df = ds.to_pandas()
        except Exception:
            print(f"  ERROR: Cannot find data files in {input_dir}")
            return pd.DataFrame()
    else:
        if files[0].endswith(".parquet"):
            df = pd.concat([pd.read_parquet(f) for f in sorted(files)])
        else:
            df = pd.concat([pd.read_json(f, lines=True) for f in sorted(files)])

    # Filter by difficulty
    if "difficulty" in df.columns:
        before = len(df)
        df = df[df["difficulty"] >= min_difficulty].reset_index(drop=True)
        print(f"  Filtered difficulty >= {min_difficulty}: {before} -> {len(df)}")

    # Convert to verl format
    system_prompt = get_system_prompt("math")
    rows = []
    for _, row in df.iterrows():
        question = row.get("question", row.get("problem", ""))
        answer = row.get("final_answer", row.get("answer", ""))
        solution = row.get("r1_solution_1", row.get("solution", ""))

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question} Let's think step by step and output the final answer after \"####\"."},
        ]

        rows.append({
            "data_source": "math/deepmath",
            "prompt": prompt,
            "ability": "math",
            "reward_model": json.dumps({"ground_truth": str(answer), "style": "rule"}),
            "extra_info": json.dumps({
                "difficulty": row.get("difficulty", 0),
                "topic": row.get("topic", ""),
                "solution": solution[:500] if solution else "",
            }),
        })

    result = pd.DataFrame(rows)
    if max_samples and len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  DeepMath: {len(result)} samples")
    return result


def convert_sciknow(input_dir: str, level: str = "L3", max_samples: int = None) -> pd.DataFrame:
    """Convert SciKnowEval to verl format with science system prompt."""
    print(f"  Loading SciKnowEval from {input_dir}...")

    try:
        from datasets import load_dataset
        ds = load_dataset(input_dir, split="train")
        df = ds.to_pandas()
    except Exception:
        import glob
        files = glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
        if not files:
            files = glob.glob(os.path.join(input_dir, "**/*.json*"), recursive=True)
        if not files:
            print(f"  ERROR: Cannot find data files in {input_dir}")
            return pd.DataFrame()
        if files[0].endswith(".parquet"):
            df = pd.concat([pd.read_parquet(f) for f in sorted(files)])
        else:
            df = pd.concat([pd.read_json(f, lines=True) for f in sorted(files)])

    # Filter by level if column exists
    if "level" in df.columns:
        before = len(df)
        df = df[df["level"] == level].reset_index(drop=True)
        print(f"  Filtered level={level}: {before} -> {len(df)}")

    system_prompt = get_system_prompt("science")
    rows = []
    for _, row in df.iterrows():
        question = row.get("question", row.get("input", row.get("text", "")))
        answer = row.get("answer", row.get("output", row.get("label", "")))
        domain = row.get("domain", row.get("subject", "science"))

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ]

        rows.append({
            "data_source": f"science/sciknow",
            "prompt": prompt,
            "ability": "science",
            "reward_model": json.dumps({"ground_truth": str(answer), "style": "rule"}),
            "extra_info": json.dumps({
                "domain": domain,
                "level": row.get("level", ""),
                "task_type": row.get("task_type", row.get("type", "")),
            }),
        })

    result = pd.DataFrame(rows)
    if max_samples and len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  SciKnow: {len(result)} samples")
    return result


def convert_livecode(input_dir: str, release: str = "release_v6", max_samples: int = None) -> pd.DataFrame:
    """Convert LiveCodeBench to verl format with code system prompt."""
    print(f"  Loading LiveCodeBench from {input_dir}...")

    try:
        from datasets import load_dataset
        ds = load_dataset(input_dir, split="test")
        df = ds.to_pandas()
    except Exception:
        import glob
        files = glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
        if not files:
            print(f"  ERROR: Cannot find data files in {input_dir}")
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in sorted(files)])

    # Filter by release if column exists
    if "release" in df.columns:
        before = len(df)
        df = df[df["release"] == release].reset_index(drop=True)
        print(f"  Filtered release={release}: {before} -> {len(df)}")

    system_prompt = get_system_prompt("code")
    rows = []
    for _, row in df.iterrows():
        question = row.get("question_content", row.get("problem", row.get("question", "")))
        tests = row.get("public_test_cases", row.get("tests", ""))

        user_content = str(question)
        if tests:
            user_content += f"\n\nExample test cases:\n{str(tests)[:500]}"
        user_content += "\n\nWrite a Python solution. Put your code in ```python``` blocks."

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        rows.append({
            "data_source": "code/livecode",
            "prompt": prompt,
            "ability": "code",
            "reward_model": json.dumps({"ground_truth": "", "style": "code"}),
            "extra_info": json.dumps({
                "problem_id": row.get("question_id", row.get("id", "")),
                "difficulty": row.get("difficulty", ""),
                "release": row.get("release", ""),
                "platform": row.get("platform", ""),
            }),
        })

    result = pd.DataFrame(rows)
    if max_samples and len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  LiveCodeBench: {len(result)} samples")
    return result


def convert_toolalpaca(input_dir: str, max_samples: int = None) -> pd.DataFrame:
    """Convert ToolAlpaca to verl format with tool-use system prompt."""
    print(f"  Loading ToolAlpaca from {input_dir}...")

    train_file = os.path.join(input_dir, "data", "train_data.json")
    if not os.path.exists(train_file):
        train_file = os.path.join(input_dir, "train_data.json")
    if not os.path.exists(train_file):
        import glob
        candidates = glob.glob(os.path.join(input_dir, "**/*train*.json"), recursive=True)
        if candidates:
            train_file = candidates[0]
        else:
            print(f"  ERROR: Cannot find train_data.json in {input_dir}")
            return pd.DataFrame()

    with open(train_file) as f:
        data = json.load(f)

    system_prompt = get_system_prompt("tool_use")
    rows = []
    for item in data:
        apis = item.get("api_list", [])
        user_request = item.get("user_request", item.get("query", ""))
        api_desc = json.dumps(apis, ensure_ascii=False)[:1000] if apis else ""

        user_content = user_request
        if api_desc:
            user_content = f"Available tools:\n{api_desc}\n\nUser request: {user_request}"

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Ground truth: the expected tool calls
        gt = item.get("answer", item.get("expected_output", ""))

        rows.append({
            "data_source": "tool/toolalpaca",
            "prompt": prompt,
            "ability": "tool_use",
            "reward_model": json.dumps({"ground_truth": str(gt), "style": "rule"}),
            "extra_info": json.dumps({
                "num_apis": len(apis),
                "api_names": [a.get("name", "") for a in apis[:5]],
            }),
        })

    result = pd.DataFrame(rows)
    if max_samples and len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  ToolAlpaca: {len(result)} samples")
    return result


# ============================================================
# Main
# ============================================================

CONVERTERS = {
    "gsm8k": convert_gsm8k,
    "deepmath": convert_deepmath,
    "sciknow": convert_sciknow,
    "livecode": convert_livecode,
    "toolalpaca": convert_toolalpaca,
}


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to verl OPD format with system prompts")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"],
                        choices=list(CONVERTERS.keys()),
                        help="Which datasets to convert")
    parser.add_argument("--data-dir", default="datasets",
                        help="Root directory containing downloaded datasets")
    parser.add_argument("--gsm8k-path", default=None,
                        help="Path to existing GSM8K parquet (default: auto-detect)")
    parser.add_argument("--output", default="merged_train.parquet",
                        help="Output merged parquet file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--min-difficulty", type=float, default=6.0,
                        help="Min difficulty for DeepMath filtering")
    parser.add_argument("--no-merge", action="store_true",
                        help="Save each dataset separately instead of merging")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_dfs = []

    for name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Converting: {name}")
        print(f"{'='*50}")

        if name == "gsm8k":
            # Try to find existing GSM8K parquet
            gsm8k_path = args.gsm8k_path
            if not gsm8k_path:
                for candidate in [
                    "/root/data/gsm8k/train.parquet",
                    "data/gsm8k/train.parquet",
                    str(data_dir / "gsm8k" / "train.parquet"),
                ]:
                    if os.path.exists(candidate):
                        gsm8k_path = candidate
                        break
            if not gsm8k_path:
                print("  ERROR: GSM8K parquet not found. Use --gsm8k-path to specify.")
                continue
            df = convert_gsm8k(gsm8k_path, args.max_samples)

        elif name == "deepmath":
            df = convert_deepmath(str(data_dir / "DeepMath-103K"), args.min_difficulty, args.max_samples)

        elif name == "sciknow":
            df = convert_sciknow(str(data_dir / "SciKnowEval"), max_samples=args.max_samples)

        elif name == "livecode":
            df = convert_livecode(str(data_dir / "LiveCodeBench"), max_samples=args.max_samples)

        elif name == "toolalpaca":
            df = convert_toolalpaca(str(data_dir / "ToolAlpaca"), args.max_samples)

        if len(df) > 0:
            all_dfs.append(df)

            if args.no_merge:
                out_path = f"{name}_train.parquet"
                df.to_parquet(out_path, index=False)
                print(f"  Saved: {out_path}")

    if not all_dfs:
        print("\nERROR: No datasets converted.")
        sys.exit(1)

    # Merge
    merged = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*50}")
    print(f"Merged dataset summary")
    print(f"{'='*50}")
    print(f"Total samples: {len(merged)}")
    print(f"\nBy data_source:")
    for src, count in merged["data_source"].value_counts().items():
        print(f"  {src}: {count}")
    print(f"\nBy ability:")
    for ability, count in merged["ability"].value_counts().items():
        print(f"  {ability}: {count}")

    # Show sample
    print(f"\nSample prompts (first per domain):")
    for src in merged["data_source"].unique():
        sample = merged[merged["data_source"] == src].iloc[0]
        prompt = sample["prompt"]
        sys_msg = next((m["content"][:80] for m in prompt if m["role"] == "system"), "N/A")
        user_msg = next((m["content"][:80] for m in prompt if m["role"] == "user"), "N/A")
        print(f"\n  [{src}]")
        print(f"    system: {sys_msg}...")
        print(f"    user:   {user_msg}...")

    merged.to_parquet(args.output, index=False)
    print(f"\nSaved merged dataset: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")

    print(f"\n{'='*50}")
    print(f"Next steps:")
    print(f"{'='*50}")
    print(f"1. Copy to container:")
    print(f"   docker cp {args.output} verl-vllm:/root/data/mopd/train.parquet")
    print(f"")
    print(f"2. Update run_opd.sh for MOPD routing:")
    print(f"   data.train_files=/root/data/mopd/train.parquet")
    print(f"   distillation.teacher_key=data_source")
    print(f"")
    print(f"3. For pseudo-MOPD (single teacher, different roles):")
    print(f"   No config change needed! The system prompts in the data")
    print(f"   will condition the teacher's logprobs automatically.")
    print(f"")
    print(f"4. For true MOPD (multiple teacher models):")
    print(f"   +distillation.teacher_models.math.key=math/gsm8k")
    print(f"   +distillation.teacher_models.math.model_path=...")
    print(f"   +distillation.teacher_models.code.key=code/livecode")
    print(f"   +distillation.teacher_models.code.model_path=...")


if __name__ == "__main__":
    main()

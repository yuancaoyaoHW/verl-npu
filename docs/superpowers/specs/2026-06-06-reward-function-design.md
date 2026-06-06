# Cascade RL Reward Function Design

**Date:** 2026-06-06
**Project:** Haidass OPD on NPU — Multi-Domain RL + Distillation
**Companion to:** [2026-06-05-cascade-rl-design.md](2026-06-05-cascade-rl-design.md)
**Status:** Design Complete — Ready for Implementation

---

## 1. Overview

This spec defines the reward function architecture for the 4-stage Cascade RL pipeline (Math → Science → Code → Tool-Use). It supersedes the reward sections in the cascade RL design doc.

### Design Principles

1. **Binary task rewards (0/1)** — GRPO normalizes within uid groups; absolute scale is irrelevant.
2. **Strict per-domain format rewards (0.1 weight)** — Detect format compliance, not teach format. System prompt + task reward naturally guide format learning.
3. **Self-contained reward functions** — Each reward function owns its full computation, including external API calls (SandboxFusion). No external pre-processing dependency.
4. **Dual-verifier math grading** — String normalization + `math_verify` library (OR logic), matching DeepMath authors' own pipeline.
5. **Dynamic Sampling** — Filter zero-gradient groups at the data loading layer to maintain training signal quality.

---

## 2. Reward Architecture

### 2.1 Total Reward Formula

```
total_reward = task_reward + FORMAT_WEIGHT × format_reward

task_reward ∈ {0.0, 1.0}     # Binary: correct or not
format_reward ∈ [0.0, 1.0]   # Continuous: degree of format compliance
FORMAT_WEIGHT = 0.1           # Fixed across all domains
```

### 2.2 GRPO Integration

```
uid = "{data_source}_{index}"

For each uid group (rollout.n=16 responses to the same prompt):
  rewards = [r_1, r_2, ..., r_16]
  mean = mean(rewards)
  std = std(rewards)
  advantages = [(r_i - mean) / (std + eps) for r_i in rewards]
```

**Zero-gradient groups** (std=0): Filtered via Dynamic Sampling (§6). Remaining groups contribute zero gradient but are bounded by KL penalty (§6.3).

### 2.3 Return Value Contract

The reward function returns a `dict` for verl's `NaiveRewardManager`:

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """
    Returns:
        {
            "score": float,           # total_reward (task + format)
            "task_reward": float,     # 0.0 or 1.0
            "format_ok": bool,        # format compliance
            "domain": str,            # "math", "science", "code", "tool_use"
        }
    """
```

All dict keys become `reward_extra_info` in verl and are propagated to metrics/logging.

---

## 3. Domain-Specific Reward Functions

### 3.1 Math Reward

**Datasets:** GSM8K (integers only), DeepMath-103K (mixed formats)

#### Task Reward

```python
def _math_reward(solution_str: str, ground_truth: str) -> tuple[float, bool]:
    """
    Returns: (task_reward, format_ok)
    """
    FORMAT_WEIGHT = 0.1

    # 1. Extract predicted answer
    pred = _extract_math_answer(solution_str)
    format_ok = pred is not None

    if not format_ok:
        return 0.0, False

    # 2. Dual-verifier grading (OR logic)
    correct = (
        _math_string_verify(pred, ground_truth)
        or _math_library_verify(pred, ground_truth)
    )

    return (1.0 if correct else 0.0), format_ok
```

#### Answer Extraction

Support two extraction patterns:

```python
def _extract_math_answer(solution_str: str) -> str | None:
    """Extract answer from model response. Try \boxed{} first, then ####."""

    # Pattern 1: \boxed{...} (common for DeepMath-style responses)
    boxed_matches = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", solution_str)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Pattern 2: #### <answer> (GSM8K-style)
    hash_match = re.search(r"####\s*(.+)$", solution_str, re.MULTILINE)
    if hash_match:
        return hash_match.group(1).strip()

    return None
```

#### Dual Verifier

**Verifier 1: String normalization + numerical/symbolic comparison**

```python
def _math_string_verify(prediction: str, ground_truth: str) -> bool:
    """String normalization → exact match → numerical → symbolic."""

    pred = _normalize_math_string(prediction)
    gt = _normalize_math_string(ground_truth)

    # Exact match after normalization
    if pred == gt:
        return True

    # Case-insensitive (for Yes/No/True/False)
    if pred.lower() == gt.lower():
        return True

    # Numerical comparison
    pred_num = _try_parse_number(pred)
    gt_num = _try_parse_number(gt)
    if pred_num is not None and gt_num is not None:
        return math.isclose(pred_num, gt_num, rel_tol=1e-4)

    # Symbolic comparison (sympy, with timeout)
    try:
        pred_expr = sympy.parse_expr(_latex_to_sympy(pred))
        gt_expr = sympy.parse_expr(_latex_to_sympy(gt))
        diff = sympy.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        pass

    return False


def _normalize_math_string(s: str) -> str:
    """Normalize math answer string for comparison."""
    s = s.strip()
    s = s.replace(",", "")                          # Thousand separators
    s = s.replace("$", "").replace("\\$", "")       # Dollar signs
    s = s.replace("\\dfrac", "\\frac")              # Normalize frac variants
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")  # Delimiter commands
    s = s.replace("\\ ", " ").replace("  ", " ")   # Whitespace
    s = s.strip(".")                                 # Trailing period
    s = s.strip()
    return s
```

**Verifier 2: `math_verify` library (ANTLR-based LaTeX parsing)**

```python
def _math_library_verify(prediction: str, ground_truth: str) -> bool:
    """Use math_verify library for ANTLR-based LaTeX comparison."""
    try:
        from math_verify import parse, verify

        # DeepMath's final_answer is NOT wrapped in \boxed{} — wrap it
        gold = parse(f"\\boxed{{{ground_truth}}}")
        pred = parse(prediction)  # Auto-extracts from \boxed{} if present
        return verify(gold, pred)
    except Exception:
        return False
```

#### Answer Format Coverage (DeepMath-103K, difficulty ≥ 6)

| Format | Frequency | Example | Handled By |
|--------|-----------|---------|------------|
| Integer | ~37% | `34`, `-1` | String verify (exact match) |
| LaTeX fraction | ~15% | `\dfrac{1}{2}` | Both verifiers |
| LaTeX expression | ~15% | `\sqrt{2}`, `\dfrac{3\pi}{4}` | Library verify (ANTLR) |
| Yes/No/True/False | ~15% | `Yes`, `No` | String verify (case-insensitive) |
| Algebraic expression | ~5% | `e^{-2}`, `2^n` | Sympy symbolic |
| Decimal | ~3% | `0.5`, `3.241` | Numerical comparison |
| Equation | ~2% | `m^2 + 1 = 0` | String match (normalized) |
| Tuple/Point | ~1% | `(11, 11)` | String match |
| Other | ~7% | `\|A\|`, single letters | String match (fallback) |

#### Format Reward

```python
def _math_format_check(solution_str: str) -> float:
    """Check if response follows expected math format."""
    score = 0.0

    # Has answer delimiter (\boxed{} or ####)
    has_boxed = "\\boxed{" in solution_str
    has_hash = re.search(r"####\s*.+", solution_str, re.MULTILINE) is not None
    if has_boxed or has_hash:
        score += 0.5

    # Has reasoning steps (not just a bare answer)
    has_reasoning = len(solution_str) > 100
    if has_reasoning:
        score += 0.3

    # Has think tags (optional but encouraged)
    has_think = "</think>" in solution_str
    if has_think:
        score += 0.2

    return score
```

### 3.2 Science Reward

**Dataset:** SciKnowEval (6 task types)

#### Task Reward

```python
def _science_reward(solution_str: str, ground_truth: str, extra_info: dict) -> tuple[float, bool]:
    """
    Route by task type stored in extra_info["type"].
    Returns: (task_reward, format_ok)
    """
    task_type = extra_info.get("type", "mcq-4-choices")
    pred = _extract_science_answer(solution_str, task_type)
    format_ok = pred is not None

    if not format_ok:
        return 0.0, False

    correct = _science_match(pred, ground_truth, task_type)
    return (1.0 if correct else 0.0), format_ok


def _extract_science_answer(solution_str: str, task_type: str) -> str | None:
    """Extract answer based on task type."""

    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        # Look for single letter answer
        # Try "Answer: X" pattern first
        match = re.search(r"(?:Answer|答案)[:\s]*([A-Da-d])\b", solution_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback: last standalone letter A-D
        letters = re.findall(r"\b([A-D])\b", solution_str)
        return letters[-1] if letters else None

    elif task_type == "true_or_false":
        match = re.search(r"\b(Yes|No|True|False)\b", solution_str, re.IGNORECASE)
        return match.group(1).capitalize() if match else None

    elif task_type == "relation_extraction":
        # Expect tuple format: (entity1, relation, entity2)
        match = re.search(r"\(([^)]+)\)", solution_str)
        return match.group(0).strip() if match else None

    elif task_type == "filling":
        # Expect equation or short expression, no explanation
        lines = [l.strip() for l in solution_str.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    elif task_type == "open-ended-qa":
        # Return the full response for comparison
        return solution_str.strip()

    return None


def _science_match(pred: str, ground_truth: str, task_type: str) -> bool:
    """Match predicted answer against ground truth."""

    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        return pred.upper().strip() == ground_truth.upper().strip()

    elif task_type == "true_or_false":
        # Normalize Yes/No/True/False
        pred_norm = pred.lower().strip()
        gt_norm = ground_truth.lower().strip()
        yes_equivs = {"yes", "true"}
        no_equivs = {"no", "false"}
        if pred_norm in yes_equivs and gt_norm in yes_equivs:
            return True
        if pred_norm in no_equivs and gt_norm in no_equivs:
            return True
        return pred_norm == gt_norm

    elif task_type == "relation_extraction":
        # Compare tuples element-wise (whitespace-insensitive)
        pred_parts = [p.strip() for p in pred.strip("()").split(",")]
        gt_parts = [p.strip() for p in ground_truth.strip("()").split(",")]
        return pred_parts == gt_parts

    elif task_type == "filling":
        # Normalize chemical equations: strip whitespace, compare
        pred_norm = re.sub(r"\s+", "", pred)
        gt_norm = re.sub(r"\s+", "", ground_truth)
        return pred_norm == gt_norm

    elif task_type == "open-ended-qa":
        # Fuzzy match: check if key terms from ground_truth appear in prediction
        gt_words = set(ground_truth.lower().split())
        pred_words = set(pred.lower().split())
        overlap = len(gt_words & pred_words) / max(len(gt_words), 1)
        return overlap > 0.5

    return pred.strip().lower() == ground_truth.strip().lower()
```

#### Format Reward

```python
def _science_format_check(solution_str: str, task_type: str) -> float:
    """Check format compliance per SciKnowEval task type."""
    score = 0.0

    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        # Should contain "Answer: X" or at least a clear letter choice
        has_answer_prefix = re.search(r"(?:Answer|答案)[:\s]*[A-D]", solution_str, re.IGNORECASE) is not None
        has_letter = re.search(r"\b[A-D]\b", solution_str) is not None
        if has_answer_prefix:
            score += 0.7
        elif has_letter:
            score += 0.4
        # Brevity bonus (MCQ shouldn't have essays)
        if len(solution_str) < 500:
            score += 0.3

    elif task_type == "true_or_false":
        has_yesno = re.search(r"\b(Yes|No|True|False)\b", solution_str, re.IGNORECASE) is not None
        if has_yesno:
            score += 0.7
        if len(solution_str) < 500:
            score += 0.3

    elif task_type == "relation_extraction":
        has_tuple = re.search(r"\([^)]+,[^)]+,[^)]+\)", solution_str) is not None
        if has_tuple:
            score += 1.0

    elif task_type == "filling":
        has_equation = "=" in solution_str
        is_short = len(solution_str) < 200
        if has_equation:
            score += 0.6
        if is_short:
            score += 0.4

    elif task_type == "open-ended-qa":
        # Should be a concise paragraph, not empty or excessively long
        word_count = len(solution_str.split())
        if 10 < word_count < 200:
            score += 1.0
        elif word_count >= 5:
            score += 0.5

    return score
```

#### SciKnowEval Task Type Distribution

| Type | Count | Ground Truth Field | Answer Format |
|------|-------|--------------------|---------------|
| `mcq-4-choices` | 18,470 | `answerKey` ("A"/"B"/"C"/"D") | Single letter |
| `open-ended-qa` | 4,830 | `answer` (text) | Short paragraph |
| `true_or_false` | 3,228 | `answer` ("Yes"/"No") | Yes/No |
| `relation_extraction` | 1,364 | `answer` (tuple string) | `(entity, relation, entity)` |
| `filling` | 300 | `answer` (equation) | Balanced equation |
| `mcq-2-choices` | 200 | `answerKey` ("A"/"B") | Single letter |

### 3.3 Code Reward

**Dataset:** LiveCodeBench (release_v6)

#### Architecture: Self-Contained Sandbox Execution

The reward function calls SandboxFusion directly — no external pre-processing.

```python
def _code_reward(solution_str: str, extra_info: dict, **kwargs) -> tuple[float, bool]:
    """
    Extract code → execute via SandboxFusion → compare with test cases.
    Returns: (task_reward, format_ok)
    """
    FORMAT_WEIGHT = 0.1

    # 1. Extract code from response
    code = _extract_python_code(solution_str)
    format_ok = code is not None

    if not format_ok:
        return 0.0, False

    # 2. Get test cases from extra_info
    test_cases = _parse_test_cases(extra_info)
    if not test_cases:
        # No test cases available — can't grade, return format-only
        return 0.0, True

    # 3. Get starter code if present (LeetCode-style function completion)
    starter_code = extra_info.get("starter_code", "")
    if starter_code:
        code = starter_code + "\n" + code

    # 4. Execute via SandboxFusion
    sandbox_url = (
        kwargs.get("sandbox_fusion_url")
        or os.environ.get("SANDBOX_FUSION_URL")
        or "http://localhost:8080/run_code"
    )
    timeout = kwargs.get("sandbox_timeout", 10)

    all_passed = _execute_and_check(code, test_cases, sandbox_url, timeout)
    return (1.0 if all_passed else 0.0), True
```

#### Code Extraction

```python
def _extract_python_code(solution_str: str) -> str | None:
    """Extract Python code from response. Mirrors LiveCodeBench extraction."""
    # Look for ```python ... ``` blocks
    matches = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last code block

    # Fallback: any ``` block
    matches = re.findall(r"```\s*\n(.*?)```", solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return None
```

#### SandboxFusion Integration

```python
def _execute_and_check(
    code: str,
    test_cases: list[dict],
    sandbox_url: str,
    timeout: int = 10,
) -> bool:
    """
    Execute code against test cases via SandboxFusion API.
    Returns True only if ALL test cases pass.
    """
    import requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for tc in test_cases:
        test_input = tc["input"]
        expected_output = tc["output"]
        test_type = tc.get("testtype", "stdin")

        if test_type == "functional":
            # LeetCode-style: call function and compare return value
            payload = {
                "code": code,
                "input": test_input,
                "mode": "functional",
                "timeout": timeout,
            }
        else:
            # stdin/stdout mode
            payload = {
                "code": code,
                "input": test_input,
                "mode": "stdin",
                "timeout": timeout,
            }

        try:
            resp = session.post(sandbox_url, json=payload, timeout=timeout + 5)
            if resp.status_code != 200:
                return False

            result = resp.json()
            actual_output = result.get("output", "").strip()
            expected = expected_output.strip()

            if actual_output != expected:
                return False

        except (requests.Timeout, requests.ConnectionError):
            return False

    return True
```

#### LiveCodeBench Execution Modes

| Mode | `starter_code` present | `testtype` | How it works |
|------|----------------------|------------|--------------|
| Function completion | Yes | `functional` | Model completes `class Solution` method; harness calls `Solution().func(*args)` |
| Full program | No | `stdin` | Model writes complete program; harness pipes input to stdin, checks stdout |

#### Format Reward

```python
def _code_format_check(solution_str: str, extra_info: dict) -> float:
    """Check code response format compliance."""
    score = 0.0

    # 1. Has a code block (0.3)
    code = _extract_python_code(solution_str)
    if code is None:
        return 0.0
    score += 0.3

    # 2. Code is syntactically valid Python (0.3)
    try:
        ast.parse(code)
        score += 0.3
    except SyntaxError:
        pass

    # 3. Structural correctness (0.4)
    starter_code = extra_info.get("starter_code", "")
    if starter_code:
        # Function completion mode: should contain the function/class
        metadata = extra_info.get("metadata", "{}")
        func_name = json.loads(metadata).get("func_name", "") if isinstance(metadata, str) else ""
        has_func = func_name and func_name in code
        has_class = "class Solution" in code
        if has_func:
            score += 0.2
        if has_class:
            score += 0.2
    else:
        # stdin mode: should read input and print output
        has_input = "input()" in code or "sys.stdin" in code
        has_print = "print(" in code
        if has_input:
            score += 0.2
        if has_print:
            score += 0.2

    return score
```

### 3.4 Tool-Use Reward

**Dataset:** ToolAlpaca (ReAct format)

#### Task Reward

```python
def _tool_reward(solution_str: str, ground_truth: str, extra_info: dict) -> tuple[float, bool]:
    """
    Match predicted tool calls against expected tool calls.
    ground_truth contains the expected intermediate_steps.
    Returns: (task_reward, format_ok)
    """
    # 1. Extract tool calls from response
    predicted_calls = _extract_tool_calls(solution_str)
    format_ok = len(predicted_calls) > 0

    if not format_ok:
        return 0.0, False

    # 2. Parse expected tool calls from ground_truth
    expected_calls = _parse_expected_tool_calls(ground_truth)
    if not expected_calls:
        return 0.0, True

    # 3. Match tool calls
    score = _match_tool_calls(predicted_calls, expected_calls)
    return score, True


def _extract_tool_calls(solution_str: str) -> list[dict]:
    """Extract ReAct-format tool calls from response."""
    calls = []

    # Pattern: Action: <tool_name>\nAction Input: <json>
    pattern = r"Action:\s*(\w+)\s*\n\s*Action Input:\s*(\{[^}]*\})"
    matches = re.findall(pattern, solution_str, re.DOTALL)

    for tool_name, action_input in matches:
        try:
            params = json.loads(action_input)
            calls.append({"tool": tool_name, "params": params})
        except json.JSONDecodeError:
            calls.append({"tool": tool_name, "params": None})

    return calls


def _parse_expected_tool_calls(ground_truth: str) -> list[dict]:
    """Parse expected tool calls from ground_truth JSON."""
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if isinstance(gt, list):
            calls = []
            for step in gt:
                # Each step: [[tool_name, params_json, thought], observation]
                if isinstance(step, list) and len(step) >= 1:
                    action = step[0]
                    if isinstance(action, list) and len(action) >= 2:
                        tool_name = action[0]
                        try:
                            params = json.loads(action[1]) if isinstance(action[1], str) else action[1]
                        except json.JSONDecodeError:
                            params = {}
                        calls.append({"tool": tool_name, "params": params})
            return calls
        elif isinstance(gt, dict):
            return [gt]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _match_tool_calls(predicted: list[dict], expected: list[dict]) -> float:
    """
    Match predicted vs expected tool calls.
    Returns: 1.0 (exact match), 0.5 (partial), 0.0 (no match)
    """
    if not predicted or not expected:
        return 0.0

    # Check if same number of calls
    if len(predicted) != len(expected):
        # Partial credit: check if at least the first call matches
        if predicted[0]["tool"] == expected[0]["tool"]:
            return 0.3
        return 0.0

    # Check each call
    tool_matches = 0
    param_matches = 0

    for pred, exp in zip(predicted, expected):
        if pred["tool"] == exp["tool"]:
            tool_matches += 1
            if pred.get("params") == exp.get("params"):
                param_matches += 1

    n = len(expected)
    if tool_matches == n and param_matches == n:
        return 1.0  # Exact match
    elif tool_matches == n:
        return 0.7  # All tools correct, some params wrong
    elif tool_matches > 0:
        return 0.3  # Some tools correct
    else:
        return 0.0
```

#### Format Reward

```python
def _tool_format_check(solution_str: str) -> float:
    """Check ReAct format compliance."""
    score = 0.0

    # Thought present (0.2)
    if re.search(r"Thought:", solution_str, re.IGNORECASE):
        score += 0.2

    # Action present (0.2)
    if re.search(r"Action:\s*\w+", solution_str):
        score += 0.2

    # Action Input is valid JSON (0.2)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", solution_str, re.DOTALL)
    if input_match:
        try:
            json.loads(input_match.group(1))
            score += 0.2
        except json.JSONDecodeError:
            pass

    # Response/Final answer present (0.2)
    if re.search(r"Response:", solution_str, re.IGNORECASE):
        score += 0.2

    # Overall ReAct chain structure (0.2)
    has_chain = "Thought:" in solution_str and "Action:" in solution_str
    if has_chain:
        score += 0.2

    return score
```

---

## 4. Unified Entry Point

### 4.1 `scripts/my_rewards.py`

```python
"""
Unified reward function for Cascade RL + OPD training.

Routes to domain-specific reward logic based on data_source field.
Each domain returns: task_reward (0/1) + FORMAT_WEIGHT × format_reward

Usage in verl config:
    reward.custom_reward_function.path=scripts/my_rewards.py
    reward.custom_reward_function.name=compute_score
    reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=http://localhost:8080/run_code
"""

import ast
import json
import math
import os
import re
from typing import Any

FORMAT_WEIGHT = 0.1


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Entry point for verl NaiveRewardManager."""

    extra_info = extra_info or {}

    # Parse ground_truth if JSON string
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, dict) and "ground_truth" in gt:
                ground_truth = gt["ground_truth"]
        except (json.JSONDecodeError, AttributeError):
            pass

    # Parse extra_info if JSON string
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, AttributeError):
            extra_info = {}

    # Route by domain
    if data_source.startswith("math/"):
        task_reward, format_ok = _math_reward(solution_str, str(ground_truth))
        format_score = _math_format_check(solution_str)
        domain = "math"

    elif data_source.startswith("science/"):
        task_reward, format_ok = _science_reward(
            solution_str, str(ground_truth), extra_info
        )
        task_type = extra_info.get("type", "mcq-4-choices")
        format_score = _science_format_check(solution_str, task_type)
        domain = "science"

    elif data_source.startswith("code/"):
        task_reward, format_ok = _code_reward(
            solution_str, extra_info, **kwargs
        )
        format_score = _code_format_check(solution_str, extra_info)
        domain = "code"

    elif data_source.startswith("tool/"):
        task_reward, format_ok = _tool_reward(
            solution_str, str(ground_truth), extra_info
        )
        format_score = _tool_format_check(solution_str)
        domain = "tool_use"

    else:
        task_reward, format_ok = _default_reward(solution_str, str(ground_truth))
        format_score = 1.0 if format_ok else 0.0
        domain = "unknown"

    total = task_reward + FORMAT_WEIGHT * format_score

    return {
        "score": total,
        "task_reward": task_reward,
        "format_ok": format_ok,
        "format_score": format_score,
        "domain": domain,
    }
```

### 4.2 Config Integration

```bash
# In each stage's launch script:
reward.custom_reward_function.path=$(pwd)/scripts/my_rewards.py \
reward.custom_reward_function.name=compute_score \
reward.num_workers=4 \

# Stage 3 (Code) additionally:
reward.custom_reward_function.reward_kwargs.sandbox_fusion_url=http://localhost:8080/run_code \
reward.custom_reward_function.reward_kwargs.sandbox_timeout=10 \
```

---

## 5. Dependencies

### 5.1 Python Packages

```
# Required for math dual-verifier
math-verify[antlr4_11_0]==0.7.0
sympy>=1.12

# Required for code reward (sandbox HTTP calls)
requests>=2.28

# Already available in verl environment
# json, re, ast, math — stdlib
```

### 5.2 External Services

| Service | Domain | Deployment | Required By |
|---------|--------|------------|-------------|
| SandboxFusion | Code | `bash scripts/deploy_sandbox.sh` (Docker) | Stage 3 only |

---

## 6. Dynamic Sampling & Zero-Gradient Mitigation

### 6.1 Problem

With `rollout.n=16` and binary rewards, a uid group produces zero gradient when all 16 responses get the same reward (all correct or all wrong). This wastes compute and shrinks effective batch size.

**When it happens:**
- Training early (model bad): many groups all-wrong → zero gradient
- Training late (model good): many groups all-correct → zero gradient
- Easy/hard prompts: permanently zero gradient regardless of training stage

### 6.2 Dynamic Sampling (DAPO)

Filter zero-gradient groups at the data loading layer, over-sample to fill the batch.

```
Each training step:
1. Sample 2× batch_size prompts from the dataset
2. Generate n=16 rollouts per prompt
3. Compute rewards for all rollouts
4. Compute per-group std(rewards)
5. Discard groups where std == 0 (all same reward)
6. If remaining groups < batch_size:
   a. Sample more prompts
   b. Repeat from step 2
7. Train on the filtered batch (all groups have non-zero gradient)
```

**Implementation location:** verl's data loader / batch preparation pipeline. This is a data-layer change, not a reward function change.

**Cost analysis:**
- Generation is the bottleneck, but filtered prompts tend to be short (all-correct = easy, all-wrong = hard → short responses)
- DAPO paper reports the extra sampling cost is offset by more efficient gradient updates
- Effective batch size stays constant → no throughput loss

**verl support:** verl's `core_algos.py` has `norm_adv_by_std_in_grpo` flag. The Dynamic Sampling logic needs to be implemented in the data loading layer (custom `DataLoader` or batch filter in `ray_trainer.py`).

### 6.3 KL Penalty as Gradient Floor

Even when advantage=0 (group not filtered by Dynamic Sampling), the KL penalty provides a non-zero gradient that prevents policy collapse.

```bash
# Enable in training config:
algorithm.use_kl_in_reward=True
algorithm.kl_penalty=kl          # Standard KL divergence
algorithm.kl_ctrl.kl_coef=0.01   # Small beta — gradient floor, not dominant signal
```

**Why beta=0.01:** Large enough to prevent collapse, small enough that task rewards dominate the learning signal when they exist.

### 6.4 Monitoring

Track these metrics per stage:

```
reward/frac_zero_std          # Fraction of groups with std=0 (should be < 30%)
reward/task_reward_mean       # Average task reward (0.3-0.7 healthy range)
reward/format_reward_mean     # Format compliance rate
reward/dynamic_sampling_rate  # Fraction of prompts discarded by dynamic sampling
actor/kl_divergence           # KL to reference model
```

**Alert thresholds:**
- `frac_zero_std > 50%`: Training is wasting compute. Increase temperature or add harder prompts.
- `dynamic_sampling_rate > 70%`: Dataset difficulty mismatch. Most prompts are too easy or too hard.
- `kl_divergence > 0.5`: Policy has drifted far from reference. Consider lowering LR.

---

## 7. Data Schema Requirements

### 7.1 `extra_info` Fields Per Domain

The `convert_datasets.py` script must populate these fields in the parquet `extra_info` column:

**Math:**
```json
{
    "difficulty": 6.5,
    "topic": "Mathematics -> Algebra -> ...",
    "split": "train"
}
```

**Science:**
```json
{
    "type": "mcq-4-choices",
    "domain": "Biology",
    "level": "L3",
    "choices": {"text": ["...", "..."], "label": ["A", "B", "C", "D"]}
}
```

**Code:**
```json
{
    "starter_code": "class Solution:\n    def func(self, nums):\n        ",
    "test_cases": [
        {"input": "[[7,2,1],[6,4,2]]", "output": "15", "testtype": "functional"}
    ],
    "metadata": {"func_name": "matrixSum"},
    "platform": "leetcode",
    "difficulty": "medium"
}
```

**Tool-Use:**
```json
{
    "api_name": "Httpbin",
    "tool_names": ["sendHttpRequest", "getClientRequestData"],
    "nl_documentation": "sendHttpRequest: Send an HTTP request..."
}
```

### 7.2 `reward_model.ground_truth` Format Per Domain

| Domain | ground_truth content | Example |
|--------|---------------------|---------|
| Math | Numeric answer or LaTeX expression | `"72"`, `"\dfrac{1}{2}"`, `"\sqrt{2}"` |
| Science | Answer key or answer text | `"B"`, `"Yes"`, `"(drug, interaction, drug)"` |
| Code | JSON string of expected test case outputs | `"[{\"input\": \"...\", \"output\": \"15\"}]"` |
| Tool-Use | JSON string of expected intermediate_steps | `"[[[\"tool\", \"{params}\"], \"obs\"]]"` |

---

## 8. Comparison with Previous Design

| Aspect | Previous (cascade-rl-design §5) | This Spec |
|--------|-------------------------------|-----------|
| Math answer extraction | `####` only | `\boxed{}` + `####` (dual pattern) |
| Math verification | String normalize + exact match | Dual verifier (string + math_verify library) |
| DeepMath coverage | ~37% (integers only) | ~90%+ (fractions, expressions, symbols) |
| Science answer extraction | `Answer:` or last line | Type-aware extraction (MCQ/T-F/relation/filling/open) |
| Science format check | `len > 20` (trivially true) | Per-type format validation |
| Code execution | External pre-processing via `extra_info["execution_result"]` | Self-contained: reward function calls SandboxFusion directly |
| Tool-Use matching | Exact JSON match or name-only | Graded: exact(1.0) → tools+params(0.7) → tools only(0.3) |
| GRPO zero-gradient | Not addressed | Dynamic Sampling + KL penalty (beta=0.01) |
| Return value | `float` | `dict` with score + metrics |
| Dependencies | None | `math-verify`, `sympy`, `requests` |

---

## 9. Implementation Checklist

| # | Task | File | Priority |
|---|------|------|----------|
| 1 | Implement dual-verifier math reward | `scripts/my_rewards.py` | P0 |
| 2 | Implement type-aware science reward | `scripts/my_rewards.py` | P0 |
| 3 | Implement self-contained code reward with SandboxFusion | `scripts/my_rewards.py` | P0 |
| 4 | Implement ReAct-format tool reward | `scripts/my_rewards.py` | P0 |
| 5 | Update `convert_datasets.py` to populate `extra_info` per §7.1 | `scripts/convert_datasets.py` | P0 |
| 6 | Add `reward.custom_reward_function` config to stage scripts | `scripts/run_cascade_stage*.sh` | P0 |
| 7 | Install `math-verify` and `sympy` in container | `docker/` | P0 |
| 8 | Implement Dynamic Sampling in data loader | `patches/ray_trainer.py` or custom loader | P1 |
| 9 | Enable KL penalty (`use_kl_in_reward=True`, `kl_coef=0.01`) | Stage scripts | P1 |
| 10 | Add monitoring metrics (`frac_zero_std`, `dynamic_sampling_rate`) | `patches/ray_trainer.py` | P1 |
| 11 | Unit test each domain reward function | `tests/test_rewards.py` | P1 |
| 12 | Integration test: reward function with actual parquet data | `tests/test_rewards_integration.py` | P2 |

---

## 10. Decision Log

| Decision | Rationale |
|----------|-----------|
| Binary task rewards (0/1) | GRPO auto-normalizes within uid groups; absolute scale irrelevant. Binary is simplest and most robust. |
| FORMAT_WEIGHT=0.1 (fixed) | Format reward is a guardrail, not a teaching signal. System prompt + task reward handle format learning. |
| Dual math verifier (OR logic) | DeepMath authors use this exact pipeline. String normalize catches 37% integer cases fast; math_verify catches LaTeX/symbolic cases. OR logic maximizes recall. |
| `\boxed{}` extraction priority over `####` | DeepMath-style responses use `\boxed{}`; GSM8K uses `####`. Try `\boxed{}` first since it's more specific. |
| Type-aware science extraction | SciKnowEval has 6 task types with different answer formats. A single extraction pattern would fail on 4 of them. |
| Self-contained code reward | Eliminates coupling between reward function and external pre-processing. SandboxFusion calls from within the reward function are safe (ThreadPoolExecutor). |
| Graded tool matching (1.0/0.7/0.3/0.0) | Binary exact-match is too harsh for multi-step tool chains. Partial credit for correct tool names encourages exploration. |
| Dynamic Sampling over Replay Buffer | DAPO's Dynamic Sampling is simpler, well-studied, and doesn't require buffer management. Replay Buffer is TRL-experimental. |
| KL penalty beta=0.01 | Provides gradient floor without dominating task reward signal. Standard value from GRPO/DAPO literature. |
| Dict return value | Enables per-domain metric tracking (task_reward, format_ok, domain) in verl's logging infrastructure. |

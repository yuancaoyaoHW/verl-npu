"""
Unified reward function for Cascade RL + OPD training.

Routes to domain-specific reward logic based on data_source field.
Each domain returns: task_reward (0/1) + FORMAT_WEIGHT * format_reward

verl config:
    custom_reward_function.path=scripts/my_rewards.py
    custom_reward_function.name=compute_score
    custom_reward_function.reward_kwargs.sandbox_fusion_url=http://localhost:8080/run_code
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
    extra_info = extra_info or {}

    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, dict) and "ground_truth" in gt:
                ground_truth = gt["ground_truth"]
        except (json.JSONDecodeError, AttributeError):
            pass

    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, AttributeError):
            extra_info = {}

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


# ============================================================
# Math
# ============================================================


def _math_reward(solution_str: str, ground_truth: str) -> tuple[float, bool]:
    pred = _extract_math_answer(solution_str)
    format_ok = pred is not None

    if not format_ok:
        return 0.0, False

    correct = (
        _math_string_verify(pred, ground_truth)
        or _math_library_verify(pred, ground_truth)
    )

    return (1.0 if correct else 0.0), format_ok


def _extract_math_answer(solution_str: str) -> str | None:
    # Pattern 1: \boxed{...} (handles up to 2 levels of nested braces)
    boxed_matches = re.findall(
        r"\\boxed\{((?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        solution_str,
    )
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Pattern 2: #### <answer> (GSM8K-style)
    hash_match = re.search(r"####\s*(.+)$", solution_str, re.MULTILINE)
    if hash_match:
        return hash_match.group(1).strip()

    return None


def _math_string_verify(prediction: str, ground_truth: str) -> bool:
    pred = _normalize_math_string(prediction)
    gt = _normalize_math_string(ground_truth)

    if pred == gt:
        return True

    if pred.lower() == gt.lower():
        return True

    pred_num = _try_parse_number(pred)
    gt_num = _try_parse_number(gt)
    if pred_num is not None and gt_num is not None:
        return math.isclose(pred_num, gt_num, rel_tol=1e-4)

    try:
        import sympy
        pred_expr = sympy.parse_expr(_latex_to_sympy(pred))
        gt_expr = sympy.parse_expr(_latex_to_sympy(gt))
        diff = sympy.simplify(pred_expr - gt_expr)
        return diff == 0
    except Exception:
        pass

    return False


def _normalize_math_string(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\ ", " ").replace("  ", " ")
    s = s.strip(".")
    s = s.strip()
    return s


def _try_parse_number(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _latex_to_sympy(s: str) -> str:
    s = s.replace("\\frac", "/")
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("^", "**")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s


def _math_library_verify(prediction: str, ground_truth: str) -> bool:
    try:
        from math_verify import parse, verify
        gold = parse(f"\\boxed{{{ground_truth}}}")
        if "\\boxed{" not in prediction:
            prediction = f"\\boxed{{{prediction}}}"
        pred = parse(prediction)
        return verify(gold, pred)
    except Exception:
        return False


def _math_format_check(solution_str: str) -> float:
    score = 0.0

    has_boxed = "\\boxed{" in solution_str
    has_hash = re.search(r"####\s*.+", solution_str, re.MULTILINE) is not None
    if has_boxed or has_hash:
        score += 0.5

    if len(solution_str) > 100:
        score += 0.3

    if "</think>" in solution_str:
        score += 0.2

    return score


# ============================================================
# Science
# ============================================================


def _science_reward(
    solution_str: str, ground_truth: str, extra_info: dict
) -> tuple[float, bool]:
    task_type = extra_info.get("type", "mcq-4-choices")
    pred = _extract_science_answer(solution_str, task_type)
    format_ok = pred is not None

    if not format_ok:
        return 0.0, False

    correct = _science_match(pred, ground_truth, task_type)
    return (1.0 if correct else 0.0), format_ok


def _extract_science_answer(solution_str: str, task_type: str) -> str | None:
    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        match = re.search(
            r"(?:Answer|答案)[:\s]*([A-Da-d])\b", solution_str, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        letters = re.findall(r"\b([A-D])\b", solution_str)
        return letters[-1] if letters else None

    elif task_type == "true_or_false":
        match = re.search(r"\b(Yes|No|True|False)\b", solution_str, re.IGNORECASE)
        return match.group(1).capitalize() if match else None

    elif task_type == "relation_extraction":
        match = re.search(r"\(([^)]+)\)", solution_str)
        return match.group(0).strip() if match else None

    elif task_type == "filling":
        lines = [l.strip() for l in solution_str.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    elif task_type == "open-ended-qa":
        return solution_str.strip()

    return None


def _science_match(pred: str, ground_truth: str, task_type: str) -> bool:
    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        return pred.upper().strip() == ground_truth.upper().strip()

    elif task_type == "true_or_false":
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
        pred_parts = [p.strip() for p in pred.strip("()").split(",")]
        gt_parts = [p.strip() for p in ground_truth.strip("()").split(",")]
        return pred_parts == gt_parts

    elif task_type == "filling":
        pred_norm = re.sub(r"\s+", "", pred)
        gt_norm = re.sub(r"\s+", "", ground_truth)
        return pred_norm == gt_norm

    elif task_type == "open-ended-qa":
        gt_words = set(ground_truth.lower().split())
        pred_words = set(pred.lower().split())
        overlap = len(gt_words & pred_words) / max(len(gt_words), 1)
        return overlap > 0.5

    return pred.strip().lower() == ground_truth.strip().lower()


def _science_format_check(solution_str: str, task_type: str) -> float:
    score = 0.0

    if task_type in ("mcq-4-choices", "mcq-2-choices"):
        has_answer_prefix = (
            re.search(r"(?:Answer|答案)[:\s]*[A-D]", solution_str, re.IGNORECASE)
            is not None
        )
        has_letter = re.search(r"\b[A-D]\b", solution_str) is not None
        if has_answer_prefix:
            score += 0.7
        elif has_letter:
            score += 0.4
        if len(solution_str) < 500:
            score += 0.3

    elif task_type == "true_or_false":
        has_yesno = (
            re.search(r"\b(Yes|No|True|False)\b", solution_str, re.IGNORECASE)
            is not None
        )
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
        word_count = len(solution_str.split())
        if 10 < word_count < 200:
            score += 1.0
        elif word_count >= 5:
            score += 0.5

    return score


# ============================================================
# Code
# ============================================================


def _code_reward(
    solution_str: str, extra_info: dict, **kwargs
) -> tuple[float, bool]:
    code = _extract_python_code(solution_str)
    format_ok = code is not None

    if not format_ok:
        return 0.0, False

    test_cases = _parse_test_cases(extra_info)
    if not test_cases:
        return 0.0, True

    starter_code = extra_info.get("starter_code", "")
    if starter_code:
        code = starter_code + "\n" + code

    sandbox_url = (
        kwargs.get("sandbox_fusion_url")
        or os.environ.get("SANDBOX_FUSION_URL")
        or "http://localhost:8080/run_code"
    )
    timeout = kwargs.get("sandbox_timeout", 10)

    all_passed = _execute_and_check(code, test_cases, sandbox_url, timeout)
    return (1.0 if all_passed else 0.0), True


def _extract_python_code(solution_str: str) -> str | None:
    matches = re.findall(
        r"```(?:python|Python)?\s*\n(.*?)```", solution_str, re.DOTALL
    )
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"```\s*\n(.*?)```", solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return None


def _parse_test_cases(extra_info: dict) -> list[dict] | None:
    test_cases = extra_info.get("test_cases")
    if test_cases:
        if isinstance(test_cases, str):
            try:
                return json.loads(test_cases)
            except json.JSONDecodeError:
                return None
        return test_cases
    return None


def _execute_and_check(
    code: str,
    test_cases: list[dict],
    sandbox_url: str,
    timeout: int = 10,
) -> bool:
    import requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for tc in test_cases:
        test_input = tc.get("input", "")
        expected_output = tc.get("output", "")
        test_type = tc.get("testtype", "stdin")

        if test_type == "functional":
            payload = {
                "code": code,
                "input": test_input,
                "mode": "functional",
                "timeout": timeout,
            }
        else:
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


def _code_format_check(solution_str: str, extra_info: dict) -> float:
    score = 0.0

    code = _extract_python_code(solution_str)
    if code is None:
        return 0.0
    score += 0.3

    try:
        ast.parse(code)
        score += 0.3
    except SyntaxError:
        pass

    starter_code = extra_info.get("starter_code", "")
    if starter_code:
        metadata = extra_info.get("metadata", "{}")
        func_name = (
            json.loads(metadata).get("func_name", "")
            if isinstance(metadata, str)
            else ""
        )
        has_func = func_name and func_name in code
        has_class = "class Solution" in code
        if has_func:
            score += 0.2
        if has_class:
            score += 0.2
    else:
        has_input = "input()" in code or "sys.stdin" in code
        has_print = "print(" in code
        if has_input:
            score += 0.2
        if has_print:
            score += 0.2

    return score


# ============================================================
# Tool-Use
# ============================================================


def _tool_reward(
    solution_str: str, ground_truth: str, extra_info: dict
) -> tuple[float, bool]:
    predicted_calls = _extract_tool_calls(solution_str)
    format_ok = len(predicted_calls) > 0

    if not format_ok:
        return 0.0, False

    expected_calls = _parse_expected_tool_calls(ground_truth)
    if not expected_calls:
        return 0.0, True

    score = _match_tool_calls(predicted_calls, expected_calls)
    return score, True


def _extract_tool_calls(solution_str: str) -> list[dict]:
    calls = []
    # ReAct pattern: Action: <tool_name>\nAction Input: <json>
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
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if isinstance(gt, list):
            calls = []
            for step in gt:
                if isinstance(step, list) and len(step) >= 1:
                    action = step[0]
                    if isinstance(action, list) and len(action) >= 2:
                        tool_name = action[0]
                        try:
                            params = (
                                json.loads(action[1])
                                if isinstance(action[1], str)
                                else action[1]
                            )
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
    if not predicted or not expected:
        return 0.0

    if len(predicted) != len(expected):
        if predicted[0]["tool"] == expected[0]["tool"]:
            return 0.3
        return 0.0

    tool_matches = 0
    param_matches = 0

    for pred, exp in zip(predicted, expected):
        if pred["tool"] == exp["tool"]:
            tool_matches += 1
            if pred.get("params") == exp.get("params"):
                param_matches += 1

    n = len(expected)
    if tool_matches == n and param_matches == n:
        return 1.0
    elif tool_matches == n:
        return 0.7
    elif tool_matches > 0:
        return 0.3
    else:
        return 0.0


def _tool_format_check(solution_str: str) -> float:
    score = 0.0

    if re.search(r"Thought:", solution_str, re.IGNORECASE):
        score += 0.2

    if re.search(r"Action:\s*\w+", solution_str):
        score += 0.2

    input_match = re.search(
        r"Action Input:\s*(\{.*?\})", solution_str, re.DOTALL
    )
    if input_match:
        try:
            json.loads(input_match.group(1))
            score += 0.2
        except json.JSONDecodeError:
            pass

    if re.search(r"Response:", solution_str, re.IGNORECASE):
        score += 0.2

    has_chain = "Thought:" in solution_str and "Action:" in solution_str
    if has_chain:
        score += 0.2

    return score


# ============================================================
# Default
# ============================================================


def _default_reward(solution_str: str, ground_truth: str) -> tuple[float, bool]:
    lines = [line.strip() for line in solution_str.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0, False

    predicted = lines[-1].lower().strip()
    expected = ground_truth.lower().strip()

    if predicted == expected:
        return 1.0, True

    return 0.0, len(solution_str.strip()) > 20


# ============================================================
# Tests
# ============================================================

if __name__ == "__main__":
    print("=== Math Reward ===")

    r = compute_score("math/gsm8k", "Step by step...\n#### 42", "42")
    print(f"  GSM8K correct:       {r}")
    assert r["task_reward"] == 1.0 and r["domain"] == "math"

    r = compute_score("math/gsm8k", "Step by step...\n#### 43", "42")
    print(f"  GSM8K wrong:         {r}")
    assert r["task_reward"] == 0.0

    r = compute_score("math/gsm8k", "No answer here.", "42")
    print(f"  GSM8K no format:     {r}")
    assert r["task_reward"] == 0.0 and r["format_ok"] is False

    r = compute_score("math/deepmath", "So $\\boxed{\\frac{1}{2}}$", "\\frac{1}{2}")
    print(f"  DeepMath boxed:      {r}")
    assert r["task_reward"] == 1.0 and r["format_ok"] is True

    r = compute_score("math/gsm8k", "#### -42", "-42")
    print(f"  Negative:            {r}")
    assert r["task_reward"] == 1.0

    r = compute_score("math/gsm8k", "#### 3.14", "3.14")
    print(f"  Decimal:             {r}")
    assert r["task_reward"] == 1.0

    r = compute_score("math/gsm8k", "#### $1,234", "1234")
    print(f"  Dollar+comma:        {r}")
    assert r["task_reward"] == 1.0

    r = compute_score("math/gsm8k", "#### 42", '{"ground_truth": "42", "style": "rule"}')
    print(f"  JSON ground_truth:   {r}")
    assert r["task_reward"] == 1.0

    print("\n=== Science Reward ===")

    r = compute_score(
        "science/sciknow", "Answer: B", "B",
        extra_info={"type": "mcq-4-choices"},
    )
    print(f"  MCQ correct:         {r}")
    assert r["task_reward"] == 1.0

    r = compute_score(
        "science/sciknow", "The answer is C", "B",
        extra_info={"type": "mcq-4-choices"},
    )
    print(f"  MCQ wrong:           {r}")
    assert r["task_reward"] == 0.0

    r = compute_score(
        "science/sciknow", "Yes, this is correct.", "Yes",
        extra_info={"type": "true_or_false"},
    )
    print(f"  T/F Yes:             {r}")
    assert r["task_reward"] == 1.0

    r = compute_score(
        "science/sciknow", "True", "Yes",
        extra_info={"type": "true_or_false"},
    )
    print(f"  T/F True==Yes:       {r}")
    assert r["task_reward"] == 1.0

    r = compute_score(
        "science/sciknow", "The relation is (drug, interacts, protein)",
        "(drug, interacts, protein)",
        extra_info={"type": "relation_extraction"},
    )
    print(f"  Relation:            {r}")
    assert r["task_reward"] == 1.0

    print("\n=== Code Reward ===")

    r = compute_score("code/livecode", "```python\nprint(2+2)\n```", "")
    print(f"  Code format (no sb): {r}")
    assert r["format_ok"] is True and r["domain"] == "code"

    print("\n=== Tool Reward ===")

    tool_response = (
        "Thought: I need to search.\n"
        'Action: search\n'
        'Action Input: {"query": "weather"}\n'
        "Response: Found results."
    )
    tool_gt = json.dumps([
        [["search", '{"query": "weather"}'], "observation"]
    ])
    r = compute_score("tool/toolalpaca", tool_response, tool_gt)
    print(f"  ReAct exact:         {r}")
    assert r["task_reward"] == 1.0

    tool_gt_wrong = json.dumps([
        [["search", '{"query": "news"}'], "observation"]
    ])
    r = compute_score("tool/toolalpaca", tool_response, tool_gt_wrong)
    print(f"  ReAct tool match:    {r}")
    assert r["task_reward"] == 0.7

    r = compute_score("tool/toolalpaca", "No tool calls here.", "[]")
    print(f"  No tool call:        {r}")
    assert r["task_reward"] == 0.0 and r["format_ok"] is False

    print("\n=== Format Scores ===")

    math_fmt = _math_format_check("#### 42\nShort.")
    print(f"  Math (short, ####):  {math_fmt}")

    math_fmt2 = _math_format_check(
        "\\boxed{42}\n" + "x" * 100 + "\n</think>"
    )
    print(f"  Math (boxed+think):  {math_fmt2}")
    assert math_fmt2 == 1.0

    sci_fmt = _science_format_check("Answer: B", "mcq-4-choices")
    print(f"  Science MCQ:         {sci_fmt}")

    tool_fmt = _tool_format_check(tool_response)
    print(f"  Tool ReAct:          {tool_fmt}")

    print("\n✅ All tests passed!")

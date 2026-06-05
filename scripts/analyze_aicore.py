#!/usr/bin/env python3
import sys
import re
import argparse

def parse_watch_log(log_file):
    data = {}
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    npu_id = int(parts[0])
                    aicore = int(parts[4])
                    if npu_id not in data:
                        data[npu_id] = []
                    data[npu_id].append(aicore)
                except (ValueError, IndexError):
                    continue
    return data

def compute_stats(values):
    if not values:
        return {"peak": 0, "avg": 0, "p50": 0, "p95": 0, "min": 0, "count": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "peak": max(values),
        "avg": sum(values) / n,
        "min": min(values),
        "p50": sorted_v[n // 2],
        "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
        "count": n,
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze AICore utilization trace log")
    parser.add_argument("log_file", help="Path to npu-smi trace log")
    parser.add_argument("--student-only", action="store_true", help="Only show student NPUs (0-3)")
    parser.add_argument("--teacher-only", action="store_true", help="Only show teacher NPUs (4-7)")
    args = parser.parse_args()

    data = parse_watch_log(args.log_file)
    if not data:
        print("No data found in log file")
        sys.exit(1)

    student_all = []
    teacher_all = []

    print(f"{'NPU':>4} {'Peak':>6} {'Avg':>6} {'Min':>6} {'P50':>6} {'P95':>6} {'Samples':>8}")
    print("-" * 52)

    for npu_id in sorted(data.keys()):
        if args.student_only and npu_id > 3:
            continue
        if args.teacher_only and npu_id < 4:
            continue
        values = data[npu_id]
        stats = compute_stats(values)
        print(f"{npu_id:>4} {stats['peak']:>5.0f}% {stats['avg']:>5.1f}% {stats['min']:>5.0f}% {stats['p50']:>5.0f}% {stats['p95']:>5.0f}% {stats['count']:>8}")
        if npu_id <= 3:
            student_all.extend(values)
        else:
            teacher_all.extend(values)

    print("-" * 52)
    if student_all:
        s = compute_stats(student_all)
        print(f"{'Student (0-3)':>14} {s['peak']:>5.0f}% {s['avg']:>5.1f}% {s['min']:>5.0f}% {s['p50']:>5.0f}% {s['p95']:>5.0f}% {s['count']:>8}")
    if teacher_all:
        s = compute_stats(teacher_all)
        print(f"{'Teacher (4-7)':>14} {s['peak']:>5.0f}% {s['avg']:>5.1f}% {s['min']:>5.0f}% {s['p50']:>5.0f}% {s['p95']:>5.0f}% {s['count']:>8}")
    if student_all and teacher_all:
        all_vals = student_all + teacher_all
        s = compute_stats(all_vals)
        print(f"{'All (0-7)':>14} {s['peak']:>5.0f}% {s['avg']:>5.1f}% {s['min']:>5.0f}% {s['p50']:>5.0f}% {s['p95']:>5.0f}% {s['count']:>8}")

if __name__ == "__main__":
    main()

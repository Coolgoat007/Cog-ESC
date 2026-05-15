#!/usr/bin/env python3
"""
监控生成和评估进度
"""

import os
import json
import time
from datetime import datetime

def check_generation_progress():
    """检查生成进度"""
    print("\n" + "="*60)
    print("Generation Progress")
    print("="*60)

    files = {
        "GPT-4o": "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o/generated_predictions.jsonl",
        "GPT-4o-mini": "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o_mini/generated_predictions.jsonl",
        "Claude": "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/claude_sonnet/generated_predictions.jsonl",
    }

    total_expected = 682

    for name, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                count = sum(1 for _ in f)
            percentage = (count / total_expected) * 100
            status = "✓ Complete" if count == total_expected else f"⏳ {percentage:.1f}%"
            print(f"{name:<15} {count:>3}/{total_expected} {status}")
        else:
            print(f"{name:<15}   0/{total_expected} ✗ Not started")

def check_evaluation_progress():
    """检查评估进度"""
    print("\n" + "="*60)
    print("Evaluation Progress")
    print("="*60)

    evaluations = {
        "Sequence Eval": "sequence_eval_with_baselines_results/all_sequence_summary.csv",
        "5-Dim Eval": "eval_with_baselines_results/",
        "Pairwise Eval": "pairwise_with_baselines_results/",
    }

    for name, path in evaluations.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"{name:<20} ✓ Complete")
            else:
                files = os.listdir(path)
                print(f"{name:<20} ✓ Complete ({len(files)} files)")
        else:
            print(f"{name:<20} ✗ Not started")

def check_process_running():
    """检查进程是否在运行"""
    print("\n" + "="*60)
    print("Running Processes")
    print("="*60)

    import subprocess

    processes = [
        "generate_strong_baseline_responses.py",
        "sequence_eval_llm_v2_with_baselines.py",
        "llm_eval_with_baselines.py",
        "pairwise_eval_v2_with_baselines.py",
    ]

    for proc in processes:
        result = subprocess.run(
            f"ps aux | grep {proc} | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"⏳ {proc} is running")
        else:
            print(f"✗ {proc} is not running")

def main():
    print("\n" + "="*60)
    print("Progress Monitor")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    check_process_running()
    check_generation_progress()
    check_evaluation_progress()

    print("\n" + "="*60)
    print("Refresh this script to see updated progress")
    print("="*60)

if __name__ == "__main__":
    main()

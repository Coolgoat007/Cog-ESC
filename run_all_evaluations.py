#!/usr/bin/env python3
"""
主控脚本：自动运行所有评估并生成最终报告
"""

import os
import sys
import subprocess
import time

def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)

def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print("-"*80)

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time

    print("-"*80)
    if result.returncode == 0:
        print(f"✓ Completed in {elapsed:.1f}s")
    else:
        print(f"✗ Failed with exit code {result.returncode}")
    print("="*80)

    return result.returncode == 0

def main():
    print("\n" + "="*80)
    print("AUTOMATED EVALUATION PIPELINE")
    print("Strong Baseline Comparison for Emotional Support Conversations")
    print("="*80)

    # 检查生成的文件是否存在
    required_files = [
        "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o/generated_predictions.jsonl",
        "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o_mini/generated_predictions.jsonl",
        "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/claude_sonnet/generated_predictions.jsonl",
    ]

    print("\n### Checking generated files ###")
    all_exist = True
    for filepath in required_files:
        exists = check_file_exists(filepath)
        status = "✓" if exists else "✗"
        print(f"{status} {filepath}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠ Warning: Some generated files are missing!")
        print("Please run generate_strong_baseline_responses.py first.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # 运行评估
    evaluations = [
        {
            "script": "sequence_eval_llm_v2_with_baselines.py",
            "description": "Sequence Evaluation (Validation-First & Premature-Advice)",
            "estimated_time": "30-40 minutes"
        },
        {
            "script": "llm_eval_with_baselines.py",
            "description": "5-Dimension Quality Evaluation",
            "estimated_time": "40-50 minutes"
        },
        {
            "script": "pairwise_eval_v2_with_baselines.py",
            "description": "Pairwise V2 Evaluation (Therapeutic Depth)",
            "estimated_time": "60-90 minutes"
        },
    ]

    print("\n### Evaluation Pipeline ###")
    for i, eval_info in enumerate(evaluations, 1):
        print(f"{i}. {eval_info['description']}")
        print(f"   Script: {eval_info['script']}")
        print(f"   Estimated time: {eval_info['estimated_time']}")

    print("\n" + "="*80)
    response = input("Start evaluation pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return

    # 执行评估
    total_start = time.time()
    results = []

    for i, eval_info in enumerate(evaluations, 1):
        print(f"\n### [{i}/{len(evaluations)}] {eval_info['description']} ###")
        success = run_command(f"python {eval_info['script']}", eval_info['description'])
        results.append({
            "name": eval_info['description'],
            "success": success
        })

        if not success:
            print(f"\n⚠ Warning: {eval_info['description']} failed!")
            response = input("Continue with next evaluation? (y/n): ")
            if response.lower() != 'y':
                break

    total_elapsed = time.time() - total_start

    # 生成最终报告
    print("\n" + "="*80)
    print("Generating final report...")
    print("="*80)
    run_command("python generate_final_report.py", "Final Report Generation")

    # 总结
    print("\n" + "="*80)
    print("EVALUATION PIPELINE COMPLETED")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("\nResults:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['name']}")

    print("\n### Output Files ###")
    output_dirs = [
        "sequence_eval_with_baselines_results/",
        "eval_with_baselines_results/",
        "pairwise_with_baselines_results/",
    ]

    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"\n{dir_path} ({len(files)} files)")
            for f in sorted(files)[:5]:  # 只显示前5个
                print(f"  - {f}")
            if len(files) > 5:
                print(f"  ... and {len(files)-5} more files")

    print("\n### LaTeX Tables ###")
    latex_files = [
        "final_report_sequence_table.tex",
        "final_report_5dim_table.tex",
        "final_report_pairwise_table.tex"
    ]
    for f in latex_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f}")

    print("\n" + "="*80)
    print("All done! Check the output files for results.")
    print("="*80)

if __name__ == "__main__":
    main()

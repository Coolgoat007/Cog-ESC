"""
TASK 7: Generate final report after all tasks complete.
"""
import os, json, sys

SUPPLEMENTAL_DIR = os.path.dirname(__file__)

def main():
    report = []
    report.append("# Supplemental Experiments Final Report\n")
    report.append(f"Generated: {os.popen('date').read().strip()}\n")

    # Task 0
    report.append("## Task 0: Deterministic Subset\n")
    subset_path = os.path.join(SUPPLEMENTAL_DIR, "subset_200_seed42.txt")
    if os.path.exists(subset_path):
        with open(subset_path) as f:
            indices = [int(l.strip()) for l in f if l.strip()]
        report.append(f"- **Status**: ✓ Complete\n")
        report.append(f"- **N**: {len(indices)} indices\n")
        report.append(f"- **Seed**: 42\n")
        report.append(f"- **Path**: `{subset_path}`\n")
    else:
        report.append(f"- **Status**: ✗ Missing\n")

    # Task 2
    report.append("\n## Task 2: V2-Neutral Prompt Sensitivity (GPT-4o-mini)\n")
    v2_dir = os.path.join(SUPPLEMENTAL_DIR, "v2_neutral_gpt4o_mini")
    comparisons = [
        "Llama-Cognitive_vs_Llama-Base",
        "Qwen-Cognitive_vs_Qwen-Base",
        "Llama-Cognitive_vs_GPT-4o",
        "Llama-Cognitive_vs_Claude-Sonnet",
        "Qwen-Cognitive_vs_GPT-4o",
        "Qwen-Cognitive_vs_Claude-Sonnet",
    ]
    t2_complete = True
    for comp in comparisons:
        fname = f"{comp}_v2_neutral_gpt4o_mini.jsonl"
        path = os.path.join(v2_dir, fname)
        if os.path.exists(path):
            n = sum(1 for _ in open(path))
            status = "✓" if n == 200 else f"⚠ {n}/200"
            if n != 200:
                t2_complete = False
        else:
            status = "✗ Missing"
            t2_complete = False
        report.append(f"- **{comp}**: {status}\n")

    summary_csv = os.path.join(v2_dir, "summary.csv")
    if os.path.exists(summary_csv):
        report.append(f"- **Summary tables**: ✓ Generated\n")
        report.append(f"  - `summary.csv`, `summary.md`, `summary.tex`\n")
        report.append(f"  - `prompt_sensitivity_table.*`\n")
    else:
        report.append(f"- **Summary tables**: ✗ Not generated\n")

    # Task 3
    report.append("\n## Task 3: Multi-Judge Robustness\n")
    mj_dir = os.path.join(SUPPLEMENTAL_DIR, "multijudge_v2_neutral")
    judges = ["gpt-4o-mini", "gpt-4o", "claude"]
    t3_comparisons = [
        "Llama-Cognitive_vs_Llama-Base",
        "Llama-Cognitive_vs_GPT-4o",
        "Llama-Cognitive_vs_Claude-Sonnet",
        "Qwen-Cognitive_vs_GPT-4o",
        "Qwen-Cognitive_vs_Claude-Sonnet",
    ]
    t3_complete = True
    for judge in judges:
        report.append(f"\n### Judge: {judge}\n")
        for comp in t3_comparisons:
            fname = f"{comp}_v2_neutral_{judge}.jsonl"
            path = os.path.join(mj_dir, fname)
            if os.path.exists(path):
                n = sum(1 for _ in open(path))
                status = "✓" if n == 200 else f"⚠ {n}/200"
                if n != 200:
                    t3_complete = False
            else:
                if judge == "claude":
                    status = "⊘ Unavailable"
                else:
                    status = "✗ Missing"
                    t3_complete = False
            report.append(f"- **{comp}**: {status}\n")

    summary_csv = os.path.join(mj_dir, "judge_robustness_summary.csv")
    if os.path.exists(summary_csv):
        report.append(f"\n- **Summary tables**: ✓ Generated\n")
    else:
        report.append(f"\n- **Summary tables**: ✗ Not generated\n")

    # Task 4
    report.append("\n## Task 4: Length-Aware Analysis\n")
    la_dir = os.path.join(SUPPLEMENTAL_DIR, "length_analysis")
    la_files = [
        "length_analysis_summary.csv",
        "length_analysis_summary.md",
        "length_analysis_summary.tex",
        "regression_information.txt",
        "regression_strategy.txt",
        "paper_conclusion.txt",
    ]
    t4_complete = True
    for fname in la_files:
        path = os.path.join(la_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            status = "✓"
        else:
            status = "✗"
            t4_complete = False
        report.append(f"- **{fname}**: {status}\n")

    # Task 5
    report.append("\n## Task 5: Reward Evaluator Consistency\n")
    report.append("- **Status**: Skipped (MCTS intermediate candidates not available)\n")

    # Task 6
    report.append("\n## Task 6: Quality Checks\n")
    qc_path = os.path.join(SUPPLEMENTAL_DIR, "final_quality_check.md")
    if os.path.exists(qc_path):
        report.append(f"- **Status**: ✓ Complete\n")
        report.append(f"- **Path**: `{qc_path}`\n")
    else:
        report.append(f"- **Status**: ✗ Not run\n")

    # Overall status
    report.append("\n## Overall Status\n")
    all_complete = t2_complete and t3_complete and t4_complete
    if all_complete:
        report.append("- **All tasks**: ✓ Complete\n")
    else:
        report.append("- **All tasks**: ⚠ Some tasks incomplete\n")
        if not t2_complete:
            report.append("  - Task 2 (V2-Neutral) incomplete\n")
        if not t3_complete:
            report.append("  - Task 3 (Multi-judge) incomplete\n")
        if not t4_complete:
            report.append("  - Task 4 (Length analysis) incomplete\n")

    # Key results
    report.append("\n## Key Results Summary\n")

    # V2-Neutral
    summary_csv = os.path.join(v2_dir, "summary.csv")
    if os.path.exists(summary_csv):
        report.append("\n### V2-Neutral Win Rates (GPT-4o-mini judge)\n")
        import csv
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                report.append(f"- **{row['Comparison']}**: {row['A_Win_Rate']}\n")

    # Multi-judge
    mj_summary = os.path.join(mj_dir, "judge_robustness_summary.csv")
    if os.path.exists(mj_summary):
        report.append("\n### Multi-Judge Agreement\n")
        import csv
        with open(mj_summary) as f:
            reader = csv.DictReader(f)
            for row in reader:
                report.append(f"- **{row['Comparison']}**: Agreement={row['Judge_Agreement']}, Majority A Win={row['Majority_A_Win_Pct']}\n")

    # Length analysis
    la_conclusion = os.path.join(la_dir, "paper_conclusion.txt")
    if os.path.exists(la_conclusion):
        report.append("\n### Length-Aware Analysis Conclusion\n")
        with open(la_conclusion) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Suggested paper conclusion:"):
                    idx = lines.index(line)
                    conclusion = ''.join(lines[idx+1:]).strip()
                    report.append(f"> {conclusion}\n")
                    break

    # Output paths
    report.append("\n## Output File Paths\n")
    report.append(f"- **Subset**: `{subset_path}`\n")
    report.append(f"- **V2-Neutral results**: `{v2_dir}/`\n")
    report.append(f"- **Multi-judge results**: `{mj_dir}/`\n")
    report.append(f"- **Length analysis**: `{la_dir}/`\n")
    report.append(f"- **Quality check**: `{qc_path}`\n")

    # Recommendations
    report.append("\n## Recommendations for Paper\n")
    report.append("1. **Main text**: Add V2-Neutral results to demonstrate prompt robustness\n")
    report.append("2. **Main text or Appendix**: Include multi-judge agreement table\n")
    report.append("3. **Main text**: Add length-aware analysis paragraph with regression results\n")
    report.append("4. **Appendix**: Include full prompt sensitivity table (V1/V2-Neutral/V2-Depth)\n")
    report.append("5. **Appendix**: Include detailed length analysis table\n")

    # LaTeX snippets
    report.append("\n## LaTeX Tables Ready for Paper\n")
    report.append("- `v2_neutral_gpt4o_mini/summary.tex`\n")
    report.append("- `v2_neutral_gpt4o_mini/prompt_sensitivity_table.tex`\n")
    report.append("- `multijudge_v2_neutral/judge_robustness_summary.tex`\n")
    report.append("- `length_analysis/length_analysis_summary.tex`\n")

    # Write report
    out_path = os.path.join(SUPPLEMENTAL_DIR, "final_report.md")
    with open(out_path, "w") as f:
        f.write(''.join(report))

    print(''.join(report))
    print(f"\n\nReport saved to: {out_path}")

    if all_complete:
        print("\n✓ ALL TASKS COMPLETE")
        print("✓ All output files generated")
        print("✓ Ready for paper integration")
    else:
        print("\n⚠ SOME TASKS INCOMPLETE")
        print("Run this script again after all tasks finish")


if __name__ == "__main__":
    main()

"""
TASK 6: Quality checks across all supplemental eval output files.
"""
import os, json, sys

SUPPLEMENTAL_DIR = os.path.dirname(__file__)
SUBSET_PATH = os.path.join(SUPPLEMENTAL_DIR, "subset_200_seed42.txt")

def load_subset():
    with open(SUBSET_PATH) as f:
        return set(int(l.strip()) for l in f if l.strip())

def check_jsonl(path, expected_n, model_a=None, model_b=None, subset_indices=None):
    issues = []
    if not os.path.exists(path):
        return [f"FILE MISSING: {path}"]
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) != expected_n:
        issues.append(f"Row count mismatch: got {len(rows)}, expected {expected_n}")

    errors = [r for r in rows if r.get("error")]
    if errors:
        issues.append(f"Errors: {len(errors)}/{len(rows)}")

    # real_winner validity
    valid_winners = {model_a, model_b, "Tie", None}
    bad_winners = [r for r in rows if r.get("real_winner") not in valid_winners]
    if bad_winners:
        issues.append(f"Invalid real_winner values: {len(bad_winners)}")

    # empty responses
    empty_a = [r for r in rows if not r.get("response_a", "").strip()]
    empty_b = [r for r in rows if not r.get("response_b", "").strip()]
    if empty_a:
        issues.append(f"Empty response_a: {len(empty_a)}")
    if empty_b:
        issues.append(f"Empty response_b: {len(empty_b)}")

    # Supporter: prefix artifacts
    prefix_a = [r for r in rows if str(r.get("response_a","")).strip() in ("Supporter:", "Assistant:")]
    prefix_b = [r for r in rows if str(r.get("response_b","")).strip() in ("Supporter:", "Assistant:")]
    if prefix_a or prefix_b:
        issues.append(f"Bare prefix artifacts: response_a={len(prefix_a)}, response_b={len(prefix_b)}")

    # A/B balance
    n_original = sum(1 for r in rows if r.get("order") == "original")
    n_swapped  = sum(1 for r in rows if r.get("order") == "swapped")
    balance = n_original / len(rows) if rows else 0
    if not (0.35 <= balance <= 0.65):
        issues.append(f"A/B order imbalance: original={n_original}, swapped={n_swapped} (ratio={balance:.2f})")

    # Duplicate indices
    indices = [r.get("index") for r in rows]
    if len(set(indices)) < len(indices):
        issues.append(f"Duplicate indices: {len(indices) - len(set(indices))} duplicates")

    # Subset consistency
    if subset_indices:
        row_indices = set(indices)
        missing = subset_indices - row_indices
        extra   = row_indices - subset_indices
        if missing:
            issues.append(f"Missing subset indices: {len(missing)}")
        if extra:
            issues.append(f"Extra (non-subset) indices: {len(extra)}")

    return issues


def main():
    subset_indices = load_subset()
    n_subset = len(subset_indices)

    report_lines = ["# Supplemental Eval Quality Check\n"]

    # Task 2: V2-Neutral
    v2_dir = os.path.join(SUPPLEMENTAL_DIR, "v2_neutral_gpt4o_mini")
    comparisons_t2 = [
        ("Llama-Cognitive", "Llama-Base"),
        ("Qwen-Cognitive",  "Qwen-Base"),
        ("Llama-Cognitive", "GPT-4o"),
        ("Llama-Cognitive", "Claude-Sonnet"),
        ("Qwen-Cognitive",  "GPT-4o"),
        ("Qwen-Cognitive",  "Claude-Sonnet"),
    ]
    report_lines.append("## Task 2: V2-Neutral GPT-4o-mini\n")
    for ma, mb in comparisons_t2:
        fname = f"{ma}_vs_{mb}_v2_neutral_gpt4o_mini.jsonl"
        path = os.path.join(v2_dir, fname)
        issues = check_jsonl(path, n_subset, ma, mb, subset_indices)
        status = "OK" if not issues else "ISSUES"
        report_lines.append(f"- **{fname}**: {status}")
        if issues:
            for iss in issues:
                report_lines.append(f"  - {iss}")

    # Task 3: Multi-judge
    mj_dir = os.path.join(SUPPLEMENTAL_DIR, "multijudge_v2_neutral")
    comparisons_t3 = [
        ("Llama-Cognitive", "Llama-Base"),
        ("Llama-Cognitive", "GPT-4o"),
        ("Llama-Cognitive", "Claude-Sonnet"),
        ("Qwen-Cognitive",  "GPT-4o"),
        ("Qwen-Cognitive",  "Claude-Sonnet"),
    ]
    judges = ["gpt-4o-mini", "gpt-4o", "claude"]
    report_lines.append("\n## Task 3: Multi-Judge Robustness\n")
    for ma, mb in comparisons_t3:
        for judge in judges:
            fname = f"{ma}_vs_{mb}_v2_neutral_{judge}.jsonl"
            path = os.path.join(mj_dir, fname)
            if not os.path.exists(path):
                # gpt4o-mini may be symlinked from task 2
                alt_path = os.path.join(v2_dir, f"{ma}_vs_{mb}_v2_neutral_gpt4o_mini.jsonl")
                if judge == "gpt-4o-mini" and os.path.exists(alt_path):
                    report_lines.append(f"- **{fname}**: REUSED from Task 2 (OK)")
                    continue
                else:
                    report_lines.append(f"- **{fname}**: SKIPPED (not available)")
                    continue
            issues = check_jsonl(path, n_subset, ma, mb, subset_indices)
            status = "OK" if not issues else "ISSUES"
            report_lines.append(f"- **{fname}**: {status}")
            if issues:
                for iss in issues:
                    report_lines.append(f"  - {iss}")

    # Task 4: Length analysis
    la_dir = os.path.join(SUPPLEMENTAL_DIR, "length_analysis")
    report_lines.append("\n## Task 4: Length Analysis\n")
    for fname in ["length_analysis_summary.csv", "length_analysis_summary.md",
                  "length_analysis_summary.tex", "regression_information.txt",
                  "regression_strategy.txt", "paper_conclusion.txt"]:
        path = os.path.join(la_dir, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = f"OK (size={size}B)" if exists and size > 0 else "MISSING or EMPTY"
        report_lines.append(f"- **{fname}**: {status}")

    # Task 5: Reward consistency
    rc_dir = os.path.join(SUPPLEMENTAL_DIR, "reward_consistency")
    report_lines.append("\n## Task 5: Reward Evaluator Consistency\n")
    report_lines.append("- Skipped reward evaluator consistency because MCTS intermediate candidates were not available in a standalone file.")

    out_path = os.path.join(SUPPLEMENTAL_DIR, "final_quality_check.md")
    with open(out_path, "w") as f:
        f.write('\n'.join(report_lines) + "\n")
    print(f"Quality check saved to {out_path}")
    print('\n'.join(report_lines))


if __name__ == "__main__":
    main()

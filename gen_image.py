import json
import numpy as np
from scipy import stats

COG_FILE = "ablation_experiments_qwen/eval_cognitive_qwens_eval_details_qwen.jsonl"
ABL_FILE = "ablation_experiments_qwen/eval_ablation_qwen_eval_details_qwen.jsonl"

DIMS = [
    ("Empathy", "dim_1"),
    ("Information", "dim_2"),
    ("Humanoid", "dim_3"),
    ("Strategies", "dim_4"),
    ("Cognitive Alignment", "dim_5"),
]

def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def scores(rows, key):
    return np.array([r["eval_scores"][key] for r in rows], dtype=float)

def sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def main():
    cog = load(COG_FILE)
    abl = load(ABL_FILE)

    print("N cognitive:", len(cog))
    print("N ablation :", len(abl))
    assert len(cog) == len(abl)

    print("\nLaTeX table:\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Metric & Ablation & Full Cognitive & $p$ \\\\")
    print("\\midrule")

    for name, key in DIMS:
        a = scores(abl, key)
        c = scores(cog, key)
        _, p = stats.ttest_rel(c, a)
        print(f"{name} & {a.mean():.2f} & {c.mean():.2f}{sig(p)} & {p:.4f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Point-wise ablation results comparing the full cognitive model against the variant without Emotion-Validation-First. Significance is computed using paired t-tests.}")
    print("\\label{tab:ablation_pointwise}")
    print("\\end{table}")

if __name__ == "__main__":
    main()

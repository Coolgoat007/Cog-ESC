# length_matched_pairwise.py
import json
import re
import numpy as np
from scipy import stats

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)

            base = d["response_Base"]
            cog = d["response_Cognitive"]
            winner = d["real_winner"]

            len_base = word_count(base)
            len_cog = word_count(cog)
            len_diff = len_cog - len_base

            rows.append({
                "len_base": len_base,
                "len_cog": len_cog,
                "len_diff": len_diff,
                "winner": winner,
            })
    return rows

def analyze_matched(path, name, max_abs_diff=30):
    rows = load_rows(path)

    matched = [
        r for r in rows
        if abs(r["len_diff"]) <= max_abs_diff and r["winner"] != "Tie"
    ]

    cog_wins = sum(1 for r in matched if r["winner"] == "Cognitive")
    base_wins = sum(1 for r in matched if r["winner"] == "Base")

    total = cog_wins + base_wins

    print("=" * 70)
    print(name)
    print(f"Length-matched subset: abs(length_diff) <= {max_abs_diff}")
    print("=" * 70)
    print(f"Matched decisive samples: {total}")
    print(f"Cognitive wins: {cog_wins}")
    print(f"Other wins: {base_wins}")

    if total > 0:
        win_rate = cog_wins / total
        p = stats.binomtest(cog_wins, total, p=0.5, alternative="two-sided").pvalue
        print(f"Cognitive decisive win rate: {win_rate * 100:.2f}%")
        print(f"Binomial p-value: {p:.6f}")

def main():
    V2_ABLATION_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/pairwise_eval_v2_details_qwen_cognitive_ablation.jsonl"

    for threshold in [10, 20, 30, 40, 50]:
        analyze_matched(
            V2_ABLATION_FILE,
            "V2 Therapeutic-Depth: Full Cognitive vs Ablation",
            max_abs_diff=threshold
        )
        print()

if __name__ == "__main__":
    main()

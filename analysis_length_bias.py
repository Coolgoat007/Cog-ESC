# analysis_length_bias.py
import json
import re
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def load_pairwise_details(path):
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

            if winner == "Cognitive":
                win = 1
            elif winner == "Base":
                win = 0
            else:
                win = 0.5

            rows.append({
                "len_base": len_base,
                "len_cog": len_cog,
                "len_diff": len_diff,
                "winner": winner,
                "win": win,
            })
    return rows

def analyze(path, name):
    rows = load_pairwise_details(path)

    decisive = [r for r in rows if r["winner"] != "Tie"]
    x = np.array([r["len_diff"] for r in decisive])
    y = np.array([r["win"] for r in decisive])

    rho, p = stats.spearmanr(x, y)

    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Total samples: {len(rows)}")
    print(f"Decisive samples: {len(decisive)}")
    print(f"Mean Base length: {np.mean([r['len_base'] for r in rows]):.2f}")
    print(f"Mean Cognitive length: {np.mean([r['len_cog'] for r in rows]):.2f}")
    print(f"Mean length diff Cog-Base: {np.mean([r['len_diff'] for r in rows]):.2f}")
    print()
    print("Spearman correlation between length_diff and Cognitive win:")
    print(f"rho = {rho:.4f}, p = {p:.6f}")

    X = x.reshape(-1, 1)
    clf = LogisticRegression()
    clf.fit(X, y)

    print()
    print("Logistic regression:")
    print(f"coef length_diff = {clf.coef_[0][0]:.6f}")
    print(f"intercept = {clf.intercept_[0]:.6f}")
    print()

def main():
    # 按你的实际路径修改
    V1_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/pairwise_eval_details.jsonl"
    V2_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/pairwise_eval_v2_details_qwen_cognitive_ablation.jsonl"

    analyze(V1_FILE, "V1 Standard Pairwise")
    analyze(V2_FILE, "V2 Therapeutic-Depth Pairwise")

if __name__ == "__main__":
    main()

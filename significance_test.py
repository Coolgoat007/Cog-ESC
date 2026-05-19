# significance_test.py
import json
import numpy as np
from scipy import stats

def load_scores(file_path):
    scores = {"dim_1": [], "dim_2": [], "dim_3": [], "dim_4": [], "dim_5": []}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for i in range(1, 6):
                scores[f"dim_{i}"].append(data['eval_scores'][f"dim_{i}"])
    return scores

if __name__ == "__main__":
    # Cognitive vs Base (LLaMA)
    # BASE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/generated_predictions_eval_details.jsonl"
    # COGNITIVE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions_eval_details.jsonl"

    # Cognitive vs Ablation (Qwen)
    ABLATION_FILE = "ablation_experiments_qwen/eval_ablation_qwen_eval_details_qwen.jsonl"
    COGNITIVE_FILE = "ablation_experiments_qwen/eval_cognitive_qwens_eval_details_qwen.jsonl"

    print("Loading data...")
    ablation_scores = load_scores(ABLATION_FILE)
    cog_scores = load_scores(COGNITIVE_FILE)

    dim_names = [
        "1. Empathy",
        "2. Information",
        "3. Humanoid",
        "4. Strategies",
        "5. Cognitive Alignment"
    ]

    print(f"\n{'='*60}")
    print("Paired T-Test Results (Cognitive vs Ablation)")
    print(f"{'='*60}")
    for i in range(1, 6):
        k = f"dim_{i}"
        t_stat, p_val = stats.ttest_rel(cog_scores[k], ablation_scores[k])
        mean_abl = np.mean(ablation_scores[k])
        mean_cog = np.mean(cog_scores[k])
        sig = "** (p<0.01)" if p_val < 0.01 else ("* (p<0.05)" if p_val < 0.05 else "(n.s.)")
        print(f"{dim_names[i-1]:<26} | Ablation: {mean_abl:.3f} | Cog: {mean_cog:.3f} | p={p_val:.4f} {sig}")

    print(f"\n{'='*60}")
    print("Wilcoxon Signed-Rank Test Results (Cognitive vs Ablation)")
    print(f"{'='*60}")
    for i in range(1, 6):
        k = f"dim_{i}"
        stat, p_val = stats.wilcoxon(cog_scores[k], ablation_scores[k])
        mean_abl = np.mean(ablation_scores[k])
        mean_cog = np.mean(cog_scores[k])
        sig = "** (p<0.01)" if p_val < 0.01 else ("* (p<0.05)" if p_val < 0.05 else "(n.s.)")
        print(f"{dim_names[i-1]:<26} | Ablation: {mean_abl:.3f} | Cog: {mean_cog:.3f} | p={p_val:.4f} {sig}")

    print(f"{'='*60}\n")

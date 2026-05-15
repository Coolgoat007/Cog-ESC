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
    # 替换为你实际的路径 ( Point-wise 绝对打分的 details 文件)
    # BASE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions_eval_details_qwen.jsonl"
    # BASE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/generated_predictions_eval_details.jsonl"
    # COGNITIVE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions_eval_details_qwen.jsonl"
    # COGNITIVE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions_eval_details.jsonl"
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
    
    print(f"\n{'='*50}")
    print("Paired T-Test Results (Cognitive vs Ablation)")
    print(f"{'='*50}")
    
    for i in range(1, 6):
        dim_key = f"dim_{i}"
        # 进行配对样本 t 检验
        t_stat, p_val = stats.ttest_rel(cog_scores[dim_key], ablation_scores[dim_key])
        
        # 计算均值
        mean_ablation = np.mean(ablation_scores[dim_key])
        mean_cog = np.mean(cog_scores[dim_key])
        
        # 判断显著性符号
        sig_marker = ""
        if p_val < 0.01:
            sig_marker = "** (p<0.01, Extremely Significant)"
        elif p_val < 0.05:
            sig_marker = "*  (p<0.05, Significant)"
        else:
            sig_marker = "   (p>=0.05, Not Significant)"
            
        print(f"{dim_names[i-1]:<22} | Ablation: {mean_ablation:.3f} | Cog: {mean_cog:.3f} | p-value: {p_val:.4f} {sig_marker}")
        
    print(f"{'='*50}\n")

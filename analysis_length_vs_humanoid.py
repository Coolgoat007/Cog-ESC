# analysis_length_vs_humanoid.py
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def parse_llama_factory_prompt(prompt_text):
    """提取对话历史"""
    try:
        chat_part = prompt_text.split("Conversation:\n")[1]
        chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
        return chat_history_clean
    except:
        return prompt_text

def load_and_analyze(detail_file, label):
    """加载评估详情文件，提取长度和Humanoid得分"""
    lengths = []
    humanoid_scores = []
    
    with open(detail_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 提取回复长度（按词数或字符数）
            response = data['predict'].strip()
            length = len(response.split())  # 词数
            # 或者用字符数: length = len(response)
            
            humanoid = data['eval_scores']['dim_3']  # Humanoid 是 dim_3
            
            lengths.append(length)
            humanoid_scores.append(humanoid)
    
    return lengths, humanoid_scores

# ===== 修改这里的路径 =====
LLAMA_BASE_DETAIL = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/generated_predictions_eval_details.jsonl"
LLAMA_COG_DETAIL = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions_eval_details.jsonl"
QWEN_BASE_DETAIL = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions_eval_details_qwen.jsonl"
QWEN_COG_DETAIL = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions_eval_details_qwen.jsonl"

# 加载数据
llama_base_len, llama_base_hum = load_and_analyze(LLAMA_BASE_DETAIL, "Llama-Base")
llama_cog_len, llama_cog_hum = load_and_analyze(LLAMA_COG_DETAIL, "Llama-Cognitive")
qwen_base_len, qwen_base_hum = load_and_analyze(QWEN_BASE_DETAIL, "Qwen-Base")
qwen_cog_len, qwen_cog_hum = load_and_analyze(QWEN_COG_DETAIL, "Qwen-Cognitive")

# 打印统计
print("=" * 60)
print("回复长度统计（词数）")
print("=" * 60)
for name, lens in [("Llama-Base", llama_base_len), ("Llama-Cognitive", llama_cog_len),
                    ("Qwen-Base", qwen_base_len), ("Qwen-Cognitive", qwen_cog_len)]:
    print(f"{name:20s}: Mean={np.mean(lens):.1f}, Median={np.median(lens):.1f}, Std={np.std(lens):.1f}")

# 计算 Spearman 相关系数（长度 vs Humanoid）
print("\n" + "=" * 60)
print("Spearman 相关性：回复长度 vs Humanoid 得分")
print("=" * 60)
for name, lens, hums in [("Llama-Base", llama_base_len, llama_base_hum),
                          ("Llama-Cognitive", llama_cog_len, llama_cog_hum),
                          ("Qwen-Base", qwen_base_len, qwen_base_hum),
                          ("Qwen-Cognitive", qwen_cog_len, qwen_cog_hum)]:
    rho, pval = stats.spearmanr(lens, hums)
    print(f"{name:20s}: ρ = {rho:.3f} (p = {pval:.4f})")

# 画图：两张子图并排（Llama vs Qwen）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, title, base_len, base_hum, cog_len, cog_hum in [
    (axes[0], "Llama-3.1-8B", llama_base_len, llama_base_hum, llama_cog_len, llama_cog_hum),
    (axes[1], "Qwen-2.5-7B", qwen_base_len, qwen_base_hum, qwen_cog_len, qwen_cog_hum)
]:
    ax.scatter(base_len, base_hum, alpha=0.5, label="Base", c="blue", marker="o")
    ax.scatter(cog_len, cog_hum, alpha=0.5, label="Cognitive", c="red", marker="x")
    
    # 趋势线
    for lens, hums, color, ls in [(base_len, base_hum, "blue", "--"),
                                   (cog_len, cog_hum, "red", "-")]:
        if len(lens) > 0:
            z = np.polyfit(lens, hums, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(lens), max(lens), 100)
            ax.plot(x_range, p(x_range), color=color, linestyle=ls, alpha=0.7)
    
    ax.set_xlabel("Response Length (words)")
    ax.set_ylabel("Humanoid Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("length_vs_humanoid.png", dpi=150)
print("\n图片已保存: length_vs_humanoid.png")
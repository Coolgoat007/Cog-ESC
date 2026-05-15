import matplotlib.pyplot as plt
import numpy as np

# ========== 数据 ==========
models = ['Llama-3.1-8B', 'Qwen-2.5-7B']

# 数据格式: [V1 Base, V1 Cognitive, V2 Base, V2 Cognitive]
llama = [50.9, 37.1, 38.7, 57.0]
qwen  = [41.6, 36.4, 45.9, 47.5]

# 显著性标记 (V1: Cognitive < Base, V2: Cognitive > Base)
# 使用你实际的二项检验结果：Llama V1 p<0.0001, V2 p<0.0001; Qwen V1 p=0.065, V2 p=0.346
sig_marks = [
    ['', '***', '', '***'],   # Llama: V1 Cog 显著低, V2 Cog 显著高
    ['', 'ns.', '', 'ns.']    # Qwen: 都不显著 (p>0.05)
]

# ========== 画图 ==========
x = np.arange(len(models))  # [0, 1]
width = 0.18                 # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 四个系列的柱子位置
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝, 橙, 绿, 红
labels = ['V1 Standard - Base', 'V1 Standard - Cognitive', 
          'V2 Therapeutic - Base', 'V2 Therapeutic - Cognitive']

for i, (label, offset, color) in enumerate(zip(labels, offsets, colors)):
    values = [llama[i], qwen[i]]
    bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)

    # 在柱子上方标注数值和显著性
    for j, bar in enumerate(bars):
        height = bar.get_height()
        sig = sig_marks[j][i] if i < len(sig_marks[0]) else ''
        annot = f'{height:.1f}%{sig}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                annot, ha='center', va='bottom', fontsize=8, fontweight='bold')

# 添加图例和标签
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 70)
ax.set_title('Pairwise Blind Test: V1 vs. V2 Evaluation Paradigms', fontsize=14, fontweight='bold')

# 添加显著性说明
ax.text(0.02, 0.98, '*** p<0.001 | ns. not significant', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_pairwise_winrate.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved as figure2_pairwise_winrate.png")
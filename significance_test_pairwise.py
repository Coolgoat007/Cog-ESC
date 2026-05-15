# significance_test_pairwise.py
from scipy.stats import binomtest

# ===== 你的实际数据 =====
# Llama
llama_v2_total = 682
llama_cog_win = 389
llama_base_win = 264

# Qwen
qwen_v2_total = 682
qwen_cog_win = 324
qwen_base_win = 313

# 二项检验 (假设胜率应为 50%)
def test_pairwise(total, win, opponent_win, model_name, alternative='greater'):
    decisive = win + opponent_win
    result = binomtest(win, n=decisive, p=0.5, alternative=alternative)
    alt_str = "Cognitive > 50%" if alternative == 'greater' else "Cognitive < 50%"
    print(f"{model_name}: {win}/{decisive} decisive wins ({win/decisive*100:.1f}%), H1: {alt_str}, p = {result.pvalue:.6f}")

print("=" * 60)
print("V2 Pairwise 二项检验 (H1: Cognitive 胜率 > 50%)")
print("=" * 60)
test_pairwise(llama_v2_total, llama_cog_win, llama_base_win, "Llama-Cognitive V2", 'greater')
test_pairwise(qwen_v2_total, qwen_cog_win, qwen_base_win, "Qwen-Cognitive V2", 'greater')

# V1 检验
llama_v1_cog, llama_v1_base = 253, 347
qwen_v1_cog, qwen_v1_base = 248, 284

print("\n" + "=" * 60)
print("V1 Pairwise 二项检验 (H1: Cognitive 胜率 < 50%)")
print("=" * 60)
test_pairwise(llama_v2_total, llama_v1_cog, llama_v1_base, "Llama-Cognitive V1", 'less')
test_pairwise(qwen_v2_total, qwen_v1_cog, qwen_v1_base, "Qwen-Cognitive V1", 'less')
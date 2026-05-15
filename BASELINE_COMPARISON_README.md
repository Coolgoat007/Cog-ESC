# Strong Baseline Comparison Experiments

## 概述

本实验补充了与强大基线模型（GPT-4o, Claude-3.5-Sonnet, GPT-4o-mini）的对比，用于验证认知注入方法的有效性。

## 实验目标

1. **序列评估**：验证GPT-4o/Claude是否也存在"premature advice"问题
2. **5维度评估**：对比小模型与大模型在各维度的表现
3. **Pairwise评估**：在therapeutic depth导向下，对比Cognitive模型与强基线

## 文件说明

### 生成脚本
- `generate_strong_baseline_responses.py` - 生成GPT-4o/Claude/GPT-4o-mini的回复

### 评估脚本
- `sequence_eval_llm_v2_with_baselines.py` - 序列评估（Validation-First & Premature-Advice）
- `llm_eval_with_baselines.py` - 5维度质量评估
- `pairwise_eval_v2_with_baselines.py` - Pairwise V2评估（Therapeutic Depth）

### 辅助脚本
- `generate_final_report.py` - 生成最终报告和LaTeX表格
- `run_all_evaluations.py` - 自动运行所有评估的主控脚本

## 使用流程

### Step 1: 生成强基线回复

```bash
python generate_strong_baseline_responses.py
```

**输入**：
- `/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl`

**输出**：
- `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o/generated_predictions.jsonl`
- `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o_mini/generated_predictions.jsonl`
- `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/claude_sonnet/generated_predictions.jsonl`

**时间**：约1小时（682条 × 3个模型）
**成本**：约$4-5

### Step 2: 运行评估

#### 方式A：自动运行所有评估（推荐）

```bash
python run_all_evaluations.py
```

这会依次运行：
1. 序列评估（30-40分钟）
2. 5维度评估（40-50分钟）
3. Pairwise评估（60-90分钟）
4. 生成最终报告

**总时间**：约2-3小时

#### 方式B：手动运行单个评估

```bash
# 1. 序列评估
python sequence_eval_llm_v2_with_baselines.py

# 2. 5维度评估
python llm_eval_with_baselines.py

# 3. Pairwise评估
python pairwise_eval_v2_with_baselines.py

# 4. 生成报告
python generate_final_report.py
```

### Step 3: 查看结果

评估完成后，结果保存在以下目录：

```
sequence_eval_with_baselines_results/
├── all_sequence_summary.csv
├── gpt4o_sequence_details.jsonl
├── claude_sonnet_sequence_details.jsonl
├── llama_cognitive_sequence_details.jsonl
└── ...

eval_with_baselines_results/
├── gpt4o_eval_summary.txt
├── gpt4o_eval_details.jsonl
├── claude_sonnet_eval_summary.txt
└── ...

pairwise_with_baselines_results/
├── Llama-Cognitive_vs_GPT-4o_summary.txt
├── Llama-Cognitive_vs_GPT-4o_details.jsonl
├── Qwen-Cognitive_vs_GPT-4o_summary.txt
└── ...
```

LaTeX表格保存在：
```
final_report_sequence_table.tex
final_report_5dim_table.tex
final_report_pairwise_table.tex
```

## 评估指标说明

### 1. 序列评估指标

- **Validation-First Rate (↑)**：验证优先率，越高越好
- **Premature-Advice Rate (↓)**：过早建议率，越低越好
- **Avg Length**：平均响应长度（词数）

### 2. 5维度评估指标（0-4分）

1. **Empathy**：共情能力
2. **Information**：信息丰富度
3. **Humanoid**：类人性
4. **Strategies**：策略适当性
5. **Cognitive Alignment**：认知对齐度（核心创新）

### 3. Pairwise评估指标

- **Win Rate**：在therapeutic depth导向评估下的胜率
- **Tie Rate**：平局率

## 预期发现

### 假设1：强基线也存在premature advice问题
如果GPT-4o的Premature-Advice Rate > 40%，说明这是LLM的普遍问题。

### 假设2：小模型通过认知注入可以接近大模型
如果Llama-Cognitive在某些维度接近GPT-4o，证明方法有效。

### 假设3：Therapeutic depth评估更能反映真实质量
如果Cognitive模型在V2评估下表现更好，验证了评估协议的有效性。

## 成本估算

### 生成成本
- GPT-4o: ~$2
- GPT-4o-mini: ~$0.1
- Claude: ~$2.5
- **总计**: ~$4.5

### 评估成本
- 序列评估: ~$0.4
- 5维度评估: ~$0.4
- Pairwise评估: ~$0.7
- **总计**: ~$1.5

### 总成本：约$6

## 时间估算

- 生成：1小时
- 评估：2-3小时
- **总计**：3-4小时

## 注意事项

1. **API密钥配置**：确保`.env`文件中配置了正确的API密钥
2. **并发控制**：评估脚本默认使用30个并发线程，可根据API限制调整
3. **错误处理**：脚本包含重试机制，但如果频繁失败，请检查API配额
4. **磁盘空间**：确保有足够空间存储结果文件（约500MB）

## 论文使用建议

### 新增章节：Comparison with Strong Baselines

建议在Results部分新增一个小节：

```latex
\subsection{Comparison with Strong Baselines}

To contextualize our results, we compare our cognitive-aligned models 
with state-of-the-art commercial models including GPT-4o, Claude-3.5-Sonnet, 
and GPT-4o-mini.

\subsubsection{Sequencing Behavior}
Table~\ref{tab:sequence_baselines} shows that even GPT-4o exhibits 
premature advice behavior (X\% premature-advice rate), suggesting that 
therapeutic pacing remains a challenge for current LLMs...

\subsubsection{Therapeutic Depth}
Under V2 evaluation (Table~\ref{tab:pairwise_baselines}), our 
Llama-Cognitive model achieves Y\% win rate against GPT-4o, 
demonstrating that cognitive-guided optimization can help smaller 
models approach the therapeutic depth of much larger models...
```

## 故障排除

### 问题1：API密钥错误
```bash
# 检查.env文件
cat .env

# 应该包含：
# openai_key=sk-...
# openai_url=https://...
```

### 问题2：生成失败
```bash
# 查看日志
tail -100 generation_log.txt

# 重新运行生成
python generate_strong_baseline_responses.py
```

### 问题3：评估失败
```bash
# 检查生成的文件是否存在
ls -lh /root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/*/generated_predictions.jsonl

# 单独运行失败的评估
python sequence_eval_llm_v2_with_baselines.py
```

## 联系方式

如有问题，请检查：
1. API密钥是否正确
2. 生成的文件是否完整
3. 评估脚本的输出日志

---

**创建时间**：2026-05-15
**版本**：1.0

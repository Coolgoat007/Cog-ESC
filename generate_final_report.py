#!/usr/bin/env python3
"""
生成最终的对比报告
整合所有评估结果，生成论文可用的表格和图表数据
"""

import json
import os
import pandas as pd
import numpy as np

def load_sequence_results():
    """加载序列评估结果"""
    csv_file = "sequence_eval_with_baselines_results/all_sequence_summary.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df
    return None

def load_5dim_results():
    """加载5维度评估结果"""
    results = {}
    result_dir = "eval_with_baselines_results"

    if not os.path.exists(result_dir):
        return None

    for file in os.listdir(result_dir):
        if file.endswith("_eval_summary.txt"):
            model_name = file.replace("_eval_summary.txt", "")
            with open(os.path.join(result_dir, file), 'r') as f:
                content = f.read()
                # 解析分数
                scores = {}
                for line in content.split('\n'):
                    if "Empathy:" in line:
                        scores['Empathy'] = float(line.split(':')[1].split('/')[0].strip())
                    elif "Information:" in line:
                        scores['Information'] = float(line.split(':')[1].split('/')[0].strip())
                    elif "Humanoid:" in line:
                        scores['Humanoid'] = float(line.split(':')[1].split('/')[0].strip())
                    elif "Strategies:" in line:
                        scores['Strategies'] = float(line.split(':')[1].split('/')[0].strip())
                    elif "Cognitive Alignment:" in line:
                        scores['Cognitive Alignment'] = float(line.split(':')[1].split('/')[0].strip())
                results[model_name] = scores

    return results

def load_pairwise_results():
    """加载Pairwise评估结果"""
    results = {}
    result_dir = "pairwise_with_baselines_results"

    if not os.path.exists(result_dir):
        return None

    for file in os.listdir(result_dir):
        if file.endswith("_summary.txt"):
            comparison = file.replace("_summary.txt", "")
            with open(os.path.join(result_dir, file), 'r') as f:
                content = f.read()
                # 解析胜率：格式 "🥇 ModelName Wins : 675 (99.0%)"
                import re
                wins = {}
                for line in content.split('\n'):
                    m = re.search(r'([\d.]+)%\)', line)
                    if m and 'Wins' in line:
                        pct = float(m.group(1))
                        # 提取模型名
                        name_part = re.sub(r'[🥇🥈🤝]', '', line.split('Wins')[0]).strip()
                        wins[name_part] = pct
                results[comparison] = wins

    return results

def generate_latex_table_sequence(df):
    """生成序列评估的LaTeX表格"""
    print("\n" + "="*60)
    print("LaTeX Table: Sequence Evaluation")
    print("="*60)

    latex = """\\begin{table}[t]
\\centering
\\caption{Sequencing behavior comparison with strong baselines.}
\\label{tab:sequence_baselines}
\\begin{tabular}{lccc}
\\toprule
Model & Avg. Length & Validation-First ↑ & Premature-Advice ↓ \\\\
\\midrule
"""

    # 按照特定顺序排列
    order = ['gpt4o', 'claude_sonnet', 'gpt4o_mini', 'llama_cognitive', 'qwen_cognitive', 'llama_base', 'qwen_base']

    for model in order:
        row = df[df['name'] == model]
        if not row.empty:
            name = model.replace('_', '-').title()
            length = row['avg_length'].values[0]
            vf = row['validation_first_rate'].values[0] * 100
            pa = row['premature_advice_rate'].values[0] * 100
            latex += f"{name} & {length:.1f} & {vf:.1f}\\% & {pa:.1f}\\% \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    print(latex)
    return latex

def generate_latex_table_5dim(results):
    """生成5维度评估的LaTeX表格"""
    print("\n" + "="*60)
    print("LaTeX Table: 5-Dimension Evaluation")
    print("="*60)

    latex = """\\begin{table}[t]
\\centering
\\caption{5-dimension evaluation with strong baselines.}
\\label{tab:5dim_baselines}
\\begin{tabular}{lccccc}
\\toprule
Model & Empathy & Info & Humanoid & Strategy & Cognitive \\\\
\\midrule
"""

    order = ['gpt4o', 'claude_sonnet', 'gpt4o_mini', 'llama_cognitive', 'qwen_cognitive', 'llama_base', 'qwen_base']

    for model in order:
        if model in results:
            scores = results[model]
            name = model.replace('_', '-').title()
            latex += f"{name} & {scores.get('Empathy', 0):.2f} & {scores.get('Information', 0):.2f} & {scores.get('Humanoid', 0):.2f} & {scores.get('Strategies', 0):.2f} & {scores.get('Cognitive Alignment', 0):.2f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    print(latex)
    return latex

def generate_latex_table_pairwise(results):
    """生成Pairwise评估的LaTeX表格"""
    print("\n" + "="*60)
    print("LaTeX Table: Pairwise Evaluation")
    print("="*60)

    latex = """\\begin{table}[t]
\\centering
\\caption{Pairwise V2 (Therapeutic Depth) evaluation with strong baselines.}
\\label{tab:pairwise_baselines}
\\begin{tabular}{lcc}
\\toprule
Comparison & Model 1 Win Rate & Model 2 Win Rate \\\\
\\midrule
"""

    for comparison, wins in results.items():
        parts = comparison.split('_vs_')
        if len(parts) == 2:
            model1 = parts[0].replace('_', '-')
            model2 = parts[1].replace('_', '-')
            vals = list(wins.values())
            rate1 = vals[0] if len(vals) > 0 else 0.0
            rate2 = vals[1] if len(vals) > 1 else 0.0
            latex += f"{model1} vs {model2} & {rate1:.1f}\\% & {rate2:.1f}\\% \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    print(latex)
    return latex

def generate_summary_report():
    """生成完整的总结报告"""
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT: Strong Baseline Comparison")
    print("="*80)

    # 1. 序列评估
    seq_df = load_sequence_results()
    if seq_df is not None:
        print("\n### 1. SEQUENCE EVALUATION ###")
        print(seq_df.to_string(index=False))
        latex_seq = generate_latex_table_sequence(seq_df)

        # 保存LaTeX
        with open("final_report_sequence_table.tex", 'w') as f:
            f.write(latex_seq)

    # 2. 5维度评估
    dim5_results = load_5dim_results()
    if dim5_results:
        print("\n### 2. 5-DIMENSION EVALUATION ###")
        for model, scores in dim5_results.items():
            print(f"\n{model}:")
            for dim, score in scores.items():
                print(f"  {dim}: {score:.2f}")

        latex_5dim = generate_latex_table_5dim(dim5_results)
        with open("final_report_5dim_table.tex", 'w') as f:
            f.write(latex_5dim)

    # 3. Pairwise评估
    pairwise_results = load_pairwise_results()
    if pairwise_results:
        print("\n### 3. PAIRWISE EVALUATION ###")
        for comparison, wins in pairwise_results.items():
            print(f"\n{comparison}:")
            for model, count in wins.items():
                print(f"  {model}: {count}")

        latex_pairwise = generate_latex_table_pairwise(pairwise_results)
        with open("final_report_pairwise_table.tex", 'w') as f:
            f.write(latex_pairwise)

    # 4. 关键发现
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if seq_df is not None:
        print("\n### Sequencing Metrics ###")

        # GPT-4o的表现
        gpt4o_row = seq_df[seq_df['name'] == 'gpt4o']
        if not gpt4o_row.empty:
            vf = gpt4o_row['validation_first_rate'].values[0] * 100
            pa = gpt4o_row['premature_advice_rate'].values[0] * 100
            print(f"GPT-4o: Validation-First={vf:.1f}%, Premature-Advice={pa:.1f}%")

        # Llama-Cognitive的表现
        llama_cog_row = seq_df[seq_df['name'] == 'llama_cognitive']
        if not llama_cog_row.empty:
            vf = llama_cog_row['validation_first_rate'].values[0] * 100
            pa = llama_cog_row['premature_advice_rate'].values[0] * 100
            print(f"Llama-Cognitive: Validation-First={vf:.1f}%, Premature-Advice={pa:.1f}%")

        # Qwen-Cognitive的表现
        qwen_cog_row = seq_df[seq_df['name'] == 'qwen_cognitive']
        if not qwen_cog_row.empty:
            vf = qwen_cog_row['validation_first_rate'].values[0] * 100
            pa = qwen_cog_row['premature_advice_rate'].values[0] * 100
            print(f"Qwen-Cognitive: Validation-First={vf:.1f}%, Premature-Advice={pa:.1f}%")

    print("\n" + "="*80)
    print("Report generation completed!")
    print("LaTeX tables saved to:")
    print("  - final_report_sequence_table.tex")
    print("  - final_report_5dim_table.tex")
    print("  - final_report_pairwise_table.tex")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report()

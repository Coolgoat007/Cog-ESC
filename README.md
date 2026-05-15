# Healing Requires More Than Helpfulness: Cognitive-Inspired Preference Optimization for Therapeutically Paced Emotional Support Conversations

> **Paper**: [arXiv link — to be added upon submission]
> **Base framework**: Built on [CSO (Chain-of-Strategy Optimization)](https://arxiv.org/abs/2503.05362)
> **Dataset**: [ESC-Pro on HuggingFace](https://huggingface.co/datasets/XingYuSSS/ESC-Pro)

## Overview

This repository contains code and evaluation results for our paper on cognitive-guided preference optimization for Emotional Support Conversations (ESC). We make two main contributions:

**1. Cognitive-guided preference optimization** — We inject five prefrontal cognitive regulation principles (Emotion-Validation-First, Non-Judgment, Gentle Cognitive Guidance, Risk Awareness, Natural Conversational Flow) into the MCTS-based preference construction pipeline of CSO, improving therapeutic depth on LLaMA-3.1-8B and Qwen-2.5-7B.

**2. Verbosity Tax identification and quantification** — We identify and quantify a systematic evaluator bias in LLM-as-a-judge evaluation: judges favor shorter, more conversational responses over longer, therapeutically complete ones when evaluator instructions do not explicitly protect against brevity preference. Key finding:

| Protocol | Llama-Cog vs GPT-4o | Llama-Cog vs Claude-Sonnet |
|---|---|---|
| V2-Depth (therapeutic depth) | **71.6%** | **70.2%** |
| V2-Neutral (length-agnostic) | 23.0% | 10.0% |
| GPT-4o judge (robustness check) | 43.0% | — |

This 31–60pp reversal is driven by judges citing "verbose" and "overwhelming" in 9–44% of commercial-wins explanations, while "comprehensive" and "detailed" dominate cognitive-wins explanations.

## Repository Structure

```
├── MCTS.py                        # Core MCTS implementation
├── run.py                         # Stage 1: generate conversation trees
├── build_data.py                  # Stage 2: build preference pairs
├── change_data.py                 # Stage 3: convert to DPO format
├── change_data_kto.py             # Stage 3 (alt): convert to KTO format
├── requirements.txt
├── .env.example                   # API key template
│
├── supplemental_eval/             # Verbosity Tax experiments (see README inside)
│   ├── configs.py                 # Centralized path config
│   ├── task2_v2_neutral_eval.py   # V2-Neutral evaluation (1200 API calls)
│   ├── task3_lc_vs_gpt4o_gpt4o_judge.py  # GPT-4o judge robustness check
│   ├── task3_multijudge_revised.py
│   ├── task4_length_analysis.py   # Length-controlled regression analysis
│   ├── task6_quality_check.py     # Evaluation quality verification
│   ├── v2_neutral_gpt4o_mini/     # Pre-computed results: 6×200 evaluations
│   ├── multijudge_v2_neutral/     # Pre-computed results: GPT-4o judge (100 instances)
│   └── length_analysis/           # Pre-computed regression outputs
│
└── eval_with_baselines_results/   # Pointwise evaluation outputs (generated locally)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Use pre-computed results (no API calls needed)

All supplemental evaluation results are pre-computed and included in the repo:

```bash
# View V2-Neutral vs V2-Depth comparison table
cat supplemental_eval/v2_neutral_gpt4o_mini/summary.md

# View keyword analysis
cat supplemental_eval/keyword_analysis.csv

# View length regression results
cat supplemental_eval/length_analysis/length_analysis_summary.md
```

### 4. Reproduce evaluations from scratch

Download model predictions from HuggingFace and set paths in `.env`:

```bash
# Set prediction file paths (see .env.example for all variables)
export LLAMA_COGNITIVE_PREDICTIONS=/path/to/llama_cognitive/generated_predictions.jsonl
export GPT4O_PREDICTIONS=/path/to/gpt4o/generated_predictions.jsonl

# Run V2-Neutral evaluation (requires OpenAI API key, ~1200 calls)
python -m supplemental_eval.task2_v2_neutral_eval

# Run length analysis (no API calls)
python -m supplemental_eval.task4_length_analysis

# Run quality checks
python -m supplemental_eval.task6_quality_check
```

## Main Results

### Cognitive vs Base (stable across all protocols)
| Comparison | V1-Standard | V2-Neutral | V2-Depth |
|---|---|---|---|
| Llama-Cognitive vs Llama-Base | 37.1% | 47.5% | 57.0% |
| Qwen-Cognitive vs Qwen-Base | 36.4% | 55.5% | 47.5% |

### Cognitive vs Commercial (protocol-sensitive)
| Comparison | V2-Neutral | V2-Depth |
|---|---|---|
| Llama-Cognitive vs GPT-4o | 23.0% | 71.6% |
| Llama-Cognitive vs Claude-Sonnet | 10.0% | 70.2% |
| Qwen-Cognitive vs GPT-4o | 21.0% | 52.1% |
| Qwen-Cognitive vs Claude-Sonnet | 9.0% | 58.4% |

### Length-controlled analysis
After OLS regression controlling for response length, the cognitive-commercial Information score gap shrinks by only **5.6%** (Spearman ρ ≤ 0.08 for all models), confirming the advantage is not a length artifact.

## Citation

If you use this code or results, please cite:

```bibtex
@article{yao2026healing,
  title={Healing Requires More Than Helpfulness: Cognitive-Inspired Preference Optimization for Therapeutically Paced Emotional Support Conversations},
  author={Yao, Jichen},
  year={2026}
}
```

Also cite the base CSO framework:

```bibtex
@article{zhao2025chain,
  title={Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter},
  author={Zhao, Weixiang and others},
  journal={arXiv preprint arXiv:2503.05362},
  year={2025}
}
```

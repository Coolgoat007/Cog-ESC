# Beyond Helpfulness: Cognitive-Inspired Preference Construction for Therapeutically Paced Emotional Support Conversations

> **Paper**: [arXiv link — to be added upon submission]
> **Base framework**: Built on [CSO (Chain-of-Strategy Optimization)](https://arxiv.org/abs/2503.05362)
> **Training data**: [ESC-Pro on HuggingFace](https://huggingface.co/datasets/XingYuSSS/ESC-Pro)
> **DPO preference data**: [Cog-ESC-DPO on HuggingFace](https://huggingface.co/datasets/shunqiziran123/Cog-ESC-DPO) — base and cognitive preference pairs
> **Evaluation data**: [Cog-ESC-Eval on HuggingFace](https://huggingface.co/datasets/shunqiziran123/Cog-ESC-Eval) — predictions, pointwise scores, verbosity tax results

## Overview

This repository contains code for our paper on cognitively inspired preference-data construction for Emotional Support Conversations (ESC). Two main contributions:

**1. CogCSO+MCTS preference construction** — Five cognitively inspired regulation principles (Emotion-Validation-First, Non-Judgment, Gentle Cognitive Guidance, Risk Awareness, Natural Conversational Flow) injected into the CSO+MCTS preference-data construction pipeline. Base and Cognitive models are both DPO/LoRA-tuned; the only difference is whether their preference data come from the original CSO+MCTS pipeline or our CogCSO+MCTS variant. Training on CogCSO+MCTS-constructed data improves Information and Strategy scores on LLaMA-3.1-8B and Qwen-2.5-7B.

**2. Verbosity Tax hypothesis** — A verbosity-related evaluation pattern in LLM-as-a-judge settings: evaluators tend to favor shorter, more conversational responses unless support-content completeness is explicitly emphasized.

| Protocol | Llama-Cog vs GPT-4o | Llama-Cog vs Claude-Sonnet |
|---|---|---|
| V2-Depth (therapeutic depth) | **71.6%** | **70.2%** |
| V2-Neutral (length-agnostic) | 23.0% | 10.0% |
| GPT-4o judge (robustness check) | 43.0% | — |

## Repository Structure

```
├── MCTS.py                              # Core MCTS implementation
├── build_data.py                        # Build preference pairs from trees
├── change_data.py / change_data_kto.py  # Convert to DPO / KTO format
├── util.py                              # Shared utilities
├── run_base.py                          # Run base model training
├── run_full_cognitive.py                # Run cognitive-guided training (reward weights: 0.1/0.1/0.1/0.6/0.1)
├── run_ablation_cognitive.py            # Run ablation (w/o Emotion-Validation-First)
├── llm_eval_with_baselines.py           # Pointwise evaluation
├── pairwise_eval_v2_with_baselines.py   # Pairwise evaluation (V2-Depth protocol)
├── sequence_eval_llm_v2_with_baselines.py  # Sequencing behavior evaluation
├── significance_test.py                 # Paired t-test + Wilcoxon signed-rank tests
├── ablation_experiments_qwen/           # Ablation eval data (Qwen-2.5-7B backbone)
│   ├── eval_{base,cognitive,ablation}_qwens_eval_details_qwen.jsonl  # Pointwise scores
│   ├── eval_{base,cognitive,ablation}_qwens_eval_summary_qwen.txt    # Score summaries
│   ├── pairwise_eval_v2_details_qwen_cognitive_ablation.jsonl        # Pairwise results
│   └── pairwise_eval_v2_summary_qwen_cognitive_ablation.txt          # 58.3% win rate summary
└── supplemental_eval/                   # Verbosity Tax experiments
```

## Reproducing the Main Pipeline

### 1. Install dependencies
```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI API key
```

### 2. Generate conversation trees
```bash
python run_base.py          # base model
python run_full_cognitive.py  # cognitive-guided model
```

### 3. Build preference data and train
```bash
python build_data.py
python change_data.py       # DPO format
# or: python change_data_kto.py  # KTO format
```

### 4. Evaluate
```bash
python llm_eval_with_baselines.py           # pointwise scores
python pairwise_eval_v2_with_baselines.py   # pairwise preferences (V2-Depth)
python sequence_eval_llm_v2_with_baselines.py  # sequencing behavior
python significance_test.py                 # paired t-test + Wilcoxon signed-rank (Cognitive vs Ablation)
```

> **Pre-computed ablation results** are available in `ablation_experiments_qwen/` (Qwen-2.5-7B backbone).
> The pairwise ablation comparison (Full Cognitive vs. Ablation, V2-Depth, N=682) yields a **58.3% decisive win rate**
> (372W / 266L / 44T); raw judgments are in `pairwise_eval_v2_details_qwen_cognitive_ablation.jsonl`.

### 5. Reproduce Verbosity Tax experiments
See [`supplemental_eval/README.md`](supplemental_eval/README.md).

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
After OLS regression controlling for response length, the cognitive-commercial Information score gap shrinks by only **5.6%**, suggesting the advantage is not explained by response length alone.

## Citation

```bibtex
@article{yao2026beyond,
  title={Beyond Helpfulness: Cognitive-Inspired Preference Construction for Therapeutically Paced Emotional Support Conversations},
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

# Healing Requires More Than Helpfulness: Cognitive-Inspired Preference Optimization for Therapeutically Paced Emotional Support Conversations

> **Paper**: [arXiv link — to be added upon submission]
> **Base framework**: Built on [CSO (Chain-of-Strategy Optimization)](https://arxiv.org/abs/2503.05362)
> **Training data**: [ESC-Pro on HuggingFace](https://huggingface.co/datasets/XingYuSSS/ESC-Pro)

## Overview

This repository contains code for our paper on cognitive-guided preference optimization for Emotional Support Conversations (ESC). Two main contributions:

**1. Cognitive-guided preference optimization** — Five prefrontal cognitive regulation principles (Emotion-Validation-First, Non-Judgment, Gentle Cognitive Guidance, Risk Awareness, Natural Conversational Flow) injected into the MCTS-based preference construction pipeline of CSO, improving therapeutic depth on LLaMA-3.1-8B and Qwen-2.5-7B.

**2. Verbosity Tax identification** — A systematic evaluator bias where LLM judges favor shorter responses over therapeutically complete ones when not explicitly instructed otherwise:

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
├── run_full_cognitive.py                # Run cognitive-guided training
├── run_ablation_cognitive.py            # Run ablation (w/o Emotion-Validation-First)
├── llm_eval_with_baselines.py           # Pointwise evaluation
├── pairwise_eval_v2_with_baselines.py   # Pairwise evaluation (V2-Depth protocol)
├── sequence_eval_llm_v2_with_baselines.py  # Sequencing behavior evaluation
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
```

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
After OLS regression controlling for response length, the cognitive-commercial Information score gap shrinks by only **5.6%**, confirming the advantage is not a length artifact.

## Citation

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

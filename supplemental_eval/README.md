# Supplemental Evaluation: Verbosity Tax Experiments

Code for the supplemental experiments in Section 5.5 and Discussion of the paper.
Pre-computed results and raw evaluation outputs are available on HuggingFace: [link TBD].

## Evaluation Protocols

| Protocol | Key instruction | Effect |
|---|---|---|
| **V1-Standard** | Standard pairwise quality | Brevity-favoring |
| **V2-Neutral** | "Penalize verbose but repetitive responses" | Moderately brevity-favoring |
| **V2-Depth** | "Do not penalize a response for being detailed or longer" | Verbosity-tolerant |

## Reproducing the Experiments

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

Download model predictions from HuggingFace and set paths in `.env` (see `.env.example`).

### Task 2: V2-Neutral evaluation (GPT-4o-mini judge, ~1200 API calls)

```bash
python -m supplemental_eval.task2_v2_neutral_eval
```

Runs 6 pairwise comparisons × 200 instances. Resumable — skips already-evaluated indices.
Output: `supplemental_eval/v2_neutral_gpt4o_mini/`

### Task 3: GPT-4o judge robustness check (~100 API calls)

```bash
python -m supplemental_eval.task3_lc_vs_gpt4o_gpt4o_judge
```

Replicates Llama-Cognitive vs GPT-4o under V2-Neutral using GPT-4o as judge.
Requires Task 2 output to reuse the same A/B swap decisions.
Output: `supplemental_eval/multijudge_v2_neutral/`

### Task 4: Length-aware analysis (no API calls)

```bash
python -m supplemental_eval.task4_length_analysis
```

OLS regression of quality scores on log(length) + model fixed effects.
Requires model prediction files and pointwise evaluation files.
Output: `supplemental_eval/length_analysis/`

### Task 6: Quality checks (no API calls)

```bash
python -m supplemental_eval.task6_quality_check
```

Verifies N, errors, winner validity, and A/B balance across all output files.

## Configuration

Set prediction file paths via environment variables or edit `configs.py`:

```bash
export LLAMA_COGNITIVE_PREDICTIONS=/path/to/llama_cognitive/generated_predictions.jsonl
export GPT4O_PREDICTIONS=/path/to/gpt4o/generated_predictions.jsonl
# ... see .env.example for all variables
```

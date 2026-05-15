# Supplemental Evaluation: Verbosity Tax Experiments

This directory contains all code and pre-computed results for the supplemental experiments in Section 5.5 and the Discussion of the paper.

## Pre-computed Results (no API calls needed)

All results are already computed and included. You can inspect them directly:

| File/Directory | Contents |
|---|---|
| `v2_neutral_gpt4o_mini/` | 6×200 V2-Neutral evaluations (GPT-4o-mini judge) |
| `multijudge_v2_neutral/` | 100 V2-Neutral evaluations (GPT-4o judge, robustness check) |
| `length_analysis/` | OLS regression outputs, length-stratified statistics |
| `keyword_analysis.csv` | Keyword frequencies in judge explanations |
| `winner_length_analysis.csv` | Response lengths stratified by winner |
| `quality_check_task2.csv` | Quality verification: N, errors, A/B balance for all 6 comparisons |

## Evaluation Protocols

| Protocol | Key instruction | Bias direction |
|---|---|---|
| **V1-Standard** | Standard pairwise quality | Strongly brevity-favoring |
| **V2-Neutral** | "Penalize verbose but repetitive responses" | Moderately brevity-favoring |
| **V2-Depth** | "Do not penalize a response for being detailed or longer" | Verbosity-tolerant |

The divergence between V2-Neutral and V2-Depth is the core evidence for the **Verbosity Tax**: removing the explicit verbosity-tolerance instruction causes a 31–60pp drop in cognitive model win rates against commercial baselines.

## Scripts

### Task 0: Build evaluation subsets
```bash
python supplemental_eval/task0_build_subset.py    # 200-instance subset (seed=42)
python supplemental_eval/task0b_build_subset_100.py  # 100-instance subset for Task 3
```
Outputs: `subset_200_seed42.txt`, `subset_100_seed42.txt`

### Task 2: V2-Neutral evaluation (GPT-4o-mini judge)
```bash
python -m supplemental_eval.task2_v2_neutral_eval
```
- 6 comparisons × 200 instances = 1,200 API calls
- Resumable: skips already-evaluated indices
- Requires: `OPENAI_API_KEY`, all 6 model prediction files

### Task 3: GPT-4o judge robustness check
```bash
python -m supplemental_eval.task3_lc_vs_gpt4o_gpt4o_judge
```
- Llama-Cognitive vs GPT-4o only, 100 instances, GPT-4o judge
- Reuses same A/B swap decisions as Task 2 for per-instance comparison
- Requires: `OPENAI_API_KEY`, Llama-Cognitive and GPT-4o prediction files

### Task 4: Length-aware analysis (no API calls)
```bash
python -m supplemental_eval.task4_length_analysis
```
- OLS regression: score ~ log(length) + model fixed effects
- Spearman correlation between length and quality scores
- Requires: all model prediction files + pointwise evaluation files

### Task 6: Quality checks (no API calls)
```bash
python -m supplemental_eval.task6_quality_check
```
- Verifies N, errors, winner validity, A/B balance for all output files
- No external dependencies beyond the output JSONL files

## Configuration

Set prediction file paths via environment variables (see `../.env.example`) or edit `configs.py` directly.

```bash
# Minimum required for Task 3
export LLAMA_COGNITIVE_PREDICTIONS=/path/to/llama_cognitive/generated_predictions.jsonl
export GPT4O_PREDICTIONS=/path/to/gpt4o/generated_predictions.jsonl
```

## Key Findings Summary

**Asymmetric preference reversal** (V2-Neutral vs V2-Depth):
- Cognitive vs Base: stable (±10pp), because length gap is small (~1.1–1.5×)
- Cognitive vs Commercial: reverses 31–60pp, because length gap is large (~2–4×)

**Flip analysis** (within 200-instance subset):
- 68–88% of cognitive wins under V2-Depth flip to commercial wins under V2-Neutral
- 0% of commercial wins flip to cognitive wins

**Judge robustness**: GPT-4o judge gives 43% (vs 23% with mini, vs 71.6% under V2-Depth) — stronger judge partially resists brevity bias but does not eliminate it.

**Length regression**: Cognitive-commercial Information gap shrinks only 5.6% after controlling for length (N=4,774, OLS with model fixed effects).

# Commercial Baseline Rerun Notes

## Why We Reran

1. The original `generate_strong_baseline_responses.py` included an inference-time brevity constraint in the system prompt (`Most of the time you speak less than 25 words at a time`), making commercial baseline comparisons unfair.
2. The original Claude-Sonnet run produced 528/682 empty outputs due to silent API failures.
3. GPT-4o-mini original results were brevity-contaminated (avg=24 words, median=24) and had systematic `Supporter:` prefix artifacts (626/682 lines).

## New Generation Settings

- Removed explicit brevity constraint from `SUPPORTER_SYS_PROMPT` in `generate_strong_baseline_responses.py`.
- GPT-4o: retained new no-brevity version (confirmed 0/682 lines identical to backup; avg=75 vs backup avg=27).
- Claude-Sonnet: resumed from 154 valid entries, regenerated 528 empty outputs via `fix_baselines.py`.
- GPT-4o-mini: full regeneration (force_regen=True) to replace brevity-contaminated results.
- After generation, stripped leading speaker labels (`Supporter:`, `Assistant:`, full-width variants) from `predict` field using regex `^(Supporter|Assistant)\s*[：:]\s*`. No content was modified.

## Backup Paths

| Backup | Path |
|--------|------|
| Old brevity version | `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines_backup_brevity/` |
| New no-brevity raw (before prefix clean) | `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines_no_brevity_raw_before_prefix_clean/` |
| Current eval input (prefix-cleaned) | `/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/` |

## Health Check Results (post prefix-clean)

| model | total | empty | errors | avg words | median | p10 | p90 | Supporter: | short<5 |
|-------|-------|-------|--------|-----------|--------|-----|-----|-----------|---------|
| gpt4o | 682 | 0 | 0 | 75.1 | 71 | 52 | 96 | 0 | 0 |
| gpt4o_mini | 682 | 0 | 0 | 53.2 | 52 | 36 | 73 | 0 | 0 |
| claude_sonnet | 682 | 0 | 0 | 67.5 | 82 | 17 | 94 | 0 | 2 |

## Evaluation Settings

- Judge model: `gpt-4o` (confirmed in `util.py`)
- Eval input: prefix-cleaned `strong_baselines/` (not backup)
- All eval scripts use `open('w')` — fully overwrite prior results, no skip logic
- Eval log: `eval_rerun_no_brevity_gpt4o_judge_prefix_clean.log`
- Eval PID: 18035/18037 (started 2026-05-15 13:22)

## Paper Note

> We removed leading speaker labels (e.g., "Supporter:") from generated outputs before evaluation, without modifying response content. This is an engineering normalization step, not semantic post-processing.

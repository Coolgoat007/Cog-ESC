import json
import random
import os

INPUT = "sequence_eval_llm_v2_results/llama_cognitive_sequence_details.jsonl"
OUT = "llama_cognitive_sequencing_failures_sample.txt"

random.seed(42)

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

rows = load_rows(INPUT)

# 抽取 Llama-Cognitive 中 sequencing 失败样本
failures = [
    r for r in rows
    if (not r["validation_first"]) or r["premature_advice"]
]

print(f"Total rows: {len(rows)}")
print(f"Failure rows: {len(failures)}")

sample = random.sample(failures, min(50, len(failures)))

with open(OUT, "w", encoding="utf-8") as f:
    for i, r in enumerate(sample, 1):
        f.write("=" * 100 + "\n")
        f.write(f"Sample {i}\n")
        f.write(f"Index: {r['index']}\n")
        f.write(f"Length: {r['length']}\n")
        f.write(f"Validation-first: {r['validation_first']}\n")
        f.write(f"Premature-advice: {r['premature_advice']}\n\n")
        f.write("Response:\n")
        f.write(r["response"].strip() + "\n\n")
        f.write("Judge output:\n")
        f.write(r["judge_output"].strip() + "\n\n")

print(f"Saved to: {OUT}")

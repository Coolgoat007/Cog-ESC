# inspect_sequence_errors.py
import json
import random

FILES = [
    "sequence_eval_llm_results/llama_cognitive_sequence_details.jsonl",
    "sequence_eval_llm_results/qwen_cognitive_sequence_details.jsonl",
    "sequence_eval_llm_results/qwen_base_sequence_details.jsonl",
]

random.seed(42)

for path in FILES:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    sample = random.sample(rows, 10)

    for r in sample:
        print("\n" + "-" * 80)
        print("INDEX:", r["index"])
        print("VALIDATION_FIRST:", r["validation_first"])
        print("PREMATURE_ADVICE:", r["premature_advice"])
        print("\nRESPONSE:")
        print(r["response"])
        print("\nJUDGE:")
        print(r["judge_output"])

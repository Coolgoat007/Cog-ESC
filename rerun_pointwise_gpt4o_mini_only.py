import sys
sys.path.insert(0, '/root/baseline/CSO-main')
from llm_eval_with_baselines import evaluate_file

evaluate_file(
    "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o_mini/generated_predictions.jsonl",
    max_workers=30
)

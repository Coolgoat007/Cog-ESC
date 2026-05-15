# sequence_eval_llm.py
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from util import retry, generate

FILES = {
    "llama_base": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/generated_predictions.jsonl",
    "llama_cognitive": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions.jsonl",
    "llama_ablation": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_llama/generated_predictions.jsonl",

    "qwen_base": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl",
    "qwen_cognitive": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions.jsonl",
    "qwen_ablation": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/generated_predictions.jsonl",
}

OUT_DIR = "sequence_eval_llm_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCE_PROMPT = """
You are evaluating one response in an emotional support conversation.

Your task is to judge the temporal sequence of support behaviors, not the overall quality.

Definitions:

Validation means the supporter explicitly acknowledges, accepts, or normalizes the user's emotional experience.
Examples: "That sounds really painful", "I can understand why you feel that way", "Your feelings make sense."

Intervention means advice, planning, reframing, problem-solving, corrective interpretation, or suggesting actions.
Examples: "You should...", "Try to...", "It may help to...", "Consider...", "Let's make a plan", "Maybe look at it this way."

Please judge the supporter response using two binary labels:

1. validation_first:
Answer Yes if the response validates the user's emotional experience before introducing any intervention.
Answer No if it starts with advice, reframing, planning, correction, or problem-solving before sufficient validation.
If the response contains validation but no intervention, answer Yes.
If the response contains neither clear validation nor clear intervention, answer No.

2. premature_advice:
Answer Yes if the response introduces advice, reframing, planning, correction, or problem-solving before sufficient emotional validation.
Answer No otherwise.

Conversation context:
{chat_history}

Supporter response:
{response}

Output exactly in this format:
validation_first: Yes/No
premature_advice: Yes/No
reason: one short sentence
"""

def parse_llama_factory_prompt(prompt_text):
    try:
        chat_part = prompt_text.split("Conversation:\n")[1]
        chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
        chat_history_clean = re.sub(r"Supporter:\s*$", "", chat_history_clean).strip()
        return chat_history_clean
    except Exception:
        return prompt_text

def word_count(text):
    return len(re.findall(r"\b\w+\b", text or ""))

@retry()
def judge_sequence(chat_history, response):
    query = SEQUENCE_PROMPT.format(
        chat_history=chat_history,
        response=response
    )
    output = generate(query)

    vf_match = re.search(r"validation_first\s*:\s*(yes|no)", output, re.I)
    pa_match = re.search(r"premature_advice\s*:\s*(yes|no)", output, re.I)

    if not vf_match or not pa_match:
        print("Bad judge output:")
        print(repr(output))
        raise ValueError("Failed to parse sequencing judge output")

    validation_first = vf_match.group(1).lower() == "yes"
    premature_advice = pa_match.group(1).lower() == "yes"

    return {
        "validation_first": validation_first,
        "premature_advice": premature_advice,
        "judge_output": output
    }

def process_line(line, index):
    data = json.loads(line)
    chat_history = parse_llama_factory_prompt(data["prompt"])
    response = data["predict"].strip()

    result = judge_sequence(chat_history, response)

    return {
        "index": index,
        "chat_history": chat_history,
        "response": response,
        "length": word_count(response),
        "validation_first": result["validation_first"],
        "premature_advice": result["premature_advice"],
        "judge_output": result["judge_output"]
    }

def evaluate_file(name, path, max_workers=20):
    print("=" * 90)
    print(f"Evaluating: {name}")
    print(path)
    print("=" * 90)

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    detailed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_line, line, i)
            for i, line in enumerate(lines)
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=name):
            try:
                detailed.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    detailed.sort(key=lambda x: x["index"])

    n = len(detailed)
    avg_len = np.mean([x["length"] for x in detailed])
    vf_rate = np.mean([x["validation_first"] for x in detailed])
    pa_rate = np.mean([x["premature_advice"] for x in detailed])

    summary = {
        "name": name,
        "path": path,
        "n": n,
        "avg_length": avg_len,
        "validation_first_rate": vf_rate,
        "premature_advice_rate": pa_rate
    }

    detail_path = os.path.join(OUT_DIR, f"{name}_sequence_details.jsonl")
    summary_path = os.path.join(OUT_DIR, f"{name}_sequence_summary.txt")

    with open(detail_path, "w", encoding="utf-8") as f:
        for item in detailed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_text = (
        f"{'=' * 60}\n"
        f"{name}\n"
        f"{'=' * 60}\n"
        f"N                         : {n}\n"
        f"Avg length                : {avg_len:.2f}\n"
        f"Validation-First Rate     : {vf_rate * 100:.2f}%\n"
        f"Premature-Advice Rate     : {pa_rate * 100:.2f}%\n"
        f"Details                   : {detail_path}\n"
        f"{'=' * 60}\n"
    )

    print(summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary

def main():
    all_summaries = []

    for name, path in FILES.items():
        summary = evaluate_file(name, path, max_workers=30)
        if summary:
            all_summaries.append(summary)

    final_path = os.path.join(OUT_DIR, "all_sequence_summary.csv")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("name,n,avg_length,validation_first_rate,premature_advice_rate\n")
        for s in all_summaries:
            f.write(
                f"{s['name']},{s['n']},{s['avg_length']:.4f},"
                f"{s['validation_first_rate']:.6f},"
                f"{s['premature_advice_rate']:.6f}\n"
            )

    print(f"\nAll summaries saved to: {final_path}")

if __name__ == "__main__":
    main()

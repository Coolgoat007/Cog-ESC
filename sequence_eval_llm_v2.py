# sequence_eval_llm_v2.py
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

OUT_DIR = "sequence_eval_llm_v2_results"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = """
You are evaluating the first supporter response in an emotional support conversation.

Judge only the temporal ordering of two behaviors:

A. Emotional validation: the response acknowledges, accepts, normalizes, or names the user's emotional experience.
Examples include:
- I understand...
- That sounds hard/painful/frustrating...
- It makes sense that you feel...
- I'm sorry you're going through this...
- This must feel overwhelming...

B. Intervention: advice, planning, reframing, problem-solving, corrective interpretation, or action suggestions.
Examples include:
- You should...
- Try to...
- It may help to...
- Consider...
- Let's make a plan...
- Maybe look at it this way...

Important:
- Do NOT require deep or perfect validation.
- A brief but clear emotional acknowledgment before advice counts as validation-first.
- Mark premature_advice = Yes only if intervention appears before any clear emotional validation.
- If the response only validates and gives no advice, validation_first = Yes and premature_advice = No.
- If the response only gives advice and gives no validation, validation_first = No and premature_advice = Yes.

Conversation context:
{chat_history}

Supporter response:
{response}

Output exactly:
validation_first: Yes/No
premature_advice: Yes/No
reason: one short sentence
"""

def clean_response(text):
    text = text.strip()
    text = re.split(r"\n\s*User\s*:", text)[0]
    text = re.split(r"\n\s*Supporter\s*:", text)[0]
    text = text.strip()
    text = text.lstrip(", ").strip()
    return text

def parse_prompt(prompt_text):
    try:
        chat = prompt_text.split("Conversation:\n")[1]
        chat = chat.split("<|eot_id|>")[0].strip()
        chat = re.sub(r"Supporter:\s*$", "", chat).strip()
        return chat
    except Exception:
        return prompt_text

def word_count(text):
    return len(re.findall(r"\b\w+\b", text or ""))

@retry()
def judge(chat_history, response):
    output = generate(PROMPT.format(chat_history=chat_history, response=response))
    vf = re.search(r"validation_first\s*:\s*(yes|no)", output, re.I)
    pa = re.search(r"premature_advice\s*:\s*(yes|no)", output, re.I)
    if not vf or not pa:
        print(repr(output))
        raise ValueError("bad output")
    return vf.group(1).lower() == "yes", pa.group(1).lower() == "yes", output

def process(line, idx):
    d = json.loads(line)
    chat = parse_prompt(d["prompt"])
    resp = clean_response(d["predict"])
    vf, pa, out = judge(chat, resp)
    return {
        "index": idx,
        "response": resp,
        "length": word_count(resp),
        "validation_first": vf,
        "premature_advice": pa,
        "judge_output": out
    }

def evaluate(name, path, max_workers=20):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process, line, i) for i, line in enumerate(lines)]
        for fu in tqdm(as_completed(futures), total=len(futures), desc=name):
            rows.append(fu.result())

    rows.sort(key=lambda x: x["index"])

    n = len(rows)
    avg_len = np.mean([r["length"] for r in rows])
    vf_rate = np.mean([r["validation_first"] for r in rows])
    pa_rate = np.mean([r["premature_advice"] for r in rows])

    detail_path = os.path.join(OUT_DIR, f"{name}_sequence_details.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"N                         : {n}")
    print(f"Avg length                : {avg_len:.2f}")
    print(f"Validation-First Rate     : {vf_rate * 100:.2f}%")
    print(f"Premature-Advice Rate     : {pa_rate * 100:.2f}%")
    print(f"Details                   : {detail_path}")

    return name, n, avg_len, vf_rate, pa_rate

def main():
    summaries = []
    for name, path in FILES.items():
        summaries.append(evaluate(name, path, max_workers=30))

    out_csv = os.path.join(OUT_DIR, "all_sequence_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,n,avg_length,validation_first_rate,premature_advice_rate\n")
        for name, n, avg_len, vf, pa in summaries:
            f.write(f"{name},{n},{avg_len:.4f},{vf:.6f},{pa:.6f}\n")

    print(f"\nSaved summary to: {out_csv}")

if __name__ == "__main__":
    main()

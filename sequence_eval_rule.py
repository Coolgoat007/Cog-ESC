# sequence_eval_rule.py
import json
import re
import os
import numpy as np

FILES = {
    "llama_base": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/generated_predictions.jsonl",
    "llama_cognitive": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions.jsonl",
    "llama_ablation": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_llama/generated_predictions.jsonl",

    "qwen_base": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl",
    "qwen_cognitive": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions.jsonl",
    "qwen_ablation": "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/generated_predictions.jsonl",
}

VALIDATION_PATTERNS = [
    r"\bi understand\b",
    r"\bi can understand\b",
    r"\bthat sounds\b",
    r"\bit sounds\b",
    r"\bi hear\b",
    r"\bi'm sorry\b",
    r"\bi am sorry\b",
    r"\bit makes sense\b",
    r"\bit's understandable\b",
    r"\bthat's understandable\b",
    r"\bmust be hard\b",
    r"\bmust be really hard\b",
    r"\bthat must\b",
    r"\byour feelings are valid\b",
    r"\bfeelings are valid\b",
    r"\boverwhelming\b",
    r"\bpainful\b",
    r"\bdifficult\b",
    r"\bhard\b",
    r"\bfrustrating\b",
    r"\bupsetting\b",
]

ADVICE_PATTERNS = [
    r"\byou should\b",
    r"\byou need to\b",
    r"\byou have to\b",
    r"\byou might want to\b",
    r"\byou could\b",
    r"\byou can\b",
    r"\bit may be helpful\b",
    r"\bit might be helpful\b",
    r"\btry to\b",
    r"\bconsider\b",
    r"\bi suggest\b",
    r"\bi recommend\b",
    r"\blet's\b",
    r"\bwe can\b",
    r"\bplan\b",
    r"\bstrategy\b",
    r"\breframe\b",
    r"\bfocus on\b",
]

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def first_match_pos(text, patterns):
    text_l = text.lower()
    positions = []
    for p in patterns:
        m = re.search(p, text_l)
        if m:
            positions.append(m.start())
    return min(positions) if positions else None

def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()

def analyze_response(resp):
    resp = resp.strip()
    first_sent = first_sentence(resp)

    val_pos = first_match_pos(resp, VALIDATION_PATTERNS)
    adv_pos = first_match_pos(resp, ADVICE_PATTERNS)

    has_validation = val_pos is not None
    has_advice = adv_pos is not None

    validation_first = False
    premature_advice = False

    if has_validation and has_advice:
        validation_first = val_pos < adv_pos
        premature_advice = adv_pos < val_pos
    elif has_validation and not has_advice:
        validation_first = True
        premature_advice = False
    elif has_advice and not has_validation:
        validation_first = False
        premature_advice = True

    # 更严格地看“开头是否建议”
    first_sent_adv = first_match_pos(first_sent, ADVICE_PATTERNS) is not None
    first_sent_val = first_match_pos(first_sent, VALIDATION_PATTERNS) is not None

    if first_sent_adv and not first_sent_val:
        premature_advice = True

    return {
        "length": word_count(resp),
        "has_validation": has_validation,
        "has_advice": has_advice,
        "validation_first": validation_first,
        "premature_advice": premature_advice,
    }

def evaluate_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            resp = d["predict"]
            rows.append(analyze_response(resp))

    n = len(rows)
    return {
        "n": n,
        "avg_length": np.mean([r["length"] for r in rows]),
        "validation_first_rate": np.mean([r["validation_first"] for r in rows]),
        "premature_advice_rate": np.mean([r["premature_advice"] for r in rows]),
        "has_validation_rate": np.mean([r["has_validation"] for r in rows]),
        "has_advice_rate": np.mean([r["has_advice"] for r in rows]),
    }

def main():
    print("=" * 90)
    print("Rule-based sequencing evaluation")
    print("=" * 90)

    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"[Missing] {name}: {path}")
            continue

        res = evaluate_file(path)
        print(f"\n{name}")
        print("-" * 60)
        print(f"N                         : {res['n']}")
        print(f"Avg length                : {res['avg_length']:.2f}")
        print(f"Validation-First Rate     : {res['validation_first_rate'] * 100:.2f}%")
        print(f"Premature-Advice Rate     : {res['premature_advice_rate'] * 100:.2f}%")
        print(f"Has Validation Rate       : {res['has_validation_rate'] * 100:.2f}%")
        print(f"Has Advice Rate           : {res['has_advice_rate'] * 100:.2f}%")

if __name__ == "__main__":
    main()

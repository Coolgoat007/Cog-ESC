"""
Task 3 (reduced): LC vs GPT-4o, 100 instances, gpt-4o judge, V2-Neutral prompt.
Uses SAME swap decisions as the gpt-4o-mini run for a direct per-instance comparison.
Resumable.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supplemental_eval.configs import RESPONSE_FILES

SUPPLEMENTAL_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(SUPPLEMENTAL_DIR, "multijudge_v2_neutral")
MINI_DIR = os.path.join(SUPPLEMENTAL_DIR, "v2_neutral_gpt4o_mini")
SUBSET_PATH = os.path.join(SUPPLEMENTAL_DIR, "subset_100_seed42.txt")
os.makedirs(OUT_DIR, exist_ok=True)

COMP = "Llama-Cognitive_vs_GPT-4o"
MODEL_A = "Llama-Cognitive"
MODEL_B = "GPT-4o"
JUDGE = "gpt-4o"
OUT_PATH = os.path.join(OUT_DIR, f"{COMP}_v2_neutral_gpt-4o.jsonl")

V2_NEUTRAL_SYSTEM = (
    "You are an evaluator of emotional support conversations. "
    "Your task is to compare two supporter responses to the same conversation context. "
    "Evaluate which response provides better emotional support."
)

V2_NEUTRAL_USER_TEMPLATE = """\
Conversation context:
{chat_history}

Response A:
{response_a}

Response B:
{response_b}

Please compare Response A and Response B according to the following criteria:

1. Emotional validation: Does the response accurately acknowledge and normalize the user's emotional experience?
2. Intervention timing: Does the response avoid moving into advice, reframing, or problem-solving before sufficient validation?
3. Constructive support: Does the response provide useful emotional, informational, or cognitive support when appropriate?
4. Non-judgment and safety: Does the response avoid blame, minimization, coercive advice, or emotionally escalating suggestions?
5. Conversational naturalness: Does the response sound natural, supportive, and responsive to the user's immediate state?

Important instructions:
- Do not assume that a longer response is better.
- Do not assume that a shorter response is better.
- Prefer the response that better balances validation, appropriate timing, concrete support, and naturalness.
- Penalize responses that are verbose but repetitive, generic, or unnatural.
- Penalize responses that are concise but shallow, dismissive, or lacking useful support.
- If both responses are similarly good or similarly flawed, choose Tie.

Return only valid JSON:
{{
  "winner": "A" or "B" or "Tie",
  "confidence": a number from 0 to 1,
  "explanation": "brief explanation"
}}"""


def load_responses(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_chat_history(prompt_str):
    lines = prompt_str.split('\n')
    history_lines = [l.strip() for l in lines
                     if l.strip().startswith(('seeker:', 'supporter:', 'User:'))]
    return '\n'.join(history_lines) if history_lines else prompt_str[-800:]


def parse_judge_output(raw):
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    try:
        obj = json.loads(cleaned)
        winner = obj.get("winner", "").strip()
        if winner not in ("A", "B", "Tie"):
            return None, None, None, f"Invalid winner: {winner!r}"
        conf = float(obj["confidence"]) if obj.get("confidence") is not None else None
        return winner, conf, str(obj.get("explanation", "")), None
    except Exception as e:
        return None, None, None, str(e)


def call_judge(prompt_text, system_text, model, max_attempts=3, delay=2):
    import time
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('openai_key'),
        base_url=os.getenv('openai_url'),
    )
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_text},
                          {"role": "user",   "content": prompt_text}],
                stream=False,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise e


def main():
    # Load subset
    with open(SUBSET_PATH) as f:
        subset = [int(l.strip()) for l in f if l.strip()]
    print(f"Subset: {len(subset)} indices")

    # Load swap decisions from mini run (MUST match for per-instance comparison)
    mini_path = os.path.join(MINI_DIR, f"{COMP}_v2_neutral_gpt4o_mini.jsonl")
    mini_rows = {r['index']: r for r in
                 [json.loads(l) for l in open(mini_path) if l.strip()]}
    swap_map = {idx: (mini_rows[idx]['order'] == 'swapped') for idx in subset}
    print(f"Swap map loaded: {sum(swap_map.values())} swapped, {sum(1 for v in swap_map.values() if not v)} original")

    # Load responses
    resp_lc  = load_responses(RESPONSE_FILES["Llama-Cognitive"])
    resp_gpt = load_responses(RESPONSE_FILES["GPT-4o"])

    # Load already-done
    done = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done[obj['index']] = obj
    remaining = [i for i in subset if i not in done]
    print(f"Already done: {len(done)}, remaining: {len(remaining)}")

    with open(OUT_PATH, "a") as fout:
        for idx in remaining:
            lc_text  = resp_lc[idx]["predict"]
            gpt_text = resp_gpt[idx]["predict"]
            chat_hist = extract_chat_history(resp_lc[idx]["prompt"])

            swap = swap_map[idx]
            if swap:
                display_a, display_b = gpt_text, lc_text
                order = "swapped"
            else:
                display_a, display_b = lc_text, gpt_text
                order = "original"

            prompt_text = V2_NEUTRAL_USER_TEMPLATE.format(
                chat_history=chat_hist,
                response_a=display_a,
                response_b=display_b,
            )

            error = None
            winner_raw = confidence = explanation = None
            raw_output = ""
            try:
                raw_output = call_judge(prompt_text, V2_NEUTRAL_SYSTEM, JUDGE)
                winner_raw, confidence, explanation, parse_err = parse_judge_output(raw_output)
                if parse_err:
                    raw_output2 = call_judge(
                        prompt_text + "\n\nReturn valid JSON only. No markdown.",
                        V2_NEUTRAL_SYSTEM, JUDGE)
                    winner_raw, confidence, explanation, parse_err2 = parse_judge_output(raw_output2)
                    if parse_err2:
                        error = f"parse_err: {parse_err2}"
                        raw_output = raw_output2
            except Exception as e:
                error = str(e)

            if winner_raw and not error:
                if winner_raw == "Tie":
                    real_winner = "Tie"
                elif winner_raw == "A":
                    real_winner = MODEL_B if swap else MODEL_A
                else:
                    real_winner = MODEL_A if swap else MODEL_B
            else:
                real_winner = None

            record = {
                "index": idx, "comparison": COMP,
                "model_a": MODEL_A, "model_b": MODEL_B,
                "order": order, "judge_model": JUDGE,
                "winner": winner_raw, "real_winner": real_winner,
                "confidence": confidence, "explanation": explanation,
                "raw_judge_output": raw_output, "error": error,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            status = f"real_winner={real_winner}" if not error else f"ERROR: {error[:50]}"
            print(f"  idx={idx:3d}  {status}")

    # Final check
    all_rows = [json.loads(l) for l in open(OUT_PATH) if l.strip()]
    n_err = sum(1 for r in all_rows if r.get('error'))
    lc_wins  = sum(1 for r in all_rows if r.get('real_winner') == MODEL_A)
    gpt_wins = sum(1 for r in all_rows if r.get('real_winner') == MODEL_B)
    ties     = sum(1 for r in all_rows if r.get('real_winner') == 'Tie')
    print(f"\n=== DONE: N={len(all_rows)}, LC={lc_wins}({lc_wins/len(all_rows)*100:.1f}%), "
          f"GPT-4o={gpt_wins}({gpt_wins/len(all_rows)*100:.1f}%), Tie={ties}, Err={n_err} ===")


if __name__ == "__main__":
    main()

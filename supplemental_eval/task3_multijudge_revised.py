"""
TASK 3 (REVISED): Multi-judge robustness - REDUCED SCOPE
- Only 3 comparisons (Llama-Cognitive vs Base/GPT-4o/Claude-Sonnet)
- Only 100 instances per comparison (not 200)
- Only 2 judges: gpt-4o-mini (reuse from Task 2), gpt-4o (new)
- No Claude judge
"""
import sys, os, json, random, re, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supplemental_eval.configs import RESPONSE_FILES

SEED = 42
SUPPLEMENTAL_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(SUPPLEMENTAL_DIR, "multijudge_v2_neutral")
V2_NEUTRAL_DIR = os.path.join(SUPPLEMENTAL_DIR, "v2_neutral_gpt4o_mini")
SUBSET_PATH = os.path.join(SUPPLEMENTAL_DIR, "subset_100_seed42.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# CHANGED: Only 3 comparisons
COMPARISONS = [
    ("Llama-Cognitive", "Llama-Base"),
    ("Llama-Cognitive", "GPT-4o"),
    ("Llama-Cognitive", "Claude-Sonnet"),
]

# CHANGED: Only 2 judges
JUDGES = ["gpt-4o-mini", "gpt-4o"]

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
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_chat_history(prompt_str):
    lines = prompt_str.split('\n')
    history_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('seeker:') or line.startswith('supporter:') or line.startswith('User:'):
            history_lines.append(line)
    if history_lines:
        return '\n'.join(history_lines)
    return prompt_str[-800:]


def parse_judge_output(raw):
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    try:
        obj = json.loads(cleaned)
        winner = obj.get("winner", "").strip()
        if winner not in ("A", "B", "Tie"):
            return None, None, None, f"Invalid winner: {winner!r}"
        conf = obj.get("confidence", None)
        if conf is not None:
            try:
                conf = float(conf)
            except Exception:
                conf = None
        expl = str(obj.get("explanation", ""))
        return winner, conf, expl, None
    except Exception as e:
        return None, None, None, str(e)


def call_judge_openai(prompt_text, system_text, judge_model):
    from openai import OpenAI
    from dotenv import load_dotenv
    import os as _os
    load_dotenv()
    client = OpenAI(
        api_key=_os.getenv('openai_key'),
        base_url=_os.getenv('openai_url'),
    )
    msgs = [{"role": "system", "content": system_text},
            {"role": "user",   "content": prompt_text}]
    resp = client.chat.completions.create(
        model=judge_model,
        messages=msgs,
        stream=False,
    )
    return resp.choices[0].message.content


def call_judge_with_retry(prompt_text, system_text, judge_model, max_attempts=3, delay=2):
    import time
    last_exc = None
    for _ in range(max_attempts):
        try:
            return call_judge_openai(prompt_text, system_text, judge_model)
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    raise last_exc


def load_existing_mini_results(comp_key, subset_indices):
    """Load gpt-4o-mini results from Task 2 output, filter to subset_100."""
    fname = f"{comp_key}_v2_neutral_gpt4o_mini.jsonl"
    path = os.path.join(V2_NEUTRAL_DIR, fname)
    if not os.path.exists(path):
        return
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj["index"] in subset_indices:
                    results[obj["index"]] = obj
    return results


def run_judge_for_comparison(model_a_name, model_b_name, judge_model, responses_a, responses_b,
                              subset_indices, rng_swap_map, out_path):
    """Run a single judge over a comparison. Skip already done indices."""
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done[obj["index"]] = obj

    remaining = [i for i in subset_indices if i not in done]
    print(f"  Judge={judge_model}, {model_a_name} vs {model_b_name}: done={len(done)}, remaining={len(remaining)}")

    with open(out_path, "a") as fout:
        for idx in remaining:
            row_a = responses_a[idx]
            row_b = responses_b[idx]
            resp_a_text = row_a["predict"]
            resp_b_text = row_b["predict"]
            chat_hist = extract_chat_history(row_a["prompt"])

            swap = rng_swap_map.get(idx, False)
            if swap:
                display_a, display_b = resp_b_text, resp_a_text
                order = "swapped"
            else:
                display_a, display_b = resp_a_text, resp_b_text
                order = "original"

            prompt_text = V2_NEUTRAL_USER_TEMPLATE.format(
                chat_history=chat_hist,
                response_a=display_a,
                response_b=display_b,
            )

            error = None
            winner_raw = None
            confidence = None
            explanation = ""
            raw_output = ""

            try:
                raw_output = call_judge_with_retry(prompt_text, V2_NEUTRAL_SYSTEM, judge_model)
                winner_raw, confidence, explanation, parse_err = parse_judge_output(raw_output)
                if parse_err:
                    retry_prompt = prompt_text + "\n\nYou MUST respond with valid JSON only. No markdown."
                    raw_output2 = call_judge_with_retry(retry_prompt, V2_NEUTRAL_SYSTEM, judge_model)
                    winner_raw, confidence, explanation, parse_err2 = parse_judge_output(raw_output2)
                    if parse_err2:
                        error = f"parse_err: {parse_err2}"
                        raw_output = raw_output2
            except Exception as e:
                error = str(e)

            if winner_raw is not None and error is None:
                if winner_raw == "Tie":
                    real_winner = "Tie"
                elif winner_raw == "A":
                    real_winner = model_b_name if swap else model_a_name
                else:
                    real_winner = model_a_name if swap else model_b_name
            else:
                real_winner = None

            record = {
                "index": idx,
                "comparison": f"{model_a_name}_vs_{model_b_name}",
                "model_a": model_a_name,
                "model_b": model_b_name,
                "order": order,
                "judge_model": judge_model,
                "winner": winner_raw,
                "real_winner": real_winner,
                "confidence": confidence,
                "explanation": explanation,
                "raw_judge_output": raw_output,
                "error": error,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            status = f"winner={real_winner}" if error is None else f"ERROR: {error[:40]}"
            print(f"    idx={idx:3d}  {status}")

    all_results = {}
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_results[obj["index"]] = obj
    return all_results


def compute_majority(per_judge_results, subset_indices, model_a, model_b):
    """For each index, compute majority vote across judges."""
    majority = {}
    for idx in subset_indices:
        votes = []
        for judge, results in per_judge_results.items():
            r = results.get(idx)
            if r and r.get("real_winner") and not r.get("error"):
                votes.append(r["real_winner"])
        if not votes:
            majority[idx] = None
            continue
        from collections import Counter
        c = Counter(votes)
        top = c.most_common(1)[0]
        if top[1] > len(votes) / 2:
            majority[idx] = top[0]
        else:
            majority[idx] = "Tie"
    return majority


def compute_agreement(per_judge_results, subset_indices):
    """Fraction of indices where all available judges agree."""
    agree_count = 0
    total_valid = 0
    for idx in subset_indices:
        votes = []
        for judge, results in per_judge_results.items():
            r = results.get(idx)
            if r and r.get("real_winner") and not r.get("error"):
                votes.append(r["real_winner"])
        if len(votes) >= 2:
            total_valid += 1
            if len(set(votes)) == 1:
                agree_count += 1
    return agree_count / total_valid if total_valid > 0 else 0.0


def build_summary_row(comp_key, model_a, model_b, per_judge_results, subset_indices):
    row = {"Comparison": comp_key.replace("_vs_", " vs ").replace("_", "-")}
    for judge in ["gpt-4o-mini", "gpt-4o"]:
        col = f"{judge}_A_Win_Pct"
        if judge in per_judge_results:
            res = per_judge_results[judge]
            a_wins = sum(1 for idx in subset_indices if res.get(idx, {}).get("real_winner") == model_a)
            n_valid = sum(1 for idx in subset_indices if res.get(idx, {}).get("real_winner") is not None and not res.get(idx, {}).get("error"))
            row[col] = f"{a_wins/len(subset_indices)*100:.1f}%" if len(subset_indices) > 0 else "N/A"
        else:
            row[col] = "N/A"

    majority = compute_majority(per_judge_results, subset_indices, model_a, model_b)
    maj_a = sum(1 for idx in subset_indices if majority.get(idx) == model_a)
    row["Majority_A_Win_Pct"] = f"{maj_a/len(subset_indices)*100:.1f}%"
    row["Judge_Agreement"] = f"{compute_agreement(per_judge_results, subset_indices)*100:.1f}%"
    row["N"] = len(subset_indices)
    errors = sum(
        sum(1 for r in res.values() if r.get("error"))
        for res in per_judge_results.values()
    )
    row["Errors"] = errors
    return row


def write_robustness_summary(summary_rows, out_dir):
    fields = ["Comparison", "gpt-4o-mini_A_Win_Pct", "gpt-4o_A_Win_Pct",
              "Majority_A_Win_Pct", "Judge_Agreement", "N", "Errors"]

    csv_path = os.path.join(out_dir, "judge_robustness_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved {csv_path}")

    md_path = os.path.join(out_dir, "judge_robustness_summary.md")
    header = "| Comparison | GPT-4o-mini A Win% | GPT-4o A Win% | Majority-Judge A Win% | Judge Agreement | N | Errors |"
    sep    = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in summary_rows:
        lines.append(
            f"| {r['Comparison']} | {r['gpt-4o-mini_A_Win_Pct']} | {r['gpt-4o_A_Win_Pct']} | "
            f"{r['Majority_A_Win_Pct']} | {r['Judge_Agreement']} | {r['N']} | {r['Errors']} |"
        )
    note = "\n\n> Note: Reduced scope - only 3 comparisons, 100 instances each, 2 judges (no Claude).\n"
    with open(md_path, "w") as f:
        f.write("# Multi-Judge Robustness Summary (V2-Neutral, Reduced Scope)\n\n")
        f.write('\n'.join(lines) + "\n" + note)
    print(f"Saved {md_path}")

    tex_path = os.path.join(out_dir, "judge_robustness_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Comparison & GPT-4o-mini & GPT-4o & Majority & Agreement \\\\\n\\midrule\n")
        for r in summary_rows:
            f.write(
                f"{r['Comparison']} & {r['gpt-4o-mini_A_Win_Pct']} & {r['gpt-4o_A_Win_Pct']} & "
                f"{r['Majority_A_Win_Pct']} & {r['Judge_Agreement']} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Multi-judge robustness (V2-Neutral, 100 instances, 2 judges). N/A indicates judge unavailable.}\n")
        f.write("\\label{tab:multijudge}\n\\end{table}\n")
    print(f"Saved {tex_path}")


def main():
    with open(SUBSET_PATH) as f:
        subset_indices = set(int(l.strip()) for l in f if l.strip())
    print(f"Loaded {len(subset_indices)} subset indices (100-sample)")

    print("Loading response files...")
    responses =
    for name, path in RESPONSE_FILES.items():
        responses[name] = load_responses(path)
        print(f"  {name}: {len(responses[name])} rows")

    # Build stable swap map per index
    rng = random.Random(SEED)
    swap_maps = {}
    for model_a, model_b in COMPARISONS:
        comp_key = f"{model_a}_vs_{model_b}"
        rng2 = random.Random(SEED + hash(comp_key) % 100000)
        swap_maps[comp_key] = {idx: (rng2.random() < 0.5) for idx in subset_indices}

    all_summary_rows = []

    for model_a, model_b in COMPARISONS:
        comp_key = f"{model_a}_vs_{model_b}"
        per_judge =

        # gpt-4o-mini: reuse Task 2 results if available (filter to 100 subset)
        mini_existing = load_existing_mini_results(comp_key, subset_indices)
        if len(mini_existing) == len(subset_indices):
            print(f"  Reusing Task 2 gpt-4o-mini results for {comp_key} ({len(mini_existing)} rows from 100-subset)")
            per_judge["gpt-4o-mini"] = mini_existing
        else:
            print(f"  Task 2 results incomplete for 100-subset ({len(mini_existing)}/{len(subset_indices)}), running gpt-4o-mini")
            out_path = os.path.join(OUT_DIR, f"{comp_key}_v2_neutral_gpt-4o-mini.jsonl")
            results = run_judge_for_comparison(
                model_a, model_b, "gpt-4o-mini",
                responses[model_a], responses[model_b],
                list(subset_indices), swap_maps[comp_key], out_path
            )
            per_judge["gpt-4o-mini"] = results

        # gpt-4o
        gpt4o_path = os.path.join(OUT_DIR, f"{comp_key}_v2_neutral_gpt-4o.jsonl")
        gpt4o_results = run_judge_for_comparison(
            model_a, model_b, "gpt-4o",
            responses[model_a], responses[model_b],
            list(subset_indices), swap_maps[comp_key], gpt4o_path
        )
        per_judge["gpt-4o"] = gpt4o_results

        row = build_summary_row(comp_key, model_a, model_b, per_judge, list(subset_indices))
        all_summary_rows.append(row)

    write_robustness_summary(all_summary_rows, OUT_DIR)
    print("\n=== TASK 3 (REVISED) COMPLETE ===")
    print("Scope: 3 comparisons × 100 instances × 2 judges = 600 total evaluations")


if __name__ == "__main__":
    main()

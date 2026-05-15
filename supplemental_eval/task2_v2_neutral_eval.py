"""
TASK 2: V2-Neutral pairwise prompt sensitivity eval using GPT-4o-mini.
Resumable: skips already-evaluated indices per comparison file.
"""
import sys, os, json, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supplemental_eval.configs import RESPONSE_FILES

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "v2_neutral_gpt4o_mini")
SUBSET_PATH = os.path.join(os.path.dirname(__file__), "subset_200_seed42.txt")
os.makedirs(OUT_DIR, exist_ok=True)

COMPARISONS = [
    ("Llama-Cognitive", "Llama-Base"),
    ("Qwen-Cognitive",  "Qwen-Base"),
    ("Llama-Cognitive", "GPT-4o"),
    ("Llama-Cognitive", "Claude-Sonnet"),
    ("Qwen-Cognitive",  "GPT-4o"),
    ("Qwen-Cognitive",  "Claude-Sonnet"),
]

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
    """Extract the human-readable chat from the prompt field."""
    # Try to find seeker/supporter turns
    lines = prompt_str.split('\n')
    history_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('seeker:') or line.startswith('supporter:') or line.startswith('User:'):
            history_lines.append(line)
    if history_lines:
        return '\n'.join(history_lines)
    # Fallback: return last 800 chars of prompt
    return prompt_str[-800:]


def parse_judge_output(raw):
    """Parse JSON from judge output; returns (winner, confidence, explanation, error)."""
    # Strip markdown code blocks
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


@retry(max_attempts=3, delay=2)
def call_judge(prompt_text, system_text, judge_model):
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    import os as _os
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


def run_comparison(model_a_name, model_b_name, responses_a, responses_b, subset_indices,
                   judge_model, out_path, rng):
    # Load already done indices
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add(obj["index"])
    print(f"\n[{model_a_name} vs {model_b_name}] already done: {len(done)}, remaining: {len(subset_indices)-len(done)}")

    with open(out_path, "a") as fout:
        for idx in subset_indices:
            if idx in done:
                continue
            row_a = responses_a[idx]
            row_b = responses_b[idx]
            resp_a_text = row_a["predict"]
            resp_b_text = row_b["predict"]
            # Extract chat history
            chat_hist = extract_chat_history(row_a["prompt"])

            # Decide swap
            swap = rng.random() < 0.5
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
                raw_output = call_judge(prompt_text, V2_NEUTRAL_SYSTEM, judge_model)
                winner_raw, confidence, explanation, parse_err = parse_judge_output(raw_output)
                if parse_err:
                    # One retry with explicit JSON instruction
                    retry_prompt = prompt_text + "\n\nYou MUST respond with valid JSON only. No markdown."
                    raw_output2 = call_judge(retry_prompt, V2_NEUTRAL_SYSTEM, judge_model)
                    winner_raw, confidence, explanation, parse_err2 = parse_judge_output(raw_output2)
                    if parse_err2:
                        error = f"parse_err: {parse_err2}"
                        raw_output = raw_output2
            except Exception as e:
                error = str(e)

            # Map display winner back to real winner
            if winner_raw is not None and error is None:
                if winner_raw == "Tie":
                    real_winner = "Tie"
                elif winner_raw == "A":
                    real_winner = model_b_name if swap else model_a_name
                else:  # B
                    real_winner = model_a_name if swap else model_b_name
            else:
                real_winner = None

            record = {
                "index": idx,
                "comparison": f"{model_a_name}_vs_{model_b_name}",
                "model_a": model_a_name,
                "model_b": model_b_name,
                "response_a": resp_a_text,
                "response_b": resp_b_text,
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
            status = f"winner={real_winner}" if error is None else f"ERROR: {error[:60]}"
            print(f"  idx={idx:3d}  {status}")

    # Final count check
    results = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"  => Total rows in file: {len(results)} (expected {len(subset_indices)})")
    n_errors = sum(1 for r in results if r.get("error"))
    n_empty  = sum(1 for r in results if not r.get("real_winner"))
    print(f"  => Errors: {n_errors}, Empty real_winner: {n_empty}")
    return results


def compute_summary(results_by_comparison):
    rows = []
    for comp_name, results in results_by_comparison.items():
        model_a = results[0]["model_a"] if results else ""
        model_b = results[0]["model_b"] if results else ""
        a_wins = sum(1 for r in results if r.get("real_winner") == model_a)
        b_wins = sum(1 for r in results if r.get("real_winner") == model_b)
        ties   = sum(1 for r in results if r.get("real_winner") == "Tie")
        errors = sum(1 for r in results if r.get("error"))
        n = len(results)
        decisive = a_wins + b_wins
        a_win_rate = a_wins / n * 100 if n > 0 else 0
        decisive_a = a_wins / decisive * 100 if decisive > 0 else 0
        rows.append({
            "Comparison": comp_name,
            "A_Wins": a_wins, "B_Wins": b_wins, "Ties": ties,
            "A_Win_Rate": f"{a_win_rate:.1f}%",
            "Decisive_A_Win_Rate": f"{decisive_a:.1f}%",
            "Errors": errors, "N": n,
        })
    return rows


def write_summary(rows, out_dir):
    import csv

    # CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    fields = ["Comparison", "A_Wins", "B_Wins", "Ties", "A_Win_Rate", "Decisive_A_Win_Rate", "Errors", "N"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    # Markdown
    md_path = os.path.join(out_dir, "summary.md")
    header = "| Comparison | A Wins | B Wins | Ties | A Win Rate | Decisive A Win% | Errors | N |"
    sep    = "|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for r in rows:
        lines.append(f"| {r['Comparison']} | {r['A_Wins']} | {r['B_Wins']} | {r['Ties']} | {r['A_Win_Rate']} | {r['Decisive_A_Win_Rate']} | {r['Errors']} | {r['N']} |")
    with open(md_path, "w") as f:
        f.write("# V2-Neutral GPT-4o-mini Summary\n\n")
        f.write('\n'.join(lines) + "\n")
    print(f"Saved {md_path}")

    # LaTeX
    tex_path = os.path.join(out_dir, "summary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Comparison & A Wins & B Wins & Ties & A Win\\% & Decisive A Win\\% & Errors \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['Comparison'].replace('_',' ')} & {r['A_Wins']} & {r['B_Wins']} & {r['Ties']} & {r['A_Win_Rate']} & {r['Decisive_A_Win_Rate']} & {r['Errors']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{V2-Neutral pairwise evaluation results (GPT-4o-mini judge, 200 subset).}\n")
        f.write("\\label{tab:v2neutral_summary}\n\\end{table}\n")
    print(f"Saved {tex_path}")


def write_sensitivity_table(rows_dict, out_dir):
    """Produces prompt_sensitivity_table with V1 / V2-Neutral / V2-Depth columns."""
    V1 = {
        "Llama-Cognitive vs Llama-Base": "37.1%",
        "Qwen-Cognitive vs Qwen-Base":   "36.4%",
        "Llama-Cognitive vs GPT-4o":     "N/A",
        "Llama-Cognitive vs Claude-Sonnet": "N/A",
        "Qwen-Cognitive vs GPT-4o":      "N/A",
        "Qwen-Cognitive vs Claude-Sonnet": "N/A",
    }
    V2_DEPTH = {
        "Llama-Cognitive vs Llama-Base": "57.0%",
        "Qwen-Cognitive vs Qwen-Base":   "47.5%",
        "Llama-Cognitive vs GPT-4o":     "71.6%",
        "Llama-Cognitive vs Claude-Sonnet": "70.2%",
        "Qwen-Cognitive vs GPT-4o":      "52.1%",
        "Qwen-Cognitive vs Claude-Sonnet": "58.4%",
    }
    # Build lookup: "model_a vs model_b" -> A Win Rate from new eval
    neutral_rates = {}
    for r in rows_dict:
        key = r["Comparison"].replace("_vs_", " vs ").replace("_", "-")
        neutral_rates[key] = r["A_Win_Rate"]

    comps = list(V1.keys())

    # Markdown
    md = "# Prompt Sensitivity Table\n\n"
    md += "| Comparison | V1 Standard | V2-Neutral (new) | V2 Depth |\n"
    md += "|---|---|---|---|\n"
    for c in comps:
        nk = c
        md += f"| {c} | {V1[c]} | {neutral_rates.get(nk, 'N/A')} | {V2_DEPTH[c]} |\n"
    with open(os.path.join(out_dir, "prompt_sensitivity_table.md"), "w") as f:
        f.write(md)

    # LaTeX
    tex = "\\begin{table}[t]\n\\centering\\small\n"
    tex += "\\begin{tabular}{lccc}\n\\toprule\n"
    tex += "Comparison & V1 Standard & V2-Neutral & V2 Depth \\\\\n\\midrule\n"
    for c in comps:
        nk = c
        tex += f"{c} & {V1[c]} & {neutral_rates.get(nk, 'N/A')} & {V2_DEPTH[c]} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex += "\\caption{Prompt sensitivity: win rates under V1, V2-Neutral, and V2-Depth evaluation protocols. V2-Neutral removes the explicit encouragement of longer responses present in V2-Depth.}\n"
    tex += "\\label{tab:prompt_sensitivity}\n\\end{table}\n"
    with open(os.path.join(out_dir, "prompt_sensitivity_table.tex"), "w") as f:
        f.write(tex)

    # CSV
    import csv
    with open(os.path.join(out_dir, "prompt_sensitivity_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Comparison", "V1_Standard", "V2_Neutral", "V2_Depth"])
        for c in comps:
            w.writerow([c, V1[c], neutral_rates.get(c, "N/A"), V2_DEPTH[c]])

    print(f"Saved prompt_sensitivity_table.* to {out_dir}")


def main():
    # Load subset
    with open(SUBSET_PATH) as f:
        subset_indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(subset_indices)} subset indices")

    # Load all responses
    print("Loading response files...")
    responses = {}
    for name, path in RESPONSE_FILES.items():
        responses[name] = load_responses(path)
        print(f"  {name}: {len(responses[name])} rows")

    rng = random.Random(SEED)

    results_by_comparison = {}
    for model_a, model_b in COMPARISONS:
        comp_key = f"{model_a}_vs_{model_b}"
        fname = f"{comp_key}_v2_neutral_gpt4o_mini.jsonl"
        out_path = os.path.join(OUT_DIR, fname)
        results = run_comparison(
            model_a, model_b,
            responses[model_a], responses[model_b],
            subset_indices,
            judge_model="gpt-4o-mini",
            out_path=out_path,
            rng=rng,
        )
        results_by_comparison[comp_key] = results

    summary_rows = compute_summary(results_by_comparison)
    write_summary(summary_rows, OUT_DIR)
    write_sensitivity_table(summary_rows, OUT_DIR)
    print("\n=== TASK 2 COMPLETE ===")


if __name__ == "__main__":
    main()

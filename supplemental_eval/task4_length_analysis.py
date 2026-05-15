"""
TASK 4: Length-aware analysis for Information and Strategy scores.
No API calls. Uses existing pointwise detail files and response files.
"""
import sys, os, json, csv, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supplemental_eval.configs import RESPONSE_FILES, POINTWISE_FILES

OUT_DIR = os.path.join(os.path.dirname(__file__), "length_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {name: {"response_file": RESPONSE_FILES[name], "pointwise_file": POINTWISE_FILES[name]}
          for name in ["Llama-Base", "Llama-Cognitive", "Qwen-Base", "Qwen-Cognitive",
                       "GPT-4o", "Claude-Sonnet", "GPT-4o-mini"]}

# dim_2 = Information, dim_4 = Strategies
DIM_INFO = "dim_2"
DIM_STRAT = "dim_4"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(text):
    return len(text.split())


def spearman_r(x, y):
    """Compute Spearman rank correlation."""
    n = len(x)
    assert n == len(y)
    def rank(arr):
        sorted_i = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorted_i[j]] == arr[sorted_i[i]]:
                j += 1
            rank_val = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                r[sorted_i[k]] = rank_val
            i = j
        return r
    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(sum((rx[i] - mean_rx)**2 for i in range(n)) *
                    sum((ry[i] - mean_ry)**2 for i in range(n)))
    return num / den if den != 0 else 0.0


def ols_fit(X, y):
    """X is (n, p) including bias; returns coefficients via normal equations."""
    n, p = len(X), len(X[0])
    # XtX
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    # Solve using numpy if available, else manual (small p so fine)
    try:
        import numpy as np
        XtX_np = np.array(XtX)
        Xty_np = np.array(Xty)
        coef = np.linalg.solve(XtX_np, Xty_np)
        return coef.tolist()
    except Exception:
        # Simple Gaussian elimination
        aug = [XtX[i][:] + [Xty[i]] for i in range(p)]
        for col in range(p):
            pivot = None
            for row in range(col, p):
                if abs(aug[row][col]) > 1e-12:
                    pivot = row
                    break
            if pivot is None:
                continue
            aug[col], aug[pivot] = aug[pivot], aug[col]
            factor = aug[col][col]
            aug[col] = [x / factor for x in aug[col]]
            for row in range(p):
                if row != col and abs(aug[row][col]) > 1e-12:
                    mult = aug[row][col]
                    aug[row] = [aug[row][j] - mult * aug[col][j] for j in range(p + 1)]
        return [aug[i][p] for i in range(p)]


def run_regression(data_rows, outcome_col, out_path_txt):
    """
    OLS: outcome ~ log(length+1) + model_dummies
    data_rows: list of dicts with 'model', 'length', outcome_col
    """
    models_list = sorted(set(r["model"] for r in data_rows))
    ref_model = models_list[0]  # reference category

    X = []
    y = []
    for r in data_rows:
        length_feat = math.log(r["length"] + 1)
        dummies = [1.0 if r["model"] == m else 0.0 for m in models_list[1:]]
        X.append([1.0, length_feat] + dummies)
        y.append(float(r[outcome_col]))

    coef = ols_fit(X, y)
    feature_names = ["intercept", "log_length"] + [f"model={m}" for m in models_list[1:]]

    with open(out_path_txt, "w") as f:
        f.write(f"OLS Regression: {outcome_col} ~ log(length+1) + model_fixed_effects\n")
        f.write(f"Reference model: {ref_model}\n")
        f.write(f"N = {len(y)}\n\n")
        f.write("Coefficients:\n")
        for name, c in zip(feature_names, coef):
            f.write(f"  {name:<40s} = {c:+.4f}\n")

        # Adjusted means: predicted at mean log_length
        mean_log_len = sum(math.log(r["length"] + 1) for r in data_rows) / len(data_rows)
        f.write(f"\nAdjusted means (at mean log_length={mean_log_len:.3f}):\n")
        adjusted = {}
        # ref model
        ref_mean = coef[0] + coef[1] * mean_log_len
        adjusted[ref_model] = ref_mean
        f.write(f"  {ref_model:<30s} = {ref_mean:.4f}\n")
        for i, m in enumerate(models_list[1:]):
            am = coef[0] + coef[1] * mean_log_len + coef[2 + i]
            adjusted[m] = am
            f.write(f"  {m:<30s} = {am:.4f}\n")

    print(f"Saved regression results to {out_path_txt}")
    return coef, feature_names, adjusted


def main():
    # Load data
    all_data = []
    model_stats = {}

    for model_name, paths in MODELS.items():
        resp_rows = load_jsonl(paths["response_file"])
        pw_rows   = load_jsonl(paths["pointwise_file"])

        if len(resp_rows) != len(pw_rows):
            print(f"WARNING: {model_name} response={len(resp_rows)} vs pointwise={len(pw_rows)}")

        lengths = []
        info_scores = []
        strat_scores = []

        for i, (resp, pw) in enumerate(zip(resp_rows, pw_rows)):
            text = resp["predict"]
            wc = word_count(text)
            info = pw["eval_scores"].get(DIM_INFO, None)
            strat = pw["eval_scores"].get(DIM_STRAT, None)
            if info is None or strat is None:
                continue
            lengths.append(wc)
            info_scores.append(float(info))
            strat_scores.append(float(strat))
            all_data.append({
                "model": model_name,
                "length": wc,
                "information": float(info),
                "strategy": float(strat),
            })

        n = len(lengths)
        avg_len    = sum(lengths) / n
        sorted_len = sorted(lengths)
        med_len    = sorted_len[n // 2] if n % 2 == 1 else (sorted_len[n//2-1] + sorted_len[n//2]) / 2
        avg_info   = sum(info_scores) / n
        avg_strat  = sum(strat_scores) / n
        info_per100  = avg_info / avg_len * 100 if avg_len > 0 else 0
        strat_per100 = avg_strat / avg_len * 100 if avg_len > 0 else 0
        spear_info  = spearman_r(lengths, info_scores)
        spear_strat = spearman_r(lengths, strat_scores)

        model_stats[model_name] = {
            "N": n,
            "Avg_Length": round(avg_len, 1),
            "Median_Length": round(med_len, 1),
            "Avg_Information": round(avg_info, 4),
            "Avg_Strategy": round(avg_strat, 4),
            "Info_per_100w": round(info_per100, 4),
            "Strat_per_100w": round(strat_per100, 4),
            "Spearman_Info_vs_Length": round(spear_info, 4),
            "Spearman_Strat_vs_Length": round(spear_strat, 4),
        }
        print(f"{model_name}: N={n}, avg_len={avg_len:.1f}, avg_info={avg_info:.3f}, avg_strat={avg_strat:.3f}, "
              f"spear_info={spear_info:.3f}, spear_strat={spear_strat:.3f}")

    # Write summary CSV
    fields = ["Model", "N", "Avg_Length", "Median_Length", "Avg_Information", "Avg_Strategy",
              "Info_per_100w", "Strat_per_100w", "Spearman_Info_vs_Length", "Spearman_Strat_vs_Length"]
    csv_path = os.path.join(OUT_DIR, "length_analysis_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for model_name in MODELS:
            s = model_stats[model_name]
            w.writerow([model_name] + [s[k] for k in fields[1:]])
    print(f"Saved {csv_path}")

    # Markdown
    md_path = os.path.join(OUT_DIR, "length_analysis_summary.md")
    header = "| Model | N | Avg Length | Median Length | Avg Info | Avg Strat | Info/100w | Strat/100w | Spearman(Info,Len) | Spearman(Strat,Len) |"
    sep    = "|---|---|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for model_name in MODELS:
        s = model_stats[model_name]
        lines.append(
            f"| {model_name} | {s['N']} | {s['Avg_Length']} | {s['Median_Length']} | "
            f"{s['Avg_Information']} | {s['Avg_Strategy']} | {s['Info_per_100w']} | {s['Strat_per_100w']} | "
            f"{s['Spearman_Info_vs_Length']} | {s['Spearman_Strat_vs_Length']} |"
        )
    with open(md_path, "w") as f:
        f.write("# Length-Aware Analysis: Information and Strategy Scores\n\n")
        f.write('\n'.join(lines) + "\n")
    print(f"Saved {md_path}")

    # LaTeX
    tex_path = os.path.join(OUT_DIR, "length_analysis_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lccccccccc}\n\\toprule\n")
        f.write("Model & N & Avg Len & Med Len & Avg Info & Avg Strat & Info/100w & Strat/100w & $\\rho$(Info,Len) & $\\rho$(Strat,Len) \\\\\n\\midrule\n")
        for model_name in MODELS:
            s = model_stats[model_name]
            f.write(f"{model_name} & {s['N']} & {s['Avg_Length']} & {s['Median_Length']} & "
                    f"{s['Avg_Information']} & {s['Avg_Strategy']} & {s['Info_per_100w']} & {s['Strat_per_100w']} & "
                    f"{s['Spearman_Info_vs_Length']} & {s['Spearman_Strat_vs_Length']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Length-aware analysis. Info/100w and Strat/100w normalize scores by response length. "
                "Spearman $\\rho$ measures length--score correlation within each model.}\n")
        f.write("\\label{tab:length_analysis}\n\\end{table*}\n")
    print(f"Saved {tex_path}")

    # OLS Regressions
    info_coef, info_features, info_adjusted = run_regression(
        all_data, "information",
        os.path.join(OUT_DIR, "regression_information.txt")
    )
    strat_coef, strat_features, strat_adjusted = run_regression(
        all_data, "strategy",
        os.path.join(OUT_DIR, "regression_strategy.txt")
    )

    # Write paper-ready conclusion
    # Check if cognitive models retain advantage after length control
    cog_models = ["Llama-Cognitive", "Qwen-Cognitive"]
    commercial_models = ["GPT-4o", "Claude-Sonnet", "GPT-4o-mini"]

    avg_cog_info_raw  = sum(model_stats[m]["Avg_Information"] for m in cog_models) / len(cog_models)
    avg_com_info_raw  = sum(model_stats[m]["Avg_Information"] for m in commercial_models) / len(commercial_models)
    avg_cog_info_adj  = sum(info_adjusted.get(m, 0) for m in cog_models) / len(cog_models)
    avg_com_info_adj  = sum(info_adjusted.get(m, 0) for m in commercial_models) / len(commercial_models)

    raw_gap = avg_cog_info_raw - avg_com_info_raw
    adj_gap = avg_cog_info_adj - avg_com_info_adj
    shrinkage = (raw_gap - adj_gap) / abs(raw_gap) * 100 if raw_gap != 0 else 0

    conclusion_path = os.path.join(OUT_DIR, "paper_conclusion.txt")
    with open(conclusion_path, "w") as f:
        f.write("=== Length-Aware Analysis: Paper-Ready Conclusion ===\n\n")
        f.write(f"Raw gap (Cognitive - Commercial) for Information: {raw_gap:+.4f}\n")
        f.write(f"Adjusted gap (Cognitive - Commercial) for Information: {adj_gap:+.4f}\n")
        f.write(f"Shrinkage after length control: {shrinkage:.1f}%\n\n")

        if adj_gap > 0.05:
            conclusion = (
                "The cognitive-aligned models retain stronger adjusted Information and Strategy scores "
                "after controlling for response length, suggesting that the advantage is not solely an "
                "artifact of longer responses."
            )
        else:
            conclusion = (
                "The advantage in Information and Strategy scores becomes smaller after controlling for "
                "response length, indicating that these scores partially reflect response elaboration. "
                "We therefore interpret them as task-specific measures of supportive development rather "
                "than length-independent capability measures."
            )
        f.write("Suggested paper conclusion:\n")
        f.write(conclusion + "\n")
        print(f"\nPaper conclusion: {conclusion}")

    print(f"Saved {conclusion_path}")
    print("\n=== TASK 4 COMPLETE ===")


if __name__ == "__main__":
    main()

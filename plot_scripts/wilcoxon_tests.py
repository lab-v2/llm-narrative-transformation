"""
Wilcoxon signed-rank tests: Abduction vs each baseline (per LLM x direction x metric).

Paired, two-sided tests (same test stories transformed by both methods). For each
(metric, direction, model, baseline) we pair per-story values and report:
  - median Delta = median(abduction_i - baseline_i)   (raw sign; interpret w/ metric direction)
  - favors        = which method is ahead (metric-aware), or n.s.
  - W, p_raw      = scipy.stats.wilcoxon (two-sided)
  - p_holm        = Holm step-down, applied WITHIN each (model x direction x metric) family
                    across the 4 baseline comparisons
  - effect_r      = matched-pairs rank-biserial correlation, ORIENTED so + = abduction better
  - n             = number of non-zero paired differences

Metrics & "better" direction:
  diagnosis : median cultural rating   (forward: lower better; inverse: higher better)
  kl        : KL(transformed||original) (lower better, both directions)
  bertscore : BERTScore F1              (higher better, both directions)

Note on interpretation: a non-significant two-sided test at n~28 is *absence of evidence of a
difference*, NOT evidence of equivalence. Report as "no significant difference detected."

Output: <output-dir>/wilcoxon_all.csv (long) + wilcoxon_{forward,inverse}.csv (paper layout).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

TECHS = [("zeroshot", "Zero-Shot"), ("fewshot", "Few-Shot"),
         ("cot", "CoT"), ("iterative", "Self-Refine")]
DEFAULT_MODELS = [
    "gpt-4o", "gpt-5.2", "xai/grok-4-fast-reasoning", "claude-sonnet-4-5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0", "bedrock/us.deepseek.r1-v1:0",
]
SHORT = {
    "gpt-4o": "GPT-4o", "gpt-5.2": "GPT-5.2", "xai/grok-4-fast-reasoning": "Grok-4",
    "claude-sonnet-4-5": "Claude-4.5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": "Llama-4",
    "bedrock/us.deepseek.r1-v1:0": "DeepSeek-R1",
}


def sanitize(m):
    return m.replace("/", "-").replace(":", "-")


def short(m):
    return SHORT.get(m, sanitize(m))


# ---------- paired-value loaders: return (abduction[], baseline[]) aligned by story ----------

def pairs_diagnosis(analysis, tech, model, problem):
    f = Path(analysis) / f"cross_model_summary_{tech}" / problem / "model_comparison_median.csv"
    if not f.exists():
        return [], []
    d = pd.read_csv(f)
    a, b = f"{model}_transformed", f"{model}_baseline"
    if a not in d.columns or b not in d.columns:
        return [], []
    d = d[[a, b]].dropna()
    return d[a].tolist(), d[b].tolist()


def pairs_kl(analysis, tech, model, problem):
    f = Path(analysis) / f"divergences_{tech}" / sanitize(model) / problem / "kl_divergence_transformed_original.csv"
    opt = Path(analysis) / "phase2" / sanitize(model) / problem / "optimal_iterations.csv"
    if not f.exists() or not opt.exists():
        return [], []
    dk, do = pd.read_csv(f), pd.read_csv(opt)
    opt_map = dict(zip(do["story_name"], do["optimal_iteration"]))
    abd, base = [], []
    for _, r in dk.iterrows():
        it = opt_map.get(r["story_name"])
        if pd.isna(it):
            continue
        c = f"abductive_iter_{int(it)}_kl"
        if c in r.index and pd.notna(r[c]) and pd.notna(r["baseline_kl"]):
            abd.append(r[c]); base.append(r["baseline_kl"])
    return abd, base


def pairs_bertscore(analysis, tech, model, problem):
    f = Path(analysis) / f"bert_scores_csv_{tech}" / f"bertscore_results_{sanitize(model)}_{problem}.csv"
    if not f.exists():
        return [], []
    d = pd.read_csv(f)
    d = d[["abduction_bertscore_f1", "baseline_bertscore_f1"]].apply(pd.to_numeric, errors="coerce").dropna()
    return d["abduction_bertscore_f1"].tolist(), d["baseline_bertscore_f1"].tolist()


LOADERS = {"diagnosis": pairs_diagnosis, "kl": pairs_kl, "bertscore": pairs_bertscore}


def better_is_higher(metric, problem):
    if metric == "bertscore":
        return True
    if metric == "kl":
        return False
    return problem == "inverse"  # diagnosis: inverse -> higher better, forward -> lower better


# ---------- stats ----------

def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation on non-zero diffs (sign follows diffs)."""
    d = np.array([x for x in diffs if x != 0], dtype=float)
    n = len(d)
    if n == 0:
        return np.nan, 0
    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = n * (n + 1) / 2
    return (w_pos - w_neg) / total, n


def holm(pvals):
    """Holm step-down adjusted p-values (NaNs passed through, excluded from the family)."""
    idx = [i for i, p in enumerate(pvals) if not (p is None or (isinstance(p, float) and np.isnan(p)))]
    out = [np.nan] * len(pvals)
    if not idx:
        return out
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run():
    ap = argparse.ArgumentParser(description="Wilcoxon signed-rank: abduction vs each baseline")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--metrics", nargs="+", default=["diagnosis", "kl", "bertscore"],
                    choices=["diagnosis", "kl", "bertscore"])
    ap.add_argument("--analysis-dir", default="output_analysis")
    ap.add_argument("--output-dir", default="output_analysis/tables_baselines")
    ap.add_argument("--run-batch", default="main",
                    help="Provenance tag stamped on every row of wilcoxon_paper_export.csv")
    args = ap.parse_args()

    rows = []  # long format
    # Holm is applied within each (model, direction, metric) family across the 4 baselines.
    for metric in args.metrics:
        for problem in ["forward", "inverse"]:
            hi = better_is_higher(metric, problem)
            sign = 1.0 if hi else -1.0
            for model in args.models:
                fam = []  # (row_index_placeholder, p_raw) for Holm within this family
                fam_rows = []
                for tech, lbl in TECHS:
                    abd, base = LOADERS[metric](args.analysis_dir, tech, model, problem)
                    if len(abd) < 3:
                        rec = dict(metric=metric, direction=problem, model=short(model), baseline=lbl,
                                   n=len(abd), median_abd=np.nan, median_base=np.nan,
                                   median_diff=np.nan, W=np.nan, p_raw=np.nan, effect_r=np.nan)
                        fam_rows.append(rec); fam.append(np.nan); continue
                    abd = np.array(abd, float); base = np.array(base, float)
                    diffs = abd - base
                    r_rb, n_nz = rank_biserial(diffs)
                    try:
                        W, p = wilcoxon(abd, base, alternative="two-sided", zero_method="wilcox")
                    except ValueError:
                        W, p = np.nan, np.nan
                    rec = dict(metric=metric, direction=problem, model=short(model), baseline=lbl,
                               n=n_nz, median_abd=round(float(np.median(abd)), 4),
                               median_base=round(float(np.median(base)), 4),
                               median_diff=round(float(np.median(diffs)), 4),
                               W=(np.nan if np.isnan(W) else float(W)),
                               p_raw=(np.nan if (p is None or np.isnan(p)) else float(p)),
                               effect_r=(np.nan if np.isnan(r_rb) else round(float(r_rb * sign), 3)))
                    fam_rows.append(rec); fam.append(rec["p_raw"])
                p_holm = holm(fam)
                for rec, ph in zip(fam_rows, p_holm):
                    rec["p_holm"] = (np.nan if ph is None or np.isnan(ph) else round(float(ph), 4))
                    rec["p_raw"] = (np.nan if np.isnan(rec["p_raw"]) else round(rec["p_raw"], 4))
                    rec["sig"] = stars(rec["p_holm"])
                    # Direction from the signed-rank effect size (already oriented so + = abduction
                    # better). NOT from median_diff: diagnosis scores are discrete (1-5), so the
                    # median paired difference is frequently 0.0 even when the test is significant.
                    if rec["sig"] == "n.s." or rec["effect_r"] is None or np.isnan(rec["effect_r"]):
                        rec["favors"] = "n.s."
                    else:
                        rec["favors"] = "Abduction" if rec["effect_r"] > 0 else rec["baseline"]
                    rows.append(rec)

    df = pd.DataFrame(rows, columns=["metric", "direction", "model", "baseline", "n",
                                     "median_abd", "median_base", "median_diff",
                                     "W", "p_raw", "p_holm", "effect_r", "favors", "sig"])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "wilcoxon_all.csv", index=False)
    print(f"✓ saved {out/'wilcoxon_all.csv'}  ({len(df)} tests)")

    # ---- paper export: one row per (direction, model, baseline, metric) ----
    baseline_key = {"Zero-Shot": "zero_shot", "Few-Shot": "few_shot",
                    "CoT": "cot", "Self-Refine": "self_refine"}
    exp = df.copy()
    exp["baseline"] = exp["baseline"].map(baseline_key)          # snake_case keys
    exp = exp.rename(columns={"sig": "sig_marker", "n": "n_pairs"})
    exp["run_batch"] = args.run_batch
    # effect_r already oriented so + = abduction better; favors derived from it.
    cols = ["direction", "model", "baseline", "metric", "median_abd", "median_base",
            "median_diff", "effect_r", "p_raw", "p_holm", "sig_marker", "n_pairs",
            "run_batch", "favors"]
    exp = exp[cols].sort_values(["direction", "model", "baseline", "metric"])
    exp.to_csv(out / "wilcoxon_paper_export.csv", index=False)
    print(f"✓ saved {out/'wilcoxon_paper_export.csv'}  ({len(exp)} rows, run_batch={args.run_batch})")

    # ---- Pareto cross-reference: cells where a baseline significantly beats abduction on
    #      diagnosis (shift), with KL & BERTScore for the SAME cell (the fidelity price paid) ----
    diag = df[(df.metric == "diagnosis") & (df.sig != "n.s.") & (df.favors != "Abduction")]
    prows = []
    for _, r in diag.iterrows():
        rec = {"direction": r.direction, "model": r.model, "baseline": r.baseline,
               "diag_favors": r.favors, "diag_effect_r": r.effect_r,
               "diag_p_holm": r.p_holm, "diag_sig": r.sig}
        for metric, pfx in [("kl", "kl"), ("bertscore", "bert")]:
            mm = df[(df.metric == metric) & (df.direction == r.direction)
                    & (df.model == r.model) & (df.baseline == r.baseline)]
            if len(mm):
                mm = mm.iloc[0]
                rec[f"{pfx}_favors"] = mm.favors
                rec[f"{pfx}_effect_r"] = mm.effect_r
                rec[f"{pfx}_p_holm"] = mm.p_holm
                rec[f"{pfx}_sig"] = mm.sig
        prows.append(rec)
    pareto = pd.DataFrame(prows).sort_values(["direction", "model", "baseline"]) if prows else pd.DataFrame(prows)
    pareto.to_csv(out / "pareto_crossref.csv", index=False)
    print(f"✓ saved {out/'pareto_crossref.csv'}  ({len(pareto)} cells where a baseline out-shifts abduction)")

    # paper-layout tables: one per direction, rows = model x baseline,
    # cols = <metric>_medianDiff / _p_holm / _sig
    for problem in ["forward", "inverse"]:
        sub = df[df["direction"] == problem]
        recs = []
        for model in [short(m) for m in args.models]:
            for _, lbl in TECHS:
                r = {"Model": model, "Baseline": lbl}
                for metric in ["diagnosis", "kl", "bertscore"]:
                    m = sub[(sub["model"] == model) & (sub["baseline"] == lbl) & (sub["metric"] == metric)]
                    if len(m):
                        row = m.iloc[0]
                        r[f"{metric}_dDiff"] = row["median_diff"]
                        r[f"{metric}_pholm"] = row["p_holm"]
                        r[f"{metric}_sig"] = row["sig"]
                recs.append(r)
        pd.DataFrame(recs).to_csv(out / f"wilcoxon_{problem}.csv", index=False)
        print(f"✓ saved {out/f'wilcoxon_{problem}.csv'}")


if __name__ == "__main__":
    run()

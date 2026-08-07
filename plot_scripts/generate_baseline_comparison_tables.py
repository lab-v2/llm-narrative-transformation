"""
Baseline Comparison Tables

Builds combined per-direction tables comparing each baseline prompting technique
(zero-shot, CoT, few-shot, self-refine) against the abduction method, for three
metrics:

  - improvement   : average gap-to-target closed on the MEDIAN cultural rating (%)
  - kl            : median KL(transformed || original)  (lower = better fidelity)
  - bertscore     : median BERTScore F1                 (higher = better fidelity)

Each table: rows = models, columns = <techniques...> + Abduction. The abduction
column is identical across techniques (it's the method, independent of baseline).

Inputs (produced by the per-baseline eval pipeline; see README
"Per-baseline evaluation"):
  improvement : <analysis>/cross_model_summary_<tech>/<problem>/model_comparison_median.csv
  kl          : <analysis>/divergences_<tech>/<model>/<problem>/kl_divergence_transformed_original.csv
                + <analysis>/phase2/<model>/<problem>/optimal_iterations.csv
  bertscore   : <analysis>/bert_scores_csv_<tech>/bertscore_results_<model>_<problem>.csv

Output: <output-dir>/{improvement,kl_median,bertscore_f1_median}_{forward,inverse}.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TECHS = [("zeroshot", "Zero-Shot"), ("cot", "CoT"),
         ("fewshot", "Few-Shot"), ("iterative", "Self-Refine")]

DEFAULT_MODELS = [
    "gpt-4o", "gpt-5.2", "xai/grok-4-fast-reasoning", "claude-sonnet-4-5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0", "bedrock/us.deepseek.r1-v1:0",
]

SHORT = {
    "gpt-4o": "GPT-4o", "gpt-5.2": "GPT-5.2",
    "xai/grok-4-fast-reasoning": "Grok-4", "claude-sonnet-4-5": "Claude-4.5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": "Llama-4",
    "bedrock/us.deepseek.r1-v1:0": "DeepSeek-R1",
}


def sanitize(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def short_name(model: str) -> str:
    return SHORT.get(model, sanitize(model))


# ---------- improvement (% of gap-to-target closed, on median rating) ----------

def _improvement(df, model, col, problem):
    o, v = f"{model}_original", f"{model}_{col}"
    if o not in df.columns or v not in df.columns:
        return None
    vals = []
    for _, r in df.iterrows():
        orig, val = r[o], r[v]
        if pd.isna(orig) or pd.isna(val):
            continue
        gap = (orig - 1.0) if problem == "forward" else (5.0 - orig)
        if gap > 0:
            vals.append(((orig - val) if problem == "forward" else (val - orig)) / gap * 100)
    return round(float(np.mean(vals)), 2) if vals else None


def improvement_cell(analysis, tech, model, problem, which):
    # which: "baseline" (technique) or "transformed" (abduction)
    f = Path(analysis) / f"cross_model_summary_{tech}" / problem / "model_comparison_median.csv"
    if not f.exists():
        return None
    return _improvement(pd.read_csv(f), model, which, problem)


# ---------- KL median ----------

def kl_baseline_median(analysis, tech, model, problem):
    f = Path(analysis) / f"divergences_{tech}" / sanitize(model) / problem / "kl_divergence_transformed_original.csv"
    if not f.exists():
        return None
    v = pd.read_csv(f)["baseline_kl"].dropna()
    return round(float(v.median()), 4) if len(v) else None


def kl_abduction_median(analysis, model, problem):
    f = Path(analysis) / "divergences_zeroshot" / sanitize(model) / problem / "kl_divergence_transformed_original.csv"
    opt = Path(analysis) / "phase2" / sanitize(model) / problem / "optimal_iterations.csv"
    if not f.exists() or not opt.exists():
        return None
    dk, do = pd.read_csv(f), pd.read_csv(opt)
    vals = []
    for _, r in dk.iterrows():
        orow = do[do["story_name"] == r["story_name"]]
        if len(orow) and pd.notna(orow["optimal_iteration"].values[0]):
            it = int(orow["optimal_iteration"].values[0])
            c = f"abductive_iter_{it}_kl"
            if c in r.index and pd.notna(r[c]):
                vals.append(r[c])
    return round(float(np.median(vals)), 4) if vals else None


# ---------- BERTScore F1 median ----------

def bert_median(analysis, tech, model, problem, col):
    f = Path(analysis) / f"bert_scores_csv_{tech}" / f"bertscore_results_{sanitize(model)}_{problem}.csv"
    if not f.exists():
        return None
    v = pd.to_numeric(pd.read_csv(f)[col], errors="coerce").dropna()
    return round(float(v.median()), 4) if len(v) else None


# ---------- table assembly ----------

def build_table(metric, problem, models, analysis):
    rows = []
    for m in models:
        row = {"Model": short_name(m)}
        for tech, lbl in TECHS:
            if metric == "improvement":
                row[lbl] = improvement_cell(analysis, tech, m, problem, "baseline")
            elif metric == "kl":
                row[lbl] = kl_baseline_median(analysis, tech, m, problem)
            else:  # bertscore
                row[lbl] = bert_median(analysis, tech, m, problem, "baseline_bertscore_f1")
        if metric == "improvement":
            row["Abduction"] = improvement_cell(analysis, "zeroshot", m, problem, "transformed")
        elif metric == "kl":
            row["Abduction"] = kl_abduction_median(analysis, m, problem)
        else:
            row["Abduction"] = bert_median(analysis, "zeroshot", m, problem, "abduction_bertscore_f1")
        rows.append(row)
    return pd.DataFrame(rows)


METRIC_INFO = {
    "improvement": ("improvement", "Average Improvement % (median rating, gap-to-target closed; higher=better)"),
    "kl": ("kl_median", "KL median (lower=better fidelity)"),
    "bertscore": ("bertscore_f1_median", "BERTScore F1 median (higher=better fidelity)"),
}


def main():
    ap = argparse.ArgumentParser(description="Build combined baseline-vs-abduction comparison tables")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names (full LiteLLM strings)")
    ap.add_argument("--metrics", nargs="+", default=["improvement", "kl", "bertscore"],
                    choices=["improvement", "kl", "bertscore"])
    ap.add_argument("--analysis-dir", default="output_analysis",
                    help="Root holding divergences_<tech>/, bert_scores_csv_<tech>/, "
                         "cross_model_summary_<tech>/, phase2/ (default: output_analysis)")
    ap.add_argument("--output-dir", default="output_analysis/tables_baselines")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        name, title = METRIC_INFO[metric]
        for problem in ["forward", "inverse"]:
            df = build_table(metric, problem, args.models, args.analysis_dir)
            f = outdir / f"{name}_{problem}.csv"
            df.to_csv(f, index=False)
            print(f"\n===== {title} — {problem.upper()} =====")
            print(df.to_string(index=False))
            print(f"(saved {f})")


if __name__ == "__main__":
    main()

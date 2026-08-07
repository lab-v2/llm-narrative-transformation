"""
Combined 5-box baseline comparison plots (for the paper).

Each subfigure shows FIVE boxes for one model + direction:
    ZS (zero-shot), FS (few-shot), CoT, SR (self-refine), and Abduction-Guided.

Two metrics:
  - kl        : KL(transformed || original)  (lower = better fidelity), y 0..max-y-tick
  - bertscore : BERTScore F1                 (higher = better fidelity), y y-min..y-max

Reads the per-technique eval outputs (see README "Per-baseline evaluation"):
  kl        : <analysis>/divergences_<tech>/<model>/<problem>/kl_divergence_transformed_original.csv
              + <analysis>/phase2/<model>/<problem>/optimal_iterations.csv   (for the abduction box)
  bertscore : <analysis>/bert_scores_csv_<tech>/bertscore_results_<model>_<problem>.csv

Output: <output-dir>/<metric>_<problem>_<sanitized-model>.png   (default output-dir: figures-paper1)
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Baseline techniques in display order + their abbreviated tick labels.
BASELINES = [("zeroshot", "ZS"), ("fewshot", "FS"), ("cot", "CoT"), ("iterative", "SR")]
ABD_LABEL = "Abduction-Guided"

COLOR_BASELINE = "#2ECC71"   # green
COLOR_ABDUCTION = "#E74C3C"  # red
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
BOX_ALPHA = 0.7


def sanitize(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


# ---------- data loading ----------

def kl_baseline_vals(analysis, tech, model, problem):
    f = Path(analysis) / f"divergences_{tech}" / sanitize(model) / problem / "kl_divergence_transformed_original.csv"
    if not f.exists():
        return []
    return pd.read_csv(f)["baseline_kl"].dropna().tolist()


def kl_abduction_vals(analysis, model, problem):
    f = Path(analysis) / "divergences_zeroshot" / sanitize(model) / problem / "kl_divergence_transformed_original.csv"
    opt = Path(analysis) / "phase2" / sanitize(model) / problem / "optimal_iterations.csv"
    if not f.exists() or not opt.exists():
        return []
    dk, do = pd.read_csv(f), pd.read_csv(opt)
    vals = []
    for _, r in dk.iterrows():
        orow = do[do["story_name"] == r["story_name"]]
        if len(orow) and pd.notna(orow["optimal_iteration"].values[0]):
            it = int(orow["optimal_iteration"].values[0])
            c = f"abductive_iter_{it}_kl"
            if c in r.index and pd.notna(r[c]):
                vals.append(r[c])
    return vals


def bert_vals(analysis, tech, model, problem, col):
    f = Path(analysis) / f"bert_scores_csv_{tech}" / f"bertscore_results_{sanitize(model)}_{problem}.csv"
    if not f.exists():
        return []
    return pd.to_numeric(pd.read_csv(f)[col], errors="coerce").dropna().tolist()


def collect(metric, analysis, model, problem):
    """Return (labels, data) for the 5 boxes: ZS, FS, CoT, SR, Abduction."""
    data, labels = [], []
    for tech, abbr in BASELINES:
        if metric == "kl":
            data.append(kl_baseline_vals(analysis, tech, model, problem))
        else:
            data.append(bert_vals(analysis, tech, model, problem, "baseline_bertscore_f1"))
        labels.append(abbr)
    if metric == "kl":
        data.append(kl_abduction_vals(analysis, model, problem))
    else:
        data.append(bert_vals(analysis, "zeroshot", model, problem, "abduction_bertscore_f1"))
    labels.append(ABD_LABEL)
    return labels, data


# ---------- plotting ----------

def make_plot(metric, model, problem, analysis, output_dir, max_y_tick, y_min, y_max):
    labels, data = collect(metric, analysis, model, problem)
    n_missing = sum(1 for d in data if not d)
    if n_missing:
        logger.warning(f"{model}/{problem}/{metric}: {n_missing} of 5 boxes have no data")

    ylabel = "KL(transformed || original)" if metric == "kl" else "BERTScore F1"
    colors = [COLOR_BASELINE] * len(BASELINES) + [COLOR_ABDUCTION]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(alpha=BOX_ALPHA, linewidth=1.5),
        whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.5),
    )
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        bp["fliers"][i].set_markerfacecolor(color)
        bp["fliers"][i].set_markeredgecolor(color)

    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    if metric == "kl":
        ax.set_ylim(0, max_y_tick)
    else:
        ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric_name = "kl_divergence" if metric == "kl" else "bertscore_f1"
    output_file = out / f"{metric_name}_{problem}_{sanitize(model)}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_file}")
    return output_file


def main():
    ap = argparse.ArgumentParser(description="5-box baseline-vs-abduction plots for the paper")
    ap.add_argument("--metric", choices=["kl", "bertscore"], required=True)
    ap.add_argument("--model", required=True, help="Model name (full LiteLLM string)")
    ap.add_argument("--problem", choices=["forward", "inverse"], required=True)
    ap.add_argument("--analysis-dir", default="output_analysis")
    ap.add_argument("--output-dir", default="figures-paper1")
    ap.add_argument("--max-y-tick", type=float, default=5.0, help="KL y-axis max (default 5.0)")
    ap.add_argument("--y-min", type=float, default=0.75, help="BERTScore y-min (default 0.75)")
    ap.add_argument("--y-max", type=float, default=1.01, help="BERTScore y-max (default 1.01)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s - %(message)s")
    make_plot(args.metric, args.model, args.problem, args.analysis_dir,
              args.output_dir, args.max_y_tick, args.y_min, args.y_max)


if __name__ == "__main__":
    main()

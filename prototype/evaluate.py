"""
Compare the LLM-free models with the LLM survey on a set of stories (the test set).

Reference: the LLM survey ratings (survey_story.py JSONs). Each model: a predictions
folder from predict_m1/m2/m3.py. Every model must cover exactly the surveyed stories.

Tables (report.txt + CSVs in --output-dir):
    overall.csv              per model: QWK [bootstrap 95% range], CV QWK, balanced acc, exact,
                             within +-1, 3-level QWK / balanced acc (means over questions)
    timing.csv               LLM survey vs each model: time per story (end to end), embed /
                             predict split, cost, model size, speed-up
    per_question.csv         QWK and balanced acc per question and model, test rating counts
    by_group.csv             overall metrics for individualistic vs collectivistic stories
    per_story_predictions.csv  LLM rating and every model's prediction per story and question
    report.txt               all of the above + paired differences vs the first model,
                             the stories each model gets most wrong, confusion matrices
    summary.json             everything above in one machine-readable file (for make_dashboard.py)
    evaluate.log             everything printed to the terminal

Bootstrap: resample the stories with replacement (--bootstrap times); the 95% range is the
2.5-97.5 percentile of the mean-over-questions QWK. Differences between models use the same
resamples (paired).

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/evaluate.py \
        --survey-dir output_test/prototype/survey/gpt-6-sol/test \
        --predictions M1=output_test/prototype/predictions/m1_s400/test \
                      M2=output_test/prototype/predictions/m2_h32-16/test \
                      M3=output_test/prototype/predictions/m3_h32-16/test \
                      M3-nearest=output_test/prototype/predictions/m3_nearest/test \
        --output-dir output_test/prototype/evaluation/test
"""

import argparse
import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from train_m1 import REPO_ROOT, SEED, answer_values, metric_names, question_metrics, setup_file_logging

logger = logging.getLogger(__name__)

METRIC_LABELS = {"qwk": "QWK", "balanced_acc": "bal_acc", "exact_acc": "exact", "within1_acc": "within1",
                 "qwk_3level": "QWK3", "balanced_acc_3level": "bal_acc3"}


def load_folder(folder: Path) -> dict:
    files = sorted(folder.glob("*.json"))
    if not files:
        sys.exit(f"No JSON files in {folder}")
    out = {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            out[p.stem] = json.load(f)
    return out


def ratings_matrix(results: dict, names: list, question_ids: list, source: str, values: list) -> np.ndarray:
    """-> (N, Q) class = index into values ([1..5], or [0, 1] for yes/no); stops on missing/failed ratings."""
    Y = np.zeros((len(names), len(question_ids)), dtype=np.int64)
    for i, n in enumerate(names):
        r = results[n]
        ratings = r.get("ratings") or {}
        if r.get("n_failed", 1) != 0 or any(ratings.get(q) not in values for q in question_ids):
            sys.exit(f"{source}: {n} has missing or failed ratings (expected values {values})")
        Y[i] = [values.index(ratings[q]) for q in question_ids]
    return Y


def probs_matrix(results: dict, names: list, question_ids: list) -> np.ndarray:
    """-> (N, Q, C) predicted probabilities (used for the yes/no AUC)."""
    return np.array([[results[n]["probabilities"][q] for q in question_ids] for n in names], dtype=float)


def qwk(t: np.ndarray, p: np.ndarray, n: int) -> float:
    """Quadratic weighted kappa (same as sklearn cohen_kappa_score(weights='quadratic'); 0 if undefined).
    With n = 2 this is plain Cohen's kappa."""
    O = np.bincount(t * n + p, minlength=n ** 2).reshape(n, n).astype(float)
    W = np.subtract.outer(np.arange(n), np.arange(n)) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else 0.0


def mean_metrics(T: np.ndarray, P: np.ndarray, n: int, probs: np.ndarray = None):
    """-> per question (Q, M) and mean over questions (M,). probs (N, Q, C) only for the yes/no AUC."""
    per_q = np.array([question_metrics(T[:, q], P[:, q], n, None if probs is None else probs[:, q])
                      for q in range(T.shape[1])])
    return per_q, np.nanmean(per_q, axis=0)


def cv_qwk(pred: dict, scale: str = "1-5"):
    """CV QWK of the model that made these predictions (from its *_final_config.json), if available and
    trained on the same answer scale as the reference survey."""
    model = next(iter(pred.values()))["model"]
    model_file = Path(model["file"])
    config_file = model_file.with_name(model_file.stem + "_config.json")
    if not config_file.exists():
        return None
    with open(config_file, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("answer_scale", "1-5") != scale:
        return None
    key = "cv_mean_nearest_prototype" if model.get("readout") == "nearest" else "cv_mean"
    return cfg.get(key, {}).get("qwk")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM-free models with the LLM survey")
    parser.add_argument("--survey-dir", type=str, required=True, help="LLM survey JSONs (the reference ratings)")
    parser.add_argument("--predictions", type=str, nargs="+", required=True,
                        help="NAME=FOLDER pairs of prediction JSONs; the first is the reference for paired differences")
    parser.add_argument("--split-csv", type=str, default=str(REPO_ROOT / "prototype" / "split.csv"),
                        help="Story -> individualistic/collectivistic")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (default: 2000)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, "evaluate.log")

    # ---------- Load ----------
    survey = load_folder(Path(args.survey_dir))
    names = sorted(survey)
    question_ids = sorted(next(iter(survey.values()))["ratings"], key=int)
    scale = next(iter(survey.values())).get("answer_scale", "1-5")
    vals = answer_values(scale)
    n_cls = len(vals)
    names_m = metric_names(n_cls)
    binary = n_cls == 2
    T = ratings_matrix(survey, names, question_ids, "survey", vals)
    N, Q = T.shape

    components = {}
    qfile = next(iter(survey.values())).get("questions_file")
    if qfile and Path(qfile).exists():
        with open(qfile, encoding="utf-8") as f:
            components = {str(q["id"]): q["component"] for q in json.load(f)["questions"]}

    with open(args.split_csv, encoding="utf-8") as f:
        label_of = {r["story_name"]: r["label"] for r in csv.DictReader(f)}
    labels = np.array([label_of.get(n, "unknown") for n in names])

    models, preds, P, PR = [], {}, {}, {}
    for spec in args.predictions:
        if "=" not in spec:
            sys.exit(f"--predictions entries must be NAME=FOLDER, got {spec}")
        name, folder = spec.split("=", 1)
        pred = load_folder(Path(folder))
        if set(pred) != set(names):
            sys.exit(f"{name}: stories differ from the survey: missing={sorted(set(names) - set(pred))} "
                     f"extra={sorted(set(pred) - set(names))}")
        models.append(name)
        preds[name] = pred
        P[name] = ratings_matrix(pred, names, question_ids, name, vals)
        PR[name] = probs_matrix(pred, names, question_ids) if binary else None
    logger.info(f"{N} stories ({dict(zip(*np.unique(labels, return_counts=True)))}), {Q} questions, "
                f"answers {scale} | models: {models}")

    # ---------- Accuracy + bootstrap ----------
    per_q, mean = {}, {}
    for m in models:
        per_q[m], mean[m] = mean_metrics(T, P[m], n_cls, PR[m])

    rng = np.random.default_rng(SEED)
    boot = np.zeros((args.bootstrap, len(models)))
    for b in range(args.bootstrap):
        idx = rng.integers(0, N, size=N)
        for j, m in enumerate(models):
            boot[b, j] = np.mean([qwk(T[idx, q], P[m][idx, q], n_cls) for q in range(Q)])
    ci = {m: np.percentile(boot[:, j], [2.5, 97.5]) for j, m in enumerate(models)}
    ref = models[0]
    diffs = {m: (mean[m][0] - mean[ref][0], np.percentile(boot[:, j] - boot[:, 0], [2.5, 97.5]))
             for j, m in enumerate(models) if m != ref}

    overall_rows = []
    for m in models:
        cvq = cv_qwk(preds[m], scale)
        overall_rows.append({"model": m, "qwk": round(mean[m][0], 4), "qwk_ci_low": round(ci[m][0], 4),
                             "qwk_ci_high": round(ci[m][1], 4), "cv_qwk": cvq,
                             **{k: round(float(v), 4) for k, v in zip(names_m[1:], mean[m][1:])}})

    # ---------- Timing / cost / size ----------
    def summary(values):
        return st.mean(values), st.median(values), max(values)

    llm_wall = [survey[n]["timing"]["story_wall_s"] for n in names]
    llm_cost = [survey[n].get("cost_usd") or 0.0 for n in names]
    llm_mean = st.mean(llm_wall)
    timing_rows = [{"system": f"LLM survey ({next(iter(survey.values()))['model']})",
                    "mean_s": round(llm_mean, 3), "median_s": round(st.median(llm_wall), 3), "max_s": round(max(llm_wall), 3),
                    "embed_mean_s": None, "predict_mean_ms": None,
                    "cost_per_story_usd": round(st.mean(llm_cost), 6), "cost_total_usd": round(sum(llm_cost), 4),
                    "model_size_mb": None, "speedup_vs_llm": 1.0}]
    for m in models:
        wall = [preds[m][n]["timing"]["story_wall_s"] for n in names]
        embed = [preds[m][n]["timing"]["embed_latency_s"] for n in names]
        predict = [preds[m][n]["timing"]["predict_latency_s"] for n in names]
        cost = [preds[m][n].get("cost_usd") or 0.0 for n in names]
        w_mean, w_median, w_max = summary(wall)
        timing_rows.append({"system": m, "mean_s": round(w_mean, 3), "median_s": round(w_median, 3), "max_s": round(w_max, 3),
                            "embed_mean_s": round(st.mean(embed), 3), "predict_mean_ms": round(st.mean(predict) * 1000, 3),
                            "cost_per_story_usd": round(st.mean(cost), 6), "cost_total_usd": round(sum(cost), 5),
                            "model_size_mb": round(next(iter(preds[m].values()))["model"]["size_bytes"] / 1e6, 2),
                            "speedup_vs_llm": round(llm_mean / w_mean, 1)})

    # ---------- By group ----------
    group_rows = []
    for g in sorted(set(labels)):
        sel = labels == g
        for m in models:
            _, gm = mean_metrics(T[sel], P[m][sel], n_cls, None if PR[m] is None else PR[m][sel])
            group_rows.append({"group": g, "n_stories": int(sel.sum()), "model": m,
                               **{k: round(float(v), 4) for k, v in zip(names_m, gm)}})

    # ---------- Write CSVs ----------
    def write_csv(filename, rows):
        with open(output_dir / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    test_counts = ["/".join(str(c) for c in np.bincount(T[:, q], minlength=n_cls)) for q in range(Q)]
    write_csv("overall.csv", overall_rows)
    write_csv("timing.csv", timing_rows)
    write_csv("by_group.csv", group_rows)
    write_csv("per_question.csv", [
        {"question_id": question_ids[q], "component": components.get(question_ids[q], ""), "test_counts_1to5": test_counts[q],
         **{f"{m}_qwk": round(float(per_q[m][q, 0]), 4) for m in models},
         **{f"{m}_balanced_acc": round(float(per_q[m][q, 1]), 4) for m in models}}
        for q in range(Q)])
    write_csv("per_story_predictions.csv", [
        {"story_name": n, "label": labels[i], "question_id": question_ids[q], "llm_rating": vals[T[i, q]],
         **{m: vals[P[m][i, q]] for m in models}}
        for i, n in enumerate(names) for q in range(Q)])

    # ---------- Machine-readable summary (used by make_dashboard.py) ----------
    summary_json = {
        "n_stories": N, "question_ids": question_ids,
        "components": [components.get(q, "") for q in question_ids], "test_counts": test_counts,
        "models": models, "reference_model": ref, "llm_model": next(iter(survey.values()))["model"],
        "bootstrap": args.bootstrap, "metric_names": names_m,
        "answer_scale": scale, "answer_values": vals, "n_classes": n_cls,
        "overall": overall_rows,
        "diffs": {m: {"diff": round(float(d), 4), "low": round(float(lo), 4), "high": round(float(hi), 4)}
                  for m, (d, (lo, hi)) in diffs.items()},
        "timing": timing_rows,
        "by_group": group_rows,
        "per_question": {m: [{k: round(float(v), 4) for k, v in zip(names_m, per_q[m][q])} for q in range(Q)]
                         for m in models},
        "confusion": {m: [confusion_matrix(T[:, q], P[m][:, q], labels=range(n_cls)).tolist() for q in range(Q)]
                      for m in models},
        "stories": [{"name": n, "label": str(labels[i]), "llm": [vals[c] for c in T[i]],
                     "pred": {m: [vals[c] for c in P[m][i]] for m in models},
                     "mae": {m: round(float(np.abs(P[m][i] - T[i]).mean()), 3) for m in models}}
                    for i, n in enumerate(names)],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=1)

    # ---------- Report ----------
    L = [f"Test comparison | {N} stories ({', '.join(f'{g}: {int((labels == g).sum())}' for g in sorted(set(labels)))}) | "
         f"{Q} questions | reference: LLM survey", ""]
    if binary:
        L += ["1. Agreement with the LLM's yes/no answers (mean over questions; kappa 0 = chance, 1 = perfect; "
              "balanced acc and AUC 0.5 = chance)",
              f"  {'model':<12} {'test kappa [95% range]':>24} {'CV kappa':>8} {'bal_acc':>7} {'accuracy':>8} {'F1 yes':>6} {'AUC':>6}"]
        for r in overall_rows:
            cvq = f"{r['cv_qwk']:.3f}" if r["cv_qwk"] is not None else "-"
            L.append(f"  {r['model']:<12} {r['qwk']:>6.3f} [{r['qwk_ci_low']:.3f}-{r['qwk_ci_high']:.3f}]   {cvq:>8} "
                     f"{r['balanced_acc']:>7.3f} {r['exact_acc']:>8.3f} {r['f1_yes']:>6.3f} {r['auc']:>6.3f}")
    else:
        L += ["1. Accuracy vs the LLM survey (mean over questions; QWK 0 = chance, 1 = perfect)",
              f"  {'model':<12} {'test QWK [95% range]':>22} {'CV QWK':>7} {'bal_acc':>7} {'exact':>6} {'within1':>7} {'QWK3':>6} {'bal_acc3':>8}"]
        for r in overall_rows:
            cvq = f"{r['cv_qwk']:.3f}" if r["cv_qwk"] is not None else "-"
            L.append(f"  {r['model']:<12} {r['qwk']:>6.3f} [{r['qwk_ci_low']:.3f}-{r['qwk_ci_high']:.3f}] {cvq:>7} "
                     f"{r['balanced_acc']:>7.3f} {r['exact_acc']:>6.3f} {r['within1_acc']:>7.3f} {r['qwk_3level']:>6.3f} "
                     f"{r['balanced_acc_3level']:>8.3f}")
    if diffs:
        L += ["", f"  Paired difference in test {'kappa' if binary else 'QWK'} vs {ref} "
                  f"(same bootstrap resamples; range excluding 0 = a real difference):"]
        for m, (d, (lo, hi)) in diffs.items():
            L.append(f"    {m:<12} {d:+.3f} [{lo:+.3f} to {hi:+.3f}]")

    L += ["", "2. Speed, cost and size per story (time = reading the story file -> 20 ratings)",
          f"  {'system':<26} {'mean s':>7} {'median s':>8} {'max s':>6} {'embed s':>7} {'predict ms':>10} "
          f"{'$/story':>9} {'$ total':>8} {'size MB':>7} {'speed-up':>8}"]
    for r in timing_rows:
        fmt = lambda v, spec: format(v, spec) if v is not None else "-"
        L.append(f"  {r['system']:<26} {r['mean_s']:>7.3f} {r['median_s']:>8.3f} {r['max_s']:>6.3f} "
                 f"{fmt(r['embed_mean_s'], '.3f'):>7} {fmt(r['predict_mean_ms'], '.3f'):>10} {r['cost_per_story_usd']:>9.5f} "
                 f"{r['cost_total_usd']:>8.4f} {fmt(r['model_size_mb'], '.2f'):>7} {r['speedup_vs_llm']:>7.1f}x")

    L += ["", "3. Kappa per question" if binary else "3. QWK per question",
          f"  {'q':>4} {'test 1/2/3/4/5':>15} " + " ".join(f"{m:>11}" for m in models) + "  component"]
    for q in range(Q):
        L.append(f"  {question_ids[q]:>4} {test_counts[q]:>15} " + " ".join(f"{per_q[m][q, 0]:>11.3f}" for m in models)
                 + f"  {components.get(question_ids[q], '')}")

    if binary:
        L += ["", "4. By story type (small groups are only indicative)",
              f"  {'group':<16} {'n':>3} {'model':<12} {'kappa':>6} {'bal_acc':>7} {'accuracy':>8} {'F1 yes':>6}"]
        for r in group_rows:
            L.append(f"  {r['group']:<16} {r['n_stories']:>3} {r['model']:<12} {r['qwk']:>6.3f} {r['balanced_acc']:>7.3f} "
                     f"{r['exact_acc']:>8.3f} {r['f1_yes']:>6.3f}")
    else:
        L += ["", "4. By story type (small groups are only indicative)",
              f"  {'group':<16} {'n':>3} {'model':<12} {'QWK':>6} {'bal_acc':>7} {'exact':>6} {'within1':>7} {'QWK3':>6}"]
        for r in group_rows:
            L.append(f"  {r['group']:<16} {r['n_stories']:>3} {r['model']:<12} {r['qwk']:>6.3f} {r['balanced_acc']:>7.3f} "
                     f"{r['exact_acc']:>6.3f} {r['within1_acc']:>7.3f} {r['qwk_3level']:>6.3f}")
    summary_len = len(L)

    L += ["", "5. Stories each model gets most wrong (mean |predicted - LLM rating| over the questions)"]
    for m in models:
        err = np.abs(P[m] - T).mean(1)
        worst = np.argsort(-err)[:5]
        L.append(f"  {m}: " + ", ".join(f"{names[i]} ({labels[i][:4]}, {err[i]:.2f})" for i in worst))

    L += ["", "6. Confusion matrices (rows = LLM rating, cols = predicted)"]
    for q in range(Q):
        L.append(f"  q{question_ids[q]} {components.get(question_ids[q], '')}")
        for m in models:
            cm = confusion_matrix(T[:, q], P[m][:, q], labels=range(n_cls))
            L.append(f"    {m}:  " + " | ".join(" ".join(f"{v:>2}" for v in row) for row in cm))
    (output_dir / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    logger.info("Results:\n\n" + "\n".join(L[:summary_len]) + "\n")
    logger.info(f"Full report (worst stories, confusion matrices): {output_dir / 'report.txt'} | CSVs in {output_dir}")


if __name__ == "__main__":
    main()

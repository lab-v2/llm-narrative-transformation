"""
Approach 1 for the russio distillation experiments: TF-IDF + logistic regression.

Student model for the LLM's yes/no answer (Russia-Ukraine war relevance). Features are word 1-2 grams
plus character 2-5 grams (within word boundaries), so one model covers all 8 languages and both
scripts without per-language word lists. Regularization strength C is chosen by 5-fold CV on the
train split only (folds stratified by language x answer); the final model is trained on all train
documents and predicts each test document one at a time (timed like the LLM survey).

Outputs (same format as the other approaches, so evaluate.py scores them all the same way):
  <models-dir>/tfidf_final.joblib + tfidf_final_config.json (with CV results)
  <predictions-dir>/<doc>.json   per test document: ratings {"1": 0/1}, probabilities, timing, cost 0

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_tfidf.py
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from train_m1 import metric_names, question_metrics

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_split(part: str, survey_dir: Path):
    """Texts, answers and languages of one split part, in split.csv order."""
    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == part]
    names = [r["story_name"] for r in rows]
    texts = [(DATA / part / f"{n}.txt").read_text(encoding="utf-8") for n in names]
    y = []
    for n in names:
        with open(survey_dir / part / f"{n}.json", encoding="utf-8") as f:
            y.append(json.load(f)["ratings"]["1"])
    return names, texts, np.array(y), np.array([r["label"] for r in rows])


def make_pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("lr", LogisticRegression(C=C, max_iter=5000, solver="liblinear"))])


def main():
    parser = argparse.ArgumentParser(description="Approach 1: TF-IDF + logistic regression (russio)")
    parser.add_argument("--survey-dir", type=str, default=str(OUT / "survey_war" / "gpt-5.4"),
                        help="Folder with train/ and test/ survey JSONs (the LLM's answers)")
    parser.add_argument("--C", type=float, nargs="+", default=[1, 3, 10, 30],
                        help="Candidate regularization strengths, chosen by CV accuracy (default: 1 3 10 30)")
    parser.add_argument("--models-dir", type=str, default=str(OUT / "models" / "tfidf_lr"))
    parser.add_argument("--predictions-dir", type=str, default=str(OUT / "predictions" / "tfidf_lr" / "test"))
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    models_dir, pred_dir = Path(args.models_dir), Path(args.predictions_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    _, X_tr, y_tr, lang_tr = load_split("train", survey_dir)
    names_te, X_te, y_te, lang_te = load_split("test", survey_dir)
    logger.info(f"train {len(X_tr)} | test {len(X_te)} | train yes {y_tr.mean():.1%}")

    # ---------- choose C by 5-fold CV on train (stratified by language x answer) ----------
    strata = [f"{l}_{a}" for l, a in zip(lang_tr, y_tr)]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_tr, strata))
    cv = {}
    for C in args.C:
        start = time.perf_counter()
        oof = np.zeros((len(X_tr), 2))
        for a, v in folds:
            pipe = make_pipeline(C).fit([X_tr[i] for i in a], y_tr[a])
            oof[v] = pipe.predict_proba([X_tr[i] for i in v])
        m = dict(zip(metric_names(2), question_metrics(y_tr, oof.argmax(1), 2, oof)))
        cv[C] = m
        logger.info(f"C={C:<5} CV accuracy {m['exact_acc']:.3%} kappa {m['qwk']:.3f} F1 {m['f1_yes']:.3f} "
                    f"AUC {m['auc']:.3f} ({time.perf_counter() - start:.0f}s)")
    best_C = max(cv, key=lambda c: cv[c]["exact_acc"])
    logger.info(f"chosen C = {best_C}")

    # ---------- final model on all train ----------
    start = time.perf_counter()
    pipe = make_pipeline(best_C).fit(X_tr, y_tr)
    train_s = time.perf_counter() - start
    model_file = models_dir / "tfidf_final.joblib"
    joblib.dump(pipe, model_file)
    size = model_file.stat().st_size
    n_features = sum(len(t.vocabulary_) for _, t in pipe.named_steps["tfidf"].transformer_list)
    train_acc = (pipe.predict(X_tr) == y_tr).mean()
    config = {
        "model": "TF-IDF (word 1-2 + char 2-5 grams) + logistic regression", "C": best_C, "n_features": n_features,
        "n_classes": 2, "answer_values": [0, 1], "answer_scale": "yes/no (1/0)", "question_ids": ["1"],
        "train_documents": len(X_tr), "train_seconds": round(train_s, 1), "train_accuracy": round(float(train_acc), 4),
        "cv_by_C": {str(c): {k: round(float(v), 4) for k, v in m.items()} for c, m in cv.items()},
        "cv_mean": {k: round(float(v), 4) for k, v in cv[best_C].items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(models_dir / "tfidf_final_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"final model: {n_features:,} features, {size / 1e6:.1f} MB, trained in {train_s:.0f}s, train accuracy {train_acc:.1%}")

    # ---------- predict test one document at a time (timed end to end) ----------
    pipe.predict_proba(["warm-up"])
    info = {"name": config["model"], "file": str(model_file), "n_parameters": n_features + 1, "size_bytes": size}
    for name in names_te:
        path = DATA / "test" / f"{name}.txt"
        start = time.perf_counter()
        text = path.read_text(encoding="utf-8")
        p = pipe.predict_proba([text])[0]
        wall_s = time.perf_counter() - start
        result = {
            "story_name": name, "story_file": str(path), "model": info, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ratings": {"1": int(p.argmax())}, "probabilities": {"1": [round(float(x), 4) for x in p]}, "n_failed": 0,
            "tokens": 0, "timing": {"story_wall_s": round(wall_s, 5), "embed_latency_s": 0.0, "predict_latency_s": round(wall_s, 6)},
            "cost_usd": 0.0,
        }
        with open(pred_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"test predictions -> {pred_dir}")


if __name__ == "__main__":
    main()

"""
Approach 2 for the russio distillation experiments: frozen OpenAI embeddings + a classifier.

Input: the whole-text embedding of each document (russio_embed.py), standardized on the training
documents. Two classifiers, each with its settings chosen by 5-fold CV on the train split only
(folds stratified by language x answer):
  lr   logistic regression (the standard frozen-embedding probe), C from --C
  mlp  one hidden layer (--hidden options) + dropout 0.3 + AdamW (weight decay 1e-4), mini-batches;
       the number of epochs is picked from --epochs checkpoints in the same CV

Test timing per document = its own embedding call (measured one request per test document by
russio_embed.py) + the classifier's prediction on that single vector (measured here).

Outputs (same format as russio_tfidf.py, scored with evaluate.py):
  <models-root>/emb_<clf>/  model + emb_<clf>_final_config.json (with CV results)
  <predictions-root>/emb_<clf>/test/<doc>.json

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_embed_clf.py --classifier lr
    python prototype/russio_embed_clf.py --classifier mlp
    python prototype/russio_embed_clf.py --classifier lr --question 2 \\
        --survey-dir output_test/prototype/russio/survey_extra/gpt-5.4      # -> emb_lr_q2/
"""

import argparse
import csv
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from train_m1 import FP_IGNORE, metric_names, question_metrics

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
torch.set_num_threads(4)


def load(emb_dir: Path, survey_dir: Path, question: str = "1"):
    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(emb_dir / "whole_meta.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    assert [m["story_name"] for m in meta] == [r["story_name"] for r in rows], "embedding order != split.csv"
    X = np.load(emb_dir / "whole.npy")
    y = []
    for r in rows:
        with open(survey_dir / r["split"] / f"{r['story_name']}.json", encoding="utf-8") as f:
            y.append(json.load(f)["ratings"][question])
    return rows, meta, X, np.array(y)


# ---------------- MLP ----------------

def train_mlp(X, y, hidden, epochs_list, seed=42, X_eval=None):
    """Returns (model, {epoch: P(eval) (n, 2)}) - probabilities at each requested epoch checkpoint."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, t = torch.from_numpy(X), torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for epoch in range(1, max(epochs_list) + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=g).split(128):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), t[idx]).backward()
            opt.step()
        if X_eval is not None and epoch in epochs_list:
            model.eval()
            with torch.no_grad():
                out[epoch] = torch.softmax(model(torch.from_numpy(X_eval)), -1).numpy()
    return model, out


def main():
    parser = argparse.ArgumentParser(description="Approach 2: OpenAI embeddings + classifier (russio)")
    parser.add_argument("--classifier", choices=["lr", "mlp"], required=True)
    parser.add_argument("--embeddings-dir", type=str, default=str(OUT / "embeddings" / "text-embedding-3-large"))
    parser.add_argument("--survey-dir", type=str, default=str(OUT / "survey_war" / "gpt-5.4"))
    parser.add_argument("--question", type=str, default="1",
                        help="Question id in the survey JSONs (1 = war relevance; 2-4 are in survey_extra). "
                             "Outputs for questions other than 1 get a _q<id> suffix")
    parser.add_argument("--C", type=float, nargs="+", default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], help="[lr] candidates")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 512], help="[mlp] hidden-size candidates")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 20, 40], help="[mlp] epoch checkpoints")
    args = parser.parse_args()

    emb_dir, survey_dir, q = Path(args.embeddings_dir), Path(args.survey_dir), args.question
    name = f"emb_{args.classifier}" + ("" if q == "1" else f"_q{q}")
    models_dir, pred_dir = OUT / "models" / name, OUT / "predictions" / name / "test"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows, meta, X, y = load(emb_dir, survey_dir, q)
    tr = np.array([r["split"] == "train" for r in rows])
    Xtr, ytr = X[tr], y[tr]
    strata = [f"{r['label']}_{a}" for r, a, keep in zip(rows, y, tr) if keep]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xtr, strata))
    logger.info(f"{name}: train {tr.sum()} | test {(~tr).sum()} | {X.shape[1]}-dim embeddings")

    # ---------- CV on train ----------
    cv = {}
    for setting in (args.C if args.classifier == "lr" else args.hidden):
        start = time.perf_counter()
        oof = {e: np.zeros((len(Xtr), 2)) for e in (args.epochs if args.classifier == "mlp" else [0])}
        for a, v in folds:
            sc = StandardScaler().fit(Xtr[a])
            with np.errstate(**FP_IGNORE):
                Za, Zv = sc.transform(Xtr[a]).astype(np.float32), sc.transform(Xtr[v]).astype(np.float32)
            if args.classifier == "lr":
                with np.errstate(**FP_IGNORE):
                    oof[0][v] = LogisticRegression(C=setting, max_iter=5000).fit(Za, ytr[a]).predict_proba(Zv)
            else:
                _, probs = train_mlp(Za, ytr[a], setting, args.epochs, X_eval=Zv)
                for e, p in probs.items():
                    oof[e][v] = p
        for e, p in oof.items():
            key = (setting, e)
            cv[key] = dict(zip(metric_names(2), question_metrics(ytr, p.argmax(1), 2, p)))
            label = f"C={setting}" if args.classifier == "lr" else f"hidden={setting} epochs={e}"
            logger.info(f"{label:22} CV accuracy {cv[key]['exact_acc']:.3%} kappa {cv[key]['qwk']:.3f} AUC {cv[key]['auc']:.3f}")
        logger.info(f"  ({time.perf_counter() - start:.0f}s)")
    best = max(cv, key=lambda k: cv[k]["exact_acc"])
    logger.info(f"chosen: {best}")

    # ---------- final model on all train ----------
    start = time.perf_counter()
    sc = StandardScaler().fit(Xtr)
    with np.errstate(**FP_IGNORE):
        Ztr = sc.transform(Xtr).astype(np.float32)
    if args.classifier == "lr":
        with np.errstate(**FP_IGNORE):
            clf = LogisticRegression(C=best[0], max_iter=5000).fit(Ztr, ytr)
        model_file = models_dir / f"{name}_final.joblib"
        joblib.dump({"scaler": sc, "clf": clf}, model_file)
        n_params = clf.coef_.size + clf.intercept_.size
        predict = lambda z: clf.predict_proba(z)
        train_acc = (clf.predict(Ztr) == ytr).mean()
    else:
        model, _ = train_mlp(Ztr, ytr, best[0], [best[1]])
        model.eval()
        model_file = models_dir / f"{name}_final.pt"
        torch.save({"state_dict": model.state_dict(), "hidden": best[0], "scaler_mean": sc.mean_, "scaler_scale": sc.scale_}, model_file)
        n_params = sum(p.numel() for p in model.parameters())

        def predict(z):
            with torch.no_grad():
                return torch.softmax(model(torch.from_numpy(z)), -1).numpy()
        train_acc = (predict(Ztr).argmax(1) == ytr).mean()
    train_s = time.perf_counter() - start
    size = model_file.stat().st_size

    config = {"model": f"OpenAI text-embedding-3-large + {'logistic regression' if args.classifier == 'lr' else 'MLP'}",
              "classifier": args.classifier, "chosen": {"setting": best[0], "epochs": best[1]},
              "n_classes": 2, "answer_values": [0, 1], "answer_scale": "yes/no (1/0)", "question_ids": [q],
              "train_documents": int(tr.sum()), "train_seconds": round(train_s, 1), "train_accuracy": round(float(train_acc), 4),
              "cv_all": {f"{k[0]}|{k[1]}": {m: round(float(v), 4) for m, v in r.items()} for k, r in cv.items()},
              "cv_mean": {m: round(float(v), 4) for m, v in cv[best].items()},
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    with open(models_dir / f"{name}_final_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"final model: {n_params:,} parameters, {size / 1e6:.2f} MB, train accuracy {train_acc:.1%}")

    # ---------- test predictions (embedding latency + single-vector prediction) ----------
    info = {"name": config["model"], "file": str(model_file), "n_parameters": int(n_params), "size_bytes": size}
    predict(sc.transform(X[~tr][:1]).astype(np.float32))  # warm-up
    for i in np.where(~tr)[0]:
        start = time.perf_counter()
        with np.errstate(**FP_IGNORE):
            p = predict(sc.transform(X[i:i + 1]).astype(np.float32))[0]
        pred_s = time.perf_counter() - start
        embed_s = float(meta[i]["latency_s"])
        doc = rows[i]["story_name"]
        result = {"story_name": doc, "story_file": str(DATA / "test" / f"{doc}.txt"), "model": info,
                  "embedding_model": "text-embedding-3-large", "timestamp": datetime.now().isoformat(timespec="seconds"),
                  "ratings": {q: int(p.argmax())}, "probabilities": {q: [round(float(x), 4) for x in p]}, "n_failed": 0,
                  "tokens": int(meta[i]["tokens"]),
                  "timing": {"story_wall_s": round(embed_s + pred_s, 4), "embed_latency_s": embed_s, "predict_latency_s": round(pred_s, 6)},
                  "cost_usd": round(min(int(meta[i]["tokens"]), 8191) * 0.13 / 1e6, 8)}
        with open(pred_dir / f"{doc}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"test predictions -> {pred_dir}")


if __name__ == "__main__":
    main()

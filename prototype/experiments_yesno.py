"""
Experiments to improve the yes/no models, one idea ("recipe") at a time, all with the same
folds and the same scoring against the LLM's yes/no answers.

For each recipe it records:
  train accuracy  final model (all 80 train stories) scored on those same stories
  CV metrics      3x5 stratified folds on the 80 train stories (pooled out-of-fold predictions,
                  same as train_m1.py). Decisions between recipes are made on these.
  test metrics    the 26 test stories, recorded for information only (never used to choose)
and appends one row per recipe to --log (and prints the whole log).

Recipes:
  baseline_mlp          M1 network: PCA 32 -> hidden 16 per question (the chosen yes/no M1)
  baseline_logreg       M1 logistic regression: PCA 16, C = 0.01 per question
  baseline_m4           M4: one shared network on story PCA 32 + question PCA 8, hidden 16
  ensemble              average probabilities of baseline_mlp, baseline_logreg and baseline_m4
  soft_rating_mlp/_logreg   train on P(yes | the LLM's earlier 1-5 rating) instead of 0/1
  soft_vote_mlp/_logreg     train on the mean of the yes/no answer and P(yes | 1-5 rating)
                            (two LLM runs as two votes)
  The P(yes | rating) table is estimated from the training stories of each fold only.
  nopca_mlp             network M1 on the full 3072-number embedding (standardized), no PCA
  nopca_mlp_raw         network M1 on the full embedding exactly as returned by OpenAI, no PCA
  nopca_mlp_wd01        nopca_mlp with weight decay 0.1 (each network has ~49k weights instead of 562)
  m4_onehot             M4 + a one-hot "which question" input (shared patterns + per-question adjustments)
  m4_interact           M4 + story x question products (both projected with the story PCA)
  m4_interact_onehot    both of the above
  m4_big                M4 with hidden 64-32 and 800 steps
  votes3_mlp/_logreg    (with --labels majority) trained on the share of the three LLM runs that said yes
  votes3_ensemble       average of votes3 network + votes3 logistic + M4
  parts_*               the same models with the story read in parts: input = whole-story embedding +
                        beginning / middle / end embeddings (embed_parts.py), compressed by PCA as before
  aug_*                 the same models with extra training stories: LLM-transformed versions of the
                        train stories (make_augment_set.py), surveyed yes/no. A version is used only when
                        its source story is in the fold's training part; validation and test use original
                        stories only. Extra stories have no 1-5 rating, so soft recipes use their 0/1 answer.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/experiments_yesno.py --recipe baseline_mlp
"""

import argparse
import csv
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from train_m1 import (FP_IGNORE, LOSSES, REPO_ROOT, SEED, PerQuestionMLP, class_weights, fit_features, load_data,
                      predict_proba, question_metrics, to_features, train, train_logreg)
from train_m4 import load_question_embeddings, pair_features, predict_shared, train_shared

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

P = REPO_ROOT / "output_test" / "prototype"
EMB = P / "embeddings" / "text-embedding-3-large"
MLP_ARGS = SimpleNamespace(hidden=[16], dropout=0.3, lr=1e-3, weight_decay=1e-2, steps=400, loss="ce")
M4_ARGS = SimpleNamespace(hidden=[16], dropout=0.3, lr=1e-3, weight_decay=1e-2, steps=400, loss="ce")


# ==========================================================
# Data
# ==========================================================

def load_all(embedding: str = "large", labels: str = "single"):
    """embedding: 'large', 'small' (text-embedding-3-small) or 'large+small' (both, concatenated).
    labels: 'single' = one yes/no LLM run; 'majority' = majority of three runs (make_majority_labels.py)."""
    split = REPO_ROOT / "prototype" / "split.csv"
    survey = P / ("survey_binary" if labels == "single" else "survey_binary_majority") / "gpt-6-sol"
    if embedding == "large+small":
        tr, te, R, QE = load_all("large", labels)
        tr2, te2, _, QE2 = load_all("small", labels)
        assert tr["names"] == tr2["names"] and te["names"] == te2["names"]
        tr["X"], te["X"] = np.hstack([tr["X"], tr2["X"]]), np.hstack([te["X"], te2["X"]])
        return tr, te, R, np.hstack([QE, QE2])
    emb = EMB if embedding == "large" else P / "embeddings" / "text-embedding-3-small"
    tr = load_data(emb / "train", survey / "train", split)
    te = load_data(emb / "test", survey / "test", split)
    if labels == "majority":                       # share of the three runs that said yes (train stories)
        shares = {}
        for p in (survey / "train").glob("*.json"):
            with open(p, encoding="utf-8") as f:
                shares[p.stem] = json.load(f)["yes_share"]
        tr["V"] = np.array([[shares[n][q] for q in tr["question_ids"]] for n in tr["names"]], dtype=np.float32)
    assert tr["question_ids"] == te["question_ids"] and tr["n_classes"] == 2
    ratings = {}
    for p in (P / "survey" / "gpt-6-sol" / "train").glob("*.json"):
        with open(p, encoding="utf-8") as f:
            ratings[p.stem] = json.load(f)["ratings"]
    R = np.array([[ratings[n][q] for q in tr["question_ids"]] for n in tr["names"]])  # earlier 1-5 ratings (train)
    QE = load_question_embeddings(emb / "questions" / "collectivistic_questions.json", tr["question_ids"], tr["embedding_model"])
    return tr, te, R, QE


def load_parts(data: dict, folder: str) -> np.ndarray:
    """(N, n_parts * D) part embeddings in the same story order as data["names"]."""
    out = []
    for n in data["names"]:
        with open(EMB / folder / f"{n}.json", encoding="utf-8") as f:
            out.append(np.concatenate(json.load(f)["parts"]))
    return np.asarray(out, dtype=np.float32)


def load_augment(tr: dict):
    """Extra training stories: (X, Y, index of the source story in tr["names"]). Stories without a complete
    yes/no survey or embedding are skipped."""
    manifest = REPO_ROOT / "prototype" / "augment_manifest.csv"
    if not manifest.exists():
        return None
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pos = {n: i for i, n in enumerate(tr["names"])}
    X, Y, src, skipped = [], [], [], 0
    for r in rows:
        e, sv = EMB / "augment" / f"{r['story_name']}.json", P / "survey_binary" / "gpt-6-sol" / "augment" / f"{r['story_name']}.json"
        if not (e.exists() and sv.exists()):
            skipped += 1
            continue
        with open(e, encoding="utf-8") as f:
            emb = json.load(f)
        with open(sv, encoding="utf-8") as f:
            survey = json.load(f)
        if survey.get("n_failed", 1) != 0 or emb["model"] != tr["embedding_model"]:
            skipped += 1
            continue
        X.append(emb["embedding"])
        Y.append([survey["ratings"][q] for q in tr["question_ids"]])
        src.append(pos[r["source_story"]])
    if skipped:
        print(f"(augment: {skipped} extra stories skipped - not yet surveyed/embedded)")
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64), np.asarray(src)


# ==========================================================
# Models (each fit returns predict(X) -> probabilities (Q, n, 2))
# ==========================================================

def soft_targets(Ybin: np.ndarray, R: np.ndarray, kind: str) -> np.ndarray:
    """P(yes) per (story, question). 'rating': P(yes | 1-5 rating) estimated on these stories;
    'vote': mean of the yes/no answer and P(yes | rating)."""
    table = {r: Ybin[R == r].mean() if (R == r).any() else (0.0 if r <= 2 else 1.0) for r in range(1, 6)}
    p_rating = np.vectorize(table.get)(R).astype(np.float32)
    return p_rating if kind == "rating" else (0.5 * Ybin + 0.5 * p_rating).astype(np.float32)


def train_mlp_soft(Z: np.ndarray, Ybin: np.ndarray, soft: np.ndarray, seed: int) -> PerQuestionMLP:
    """Per-question network trained on soft targets (cross-entropy against P(yes)); class weights from the
    hard yes/no answer, as in train_m1."""
    torch.manual_seed(seed)
    a = MLP_ARGS
    model = PerQuestionMLP(Ybin.shape[1], Z.shape[1], tuple(a.hidden), 2, a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    x, y = torch.from_numpy(Z), torch.from_numpy(Ybin.T.copy())
    w = class_weights(Ybin, 2).gather(1, y)                                    # (Q, N)
    p = torch.from_numpy(soft.T.copy())                                        # (Q, N)
    model.train()
    for _ in range(a.steps):
        opt.zero_grad()
        logp = torch.log_softmax(model(x), dim=-1)
        nll = -(p * logp[..., 1] + (1 - p) * logp[..., 0])
        ((w * nll).sum(1) / w.sum(1)).sum().backward()
        opt.step()
    return model


def train_logreg_soft(Z: np.ndarray, soft: np.ndarray, C: float) -> PerQuestionMLP:
    """Logistic regression on soft targets: each story appears once as yes (weight p) and once as no
    (weight 1 - p), with balanced class weights from the soft label mass. Packed like train_logreg."""
    Q, k = soft.shape[1], Z.shape[1]
    W = np.zeros((Q, k, 2), dtype=np.float32)
    b = np.zeros((Q, 1, 2), dtype=np.float32)
    Z2 = np.vstack([Z, Z])
    y2 = np.r_[np.ones(len(Z)), np.zeros(len(Z))]
    for q in range(Q):
        p = soft[:, q]
        mass_yes, mass_no = p.sum(), (1 - p).sum()
        w2 = np.r_[p * len(Z) / (2 * max(mass_yes, 1e-6)), (1 - p) * len(Z) / (2 * max(mass_no, 1e-6))]
        with np.errstate(**FP_IGNORE):
            clf = LogisticRegression(C=C, max_iter=5000).fit(Z2, y2, sample_weight=w2)
        W[q, :, 1] = clf.coef_[0]
        b[q, 0, 1] = clf.intercept_[0]
    model = PerQuestionMLP(Q, k, (), 2, 0.0)
    with torch.no_grad():
        model.weights[0].copy_(torch.from_numpy(W))
        model.biases[0].copy_(torch.from_numpy(b))
    return model


def training_set(X, Y, idx, ctx, soft=None, aug=False):
    """Training stories (+ extra versions of them when aug) and optional soft targets."""
    if soft == "votes3":
        if ctx.get("V") is None:
            sys.exit("votes3 recipes need --labels majority")
        S = ctx["V"][idx]
    else:
        S = soft_targets(Y, ctx["R"][idx], soft) if soft else None
    if aug:
        aX, aY, asrc = ctx["aug"]
        sel = np.isin(asrc, idx)                     # only versions of stories in this training part
        X, Y = np.vstack([X, aX[sel]]), np.vstack([Y, aY[sel]])
        if S is not None:
            S = np.vstack([S, aY[sel].astype(np.float32)])
    return X, Y, S


def with_pca(k: int, fit_on_Z, soft=None, aug=False):
    """Wrap a model on PCA-k features: returns fit(X, Y, idx, ctx, seed) -> predict(X)."""
    def fit(X, Y, idx, ctx, seed):
        Xt, Yt, S = training_set(X, Y, idx, ctx, soft, aug)
        pca, sc = fit_features(Xt, k)
        model = fit_on_Z(to_features(pca, sc, Xt), Yt, S, seed)
        return lambda Xe: predict_proba(model, to_features(pca, sc, Xe))
    return fit


def without_pca(fit_on_Z, scale=True):
    """Like with_pca but the model sees the full embedding (standardized on the training stories if scale)."""
    def fit(X, Y, idx, ctx, seed):
        Xt, Yt, S = training_set(X, Y, idx, ctx)
        if scale:
            sc = StandardScaler().fit(Xt)
            tf = lambda A: sc.transform(A).astype(np.float32)
        else:
            tf = lambda A: A.astype(np.float32)
        with np.errstate(**FP_IGNORE):
            model = fit_on_Z(tf(Xt), Yt, S, seed)
        return lambda Xe: predict_proba(model, tf(Xe))
    return fit


def m4(aug=False, onehot=False, interact=False, hidden=(16,), steps=400):
    """M4 variants. onehot: add which-question one-hot; interact: add story x question products, with the
    question embedding projected by the story PCA so both live in the same coordinates."""
    a = SimpleNamespace(**{**vars(M4_ARGS), "hidden": list(hidden), "steps": steps})

    def fit(X, Y, idx, ctx, seed):
        Xt, Yt, _ = training_set(X, Y, idx, ctx, None, aug)
        pca, sc = fit_features(Xt, 32)
        qp, qs = fit_features(ctx["QE"], 8)
        Zq = to_features(qp, qs, ctx["QE"])
        if onehot:
            Zq = np.hstack([Zq, np.eye(len(Zq), dtype=np.float32)])
        Zq_story = to_features(pca, sc, ctx["QE"]) if interact else None   # questions in story-PCA coordinates

        def feats(Xe):
            Zs = to_features(pca, sc, Xe)
            F = pair_features(Zs, Zq)
            if interact:
                F = np.concatenate([F, (Zq_story[:, None, :] * Zs[None, :, :]).astype(np.float32)], axis=-1)
            return F
        model = train_shared(feats(Xt), Yt, a, seed, 2)
        return lambda Xe: predict_shared(model, feats(Xe))
    return fit


MLP = lambda Z, Y, S, s: train(Z, Y, MLP_ARGS, seed=s, n_classes=2)
LOGREG = lambda Z, Y, S, s: train_logreg(Z, Y, 0.01, 2)
MLP_SOFT = lambda Z, Y, S, s: train_mlp_soft(Z, Y, S, s)
LOGREG_SOFT = lambda Z, Y, S, s: train_logreg_soft(Z, S, 0.01)

RECIPES = {
    "baseline_mlp": with_pca(32, MLP),
    "baseline_logreg": with_pca(16, LOGREG),
    "baseline_m4": m4(),
    "soft_rating_mlp": with_pca(32, MLP_SOFT, soft="rating"),
    "soft_vote_mlp": with_pca(32, MLP_SOFT, soft="vote"),
    "soft_rating_logreg": with_pca(16, LOGREG_SOFT, soft="rating"),
    "soft_vote_logreg": with_pca(16, LOGREG_SOFT, soft="vote"),
    "aug_mlp": with_pca(32, MLP, aug=True),
    "aug_logreg": with_pca(16, LOGREG, aug=True),
    "aug_m4": m4(aug=True),
    "aug_soft_vote_mlp": with_pca(32, MLP_SOFT, soft="vote", aug=True),
    "aug_soft_vote_logreg": with_pca(16, LOGREG_SOFT, soft="vote", aug=True),
    "nopca_mlp": without_pca(MLP),
    "nopca_mlp_raw": without_pca(MLP, scale=False),
    "nopca_mlp_wd01": without_pca(lambda Z, Y, S, s: train(Z, Y, SimpleNamespace(**{**vars(MLP_ARGS), "weight_decay": 0.1}),
                                                           seed=s, n_classes=2)),
    "m4_onehot": m4(onehot=True),
    "m4_interact": m4(interact=True),
    "m4_interact_onehot": m4(onehot=True, interact=True),
    "m4_big": m4(hidden=(64, 32), steps=800),
    "votes3_mlp": with_pca(32, MLP_SOFT, soft="votes3"),
    "votes3_logreg": with_pca(16, LOGREG_SOFT, soft="votes3"),
}


def ensemble(*names):
    def fit(X, Y, idx, ctx, seed):
        preds = [RECIPES[n](X, Y, idx, ctx, seed) for n in names]
        return lambda Xe: np.mean([p(Xe) for p in preds], axis=0)
    return fit


RECIPES["ensemble"] = ensemble("baseline_mlp", "baseline_logreg", "baseline_m4")
RECIPES["ensemble_soft"] = ensemble("soft_vote_mlp", "soft_vote_logreg", "baseline_m4")
RECIPES["aug_ensemble"] = ensemble("aug_mlp", "aug_logreg", "aug_m4")
RECIPES["votes3_ensemble"] = ensemble("votes3_mlp", "votes3_logreg", "baseline_m4")
for _name in ("baseline_mlp", "baseline_logreg", "baseline_m4", "ensemble", "ensemble_soft"):
    RECIPES["parts_" + _name.replace("baseline_", "")] = RECIPES[_name]   # same models, parts input (see main)
RECIPES["aug_ensemble_soft"] = ensemble("aug_soft_vote_mlp", "aug_soft_vote_logreg", "aug_m4")

DESCRIPTIONS = {
    "baseline_mlp": "M1 network, PCA 32, hidden 16 (chosen yes/no M1)",
    "baseline_logreg": "M1 logistic regression, PCA 16, C 0.01",
    "baseline_m4": "M4 shared network, story PCA 32 + question PCA 8, hidden 16",
    "ensemble": "average of network M1 + logistic M1 + M4",
    "soft_rating_mlp": "network M1 trained on P(yes | earlier 1-5 rating)",
    "soft_vote_mlp": "network M1 trained on mean(yes/no answer, P(yes | 1-5 rating))",
    "soft_rating_logreg": "logistic M1 trained on P(yes | earlier 1-5 rating)",
    "soft_vote_logreg": "logistic M1 trained on mean(yes/no answer, P(yes | 1-5 rating))",
    "ensemble_soft": "average of soft-vote network + soft-vote logistic + M4",
    "nopca_mlp": "network M1 on the full 3072-number embedding, standardized (no PCA)",
    "nopca_mlp_raw": "network M1 on the full embedding as returned by OpenAI (no PCA, no scaling)",
    "nopca_mlp_wd01": "network M1 on the full standardized embedding, weight decay 0.1 (no PCA)",
    "m4_onehot": "M4 + one-hot question identity",
    "m4_interact": "M4 + story x question products (same PCA space)",
    "m4_interact_onehot": "M4 + story x question products + one-hot question identity",
    "m4_big": "M4, hidden 64-32, 800 steps",
    "votes3_mlp": "network M1 trained on the share of 3 LLM runs saying yes",
    "votes3_logreg": "logistic M1 trained on the share of 3 LLM runs saying yes",
    "votes3_ensemble": "average of votes3 network + votes3 logistic + M4",
    "parts_mlp": "network M1, input = whole + beginning/middle/end embeddings",
    "parts_logreg": "logistic M1, input = whole + beginning/middle/end embeddings",
    "parts_m4": "M4, story input = whole + beginning/middle/end embeddings",
    "parts_ensemble": "average of parts network + parts logistic + parts M4",
    "parts_ensemble_soft": "average of parts soft-vote network + soft-vote logistic + parts M4",
    "aug_mlp": "network M1 + extra transformed train stories",
    "aug_logreg": "logistic M1 + extra transformed train stories",
    "aug_m4": "M4 + extra transformed train stories",
    "aug_soft_vote_mlp": "soft-vote network M1 + extra transformed train stories",
    "aug_soft_vote_logreg": "soft-vote logistic M1 + extra transformed train stories",
    "aug_ensemble": "average of aug network + aug logistic + aug M4",
    "aug_ensemble_soft": "average of aug soft-vote network + aug soft-vote logistic + aug M4",
}


# ==========================================================
# Evaluation
# ==========================================================

def summarize(Y: np.ndarray, probs: np.ndarray, threshold: float = None) -> dict:
    """probs (Q, N, 2) -> mean over questions of kappa, balanced acc, accuracy, F1 yes, AUC.
    threshold: answer yes when P(yes) >= threshold (default: the most likely answer)."""
    pred = lambda q: probs[q].argmax(-1) if threshold is None else (probs[q][:, 1] >= threshold).astype(int)
    m = np.nanmean([question_metrics(Y[:, q], pred(q), 2, probs[q]) for q in range(Y.shape[1])], axis=0)
    return dict(zip(["kappa", "bal_acc", "acc", "f1_yes", "auc"], m))


def main():
    parser = argparse.ArgumentParser(description="Yes/no experiments: one recipe at a time, CV + test")
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--log", type=str, default=str(P / "experiments" / "yesno_log.csv"))
    parser.add_argument("--embedding", choices=["large", "small", "large+small"], default="large",
                        help="Story/question embeddings: text-embedding-3-large (default), -small, or both concatenated")
    parser.add_argument("--labels", choices=["single", "majority"], default="single",
                        help="single = one yes/no LLM run (default); majority = majority of three runs, for "
                             "training AND as the test reference")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Pick one yes/no cut-off on P(yes) that maximizes CV accuracy (train folds only) "
                             "and use it for CV and test; logged as <recipe>+thr")
    args = parser.parse_args()
    torch.set_num_threads(1)

    tr, te, R, QE = load_all(args.embedding, args.labels)
    X, Y, groups = tr["X"], tr["Y"], tr["groups"]
    Xt = te["X"]
    if args.recipe.startswith("parts"):           # read the story in parts: whole + beginning/middle/end
        X = np.hstack([X, load_parts(tr, "train_parts3")])
        Xt = np.hstack([Xt, load_parts(te, "test_parts3")])
    ctx = {"R": R, "QE": QE, "V": tr.get("V"), "aug": load_augment(tr) if args.recipe.startswith("aug") else None}
    if args.recipe.startswith("aug"):
        if ctx["aug"] is None or len(ctx["aug"][0]) == 0:
            sys.exit("No extra stories available yet (run make_augment_set.py, embed_story.py and survey_story.py first)")
        print(f"(augment: {len(ctx['aug'][0])} extra training stories)")
    fit = RECIPES[args.recipe]

    oof = np.zeros((3, Y.shape[1], len(X), 2), dtype=np.float32)
    for i, (a, v) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, groups)):
        oof[i // 5][:, v] = fit(X[a], Y[a], a, ctx, SEED + i)(X[v])
    thr = None
    if args.tune_threshold:   # one cut-off, chosen on the out-of-fold predictions of the TRAIN stories only
        grid = np.round(np.arange(0.30, 0.81, 0.025), 3)
        thr = float(max(grid, key=lambda t: np.mean([summarize(Y, oof[r], t)["acc"] for r in range(3)])))
        print(f"(chosen cut-off on train CV: answer yes when P(yes) >= {thr})")
    cv = {k: np.mean([summarize(Y, oof[r], thr)[k] for r in range(3)]) for k in ["kappa", "bal_acc", "acc", "f1_yes", "auc"]}

    final = fit(X, Y, np.arange(len(X)), ctx, SEED)
    train_acc = summarize(Y, final(X), thr)["acc"]
    test = summarize(te["Y"], final(Xt), thr)

    name = (args.recipe + ("+thr" if thr is not None else "") + ("" if args.embedding == "large" else f"@{args.embedding}")
            + ("" if args.labels == "single" else " [majority]"))
    row = {"recipe": name, "description": DESCRIPTIONS[args.recipe] + (f" | cut-off P(yes) >= {thr} (tuned on train CV)" if thr is not None else "")
                                         + ("" if args.embedding == "large" else f" | embeddings: {args.embedding}")
                                         + ("" if args.labels == "single" else " | labels: majority of 3 LLM runs (train and test)"),
           "train_acc": round(train_acc, 4), **{f"cv_{k}": round(float(v), 4) for k, v in cv.items()},
           **{f"test_{k}": round(float(v), 4) for k, v in test.items()},
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if log.exists():
        with open(log, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["recipe"] != name]
    rows.append(row)
    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)

    print(f"{'recipe':20} | {'train':>6} | {'CV acc':>6} {'kappa':>6} {'bal_acc':>7} {'AUC':>6} | {'TEST acc':>8} {'kappa':>6} {'bal_acc':>7} {'AUC':>6}")
    for r in rows:
        print(f"{r['recipe']:20} | {float(r['train_acc']):>6.1%} | {float(r['cv_acc']):>6.1%} {float(r['cv_kappa']):>6.3f} "
              f"{float(r['cv_bal_acc']):>7.3f} {float(r['cv_auc']):>6.3f} | {float(r['test_acc']):>8.1%} "
              f"{float(r['test_kappa']):>6.3f} {float(r['test_bal_acc']):>7.3f} {float(r['test_auc']):>6.3f}")


if __name__ == "__main__":
    main()

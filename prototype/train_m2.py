"""
M2 for the LLM-free survey prototype: one small neural network per survey
question on "prototype distances" (the lab's prototype method, per question).

Features for question q (built from training stories only):
    - Group the training stories by the rating the LLM gave them on q (1..5).
    - Summarize each group with up to K k-means centroids ("prototypes") of the
      raw OpenAI embeddings (K = 3 by default) -> up to 5*K prototypes for q.
    - A story's features = cosine distance from its embedding to each prototype,
      sorted nearest-first within each rating group (so feature j means "j-th
      nearest rating-r prototype" no matter how k-means ordered the clusters).
    - Groups with fewer than K stories repeat their centroids; empty groups give
      a constant (uninformative) distance.
    -> 5*K features per question, standardized, into that question's own MLP.

Cross-fitting: a training story must not be measured against prototypes built
from itself (in small groups its distance would be ~0 to its own rating's
prototype, which never happens for a new story). So each training story's
features come from prototypes built on the other inner folds; validation/test
stories use prototypes built from all training stories.

Training (same as M1): one network per question (no shared weights),
cross-entropy weighted by 1/frequency of each rating, full-batch AdamW +
dropout, fixed steps. Repeated stratified K-fold CV measures it; then the
model is retrained on all train stories and saved for the test set.

Outputs in --output-dir: same files as train_m1.py (cv_per_question.csv,
cv_oof_predictions.csv, cv_report.txt, train.log) plus m2_final.pt /
m2_final_config.json.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/train_m2.py \
        --embeddings-dir output_test/prototype/embeddings/text-embedding-3-large/train \
        --survey-dir output_test/prototype/survey/gpt-6-sol/train \
        --output-dir output_test/prototype/models/m2
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from train_m1 import (FP_IGNORE, N_CLASSES, metric_names, REPO_ROOT, SEED, load_data, predict_proba,
                      setup_file_logging, train, write_cv_outputs)

logger = logging.getLogger(__name__)

EMPTY_DISTANCE = 1.0  # used for prototype slots of ratings with no training stories (constant -> no information)


# ==========================================================
# Prototype features
# ==========================================================

def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def build_prototypes(X: np.ndarray, Y: np.ndarray, k: int, seed: int, normalize: bool = True,
                     n_classes: int = N_CLASSES):
    """Up to k k-means centroids per (question, rating) from the stories with that rating.
    X: (N, D) shared by all questions, or (Q, N, D) per question (M3's encodings).
    Returns prototypes (Q, C*k, D) (unit length if normalize) and a mask (Q, C*k) of slots that have one."""
    N, Q = Y.shape
    P = np.zeros((Q, n_classes * k, X.shape[-1]), dtype=np.float32)
    mask = np.zeros((Q, n_classes * k), dtype=bool)
    for q in range(Q):
        Xq = X[q] if X.ndim == 3 else X
        for r in range(n_classes):
            members = Xq[Y[:, q] == r]
            if len(members) == 0:
                continue
            n_clusters = min(k, len(members))
            if n_clusters == len(members):
                centroids = members
            else:
                with np.errstate(**FP_IGNORE):
                    centroids = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(members).cluster_centers_
            if normalize:
                centroids = unit(centroids)
            slots = slice(r * k, (r + 1) * k)
            P[q, slots] = centroids[[i % n_clusters for i in range(k)]]  # pad by repeating
            mask[q, slots] = True
    return P, mask


def distances(X: np.ndarray, P: np.ndarray, mask: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Distance of each story to each prototype, sorted nearest-first within each rating.
    X (N, D) shared or (Q, N, D) per question, P (Q, C*k, D) -> (Q, N, C*k)
    cosine (M2): empty rating groups get the constant EMPTY_DISTANCE.
    euclidean (M3): empty rating groups get the story's distance to its farthest real prototype."""
    with np.errstate(**FP_IGNORE):
        if metric == "cosine":
            d = 1.0 - np.einsum("nd,qsd->qns", unit(X), P)
            d = np.where(mask[:, None, :], d, EMPTY_DISTANCE)
        else:
            Xq = X if X.ndim == 3 else X[None]
            d = np.sqrt(((Xq[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1))
            far = np.where(mask[:, None, :], d, -np.inf).max(-1, keepdims=True)
            d = np.where(mask[:, None, :], d, far)
    Q, N, S = d.shape
    d = np.sort(d.reshape(Q, N, S // k, k), axis=-1).reshape(Q, N, S).astype(np.float32)
    if not np.isfinite(d).all():
        raise FloatingPointError("Prototype distances contain NaN/inf - check the embeddings")
    return d


def crossfit_distances(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, k: int, inner_folds: int, seed: int,
                       n_classes: int = N_CLASSES):
    """Distances for training stories, each from prototypes built without it (inner folds)."""
    F = np.zeros((Y.shape[1], len(X), n_classes * k), dtype=np.float32)
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for fit_idx, out_idx in inner.split(X, groups):
        P, mask = build_prototypes(X[fit_idx], Y[fit_idx], k, seed, n_classes=n_classes)
        F[:, out_idx] = distances(X[out_idx], P, mask, k)
    return F


def fit_scaler(F: np.ndarray):
    """Per question and feature over stories: F (Q, N, S) -> mean, std (Q, 1, S)."""
    mean = F.mean(1, keepdims=True)
    std = F.std(1, keepdims=True)
    std[std < 1e-8] = 1.0  # constant features (e.g. empty rating groups) stay 0 after scaling
    return mean, std


def fit_features(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, args, seed: int):
    """Everything M2 learns from the training stories before the networks:
    cross-fitted training features, prototypes from all of them, and the scaler."""
    n = getattr(args, "n_classes", N_CLASSES)
    F_train = crossfit_distances(X, Y, groups, args.k_per_rating, args.inner_folds, seed, n_classes=n)
    P, mask = build_prototypes(X, Y, args.k_per_rating, seed, n_classes=n)
    mean, std = fit_scaler(F_train)
    return (F_train - mean) / std, P, mask, mean, std


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Train M2 (per-question MLP on prototype distances), evaluated with repeated CV")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Train embeddings (<story>.json from embed_story.py)")
    parser.add_argument("--survey-dir", type=str, required=True, help="Train survey ratings (<story>.json from survey_story.py)")
    parser.add_argument("--split-csv", type=str, default=str(REPO_ROOT / "prototype" / "split.csv"),
                        help="Story -> individualistic/collectivistic, used to stratify folds")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write CV results and the final model")
    # Features
    parser.add_argument("--k-per-rating", type=int, default=3, help="k-means prototypes per (question, rating) (default: 3)")
    parser.add_argument("--inner-folds", type=int, default=5, help="Inner folds for cross-fitted training features (default: 5)")
    # Model setting
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 16],
                        help="Hidden layer sizes, e.g. --hidden 32 16 (default); --hidden with no values = no hidden layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--steps", type=int, default=400, help="Full-batch training steps (default: 400)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default: 0.001)")
    parser.add_argument("--loss", choices=["ce", "emd", "ce+emd"], default="ce",
                        help="ce = cross-entropy (default); emd = order-aware (squared earth mover's distance); ce+emd = both")
    parser.add_argument("--decode", choices=["argmax", "mean"], default="argmax",
                        help="How probabilities become a rating: argmax = most likely (default); mean = expected rating, rounded")
    # Evaluation
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, "train.log")

    data = load_data(Path(args.embeddings_dir), Path(args.survey_dir), Path(args.split_csv))
    X, Y, groups = data["X"], data["Y"], data["groups"]
    N, Q = Y.shape
    args.n_classes = data["n_classes"]
    hidden_str = "-".join(map(str, args.hidden)) or "none"
    setting = (f"k_per_rating={args.k_per_rating} inner_folds={args.inner_folds} hidden={hidden_str} "
               f"dropout={args.dropout} weight_decay={args.weight_decay} steps={args.steps} lr={args.lr} "
               f"loss={args.loss} decode={args.decode}")
    group_counts = {str(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True))}
    logger.info(f"{N} stories ({group_counts}), {Q} questions, "
                f"embeddings {data['embedding_model']} ({X.shape[1]} dims)")
    logger.info(f"Setting: {setting} | {args.n_classes * args.k_per_rating} features per question | "
                f"{args.repeats}x{args.folds}-fold CV")

    # ---------- Cross-validation ----------
    oof = np.zeros((args.repeats, Q, N, args.n_classes), dtype=np.float32)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    cv_start = time.perf_counter()
    for fold_i, (tr, va) in enumerate(cv.split(X, groups)):
        fold_start = time.perf_counter()
        Z_tr, P, mask, mean, std = fit_features(X[tr], Y[tr], groups[tr], args, seed=SEED + fold_i)
        Z_va = (distances(X[va], P, mask, args.k_per_rating) - mean) / std
        model = train(Z_tr, Y[tr], args, seed=SEED + fold_i, n_classes=args.n_classes)
        oof[fold_i // args.folds][:, va] = predict_proba(model, Z_va)
        logger.info(f"[fold {fold_i + 1}/{args.repeats * args.folds}] {time.perf_counter() - fold_start:.1f}s")
    cv_s = time.perf_counter() - cv_start
    logger.info(f"CV done: {args.repeats * args.folds} fold runs in {cv_s:.1f}s")

    cv_mean, summary = write_cv_outputs(output_dir, data, oof, "M2 per-question MLP on prototype distances",
                                        setting, args.folds, args.repeats, cv_s, args.decode)

    # ---------- Final model on all train stories ----------
    final_start = time.perf_counter()
    Z, P, mask, mean, std = fit_features(X, Y, groups, args, seed=SEED)
    model = train(Z, Y, args, seed=SEED, n_classes=args.n_classes)
    final_s = time.perf_counter() - final_start

    final_config = {
        "model": "M2 per-question MLP on prototype distances",
        "k_per_rating": args.k_per_rating, "inner_folds": args.inner_folds, "empty_distance": EMPTY_DISTANCE,
        "hidden": list(args.hidden), "dropout": args.dropout,
        "weight_decay": args.weight_decay, "steps": args.steps, "lr": args.lr,
        "loss": args.loss, "decode": args.decode,
        "n_classes": args.n_classes, "answer_values": data["answer_values"], "answer_scale": data["answer_scale"],
        "question_ids": data["question_ids"], "components": data["components"],
        "embedding_model": data["embedding_model"], "embedding_dim": int(X.shape[1]),
        "train_stories": data["names"], "seed": SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save({
        "config": final_config,
        "prototypes": torch.from_numpy(P),                       # (Q, C*k, D), unit length
        "prototype_mask": torch.from_numpy(mask),                # (Q, C*k)
        "scaler_mean": torch.from_numpy(mean.astype(np.float32)),  # (Q, 1, C*k)
        "scaler_std": torch.from_numpy(std.astype(np.float32)),    # (Q, 1, C*k)
        "state_dict": model.state_dict(),
    }, output_dir / "m2_final.pt")
    with open(output_dir / "m2_final_config.json", "w", encoding="utf-8") as f:
        json.dump({**final_config, "cv_seconds": round(cv_s, 2), "final_train_seconds": round(final_s, 3),
                   "cv_mean": {m: round(float(v), 4) for m, v in zip(metric_names(args.n_classes), cv_mean)}}, f, indent=2)

    logger.info("Results:\n\n" + summary + "\n")
    logger.info(f"Final model trained in {final_s:.2f}s -> {output_dir / 'm2_final.pt'} | "
                f"full report with confusion matrices: {output_dir / 'cv_report.txt'} | log: {output_dir / 'train.log'}")


if __name__ == "__main__":
    main()

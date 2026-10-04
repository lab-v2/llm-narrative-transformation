"""
M3 for the LLM-free survey prototype: prototype distances (M2's method) in a
task-tuned space produced by M1's networks (the "encoder").

Per question q:
    embedding -> PCA + scaling -> M1's q network without its output layer
              -> H numbers tuned for q (the encoder output)
              -> Euclidean distance to q's prototypes (up to K k-means centroids per
                 rating, built in this tuned space), sorted nearest-first per rating
              -> scaling -> new per-question MLP (the "head") -> 5 logits

The encoder is M1 trained on the same stories with M1's settings (--enc-*); it is
frozen, i.e. only its output is used. The head is trained on top of it.

Cross-fitting: the encoder has partly memorized the stories it was trained on, so
in its space they sit unrealistically close to their own rating's prototypes. Each
training story's distances therefore come from an encoder + prototypes built on
the other inner folds (distances mean the same thing in every encoder's space, so
they can be pooled). Validation/test stories use the encoder + prototypes built
from all training stories.

Also reported (no extra model): "nearest prototype" = predict the rating whose
nearest prototype is closest, no head.

Outputs in --output-dir: same files as train_m1.py (cv_per_question.csv,
cv_oof_predictions.csv, cv_report.txt, train.log) plus m3_final.pt /
m3_final_config.json.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/train_m3.py \
        --embeddings-dir output_test/prototype/embeddings/text-embedding-3-large/train \
        --survey-dir output_test/prototype/survey/gpt-6-sol/train \
        --output-dir output_test/prototype/models/m3
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from train_m1 import (METRIC_NAMES, N_CLASSES, REPO_ROOT, SEED, PerQuestionMLP, fit_features as fit_pca,
                      load_data, predict_proba, question_metrics, setup_file_logging, to_features, train,
                      write_cv_outputs)
from train_m2 import build_prototypes, distances, fit_scaler

logger = logging.getLogger(__name__)


# ==========================================================
# Encoder (M1 without its output layer)
# ==========================================================

def fit_encoder(X: np.ndarray, Y: np.ndarray, args, seed: int):
    """Train M1 on (X, Y) with the --enc-* settings; return (pca, scaler, model)."""
    enc_args = SimpleNamespace(hidden=args.enc_hidden, dropout=args.enc_dropout, lr=args.enc_lr,
                               weight_decay=args.enc_weight_decay, steps=args.enc_steps, loss=args.enc_loss)
    pca, scaler = fit_pca(X, args.enc_pca_k)
    model = train(to_features(pca, scaler, X), Y, enc_args, seed=seed)
    return pca, scaler, model


def hidden_output(model: PerQuestionMLP, Z: torch.Tensor) -> torch.Tensor:
    """Last hidden layer (after ReLU) of each question's network: Z (N, F) -> (Q, N, H)."""
    h = Z.unsqueeze(0).expand(model.weights[0].shape[0], -1, -1)
    for w, b in zip(model.weights[:-1], model.biases[:-1]):
        h = torch.relu(torch.bmm(h, w) + b)
    return h


def encode(encoder, X: np.ndarray) -> np.ndarray:
    pca, scaler, model = encoder
    model.eval()
    with torch.no_grad():
        return hidden_output(model, torch.from_numpy(to_features(pca, scaler, X))).numpy()


# ==========================================================
# Features
# ==========================================================

def crossfit_distances(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, args, seed: int) -> np.ndarray:
    """Distances for training stories, each from an encoder + prototypes built without it (inner folds)."""
    k = args.k_per_rating
    F = np.zeros((Y.shape[1], len(X), N_CLASSES * k), dtype=np.float32)
    inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
    for fit_idx, out_idx in inner.split(X, groups):
        encoder = fit_encoder(X[fit_idx], Y[fit_idx], args, seed)
        P, mask = build_prototypes(encode(encoder, X[fit_idx]), Y[fit_idx], k, seed, normalize=False)
        F[:, out_idx] = distances(encode(encoder, X[out_idx]), P, mask, k, metric="euclidean")
    return F


def fit_features(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, args, seed: int):
    """Everything M3 learns before the head: cross-fitted training features, the encoder
    and prototypes from all training stories, and the scaler."""
    F_train = crossfit_distances(X, Y, groups, args, seed)
    encoder = fit_encoder(X, Y, args, seed)
    P, mask = build_prototypes(encode(encoder, X), Y, args.k_per_rating, seed, normalize=False)
    mean, std = fit_scaler(F_train)
    return (F_train - mean) / std, encoder, P, mask, mean, std


def nearest_prototype_probs(D: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Rating of the nearest prototype, as one-hot probabilities. D (Q, N, C*k) sorted per rating."""
    Q, N, S = D.shape
    nearest = D.reshape(Q, N, N_CLASSES, k)[..., 0]                              # (Q, N, C)
    nearest = np.where(mask.reshape(Q, N_CLASSES, k)[:, None, :, 0], nearest, np.inf)
    return np.eye(N_CLASSES, dtype=np.float32)[nearest.argmin(-1)]


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Train M3 (prototype distances in M1's tuned space), evaluated with repeated CV")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Train embeddings (<story>.json from embed_story.py)")
    parser.add_argument("--survey-dir", type=str, required=True, help="Train survey ratings (<story>.json from survey_story.py)")
    parser.add_argument("--split-csv", type=str, default=str(REPO_ROOT / "prototype" / "split.csv"),
                        help="Story -> individualistic/collectivistic, used to stratify folds")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write CV results and the final model")
    # Encoder (M1) setting
    parser.add_argument("--enc-pca-k", type=int, default=32, help="Encoder PCA components (default: 32)")
    parser.add_argument("--enc-hidden", type=int, nargs="+", default=[64, 32],
                        help="Encoder hidden layers; the last one is the tuned space (default: 64 32)")
    parser.add_argument("--enc-dropout", type=float, default=0.3, help="Encoder dropout (default: 0.3)")
    parser.add_argument("--enc-weight-decay", type=float, default=1e-2, help="Encoder weight decay (default: 0.01)")
    parser.add_argument("--enc-steps", type=int, default=400, help="Encoder training steps (default: 400)")
    parser.add_argument("--enc-lr", type=float, default=1e-3, help="Encoder learning rate (default: 0.001)")
    parser.add_argument("--enc-loss", choices=["ce", "emd", "ce+emd"], default="ce", help="Encoder loss (default: ce)")
    # Prototype features
    parser.add_argument("--k-per-rating", type=int, default=3, help="k-means prototypes per (question, rating) (default: 3)")
    parser.add_argument("--inner-folds", type=int, default=5, help="Inner folds for cross-fitted training features (default: 5)")
    # Head setting
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 16],
                        help="Head hidden layers, e.g. --hidden 32 16 (default); --hidden with no values = no hidden layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Head dropout (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Head weight decay (default: 0.01)")
    parser.add_argument("--steps", type=int, default=400, help="Head training steps (default: 400)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate (default: 0.001)")
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
    k = args.k_per_rating
    enc_str = "-".join(map(str, args.enc_hidden))
    hidden_str = "-".join(map(str, args.hidden)) or "none"
    setting = (f"encoder: pca_k={args.enc_pca_k} hidden={enc_str} dropout={args.enc_dropout} "
               f"weight_decay={args.enc_weight_decay} steps={args.enc_steps} lr={args.enc_lr} loss={args.enc_loss} | "
               f"prototypes: k_per_rating={k} inner_folds={args.inner_folds} euclidean | "
               f"head: hidden={hidden_str} dropout={args.dropout} weight_decay={args.weight_decay} "
               f"steps={args.steps} lr={args.lr} loss={args.loss} decode={args.decode}")
    group_counts = {str(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True))}
    logger.info(f"{N} stories ({group_counts}), {Q} questions, "
                f"embeddings {data['embedding_model']} ({X.shape[1]} dims)")
    logger.info(f"Setting: {setting} | {args.repeats}x{args.folds}-fold CV")

    # ---------- Cross-validation ----------
    oof = np.zeros((args.repeats, Q, N, N_CLASSES), dtype=np.float32)
    oof_nearest = np.zeros_like(oof)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    cv_start = time.perf_counter()
    for fold_i, (tr, va) in enumerate(cv.split(X, groups)):
        fold_start = time.perf_counter()
        Z_tr, encoder, P, mask, mean, std = fit_features(X[tr], Y[tr], groups[tr], args, seed=SEED + fold_i)
        D_va = distances(encode(encoder, X[va]), P, mask, k, metric="euclidean")
        model = train(Z_tr, Y[tr], args, seed=SEED + fold_i)
        repeat = fold_i // args.folds
        oof[repeat][:, va] = predict_proba(model, (D_va - mean) / std)
        oof_nearest[repeat][:, va] = nearest_prototype_probs(D_va, mask, k)
        logger.info(f"[fold {fold_i + 1}/{args.repeats * args.folds}] {time.perf_counter() - fold_start:.1f}s")
    cv_s = time.perf_counter() - cv_start
    logger.info(f"CV done: {args.repeats * args.folds} fold runs in {cv_s:.1f}s")

    cv_mean, summary = write_cv_outputs(output_dir, data, oof, "M3 per-question MLP on prototype distances "
                                        "in M1's tuned space", setting, args.folds, args.repeats, cv_s, args.decode)
    nearest_mean = np.mean([[question_metrics(Y[:, q], oof_nearest[r, q].argmax(-1)) for q in range(Q)]
                            for r in range(args.repeats)], axis=(0, 1))
    nearest_line = (f"Nearest prototype (no head), same CV: QWK={nearest_mean[0]:.3f}  "
                    f"balanced_acc={nearest_mean[1]:.3f}  exact={nearest_mean[2]:.3f}  within1={nearest_mean[3]:.3f}  "
                    f"QWK3={nearest_mean[4]:.3f}")
    with open(output_dir / "cv_report.txt", "a", encoding="utf-8") as f:
        f.write("\n" + nearest_line + "\n")

    # ---------- Final model on all train stories ----------
    final_start = time.perf_counter()
    Z, encoder, P, mask, mean, std = fit_features(X, Y, groups, args, seed=SEED)
    model = train(Z, Y, args, seed=SEED)
    final_s = time.perf_counter() - final_start
    pca, scaler, enc_model = encoder

    final_config = {
        "model": "M3 per-question MLP on prototype distances in M1's tuned space",
        "enc_pca_k": args.enc_pca_k, "enc_hidden": list(args.enc_hidden), "enc_dropout": args.enc_dropout,
        "enc_weight_decay": args.enc_weight_decay, "enc_steps": args.enc_steps, "enc_lr": args.enc_lr,
        "enc_loss": args.enc_loss,
        "k_per_rating": k, "inner_folds": args.inner_folds, "distance": "euclidean",
        "hidden": list(args.hidden), "dropout": args.dropout,
        "weight_decay": args.weight_decay, "steps": args.steps, "lr": args.lr,
        "loss": args.loss, "decode": args.decode,
        "n_classes": N_CLASSES, "question_ids": data["question_ids"], "components": data["components"],
        "embedding_model": data["embedding_model"], "embedding_dim": int(X.shape[1]),
        "train_stories": data["names"], "seed": SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save({
        "config": final_config,
        "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "encoder_state_dict": enc_model.state_dict(),
        "prototypes": torch.from_numpy(P),                          # (Q, C*k, H)
        "prototype_mask": torch.from_numpy(mask),                   # (Q, C*k)
        "dist_mean": torch.from_numpy(mean.astype(np.float32)),     # (Q, 1, C*k)
        "dist_std": torch.from_numpy(std.astype(np.float32)),       # (Q, 1, C*k)
        "state_dict": model.state_dict(),
    }, output_dir / "m3_final.pt")
    with open(output_dir / "m3_final_config.json", "w", encoding="utf-8") as f:
        json.dump({**final_config, "cv_seconds": round(cv_s, 2), "final_train_seconds": round(final_s, 3),
                   "cv_mean": {m: round(float(v), 4) for m, v in zip(METRIC_NAMES, cv_mean)},
                   "cv_mean_nearest_prototype": {m: round(float(v), 4) for m, v in zip(METRIC_NAMES, nearest_mean)}},
                  f, indent=2)

    logger.info("Results:\n\n" + summary + "\n\n" + nearest_line + "\n")
    logger.info(f"Final model trained in {final_s:.2f}s -> {output_dir / 'm3_final.pt'} | "
                f"full report with confusion matrices: {output_dir / 'cv_report.txt'} | log: {output_dir / 'train.log'}")


if __name__ == "__main__":
    main()

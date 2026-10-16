"""
M4 for the LLM-free survey prototype: ONE shared network for all questions that
reads the story AND the question's text.

Input for a (story, question) pair:
    story embedding    -> PCA to --pca-k (fitted on training stories) -> standardize
    question embedding -> PCA to --question-pca-k (fitted on training questions) -> standardize
    concatenated -> shared MLP -> n_classes logits
Every (story, question) pair is a training example. The loss keeps each question's own
class weights (rare answers count more), as in M1-M3.

Two evaluations:
  1. Repeated stratified K-fold over stories (new stories, same questions), comparable
     to M1-M3. Outputs as train_m1.py: cv_per_question.csv, cv_oof_predictions.csv,
     cv_report.txt.
  2. Unseen questions (--question-folds > 0): stories split into --folds folds and
     questions into --question-folds folds; each model is trained on training stories x
     training questions and predicts held-out stories x held-out questions, so both the
     story and the question are new. Same files, written to <output-dir>/unseen_questions/.

Final model (all train stories x all questions) -> m4_final.pt / m4_final_config.json,
including the question PCA so new questions could be projected later.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/train_m4.py \
        --embeddings-dir output_test/prototype/embeddings/text-embedding-3-large/train \
        --survey-dir output_test/prototype/survey_binary/gpt-6-sol/train \
        --question-embeddings output_test/prototype/embeddings/text-embedding-3-large/questions/collectivistic_questions.json \
        --output-dir output_test/prototype/models/m4_binary
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
import torch.nn as nn
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold

from train_m1 import (LOSSES, REPO_ROOT, SEED, class_weights, fit_features, load_data, metric_names,
                      setup_file_logging, to_features, write_cv_outputs)

logger = logging.getLogger(__name__)


# ==========================================================
# Model
# ==========================================================

class SharedMLP(nn.Module):
    """One MLP for all questions: (..., F) -> (..., n_classes)."""

    def __init__(self, in_dim: int, hidden: tuple, n_classes: int, dropout: float):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pair_features(Zs: np.ndarray, Zq: np.ndarray) -> np.ndarray:
    """Story features (N, Fs) x question features (Q, Fq) -> (Q, N, Fs + Fq) for every pair."""
    Q, N = len(Zq), len(Zs)
    return np.concatenate([np.broadcast_to(Zs[None], (Q, N, Zs.shape[1])),
                           np.broadcast_to(Zq[:, None], (Q, N, Zq.shape[1]))], axis=-1).astype(np.float32)


def train_shared(X: np.ndarray, Y: np.ndarray, args, seed: int, n_classes: int) -> SharedMLP:
    """X (Q, N, F) pair features, Y (N, Q) classes."""
    torch.manual_seed(seed)
    model = SharedMLP(X.shape[-1], tuple(args.hidden), n_classes, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x, y, cw = torch.from_numpy(X), torch.from_numpy(Y.T.copy()), class_weights(Y, n_classes)
    loss_fn = LOSSES[args.loss]
    model.train()
    for _ in range(args.steps):
        opt.zero_grad()
        loss_fn(model(x), y, cw).backward()  # per-question weighted loss, summed over questions
        opt.step()
    return model


def predict_shared(model: SharedMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=-1).numpy()


# ==========================================================
# Data
# ==========================================================

def load_question_embeddings(path: Path, question_ids: list, embedding_model: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        qe = json.load(f)
    if qe["model"] != embedding_model:
        sys.exit(f"Question embeddings are {qe['model']}, story embeddings are {embedding_model}")
    by_id = {q["id"]: q["embedding"] for q in qe["questions"]}
    missing = [q for q in question_ids if q not in by_id]
    if missing:
        sys.exit(f"No question embedding for questions {missing}")
    return np.asarray([by_id[q] for q in question_ids], dtype=np.float32)


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Train M4 (one shared network on story + question embeddings)")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Train story embeddings (from embed_story.py)")
    parser.add_argument("--survey-dir", type=str, required=True, help="Train survey answers (from survey_story.py)")
    parser.add_argument("--question-embeddings", type=str, required=True, help="Question embeddings (from embed_questions.py)")
    parser.add_argument("--split-csv", type=str, default=str(REPO_ROOT / "prototype" / "split.csv"),
                        help="Story -> individualistic/collectivistic, used to stratify folds")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write CV results and the final model")
    parser.add_argument("--pca-k", type=int, default=32, help="Story PCA components (default: 32)")
    parser.add_argument("--question-pca-k", type=int, default=8,
                        help="Question PCA components; at most the number of training questions - 1 (default: 8)")
    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 32], help="Hidden layer sizes (default: 64 32)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--steps", type=int, default=400, help="Full-batch training steps (default: 400)")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default: 0.001)")
    parser.add_argument("--loss", choices=["ce", "emd", "ce+emd"], default="ce", help="Loss (default: ce)")
    parser.add_argument("--decode", choices=["argmax", "mean"], default="argmax", help="Answer rule (default: argmax)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--question-folds", type=int, default=5,
                        help="Folds over questions for the unseen-question evaluation; 0 skips it (default: 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, "train.log")

    data = load_data(Path(args.embeddings_dir), Path(args.survey_dir), Path(args.split_csv))
    X, Y, groups = data["X"], data["Y"], data["groups"]
    N, Q = Y.shape
    n_classes = data["n_classes"]
    QE = load_question_embeddings(Path(args.question_embeddings), data["question_ids"], data["embedding_model"])
    hidden_str = "-".join(map(str, args.hidden)) or "none"
    setting = (f"pca_k={args.pca_k} question_pca_k={args.question_pca_k} hidden={hidden_str} dropout={args.dropout} "
               f"weight_decay={args.weight_decay} steps={args.steps} lr={args.lr} loss={args.loss} decode={args.decode}")
    group_counts = {str(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True))}
    logger.info(f"{N} stories ({group_counts}), {Q} questions, answers {data['answer_scale']}, "
                f"embeddings {data['embedding_model']} ({X.shape[1]} dims)")
    logger.info(f"Setting: {setting} | {args.repeats}x{args.folds}-fold CV over stories"
                + (f" + unseen questions ({args.folds} story folds x {args.question_folds} question folds)" if args.question_folds else ""))

    # ---------- 1. New stories, same questions ----------
    oof = np.zeros((args.repeats, Q, N, n_classes), dtype=np.float32)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    cv_start = time.perf_counter()
    q_pca, q_scaler = fit_features(QE, args.question_pca_k)
    Zq_all = to_features(q_pca, q_scaler, QE)
    for fold_i, (tr, va) in enumerate(cv.split(X, groups)):
        pca, scaler = fit_features(X[tr], args.pca_k)
        model = train_shared(pair_features(to_features(pca, scaler, X[tr]), Zq_all), Y[tr], args, SEED + fold_i, n_classes)
        oof[fold_i // args.folds][:, va] = predict_shared(model, pair_features(to_features(pca, scaler, X[va]), Zq_all))
    cv_s = time.perf_counter() - cv_start
    logger.info(f"CV over stories done: {args.repeats * args.folds} fold runs in {cv_s:.1f}s")
    cv_mean, summary = write_cv_outputs(output_dir, data, oof, "M4 shared network on story + question embeddings",
                                        setting, args.folds, args.repeats, cv_s, args.decode)

    # ---------- 2. New stories AND new questions ----------
    unseen_mean, unseen_summary = None, ""
    if args.question_folds:
        unseen_dir = output_dir / "unseen_questions"
        unseen_dir.mkdir(exist_ok=True)
        oof_u = np.zeros((1, Q, N, n_classes), dtype=np.float32)
        story_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        question_cv = KFold(n_splits=args.question_folds, shuffle=True, random_state=SEED)
        u_start = time.perf_counter()
        for s_i, (tr, va) in enumerate(story_cv.split(X, groups)):
            pca, scaler = fit_features(X[tr], args.pca_k)
            Zs_tr, Zs_va = to_features(pca, scaler, X[tr]), to_features(pca, scaler, X[va])
            for q_i, (qtr, qva) in enumerate(question_cv.split(QE)):
                qp, qs = fit_features(QE[qtr], min(args.question_pca_k, len(qtr) - 1))  # fitted on training questions only
                model = train_shared(pair_features(Zs_tr, to_features(qp, qs, QE[qtr])), Y[np.ix_(tr, qtr)],
                                     args, SEED + 100 * s_i + q_i, n_classes)
                probs = predict_shared(model, pair_features(Zs_va, to_features(qp, qs, QE[qva])))  # (len(qva), len(va), C)
                oof_u[0][np.ix_(qva, va)] = probs
        u_s = time.perf_counter() - u_start
        logger.info(f"Unseen-question evaluation done: {args.folds * args.question_folds} runs in {u_s:.1f}s")
        unseen_mean, unseen_summary = write_cv_outputs(
            unseen_dir, data, oof_u, "M4 on UNSEEN questions (new stories x new questions)", setting,
            args.folds, 1, u_s, args.decode)

    # ---------- Final model on all train stories x all questions ----------
    final_start = time.perf_counter()
    pca, scaler = fit_features(X, args.pca_k)
    model = train_shared(pair_features(to_features(pca, scaler, X), Zq_all), Y, args, SEED, n_classes)
    final_s = time.perf_counter() - final_start

    names = metric_names(n_classes)
    final_config = {
        "model": "M4 shared network on story + question embeddings",
        "pca_k": args.pca_k, "question_pca_k": args.question_pca_k, "hidden": list(args.hidden),
        "dropout": args.dropout, "weight_decay": args.weight_decay, "steps": args.steps, "lr": args.lr,
        "loss": args.loss, "decode": args.decode,
        "n_classes": n_classes, "answer_values": data["answer_values"], "answer_scale": data["answer_scale"],
        "question_ids": data["question_ids"], "components": data["components"],
        "question_embeddings": args.question_embeddings,
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
        "question_features": torch.from_numpy(Zq_all),                     # (Q, question_pca_k), for the 20 questions
        "question_pca_components": torch.from_numpy(q_pca.components_.astype(np.float32)),
        "question_pca_mean": torch.from_numpy(q_pca.mean_.astype(np.float32)),
        "question_scaler_mean": torch.from_numpy(q_scaler.mean_.astype(np.float32)),
        "question_scaler_scale": torch.from_numpy(q_scaler.scale_.astype(np.float32)),
        "state_dict": model.state_dict(),
    }, output_dir / "m4_final.pt")
    with open(output_dir / "m4_final_config.json", "w", encoding="utf-8") as f:
        json.dump({**final_config, "cv_seconds": round(cv_s, 2), "final_train_seconds": round(final_s, 3),
                   "cv_mean": {m: round(float(v), 4) for m, v in zip(names, cv_mean)},
                   "cv_mean_unseen_questions": (None if unseen_mean is None else
                                                {m: round(float(v), 4) for m, v in zip(names, unseen_mean)})}, f, indent=2)

    logger.info("Results (new stories, same questions):\n\n" + summary + "\n")
    if unseen_summary:
        logger.info("Results (new stories AND new questions):\n\n" + unseen_summary + "\n")
    logger.info(f"Final model trained in {final_s:.2f}s -> {output_dir / 'm4_final.pt'} | "
                f"reports: {output_dir / 'cv_report.txt'}, {output_dir / 'unseen_questions' / 'cv_report.txt'}")


if __name__ == "__main__":
    main()

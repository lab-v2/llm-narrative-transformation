"""
M1 for the LLM-free survey prototype: one small neural network per survey
question, predicting the LLM's 1-5 rating from the story's OpenAI embedding.

Pipeline (fitted on training stories only):
    embedding (e.g. 3072) -> PCA to k -> standardize -> per-question MLP -> 5 logits

- One network per question, no shared weights. For speed the 20 networks are
  trained side by side in one module (PerQuestionMLP); since their parameters
  are disjoint and AdamW updates each parameter independently, this is
  identical to training them one by one.
- Loss: cross-entropy weighted by 1/frequency of each rating in the training
  stories (sklearn "balanced" weights), so rare ratings count more.
- Full-batch AdamW (weight decay) + dropout, fixed number of steps.
- One fixed setting (override with the flags below). Repeated stratified K-fold
  CV (stratified by individualistic/collectivistic) measures how well it does;
  then the model is retrained on all train stories and saved for the test set.

Outputs in --output-dir:
    cv_per_question.csv        metrics per question (out-of-fold, averaged over repeats)
    cv_oof_predictions.csv     out-of-fold prediction + probabilities per story/question/repeat
    cv_report.txt              readable summary + per-question confusion matrices
    m1_final.pt / m1_final_config.json   final model (PCA + scaler + 20 networks)
    train.log                  everything printed to the terminal (command, progress, warnings, results)

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/train_m1.py \
        --embeddings-dir output_test/prototype/embeddings/text-embedding-3-large/train \
        --survey-dir output_test/prototype/survey/gpt-6-sol/train \
        --output-dir output_test/prototype/models/m1
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
torch.set_num_threads(1)  # tiny networks run faster single-threaded

SEED = 42
N_CLASSES = 5                              # ratings 1..5 -> class 0..4
TO_3_LEVEL = np.array([0, 0, 1, 2, 2])     # 1-2 low, 3 mid, 4-5 high
METRIC_NAMES = ["qwk", "balanced_acc", "exact_acc", "within1_acc", "qwk_3level", "balanced_acc_3level"]
YES_NO_SCALE = "yes/no (1/0)"   # survey_story.py's answer_scale for a yes/no questionnaire


def answer_values(scale: str) -> list:
    """Raw answer values for a survey's answer_scale: yes/no -> [0, 1]; the 1-5 survey -> [1, 2, 3, 4, 5]."""
    return [0, 1] if scale == YES_NO_SCALE else [1, 2, 3, 4, 5]


def metric_names(n_classes: int) -> list:
    """Yes/no: kappa (stored as "qwk"; QWK on two classes is plain kappa), balanced acc, accuracy, F1 of yes, AUC."""
    return ["qwk", "balanced_acc", "exact_acc", "f1_yes", "auc"] if n_classes == 2 else METRIC_NAMES


# ==========================================================
# Data
# ==========================================================

def load_data(embeddings_dir: Path, survey_dir: Path, split_csv: Path):
    with open(split_csv, encoding="utf-8") as f:
        group = {r["story_name"]: r["label"] for r in csv.DictReader(f)}

    emb_files = {p.stem: p for p in embeddings_dir.glob("*.json")}
    survey_files = {p.stem: p for p in survey_dir.glob("*.json")}
    if not emb_files:
        sys.exit(f"No embeddings found in {embeddings_dir}")
    if set(emb_files) != set(survey_files):
        sys.exit(f"Embeddings and surveys don't cover the same stories:\n"
                 f"  only embeddings: {sorted(set(emb_files) - set(survey_files))}\n"
                 f"  only surveys:    {sorted(set(survey_files) - set(emb_files))}")
    names = sorted(emb_files)
    missing = [n for n in names if n not in group]
    if missing:
        sys.exit(f"Stories missing from {split_csv}: {missing}")

    X, Y, emb_models, question_ids, questions_file, scale = [], [], set(), None, None, None
    for name in names:
        with open(emb_files[name], encoding="utf-8") as f:
            emb = json.load(f)
        with open(survey_files[name], encoding="utf-8") as f:
            survey = json.load(f)

        ratings = survey["ratings"]
        if question_ids is None:
            question_ids = sorted(ratings, key=int)
            questions_file = survey.get("questions_file")
            scale = survey.get("answer_scale", "1-5")
            values = answer_values(scale)
        if survey.get("answer_scale", "1-5") != scale:
            sys.exit(f"{name}: answer scale {survey.get('answer_scale', '1-5')} differs from the other stories ({scale})")
        if sorted(ratings, key=int) != question_ids or any(ratings[q] not in values for q in question_ids):
            sys.exit(f"{name}: survey ratings incomplete or don't match the other stories")

        X.append(emb["embedding"])
        Y.append([values.index(ratings[q]) for q in question_ids])
        emb_models.add(emb["model"])

    if len(emb_models) != 1:
        sys.exit(f"Mixed embedding models: {emb_models}")

    components = {}
    if questions_file and Path(questions_file).exists():
        with open(questions_file, encoding="utf-8") as f:
            components = {str(q["id"]): q["component"] for q in json.load(f)["questions"]}

    return {
        "names": names,
        "groups": np.array([group[n] for n in names]),
        "X": np.asarray(X, dtype=np.float32),
        "Y": np.asarray(Y, dtype=np.int64),          # (N, Q), class = index into answer_values
        "n_classes": len(values),
        "answer_values": values,                     # [1..5], or [0, 1] for yes/no
        "answer_scale": scale,
        "question_ids": question_ids,
        "components": [components.get(q, "") for q in question_ids],
        "embedding_model": emb_models.pop(),
    }


# ==========================================================
# Model
# ==========================================================

class PerQuestionMLP(nn.Module):
    """n_questions independent MLPs evaluated side by side (no shared weights)."""

    def __init__(self, n_questions: int, in_dim: int, hidden: tuple, n_classes: int, dropout: float):
        super().__init__()
        dims = [in_dim, *hidden, n_classes]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims, dims[1:]):
            bound = 1 / math.sqrt(d_in)  # same init range as nn.Linear
            self.weights.append(nn.Parameter(torch.empty(n_questions, d_in, d_out).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n_questions, 1, d_out).uniform_(-bound, bound)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) shared by all questions, or (Q, N, in_dim) per question -> logits (Q, N, n_classes)"""
        h = x if x.dim() == 3 else x.unsqueeze(0).expand(self.weights[0].shape[0], -1, -1)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.bmm(h, w) + b
            if i < last:
                h = self.dropout(torch.relu(h))
        return h


def class_weights(Y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """(N, Q) -> (Q, C): n / (n_present_classes * count) per class, 0 for absent classes."""
    weights = np.zeros((Y.shape[1], n_classes), dtype=np.float32)
    for q in range(Y.shape[1]):
        counts = np.bincount(Y[:, q], minlength=n_classes)
        present = counts > 0
        weights[q, present] = len(Y) / (present.sum() * counts[present])
    return torch.from_numpy(weights)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Per-question weighted cross-entropy (same as F.cross_entropy(weight=...)), summed over questions."""
    nll = -torch.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (Q, N)
    w = cw.gather(1, y)                                                                # (Q, N)
    return ((w * nll).sum(1) / w.sum(1)).sum()


def weighted_emd(logits: torch.Tensor, y: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """Order-aware loss: squared earth mover's distance between the predicted distribution and the
    true rating, i.e. sum over k of (P(pred <= k) - [true <= k])^2. Probability mass far from the
    true rating costs more than mass next to it. Same class weighting as weighted_ce."""
    cdf = torch.cumsum(torch.softmax(logits, dim=-1), dim=-1)[..., :-1]              # (Q, N, C-1)
    target = (torch.arange(cdf.shape[-1]) >= y.unsqueeze(-1)).float()                # CDF of the true rating
    emd = ((cdf - target) ** 2).sum(-1)                                               # (Q, N)
    w = cw.gather(1, y)
    return ((w * emd).sum(1) / w.sum(1)).sum()


LOSSES = {
    "ce": weighted_ce,
    "emd": weighted_emd,
    "ce+emd": lambda logits, y, cw: weighted_ce(logits, y, cw) + weighted_emd(logits, y, cw),
}


def decode(probs: np.ndarray, mode: str = "argmax") -> np.ndarray:
    """Probabilities (..., C) -> class indices. argmax = most likely rating;
    mean = expected rating rounded to the nearest class (uses the order of the ratings)."""
    if mode == "mean":
        return np.clip(np.floor((probs * np.arange(probs.shape[-1])).sum(-1) + 0.5), 0, probs.shape[-1] - 1).astype(np.int64)
    return probs.argmax(-1)


# numpy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow / invalid value
# encountered in matmul" warnings inside PCA even though the results are correct; silence
# them here and check the features are finite instead.
FP_IGNORE = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


def fit_features(X_fit: np.ndarray, k: int):
    """PCA to k + standardize, fitted on X_fit only."""
    with np.errstate(**FP_IGNORE):
        pca = PCA(n_components=k, svd_solver="full").fit(X_fit)
        scaler = StandardScaler().fit(pca.transform(X_fit))
    return pca, scaler


def to_features(pca, scaler, X: np.ndarray) -> np.ndarray:
    with np.errstate(**FP_IGNORE):
        Z = scaler.transform(pca.transform(X)).astype(np.float32)
    if not np.isfinite(Z).all():
        raise FloatingPointError("PCA features contain NaN/inf - check the embeddings")
    return Z


def train(Z: np.ndarray, Y: np.ndarray, args, seed: int, n_classes: int = N_CLASSES) -> PerQuestionMLP:
    """Z: (N, F) shared by all questions, or (Q, N, F) per question."""
    torch.manual_seed(seed)
    model = PerQuestionMLP(Y.shape[1], Z.shape[-1], tuple(args.hidden), n_classes, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x, y, cw = torch.from_numpy(Z), torch.from_numpy(Y.T.copy()), class_weights(Y, n_classes)
    loss_fn = LOSSES[getattr(args, "loss", "ce")]
    model.train()
    for _ in range(args.steps):
        opt.zero_grad()
        loss_fn(model(x), y, cw).backward()
        opt.step()
    return model


def train_logreg(Z: np.ndarray, Y: np.ndarray, C: float, n_classes: int = N_CLASSES) -> PerQuestionMLP:
    """One L2-regularized logistic regression per question (rare answers weighted up, like the networks),
    packed into a PerQuestionMLP with no hidden layer so prediction code is unchanged."""
    Q, k = Y.shape[1], Z.shape[1]
    W = np.zeros((Q, k, n_classes), dtype=np.float32)
    b = np.full((Q, 1, n_classes), -1e4, dtype=np.float32)       # answers never seen in training stay impossible
    for q in range(Q):
        classes = np.unique(Y[:, q])
        if len(classes) == 1:
            b[q, 0, classes[0]] = 0.0
            continue
        with np.errstate(**FP_IGNORE):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z, Y[:, q])
        if len(classes) == 2:                                      # sklearn: P(classes[1]) = sigmoid(w.z + b)
            W[q, :, classes[1]] = clf.coef_[0]
            b[q, 0, classes[0]], b[q, 0, classes[1]] = 0.0, clf.intercept_[0]
        else:                                                      # multinomial: softmax(coef.z + intercept)
            W[q][:, classes] = clf.coef_.T
            b[q, 0, classes] = clf.intercept_
    model = PerQuestionMLP(Q, k, (), n_classes, 0.0)
    with torch.no_grad():
        model.weights[0].copy_(torch.from_numpy(W))
        model.biases[0].copy_(torch.from_numpy(b))
    return model


def predict_proba(model: PerQuestionMLP, Z: np.ndarray) -> np.ndarray:
    """-> (Q, N, C) probabilities"""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(Z)), dim=-1).numpy()


# ==========================================================
# Metrics
# ==========================================================

def question_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES, probs: np.ndarray = None) -> list:
    """Metrics in metric_names(n_classes) order. probs (N, C) is only needed for the yes/no AUC."""
    if n_classes == 2:
        auc = roc_auc_score(y_true, probs[:, 1]) if probs is not None and len(np.unique(y_true)) == 2 else np.nan
        return [
            float(np.nan_to_num(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1]))),
            balanced_accuracy_score(y_true, y_pred),
            float(np.mean(y_true == y_pred)),
            f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            float(auc),
        ]
    t3, p3 = TO_3_LEVEL[y_true], TO_3_LEVEL[y_pred]
    return [
        float(np.nan_to_num(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(N_CLASSES)))),
        balanced_accuracy_score(y_true, y_pred),
        float(np.mean(y_true == y_pred)),
        float(np.mean(np.abs(y_true - y_pred) <= 1)),
        float(np.nan_to_num(cohen_kappa_score(t3, p3, weights="quadratic", labels=range(3)))),
        balanced_accuracy_score(t3, p3),
    ]


# ==========================================================
# Shared by the M1/M2 training scripts
# ==========================================================

def setup_file_logging(output_dir: Path, filename: str):
    """Everything shown in the terminal (logs + any warnings) also goes to <output_dir>/<filename>."""
    file_handler = logging.FileHandler(output_dir / filename, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.captureWarnings(True)
    logger.info("Command: " + " ".join(sys.argv))


def write_cv_outputs(output_dir: Path, data: dict, oof: np.ndarray, title: str, setting: str,
                     folds: int, repeats: int, cv_s: float, decode_mode: str = "argmax"):
    """Score out-of-fold probabilities oof (R, Q, N, C), turned into ratings with decode(decode_mode);
    write cv_per_question.csv, cv_oof_predictions.csv and cv_report.txt. Returns (mean metrics, summary text)."""
    Y, groups = data["Y"], data["groups"]
    N, Q = Y.shape
    n = data.get("n_classes", N_CLASSES)
    vals = data.get("answer_values", list(range(1, N_CLASSES + 1)))
    names = metric_names(n)
    pred = decode(oof, decode_mode)                                 # (R, Q, N)
    per_repeat = np.array([[question_metrics(Y[:, q], pred[r, q], n, oof[r, q]) for q in range(Q)]
                           for r in range(repeats)])               # (R, Q, M)
    per_q = np.nanmean(per_repeat, axis=0)                          # (Q, M)
    mean = np.nanmean(per_q, axis=0)                                # (M,)
    qwk_sd = per_repeat[:, :, 0].mean(1).std()
    train_counts = ["/".join(str(c) for c in np.bincount(Y[:, q], minlength=n)) for q in range(Q)]
    confusions = [sum(confusion_matrix(Y[:, q], pred[r, q], labels=range(n))
                      for r in range(repeats)) for q in range(Q)]

    with open(output_dir / "cv_per_question.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "component", f"train_counts_{vals[0]}to{vals[-1]}", *names])
        for q in range(Q):
            writer.writerow([data["question_ids"][q], data["components"][q], train_counts[q],
                             *[round(float(v), 4) for v in per_q[q]]])

    with open(output_dir / "cv_oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["story_name", "label", "repeat", "question_id", "true_rating", "pred_rating",
                         *[f"p{v}" for v in vals]])
        for r in range(repeats):
            for q in range(Q):
                for i in range(N):
                    p = oof[r, q, i]
                    writer.writerow([data["names"][i], groups[i], r, data["question_ids"][q], vals[Y[i, q]],
                                     vals[int(pred[r, q, i])], *[round(float(v), 4) for v in p]])

    if n == 2:
        lines = [
            f"{title} | {N} train stories | {Q} questions | yes/no answers | {repeats}x{folds}-fold CV ({cv_s:.1f}s)",
            f"Setting: {setting}",
            "",
            "Mean over questions (out-of-fold, averaged over repeats):",
            f"  kappa={mean[0]:.3f} (sd over repeats {qwk_sd:.3f})  balanced_acc={mean[1]:.3f}  accuracy={mean[2]:.3f}  "
            f"F1(yes)={mean[3]:.3f}  AUC={mean[4]:.3f}",
            "  (kappa 0 = chance, 1 = perfect; balanced_acc and AUC 0.5 = chance)",
            "",
            "Per question:",
            f"  {'q':>4} {'train no/yes':>13} {'kappa':>6} {'balacc':>6} {'acc':>6} {'F1yes':>6} {'AUC':>6}  component",
        ]
        for q in range(Q):
            m = per_q[q]
            lines.append(f"  {data['question_ids'][q]:>4} {train_counts[q]:>13} {m[0]:>6.3f} {m[1]:>6.3f} {m[2]:>6.3f} "
                         f"{m[3]:>6.3f} {m[4]:>6.3f}  {data['components'][q]}")
        return _finish_report(output_dir, data, lines, confusions, vals, repeats, mean)

    lines = [
        f"{title} | {N} train stories | {Q} questions | {repeats}x{folds}-fold CV ({cv_s:.1f}s)",
        f"Setting: {setting}",
        "",
        "Mean over questions (out-of-fold, averaged over repeats):",
        f"  1-5 scale : QWK={mean[0]:.3f} (sd over repeats {qwk_sd:.3f})  balanced_acc={mean[1]:.3f}  "
        f"exact={mean[2]:.3f}  within1={mean[3]:.3f}",
        f"  3-level   : QWK={mean[4]:.3f}  balanced_acc={mean[5]:.3f}",
        "  (QWK 0 = chance, 1 = perfect; balanced_acc for always predicting one rating ~0.20-0.25)",
        "",
        "Per question:",
        f"  {'q':>4} {'train 1/2/3/4/5':>17} {'QWK':>6} {'balacc':>6} {'exact':>6} {'within1':>7} {'QWK3':>6}  component",
    ]
    for q in range(Q):
        m = per_q[q]
        lines.append(f"  {data['question_ids'][q]:>4} {train_counts[q]:>17} {m[0]:>6.3f} {m[1]:>6.3f} {m[2]:>6.3f} "
                     f"{m[3]:>7.3f} {m[4]:>6.3f}  {data['components'][q]}")
    return _finish_report(output_dir, data, lines, confusions, vals, repeats, mean)


def _finish_report(output_dir: Path, data: dict, lines: list, confusions: list, vals: list, repeats: int, mean):
    """Append the confusion matrices, write cv_report.txt; return (mean metrics, summary text)."""
    summary_len = len(lines)
    lines += ["", f"Confusion matrices (rows = LLM rating, cols = predicted; summed over {repeats} repeats, "
                  f"so each story counts {repeats}x):"]
    for q in range(len(data["question_ids"])):
        lines.append(f"  q{data['question_ids'][q]} {data['components'][q]}")
        lines.append("        " + " ".join(f"{p:>4}" for p in vals))
        for t, v_true in enumerate(vals):
            lines.append(f"     {v_true}: " + " ".join(f"{v:>4}" for v in confusions[q][t]))
    (output_dir / "cv_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return mean, "\n".join(lines[:summary_len])


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Train M1 (per-question MLP on embeddings), evaluated with repeated CV")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Train embeddings (<story>.json from embed_story.py)")
    parser.add_argument("--survey-dir", type=str, required=True, help="Train survey ratings (<story>.json from survey_story.py)")
    parser.add_argument("--split-csv", type=str, default=str(REPO_ROOT / "prototype" / "split.csv"),
                        help="Story -> individualistic/collectivistic, used to stratify folds")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write CV results and the final model")
    # Model setting
    parser.add_argument("--pca-k", type=int, default=32, help="PCA components (default: 32)")
    parser.add_argument("--classifier", choices=["mlp", "logreg"], default="mlp",
                        help="mlp = small neural network per question (default); logreg = L2 logistic regression per question")
    parser.add_argument("--C", type=float, default=0.01,
                        help="[logreg] inverse regularization strength; smaller = stronger (default: 0.01)")
    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 32],
                        help="Hidden layer sizes, e.g. --hidden 64 32 (default); --hidden with no values = no hidden layer")
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
    n_classes = data["n_classes"]
    hidden_str = "-".join(map(str, args.hidden)) or "none"
    if args.classifier == "logreg":
        args.hidden = []
        setting = f"pca_k={args.pca_k} classifier=logreg C={args.C} class_weight=balanced decode={args.decode}"
    else:
        setting = (f"pca_k={args.pca_k} hidden={hidden_str} dropout={args.dropout} "
                   f"weight_decay={args.weight_decay} steps={args.steps} lr={args.lr} loss={args.loss} decode={args.decode}")
    title = "M1 per-question logistic regression" if args.classifier == "logreg" else "M1 per-question MLP"
    fit = (lambda Z, Yt, seed: train_logreg(Z, Yt, args.C, n_classes)) if args.classifier == "logreg" else \
          (lambda Z, Yt, seed: train(Z, Yt, args, seed=seed, n_classes=n_classes))
    group_counts = {str(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True))}
    logger.info(f"{N} stories ({group_counts}), {Q} questions, answers {data['answer_scale']}, "
                f"embeddings {data['embedding_model']} ({X.shape[1]} dims)")
    logger.info(f"Setting: {setting} | {args.repeats}x{args.folds}-fold CV")

    # ---------- Cross-validation ----------
    oof = np.zeros((args.repeats, Q, N, n_classes), dtype=np.float32)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    cv_start = time.perf_counter()
    for fold_i, (tr, va) in enumerate(cv.split(X, groups)):
        pca, scaler = fit_features(X[tr], args.pca_k)
        model = fit(to_features(pca, scaler, X[tr]), Y[tr], SEED + fold_i)
        oof[fold_i // args.folds][:, va] = predict_proba(model, to_features(pca, scaler, X[va]))
    cv_s = time.perf_counter() - cv_start
    logger.info(f"CV done: {args.repeats * args.folds} fold runs in {cv_s:.1f}s")

    mean, summary = write_cv_outputs(output_dir, data, oof, title, setting,
                                     args.folds, args.repeats, cv_s, args.decode)

    # ---------- Final model on all train stories ----------
    final_start = time.perf_counter()
    pca, scaler = fit_features(X, args.pca_k)
    model = fit(to_features(pca, scaler, X), Y, SEED)
    final_s = time.perf_counter() - final_start

    final_config = {
        "model": title, "classifier": args.classifier, "C": args.C if args.classifier == "logreg" else None,
        "pca_k": args.pca_k, "hidden": list(args.hidden), "dropout": args.dropout,
        "weight_decay": args.weight_decay, "steps": args.steps, "lr": args.lr,
        "loss": args.loss, "decode": args.decode,
        "n_classes": n_classes, "answer_values": data["answer_values"], "answer_scale": data["answer_scale"],
        "question_ids": data["question_ids"], "components": data["components"],
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
        "state_dict": model.state_dict(),
    }, output_dir / "m1_final.pt")
    with open(output_dir / "m1_final_config.json", "w", encoding="utf-8") as f:
        json.dump({**final_config, "cv_seconds": round(cv_s, 2), "final_train_seconds": round(final_s, 3),
                   "cv_mean": {m: round(float(v), 4) for m, v in zip(metric_names(n_classes), mean)}}, f, indent=2)

    logger.info("Results:\n\n" + summary + "\n")
    logger.info(f"Final model trained in {final_s:.2f}s -> {output_dir / 'm1_final.pt'} | "
                f"full report with confusion matrices: {output_dir / 'cv_report.txt'} | log: {output_dir / 'train.log'}")


if __name__ == "__main__":
    main()

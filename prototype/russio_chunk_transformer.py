"""
Approach 3b for the russio distillation experiments: a small transformer over frozen OpenAI chunk
embeddings (the supervisor's "transformer on top of the embeddings"; ToBERT-style, Pappagari et al. 2019).

Each document is a sequence of chunk embeddings (russio_embed.py, up to 400 tokens per chunk, first
--max-chunks chunks). The model projects each 3072-d chunk embedding to --d, prepends a learned [CLS]
token, adds learned position embeddings, runs --layers transformer encoder layers (self-attention over
the chunks) and predicts yes/no from the [CLS] output. Only this small transformer is trained; the
OpenAI embeddings stay frozen.

Epochs are chosen by 5-fold CV on the train split (folds stratified by language x answer); the final
model is trained on all train documents. Test timing: each test document's chunks are embedded again in
ONE request per document (to measure a realistic per-document latency), plus the model's prediction.

Outputs (same format as the other approaches, scored with evaluate.py):
  <models-dir>/chunk_transformer_final.pt + chunk_transformer_final_config.json
  <predictions-dir>/<doc>.json

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_chunk_transformer.py
"""

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import StratifiedKFold

from train_m1 import metric_names, question_metrics

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
load_dotenv(REPO / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChunkTransformer(nn.Module):
    def __init__(self, emb_dim=3072, d=256, layers=2, heads=4, max_chunks=32, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d)
        self.pos = nn.Embedding(max_chunks + 1, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=heads, dim_feedforward=4 * d, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, 2)

    def forward(self, x, pad_mask):                      # x (B, n, emb_dim), pad_mask (B, n) True = padding
        B, n, _ = x.shape
        h = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], dim=1)
        h = h + self.pos(torch.arange(n + 1, device=x.device))
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1)
        return self.head(self.enc(h, src_key_padding_mask=mask)[:, 0])


def pack(seqs, max_chunks):
    """List of (n_i, D) arrays -> padded (N, n_max, D) tensor + padding mask."""
    n = min(max_chunks, max(len(s) for s in seqs))
    X = np.zeros((len(seqs), n, seqs[0].shape[1]), dtype=np.float32)
    M = np.ones((len(seqs), n), dtype=bool)
    for i, s in enumerate(seqs):
        k = min(len(s), n)
        X[i, :k], M[i, :k] = s[:k], False
    return torch.from_numpy(X), torch.from_numpy(M)


def train_model(seqs, y, args, epochs_list, seed=42, eval_seqs=None):
    torch.manual_seed(seed)
    model = ChunkTransformer(seqs[0].shape[1], args.d, args.layers, args.heads, args.max_chunks, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = np.argsort([len(s) for s in seqs])                     # batches of similar length: little padding
    batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    packed = [(pack([seqs[j] for j in b], args.max_chunks), torch.from_numpy(y[b])) for b in batches]
    rng = np.random.default_rng(seed)
    out = {}
    for epoch in range(1, max(epochs_list) + 1):
        model.train()
        for k in rng.permutation(len(packed)):
            (xb, mb), yb = packed[k]
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb, mb), yb).backward()
            opt.step()
        if eval_seqs is not None and epoch in epochs_list:
            out[epoch] = predict(model, eval_seqs, args.max_chunks)
    return model, out


@torch.no_grad()
def predict(model, seqs, max_chunks, batch_size=64):
    model.eval()
    probs = np.zeros((len(seqs), 2), dtype=np.float32)
    for s in range(0, len(seqs), batch_size):
        xb, mb = pack(seqs[s:s + batch_size], max_chunks)
        probs[s:s + batch_size] = torch.softmax(model(xb, mb), -1).numpy()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Approach 3b: small transformer over OpenAI chunk embeddings (russio)")
    parser.add_argument("--embeddings-dir", type=str, default=str(OUT / "embeddings" / "text-embedding-3-large"))
    parser.add_argument("--survey-dir", type=str, default=str(OUT / "survey_war" / "gpt-5.4"))
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 20, 30], help="epoch checkpoints compared in CV")
    parser.add_argument("--chunk-tokens", type=int, default=400, help="must match russio_embed.py (for the timing pass)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--question", type=str, default="1",
                        help="Question id in the survey JSONs (1 = war relevance; 2-4 are in survey_extra). "
                             "Outputs for questions other than 1 get a _q<id> suffix")
    parser.add_argument("--reuse-embed-timing", type=str, default=None,
                        help="Prediction folder of an earlier run (e.g. predictions/chunk_transformer/test): reuse its "
                             "measured per-article embedding latency and tokens instead of embedding the test articles again")
    args = parser.parse_args()
    torch.set_num_threads(8)
    q = args.question
    name = "chunk_transformer" + ("" if q == "1" else f"_q{q}")

    emb_dir = Path(args.embeddings_dir)
    models_dir, pred_dir = OUT / "models" / name, OUT / "predictions" / name / "test"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    C = np.load(emb_dir / "chunks.npy")
    by_doc = defaultdict(list)
    with open(emb_dir / "chunks_meta.csv", encoding="utf-8") as f:
        for k, m in enumerate(csv.DictReader(f)):
            by_doc[m["story_name"]].append(k)
    seqs = [C[by_doc[r["story_name"]]] for r in rows]
    y = []
    for r in rows:
        with open(Path(args.survey_dir) / r["split"] / f"{r['story_name']}.json", encoding="utf-8") as f:
            y.append(json.load(f)["ratings"][q])
    y = np.array(y)
    tr = np.array([r["split"] == "train" for r in rows])
    tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]
    logger.info(f"train {len(tr_idx)} | test {len(te_idx)} | chunks per document: median {np.median([len(s) for s in seqs]):.0f}, "
                f"> {args.max_chunks}: {sum(len(s) > args.max_chunks for s in seqs)} documents (first {args.max_chunks} kept)")

    # ---------- CV on train: pick the epoch count ----------
    strata = [f"{rows[i]['label']}_{y[i]}" for i in tr_idx]
    oof = {e: np.zeros((len(tr_idx), 2)) for e in args.epochs}
    start = time.perf_counter()
    for f_i, (a, v) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(tr_idx, strata)):
        _, probs = train_model([seqs[tr_idx[i]] for i in a], y[tr_idx[a]], args, args.epochs, seed=42 + f_i,
                               eval_seqs=[seqs[tr_idx[i]] for i in v])
        for e, p in probs.items():
            oof[e][v] = p
        logger.info(f"fold {f_i + 1}/5 done ({time.perf_counter() - start:.0f}s)")
    cv = {e: dict(zip(metric_names(2), question_metrics(y[tr_idx], p.argmax(1), 2, p))) for e, p in oof.items()}
    for e, m in cv.items():
        logger.info(f"epochs={e:<3} CV accuracy {m['exact_acc']:.3%} kappa {m['qwk']:.3f} AUC {m['auc']:.3f}")
    best_e = max(cv, key=lambda e: cv[e]["exact_acc"])
    logger.info(f"chosen epochs = {best_e}")

    # ---------- final model ----------
    start = time.perf_counter()
    model, _ = train_model([seqs[i] for i in tr_idx], y[tr_idx], args, [best_e])
    train_s = time.perf_counter() - start
    model_file = models_dir / "chunk_transformer_final.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "epochs": best_e}, model_file)
    n_params = sum(p.numel() for p in model.parameters())
    train_acc = (predict(model, [seqs[i] for i in tr_idx], args.max_chunks).argmax(1) == y[tr_idx]).mean()
    config = {"model": f"transformer ({args.layers} layers, d={args.d}) over OpenAI chunk embeddings",
              "max_chunks": args.max_chunks, "d": args.d, "layers": args.layers, "heads": args.heads, "epochs": best_e,
              "lr": args.lr, "weight_decay": args.weight_decay, "dropout": args.dropout,
              "n_classes": 2, "answer_values": [0, 1], "answer_scale": "yes/no (1/0)", "question_ids": [q],
              "train_documents": int(len(tr_idx)), "train_seconds": round(train_s, 1), "train_accuracy": round(float(train_acc), 4),
              "cv_all": {str(e): {k: round(float(v), 4) for k, v in m.items()} for e, m in cv.items()},
              "cv_mean": {k: round(float(v), 4) for k, v in cv[best_e].items()},
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    with open(models_dir / "chunk_transformer_final_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"final model: {n_params:,} parameters, train accuracy {train_acc:.1%}, trained in {train_s:.0f}s")

    # ---------- test: one embedding request per document (timed) + prediction ----------
    enc = tiktoken.get_encoding("cl100k_base")
    client = OpenAI(max_retries=6)

    def timed_embed(i):
        toks = enc.encode((DATA / "test" / f"{rows[i]['story_name']}.txt").read_text(encoding="utf-8"))
        pieces = [enc.decode(toks[s:s + args.chunk_tokens]) for s in range(0, max(len(toks), 1), args.chunk_tokens)][:args.max_chunks]
        start = time.perf_counter()
        resp = client.embeddings.create(model="text-embedding-3-large", input=pieces)
        return i, time.perf_counter() - start, resp.usage.prompt_tokens

    if args.reuse_embed_timing:                                # same test articles, same chunks: same embedding call
        timing = {}
        for i in te_idx:
            with open(Path(args.reuse_embed_timing) / f"{rows[i]['story_name']}.json", encoding="utf-8") as f:
                old = json.load(f)
            timing[i] = (old["timing"]["embed_latency_s"], old["tokens"])
        logger.info(f"embedding latency reused from {args.reuse_embed_timing}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            timing = {i: (s, n) for i, s, n in pool.map(timed_embed, te_idx)}
    info = {"name": config["model"], "file": str(model_file), "n_parameters": int(n_params), "size_bytes": model_file.stat().st_size}
    predict(model, [seqs[te_idx[0]]], args.max_chunks)  # warm-up
    for i in te_idx:
        start = time.perf_counter()
        p = predict(model, [seqs[i]], args.max_chunks)[0]
        pred_s = time.perf_counter() - start
        embed_s, n_tok = timing[i]
        doc = rows[i]["story_name"]
        result = {"story_name": doc, "story_file": str(DATA / "test" / f"{doc}.txt"), "model": info,
                  "embedding_model": "text-embedding-3-large", "timestamp": datetime.now().isoformat(timespec="seconds"),
                  "ratings": {q: int(p.argmax())}, "probabilities": {q: [round(float(x), 4) for x in p]}, "n_failed": 0,
                  "tokens": int(n_tok),
                  "timing": {"story_wall_s": round(embed_s + pred_s, 4), "embed_latency_s": round(embed_s, 4), "predict_latency_s": round(pred_s, 6)},
                  "cost_usd": round(n_tok * 0.13 / 1e6, 8)}
        with open(pred_dir / f"{doc}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"test predictions -> {pred_dir}")


if __name__ == "__main__":
    main()

"""
OpenAI embeddings for the russio documents, for approaches 2 (whole-text embedding + classifier)
and 3b (small transformer over chunk embeddings).

  whole   one embedding per document; texts longer than the model's input limit (8,191 tokens) are
          trimmed to their first 8,191 tokens (51 of 7,784 documents). Train documents are sent in
          batches; TEST documents are sent one per request so their per-document latency is realistic
          for the speed comparison with the LLM.
  chunks  each document split into consecutive --chunk-tokens token pieces (default 400), batched.

Outputs in <output-dir> (row order = data/russio-prototype/split.csv):
  whole.npy (N, D) float32 + whole_meta.csv (story_name, split, tokens, trimmed, latency_s for test docs)
  chunks.npy (M, D) float32 + chunks_meta.csv (story_name, split, chunk, start_token, tokens)
  run.json (model, totals, cost)

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_embed.py
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from embed_story import PRICING

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
load_dotenv(REPO / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MAX_INPUT = 8191


def batches(items, max_items=128, max_tokens=200_000):
    """Group (index, text, n_tokens) items into requests under the API's per-request limits."""
    batch, tokens = [], 0
    for it in items:
        if batch and (len(batch) >= max_items or tokens + it[2] > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(it)
        tokens += it[2]
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description="OpenAI whole-text and chunk embeddings for the russio documents")
    parser.add_argument("--model", type=str, default="text-embedding-3-large")
    parser.add_argument("--chunk-tokens", type=int, default=400)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: output_test/prototype/russio/embeddings/<model>")
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else REPO / "output_test" / "prototype" / "russio" / "embeddings" / args.model
    if out.resolve().is_relative_to((REPO / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results).")
    out.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")  # tokenizer of the text-embedding-3 models
    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    toks = [enc.encode((DATA / r["split"] / f"{r['story_name']}.txt").read_text(encoding="utf-8")) for r in rows]
    client = OpenAI(max_retries=6)
    price = PRICING.get(args.model, 0.0)

    def embed(texts):
        start = time.perf_counter()
        resp = client.embeddings.create(model=args.model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)], resp.usage.prompt_tokens, time.perf_counter() - start

    # ---------- whole-text embeddings ----------
    whole_inputs = [(i, enc.decode(t[:MAX_INPUT]), min(len(t), MAX_INPUT)) for i, t in enumerate(toks)]
    whole = [None] * len(rows)
    latency = [None] * len(rows)
    total_tokens = 0
    train_items = [it for it in whole_inputs if rows[it[0]]["split"] == "train"]
    test_items = [it for it in whole_inputs if rows[it[0]]["split"] == "test"]

    def run_batch(batch):
        vecs, n, _ = embed([b[1] for b in batch])
        return batch, vecs, n

    def run_single(item):
        vecs, n, s = embed([item[1]])
        return item, vecs[0], n, s

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch, vecs, n in pool.map(run_batch, list(batches(train_items))):
            for (i, _, _), v in zip(batch, vecs):
                whole[i] = v
            total_tokens += n
        logger.info(f"whole-text, train: {len(train_items)} documents in {time.perf_counter() - start:.0f}s")
        for item, v, n, s in pool.map(run_single, test_items):   # one request per test document (timed)
            whole[item[0]], latency[item[0]] = v, s
            total_tokens += n
    logger.info(f"whole-text, train + test: done in {time.perf_counter() - start:.0f}s")
    np.save(out / "whole.npy", np.asarray(whole, dtype=np.float32))
    with open(out / "whole_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_name", "split", "tokens", "trimmed", "latency_s"])
        for r, t, lat in zip(rows, toks, latency):
            w.writerow([r["story_name"], r["split"], len(t), int(len(t) > MAX_INPUT), "" if lat is None else round(lat, 4)])
    whole_tokens = total_tokens

    # ---------- chunk embeddings ----------
    chunk_meta, chunk_items = [], []
    for i, t in enumerate(toks):
        for c, s in enumerate(range(0, max(len(t), 1), args.chunk_tokens)):
            piece = t[s:s + args.chunk_tokens]
            chunk_items.append((len(chunk_meta), enc.decode(piece), len(piece)))
            chunk_meta.append([rows[i]["story_name"], rows[i]["split"], c, s, len(piece)])
    chunks = [None] * len(chunk_meta)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch, vecs, n in pool.map(run_batch, list(batches(chunk_items))):
            for (j, _, _), v in zip(batch, vecs):
                chunks[j] = v
            total_tokens += n
    logger.info(f"chunks: {len(chunk_meta)} pieces of up to {args.chunk_tokens} tokens in {time.perf_counter() - start:.0f}s")
    np.save(out / "chunks.npy", np.asarray(chunks, dtype=np.float32))
    with open(out / "chunks_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_name", "split", "chunk", "start_token", "tokens"])
        w.writerows(chunk_meta)

    run = {"model": args.model, "documents": len(rows), "trimmed_documents": sum(len(t) > MAX_INPUT for t in toks),
           "chunk_tokens": args.chunk_tokens, "chunks": len(chunk_meta), "whole_tokens": whole_tokens,
           "chunk_tokens_total": total_tokens - whole_tokens, "cost_usd": round(total_tokens * price / 1e6, 4),
           "test_latency_mean_s": round(float(np.mean([x for x in latency if x is not None])), 4),
           "created_at": datetime.now().isoformat(timespec="seconds")}
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    logger.info(f"Done: {run}")


if __name__ == "__main__":
    main()

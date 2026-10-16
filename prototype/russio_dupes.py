"""
Near-duplicate check for the russio train/test split (e.g. the same News Front story published in
several languages). For every test document, finds the most similar train document by cosine
similarity of the whole-text embeddings (text-embedding-3-large is multilingual, so translations
land close together), and reports:
  - how many test documents have a train document above each similarity threshold, and whether
    that neighbour is in the same or another language
  - examples near the chosen threshold, to check what the similarity means
  - each approach's test accuracy on test documents WITH vs WITHOUT a near-duplicate in train

Writes output_test/prototype/russio/evaluation/near_duplicates.csv.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_dupes.py --threshold 0.9 --predictions TF-IDF=... Emb-LR=...
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"


def main():
    parser = argparse.ArgumentParser(description="Near-duplicate check between russio train and test")
    parser.add_argument("--embeddings-dir", type=str, default=str(OUT / "embeddings" / "text-embedding-3-large"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--predictions", type=str, nargs="*", default=[], help="NAME=FOLDER prediction folders to split")
    args = parser.parse_args()

    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.load(Path(args.embeddings_dir) / "whole.npy")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    tr = np.array([r["split"] == "train" for r in rows])
    tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]
    S = X[te_idx] @ X[tr_idx].T                                  # (n_test, n_train) cosine similarities
    best = S.argmax(1)
    best_sim = S[np.arange(len(te_idx)), best]
    title = lambda i: (DATA / rows[i]["split"] / f"{rows[i]['story_name']}.txt").read_text(encoding="utf-8").split("\n", 1)[0][:85]

    out_rows = []
    for k, i in enumerate(te_idx):
        j = tr_idx[best[k]]
        out_rows.append({"test_doc": rows[i]["story_name"], "test_lang": rows[i]["label"], "test_answer": rows[i]["answer"],
                         "nearest_train_doc": rows[j]["story_name"], "train_lang": rows[j]["label"],
                         "train_answer": rows[j]["answer"], "similarity": round(float(best_sim[k]), 4)})
    path = OUT / "evaluation" / "near_duplicates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(te_idx)} test documents: similarity to the closest train document")
    print(f"  median {np.median(best_sim):.3f} | 90th pct {np.percentile(best_sim, 90):.3f} | max {best_sim.max():.3f}")
    for t in (0.95, 0.9, 0.85, 0.8):
        m = best_sim >= t
        same = sum(1 for k in np.where(m)[0] if out_rows[k]["test_lang"] == out_rows[k]["train_lang"])
        agree = sum(1 for k in np.where(m)[0] if out_rows[k]["test_answer"] == out_rows[k]["train_answer"])
        print(f"  >= {t}: {m.sum():4d} test docs ({m.mean():.1%}) | same language {same}, other language {m.sum() - same} | "
              f"nearest train doc has the same LLM answer: {agree}")
    print(f"\nexamples around the threshold {args.threshold} (test title || nearest train title):")
    order = np.argsort(np.abs(best_sim - args.threshold))[:6]
    for k in sorted(order, key=lambda k: -best_sim[k]):
        i, j = te_idx[k], tr_idx[best[k]]
        print(f"  {best_sim[k]:.3f} [{rows[i]['label']}->{rows[j]['label']}] {title(i)}  ||  {title(j)}")
    print("highest-similarity pairs:")
    for k in np.argsort(-best_sim)[:4]:
        i, j = te_idx[k], tr_idx[best[k]]
        print(f"  {best_sim[k]:.3f} [{rows[i]['label']}->{rows[j]['label']}] {title(i)}  ||  {title(j)}")

    dup = {out_rows[k]["test_doc"] for k in range(len(te_idx)) if best_sim[k] >= args.threshold}
    if args.predictions:
        print(f"\ntest accuracy with vs without a near-duplicate (similarity >= {args.threshold}) in train:")
        for spec in args.predictions:
            name, folder = spec.split("=", 1)
            hits = {"dup": [], "clean": []}
            for r in out_rows:
                with open(Path(folder) / f"{r['test_doc']}.json", encoding="utf-8") as f:
                    pred = json.load(f)["ratings"]["1"]
                hits["dup" if r["test_doc"] in dup else "clean"].append(pred == int(r["test_answer"]))
            a = lambda v: f"{np.mean(v):.1%} (n={len(v)})" if v else "-"
            print(f"  {name:10} all {np.mean(hits['dup'] + hits['clean']):.1%} | with near-duplicate {a(hits['dup'])} | "
                  f"without {a(hits['clean'])}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

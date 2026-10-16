"""
How tightly does each encoder cluster the russio articles? (the supervisor's reason for a transformer)

The supervisor's argument for a transformer was that it "tends to do a good job of clustering things
tightly in the feature space", which the later prototype / out-of-distribution step needs. Test accuracy
does not show that, so this script takes the feature vector each model produces for an article (the
layer right before its yes/no output; for the frozen OpenAI embedding, the embedding itself) and
measures on the TEST articles:

  silhouette (answer)    how tightly yes-articles and no-articles group, cosine distance;
                         -1..1, higher = tighter and better separated (Rousseeuw 1987)
  silhouette (language)  the same with the article's language as the group; high = the space groups
                         articles by language
  prototype accuracy     each test article gets the answer of the nearer of two prototypes (the mean
                         train feature of the yes and of the no articles), cosine similarity;
                         95% bootstrap range
  error-flag AUC         does the prototype margin (how much nearer one prototype is than the other)
                         tell right from wrong prototype answers? 0.5 = no better than chance, 1 = perfectly

All features are centered on the mean train feature before cosine similarity (the common offset of an
embedding space otherwise makes every pair look similar). Prototypes use train articles only.

Feature spaces (nothing is retrained):
  openai             text-embedding-3-large whole-text embedding, frozen (3,072)
  emb_network        hidden layer of the Emb + network model, after ReLU (256)
  xlmr_pretrained    xlm-roberta-base BEFORE fine-tuning, mean of its last-layer token vectors (768)
  xlmr_finetuned     fine-tuned XLM-R: the classification head's hidden layer on the summary token (768)
  chunk_transformer  the chunk transformer's [CLS] notepad after its 2 layers (256)

As a check, the trained models' own output layers are applied to the extracted features and must give
the same test answers as their saved predictions.

Outputs in output_test/prototype/russio/clustering/:
  features/<space>.npy  (row order = data/russio-prototype/split.csv; reused unless --recompute)
  results.csv, results.json, report.txt, tsne_test.png, run.log

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_clusters.py
"""

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, silhouette_score

from train_m1 import FP_IGNORE

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
CLU = OUT / "clustering"
EMB = OUT / "embeddings" / "text-embedding-3-large"
logger = logging.getLogger("clusters")

SPACES = {  # name -> (label, model it belongs to, column in per_story_predictions.csv)
    "openai": ("OpenAI embedding (frozen)", "input of Emb + LR / Emb + network", None),
    "emb_network": ("Emb + network hidden layer", "Emb + network", "Emb-MLP"),
    "xlmr_pretrained": ("XLM-R before fine-tuning", "no classifier", None),
    "xlmr_finetuned": ("XLM-R fine-tuned", "XLM-R", "XLM-R"),
    "chunk_transformer": ("Chunk transformer notepad", "Chunk transformer", "Chunk-TF"),
}


# ---------------- feature extraction (each returns features + the model's own output layer, or None) ----------------

def feats_openai(rows):
    return np.load(EMB / "whole.npy"), None


def feats_emb_network(rows):
    ck = torch.load(OUT / "models" / "emb_mlp" / "emb_mlp_final.pt", weights_only=False)
    X = np.load(EMB / "whole.npy")
    with np.errstate(**FP_IGNORE):
        Z = ((X - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)
    sd = ck["state_dict"]                                  # Sequential: 0 Linear, 1 ReLU, 2 Dropout, 3 Linear
    first, last = nn.Linear(X.shape[1], ck["hidden"]), nn.Linear(ck["hidden"], 2)
    first.load_state_dict({"weight": sd["0.weight"], "bias": sd["0.bias"]})
    last.load_state_dict({"weight": sd["3.weight"], "bias": sd["3.bias"]})
    with torch.no_grad():
        F = torch.relu(first(torch.from_numpy(Z))).numpy()
    return F, last


@torch.no_grad()
def feats_xlmr(rows, finetuned, max_length=512, batch_size=16):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    path = OUT / "models" / "xlmr_base" if finetuned else "xlm-roberta-base"
    tok = AutoTokenizer.from_pretrained(path)
    if finetuned:
        clf = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        encoder, head = clf.roberta, clf.classifier
    else:
        encoder, head = AutoModel.from_pretrained(path).to(device).eval(), None
    texts = [(DATA / r["split"] / f"{r['story_name']}.txt").read_text(encoding="utf-8") for r in rows]
    ids = tok(texts, truncation=True, max_length=max_length)["input_ids"]
    order = np.argsort([len(i) for i in ids])                 # length-sorted batches: less padding
    F = np.zeros((len(texts), encoder.config.hidden_size), dtype=np.float32)
    start = time.perf_counter()
    for s in range(0, len(order), batch_size):
        part = order[s:s + batch_size]
        b = {k: v.to(device) for k, v in tok.pad({"input_ids": [ids[i] for i in part]}, return_tensors="pt").items()}
        h = encoder(**b).last_hidden_state
        if finetuned:                                         # head = dense -> tanh -> out_proj on the <s> token
            f = torch.tanh(head.dense(h[:, 0]))
        else:                                                 # untrained head: mean of the token vectors
            m = b["attention_mask"].unsqueeze(-1).float()
            f = (h * m).sum(1) / m.sum(1)
        F[part] = f.float().cpu().numpy()
        if (s // batch_size) % 100 == 0:
            logger.info(f"  {'fine-tuned' if finetuned else 'pretrained'} XLM-R: {s + len(part)}/{len(texts)} | {time.perf_counter() - start:.0f}s")
    if device.type == "mps":
        torch.mps.empty_cache()
    return F, (head.out_proj.cpu() if finetuned else None)


@torch.no_grad()
def feats_chunk_transformer(rows):
    from russio_chunk_transformer import ChunkTransformer, pack
    ck = torch.load(OUT / "models" / "chunk_transformer" / "chunk_transformer_final.pt", weights_only=False)
    a = ck["args"]
    C = np.load(EMB / "chunks.npy")
    by_doc = defaultdict(list)
    with open(EMB / "chunks_meta.csv", encoding="utf-8") as f:
        for k, m in enumerate(csv.DictReader(f)):
            by_doc[m["story_name"]].append(k)
    seqs = [C[by_doc[r["story_name"]]] for r in rows]
    model = ChunkTransformer(C.shape[1], a["d"], a["layers"], a["heads"], a["max_chunks"], a["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    F = np.zeros((len(seqs), a["d"]), dtype=np.float32)
    for s in range(0, len(seqs), 64):                         # the model's forward(), stopped before its head
        x, pad = pack(seqs[s:s + 64], a["max_chunks"])
        B, n, _ = x.shape
        h = torch.cat([model.cls.expand(B, -1, -1), model.proj(x)], dim=1) + model.pos(torch.arange(n + 1))
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool), pad], dim=1)
        F[s:s + 64] = model.enc(h, src_key_padding_mask=mask)[:, 0].numpy()
    return F, model.head


EXTRACT = {"openai": feats_openai, "emb_network": feats_emb_network,
           "xlmr_pretrained": lambda rows: feats_xlmr(rows, finetuned=False),
           "xlmr_finetuned": lambda rows: feats_xlmr(rows, finetuned=True),
           "chunk_transformer": feats_chunk_transformer}


# ---------------- measures ----------------

def unit(F):
    return F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-12, None)


def measure(F, y, lang, tr, rng, n_boot=2000):
    te = ~tr
    with np.errstate(**FP_IGNORE):
        N = unit(F - F[tr].mean(0))                           # center on the train mean, then unit length
        sil_answer = float(silhouette_score(N[te], y[te], metric="cosine"))
        sil_lang = float(silhouette_score(N[te], lang[te], metric="cosine"))
        protos = unit(np.stack([N[tr & (y == k)].mean(0) for k in (0, 1)]))
        S = N[te] @ protos.T                                  # cosine similarity to the no / yes prototype
    hit = S.argmax(1) == y[te]
    margin = np.abs(S[:, 1] - S[:, 0])
    boot = rng.integers(0, len(hit), (n_boot, len(hit)))
    lo, hi = np.percentile(hit[boot].mean(1), [2.5, 97.5])
    return {"silhouette_answer": round(sil_answer, 4), "silhouette_language": round(sil_lang, 4),
            "prototype_accuracy": round(float(hit.mean()), 4), "prototype_acc_low": round(float(lo), 4),
            "prototype_acc_high": round(float(hi), 4), "error_flag_auc": round(float(roc_auc_score(~hit, -margin)), 4),
            "prototype_errors": int((~hit).sum())}


def tsne_plot(spaces, y, tr, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.info("matplotlib not available: no t-SNE picture")
        return
    te = ~tr
    fig, axes = plt.subplots(1, len(spaces), figsize=(4.2 * len(spaces), 4.4))
    for ax, (name, F) in zip(np.atleast_1d(axes), spaces.items()):
        with np.errstate(**FP_IGNORE):
            N = unit(F - F[tr].mean(0))[te]
        P = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(N)
        for k, (color, label) in enumerate([("#c0504d", "LLM: no"), ("#3a6ea5", "LLM: yes")]):
            ax.scatter(P[y[te] == k, 0], P[y[te] == k, 1], s=3, alpha=.55, c=color, label=label, linewidths=0)
        ax.set_title(SPACES[name][0], fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8, markerscale=3, frameon=False)
    fig.suptitle("Test articles in each feature space (t-SNE map; for illustration, distances are not exact)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cluster tightness of each encoder's feature space (russio)")
    parser.add_argument("--survey-dir", type=str, default=str(OUT / "survey_war" / "gpt-5.4"))
    parser.add_argument("--spaces", nargs="+", default=list(SPACES), choices=list(SPACES))
    parser.add_argument("--recompute", action="store_true", help="Extract features again even if saved")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    (CLU / "features").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(CLU / "run.log", mode="w")])

    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = []
    for r in rows:
        with open(Path(args.survey_dir) / r["split"] / f"{r['story_name']}.json", encoding="utf-8") as f:
            y.append(json.load(f)["ratings"]["1"])
    y = np.array(y)
    lang = np.array([r["label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    with open(OUT / "evaluation" / "all_approaches" / "per_story_predictions.csv", encoding="utf-8") as f:
        saved = {r["story_name"]: r for r in csv.DictReader(f)}
    te_names = [r["story_name"] for r, t in zip(rows, tr) if not t]
    logger.info(f"{tr.sum()} train / {(~tr).sum()} test articles | spaces: {', '.join(args.spaces)}")

    feats, results = {}, []
    rng = np.random.default_rng(0)
    for name in args.spaces:
        path = CLU / "features" / f"{name}.npy"
        head = None
        start = time.perf_counter()
        if path.exists() and not args.recompute and SPACES[name][2] is None:
            F = np.load(path)
            logger.info(f"{name}: loaded saved features {F.shape}")
        else:
            F, head = EXTRACT[name](rows)
            np.save(path, F)
            logger.info(f"{name}: extracted {F.shape} in {time.perf_counter() - start:.0f}s")
        check = None
        if head is not None:                                  # the model's own output layer on these features
            with torch.no_grad():
                pred = head(torch.from_numpy(F[~tr])).argmax(1).numpy()
            col = SPACES[name][2]
            check = float(np.mean([p == int(saved[n][col]) for p, n in zip(pred, te_names)]))
            logger.info(f"  check: output layer on these features matches the saved test answers for {check:.2%} of articles")
        m = measure(F, y, lang, tr, rng)
        results.append({"space": name, "label": SPACES[name][0], "model": SPACES[name][1], "dims": int(F.shape[1]),
                        **m, "matches_saved_predictions": None if check is None else round(check, 4)})
        logger.info(f"  silhouette answer {m['silhouette_answer']:.3f} | language {m['silhouette_language']:.3f} | "
                    f"prototype accuracy {m['prototype_accuracy']:.2%} [{m['prototype_acc_low']:.2%}-{m['prototype_acc_high']:.2%}] | "
                    f"error-flag AUC {m['error_flag_auc']:.3f}")
        feats[name] = F

    with open(CLU / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(CLU / "results.json", "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"), "test_articles": int((~tr).sum()),
                   "notes": "features centered on the train mean; cosine; prototypes = class means of train features",
                   "results": results}, f, indent=2)
    lines = [f"Cluster tightness on {(~tr).sum()} test articles (labels = LLM answers; features centered on the train mean, cosine)",
             "",
             f"  {'feature space':28} {'dims':>5} {'silhouette':>10} {'silhouette':>10} {'prototype':>9} {'95% range':>15} {'error-flag':>10}",
             f"  {'':28} {'':>5} {'(answer)':>10} {'(language)':>10} {'accuracy':>9} {'':>15} {'AUC':>10}"]
    for r in results:
        lines.append(f"  {r['label']:28} {r['dims']:>5} {r['silhouette_answer']:>10.3f} {r['silhouette_language']:>10.3f} "
                     f"{r['prototype_accuracy']:>9.1%} {r['prototype_acc_low']:>7.1%}-{r['prototype_acc_high']:<7.1%} {r['error_flag_auc']:>10.3f}")
    (CLU / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not args.no_plot:
        tsne_plot(feats, y, tr, CLU / "tsne_test.png")
        logger.info(f"t-SNE picture -> {CLU / 'tsne_test.png'}")
    logger.info(f"results -> {CLU}")


if __name__ == "__main__":
    main()

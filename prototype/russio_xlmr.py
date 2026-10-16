"""
Approach 3a for the russio distillation experiments: fine-tune XLM-RoBERTa on the LLM's yes/no answers.

The standard recipe for LLM-to-classifier distillation: a pretrained multilingual transformer encoder
(xlm-roberta-base, 100 languages) with a classification head, trained end to end (all encoder weights
update) on the train split. Inputs are the first --max-length tokens of each document (title + start).

Because a 5-fold CV of fine-tuning is expensive, settings follow the usual fine-tuning defaults and
the best epoch is chosen on a held-out 10% of the TRAIN split (stratified by language x answer); the
test split is only used at the end. Test documents are predicted one at a time (tokenize + forward
pass on the Apple GPU), so the per-document time is end to end, like the other approaches.

Outputs (same format as the other approaches, scored with evaluate.py):
  <models-dir>/                      fine-tuned model + tokenizer (save_pretrained), xlmr_final_config.json, train.log
  <predictions-dir>/<doc>.json       per test document

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_xlmr.py
    python prototype/russio_xlmr.py --question 2 --survey-dir output_test/prototype/russio/survey_extra/gpt-5.4   # -> xlmr_base_q2/
"""

import argparse
import copy
import csv
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train_m1 import metric_names, question_metrics

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
logger = logging.getLogger("xlmr")


def load_part(part: str, survey_dir: Path, question: str = "1"):
    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == part]
    texts = [(DATA / part / f"{r['story_name']}.txt").read_text(encoding="utf-8") for r in rows]
    y = []
    for r in rows:
        with open(survey_dir / part / f"{r['story_name']}.json", encoding="utf-8") as f:
            y.append(json.load(f)["ratings"][question])
    return rows, texts, np.array(y)


def batches(enc, idx, batch_size, tok, device, shuffle, rng):
    idx = list(idx)
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        part = idx[s:s + batch_size]
        b = tok.pad({"input_ids": [enc[i] for i in part]}, return_tensors="pt")
        yield part, {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def predict(model, enc, idx, tok, device, batch_size=16):
    model.eval()
    out = np.zeros((len(idx), 2), dtype=np.float32)
    pos = {i: k for k, i in enumerate(idx)}
    order = sorted(idx, key=lambda i: len(enc[i]))          # length-sorted batches: less padding, faster
    for part, b in batches(enc, order, batch_size, tok, device, False, None):
        p = torch.softmax(model(**b).logits.float(), -1).cpu().numpy()
        for i, row in zip(part, p):
            out[pos[i]] = row
    return out


def main():
    parser = argparse.ArgumentParser(description="Approach 3a: fine-tune XLM-RoBERTa (russio)")
    parser.add_argument("--model-name", type=str, default="xlm-roberta-base")
    parser.add_argument("--survey-dir", type=str, default=str(OUT / "survey_war" / "gpt-5.4"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="Run each batch in pieces of this many documents and add up their gradients before the "
                             "update (same update as the full batch, less memory). Default: the whole batch at once")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1, help="Share of steps with learning-rate warm-up")
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", type=str, default="1",
                        help="Question id in the survey JSONs (1 = war relevance; 2-4 are in survey_extra)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Folder name for model and predictions (default: xlmr_base, or xlmr_base_q<id> for other questions)")
    args = parser.parse_args()
    q = args.question
    tag = args.tag or ("xlmr_base" if q == "1" else f"xlmr_base_q{q}")

    models_dir = OUT / "models" / tag
    pred_dir = OUT / "predictions" / tag / "test"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(models_dir / "train.log", mode="w")])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rows_tr, X_tr, y_tr = load_part("train", Path(args.survey_dir), q)
    rows_te, X_te, y_te = load_part("test", Path(args.survey_dir), q)
    strata = [f"{r['label']}_{a}" for r, a in zip(rows_tr, y_tr)]
    fit_idx, val_idx = train_test_split(np.arange(len(X_tr)), test_size=args.val_share, stratify=strata, random_state=args.seed)
    logger.info(f"device {device} | train {len(fit_idx)} + validation {len(val_idx)} (from the train split) | test {len(X_te)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    enc_tr = tok(X_tr, truncation=True, max_length=args.max_length)["input_ids"]
    enc_te = tok(X_te, truncation=True, max_length=args.max_length)["input_ids"]
    truncated = np.mean([len(e) >= args.max_length for e in enc_tr + enc_te])
    logger.info(f"tokenized: {truncated:.0%} of documents are longer than {args.max_length} tokens and were cut to their start")

    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * math.ceil(len(fit_idx) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    y_t = torch.from_numpy(y_tr)
    rng = random.Random(args.seed)

    best = {"acc": -1}
    history = []
    t0 = time.perf_counter()
    step = 0
    micro = args.micro_batch or args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for part, b in batches(enc_tr, fit_idx, args.batch_size, tok, device, True, rng):
            total = 0.0
            for s in range(0, len(part), micro):             # gradient accumulation: pieces weighted by their size
                sub = {k: v[s:s + micro] for k, v in b.items()}
                loss = torch.nn.functional.cross_entropy(model(**sub).logits, y_t[part[s:s + micro]].to(device))
                loss = loss * (sub["input_ids"].shape[0] / len(part))
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running.append(total); step += 1
            if step % 50 == 0:
                logger.info(f"epoch {epoch} step {step}/{steps} | loss {np.mean(running[-50:]):.4f} | {time.perf_counter() - t0:.0f}s")
                if device.type == "mps":
                    torch.mps.empty_cache()          # batches differ in length: release cached blocks so memory stays bounded
        if device.type == "mps":
            torch.mps.empty_cache()                  # release cached training memory before scoring
        p_val = predict(model, enc_tr, val_idx, tok, device)
        m = dict(zip(metric_names(2), question_metrics(y_tr[val_idx], p_val.argmax(1), 2, p_val)))
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(running)), 4), **{k: round(float(v), 4) for k, v in m.items()}})
        logger.info(f"epoch {epoch} done | validation accuracy {m['exact_acc']:.2%} kappa {m['qwk']:.3f} AUC {m['auc']:.3f} | {time.perf_counter() - t0:.0f}s")
        if m["exact_acc"] > best["acc"]:          # keep the best epoch on disk, not in memory
            best = {"acc": m["exact_acc"], "epoch": epoch, "metrics": m}
            model.save_pretrained(models_dir / "best_epoch")
            logger.info(f"saved epoch {epoch} as the best so far -> {models_dir / 'best_epoch'}")
        if device.type == "mps":
            torch.mps.empty_cache()
    train_s = time.perf_counter() - t0
    del opt, sched
    model = AutoModelForSequenceClassification.from_pretrained(models_dir / "best_epoch").to(device)
    logger.info(f"best epoch {best['epoch']} (validation accuracy {best['acc']:.2%}); training took {train_s / 60:.1f} min")

    model.save_pretrained(models_dir)
    tok.save_pretrained(models_dir)
    size = sum(f.stat().st_size for f in models_dir.glob("*.safetensors")) or sum(f.stat().st_size for f in models_dir.glob("*.bin"))
    n_params = sum(p.numel() for p in model.parameters())
    config = {"model": f"fine-tuned {args.model_name}", "max_length": args.max_length, "epochs": args.epochs, "best_epoch": best["epoch"],
              "lr": args.lr, "batch_size": args.batch_size, "micro_batch": micro, "weight_decay": args.weight_decay, "warmup": args.warmup, "seed": args.seed,
              "n_classes": 2, "answer_values": [0, 1], "answer_scale": "yes/no (1/0)", "question_ids": [q],
              "train_documents": int(len(fit_idx)), "validation_documents": int(len(val_idx)), "share_truncated": round(float(truncated), 4),
              "train_minutes": round(train_s / 60, 1), "history": history,
              "cv_mean": {k: round(float(v), 4) for k, v in best["metrics"].items()},   # held-out validation (10% of train)
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    with open(models_dir / "xlmr_final_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---------- test: one document at a time, tokenize + forward pass (end-to-end time) ----------
    model.eval()
    info = {"name": config["model"], "file": str(models_dir / "xlmr_final.pt"), "n_parameters": int(n_params), "size_bytes": int(size)}
    with torch.no_grad():
        model(**{k: v.to(device) for k, v in tok(["warm-up"], return_tensors="pt").items()})
        for r in rows_te:
            path = DATA / "test" / f"{r['story_name']}.txt"
            start = time.perf_counter()
            b = tok([path.read_text(encoding="utf-8")], truncation=True, max_length=args.max_length, return_tensors="pt")
            p = torch.softmax(model(**{k: v.to(device) for k, v in b.items()}).logits.float(), -1)[0].cpu().numpy()
            wall_s = time.perf_counter() - start
            result = {"story_name": r["story_name"], "story_file": str(path), "model": info,
                      "timestamp": datetime.now().isoformat(timespec="seconds"),
                      "ratings": {q: int(p.argmax())}, "probabilities": {q: [round(float(x), 4) for x in p]}, "n_failed": 0,
                      "tokens": int(b["input_ids"].shape[1]),
                      "timing": {"story_wall_s": round(wall_s, 5), "embed_latency_s": 0.0, "predict_latency_s": round(wall_s, 6)},
                      "cost_usd": 0.0}
            with open(pred_dir / f"{r['story_name']}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
    # evaluate.py looks up <model file stem>_config.json next to info["file"]
    (models_dir / "xlmr_final.pt").write_text("weights are in model.safetensors (save_pretrained)\n")
    logger.info(f"test predictions -> {pred_dir}")


if __name__ == "__main__":
    main()

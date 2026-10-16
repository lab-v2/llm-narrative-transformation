"""
LLM-free survey with a trained M1 model: story text in, 20 ratings out.

For each story, end to end:
    read story -> OpenAI embedding -> PCA + scaling -> 20 per-question networks -> ratings

Output: one JSON per story at <output-dir>/<story_name>.json in the same shape
as survey_story.py's output ("ratings": {question_id: 1-5}, "n_failed",
"tokens", "timing", "cost_usd"), plus the per-question probabilities.

Timing per story (comparable to the LLM survey's timing):
    story_wall_s       total, from reading the story file to having the 20 ratings
    embed_latency_s    the OpenAI embedding call alone
    predict_latency_s  PCA + scaling + 20 networks alone
One-time setup (loading the model, creating the API client) is not counted,
as in survey_story.py; it is logged separately.

Stories whose JSON already has all ratings are skipped unless --overwrite is given.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/predict_m1.py \
        --stories-dir data/test-prototype \
        --model-dir output_test/prototype/models/m1_s400 \
        --output-dir output_test/prototype/predictions/m1_s400/test
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from openai import OpenAI

from embed_story import embed_story
from train_m1 import PerQuestionMLP, decode

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def load_model(model_dir: Path, args=None):
    model_file = model_dir / "m1_final.pt"
    ckpt = torch.load(model_file, weights_only=True)
    cfg = ckpt["config"]
    model = PerQuestionMLP(len(cfg["question_ids"]), cfg["pca_k"], tuple(cfg["hidden"]), cfg["n_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tensors = [ckpt["pca_components"], ckpt["pca_mean"], ckpt["scaler_mean"], ckpt["scaler_scale"], *ckpt["state_dict"].values()]
    info = {
        "name": cfg["model"],
        "file": str(model_file),
        "n_parameters": sum(t.numel() for t in tensors),
        "size_bytes": sum(t.numel() * t.element_size() for t in tensors),
    }
    return model, ckpt, cfg, info


def predict_one(model, ckpt, vector: torch.Tensor) -> torch.Tensor:
    """embedding (D,) -> probabilities (Q, C). Same transform as sklearn PCA.transform + StandardScaler."""
    with torch.no_grad():
        z = (vector - ckpt["pca_mean"]) @ ckpt["pca_components"].T
        z = (z - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
        return torch.softmax(model(z.unsqueeze(0)), dim=-1)[:, 0, :]


def survey_story_model(client: OpenAI, story_path: Path, model, ckpt, cfg, info, predict_fn) -> dict:
    """Story file -> embedding -> predict_fn -> survey-style result (timed end to end)."""
    question_ids = cfg["question_ids"]
    ratings = {q: None for q in question_ids}
    probabilities, tokens, cost_usd, error = None, 0, None, None
    embed_s = predict_s = None

    wall_start = time.perf_counter()
    try:
        emb = embed_story(client, story_path, cfg["embedding_model"])  # reads the story + API call
        embed_s = emb["timing"]["embed_latency_s"]
        tokens, cost_usd = emb["tokens"], emb["cost_usd"]

        predict_start = time.perf_counter()
        probs = predict_fn(model, ckpt, torch.tensor(emb["embedding"], dtype=torch.float32))
        classes = decode(probs.numpy(), cfg.get("decode", "argmax"))  # model's own rating rule
        values = cfg.get("answer_values", list(range(1, cfg["n_classes"] + 1)))  # [1..5], or [0, 1] for yes/no
        ratings = {q: values[int(classes[j])] for j, q in enumerate(question_ids)}
        predict_s = time.perf_counter() - predict_start
        probabilities = {q: [round(float(p), 4) for p in probs[j]] for j, q in enumerate(question_ids)}
    except Exception as e:
        error = str(e)
        logger.error(f"{story_path.stem}: failed: {e}")
    wall_s = time.perf_counter() - wall_start

    result = {
        "story_name": story_path.stem,
        "story_file": str(story_path),
        "model": info,
        "embedding_model": cfg["embedding_model"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ratings": ratings,
        "probabilities": probabilities,
        "n_failed": sum(r is None for r in ratings.values()),
        "tokens": tokens,
        "timing": {
            "story_wall_s": round(wall_s, 4),
            "embed_latency_s": embed_s,
            "predict_latency_s": round(predict_s, 6) if predict_s is not None else None,
        },
        "cost_usd": cost_usd,
    }
    if error:
        result["error"] = error
    return result


def is_complete(out_file: Path) -> bool:
    """True if a saved prediction exists and every question got a rating."""
    if not out_file.exists():
        return False
    with open(out_file, encoding="utf-8") as f:
        return json.load(f).get("n_failed", 1) == 0


def run(load_model_fn, predict_fn, model_label: str, add_args=None):
    """Shared driver for predict_m1/m2/m3.py. add_args(parser) adds model-specific options;
    the parsed args are passed to load_model_fn(model_dir, args)."""
    parser = argparse.ArgumentParser(description=f"LLM-free survey: story -> embedding -> {model_label} -> 20 ratings")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--story", type=str, help="Path to a single story .txt")
    src.add_argument("--stories-dir", type=str, help="Folder of story .txt files")
    parser.add_argument("--model-dir", type=str, required=True, help=f"Folder with the trained {model_label} model")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write <story_name>.json (keep it out of output/)")
    parser.add_argument("--overwrite", action="store_true", help="Redo stories that already have a complete JSON")
    if add_args:
        add_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")

    stories = [Path(args.story)] if args.story else sorted(Path(args.stories_dir).glob("*.txt"))
    if not stories or not stories[0].exists():
        sys.exit(f"No story found: {args.story or args.stories_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Everything shown in the terminal also goes to <output-dir>/predict.log
    file_handler = logging.FileHandler(output_dir / "predict.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.captureWarnings(True)
    logger.info("Command: " + " ".join(sys.argv))

    # ---------- One-time setup (not counted per story) ----------
    setup_start = time.perf_counter()
    model, ckpt, cfg, info = load_model_fn(Path(args.model_dir), args)
    predict_fn(model, ckpt, torch.zeros(cfg["embedding_dim"]))  # warm-up PyTorch
    client = OpenAI(max_retries=3)  # SDK retries rate limits / 5xx / timeouts with backoff
    logger.info(f"Model: {info['name']} | {info['n_parameters']:,} numbers ({info['size_bytes'] / 1024:.0f} KB) "
                f"incl. feature step | "
                f"embeddings: {cfg['embedding_model']} | setup {time.perf_counter() - setup_start:.3f}s")

    todo = [s for s in stories if args.overwrite or not is_complete(output_dir / f"{s.stem}.json")]
    logger.info(f"{len(todo)}/{len(stories)} stories to survey")

    walls, embeds, predicts, total_cost = [], [], [], 0.0
    for i, story_path in enumerate(todo, 1):
        result = survey_story_model(client, story_path, model, ckpt, cfg, info, predict_fn)
        out_file = output_dir / f"{story_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        t = result["timing"]
        n = len(result["ratings"])
        logger.info(f"[{i}/{len(todo)}] {story_path.stem}: {n - result['n_failed']}/{n} rated, "
                    f"{t['story_wall_s']:.3f}s total (embed {t['embed_latency_s'] or 0:.3f}s, "
                    f"predict {(t['predict_latency_s'] or 0) * 1000:.2f} ms) -> {out_file}")
        if not result["n_failed"]:
            walls.append(t["story_wall_s"])
            embeds.append(t["embed_latency_s"])
            predicts.append(t["predict_latency_s"])
            total_cost += result["cost_usd"] or 0.0

    if walls:
        k = len(walls)
        logger.info(f"Done: {k}/{len(todo)} stories rated | mean per story: total {sum(walls) / k:.3f}s, "
                    f"embed {sum(embeds) / k:.3f}s, predict {sum(predicts) / k * 1000:.2f} ms | "
                    f"total cost ${total_cost:.5f}")


def main():
    run(load_model, predict_one, "M1")


if __name__ == "__main__":
    main()

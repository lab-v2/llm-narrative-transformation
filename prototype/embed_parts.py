"""
OpenAI embeddings of each story's parts (beginning / middle / end by default), for the
"read the story in parts" experiments. One request per story (all parts together).

Output: one JSON per story at <output-dir>/<story_name>.json with the part embeddings
(in story order), the word count of each part, tokens, latency and cost. Stories that
already have a JSON are skipped unless --overwrite is given.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/embed_parts.py --stories-dir data/train-prototype \
        --output-dir output_test/prototype/embeddings/text-embedding-3-large/train_parts3
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from embed_story import PRICING

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_words(text: str, n_parts: int) -> list:
    words = text.split()
    bounds = [round(i * len(words) / n_parts) for i in range(n_parts + 1)]
    return [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]


def main():
    parser = argparse.ArgumentParser(description="OpenAI embeddings of story parts")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--story", type=str)
    src.add_argument("--stories-dir", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--parts", type=int, default=3, help="Number of equal parts by words (default: 3)")
    parser.add_argument("--model", type=str, default="text-embedding-3-large")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")
    stories = [Path(args.story)] if args.story else sorted(Path(args.stories_dir).glob("*.txt"))
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stories if args.overwrite or not (out / f"{s.stem}.json").exists()]
    logger.info(f"Model: {args.model} | {args.parts} parts | {len(todo)}/{len(stories)} stories to embed")

    client = OpenAI(max_retries=3)
    price = PRICING.get(args.model)
    total_cost, n_failed = 0.0, 0
    for i, path in enumerate(todo, 1):
        parts = split_words(path.read_text(encoding="utf-8"), args.parts)
        try:
            start = time.perf_counter()
            response = client.embeddings.create(model=args.model, input=parts)
            latency_s = time.perf_counter() - start
        except Exception as e:
            n_failed += 1
            logger.error(f"[{i}/{len(todo)}] {path.stem}: failed: {e}")
            continue
        vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        tokens = response.usage.prompt_tokens
        cost = round(tokens * price / 1e6, 8) if price else None
        with open(out / f"{path.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"story_name": path.stem, "model": args.model, "n_parts": args.parts,
                       "part_words": [len(p.split()) for p in parts], "tokens": tokens,
                       "timing": {"embed_latency_s": round(latency_s, 3)}, "cost_usd": cost,
                       "timestamp": datetime.now().isoformat(timespec="seconds"), "parts": vectors}, f)
        total_cost += cost or 0.0
    logger.info(f"Done: {len(todo) - n_failed}/{len(todo)} embedded, {n_failed} failed, ${total_cost:.4f} total")


if __name__ == "__main__":
    main()

"""
OpenAI embedding for each story, for the LLM-free survey prototype.

One request per story (the whole story text as a single input), so the recorded
latency is what the cheap predictor pays per story at test time.

Output: one JSON per story at <output-dir>/<story_name>.json with the vector
under "embedding" plus tokens, latency and cost. Stories that already have a
JSON are skipped unless --overwrite is given; failed stories are not written,
so re-running retries them.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/embed_story.py --stories-dir data/train-prototype \
        --output-dir output_test/prototype/embeddings/text-embedding-3-large/train
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

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# USD per 1M input tokens (developers.openai.com/api/docs/pricing, checked 2026-10-03)
PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def embed_story(client: OpenAI, story_path: Path, model: str) -> dict:
    text = story_path.read_text(encoding="utf-8")

    call_start = time.perf_counter()
    response = client.embeddings.create(model=model, input=text)
    latency_s = time.perf_counter() - call_start

    vector = response.data[0].embedding
    tokens = response.usage.prompt_tokens
    price = PRICING.get(model)

    return {
        "story_name": story_path.stem,
        "story_file": str(story_path),
        "model": model,
        "dimensions": len(vector),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tokens": tokens,
        "timing": {"embed_latency_s": round(latency_s, 3)},
        "cost_usd": round(tokens * price / 1e6, 8) if price else None,
        "embedding": vector,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI story embeddings for the LLM-free survey prototype")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--story", type=str, help="Path to a single story .txt")
    src.add_argument("--stories-dir", type=str, help="Folder of story .txt files (embeds all of them)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write <story_name>.json (keep it out of output/)")
    parser.add_argument("--model", type=str, default="text-embedding-3-large",
                        help="OpenAI embedding model (default: text-embedding-3-large)")
    parser.add_argument("--overwrite", action="store_true", help="Re-embed stories that already have a JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")

    stories = [Path(args.story)] if args.story else sorted(Path(args.stories_dir).glob("*.txt"))
    if not stories or not stories[0].exists():
        sys.exit(f"No story found: {args.story or args.stories_dir}")
    if args.model not in PRICING:
        logger.warning(f"No pricing for {args.model}; cost_usd will be null")

    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stories if args.overwrite or not (output_dir / f"{s.stem}.json").exists()]
    logger.info(f"Model: {args.model} | {len(todo)}/{len(stories)} stories to embed")

    client = OpenAI(max_retries=3)  # SDK retries rate limits / 5xx / timeouts with backoff
    total_cost, total_latency, n_failed = 0.0, 0.0, 0
    for i, story_path in enumerate(todo, 1):
        try:
            result = embed_story(client, story_path, args.model)
        except Exception as e:
            n_failed += 1
            logger.error(f"[{i}/{len(todo)}] {story_path.stem}: embedding failed: {e}")
            continue

        out_file = output_dir / f"{story_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f)

        total_cost += result["cost_usd"] or 0.0
        total_latency += result["timing"]["embed_latency_s"]
        logger.info(
            f"[{i}/{len(todo)}] {story_path.stem}: {result['dimensions']} dims, {result['tokens']:,} tokens, "
            f"{result['timing']['embed_latency_s']:.2f}s -> {out_file}"
        )

    if todo:
        logger.info(f"Done: {len(todo) - n_failed}/{len(todo)} embedded, {n_failed} failed, "
                    f"{total_latency:.1f}s total latency, ${total_cost:.4f} total")


if __name__ == "__main__":
    main()

"""
OpenAI embedding for each survey question's text, for M4 (the question-aware model).

Uses the same embedding model as the stories, so questions and stories live in the
same space. One request for all questions.

Output: one JSON with the model name, the questionnaire it came from, and per question
its id, component, text and embedding.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/embed_questions.py \
        --questions-file prototype/collectivistic_questions_binary.json \
        --output output_test/prototype/embeddings/text-embedding-3-large/questions/collectivistic_questions.json
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


def main():
    parser = argparse.ArgumentParser(description="OpenAI embeddings of the survey questions' text (for M4)")
    parser.add_argument("--questions-file", type=str, required=True, help="Questionnaire JSON")
    parser.add_argument("--question-type", type=str, default="collectivistic_question",
                        help="Which question text to embed (default: collectivistic_question)")
    parser.add_argument("--output", type=str, required=True, help="Where to write the JSON (keep it out of output/)")
    parser.add_argument("--model", type=str, default="text-embedding-3-large",
                        help="OpenAI embedding model; must match the story embeddings (default: text-embedding-3-large)")
    args = parser.parse_args()

    out = Path(args.output)
    if out.resolve().is_relative_to((REPO_ROOT / "output").resolve()):
        sys.exit("Refusing to write under output/ (paper results). Use e.g. output_test/prototype/...")

    with open(args.questions_file, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if args.question_type in q]
    texts = [q[args.question_type]["question"] for q in questions]

    start = time.perf_counter()
    response = OpenAI(max_retries=3).embeddings.create(model=args.model, input=texts)
    latency_s = time.perf_counter() - start
    vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
    tokens = response.usage.prompt_tokens
    price = PRICING.get(args.model)

    result = {
        "model": args.model,
        "questions_file": args.questions_file,
        "question_type": args.question_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tokens": tokens,
        "cost_usd": round(tokens * price / 1e6, 8) if price else None,
        "latency_s": round(latency_s, 3),
        "questions": [{"id": str(q["id"]), "component": q["component"], "text": t, "embedding": v}
                      for q, t, v in zip(questions, texts, vectors)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    logger.info(f"{len(questions)} questions, {len(vectors[0])} dims, {tokens} tokens, {latency_s:.2f}s -> {out}")


if __name__ == "__main__":
    main()

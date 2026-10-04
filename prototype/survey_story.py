"""
LLM survey for the LLM-free survey prototype: ONE call per story.

Sends the story plus all 20 survey questions in a single request and gets back
only the 1-5 ratings (strict JSON schema, no justifications/excerpts). Records
latency, tokens and cost per story so it can be compared with the cheap
embedding-based predictor later.

Questions and the 1-5 scale come from the project's question files:
  --problem inverse (default, individualistic stories) -> data/collectivistic_questions.json
  --problem forward (collectivistic stories)           -> data/individualistic_questions.json

Output: one JSON per story at <output-dir>/<story_name>.json with a "ratings"
dict ({question_id: 1-5}). Stories whose JSON already has all ratings are
skipped unless --overwrite is given; stories with failed ratings are redone.

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/survey_story.py --story data/train-prototype/Jay_Z.txt \
        --output-dir output_test/prototype/survey/gpt-6-sol/dry_run_single
    python prototype/survey_story.py --stories-dir data/train-prototype \
        --output-dir output_test/prototype/survey/gpt-6-sol/train
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

# USD per 1M tokens (developers.openai.com/api/docs/pricing, checked 2026-10-03)
PRICING = {
    "gpt-6-sol": {"input": 2.00, "output": 10.00},
}


# ==========================================================
# Prompt
# ==========================================================

def load_questions(questions_file: Path, question_type: str) -> list:
    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    return [q for q in questions if question_type in q]


def build_prompts(story_text: str, questions: list, question_type: str):
    """System prompt = the project's rating instructions minus the excerpt-quoting line."""
    rating_lines = [line for line in questions[0][question_type]["rating"]
                    if not line.startswith("Identify the TOP sections")]
    system_prompt = "\n".join(rating_lines).replace("analyze the question", "analyze each question")

    question_lines = "\n".join(f"q{q['id']}: {q[question_type]['question']}" for q in questions)
    user_prompt = (
        f"STORY:\n{story_text}\n\n"
        f"QUESTIONS:\n{question_lines}\n\n"
        "Rate the story on each question independently using the 1–5 scale. "
        "Return only the ratings."
    )
    return system_prompt, user_prompt


def ratings_schema(questions: list) -> dict:
    """Strict JSON schema: one integer 1-5 per question, all required."""
    props = {f"q{q['id']}": {"type": "integer", "enum": [1, 2, 3, 4, 5]} for q in questions}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "survey_ratings",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False,
            },
        },
    }


# ==========================================================
# Survey
# ==========================================================

def survey_story(client: OpenAI, story_path: Path, args) -> dict:
    question_type = "individualistic_question" if args.problem == "forward" else "collectivistic_question"
    questions_name = "individualistic_questions.json" if args.problem == "forward" else "collectivistic_questions.json"
    questions_file = Path(args.data_dir) / questions_name

    wall_start = time.perf_counter()
    questions = load_questions(questions_file, question_type)
    system_prompt, user_prompt = build_prompts(story_path.read_text(encoding="utf-8"), questions, question_type)

    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": args.max_tokens,
        "response_format": ratings_schema(questions),
    }
    if args.temperature is not None:
        request["temperature"] = args.temperature

    ratings = {str(q["id"]): None for q in questions}
    tokens = {"input": 0, "output": 0, "reasoning": None}
    error = None

    call_start = time.perf_counter()
    try:
        response = client.chat.completions.create(**request)
        latency_s = time.perf_counter() - call_start

        usage = response.usage
        tokens["input"] = usage.prompt_tokens
        tokens["output"] = usage.completion_tokens
        details = getattr(usage, "completion_tokens_details", None)
        tokens["reasoning"] = getattr(details, "reasoning_tokens", None)

        message = response.choices[0].message
        if not message.content:
            raise ValueError(f"empty response (finish_reason={response.choices[0].finish_reason}, "
                             f"refusal={getattr(message, 'refusal', None)})")
        parsed = json.loads(message.content)
        for q in questions:
            value = parsed.get(f"q{q['id']}")
            ratings[str(q["id"])] = value if value in (1, 2, 3, 4, 5) else None
    except Exception as e:
        latency_s = time.perf_counter() - call_start
        error = str(e)
        logger.error(f"{story_path.stem}: survey failed: {e}")
    wall_s = time.perf_counter() - wall_start

    pricing = PRICING.get(args.model)
    cost_usd = None
    if pricing:
        cost_usd = round((tokens["input"] * pricing["input"] + tokens["output"] * pricing["output"]) / 1e6, 6)

    result = {
        "story_name": story_path.stem,
        "story_file": str(story_path),
        "model": args.model,
        "problem_type": args.problem,
        "questions_file": str(questions_file),
        "temperature": args.temperature,  # null = model default
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ratings": ratings,
        "n_failed": sum(r is None for r in ratings.values()),
        "tokens": {**tokens, "total": tokens["input"] + tokens["output"]},
        "timing": {"llm_latency_s": round(latency_s, 3), "story_wall_s": round(wall_s, 3)},
        "cost_usd": cost_usd,
    }
    if error:
        result["error"] = error
    return result


def is_complete(out_file: Path) -> bool:
    """True if a saved survey exists and every question got a rating."""
    if not out_file.exists():
        return False
    with open(out_file, encoding="utf-8") as f:
        return json.load(f).get("n_failed", 1) == 0


def main():
    parser = argparse.ArgumentParser(description="Single-call LLM survey (20 ratings) for the LLM-free survey prototype")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--story", type=str, help="Path to a single story .txt")
    src.add_argument("--stories-dir", type=str, help="Folder of story .txt files (surveys all of them)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write <story_name>.json (keep it out of output/)")
    parser.add_argument("--model", type=str, default="gpt-6-sol", help="OpenAI model (default: gpt-6-sol)")
    parser.add_argument("--problem", type=str, choices=["forward", "inverse"], default="inverse",
                        help="inverse = individualistic stories (default), forward = collectivistic")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: not sent, model default; gpt-6-sol only accepts 1)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="max_completion_tokens, includes reasoning tokens (default: 4096)")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data"),
                        help="Folder holding the questions JSON files (default: <repo>/data)")
    parser.add_argument("--overwrite", action="store_true", help="Re-survey stories that already have a complete JSON")
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
    todo = [s for s in stories if args.overwrite or not is_complete(output_dir / f"{s.stem}.json")]
    logger.info(f"Model: {args.model} | problem: {args.problem} | {len(todo)}/{len(stories)} stories to survey")

    client = OpenAI(max_retries=3)  # SDK retries rate limits / 5xx / timeouts with backoff
    total_cost, total_latency = 0.0, 0.0
    for i, story_path in enumerate(todo, 1):
        result = survey_story(client, story_path, args)
        out_file = output_dir / f"{story_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n = len(result["ratings"])
        total_cost += result["cost_usd"] or 0.0
        total_latency += result["timing"]["llm_latency_s"]
        logger.info(
            f"[{i}/{len(todo)}] {story_path.stem}: {n - result['n_failed']}/{n} rated, "
            f"{result['timing']['llm_latency_s']:.1f}s, {result['tokens']['total']:,} tokens, "
            f"${result['cost_usd'] or 0:.4f} -> {out_file}"
        )

    if todo:
        logger.info(f"Done: {len(todo)} stories, {total_latency:.1f}s total LLM latency, ${total_cost:.4f} total")


if __name__ == "__main__":
    main()

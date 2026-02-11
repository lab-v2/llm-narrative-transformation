#!/usr/bin/env python3
"""
LLM Survey Standalone Module

Conducts surveys on pre-transformed stories from evaluation_data directory.
Separates the model used for transformation from the model used for diagnosis/evaluation.

Features:
- Surveys 3 versions per story: original, baseline_transformed, abduction_transformed
- Multi-provider support via LiteLLM (same as llm_survey.py)
- Retry with exponential backoff for rate limits
- Cost and token tracking
- Progress tracking with tqdm
- Intermediate result saving after each story
- dotenv support for API keys and configuration

Directory structure:
  Input:  evaluation_data/{tuned_model_name}/{original_stories_type}/{story_name}/{original_story.txt, baseline_transformed.txt, abduction_transformed.txt}
  Output: survey/{tuned_model_name}/{model_name_for_survey}/{original_stories_type}/{story_name}/survey_*.json
"""

import os
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

try:
    from litellm import completion
    from litellm.exceptions import RateLimitError, APIError
except ImportError:
    raise ImportError(
        "litellm is required. Install with: pip install litellm"
    )

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "tqdm is required for progress bars. Install with: pip install tqdm"
    )

logger = logging.getLogger(__name__)

# ==========================================================
# Cost Tracking (Same as llm_survey.py)
# ==========================================================

MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
    "gpt-4o-mini-2024-07-18": {"input": 0.0008, "output": 0.0032},
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm2:D5JJuHZi": {"input": 0.0008, "output": 0.0032},
    "xai/grok-4-fast-reasoning": {"input": 0.0002, "output": 0.0005},
    # Anthropic Claude
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    # AWS Bedrock - Meta Llama
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": {"input": 0.00024, "output": 0.00097},
    "meta.llama4": {"input": 0.00024, "output": 0.00097},
    "bedrock/us.meta.llama3-2-11b-instruct-v1:0": {"input": 0.00016, "output": 0.00072},
    # AWS Bedrock - DeepSeek
    "bedrock/us.deepseek.r1-v1:0": {"input": 0.00135, "output": 0.0054},
    "bedrock/deepseek-llm-r1-distill-qwen-32b": {"input": 0.00135, "output": 0.0054}
}


def get_pricing_for_model(model: str) -> Dict[str, float]:
    """Get pricing for a model."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    for key in MODEL_PRICING:
        if model.startswith(key):
            return MODEL_PRICING[key]

    logger.warning(f"No pricing found for model {model}, using default")
    return {"input": 0.0025, "output": 0.01}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model call."""
    pricing = get_pricing_for_model(model)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost


class CostTracker:
    """Tracks tokens and costs across LLM calls."""

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.per_story_breakdown = []

    def add_call(self, story_name: str, input_tokens: int, output_tokens: int):
        """Add a call to the tracker."""
        cost = calculate_cost(self.model, input_tokens, output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        self.per_story_breakdown.append({
            "story_name": story_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "timestamp": datetime.now().isoformat()
        })

    def to_dict(self) -> Dict:
        """Convert tracker to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "total_stories_processed": len(self.per_story_breakdown),
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens
            },
            "total_cost_usd": round(self.total_cost, 6),
            "per_story_breakdown": self.per_story_breakdown
        }

    def save(self, output_path: Path):
        """Save cost tracking to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Cost tracking saved to {output_path}")


# ==========================================================
# API Key Management (Same as llm_survey.py)
# ==========================================================

def get_api_key_for_model(model: str) -> Optional[str]:
    """Get the appropriate API key based on model prefix."""
    if model.startswith("openrouter/"):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it with: export OPENROUTER_API_KEY='your-key'"
            )
        return key

    if model.startswith("xai/"):
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError(
                "XAI_API_KEY environment variable not set. "
                "Set it with: export XAI_API_KEY='your-key'"
            )
        return key

    if model.startswith("anthropic/") or model.startswith("claude"):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )
        return key

    # AWS Bedrock models
    if model.startswith("bedrock/") or model.startswith("qwen"):
        return None

    # Default to OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key'"
        )
    return key


# ==========================================================
# LLM Calling with Retry Logic (Same as llm_survey.py)
# ==========================================================

def call_llm_with_retry(
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3
) -> Tuple[str, int, int]:
    """Call LLM with exponential backoff retry logic."""
    api_key = get_api_key_for_model(model)

    extra_headers = None
    if model.startswith("openrouter/"):
        extra_headers = {
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CONNECT Project"
        }

    for attempt in range(max_retries):
        try:
            # GPT-5.x models use max_completion_tokens
            if model.startswith("gpt-5"):
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature
                }
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

            if api_key is not None:
                kwargs["api_key"] = api_key

            if extra_headers is not None:
                kwargs["extra_headers"] = extra_headers

            response = completion(**kwargs)

            response_text = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {}) or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))

            return response_text, input_tokens, output_tokens

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise

        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"API error after {max_retries} attempts: {e}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            raise

    raise RuntimeError("Should not reach here")


# ==========================================================
# Response Parsing (Same as llm_survey.py)
# ==========================================================

def parse_llm_response(response_text: str) -> Tuple[Optional[int], str, str]:
    """Parse LLM response to extract rating, justification, and evidence."""
    rating = None
    rating_match = re.search(r'(?:Rating:|^)\s*(\d)', response_text, re.MULTILINE | re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
    else:
        first_line_match = re.match(r'\s*(\d)', response_text)
        if first_line_match:
            rating = int(first_line_match.group(1))

    evidence = ""
    evidence_matches = re.findall(r'"([^"]+)"', response_text)
    if evidence_matches:
        evidence = " | ".join(evidence_matches[:3])

    justification = response_text.strip()

    return rating, justification, evidence


# ==========================================================
# Questions Loading (Same as llm_survey.py)
# ==========================================================

def load_questions(questions_file: str) -> List[Dict]:
    """Load questions from JSON file."""
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["questions"]


# ==========================================================
# Standalone Survey Functions
# ==========================================================

def discover_stories(evaluation_data_dir: str, tuned_model_name: str, original_stories_type: str) -> List[str]:
    """
    Discover all story names in the evaluation_data directory.

    Args:
        evaluation_data_dir: Base evaluation data directory
        tuned_model_name: Name of tuned model
        original_stories_type: Type of original stories (individualistic/collectivistic)

    Returns:
        List of story names
    """
    stories_dir = Path(evaluation_data_dir) / tuned_model_name / original_stories_type
    
    if not stories_dir.exists():
        logger.error(f"Stories directory does not exist: {stories_dir}")
        return []

    # Get all subdirectories (story names)
    story_names = [d.name for d in stories_dir.iterdir() if d.is_dir()]
    return sorted(story_names)


def load_story_content(file_path: Path) -> Optional[str]:
    """
    Load story content from file.

    Args:
        file_path: Path to story file

    Returns:
        Story content or None if file doesn't exist
    """
    if not file_path.exists():
        logger.warning(f"Story file not found: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read story file {file_path}: {e}")
        return None


def conduct_survey_single_story_version(
        story_content: str,
        story_identifier: str,
        questions_file: str,
        problem_type: str,
        model: str,
        temperature: float = 0.0,
        cost_tracker: Optional[CostTracker] = None
) -> Dict:
    """
    Conduct LLM survey on a single story version.

    Args:
        story_content: Content of the story to survey
        story_identifier: Identifier for logging (e.g., "story_name/original")
        questions_file: Path to questions JSON file
        problem_type: "forward" or "inverse"
        model: Model name for diagnosis/survey
        temperature: LLM temperature
        cost_tracker: Optional cost tracker to update

    Returns:
        Survey result dictionary
    """
    questions = load_questions(questions_file)

    # Determine which question type to use
    question_type = "individualistic_question" if problem_type == "forward" else "collectivistic_question"

    result = {
        "story_identifier": story_identifier,
        "model": model,
        "problem_type": problem_type,
        "timestamp": datetime.now().isoformat(),
        "questions_and_answers": []
    }

    story_total_input = 0
    story_total_output = 0

    # Process each question
    for q in questions:
        if question_type not in q:
            logger.warning(f"Question {q['id']} missing {question_type}, skipping")
            continue

        question_obj = q[question_type]
        question_text = question_obj["question"]
        rating_instructions = question_obj["rating"]

        system_prompt = "\n".join(rating_instructions)
        user_prompt = (
            f"STORY:\n{story_content}\n\n"
            f"QUESTION:\n{question_text}\n\n"
            "Use the following 1–5 rating scale:\n"
            "1 = Entirely individual perspective\n"
            "2 = Primarily individual but with some group influence\n"
            "3 = Balanced between individual and group\n"
            "4 = Primarily group-oriented\n"
            "5 = Entirely group/community perspective\n\n"
            "Answer the question about the story using the scale and briefly justify your answer. "
            "Put the rating FIRST (e.g., 'Rating: 4'), THEN justify. "
            "Finally, quote a few short excerpts from the story that exemplify your chosen rating."
            " Format excerpts as a simple numbered list with plain quotes:\n"
            "1. \"excerpt text\"\n"
            "2. \"excerpt text\"\n"
            "Do NOT use bold (**), italics (*), or explanatory notes in parentheses."
        )

        try:
            response_text, input_tokens, output_tokens = call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                max_retries=3
            )

            story_total_input += input_tokens
            story_total_output += output_tokens

            rating, justification, evidence = parse_llm_response(response_text)

            entry = {
                "id": q["id"],
                "component": q["component"],
                "type": question_type,
                "question": question_text,
                "rating": rating,
                "justification": justification,
                "evidence": evidence,
                "raw_response": response_text,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens
                }
            }
            result["questions_and_answers"].append(entry)

        except Exception as e:
            logger.error(f"Failed to process question {q['id']} for {story_identifier}: {e}")
            entry = {
                "id": q["id"],
                "component": q["component"],
                "type": question_type,
                "question": question_text,
                "error": str(e),
                "rating": None,
                "raw_response": None
            }
            result["questions_and_answers"].append(entry)

    if cost_tracker:
        cost_tracker.add_call(story_identifier, story_total_input, story_total_output)

    result["total_tokens"] = {
        "input": story_total_input,
        "output": story_total_output,
        "total": story_total_input + story_total_output
    }

    return result


def conduct_surveys_standalone(
        evaluation_data_dir: str,
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        questions_file: str,
        survey_output_dir: str,
        problem_type: str,
        temperature: float,
        story_names: Optional[List[str]] = None,
        skip_existing: bool = True
) -> Tuple[int, int]:
    """
    Conduct surveys on all story versions in evaluation_data.

    For each story, surveys 3 versions:
    1. original_story.txt
    2. baseline_transformed.txt
    3. abduction_transformed.txt

    Args:
        evaluation_data_dir: Base evaluation data directory
        tuned_model_name: Name of tuned model that generated transformed stories
        model_name_for_survey: Name of model to use for diagnosis/survey
        original_stories_type: Type of original stories (individualistic/collectivistic)
        questions_file: Path to questions JSON file
        survey_output_dir: Base survey output directory
        problem_type: "forward" or "inverse"
        temperature: LLM temperature
        story_names: Optional list of specific story names to process
        skip_existing: Skip if survey JSON already exists

    Returns:
        Tuple of (total_processed, total_failed)
    """
    # Create output directory structure
    sanitized_survey_model = model_name_for_survey.replace("/", "-").replace(":", "-")
    output_base = Path(survey_output_dir) / tuned_model_name / sanitized_survey_model / original_stories_type
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Survey results will be saved to: {output_base}")

    # Discover stories if not provided
    if story_names is None:
        story_names = discover_stories(evaluation_data_dir, tuned_model_name, original_stories_type)

    if not story_names:
        logger.error("No stories found to process")
        return 0, 0

    logger.info(f"Found {len(story_names)} stories to process")

    # Initialize cost tracker
    cost_tracker = CostTracker(model_name_for_survey)

    # Track results
    total_processed = 0
    total_failed = 0

    # Story versions to survey
    versions = [
        ("original_story.txt", "survey_original_story.json"),
        ("baseline_transformed.txt", "survey_baseline_transformed.json"),
        ("abduction_transformed.txt", "survey_abduction_transformed.json")
    ]

    # Process each story
    for story_name in tqdm(story_names, desc=f"Surveying stories ({model_name_for_survey})", unit="story"):
        story_dir = Path(evaluation_data_dir) / tuned_model_name / original_stories_type / story_name
        output_story_dir = output_base / story_name
        output_story_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing story: {story_name}")

        # Process each version
        for input_filename, output_filename in versions:
            input_file = story_dir / input_filename
            output_file = output_story_dir / output_filename

            # Skip if already exists and skip_existing is True
            if output_file.exists() and skip_existing:
                logger.info(f"  ✓ {output_filename} already exists, skipping")
                continue

            # Load story content
            story_content = load_story_content(input_file)
            if story_content is None:
                logger.error(f"  ✗ Failed to load {input_filename}")
                total_failed += 1
                continue

            try:
                # Conduct survey on this version
                story_identifier = f"{story_name}/{input_filename.replace('.txt', '')}"
                result = conduct_survey_single_story_version(
                    story_content=story_content,
                    story_identifier=story_identifier,
                    questions_file=questions_file,
                    problem_type=problem_type,
                    model=model_name_for_survey,
                    temperature=temperature,
                    cost_tracker=cost_tracker
                )

                # Save result
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                logger.info(f"  ✓ Saved: {output_filename}")
                total_processed += 1

            except Exception as e:
                logger.error(f"  ✗ Failed to process {input_filename}: {e}")
                total_failed += 1
                continue

        # Save cost tracking after each story
        cost_file = output_base.parent / "cost_tracking.json"
        cost_tracker.save(cost_file)

    # Save final cost tracking
    cost_file = output_base.parent / "cost_tracking.json"
    cost_tracker.save(cost_file)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Survey complete!")
    logger.info(f"Processed: {total_processed}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"{'=' * 60}\n")

    return total_processed, total_failed


# ==========================================================
# CLI & Main
# ==========================================================

def setup_logging(log_file: str = "llm_survey_standalone.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Survey pre-transformed stories using a separate model for diagnosis"
    )

    # Required arguments
    parser.add_argument(
        "--tuned-model-name",
        required=True,
        help="Name of the fine-tuned model that generated the transformed stories"
    )
    parser.add_argument(
        "--model-name-for-survey",
        required=True,
        help="Name of the model to use for diagnosis/evaluation survey"
    )
    parser.add_argument(
        "--original-stories-type",
        required=True,
        choices=["individualistic", "collectivistic"],
        help="Type of original stories being evaluated"
    )
    parser.add_argument(
        "--questions-file",
        required=True,
        help="Path to questions JSON file"
    )

    # Optional arguments with defaults from .env
    parser.add_argument(
        "--evaluation-data-dir",
        default=os.getenv("DEFAULT_EVALUATION_DATA_DIR", "evaluation_data"),
        help="Base directory where evaluation_data lives (default: evaluation_data)"
    )
    parser.add_argument(
        "--survey-output-dir",
        default=os.getenv("DEFAULT_SURVEY_OUTPUT_DIR", "survey"),
        help="Base directory where survey results should be saved (default: survey)"
    )
    parser.add_argument(
        "--problem-type",
        default="forward",
        choices=["forward", "inverse"],
        help="Problem type: forward or inverse (default: forward)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for survey (default: 0.0)"
    )
    parser.add_argument(
        "--story-names",
        default=None,
        help="Comma-separated list of specific story names to process (default: all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if survey JSON already exists (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing survey files, re-process them"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    setup_logging()
    logger.info("Starting LLM Survey Standalone")

    args = parse_arguments()

    logger.info(f"Configuration:")
    logger.info(f"  Tuned Model: {args.tuned_model_name}")
    logger.info(f"  Survey Model: {args.model_name_for_survey}")
    logger.info(f"  Original Stories Type: {args.original_stories_type}")
    logger.info(f"  Problem Type: {args.problem_type}")
    logger.info(f"  Evaluation Data Dir: {args.evaluation_data_dir}")
    logger.info(f"  Survey Output Dir: {args.survey_output_dir}")
    logger.info(f"  Questions File: {args.questions_file}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Skip Existing: {args.skip_existing}")

    # Parse story names if provided
    story_names = None
    if args.story_names:
        story_names = [s.strip() for s in args.story_names.split(",")]
        logger.info(f"  Specific Stories: {story_names}")

    # Validate questions file exists
    if not Path(args.questions_file).exists():
        logger.error(f"Questions file not found: {args.questions_file}")
        return 1

    try:
        # Run surveys
        processed, failed = conduct_surveys_standalone(
            evaluation_data_dir=args.evaluation_data_dir,
            tuned_model_name=args.tuned_model_name,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            questions_file=args.questions_file,
            survey_output_dir=args.survey_output_dir,
            problem_type=args.problem_type,
            temperature=args.temperature,
            story_names=story_names,
            skip_existing=args.skip_existing
        )

        if failed > 0:
            logger.warning(f"Completed with {failed} failures")
            return 1

        logger.info("Survey completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

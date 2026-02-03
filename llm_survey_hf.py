"""
LLM Survey Module - HuggingFace Version

Conducts surveys on stories using HuggingFace models (Mistral, Llama, etc.)

Features:
- HuggingFace model support via hf_inference
- Cost tracking (zero cost for local models)
- Token tracking
- Progress tracking with tqdm
- Intermediate result saving after each story
"""

import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal

try:
    from hf_inference import call_hf_with_retry
except ImportError:
    raise ImportError(
        "hf_inference module required. Make sure hf_inference.py is in the same directory"
    )

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "tqdm is required for progress bars. Install with: pip install tqdm"
    )

logger = logging.getLogger(__name__)

# ==========================================================
# Cost Tracking (Zero cost for HF models)
# ==========================================================

class CostTracker:
    """Tracks tokens and costs across LLM calls. HF models have zero cost."""

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0  # Always 0 for HF models
        self.per_story_breakdown = []

    def add_call(self, story_name: str, input_tokens: int, output_tokens: int):
        """Add a call to the tracker. Cost is always 0 for HF models."""
        # HF models running locally have zero API cost
        cost = 0.0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        self.per_story_breakdown.append({
            "story_name": story_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": 0.0,  # Zero cost for HF
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
            "total_cost_usd": 0.0,  # Zero cost for HF models
            "per_story_breakdown": self.per_story_breakdown
        }

    def save(self, output_path: Path):
        """Save cost tracking to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Cost tracking saved to {output_path}")


# ==========================================================
# Survey Functions
# ==========================================================

def load_questions(questions_file: str) -> List[Dict]:
    """
    Load questions from JSON file.

    Args:
        questions_file: Path to questions JSON file

    Returns:
        List of question dictionaries
    """
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["questions"]


def conduct_survey_single_story(
        story: Dict[str, str],
        questions_file: str,
        problem_type: str,
        model: str,
        temperature: float = 0.0,
        cost_tracker: Optional[CostTracker] = None
) -> Dict:
    """
    Conduct HF survey on a single story.

    Args:
        story: Story dict with 'name', 'path', 'content'
        questions_file: Path to questions JSON file
        problem_type: "forward" or "inverse"
        model: HuggingFace model ID
        temperature: LLM temperature
        cost_tracker: Optional cost tracker to update

    Returns:
        Survey result dictionary
    """
    questions = load_questions(questions_file)

    # Determine which question type to use
    question_type = "individualistic_question" if problem_type == "forward" else "collectivistic_question"

    result = {
        "story_name": story["name"],
        "story_file": story["path"],
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

        # Format prompts (matching their style)
        system_prompt = "\n".join(rating_instructions)
        user_prompt = (
            f"STORY:\n{story['content']}\n\n"
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
            # Call HF model with retry logic
            response_text, input_tokens, output_tokens = call_hf_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                max_retries=3
            )

            story_total_input += input_tokens
            story_total_output += output_tokens

            # Parse response
            rating, justification, evidence = parse_llm_response(response_text)

            # Store result
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
            logger.error(f"Failed to process question {q['id']} for story {story['name']}: {e}")
            # Store error entry
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

    # Update cost tracker
    if cost_tracker:
        cost_tracker.add_call(story["name"], story_total_input, story_total_output)

    # Add token summary to result
    result["total_tokens"] = {
        "input": story_total_input,
        "output": story_total_output,
        "total": story_total_input + story_total_output
    }

    return result


def parse_llm_response(response_text: str) -> Tuple[Optional[int], str, str]:
    """
    Parse LLM response to extract rating, justification, and evidence.

    Expected format:
        Rating: 1

        Justification text here...

        Evidence: "Quote from story..."

    Args:
        response_text: Raw LLM response

    Returns:
        Tuple of (rating, justification, evidence)
    """
    # Try to extract rating
    rating = None
    rating_match = re.search(r'(?:Rating:|^)\s*(\d)', response_text, re.MULTILINE | re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
    else:
        # Fallback: try to find any digit at the start
        first_line_match = re.match(r'\s*(\d)', response_text)
        if first_line_match:
            rating = int(first_line_match.group(1))

    # Try to extract evidence (text in quotes)
    evidence = ""
    evidence_matches = re.findall(r'"([^"]+)"', response_text)
    if evidence_matches:
        evidence = " | ".join(evidence_matches[:3])  # Take first 3 quotes

    # Justification is everything (we keep full response for now)
    justification = response_text.strip()

    return rating, justification, evidence


def sanitize_model_name(model: str) -> str:
    """
    Sanitize model name for use in directory paths.

    Args:
        model: Model name (e.g., "mistralai/Mistral-7B-v0.3")

    Returns:
        Sanitized name (e.g., "mistralai-mistral-7b-v0-3")
    """
    return model.replace("/", "-").replace(":", "-")


def conduct_surveys(
        stories: List[Dict[str, str]],
        questions_file: str,
        problem_type: str,
        model: str,
        temperature: float,
        output_dir: str
) -> Tuple[List[Dict], List[str]]:
    """
    Conduct surveys on multiple stories using HF models.

    Args:
        stories: List of story dicts from story_loader
        questions_file: Path to questions JSON file
        problem_type: "forward" or "inverse"
        model: HuggingFace model ID
        temperature: LLM temperature
        output_dir: Base output directory

    Returns:
        Tuple of (list of survey results, list of failed story names)
    """
    # Create model-specific directory
    sanitized_model = sanitize_model_name(model)
    model_dir = Path(output_dir) / sanitized_model / problem_type
    survey_dir = model_dir / "survey_results"
    survey_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Survey results will be saved to: {survey_dir}")

    # Initialize cost tracker
    cost_tracker = CostTracker(model)

    # Track results and failures
    all_results = []
    failed_stories = []

    # Process each story with progress bar
    for story in tqdm(stories, desc=f"Surveying stories ({model})", unit="story"):
        logger.info(f"Processing story: {story['name']}")

        # Check if already processed
        output_file = survey_dir / f"{story['name']}.json"
        if output_file.exists():
            logger.info(f"Story {story['name']} already processed, skipping")
            # Load existing result for completeness
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            all_results.append(result)
            continue

        try:
            # Conduct survey
            result = conduct_survey_single_story(
                story=story,
                questions_file=questions_file,
                problem_type=problem_type,
                model=model,
                temperature=temperature,
                cost_tracker=cost_tracker
            )

            # Save result immediately
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Saved: {output_file.name}")
            all_results.append(result)

            # Save cost tracking after each story
            cost_file = model_dir / "cost_tracking.json"
            cost_tracker.save(cost_file)

        except Exception as e:
            logger.error(f"✗ Failed to process story {story['name']}: {e}")
            failed_stories.append(story['name'])
            continue

    # Save final cost tracking
    cost_file = model_dir / "cost_tracking.json"
    cost_tracker.save(cost_file)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Survey complete!")
    logger.info(f"Processed: {len(all_results)}/{len(stories)} stories")
    logger.info(f"Failed: {len(failed_stories)} stories")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"{'=' * 60}\n")

    # Save failed stories list if any
    if failed_stories:
        failed_file = model_dir / "failed_stories.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "failed_count": len(failed_stories),
                "failed_stories": failed_stories,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        logger.warning(f"Failed stories list saved to: {failed_file}")

    return all_results, failed_stories

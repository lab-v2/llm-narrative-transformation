"""
Baseline Transformation Script — Chain-of-Thought (CoT) Prompting

CoT variant of the baseline: instead of a single "make it more X" instruction,
the model is asked to reason step by step (analyze -> plan -> rewrite) and then
emit the final story inside <rewritten_story>...</rewritten_story> tags.

This is a generic CoT baseline (no scoring rubric injected, no preservation
constraint) so that the ONLY difference from the zero-shot baseline is the added
step-by-step reasoning — a clean ablation of the prompting technique.

Serves as a comparison baseline against the abduction-based approach.

Steps:
1. Load original story
2. Copy original survey from Phase 2 iteration_0
3. Transform story using the CoT prompt, extract the tagged story
4. Survey the transformed story
5. Save everything (incl. the raw reasoning) with cost tracking
"""

import json
import logging
import argparse
import re
import sys
import shutil
from pathlib import Path

# Add these two lines:
from dotenv import load_dotenv
load_dotenv()

from story_loader import load_single_story
from llm_survey import (
    call_llm_with_retry,
    conduct_survey_single_story,
    sanitize_model_name,
    CostTracker
)

logger = logging.getLogger(__name__)


# ==========================================================
# CoT Prompt Template
# ==========================================================

COT_SYSTEM_PROMPT = "You are an expert creative writer and literary analyst."


def create_cot_prompt(story_text: str, problem_type: str) -> str:
    """
    Create the chain-of-thought baseline transformation prompt.

    Args:
        story_text: Original story text
        problem_type: "forward" or "inverse"

    Returns:
        Prompt string
    """
    if problem_type == "forward":
        target, source = "individualistic", "collectivistic"
    else:  # inverse
        target, source = "collectivistic", "individualistic"

    prompt = f"""Your task is to rewrite the story below so that it reads as more {target} \
(currently it reads as more {source}).

Think step by step before writing the new version:

Step 1 — Analyze: Identify the specific elements that make the story {source} — \
e.g. the characters' goals and motivations, the source of conflict, how key \
decisions are made, how relationships and values are portrayed, and how it resolves.

Step 2 — Plan: For each element you identified, state concretely how you will \
reframe it to be more {target}.

Step 3 — Rewrite: Apply your plan and produce the full rewritten story.

Show your reasoning for Steps 1 and 2, then output ONLY the final rewritten \
story wrapped exactly in these tags:
<rewritten_story>
...complete rewritten story...
</rewritten_story>

Story:
{story_text}"""
    return prompt


def extract_rewritten_story(raw_response: str) -> str:
    """
    Extract the story from between <rewritten_story>...</rewritten_story> tags.

    Falls back to the full raw response (with a warning) if the tags are absent,
    so a malformed CoT response never silently drops the transformation.

    Args:
        raw_response: Full LLM output (reasoning + tagged story)

    Returns:
        The extracted story text (or the raw response as a fallback)
    """
    match = re.search(
        r"<rewritten_story>\s*(.*?)\s*</rewritten_story>",
        raw_response,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    logger.warning(
        "No <rewritten_story> tags found in CoT response — falling back to the "
        "full raw response. Check transformation_raw.txt for this story."
    )
    return raw_response.strip()


# ==========================================================
# Main CoT Baseline Function
# ==========================================================

def run_cot_transformation(
        story_path: str,
        problem_type: str,
        model: str,
        temperature: float,
        data_dir: str,
        phase2_dir: str,
        output_dir: str,
        max_tokens: int = 8096
) -> str:
    """
    Run CoT baseline transformation for a single story.

    Args:
        story_path: Path to original story file
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        data_dir: Data directory (for questions file)
        phase2_dir: Phase 2 output directory (to find original survey)
        output_dir: CoT baseline output directory
        max_tokens: Max output tokens for the transformation call

    Returns:
        Path to story output directory
    """
    logger.info("=" * 60)
    logger.info("COT BASELINE TRANSFORMATION")
    logger.info("=" * 60)

    # Load original story
    logger.info("Loading original story...")
    story = load_single_story(story_path)
    logger.info(f"✓ Loaded: {story['name']}")

    # Create output directory
    sanitized_model = sanitize_model_name(model)
    story_output_dir = Path(output_dir) / sanitized_model / problem_type / story['name']
    story_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {story_output_dir}")

    # Save original story
    original_story_file = story_output_dir / "original_story.txt"
    original_story_file.write_text(story['content'], encoding='utf-8')
    logger.info(f"✓ Saved original story")

    # Copy original survey from Phase 2 iteration_0
    logger.info("\nCopying original survey from Phase 2...")
    phase2_survey_path = (
            Path(phase2_dir) / sanitized_model / problem_type /
            story['name'] / "iteration_0" / "survey.json"
    )

    if not phase2_survey_path.exists():
        logger.error(f"Phase 2 survey not found: {phase2_survey_path}")
        logger.error(f"Please run Phase 2 first for this story/model/problem combination")
        sys.exit(1)

    original_survey_file = story_output_dir / "original_survey.json"
    shutil.copy2(phase2_survey_path, original_survey_file)
    logger.info(f"✓ Copied from: {phase2_survey_path}")

    # Initialize cost tracker
    cost_tracker = CostTracker(model)

    # Transform story using CoT prompt
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMING STORY (Chain-of-Thought Prompt)")
    logger.info("=" * 60)

    prompt = create_cot_prompt(story['content'], problem_type)
    target = 'individualistic' if problem_type == 'forward' else 'collectivistic'
    logger.info(f"Prompt: CoT reasoning -> rewrite story more {target}")

    try:
        raw_response, input_tokens, output_tokens = call_llm_with_retry(
            system_prompt=COT_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3
        )

        # Track cost
        cost_tracker.add_call(
            story_name=f"{story['name']}_transformation",
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        logger.info(f"✓ Story transformed")
        logger.info(f"  Tokens: {input_tokens + output_tokens:,}")
        logger.info(f"  Cost: ${cost_tracker.total_cost:.4f}")

    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        sys.exit(1)

    # Save the raw CoT response (reasoning + tagged story) for transparency
    raw_response_file = story_output_dir / "transformation_raw.txt"
    raw_response_file.write_text(raw_response, encoding='utf-8')
    logger.info(f"✓ Saved raw CoT response")

    # Extract the final rewritten story from the tags
    transformed_story = extract_rewritten_story(raw_response)

    # Save transformed story (this is what the evaluation pipeline reads)
    transformed_story_file = story_output_dir / "transformed_story.txt"
    transformed_story_file.write_text(transformed_story, encoding='utf-8')
    logger.info(f"✓ Saved transformed story")

    # Survey the transformed story
    logger.info("\n" + "=" * 60)
    logger.info("SURVEYING TRANSFORMED STORY")
    logger.info("=" * 60)

    # Determine questions file
    if problem_type == "forward":
        questions_file = Path(data_dir) / "individualistic_questions.json"
    else:  # inverse
        questions_file = Path(data_dir) / "collectivistic_questions.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    # Create temp story dict for survey
    temp_story = {
        'name': story['name'],
        'path': str(transformed_story_file),
        'content': transformed_story
    }

    transformed_survey = conduct_survey_single_story(
        story=temp_story,
        questions_file=str(questions_file),
        problem_type=problem_type,
        model=model,
        temperature=temperature,
        cost_tracker=cost_tracker
    )

    # Save transformed survey
    transformed_survey_file = story_output_dir / "transformed_survey.json"
    with open(transformed_survey_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_survey, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Survey completed and saved")

    # Save cost tracking
    cost_file = story_output_dir / "cost_tracking.json"
    cost_tracker.save(cost_file)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("COT BASELINE TRANSFORMATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Story: {story['name']}")
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"Output: {story_output_dir}")
    logger.info("=" * 60)

    return str(story_output_dir)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CoT Baseline Story Transformation - Chain-of-Thought Prompt Approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform individualistic story to collectivistic (inverse)
  python baseline_transform_cot.py \\
      --story data/individualistic-rags-to-riches-stories/Henry_Ford.txt \\
      --problem inverse \\
      --model gpt-4o

  # Transform collectivistic story to individualistic (forward)
  python baseline_transform_cot.py \\
      --story data/collectivistic-stories-all/Community_Time.txt \\
      --problem forward \\
      --model xai/grok-4-fast-reasoning
        """
    )

    parser.add_argument(
        '--story',
        type=str,
        required=True,
        help='Path to story file to transform'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='forward: collectivistic->individualistic, inverse: individualistic->collectivistic'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use (default: gpt-4o)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature (default: 0.7)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8096,
        help='Max output tokens for the transformation call (default: 8096)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory for questions files (default: data)'
    )

    parser.add_argument(
        '--phase2-dir',
        type=str,
        default='output/phase2',
        help='Phase 2 directory to find original surveys (default: output/phase2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/baseline_cot',
        help='Output directory for CoT baseline results (default: output/baseline_cot)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging — per-run log file so parallel runs (different models/problems)
    # don't interleave into one shared log
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = f'baseline_transform_cot_{sanitize_model_name(args.model)}_{args.problem}.log'
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Validate inputs
    if not Path(args.story).exists():
        logger.error(f"Story file not found: {args.story}")
        sys.exit(1)

    # Run CoT baseline transformation
    try:
        output_path = run_cot_transformation(
            story_path=args.story,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            data_dir=args.data_dir,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens
        )

        logger.info(f"\n✓ CoT baseline transformation successful!")
        logger.info(f"Results saved to: {output_path}")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"CoT baseline transformation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
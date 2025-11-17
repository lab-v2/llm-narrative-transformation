"""
Baseline Transformation Script

Simple baseline approach for story transformation.
Uses a single prompt: "Make the following story more individualistic/collectivistic"

This serves as a comparison baseline against the abduction-based approach.

Steps:
1. Load original story
2. Copy original survey from Phase 2 iteration_0
3. Transform story using simple prompt
4. Survey the transformed story
5. Save everything with cost tracking
"""

import json
import logging
import argparse
import sys
import shutil
from pathlib import Path
from typing import Tuple

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
# Baseline Prompt Template
# ==========================================================

def create_baseline_prompt(story_text: str, problem_type: str) -> str:
    """
    Create simple baseline transformation prompt.

    Args:
        story_text: Original story text
        problem_type: "forward" or "inverse"

    Returns:
        Prompt string
    """
    if problem_type == "forward":
        target = "individualistic"
    else:  # inverse
        target = "collectivistic"

    prompt = f"Make the following story more {target}:\n\n{story_text}"
    return prompt


# ==========================================================
# Main Baseline Function
# ==========================================================

def run_baseline_transformation(
        story_path: str,
        problem_type: str,
        model: str,
        temperature: float,
        data_dir: str,
        phase2_dir: str,
        output_dir: str
) -> str:
    """
    Run baseline transformation for a single story.

    Args:
        story_path: Path to original story file
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        data_dir: Data directory (for questions file)
        phase2_dir: Phase 2 output directory (to find original survey)
        output_dir: Baseline output directory

    Returns:
        Path to story output directory
    """
    logger.info("=" * 60)
    logger.info("BASELINE TRANSFORMATION")
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

    # Transform story using baseline prompt
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMING STORY (Baseline Prompt)")
    logger.info("=" * 60)

    prompt = create_baseline_prompt(story['content'], problem_type)
    logger.info(f"Prompt: Make story more {'individualistic' if problem_type == 'forward' else 'collectivistic'}")

    try:
        transformed_story, input_tokens, output_tokens = call_llm_with_retry(
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=4096,
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

    # Save transformed story
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
    logger.info("BASELINE TRANSFORMATION COMPLETE")
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
        description='Baseline Story Transformation - Simple Prompt Approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform individualistic story to collectivistic (inverse)
  python baseline_transform.py \\
      --story data/individualistic-rags-to-riches-stories/Henry_Ford.txt \\
      --problem inverse \\
      --model gpt-4o

  # Transform collectivistic story to individualistic (forward)
  python baseline_transform.py \\
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
        default='output/baseline',
        help='Output directory for baseline results (default: output/baseline)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('baseline_transform.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Validate inputs
    if not Path(args.story).exists():
        logger.error(f"Story file not found: {args.story}")
        sys.exit(1)

    # Run baseline transformation
    try:
        output_path = run_baseline_transformation(
            story_path=args.story,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            data_dir=args.data_dir,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir
        )

        logger.info(f"\n✓ Baseline transformation successful!")
        logger.info(f"Results saved to: {output_path}")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Baseline transformation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
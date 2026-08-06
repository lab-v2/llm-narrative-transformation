"""
Baseline Transformation Script — Few-Shot Prompting

Few-shot variant of the baseline: a fixed block of expert-written example pairs
(original story -> transformed story, for this direction) is prepended, then the
EXACT zero-shot instruction is applied to the test story. The example block is
the same for every test story in a direction — the only difference from the
zero-shot baseline is the added examples (a clean few-shot ablation).

Examples are loaded from:
    <examples-dir>/<problem>/<example*>/{original.txt, transformed.txt}
where `original` is the source-framing story and `transformed` is the expert
target-framing version.

Serves as a comparison baseline against the abduction-based approach.

Steps:
1. Load original story + the few-shot example pairs
2. Copy original survey from Phase 2 iteration_0
3. Transform story using the few-shot prompt
4. Survey the transformed story
5. Save everything with cost tracking
"""

import json
import logging
import argparse
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
# Few-Shot Examples
# ==========================================================

def load_fewshot_examples(problem_type: str, examples_dir: str) -> list:
    """
    Load the expert example pairs for a direction.

    Reads <examples_dir>/<problem>/<example*>/{original.txt, transformed.txt},
    sorted by folder name so ordering is deterministic.

    Args:
        problem_type: "forward" or "inverse"
        examples_dir: Root folder of few-shot examples

    Returns:
        List of dicts: {name, original, transformed}
    """
    direction_dir = Path(examples_dir) / problem_type
    if not direction_dir.exists():
        logger.error(f"Few-shot examples directory not found: {direction_dir}")
        sys.exit(1)

    examples = []
    for ex_dir in sorted(d for d in direction_dir.iterdir() if d.is_dir()):
        original_path = ex_dir / "original.txt"
        transformed_path = ex_dir / "transformed.txt"

        if not original_path.exists() or not transformed_path.exists():
            logger.warning(f"Skipping {ex_dir.name}: missing original.txt or transformed.txt")
            continue

        examples.append({
            "name": ex_dir.name,
            "original": original_path.read_text(encoding="utf-8").strip(),
            "transformed": transformed_path.read_text(encoding="utf-8").strip(),
        })

    if not examples:
        logger.error(f"No usable example pairs found in {direction_dir}")
        sys.exit(1)

    return examples


def create_fewshot_prompt(story_text: str, problem_type: str, examples: list) -> str:
    """
    Build the few-shot transformation prompt.

    The final instruction line is byte-for-byte the zero-shot baseline prompt,
    so the examples are the only thing added.

    Args:
        story_text: Original test story text
        problem_type: "forward" or "inverse"
        examples: List of {original, transformed} example pairs

    Returns:
        Prompt string
    """
    if problem_type == "forward":
        target, source = "individualistic", "collectivistic"
    else:  # inverse
        target, source = "collectivistic", "individualistic"

    source_cap = source.capitalize()
    target_cap = target.capitalize()

    parts = [f"Here are examples of rewriting a story to be more {target}.\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"Example {i}:\n"
            f"{source_cap} story:\n{ex['original']}\n\n"
            f"{target_cap} version:\n{ex['transformed']}\n"
        )

    # Final instruction is IDENTICAL to the zero-shot baseline prompt.
    parts.append(f"Make the following story more {target}:\n\n{story_text}")

    return "\n".join(parts)


# ==========================================================
# Main Few-Shot Function
# ==========================================================

def run_fewshot_transformation(
        story_path: str,
        problem_type: str,
        model: str,
        temperature: float,
        data_dir: str,
        phase2_dir: str,
        output_dir: str,
        examples_dir: str,
        max_tokens: int = 8096
) -> str:
    """
    Run few-shot baseline transformation for a single story.

    Args:
        story_path: Path to original story file
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        data_dir: Data directory (for questions file)
        phase2_dir: Phase 2 output directory (to find original survey)
        output_dir: Few-shot baseline output directory
        examples_dir: Root folder of few-shot examples
        max_tokens: Max output tokens for the transformation call

    Returns:
        Path to story output directory
    """
    logger.info("=" * 60)
    logger.info("FEW-SHOT BASELINE TRANSFORMATION")
    logger.info("=" * 60)

    # Load original story
    logger.info("Loading original story...")
    story = load_single_story(story_path)
    logger.info(f"✓ Loaded: {story['name']}")

    # Load few-shot examples for this direction
    examples = load_fewshot_examples(problem_type, examples_dir)
    logger.info(f"✓ Loaded {len(examples)} few-shot examples: {[e['name'] for e in examples]}")

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

    # Transform story using few-shot prompt
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMING STORY (Few-Shot Prompt)")
    logger.info("=" * 60)

    prompt = create_fewshot_prompt(story['content'], problem_type, examples)
    target = 'individualistic' if problem_type == 'forward' else 'collectivistic'
    logger.info(f"Prompt: {len(examples)}-shot -> make story more {target}")

    try:
        transformed_story, input_tokens, output_tokens = call_llm_with_retry(
            system_prompt="You are a helpful assistant.",
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
    logger.info("FEW-SHOT BASELINE TRANSFORMATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Story: {story['name']}")
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Examples: {len(examples)}")
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
        description='Few-Shot Baseline Story Transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform individualistic story to collectivistic (inverse)
  python baseline_transform_fewshot.py \\
      --story data/individualistic-rags-to-riches-stories/Henry_Ford.txt \\
      --problem inverse \\
      --model gpt-4o

  # Transform collectivistic story to individualistic (forward)
  python baseline_transform_fewshot.py \\
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
        '--examples-dir',
        type=str,
        default='data/fewshot_examples',
        help='Root folder of few-shot example pairs (default: data/fewshot_examples)'
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
        default='output/baseline_fewshot',
        help='Output directory for few-shot baseline results (default: output/baseline_fewshot)'
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
    log_file = f'baseline_transform_fewshot_{sanitize_model_name(args.model)}_{args.problem}.log'
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

    # Run few-shot baseline transformation
    try:
        output_path = run_fewshot_transformation(
            story_path=args.story,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            data_dir=args.data_dir,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir,
            examples_dir=args.examples_dir,
            max_tokens=args.max_tokens
        )

        logger.info(f"\n✓ Few-shot baseline transformation successful!")
        logger.info(f"Results saved to: {output_path}")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Few-shot baseline transformation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
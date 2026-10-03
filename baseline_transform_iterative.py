"""
Baseline Transformation Script — Iterative Refinement (LLM Self-Critique)

Self-refine baseline, 3 independent (stateless) LLM calls:
  1. Zero-shot transform:  original -> transformed ("make it more {target}")
  2. Critique:             (original + transformed) -> free-form critique
  3. Refine:               (transformed + critique) -> final story
                           (NOTE: the original is deliberately NOT shown in the
                            refine call — it is the most {source} version and
                            would anchor the model back toward the source framing)

Single final output, same shape as the other baselines. The 20-feature survey
is run only on the FINAL story (for evaluation) — it is never shown to the LLM.

Serves as a comparison baseline against the abduction-based approach.

Steps:
1. Load original story
2. Copy original survey from Phase 2 iteration_0
3. Zero-shot transform -> critique -> refine
4. Survey the refined story
5. Save everything with cost tracking
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
# Prompts
# ==========================================================

def create_zeroshot_prompt(story_text: str, problem_type: str) -> str:
    """Call 1: the plain zero-shot transformation prompt."""
    target = "individualistic" if problem_type == "forward" else "collectivistic"
    return f"Make the following story more {target}:\n\n{story_text}"


def create_critique_prompt(original_text: str, transformed_text: str, problem_type: str) -> str:
    """Call 2: critique the transformed story, grounded in the original."""
    if problem_type == "forward":
        target, source = "individualistic", "collectivistic"
    else:  # inverse
        target, source = "collectivistic", "individualistic"

    return (
        f"Here is an original story and a version rewritten to be more {target}.\n\n"
        f"Original story:\n{original_text}\n\n"
        f"Rewritten (more {target}) version:\n{transformed_text}\n\n"
        f"Critique the rewritten version: which parts still read as {source}, and "
        f"what specifically should change to make it more {target}? "
        f"Do not rewrite the story — only give the critique."
    )


def create_refine_prompt(transformed_text: str, critique: str, problem_type: str) -> str:
    """Call 3: refine using the critique. Original is intentionally omitted."""
    target = "individualistic" if problem_type == "forward" else "collectivistic"
    return (
        f"Below is a story that was rewritten to be more {target}, along with a critique. "
        f"Rewrite it again to be more {target}, addressing the critique.\n\n"
        f"Story:\n{transformed_text}\n\n"
        f"Critique:\n{critique}\n\n"
        f"Output ONLY the rewritten story wrapped exactly in these tags, with no "
        f"preamble and no explanation of your changes:\n"
        f"<rewritten_story>\n...complete rewritten story...\n</rewritten_story>"
    )


def extract_rewritten_story(raw_response: str) -> str:
    """
    Extract the story from between <rewritten_story>...</rewritten_story> tags.

    Falls back to the full raw response (with a warning) if the tags are absent,
    so a malformed response never silently drops the transformation.
    """
    match = re.search(
        r"<rewritten_story>\s*(.*?)\s*</rewritten_story>",
        raw_response,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    logger.warning(
        "No <rewritten_story> tags found in refine response — falling back to the "
        "full raw response. Check refine_raw.txt for this story."
    )
    return raw_response.strip()


# ==========================================================
# Main Iterative-Refinement Function
# ==========================================================

def run_iterative_transformation(
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
    Run iterative-refinement (self-critique) baseline for a single story.

    Args:
        story_path: Path to original story file
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        data_dir: Data directory (for questions file)
        phase2_dir: Phase 2 output directory (to find original survey)
        output_dir: Iterative baseline output directory
        max_tokens: Max output tokens for the transformation calls

    Returns:
        Path to story output directory
    """
    logger.info("=" * 60)
    logger.info("ITERATIVE-REFINEMENT BASELINE (SELF-CRITIQUE)")
    logger.info("=" * 60)

    # Load original story
    logger.info("Loading original story...")
    story = load_single_story(story_path)
    logger.info(f"✓ Loaded: {story['name']}")

    sanitized_model = sanitize_model_name(model)

    # Create output directory
    story_output_dir = Path(output_dir) / sanitized_model / problem_type / story['name']
    story_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {story_output_dir}")

    # Save original story
    (story_output_dir / "original_story.txt").write_text(story['content'], encoding='utf-8')
    logger.info("✓ Saved original story")

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
    shutil.copy2(phase2_survey_path, story_output_dir / "original_survey.json")
    logger.info(f"✓ Copied from: {phase2_survey_path}")

    # Initialize cost tracker
    cost_tracker = CostTracker(model)
    target = "individualistic" if problem_type == "forward" else "collectivistic"

    # --- Call 1: Zero-shot transform ---
    logger.info("\n" + "=" * 60)
    logger.info("CALL 1: ZERO-SHOT TRANSFORM")
    logger.info("=" * 60)
    try:
        zeroshot_story, in_tok, out_tok = call_llm_with_retry(
            system_prompt="You are a helpful assistant.",
            user_prompt=create_zeroshot_prompt(story['content'], problem_type),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3
        )
        cost_tracker.add_call(story_name=f"{story['name']}_zeroshot",
                              input_tokens=in_tok, output_tokens=out_tok)
        logger.info(f"✓ Zero-shot transform done ({in_tok + out_tok:,} tokens)")
    except Exception as e:
        logger.error(f"Zero-shot transform failed: {e}")
        sys.exit(1)

    (story_output_dir / "zeroshot_story.txt").write_text(zeroshot_story, encoding='utf-8')

    # --- Call 2: Critique (original + transformed) ---
    logger.info("\n" + "=" * 60)
    logger.info("CALL 2: CRITIQUE")
    logger.info("=" * 60)
    try:
        critique, in_tok, out_tok = call_llm_with_retry(
            system_prompt="You are a helpful assistant.",
            user_prompt=create_critique_prompt(story['content'], zeroshot_story, problem_type),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3
        )
        cost_tracker.add_call(story_name=f"{story['name']}_critique",
                              input_tokens=in_tok, output_tokens=out_tok)
        logger.info(f"✓ Critique generated ({in_tok + out_tok:,} tokens)")
    except Exception as e:
        logger.error(f"Critique failed: {e}")
        sys.exit(1)

    (story_output_dir / "critique.txt").write_text(critique, encoding='utf-8')

    # --- Call 3: Refine (transformed + critique, NO original) ---
    logger.info("\n" + "=" * 60)
    logger.info("CALL 3: REFINE")
    logger.info("=" * 60)
    try:
        raw_refine, in_tok, out_tok = call_llm_with_retry(
            system_prompt="You are a helpful assistant.",
            user_prompt=create_refine_prompt(zeroshot_story, critique, problem_type),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=3
        )
        cost_tracker.add_call(story_name=f"{story['name']}_refine",
                              input_tokens=in_tok, output_tokens=out_tok)
        logger.info(f"✓ Story refined ({in_tok + out_tok:,} tokens)")
        logger.info(f"  Cost so far: ${cost_tracker.total_cost:.4f}")
    except Exception as e:
        logger.error(f"Refine failed: {e}")
        sys.exit(1)

    # Save the raw refine response (may include preamble/commentary) for transparency
    (story_output_dir / "refine_raw.txt").write_text(raw_refine, encoding='utf-8')

    # Extract only the story from the tags (strips model-specific preamble/postamble)
    transformed_story = extract_rewritten_story(raw_refine)

    # Save refined story (this is what the evaluation pipeline reads)
    transformed_story_file = story_output_dir / "transformed_story.txt"
    transformed_story_file.write_text(transformed_story, encoding='utf-8')
    logger.info(f"✓ Saved refined story")

    # --- Survey the refined story (evaluation only) ---
    logger.info("\n" + "=" * 60)
    logger.info("SURVEYING REFINED STORY")
    logger.info("=" * 60)

    if problem_type == "forward":
        questions_file = Path(data_dir) / "individualistic_questions.json"
    else:  # inverse
        questions_file = Path(data_dir) / "collectivistic_questions.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

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

    with open(story_output_dir / "transformed_survey.json", 'w', encoding='utf-8') as f:
        json.dump(transformed_survey, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Survey completed and saved")

    # Save cost tracking
    cost_tracker.save(story_output_dir / "cost_tracking.json")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ITERATIVE-REFINEMENT BASELINE COMPLETE")
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
        description='Iterative-Refinement (Self-Critique) Baseline Story Transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot -> critique -> refine (inverse)
  python baseline_transform_iterative.py \\
      --story data/individualistic-rags-to-riches-stories/Henry_Ford.txt \\
      --problem inverse --model gpt-4o
        """
    )

    parser.add_argument('--story', type=str, required=True, help='Path to story file to transform')
    parser.add_argument('--problem', type=str, choices=['forward', 'inverse'], required=True,
                        help='forward: collectivistic->individualistic, inverse: individualistic->collectivistic')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM model to use (default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=8096,
                        help='Max output tokens for the transformation calls (default: 8096)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory for questions files (default: data)')
    parser.add_argument('--phase2-dir', type=str, default='output/phase2',
                        help='Phase 2 directory to find original surveys (default: output/phase2)')
    parser.add_argument('--output-dir', type=str, default='output/baseline_iterative',
                        help='Output directory for iterative baseline results (default: output/baseline_iterative)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging — per-run log file so parallel runs don't interleave
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = f'baseline_transform_iterative_{sanitize_model_name(args.model)}_{args.problem}.log'
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    if not Path(args.story).exists():
        logger.error(f"Story file not found: {args.story}")
        sys.exit(1)

    try:
        output_path = run_iterative_transformation(
            story_path=args.story,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            data_dir=args.data_dir,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens
        )
        logger.info(f"\n✓ Iterative-refinement baseline successful!")
        logger.info(f"Results saved to: {output_path}")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Iterative-refinement baseline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

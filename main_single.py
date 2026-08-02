"""
CONNECT Project - Main Script (Single-Call Transformer Variant)

Same as main.py but uses story_transformer_single.py, which transforms
ALL segments in a single LLM call instead of iterating segment-by-segment.

Author: JM Patil
Date: November 11, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from dotenv import load_dotenv
import os

load_dotenv('.env')

from story_loader import load_training_stories, load_single_story
from llm_survey import conduct_surveys, conduct_survey_single_story, sanitize_model_name, CostTracker
from rule_learner import learn_rules
from graph_builder import build_ground_atoms_from_survey, save_ground_atoms, save_segments_metadata
from pyreason_runner import run_pyreason
from abduction import run_abduction
from story_transformer_single import transform_story_single   # ← single-call version

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('connect_single.log'),   # separate log file
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CONNECT project (single-call transformer)."""

    parser = argparse.ArgumentParser(
        description='CONNECT Project - Narrative Transformation Pipeline (Single-Call Variant)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Train on individualistic stories (forward problem)
  python main_single.py --phase 1 --problem forward --data-dir data --output-dir output/phase1

  # Phase 1: Train on collectivistic stories (inverse problem)
  python main_single.py --phase 1 --problem inverse --data-dir data --output-dir output/phase1

  # Phase 2: Transform collectivistic story to individualistic (single call)
  python main_single.py --phase 2 --problem forward --rules output/phase1/rules.txt --story data/test_story.txt

  # Phase 2: Transform individualistic story to collectivistic (single call)
  python main_single.py --phase 2 --problem inverse --rules output/phase1/rules.txt --story data/test_story.txt
        """
    )

    # Core arguments
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                        help='Phase 1 (training) or Phase 2 (testing/transformation)')
    parser.add_argument('--problem', type=str, choices=['forward', 'inverse'], required=True,
                        help='forward: collectivistic→individualistic, inverse: individualistic→collectivistic')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory (default: data)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Path to output directory (default: output)')

    # Phase 1 specific
    parser.add_argument('--skip-survey', action='store_true',
                        help='Skip LLM survey step and use existing results')
    parser.add_argument('--survey-results', type=str,
                        help='[Phase 1] Path to existing survey results JSON file')

    # Phase 2 specific
    parser.add_argument('--story', type=str,
                        help='[Phase 2] Path to test story file to transform')
    parser.add_argument('--max-iterations', type=int, default=3,
                        help='[Phase 2] Maximum transformation iterations (default: 3)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='[Phase 2] Number of top features to transform per iteration (default: 3)')
    parser.add_argument('--rules', type=str, default=None,
                        help='[Phase 2] Path to learned rules file (optional - auto-determined if not provided)')

    # LLM configuration
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='LLM model to use (default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='LLM temperature (default: 0.7)')

    # General options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    if not validate_arguments(args):
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"CONNECT Project (Single-Call) - Phase {args.phase}")
    logger.info(f"Problem: {args.problem}")
    logger.info("=" * 60)

    try:
        if args.phase == 1:
            run_phase1(args)
        else:
            run_phase2(args)

        logger.info("=" * 60)
        logger.info("Execution completed successfully!")
        logger.info("=" * 60)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        logger.error("Check connect_single.log for details")
        sys.exit(1)


def validate_arguments(args):
    if args.phase == 1:
        if not args.skip_survey:
            if not Path(args.data_dir).exists():
                logger.error(f"Data directory not found: {args.data_dir}")
                return False

    if args.phase == 2:
        if not args.story:
            logger.error("--story is required for Phase 2")
            return False
        if not Path(args.story).exists():
            logger.error(f"Story file not found: {args.story}")
            return False

    return True


def run_phase1(args):
    """Phase 1 is identical to main.py - learning rules doesn't change."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 60)

    if not args.skip_survey:
        stories = load_training_stories(args.data_dir, args.problem)
        logger.info(f"Loaded {len(stories)} training stories")

        if args.problem == "forward":
            questions_file = Path(args.data_dir) / "individualistic_questions.json"
        else:
            questions_file = Path(args.data_dir) / "collectivistic_questions.json"

        if not questions_file.exists():
            logger.error(f"Questions file not found: {questions_file}")
            sys.exit(1)

        survey_results, failed_stories = conduct_surveys(
            stories=stories,
            questions_file=str(questions_file),
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            output_dir=args.output_dir
        )

        if failed_stories:
            logger.warning(f"{len(failed_stories)} stories failed. Check failed_stories.json")
    else:
        logger.info("Skipping survey step (--skip-survey flag set)")

    sanitized_model = sanitize_model_name(args.model)
    survey_results_dir = Path(args.output_dir) / sanitized_model / args.problem / "survey_results"

    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Learning PyReason Rules")
    logger.info("=" * 60)

    rules_file = learn_rules(
        survey_results_dir=str(survey_results_dir),
        problem_type=args.problem,
        model=args.model,
        output_dir=args.output_dir
    )

    logger.info(f"\n✓ Rules learned and saved to: {rules_file}")
    logger.info("\nPhase 1 completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")


def run_phase2(args):
    """
    Phase 2: Same pipeline as main.py but calls transform_story_single()
    instead of transform_story_iteratively(). Output paths are identical.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: TESTING/TRANSFORMATION (SINGLE-CALL MODE)")
    logger.info("=" * 60)

    # Step 1: Load test story
    logger.info("\n[Step 1/7] Loading test story...")
    story = load_single_story(args.story)
    logger.info(f"✓ Loaded test story: {story['name']}")

    # Step 2: Load learned rules
    logger.info("\n[Step 2/7] Loading learned rules...")

    if args.rules:
        rules_file = Path(args.rules)
    else:
        sanitized_model = sanitize_model_name(args.model)
        rules_file = Path("output/phase1") / sanitized_model / args.problem / "learned_rules" / "pyreason_rules.txt"
        logger.info(f"Auto-determined rules path: {rules_file}")

    if not rules_file.exists():
        logger.error(f"Rules file not found: {rules_file}")
        logger.error("Please ensure Phase 1 has been run for this model and problem type,")
        logger.error("or provide explicit --rules path")
        sys.exit(1)

    logger.info(f"✓ Rules loaded from: {rules_file}")

    # Create output directory (same structure as main.py)
    sanitized_model = sanitize_model_name(args.model)
    story_output_dir = Path(args.output_dir) / "phase2" / sanitized_model / args.problem / story['name']
    story_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {story_output_dir}")

    cost_tracker = CostTracker(args.model)
    logger.info("✓ Cost tracker initialized")

    # Questions file
    if args.problem == "forward":
        questions_file = Path(args.data_dir) / "individualistic_questions.json"
    else:
        questions_file = Path(args.data_dir) / "collectivistic_questions.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    current_story_text = story['content']
    current_story_path = story['path']

    # Main iteration loop
    for iteration in range(args.max_iterations):
        logger.info("\n" + "=" * 60)
        logger.info(f"ITERATION {iteration + 1}/{args.max_iterations}")
        logger.info("=" * 60)

        # Create iteration directory (same as main.py)
        iter_dir = story_output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Save current story
        story_file = iter_dir / "story.txt"
        story_file.write_text(current_story_text, encoding='utf-8')
        logger.info(f"✓ Saved current story to: {story_file}")

        # Step 3: Survey
        survey_file = iter_dir / "survey.json"

        if args.skip_survey and survey_file.exists():
            logger.info(f"\n[Step 3/7] Loading existing survey (--skip-survey)...")
            with open(survey_file, 'r', encoding='utf-8') as f:
                survey_result = json.load(f)
            logger.info(f"✓ Survey loaded from: {survey_file}")
        else:
            logger.info(f"\n[Step 3/7] Surveying story (iteration {iteration})...")

            temp_story = {
                'name': story['name'],
                'path': str(story_file),
                'content': current_story_text
            }

            survey_result = conduct_survey_single_story(
                story=temp_story,
                questions_file=str(questions_file),
                problem_type=args.problem,
                model=args.model,
                temperature=args.temperature,
                cost_tracker=cost_tracker
            )

            with open(survey_file, 'w', encoding='utf-8') as f:
                json.dump(survey_result, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Survey completed and saved to: {survey_file}")

        # Save cost tracking
        iter_cost_file = iter_dir / "cost_tracking.json"
        cost_tracker.save(iter_cost_file)
        logger.info(f"✓ Cost tracking saved to: {iter_cost_file}")

        # Step 4: Build ground atoms
        logger.info(f"\n[Step 4/7] Building ground atoms from survey...")

        ground_atoms, segments_metadata = build_ground_atoms_from_survey(
            survey_json_path=str(survey_file),
            story_name=story['name']
        )

        ground_atoms_file = iter_dir / "ground_atoms.json"
        save_ground_atoms(ground_atoms, str(ground_atoms_file))
        logger.info(f"✓ Ground atoms saved: {len(ground_atoms)} atoms")

        segments_metadata_file = iter_dir / "segments_metadata.json"
        save_segments_metadata(segments_metadata, str(segments_metadata_file))
        logger.info(f"✓ Segments metadata saved: {len(segments_metadata)} features")

        # Step 5: Run PyReason
        logger.info(f"\n[Step 5/7] Running PyReason...")

        trace_dir = run_pyreason(
            ground_atoms=ground_atoms,
            rules_file=str(rules_file),
            story_name=story['name'],
            output_dir=str(iter_dir)
        )

        logger.info(f"✓ PyReason traces saved to: {trace_dir}")

        # Step 6: Run abduction
        logger.info(f"\n[Step 6/7] Running abduction analysis...")

        trace_files = sorted(Path(trace_dir).glob("rule_trace_edges_*.csv"))
        if not trace_files:
            logger.error("No trace CSV files found from PyReason")
            sys.exit(1)

        trace_csv_path = str(trace_files[0])

        abduction_file, prescriptions_file, stop_condition_met = run_abduction(
            trace_csv_path=trace_csv_path,
            rules_file=str(rules_file),
            ground_atoms=ground_atoms,
            segments_metadata=segments_metadata,
            output_dir=str(iter_dir),
            gap_threshold=0.01
        )

        logger.info(f"✓ Abduction analysis saved to: {abduction_file}")
        logger.info(f"✓ Ranked prescriptions saved to: {prescriptions_file}")

        if stop_condition_met:
            logger.info("\n🛑 Stop condition met - all gaps < 0.01")
            logger.info("No further transformation needed!")
            break

        # Step 7: Transform story — SINGLE CALL (key difference from main.py)
        logger.info(f"\n[Step 7/7] Transforming story (single call, top-{args.top_k} features)...")

        transformed_story, log_file, cost_file = transform_story_single(
            story_text=current_story_text,
            prescriptions_file=prescriptions_file,
            top_k=args.top_k,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            output_dir=str(iter_dir)          # ← same output dir structure
        )

        logger.info(f"✓ Story transformed")
        logger.info(f"✓ Transformation log: {log_file}")
        logger.info(f"✓ Transformation costs: {cost_file}")

        current_story_text = transformed_story

    # Final cost summary (same as main.py)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 FINAL COST SUMMARY")
    logger.info("=" * 60)

    final_cost_file = story_output_dir / "cost_tracking_final.json"
    cost_tracker.save(final_cost_file)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total stories processed: 1")
    logger.info(f"Total iterations: {iteration + 1}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"  - Input: {cost_tracker.total_input_tokens:,}")
    logger.info(f"  - Output: {cost_tracker.total_output_tokens:,}")
    logger.info(f"Cost tracking saved to: {final_cost_file}")
    logger.info("=" * 60)

    logger.info("\nPhase 2 completed!")
    logger.info(f"Output directory: {story_output_dir}")


if __name__ == "__main__":
    main()

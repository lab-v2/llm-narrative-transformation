"""
CONNECT Project - Main Script

This is the main entry point for the CONNECT project.
Handles both Phase 1 (training) and Phase 2 (testing/transformation).

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

# Load environment variables from .env file
load_dotenv('.env')
print(f"DEBUG: API key loaded: {os.getenv('OPENAI_API_KEY')[:10]}...")  # Print first 10 chars


from story_loader import load_training_stories, load_single_story
from llm_survey import conduct_surveys, conduct_survey_single_story

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('connect.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CONNECT project."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='CONNECT Project - Narrative Transformation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Train on individualistic stories (forward problem)
  python main.py --phase 1 --problem forward --data-dir data --output-dir output/phase1

  # Phase 1: Train on collectivistic stories (inverse problem)
  python main.py --phase 1 --problem inverse --data-dir data --output-dir output/phase1

  # Phase 2: Transform collectivistic story to individualistic
  python main.py --phase 2 --problem forward --rules output/phase1/rules.txt --story data/test_story.txt

  # Phase 2: Transform individualistic story to collectivistic
  python main.py --phase 2 --problem inverse --rules output/phase1/rules.txt --story data/test_story.txt
        """
    )

    # Core arguments
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        required=True,
        help='Phase 1 (training) or Phase 2 (testing/transformation)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='forward: individualistic->collectivistic, inverse: collectivistic->individualistic'
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Path to output directory (default: output)'
    )

    # Phase 1 specific arguments
    parser.add_argument(
        '--skip-survey',
        action='store_true',
        help='[Phase 1] Skip LLM survey step and load existing results'
    )

    parser.add_argument(
        '--survey-results',
        type=str,
        help='[Phase 1] Path to existing survey results JSON file'
    )

    # Phase 2 specific arguments
    parser.add_argument(
        '--rules',
        type=str,
        help='[Phase 2] Path to learned rules file from Phase 1'
    )

    parser.add_argument(
        '--story',
        type=str,
        help='[Phase 2] Path to test story file to transform'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='[Phase 2] Maximum transformation iterations (default: 5)'
    )

    # LLM configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use (default: gpt-4o)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature (default: 0.0)'
    )

    # General options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Log configuration
    logger.info("=" * 60)
    logger.info(f"CONNECT Project - Phase {args.phase}")
    logger.info(f"Problem: {args.problem}")
    logger.info("=" * 60)

    # Execute appropriate phase
    try:
        if args.phase == 1:
            run_phase1(args)
        else:  # args.phase == 2
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
        logger.error("Check connect.log for details")
        sys.exit(1)


def validate_arguments(args):
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        bool: True if valid, False otherwise
    """
    # Phase 1 validation
    if args.phase == 1:
        if args.skip_survey and not args.survey_results:
            logger.error("--survey-results is required when using --skip-survey")
            return False

        if not Path(args.data_dir).exists():
            logger.error(f"Data directory not found: {args.data_dir}")
            return False

    # Phase 2 validation
    if args.phase == 2:
        if not args.rules:
            logger.error("--rules is required for Phase 2")
            return False

        if not args.story:
            logger.error("--story is required for Phase 2")
            return False

        if not Path(args.rules).exists():
            logger.error(f"Rules file not found: {args.rules}")
            return False

        if not Path(args.story).exists():
            logger.error(f"Story file not found: {args.story}")
            return False

    return True


def run_phase1(args):
    """
    Execute Phase 1: Training - Learn rules from training stories.

    This phase:
    1. Loads training stories (individualistic or collectivistic based on problem)
    2. Conducts LLM surveys on each story (20 features)
    3. Builds knowledge graph from survey results
    4. Learns PyReason rules from knowledge graph

    Args:
        args: Command line arguments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 60)

    # TODO: Import and call story_loader.py
    stories = load_training_stories(args.data_dir, args.problem)
    logger.info(f"Loaded {len(stories)} training stories")

    # TODO: Import and call llm_survey.py (unless skipping)
    # Determine questions file path based on problem type
    if args.problem == "forward":
        questions_file = Path(args.data_dir) / "individualistic_questions.json"
    else:  # inverse
        questions_file = Path(args.data_dir) / "collectivistic_questions.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    # Conduct surveys (or load existing)
    if not args.skip_survey:
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
        # Load existing survey results
        logger.info(f"Loading existing survey results from {args.survey_results}")
        with open(args.survey_results, 'r') as f:
            survey_results = json.load(f)

    # TODO: Import and call graph_builder.py
    # from graph_builder import build_knowledge_graph
    # knowledge_graph = build_knowledge_graph(survey_results, args.problem)
    # # Save knowledge graph
    # save_knowledge_graph(knowledge_graph, args.output_dir)

    # TODO: Import and call rule_learner.py
    # from rule_learner import learn_rules
    # rules = learn_rules(knowledge_graph)
    # # Save learned rules
    # save_rules(rules, args.output_dir)

    logger.info("\nPhase 1 completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")


def run_phase2(args):
    """
    Execute Phase 2: Testing/Transformation - Transform test story.

    This phase:
    1. Loads learned rules from Phase 1
    2. Conducts LLM survey on test story
    3. Runs abduction algorithm to prioritize features to change
    4. Iteratively transforms story using LLM prompts
    5. Selects best iteration based on average rating

    Args:
        args: Command line arguments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: TESTING/TRANSFORMATION")
    logger.info("=" * 60)

    # TODO: Import and load learned rules
    # from rule_learner import load_rules
    # rules = load_rules(args.rules)
    # logger.info(f"Loaded {len(rules)} rules")

    # TODO: Import and load test story
    # story = load_single_story(args.story)
    # logger.info(f"Loaded test story: {story['name']}")

    # TODO: Import and conduct survey on test story
    # # Determine questions file path
    # if args.problem == "forward":
    #     questions_file = Path(args.data_dir) / "collectivistic_questions.json"  # Note: opposite for test
    # else:
    #     questions_file = Path(args.data_dir) / "individualistic_questions.json"
    #
    # # Conduct survey on test story
    # initial_survey = conduct_survey_single_story(
    #     story=story,
    #     questions_file=str(questions_file),
    #     problem_type=args.problem,
    #     model=args.model,
    #     temperature=args.temperature
    # )
    # logger.info(f"Initial survey completed")

    # TODO: Import and run abduction algorithm
    # from abduction import run_abduction
    # feature_priorities = run_abduction(initial_survey, rules)
    # logger.info(f"Identified {len(feature_priorities)} features to modify")

    # TODO: Import and run transformation iterations
    # from story_transformer import transform_story_iteratively
    # results = transform_story_iteratively(
    #     story=story,
    #     feature_priorities=feature_priorities,
    #     rules=rules,
    #     max_iterations=args.max_iterations,
    #     model=args.model,
    #     temperature=args.temperature,
    #     target_problem=args.problem
    # )

    # TODO: Select best iteration
    # best_iteration = select_best_iteration(results)
    # logger.info(f"Best iteration: {best_iteration['iteration_num']}")
    # logger.info(f"Average rating: {best_iteration['avg_rating']:.2f}")

    # TODO: Save transformed story
    # save_transformed_story(best_iteration['story'], args.output_dir)

    logger.info("\nPhase 2 completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
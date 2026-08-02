"""
Survey Stability Experiment Script

Runs GPT-4o survey N times per story to measure consistency/stability.
Tests stories S_1 to S_5 (forward problem) and S_29 to S_33 (inverse problem).

Similar structure to main.py Phase 1.

Output: CSV with columns story_id, run_1, run_2, ..., run_N
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('.env')

from story_loader import load_single_story
from llm_survey import conduct_survey_single_story, CostTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('survey_stability.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================================
# Story Mapping & Loading
# ==========================================================

def load_story_mapping(mapping_file: str) -> pd.DataFrame:
    """Load story name mapping CSV."""
    df = pd.read_csv(mapping_file)
    logger.info(f"Loaded mapping with {len(df)} stories")
    return df


def get_story_info(
        story_id: str,
        mapping_df: pd.DataFrame,
        data_dir: str
) -> Dict:
    """
    Get story path and problem type from mapping.

    Returns:
        Dict with 'path', 'problem_type', 'original_name'
    """
    row = mapping_df[mapping_df['abbreviated_name'] == story_id]
    if len(row) == 0:
        raise ValueError(f"Story {story_id} not found in mapping")

    problem_type = row.iloc[0]['problem_type']
    original_name = row.iloc[0]['original_name']

    if problem_type == "forward":
        story_path = Path(data_dir) / "collectivistic-stories-all" / f"{original_name}.txt"
    else:  # inverse
        story_path = Path(data_dir) / "individualistic-rags-to-riches-stories-subset" / f"{original_name}.txt"

    if not story_path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    return {
        'path': str(story_path),
        'problem_type': problem_type,
        'original_name': original_name
    }


# ==========================================================
# Survey Execution
# ==========================================================

def extract_median_from_survey(survey_result: Dict) -> float:
    """
    Extract median rating from survey result.

    survey_result has structure:
    {
        "story_name": str,
        "questions_and_answers": [
            {
                "rating": float,
                ...
            },
            ...
        ]
    }
    """
    if 'questions_and_answers' not in survey_result:
        logger.warning("No 'questions_and_answers' in survey result")
        return None

    qa_list = survey_result['questions_and_answers']
    if not qa_list:
        logger.warning("Empty questions_and_answers list")
        return None

    # Extract all ratings
    ratings = []
    for qa in qa_list:
        if 'rating' in qa:
            ratings.append(qa['rating'])

    if not ratings:
        logger.warning("No ratings found in survey result")
        return None

    # Compute median
    ratings.sort()
    n = len(ratings)
    if n % 2 == 0:
        median = (ratings[n // 2 - 1] + ratings[n // 2]) / 2
    else:
        median = ratings[n // 2]

    return float(median)


def run_survey_for_story(
        story_id: str,
        story_info: Dict,
        num_runs: int,
        model: str,
        temperature: float,
        data_dir: str,
        questions_file: str,
        cost_tracker: CostTracker
) -> List[float]:
    """
    Run survey N times for a single story.

    Args:
        story_id: Story ID (e.g., "S_1")
        story_info: Dict with 'path', 'problem_type', 'original_name'
        num_runs: Number of times to run survey
        model: Model name
        temperature: Temperature for LLM
        data_dir: Data directory
        questions_file: Questions file name
        cost_tracker: Cost tracker to update

    Returns:
        List of median ratings from each run
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running survey for {story_id} ({num_runs} runs)...")
    logger.info(f"{'='*60}")

    # Load story
    story_dict = {
        'name': story_info['original_name'],
        'path': story_info['path'],
        'content': Path(story_info['path']).read_text(encoding='utf-8')
    }
    logger.info(f"  ✓ Loaded story ({len(story_dict['content'])} characters)")

    # Full path to questions file
    questions_path = Path(data_dir) / questions_file
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    median_ratings = []
    problem_type = story_info['problem_type']

    for run_num in range(1, num_runs + 1):
        try:
            logger.info(f"  Run {run_num}/{num_runs}...")

            # Conduct survey
            survey_result = conduct_survey_single_story(
                story=story_dict,
                questions_file=str(questions_path),
                problem_type=problem_type,
                model=model,
                temperature=temperature,
                cost_tracker=cost_tracker
            )

            # Extract median rating
            median = extract_median_from_survey(survey_result)
            if median is not None:
                median_ratings.append(median)
                logger.info(f"    ✓ Median: {median:.2f}")
            else:
                logger.warning(f"    ✗ Failed to extract median")
                median_ratings.append(None)

        except Exception as e:
            logger.error(f"    ✗ Error on run {run_num}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            median_ratings.append(None)

    return median_ratings


# ==========================================================
# Main Experiment
# ==========================================================

def run_stability_experiment(
        story_ids: List[str],
        num_runs: int,
        model: str,
        temperature: float,
        data_dir: str,
        mapping_file: str,
        output_file: str,
        verbose: bool = False
):
    """Run survey stability experiment."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("SURVEY STABILITY EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Runs per story: {num_runs}")
    logger.info(f"Stories: {story_ids}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Output: {output_file}")

    # Load mapping
    mapping_df = load_story_mapping(mapping_file)

    # Initialize cost tracker
    cost_tracker = CostTracker(model)

    # Results storage
    results = []

    # Run survey for each story
    for story_id in story_ids:
        try:
            # Get story info from mapping
            story_info = get_story_info(story_id, mapping_df, data_dir)
            logger.info(f"\nProcessing {story_id} ({story_info['original_name']})")
            logger.info(f"  Problem type: {story_info['problem_type']}")

            # Determine questions file based on problem type
            if story_info['problem_type'] == "forward":
                questions_file = "individualistic_questions.json"
            else:  # inverse
                questions_file = "collectivistic_questions.json"

            logger.info(f"  Questions file: {questions_file}")

            # Run surveys
            median_ratings = run_survey_for_story(
                story_id=story_id,
                story_info=story_info,
                num_runs=num_runs,
                model=model,
                temperature=temperature,
                data_dir=data_dir,
                questions_file=questions_file,
                cost_tracker=cost_tracker
            )

            # Store result
            result_row = {'story_id': story_id}
            for run_num, median in enumerate(median_ratings, 1):
                result_row[f'run_{run_num}'] = median

            results.append(result_row)

            logger.info(f"  ✓ Completed {story_id}: {[f'{m:.2f}' if m is not None else 'None' for m in median_ratings]}")

        except Exception as e:
            logger.error(f"✗ Failed to process {story_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    # Create output DataFrame
    if not results:
        logger.error("No results to save!")
        sys.exit(1)

    output_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Results saved to: {output_path}")
    logger.info(f"  Total stories: {len(output_df)}")
    logger.info(f"  Total runs per story: {num_runs}")
    logger.info(f"  Total surveys: {len(output_df) * num_runs}")

    # Cost tracking
    logger.info(f"\nCost Tracking:")
    logger.info(f"  Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"  Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"    - Input: {cost_tracker.total_input_tokens:,}")
    logger.info(f"    - Output: {cost_tracker.total_output_tokens:,}")

    logger.info("\nResults preview:")
    print("\n" + output_df.to_string(index=False))


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Survey stability experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: S_1-S_5 and S_29-S_33, 10 runs, GPT-4o
  python survey_stability_experiment.py

  # Custom runs
  python survey_stability_experiment.py --num-runs 10 --model gpt-4o

  # Custom stories
  python survey_stability_experiment.py --story-ids S_1 S_2 S_3

  # With verbose logging
  python survey_stability_experiment.py --verbose
        """
    )

    parser.add_argument('--story-ids', nargs='+',
                        default=['S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_29', 'S_30', 'S_31', 'S_32', 'S_33'],
                        help='Story IDs to test')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of survey runs per story')
    parser.add_argument('--model', default='gpt-4o',
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature (0.0 for deterministic)')
    parser.add_argument('--data-dir', default='data',
                        help='Data directory')
    parser.add_argument('--mapping-file', default='output_analysis/story_name_mapping.csv',
                        help='Story name mapping CSV file')
    parser.add_argument('--output-file', default='output_analysis/survey_stability_results.csv',
                        help='Output CSV file')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    try:
        run_stability_experiment(
            story_ids=args.story_ids,
            num_runs=args.num_runs,
            model=args.model,
            temperature=args.temperature,
            data_dir=args.data_dir,
            mapping_file=args.mapping_file,
            output_file=args.output_file,
            verbose=args.verbose
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
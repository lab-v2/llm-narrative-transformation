"""
Extract Baseline Ratings Script

Extracts ratings from baseline transformation results and generates CSV files.
Creates CSV files for original ratings and baseline transformed ratings.

Output: CSV files in output_analysis/baseline/{model}/{problem}/
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Rating Extraction
# ==========================================================

def extract_ratings_from_survey(
        survey_file: Path,
        all_features: List[str]
) -> Dict[str, float]:
    """
    Extract ratings for all features from survey JSON.

    Args:
        survey_file: Path to survey.json file
        all_features: List of all features (for consistent ordering)

    Returns:
        Dict mapping feature to rating (or NaN if missing)
    """
    with open(survey_file, 'r', encoding='utf-8') as f:
        survey_data = json.load(f)

    # Extract ratings from questions_and_answers
    ratings = {}
    for qa in survey_data.get('questions_and_answers', []):
        # Handle both question types (forward uses individualistic, inverse uses collectivistic)
        if qa.get('type') in ['individualistic_question', 'collectivistic_question']:
            component = qa.get('component', '')
            rating = qa.get('rating')

            if component and rating is not None:
                # Convert component to feature format (underscores)
                import re
                feature = re.sub(r'[^\w\s]', '_', component)
                feature = re.sub(r'[\s_]+', '_', feature).lower().strip('_')
                ratings[feature] = rating

    # Return dict with all features (use NaN for missing)
    result = {}
    for feature in all_features:
        result[feature] = ratings.get(feature, float('nan'))

    return result


def get_all_features_from_survey(survey_file: Path) -> List[str]:
    """
    Extract all feature names from survey file.

    Args:
        survey_file: Path to survey.json

    Returns:
        List of feature names in alphabetical order
    """
    with open(survey_file, 'r', encoding='utf-8') as f:
        survey_data = json.load(f)

    features = set()
    for qa in survey_data.get('questions_and_answers', []):
        # Handle both question types (forward uses individualistic, inverse uses collectivistic)
        if qa.get('type') in ['individualistic_question', 'collectivistic_question']:
            component = qa.get('component', '')
            if component:
                # Convert to feature format
                import re
                feature = re.sub(r'[^\w\s]', '_', component)
                feature = re.sub(r'[\s_]+', '_', feature).lower().strip('_')
                features.add(feature)

    return sorted(list(features))


# ==========================================================
# Story Processing
# ==========================================================

def process_baseline_story(
        story_dir: Path,
        all_features: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Process one baseline story and extract ratings.

    Args:
        story_dir: Path to story directory
        all_features: List of all features

    Returns:
        Dict with keys 'original' and 'transformed'
    """
    story_name = story_dir.name
    results = {}

    # Extract original ratings
    original_survey = story_dir / "original_survey.json"
    if not original_survey.exists():
        logger.warning(f"Missing original_survey.json for {story_name}")
        return None

    results['original'] = extract_ratings_from_survey(original_survey, all_features)

    # Extract transformed ratings
    transformed_survey = story_dir / "transformed_survey.json"
    if not transformed_survey.exists():
        logger.warning(f"Missing transformed_survey.json for {story_name}")
        return None

    results['transformed'] = extract_ratings_from_survey(transformed_survey, all_features)

    return results


# ==========================================================
# DataFrame Creation
# ==========================================================

def create_baseline_dataframe(
        all_story_results: Dict[str, Dict],
        result_type: str,
        all_features: List[str]
) -> pd.DataFrame:
    """
    Create DataFrame from baseline story results.

    Args:
        all_story_results: Dict of {story_name: {result_type: {feature: rating}}}
        result_type: 'original' or 'transformed'
        all_features: List of all features (for column order)

    Returns:
        DataFrame with columns: story_name, features..., Mean, Median, Mode
    """
    rows = []

    for story_name, story_results in sorted(all_story_results.items()):
        row = {'story_name': story_name}

        ratings = story_results.get(result_type, {})
        feature_values = []

        for feature in all_features:
            rating = ratings.get(feature, float('nan'))
            row[feature] = rating
            feature_values.append(rating)

        # Calculate statistics
        valid_values = [v for v in feature_values if not pd.isna(v)]

        if valid_values:
            row['Mean'] = round(sum(valid_values) / len(valid_values), 2)
            row['Median'] = round(pd.Series(valid_values).median(), 2)
            mode_series = pd.Series(valid_values).mode()
            row['Mode'] = mode_series[0] if len(mode_series) > 0 else float('nan')
        else:
            row['Mean'] = float('nan')
            row['Median'] = float('nan')
            row['Mode'] = float('nan')

        rows.append(row)

    # Create DataFrame with proper column order
    columns = ['story_name'] + all_features + ['Mean', 'Median', 'Mode']
    df = pd.DataFrame(rows, columns=columns)

    return df


# ==========================================================
# Main CSV Generation
# ==========================================================

def create_baseline_csvs(
        base_dir: str,
        model: str,
        problem_type: str,
        output_dir: str
):
    """
    Create CSV files from baseline results.

    Args:
        base_dir: Base directory (e.g., "output/baseline")
        model: Model name (e.g., "gpt-4o")
        problem_type: "forward" or "inverse"
        output_dir: Where to save CSV files (e.g., "output_analysis/baseline")
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")

    # Sanitize model name for directory
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Find all story directories
    results_dir = Path(base_dir) / sanitized_model / problem_type
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(story_dirs)} story directories")

    if len(story_dirs) == 0:
        logger.error("No story directories found")
        return

    # Get all features from first story
    first_story = story_dirs[0]
    original_survey = first_story / "original_survey.json"
    if not original_survey.exists():
        logger.error(f"No original survey found in {first_story}")
        return

    all_features = get_all_features_from_survey(original_survey)
    logger.info(f"Features: {len(all_features)}")

    # Process all stories
    all_story_results = {}
    for story_dir in story_dirs:
        story_name = story_dir.name
        logger.info(f"Processing: {story_name}")

        story_results = process_baseline_story(story_dir, all_features)
        if story_results:
            all_story_results[story_name] = story_results

    if not all_story_results:
        logger.error("No valid story results extracted")
        return

    logger.info(f"\nExtracted results for {len(all_story_results)} stories")

    # Create output directory
    analysis_dir = Path(output_dir) / sanitized_model / problem_type
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving CSVs to: {analysis_dir}")

    # Create and save DataFrames
    csv_files_created = []

    # 1. Original ratings
    df_original = create_baseline_dataframe(all_story_results, 'original', all_features)
    original_file = analysis_dir / "original_ratings.csv"
    df_original.to_csv(original_file, index=False)
    csv_files_created.append(original_file.name)
    logger.info(f"✓ Created: {original_file.name}")

    # 2. Baseline transformed ratings
    df_transformed = create_baseline_dataframe(all_story_results, 'transformed', all_features)
    transformed_file = analysis_dir / "baseline_transformed_ratings.csv"
    df_transformed.to_csv(transformed_file, index=False)
    csv_files_created.append(transformed_file.name)
    logger.info(f"✓ Created: {transformed_file.name}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stories processed: {len(all_story_results)}")
    logger.info(f"Features: {len(all_features)}")
    logger.info(f"CSV files created: {len(csv_files_created)}")
    logger.info(f"Output directory: {analysis_dir}")
    logger.info("=" * 60)

    # List files
    logger.info("\nGenerated CSV files:")
    for csv_file in csv_files_created:
        logger.info(f"  - {csv_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract baseline transformation results into CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract baseline results for gpt-4o forward problem
  python plot_scripts/extract_ratings_baseline.py --model gpt-4o --problem forward

  # Extract for Grok inverse problem
  python plot_scripts/extract_ratings_baseline.py --model xai/grok-4-fast-reasoning --problem inverse
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4o, xai/grok-4-fast-reasoning)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type: forward or inverse'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='output/baseline',
        help='Base directory for baseline results (default: output/baseline)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/baseline',
        help='Output directory for CSV files (default: output_analysis/baseline)'
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run extraction
    create_baseline_csvs(
        base_dir=args.base_dir,
        model=args.model,
        problem_type=args.problem,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
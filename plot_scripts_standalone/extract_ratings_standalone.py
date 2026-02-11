#!/usr/bin/env python3
"""
Extract Ratings Standalone Script

Extracts ratings from survey results (survey_original_story.json, survey_baseline_transformed.json, 
survey_abduction_transformed.json) and generates CSV files for analysis.

Creates 3 CSV files per run:
- original_ratings.csv
- baseline_transformed_ratings.csv
- abduction_transformed_ratings.csv

Each CSV contains story ratings by component with statistical summaries.

Output: CSV files in output_analysis_standalone/survey/{tuned_model}/{survey_model}/{story_type}/
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Feature/Component Extraction
# ==========================================================

def get_all_components_from_survey(survey_file: Path) -> List[Tuple[int, str]]:
    """
    Extract all component names and their question IDs from survey file.

    Args:
        survey_file: Path to survey JSON file

    Returns:
        List of tuples (question_id, component_name) in order
    """
    with open(survey_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    components = []
    for qa in data.get('questions_and_answers', []):
        q_id = qa.get('id')
        component = qa.get('component')
        if q_id and component:
            components.append((q_id, component))

    # Sort by question ID to maintain order
    components.sort(key=lambda x: x[0])
    return components


def extract_ratings_from_survey(
        survey_file: Path,
        all_components: List[Tuple[int, str]]
) -> Dict[str, float]:
    """
    Extract ratings for all components from survey JSON.

    Args:
        survey_file: Path to survey JSON file
        all_components: List of (question_id, component_name) tuples

    Returns:
        Dict mapping component_name to rating (or NaN if missing)
    """
    with open(survey_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create question_id -> rating mapping
    ratings_by_id = {}
    for qa in data.get('questions_and_answers', []):
        q_id = qa.get('id')
        rating = qa.get('rating')
        if q_id is not None and rating is not None:
            ratings_by_id[q_id] = rating

    # Return dict with all components in order
    result = {}
    for q_id, component in all_components:
        result[component] = ratings_by_id.get(q_id, float('nan'))

    return result


# ==========================================================
# Story Processing
# ==========================================================

def process_story_version(
        survey_file: Path,
        all_components: List[Tuple[int, str]]
) -> Optional[Dict[str, float]]:
    """
    Process a single story version and extract ratings.

    Args:
        survey_file: Path to survey JSON file
        all_components: List of (question_id, component_name) tuples

    Returns:
        Dict mapping component to rating, or None if file doesn't exist
    """
    if not survey_file.exists():
        logger.warning(f"Survey file not found: {survey_file}")
        return None

    try:
        ratings = extract_ratings_from_survey(survey_file, all_components)
        return ratings
    except Exception as e:
        logger.error(f"Failed to process survey file {survey_file}: {e}")
        return None


# ==========================================================
# CSV Generation
# ==========================================================

def create_csv_files(
        survey_dir: str,
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        output_dir: str
):
    """
    Create CSV files from survey results.

    Args:
        survey_dir: Base survey directory (e.g., "survey")
        tuned_model_name: Name of tuned model that generated transformed stories
        model_name_for_survey: Name of model used for diagnosis/survey
        original_stories_type: Type of original stories (individualistic/collectivistic)
        output_dir: Where to save CSV files (e.g., "output_analysis_standalone/survey")
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING SURVEY RESULTS TO CSV")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Stories Type: {original_stories_type}")

    # Sanitize model names for directory
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Find survey results directory
    survey_base = Path(survey_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    if not survey_base.exists():
        logger.error(f"Survey directory not found: {survey_base}")
        return

    # Get all story directories
    story_dirs = [d for d in survey_base.iterdir() if d.is_dir()]
    logger.info(f"Found {len(story_dirs)} stories to process")

    if len(story_dirs) == 0:
        logger.error("No story directories found")
        return

    # Get all components from first story's original survey
    first_story = story_dirs[0]
    first_survey = first_story / "survey_original_story.json"
    if not first_survey.exists():
        logger.error(f"No survey file found in {first_story}")
        return

    all_components = get_all_components_from_survey(first_survey)
    logger.info(f"Components: {len(all_components)}")
    logger.debug(f"Components: {[c[1] for c in all_components]}")

    # Process all stories and versions
    all_story_results = {}
    for story_dir in story_dirs:
        story_name = story_dir.name

        # Process each version
        original_file = story_dir / "survey_original_story.json"
        baseline_file = story_dir / "survey_baseline_transformed.json"
        abduction_file = story_dir / "survey_abduction_transformed.json"

        original_ratings = process_story_version(original_file, all_components)
        baseline_ratings = process_story_version(baseline_file, all_components)
        abduction_ratings = process_story_version(abduction_file, all_components)

        if original_ratings is None and baseline_ratings is None and abduction_ratings is None:
            logger.warning(f"No valid survey results for {story_name}")
            continue

        all_story_results[story_name] = {
            'original': original_ratings,
            'baseline': baseline_ratings,
            'abduction': abduction_ratings
        }

    if not all_story_results:
        logger.error("No valid story results extracted")
        return

    logger.info(f"Extracted results for {len(all_story_results)} stories")

    # Create output directory
    analysis_dir = Path(output_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving CSVs to: {analysis_dir}")

    # Create DataFrames and save CSVs
    csv_files_created = []

    # Extract component names for column ordering
    component_names = [c[1] for c in all_components]

    # 1. Original ratings
    df_original = create_dataframe(all_story_results, 'original', component_names)
    original_file = analysis_dir / "original_ratings.csv"
    df_original.to_csv(original_file, index=False)
    csv_files_created.append(original_file.name)
    logger.info(f"✓ Created: {original_file.name}")

    # 2. Baseline transformed ratings
    df_baseline = create_dataframe(all_story_results, 'baseline', component_names)
    baseline_file = analysis_dir / "baseline_transformed_ratings.csv"
    df_baseline.to_csv(baseline_file, index=False)
    csv_files_created.append(baseline_file.name)
    logger.info(f"✓ Created: {baseline_file.name}")

    # 3. Abduction transformed ratings
    df_abduction = create_dataframe(all_story_results, 'abduction', component_names)
    abduction_file = analysis_dir / "abduction_transformed_ratings.csv"
    df_abduction.to_csv(abduction_file, index=False)
    csv_files_created.append(abduction_file.name)
    logger.info(f"✓ Created: {abduction_file.name}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Stories processed: {len(all_story_results)}")
    logger.info(f"Components: {len(component_names)}")
    logger.info(f"CSV files created: {len(csv_files_created)}")
    logger.info(f"Output directory: {analysis_dir}")
    logger.info("=" * 70)

    # List files
    logger.info("\nGenerated CSV files:")
    for csv_file in csv_files_created:
        logger.info(f"  - {csv_file}")


def create_dataframe(
        all_story_results: Dict[str, Dict],
        result_key: str,
        component_names: List[str]
) -> pd.DataFrame:
    """
    Create DataFrame from story results.

    Args:
        all_story_results: Dict of {story_name: {version: {component: rating}}}
        result_key: Which version to extract ('original', 'baseline', 'abduction')
        component_names: List of all component names (for column order)

    Returns:
        DataFrame with columns: story_name, component_1, ..., component_N, Mean, Median, Mode
    """
    rows = []

    for story_name, story_results in sorted(all_story_results.items()):
        row = {'story_name': story_name}

        ratings = story_results.get(result_key, {})
        feature_values = []

        # If ratings is None, skip this story for this version
        if ratings is None:
            continue

        for component in component_names:
            rating = ratings.get(component, float('nan'))
            row[component] = rating
            feature_values.append(rating)

        # Calculate statistics
        # Filter out NaN values for statistics
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
    columns = ['story_name'] + component_names + ['Mean', 'Median', 'Mode']

    df = pd.DataFrame(rows, columns=columns)

    return df


# ==========================================================
# Main Execution
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract survey results into CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract for baseline gpt-4o-mini, surveyed with gpt-4o-mini
  python extract_ratings_standalone.py \\
    --tuned-model-name "gpt-4o-mini-2024-07-18" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"

  # Extract for fine-tuned model, surveyed with gpt-4o-mini
  python extract_ratings_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"
        """
    )

    parser.add_argument(
        '--tuned-model-name',
        type=str,
        required=True,
        help='Name of the fine-tuned model that generated the transformed stories'
    )

    parser.add_argument(
        '--model-name-for-survey',
        type=str,
        required=True,
        help='Name of the model used for diagnosis/evaluation survey'
    )

    parser.add_argument(
        '--original-stories-type',
        type=str,
        choices=['individualistic', 'collectivistic'],
        required=True,
        help='Type of original stories being evaluated'
    )

    parser.add_argument(
        '--survey-dir',
        type=str,
        default='survey',
        help='Base survey directory (default: survey)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/survey',
        help='Output directory for CSV files (default: output_analysis_standalone/survey)'
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

    logger.info(f"Arguments:")
    logger.info(f"  --tuned-model-name: {args.tuned_model_name}")
    logger.info(f"  --model-name-for-survey: {args.model_name_for_survey}")
    logger.info(f"  --original-stories-type: {args.original_stories_type}")
    logger.info(f"  --survey-dir: {args.survey_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --verbose: {args.verbose}")

    # Run extraction
    create_csv_files(
        survey_dir=args.survey_dir,
        tuned_model_name=args.tuned_model_name,
        model_name_for_survey=args.model_name_for_survey,
        original_stories_type=args.original_stories_type,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

"""
Extract Results Script

Extracts ratings from Phase 2 experiment results and generates CSV files.
Creates CSV files for original ratings, target ratings, and transformed ratings
for each iteration.

Output: CSV files in output_analysis/phase2/{model}/{problem}/
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Feature Extraction
# ==========================================================

def get_all_features_from_abduction(abduction_file: Path) -> List[str]:
    """
    Extract all feature names from abduction analysis file.

    Args:
        abduction_file: Path to abduction_analysis.json

    Returns:
        List of feature names in alphabetical order
    """
    with open(abduction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = set()
    for gap in data.get('feature_gaps', []):
        features.add(gap['feature'])

    return sorted(list(features))


def extract_ratings_from_abduction(
        abduction_file: Path,
        rating_type: str,
        all_features: List[str]
) -> Dict[str, float]:
    """
    Extract ratings for all features from abduction analysis.

    Args:
        abduction_file: Path to abduction_analysis.json
        rating_type: "current_rating" or "target_rating"
        all_features: List of all features (for consistent ordering)

    Returns:
        Dict mapping feature to rating (or NaN if missing)
    """
    with open(abduction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create feature -> rating mapping
    ratings = {}
    for gap in data.get('feature_gaps', []):
        feature = gap['feature']
        rating = gap.get(rating_type)
        ratings[feature] = rating

    # Return dict with all features (use NaN for missing)
    result = {}
    for feature in all_features:
        result[feature] = ratings.get(feature, float('nan'))

    return result


# ==========================================================
# Story Processing
# ==========================================================

def process_story(
        story_dir: Path,
        all_features: List[str],
        max_iterations: int
) -> Dict[str, Dict[str, float]]:
    """
    Process one story and extract all ratings.

    Args:
        story_dir: Path to story directory (e.g., output/phase2/gpt-4o/forward/Community_Time/)
        all_features: List of all features
        max_iterations: Maximum iterations

    Returns:
        Dict with keys:
            - 'original': original ratings
            - 'target_0': target ratings for iteration 0
            - 'transformed_0': transformed ratings after iteration 0
            - etc.
    """
    story_name = story_dir.name
    results = {}

    # Extract original ratings (iteration_0 current)
    iter0_abduction = story_dir / "iteration_0" / "abduction_analysis.json"
    if not iter0_abduction.exists():
        logger.warning(f"Missing iteration_0 abduction for {story_name}")
        return None

    results['original'] = extract_ratings_from_abduction(
        iter0_abduction, 'current_rating', all_features
    )

    # Extract target and transformed for each complete cycle
    for i in range(max_iterations - 1):  # 0 to N-2
        # Target for iteration i
        iter_abduction = story_dir / f"iteration_{i}" / "abduction_analysis.json"
        if iter_abduction.exists():
            results[f'target_{i}'] = extract_ratings_from_abduction(
                iter_abduction, 'target_rating', all_features
            )
        else:
            logger.warning(f"Missing iteration_{i} abduction for {story_name}")
            results[f'target_{i}'] = {f: float('nan') for f in all_features}

        # Transformed (from next iteration's current)
        next_iter_abduction = story_dir / f"iteration_{i + 1}" / "abduction_analysis.json"
        if next_iter_abduction.exists():
            results[f'transformed_{i}'] = extract_ratings_from_abduction(
                next_iter_abduction, 'current_rating', all_features
            )
        else:
            logger.warning(f"Missing iteration_{i+1} abduction for {story_name}")
            results[f'transformed_{i}'] = {f: float('nan') for f in all_features}

    return results


# ==========================================================
# CSV Generation
# ==========================================================

def create_csv_files(
        base_dir: str,
        model: str,
        problem_type: str,
        max_iterations: int,
        output_dir: str
):
    """
    Create CSV files from Phase 2 results.

    Args:
        base_dir: Base directory (e.g., "output/phase2")
        model: Model name (e.g., "gpt-4o")
        problem_type: "forward" or "inverse"
        max_iterations: Maximum iterations
        output_dir: Where to save CSV files (e.g., "output_analysis/phase2")
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING PHASE 2 RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Max iterations: {max_iterations}")

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

    # Get all features from first story (for column consistency)
    first_story = story_dirs[0]
    iter0_abduction = first_story / "iteration_0" / "abduction_analysis.json"
    if not iter0_abduction.exists():
        logger.error(f"No abduction analysis found in {first_story}")
        return

    all_features = get_all_features_from_abduction(iter0_abduction)
    logger.info(f"Features: {len(all_features)}")
    logger.debug(f"Feature list: {all_features}")

    # Process all stories
    all_story_results = {}
    for story_dir in story_dirs:
        story_name = story_dir.name
        logger.info(f"Processing: {story_name}")

        story_results = process_story(story_dir, all_features, max_iterations)
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

    # Create DataFrames and save CSVs
    csv_files_created = []

    # 1. Original ratings
    df_original = create_dataframe(all_story_results, 'original', all_features)
    original_file = analysis_dir / "original_ratings.csv"
    df_original.to_csv(original_file, index=False)
    csv_files_created.append(original_file.name)
    logger.info(f"✓ Created: {original_file.name}")

    # 2. Target and transformed for each iteration
    for i in range(max_iterations - 1):
        # Target
        df_target = create_dataframe(all_story_results, f'target_{i}', all_features)
        target_file = analysis_dir / f"target_ratings_iteration_{i}.csv"
        df_target.to_csv(target_file, index=False)
        csv_files_created.append(target_file.name)
        logger.info(f"✓ Created: {target_file.name}")

        # Transformed
        df_transformed = create_dataframe(all_story_results, f'transformed_{i}', all_features)
        transformed_file = analysis_dir / f"transformed_ratings_iteration_{i}.csv"
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


def create_dataframe(
        all_story_results: Dict[str, Dict],
        result_key: str,
        all_features: List[str]
) -> pd.DataFrame:
    """
    Create DataFrame from story results.

    Args:
        all_story_results: Dict of {story_name: {result_key: {feature: rating}}}
        result_key: Which result to extract (e.g., 'original', 'target_0')
        all_features: List of all features (for column order)

    Returns:
        DataFrame with columns: story_name, feature_1, ..., feature_20, Mean, Median, Mode
    """
    rows = []

    for story_name, story_results in sorted(all_story_results.items()):
        row = {'story_name': story_name}

        ratings = story_results.get(result_key, {})
        feature_values = []

        for feature in all_features:
            rating = ratings.get(feature, float('nan'))
            row[feature] = rating
            feature_values.append(rating)

        # Calculate statistics (only for original and transformed, not target)
        if result_key == 'original' or result_key.startswith('transformed_'):
            # Filter out NaN values for statistics
            valid_values = [v for v in feature_values if not pd.isna(v)]

            if valid_values:
                row['Mean'] = round(sum(valid_values) / len(valid_values), 2)
                row['Median'] = round(pd.Series(valid_values).median(), 2)
                row['Mode'] = pd.Series(valid_values).mode()[0] if len(pd.Series(valid_values).mode()) > 0 else float(
                    'nan')
            else:
                row['Mean'] = float('nan')
                row['Median'] = float('nan')
                row['Mode'] = float('nan')

        rows.append(row)

    # Create DataFrame with proper column order
    columns = ['story_name'] + all_features

    # Add statistics columns only for original and transformed
    if result_key == 'original' or result_key.startswith('transformed_'):
        columns += ['Mean', 'Median', 'Mode']

    df = pd.DataFrame(rows, columns=columns)

    return df

# ==========================================================
# Main Execution
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract Phase 2 results into CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract results for gpt-4o forward problem
  python plot_scripts/extract_results.py --model gpt-4o --problem forward --max-iterations 3

  # Extract for Grok inverse problem
  python plot_scripts/extract_results.py --model xai/grok-4-fast-reasoning --problem inverse --max-iterations 5
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
        '--max-iterations',
        type=int,
        required=True,
        help='Maximum iterations used in Phase 2'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='output/phase2',
        help='Base directory for Phase 2 results (default: output/phase2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/phase2',
        help='Output directory for CSV files (default: output_analysis/phase2)'
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
    create_csv_files(
        base_dir=args.base_dir,
        model=args.model,
        problem_type=args.problem,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
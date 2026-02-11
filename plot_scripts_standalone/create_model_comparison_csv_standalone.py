#!/usr/bin/env python3
"""
Cross-Model Comparison CSV Generation Script (Standalone)

Creates comparison CSV files showing results across multiple tuned models.
All models are evaluated using the same survey model.
Generates 3 CSV files: one for Mean, one for Median, one for Mode.

Each CSV has columns for each tuned model's:
- Original rating
- Baseline transformed rating
- Abduction transformed rating

Output: CSV files in output_analysis_standalone/cross_model_summary/{survey_model}/{story_type}/
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Data Loading and Processing
# ==========================================================

def load_survey_data_for_model(
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        survey_dir: str = "output_analysis_standalone/survey"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load survey data for a specific tuned model.

    Args:
        tuned_model_name: Name of tuned model
        model_name_for_survey: Name of survey model
        original_stories_type: Type of original stories
        survey_dir: Base survey directory

    Returns:
        Tuple of (original_df, baseline_df, abduction_df) or (None, None, None) if not found
    """
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    survey_path = Path(survey_dir) / sanitized_tuned / sanitized_survey / original_stories_type

    original_file = survey_path / "original_ratings.csv"
    baseline_file = survey_path / "baseline_transformed_ratings.csv"
    abduction_file = survey_path / "abduction_transformed_ratings.csv"

    # Check if all files exist
    if not all([original_file.exists(), baseline_file.exists(), abduction_file.exists()]):
        logger.warning(f"Not all survey files found for {tuned_model_name}")
        return None, None, None

    try:
        df_original = pd.read_csv(original_file)
        df_baseline = pd.read_csv(baseline_file)
        df_abduction = pd.read_csv(abduction_file)
        return df_original, df_baseline, df_abduction
    except Exception as e:
        logger.error(f"Error loading data for {tuned_model_name}: {e}")
        return None, None, None


def create_model_comparison_df(
        tuned_model_names: list,
        model_name_for_survey: str,
        original_stories_type: str,
        stat_type: str,
        survey_dir: str = "output_analysis_standalone/survey"
) -> pd.DataFrame:
    """
    Create comparison DataFrame for a specific statistic.

    Args:
        tuned_model_names: List of tuned model names
        model_name_for_survey: Survey model name
        original_stories_type: Type of original stories
        stat_type: "mean", "median", or "mode"
        survey_dir: Base survey directory

    Returns:
        Combined DataFrame with all models
    """
    logger.info(f"\nCreating {stat_type.upper()} comparison CSV...")

    # Load survey data for all tuned models
    model_data = {}

    for tuned_model in tuned_model_names:
        df_orig, df_base, df_abduction = load_survey_data_for_model(
            tuned_model, model_name_for_survey, original_stories_type, survey_dir
        )

        if df_orig is not None and df_base is not None and df_abduction is not None:
            model_data[tuned_model] = {
                'original': df_orig,
                'baseline': df_base,
                'abduction': df_abduction
            }
            logger.info(f"  Loaded {len(df_orig)} stories for {tuned_model}")
        else:
            logger.warning(f"  Could not load data for {tuned_model}")

    if not model_data:
        raise ValueError("No data loaded for any tuned model")

    # Get all unique stories (use first model as reference)
    first_model = list(model_data.keys())[0]
    all_stories = sorted(model_data[first_model]['original']['story_name'].tolist())

    # Build comparison DataFrame
    rows = []
    stat_col = stat_type.capitalize()  # "Mean", "Median", or "Mode"

    for story in all_stories:
        row = {'story_name': story}

        for tuned_model in tuned_model_names:
            if tuned_model not in model_data:
                # Model data not available
                row[f'{tuned_model}_original'] = None
                row[f'{tuned_model}_baseline'] = None
                row[f'{tuned_model}_abduction'] = None
                continue

            # Get data for this model-story combination
            df_orig = model_data[tuned_model]['original']
            df_base = model_data[tuned_model]['baseline']
            df_abduction = model_data[tuned_model]['abduction']

            # Original rating
            story_orig = df_orig[df_orig['story_name'] == story]
            if len(story_orig) > 0 and stat_col in story_orig.columns:
                row[f'{tuned_model}_original'] = story_orig[stat_col].values[0]
            else:
                row[f'{tuned_model}_original'] = None

            # Baseline rating
            story_base = df_base[df_base['story_name'] == story]
            if len(story_base) > 0 and stat_col in story_base.columns:
                row[f'{tuned_model}_baseline'] = story_base[stat_col].values[0]
            else:
                row[f'{tuned_model}_baseline'] = None

            # Abduction rating
            story_abduction = df_abduction[df_abduction['story_name'] == story]
            if len(story_abduction) > 0 and stat_col in story_abduction.columns:
                row[f'{tuned_model}_abduction'] = story_abduction[stat_col].values[0]
            else:
                row[f'{tuned_model}_abduction'] = None

        rows.append(row)

    df_comparison = pd.DataFrame(rows)
    return df_comparison


# ==========================================================
# Main Function
# ==========================================================

def generate_model_comparison_csvs(
        tuned_model_names: list,
        model_name_for_survey: str,
        original_stories_type: str,
        survey_dir: str = "output_analysis_standalone/survey",
        output_dir: str = "output_analysis_standalone/cross_model_summary"
):
    """
    Generate model comparison CSV files.

    Args:
        tuned_model_names: List of tuned model names
        model_name_for_survey: Survey model name
        original_stories_type: Type of original stories
        survey_dir: Base survey directory
        output_dir: Output directory
    """
    logger.info("=" * 70)
    logger.info("GENERATING MODEL COMPARISON CSVs (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Models: {tuned_model_names}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")

    # Create output directory
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")
    output_path = Path(output_dir) / sanitized_survey / original_stories_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate CSV for each statistic
    for stat in ['mean', 'median', 'mode']:
        df = create_model_comparison_df(
            tuned_model_names, model_name_for_survey, original_stories_type, 
            stat, survey_dir
        )

        csv_file = output_path / f"model_comparison_{stat}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"\n✓ Saved: {csv_file}")
        logger.info(f"  Stories: {len(df)}")
        logger.info(f"  Columns: {len(df.columns)}")

    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON CSVs COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_path}")
    logger.info("Files created:")
    logger.info("  - model_comparison_mean.csv")
    logger.info("  - model_comparison_median.csv")
    logger.info("  - model_comparison_mode.csv")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate cross-model comparison CSV files from survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comparison CSVs for all tuned models
  python plot_scripts_standalone/create_model_comparison_csv_standalone.py \\
    --tuned-model-names \\
      "gpt-4o-mini-2024-07-18" \\
      "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
      "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm3-D75Ahi1l" \\
      "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-D5NuhUdq" \\
      "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-en-D6rDdxB0" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"
        """
    )

    parser.add_argument(
        '--tuned-model-names',
        nargs='+',
        required=True,
        help='List of tuned model names to compare'
    )

    parser.add_argument(
        '--model-name-for-survey',
        type=str,
        required=True,
        help='Survey model name (common to all comparisons)'
    )

    parser.add_argument(
        '--original-stories-type',
        type=str,
        choices=['individualistic', 'collectivistic'],
        required=True,
        help='Type of original stories'
    )

    parser.add_argument(
        '--survey-dir',
        type=str,
        default='output_analysis_standalone/survey',
        help='Base survey directory (default: output_analysis_standalone/survey)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/cross_model_summary',
        help='Output directory (default: output_analysis_standalone/cross_model_summary)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Arguments:")
    logger.info(f"  --tuned-model-names: {args.tuned_model_names}")
    logger.info(f"  --model-name-for-survey: {args.model_name_for_survey}")
    logger.info(f"  --original-stories-type: {args.original_stories_type}")
    logger.info(f"  --survey-dir: {args.survey_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --verbose: {args.verbose}\n")

    try:
        generate_model_comparison_csvs(
            tuned_model_names=args.tuned_model_names,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            survey_dir=args.survey_dir,
            output_dir=args.output_dir
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

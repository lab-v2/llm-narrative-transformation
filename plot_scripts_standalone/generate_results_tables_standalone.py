#!/usr/bin/env python3
"""
Results Table Generation Script (Standalone)

Creates tables comparing One-Shot (Baseline) vs Abduction-Guided approaches across tuned models.
Computes average improvement over original stories as percentage of gap closed.

Output: CSV file in output_analysis_standalone/tables/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Improvement Computation
# ==========================================================

def compute_improvements(
        df: pd.DataFrame,
        tuned_model_name: str,
        original_stories_type: str
) -> tuple:
    """
    Compute average improvements for tuned model.

    Args:
        df: Model comparison DataFrame
        tuned_model_name: Name of tuned model
        original_stories_type: Type of original stories (individualistic/collectivistic)

    Returns:
        Tuple of (baseline_avg, abduction_avg)
    """
    original_col = f'{tuned_model_name}_original'
    baseline_col = f'{tuned_model_name}_baseline'
    abduction_col = f'{tuned_model_name}_abduction'

    if not all(col in df.columns for col in [original_col, baseline_col, abduction_col]):
        logger.warning(f"  Missing columns for {tuned_model_name}")
        return None, None

    baseline_improvements = []
    abduction_improvements = []

    # Determine target based on original stories type
    # If original is individualistic, we transform towards collectivistic (higher = better)
    # If original is collectivistic, we transform towards individualistic (lower = better)
    if original_stories_type == "individualistic":
        target = 5.0  # Inverse: higher is better
        is_inverse = True
    else:  # collectivistic
        target = 1.0  # Forward: lower is better
        is_inverse = False

    for _, row in df.iterrows():
        original = row[original_col]
        baseline = row[baseline_col]
        abduction = row[abduction_col]

        # Skip if any value is missing
        if pd.notna(original) and pd.notna(baseline) and pd.notna(abduction):
            if is_inverse:
                # Inverse: higher is better, target = 5.0
                gap_to_target = target - original

                if gap_to_target > 0:
                    # Baseline improvement
                    baseline_improvement = ((baseline - original) / gap_to_target) * 100
                    baseline_improvements.append(baseline_improvement)

                    # Abduction improvement
                    abduction_improvement = ((abduction - original) / gap_to_target) * 100
                    abduction_improvements.append(abduction_improvement)
            else:
                # Forward: lower is better, target = 1.0
                gap_to_target = original - target

                if gap_to_target > 0:
                    # Baseline improvement
                    baseline_improvement = ((original - baseline) / gap_to_target) * 100
                    baseline_improvements.append(baseline_improvement)

                    # Abduction improvement
                    abduction_improvement = ((original - abduction) / gap_to_target) * 100
                    abduction_improvements.append(abduction_improvement)

    # Average across all stories
    avg_baseline = np.mean(baseline_improvements) if baseline_improvements else None
    avg_abduction = np.mean(abduction_improvements) if abduction_improvements else None

    return avg_baseline, avg_abduction


# ==========================================================
# Table Generation
# ==========================================================

def generate_results_table(
        tuned_model_names: list,
        model_name_for_survey: str,
        original_stories_type: str,
        summary_dir: str = "output_analysis_standalone/cross_model_summary"
) -> pd.DataFrame:
    """
    Generate results table.

    Args:
        tuned_model_names: List of tuned model names
        model_name_for_survey: Survey model name
        original_stories_type: Type of original stories
        summary_dir: Summary directory

    Returns:
        DataFrame with table
    """
    logger.info(f"\nGenerating results table...")
    logger.info(f"  Original Stories Type: {original_stories_type}")

    # Sanitize survey model name
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Load median comparison CSV
    median_file = Path(summary_dir) / sanitized_survey / original_stories_type / "model_comparison_median.csv"

    if not median_file.exists():
        logger.error(f"Median comparison file not found: {median_file}")
        return None

    df = pd.read_csv(median_file)
    logger.info(f"  Loaded {len(df)} stories")

    # Compute improvements for each tuned model
    rows = []

    for tuned_model in tuned_model_names:
        logger.info(f"\n  Computing for {tuned_model}...")

        # Compute improvements
        baseline_avg, abduction_avg = compute_improvements(
            df, tuned_model, original_stories_type
        )

        if baseline_avg is None or abduction_avg is None:
            logger.warning(f"    Skipping {tuned_model} (missing data)")
            continue

        logger.info(f"    Zero-Shot (Baseline): {baseline_avg:.2f}%")
        logger.info(f"    Abduction-Guided: {abduction_avg:.2f}%")

        # Add two rows (one for baseline, one for abduction)
        rows.append({
            'Approach': 'Zero-Shot',
            'Tuned Model': tuned_model,
            'Avg Improvement (%)': round(baseline_avg, 2)
        })

        rows.append({
            'Approach': 'Abduction-Guided',
            'Tuned Model': tuned_model,
            'Avg Improvement (%)': round(abduction_avg, 2)
        })

    # Create DataFrame
    df_table = pd.DataFrame(rows)

    return df_table


# ==========================================================
# Main Function
# ==========================================================

def generate_results_tables(
        tuned_model_names: list,
        model_name_for_survey: str,
        original_stories_type: str,
        summary_dir: str = "output_analysis_standalone/cross_model_summary",
        output_dir: str = "output_analysis_standalone/tables"
):
    """
    Generate results table.

    Args:
        tuned_model_names: List of tuned model names
        model_name_for_survey: Survey model name
        original_stories_type: Type of original stories
        summary_dir: Summary directory
        output_dir: Output directory
    """
    logger.info("=" * 70)
    logger.info("GENERATING RESULTS TABLE (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Models: {tuned_model_names}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")

    # Create output directory
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")
    output_path = Path(output_dir) / sanitized_survey / original_stories_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate table
    df_results = generate_results_table(
        tuned_model_names, model_name_for_survey, original_stories_type, summary_dir
    )

    if df_results is not None and len(df_results) > 0:
        results_file = output_path / "table_results.csv"
        df_results.to_csv(results_file, index=False)
        logger.info(f"\n✓ Results table saved: {results_file}")
        logger.info("\nTable Preview:")
        print(df_results.to_string(index=False))
    else:
        logger.warning("No results to save")
        return

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS TABLE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info("Files:")
    logger.info("  - table_results.csv")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate results tables from survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate table for all tuned models
  python plot_scripts_standalone/generate_results_tables_standalone.py \\
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
        help='List of tuned model names to include in table'
    )

    parser.add_argument(
        '--model-name-for-survey',
        type=str,
        required=True,
        help='Survey model name'
    )

    parser.add_argument(
        '--original-stories-type',
        type=str,
        choices=['individualistic', 'collectivistic'],
        required=True,
        help='Type of original stories'
    )

    parser.add_argument(
        '--summary-dir',
        type=str,
        default='output_analysis_standalone/cross_model_summary',
        help='Summary directory (default: output_analysis_standalone/cross_model_summary)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/tables',
        help='Output directory (default: output_analysis_standalone/tables)'
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
    logger.info(f"  --summary-dir: {args.summary_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --verbose: {args.verbose}\n")

    try:
        generate_results_tables(
            tuned_model_names=args.tuned_model_names,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            summary_dir=args.summary_dir,
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

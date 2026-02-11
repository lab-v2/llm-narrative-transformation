"""
Results Table Generation Script

Creates tables comparing One-Shot vs Abduction approaches across LLMs.
Computes average improvement over original stories as percentage of gap closed.

Generates two tables:
- Table 1: Forward problem (Collectivistic → Individualistic)
- Table 2: Inverse problem (Individualistic → Collectivistic)

Output: CSV files in output_analysis/tables/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Model Name Mapping
# ==========================================================

def get_model_display_name(model_short: str) -> str:
    """Get display name for model."""
    mapping = {
        'gpt4o': 'GPT-4o',
        'gpt5o': 'GPT-5o',
        'claude45': 'Claude Sonnet 4.5',
        'claude35': 'Claude Sonnet 3.5',
        'claude': 'Claude',
        'grok': 'Grok',
        'llama4': 'Llama 4',
        'llama3': 'Llama 3',
        'deepseek_r1': 'DeepSeek R1',
        'deepseek': 'DeepSeek',
        'gpt52': 'GPT-5.2'
    }
    return mapping.get(model_short, model_short)


# ==========================================================
# Improvement Computation
# ==========================================================

def compute_improvements_forward(
        df: pd.DataFrame,
        model_short: str
) -> tuple:
    """
    Compute average improvements for forward problem.

    Args:
        df: Model comparison DataFrame
        model_short: Model short name

    Returns:
        Tuple of (baseline_avg, abduction_avg)
    """
    original_col = f'{model_short}_original'
    baseline_col = f'{model_short}_baseline'
    transformed_col = f'{model_short}_transformed'

    if original_col not in df.columns or baseline_col not in df.columns or transformed_col not in df.columns:
        logger.warning(f"  Missing columns for {model_short}")
        return None, None

    baseline_improvements = []
    abduction_improvements = []

    for _, row in df.iterrows():
        original = row[original_col]
        baseline = row[baseline_col]
        transformed = row[transformed_col]

        # Skip if any value is missing
        if pd.notna(original) and pd.notna(baseline) and pd.notna(transformed):
            # Forward: lower is better, target = 1.0
            gap_to_target = original - 1.0

            if gap_to_target > 0:
                # Baseline improvement
                baseline_improvement = ((original - baseline) / gap_to_target) * 100
                baseline_improvements.append(baseline_improvement)

                # Abduction improvement
                abduction_improvement = ((original - transformed) / gap_to_target) * 100
                abduction_improvements.append(abduction_improvement)

    # Average across all stories
    avg_baseline = np.mean(baseline_improvements) if baseline_improvements else None
    avg_abduction = np.mean(abduction_improvements) if abduction_improvements else None

    return avg_baseline, avg_abduction


def compute_improvements_inverse(
        df: pd.DataFrame,
        model_short: str
) -> tuple:
    """
    Compute average improvements for inverse problem.

    Args:
        df: Model comparison DataFrame
        model_short: Model short name

    Returns:
        Tuple of (baseline_avg, abduction_avg)
    """
    original_col = f'{model_short}_original'
    baseline_col = f'{model_short}_baseline'
    transformed_col = f'{model_short}_transformed'

    if original_col not in df.columns or baseline_col not in df.columns or transformed_col not in df.columns:
        logger.warning(f"  Missing columns for {model_short}")
        return None, None

    baseline_improvements = []
    abduction_improvements = []

    for _, row in df.iterrows():
        original = row[original_col]
        baseline = row[baseline_col]
        transformed = row[transformed_col]

        # Skip if any value is missing
        if pd.notna(original) and pd.notna(baseline) and pd.notna(transformed):
            # Inverse: higher is better, target = 5.0
            gap_to_target = 5.0 - original

            if gap_to_target > 0:
                # Baseline improvement
                baseline_improvement = ((baseline - original) / gap_to_target) * 100
                baseline_improvements.append(baseline_improvement)

                # Abduction improvement
                abduction_improvement = ((transformed - original) / gap_to_target) * 100
                abduction_improvements.append(abduction_improvement)

    # Average across all stories
    avg_baseline = np.mean(baseline_improvements) if baseline_improvements else None
    avg_abduction = np.mean(abduction_improvements) if abduction_improvements else None

    return avg_baseline, avg_abduction


# ==========================================================
# Table Generation
# ==========================================================

def generate_results_table(
        models: list,
        problem_type: str,
        summary_dir: str = "output_analysis/cross_model_summary"
) -> pd.DataFrame:
    """
    Generate results table for a specific problem.

    Args:
        models: List of model short names
        problem_type: "forward" or "inverse"
        summary_dir: Summary directory

    Returns:
        DataFrame with table
    """
    logger.info(f"\nGenerating {problem_type.upper()} table...")

    # Load median comparison CSV
    median_file = Path(summary_dir) / problem_type / "model_comparison_median.csv"

    if not median_file.exists():
        logger.error(f"Median comparison file not found: {median_file}")
        return None

    df = pd.read_csv(median_file)
    logger.info(f"  Loaded {len(df)} stories")

    # Compute improvements for each model
    rows = []

    for model_short in models:
        logger.info(f"\n  Computing for {model_short}...")

        # Compute improvements
        if problem_type == "forward":
            baseline_avg, abduction_avg = compute_improvements_forward(df, model_short)
        else:  # inverse
            baseline_avg, abduction_avg = compute_improvements_inverse(df, model_short)

        if baseline_avg is None or abduction_avg is None:
            logger.warning(f"    Skipping {model_short} (missing data)")
            continue

        logger.info(f"    Baseline: {baseline_avg:.2f}%")
        logger.info(f"    Abduction: {abduction_avg:.2f}%")

        # Add two rows (one for baseline, one for abduction)
        # model_display = get_model_display_name(model_short)
        model_display = model_short
        rows.append({
            'Approach': 'One-Shot',
            'LLM Backbone': model_display,
            'Avg Improvement (%)': round(baseline_avg, 2)
        })

        rows.append({
            'Approach': 'Abduction',
            'LLM Backbone': model_display,
            'Avg Improvement (%)': round(abduction_avg, 2)
        })

    # Create DataFrame
    df_table = pd.DataFrame(rows)

    return df_table


# ==========================================================
# Main Function
# ==========================================================

def generate_results_tables(
        models: list,
        summary_dir: str = "output_analysis/cross_model_summary",
        output_dir: str = "output_analysis/tables"
):
    """
    Generate results tables for both problems.

    Args:
        models: List of model short names
        summary_dir: Summary directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("GENERATING RESULTS TABLES")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate forward table
    df_forward = generate_results_table(models, "forward", summary_dir)

    if df_forward is not None:
        forward_file = output_path / "table_forward_results.csv"
        df_forward.to_csv(forward_file, index=False)
        logger.info(f"\n✓ Forward table saved: {forward_file}")
        logger.info("\nForward Table Preview:")
        print(df_forward.to_string(index=False))

    # Generate inverse table
    df_inverse = generate_results_table(models, "inverse", summary_dir)

    if df_inverse is not None:
        inverse_file = output_path / "table_inverse_results.csv"
        df_inverse.to_csv(inverse_file, index=False)
        logger.info(f"\n✓ Inverse table saved: {inverse_file}")
        logger.info("\nInverse Table Preview:")
        print(df_inverse.to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS TABLES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {output_path}")
    logger.info("Files:")
    logger.info("  - table_forward_results.csv")
    logger.info("  - table_inverse_results.csv")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate results tables for paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tables for all models
  python plot_scripts/generate_results_tables.py \\
      --models gpt4o claude45 grok llama4 deepseek_r1
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of model SHORT names (gpt4o, claude45, grok, llama4, deepseek_r1)'
    )
    parser.add_argument('--summary-dir', default='output_analysis/cross_model_summary')
    parser.add_argument('--output-dir', default='output_analysis/tables')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_results_tables(
            models=args.models,
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
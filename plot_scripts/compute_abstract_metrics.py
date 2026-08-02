"""
Abstract Metrics Computation Script

Computes the improvement percentages for the paper abstract:
- x%: Forward rating improvement (median)
- y%: Forward KL divergence improvement (Q2)
- z%: Inverse rating improvement (median)
- w%: Inverse KL divergence improvement (Q2)

Output: CSV file with metrics for all models
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Metric Computation
# ==========================================================

def compute_abstract_metrics(
        model_name: str,
        summary_dir: str = "output_analysis/cross_model_summary"
) -> dict:
    """
    Compute abstract metrics for a specific model.

    Args:
        model_name: Model name (short form like 'gpt4o', 'claude45')
        summary_dir: Cross-model summary directory

    Returns:
        Dict with x, y, z, w percentages
    """
    logger.info(f"\nComputing metrics for {model_name}...")

    results = {'model': model_name}

    # ==========================================
    # FORWARD PROBLEM
    # ==========================================

    # Load forward median ratings
    forward_median_file = Path(summary_dir) / "forward" / "model_comparison_median.csv"

    if forward_median_file.exists():
        df_forward = pd.read_csv(forward_median_file)

        # Get baseline and abductive medians (averaged across all stories)
        baseline_col = f'{model_name}_baseline'
        abd_col = f'{model_name}_transformed'

        if baseline_col in df_forward.columns and abd_col in df_forward.columns:
            baseline_median = df_forward[baseline_col].mean()  # Mean of medians across stories
            abd_median = df_forward[abd_col].mean()

            # x% = rating improvement (lower is better)
            if baseline_median > 0:
                x_pct = ((baseline_median - abd_median) / baseline_median) * 100
                results['x_rating_improvement_forward'] = round(x_pct, 2)
                logger.info(f"  Forward rating: Baseline={baseline_median:.2f}, Abd={abd_median:.2f}, x={x_pct:.2f}%")
            else:
                results['x_rating_improvement_forward'] = None
        else:
            logger.warning(f"  Columns not found for forward ratings")
            results['x_rating_improvement_forward'] = None
    else:
        logger.warning(f"  Forward median file not found")
        results['x_rating_improvement_forward'] = None

    # Load forward KL quartiles
    forward_kl_file = Path(summary_dir) / "forward" / "kl_divergence_quartiles.csv"

    if forward_kl_file.exists():
        df_kl_forward = pd.read_csv(forward_kl_file)
        model_row = df_kl_forward[df_kl_forward['model'] == model_name]

        if len(model_row) > 0:
            baseline_q2 = model_row['baseline_q2'].values[0]
            abd_q2 = model_row['abd_q2'].values[0]

            # y% = KL improvement (lower is better)
            if pd.notna(baseline_q2) and baseline_q2 > 0:
                y_pct = ((baseline_q2 - abd_q2) / baseline_q2) * 100
                results['y_kl_improvement_forward'] = round(y_pct, 2)
                logger.info(f"  Forward KL: Baseline_Q2={baseline_q2:.4f}, Abd_Q2={abd_q2:.4f}, y={y_pct:.2f}%")
            else:
                results['y_kl_improvement_forward'] = None
        else:
            logger.warning(f"  Model not found in forward KL quartiles")
            results['y_kl_improvement_forward'] = None
    else:
        logger.warning(f"  Forward KL file not found")
        results['y_kl_improvement_forward'] = None

    # ==========================================
    # INVERSE PROBLEM
    # ==========================================

    # Load inverse median ratings
    inverse_median_file = Path(summary_dir) / "inverse" / "model_comparison_median.csv"

    if inverse_median_file.exists():
        df_inverse = pd.read_csv(inverse_median_file)

        # Get baseline and abductive medians
        baseline_col = f'{model_name}_baseline'
        abd_col = f'{model_name}_transformed'

        if baseline_col in df_inverse.columns and abd_col in df_inverse.columns:
            baseline_median = df_inverse[baseline_col].mean()
            abd_median = df_inverse[abd_col].mean()

            # z% = rating improvement (higher is better for inverse)
            if baseline_median > 0:
                z_pct = ((abd_median - baseline_median) / baseline_median) * 100
                results['z_rating_improvement_inverse'] = round(z_pct, 2)
                logger.info(f"  Inverse rating: Baseline={baseline_median:.2f}, Abd={abd_median:.2f}, z={z_pct:.2f}%")
            else:
                results['z_rating_improvement_inverse'] = None
        else:
            logger.warning(f"  Columns not found for inverse ratings")
            results['z_rating_improvement_inverse'] = None
    else:
        logger.warning(f"  Inverse median file not found")
        results['z_rating_improvement_inverse'] = None

    # Load inverse KL quartiles
    inverse_kl_file = Path(summary_dir) / "inverse" / "kl_divergence_quartiles.csv"

    if inverse_kl_file.exists():
        df_kl_inverse = pd.read_csv(inverse_kl_file)
        model_row = df_kl_inverse[df_kl_inverse['model'] == model_name]

        if len(model_row) > 0:
            baseline_q2 = model_row['baseline_q2'].values[0]
            abd_q2 = model_row['abd_q2'].values[0]

            # w% = KL improvement (lower is better)
            if pd.notna(baseline_q2) and baseline_q2 > 0:
                w_pct = ((baseline_q2 - abd_q2) / baseline_q2) * 100
                results['w_kl_improvement_inverse'] = round(w_pct, 2)
                logger.info(f"  Inverse KL: Baseline_Q2={baseline_q2:.4f}, Abd_Q2={abd_q2:.4f}, w={w_pct:.2f}%")
            else:
                results['w_kl_improvement_inverse'] = None
        else:
            logger.warning(f"  Model not found in inverse KL quartiles")
            results['w_kl_improvement_inverse'] = None
    else:
        logger.warning(f"  Inverse KL file not found")
        results['w_kl_improvement_inverse'] = None

    return results


# ==========================================================
# Main Function
# ==========================================================

def generate_abstract_metrics(
        models: list,
        summary_dir: str = "output_analysis/cross_model_summary",
        output_dir: str = "output_analysis/cross_model_summary"
):
    """
    Generate abstract metrics CSV.

    Args:
        models: List of model short names
        summary_dir: Summary directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("COMPUTING ABSTRACT METRICS")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")

    results = []

    for model_short in models:
        metrics = compute_abstract_metrics(model_short, summary_dir)
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['model',
            'x_rating_improvement_forward', 'y_kl_improvement_forward',
            'z_rating_improvement_inverse', 'w_kl_improvement_inverse']
    df = df[cols]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "abstract_metrics.csv"
    df.to_csv(csv_file, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("ABSTRACT METRICS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Saved: {csv_file}")
    logger.info("\nSummary:")
    print("\n" + df.to_string(index=False))


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute abstract metrics (x, y, z, w percentages)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for all models
  python plot_scripts/compute_abstract_metrics.py \\
      --models gpt4o claude45 grok llama4 deepseek_r1

  # Compute for specific models
  python plot_scripts/compute_abstract_metrics.py \\
      --models gpt4o claude45
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of model SHORT names (gpt4o, claude45, grok, llama4, deepseek_r1)'
    )
    parser.add_argument('--summary-dir', default='output_analysis/cross_model_summary')
    parser.add_argument('--output-dir', default='output_analysis/cross_model_summary')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_abstract_metrics(
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
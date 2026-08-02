"""
KL Divergence Quartile Comparison CSV Script

Creates a CSV showing KL divergence distribution quartiles across models.
One row per model with Q1, Q2 (median), Q3 for baseline and abductive approaches.

Output: CSV file in output_analysis/cross_model_summary/{problem}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Data Loading
# ==========================================================

def load_kl_divergence_for_model(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2"
) -> tuple:
    """
    Load KL divergence data for a model.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        divergence_dir: Divergence directory
        phase2_dir: Phase 2 directory (for optimal iteration info)

    Returns:
        Tuple of (kl_df, optimal_df) or (None, None) if not found
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load KL divergence CSV
    kl_file = Path(divergence_dir) / sanitized_model / problem_type / "kl_divergence_transformed_original.csv"
    if not kl_file.exists():
        logger.warning(f"KL divergence not found for {model}")
        return None, None

    # Load optimal iterations
    optimal_file = Path(phase2_dir) / sanitized_model / problem_type / "optimal_iterations.csv"
    if not optimal_file.exists():
        logger.warning(f"Optimal iterations not found for {model}")
        return None, None

    df_kl = pd.read_csv(kl_file)
    df_optimal = pd.read_csv(optimal_file)

    return df_kl, df_optimal


def get_model_short_name(model: str) -> str:
    """Get short name for model."""
    # if 'gpt-4o' in model.lower():
    #     return 'gpt4o'
    # elif 'gpt-5o' in model.lower():
    #     return 'gpt5o'
    # elif 'gpt' in model.lower():
    #     return model.lower().replace('gpt-', 'gpt').replace('.', '')[:8]
    # elif 'claude' in model.lower():
    #     if 'sonnet' in model.lower():
    #         if '4-5' in model or '4.5' in model:
    #             return 'claude45'
    #         elif '3-5' in model or '3.5' in model:
    #             return 'claude35'
    #         else:
    #             return 'claude_sonnet'
    #     elif 'opus' in model.lower():
    #         return 'claude_opus'
    #     elif 'haiku' in model.lower():
    #         return 'claude_haiku'
    #     else:
    #         return 'claude'
    # elif 'grok' in model.lower():
    #     return 'grok'
    # elif 'llama' in model.lower():
    #     if 'llama4' in model.lower() or 'llama-4' in model.lower():
    #         return 'llama4'
    #     elif 'llama3' in model.lower() or 'llama-3' in model.lower():
    #         return 'llama3'
    #     else:
    #         return 'llama'
    # elif 'deepseek' in model.lower():
    #     if 'r1' in model.lower():
    #         return 'deepseek_r1'
    #     else:
    #         return 'deepseek'
    # else:
    return model


# ==========================================================
# Quartile Computation
# ==========================================================

def create_kl_quartile_comparison(
        models: list,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2"
) -> pd.DataFrame:
    """
    Create KL divergence quartile comparison DataFrame.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        divergence_dir: Divergence directory
        phase2_dir: Phase 2 directory

    Returns:
        DataFrame with quartiles for each model
    """
    logger.info("\nComputing KL divergence quartiles...")

    rows = []

    for model in models:
        logger.info(f"\nProcessing {model}...")

        # Load data
        df_kl, df_optimal = load_kl_divergence_for_model(model, problem_type, divergence_dir, phase2_dir)

        if df_kl is None or df_optimal is None:
            logger.warning(f"  Skipping {model} (data not found)")
            continue

        # Get short model name
        model_short = get_model_short_name(model)

        # Collect baseline KL values
        baseline_kl_values = df_kl['baseline_kl'].dropna().tolist()

        # Collect optimal abductive KL values
        optimal_kl_values = []
        for _, row_data in df_kl.iterrows():
            story_name = row_data['story_name']

            # Find optimal iteration for this story
            optimal_row = df_optimal[df_optimal['story_name'] == story_name]

            if len(optimal_row) > 0:
                optimal_iter = optimal_row['optimal_iteration'].values[0]

                if pd.notna(optimal_iter):
                    optimal_iter = int(optimal_iter)
                    col_name = f'abductive_iter_{optimal_iter}_kl'

                    if col_name in row_data.index and pd.notna(row_data[col_name]):
                        optimal_kl_values.append(row_data[col_name])

        # Compute quartiles
        if baseline_kl_values:
            baseline_q1 = np.percentile(baseline_kl_values, 25)
            baseline_q2 = np.percentile(baseline_kl_values, 50)
            baseline_q3 = np.percentile(baseline_kl_values, 75)
        else:
            baseline_q1 = baseline_q2 = baseline_q3 = None

        if optimal_kl_values:
            abd_q1 = np.percentile(optimal_kl_values, 25)
            abd_q2 = np.percentile(optimal_kl_values, 50)
            abd_q3 = np.percentile(optimal_kl_values, 75)
        else:
            abd_q1 = abd_q2 = abd_q3 = None

        baseline_q1_str = f"{baseline_q1:.4f}" if baseline_q1 is not None else "N/A"
        baseline_q2_str = f"{baseline_q2:.4f}" if baseline_q2 is not None else "N/A"
        baseline_q3_str = f"{baseline_q3:.4f}" if baseline_q3 is not None else "N/A"
        abd_q1_str = f"{abd_q1:.4f}" if abd_q1 is not None else "N/A"
        abd_q2_str = f"{abd_q2:.4f}" if abd_q2 is not None else "N/A"
        abd_q3_str = f"{abd_q3:.4f}" if abd_q3 is not None else "N/A"

        logger.info(f"  Baseline: Q1={baseline_q1_str}, Q2={baseline_q2_str}, Q3={baseline_q3_str}")
        logger.info(f"  Abductive: Q1={abd_q1_str}, Q2={abd_q2_str}, Q3={abd_q3_str}")
        rows.append({
            'model': model_short,
            'baseline_q1': round(baseline_q1, 4) if baseline_q1 is not None else None,
            'baseline_q2': round(baseline_q2, 4) if baseline_q2 is not None else None,
            'baseline_q3': round(baseline_q3, 4) if baseline_q3 is not None else None,
            'abd_q1': round(abd_q1, 4) if abd_q1 is not None else None,
            'abd_q2': round(abd_q2, 4) if abd_q2 is not None else None,
            'abd_q3': round(abd_q3, 4) if abd_q3 is not None else None,
            'num_stories': len(baseline_kl_values)
        })

    df_comparison = pd.DataFrame(rows)
    return df_comparison


# ==========================================================
# Main Function
# ==========================================================

def generate_kl_quartile_csv(
        models: list,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2",
        output_dir: str = "output_analysis/cross_model_summary"
):
    """
    Generate KL divergence quartile comparison CSV.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        divergence_dir: Divergence directory
        phase2_dir: Phase 2 directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("GENERATING KL DIVERGENCE QUARTILE CSV")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Problem: {problem_type}")

    # Create comparison DataFrame
    df = create_kl_quartile_comparison(models, problem_type, divergence_dir, phase2_dir)

    if df.empty:
        logger.error("No data to save")
        return

    # Save CSV
    output_path = Path(output_dir) / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "kl_divergence_quartiles.csv"
    df.to_csv(csv_file, index=False)

    logger.info(f"\n✓ Saved: {csv_file}")
    logger.info(f"  Models: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")

    logger.info("\n" + "=" * 60)
    logger.info("KL DIVERGENCE QUARTILE CSV COMPLETE")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate KL divergence quartile comparison CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create quartile comparison for all models
  python plot_scripts/create_kl_quartile_csv.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 \\
      --problem forward
        """
    )

    parser.add_argument('--models', nargs='+', required=True, help='List of model names')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--divergence-dir', default='output_analysis/divergences')
    parser.add_argument('--phase2-dir', default='output_analysis/phase2')
    parser.add_argument('--output-dir', default='output_analysis/cross_model_summary')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_kl_quartile_csv(
            models=args.models,
            problem_type=args.problem,
            divergence_dir=args.divergence_dir,
            phase2_dir=args.phase2_dir,
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
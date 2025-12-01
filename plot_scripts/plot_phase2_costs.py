"""
Phase 2 Cost Comparison Plotting Script

Creates box plots comparing LLM costs across models for Phase 2.
Generates 8 plots for different cost metrics.

Output: PNG files in output_analysis/plots/cross_model_comparison/{problem}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ==========================================================
# Plot Configuration
# ==========================================================

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12

# Colors for models
MODEL_COLORS = {
    'gpt-4o': '#3498DB',  # Blue
    'claude-sonnet-4-5': '#9B59B6',  # Purple
    'xai-grok-4-fast-reasoning': '#E67E22',  # Orange
    'bedrock-us.meta.llama4-maverick-17b-instruct-v1-0': '#27AE60'  # Green
}

BOX_ALPHA = 0.7
MEDIAN_COLOR = 'black'
MEDIAN_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_phase2_costs(
        models: list,
        problem_type: str,
        costs_dir: str = "output_analysis/costs/phase2"
) -> pd.DataFrame:
    """
    Load Phase 2 costs for all models.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        costs_dir: Costs directory

    Returns:
        Combined DataFrame
    """
    all_dfs = []

    for model in models:
        sanitized_model = model.replace("/", "-").replace(":", "-")
        csv_file = Path(costs_dir) / sanitized_model / problem_type / "llm_costs.csv"

        if not csv_file.exists():
            logger.warning(f"Cost file not found for {model}: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        df['model'] = model
        all_dfs.append(df)
        logger.info(f"Loaded {len(df)} stories for {model}")

    if not all_dfs:
        raise ValueError("No cost data found for any model")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\nTotal: {len(combined_df)} rows across {len(all_dfs)} models")

    return combined_df


# ==========================================================
# Plot Creation
# ==========================================================

def create_cost_boxplot(
        df: pd.DataFrame,
        metric: str,
        ylabel: str,
        output_path: Path,
        problem_type: str
):
    """
    Create box plot for a specific cost metric.

    Args:
        df: Combined DataFrame
        metric: Column name
        ylabel: Y-axis label
        output_path: Output path
        problem_type: Problem type
    """
    # Get unique models
    models = df['model'].unique().tolist()

    # Prepare data
    data_to_plot = []
    labels = []
    colors_list = []

    for model in models:
        model_data = df[df['model'] == model][metric].dropna().tolist()
        data_to_plot.append(model_data)

        # Clean label
        sanitized = model.replace("/", "-").replace(":", "-")
        if 'gpt' in model.lower():
            label = 'GPT-4o'
        elif 'claude' in model.lower():
            label = 'Claude'
        elif 'grok' in model.lower():
            label = 'Grok'
        elif 'llama' in model.lower():
            label = 'Llama 4'
        else:
            label = model[:15]

        labels.append(label)
        colors_list.append(MODEL_COLORS.get(sanitized, '#95A5A6'))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create box plot
    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=MEDIAN_WIDTH),
        boxprops=dict(alpha=BOX_ALPHA, linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5)
    )

    # Color boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(
        f'Phase 2 {ylabel} Distribution Across Models\n{problem_desc}',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Model', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path.name}")

    # Log stats
    for label, data in zip(labels, data_to_plot):
        logger.info(f"  {label}: Median={np.median(data):.0f}, Mean={np.mean(data):.0f}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_phase2_cost_plots(
        models: list,
        problem_type: str,
        costs_dir: str = "output_analysis/costs/phase2",
        output_dir: str = "output_analysis/plots/cross_model_comparison"
):
    """
    Create Phase 2 cost comparison plots.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        costs_dir: Costs directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("CREATING PHASE 2 COST PLOTS")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Problem: {problem_type}")

    # Load data
    df = load_phase2_costs(models, problem_type, costs_dir)

    # Output directory
    plot_dir = Path(output_dir) / problem_type
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create 8 plots
    metrics = [
        ('survey_input_tokens', 'Survey Input Tokens'),
        ('survey_output_tokens', 'Survey Output Tokens'),
        ('transform_input_tokens', 'Transform Input Tokens'),
        ('transform_output_tokens', 'Transform Output Tokens'),
        ('survey_input_tokens', 'Survey Total Tokens'),  # Will need to compute
        ('transform_input_tokens', 'Transform Total Tokens'),  # Will need to compute
        ('total_tokens', 'Total Tokens'),
        ('total_cost_usd', 'Cost (USD)')
    ]

    # Add computed columns for survey and transform totals
    df['survey_total_tokens'] = df['survey_input_tokens'] + df['survey_output_tokens']
    df['transform_total_tokens'] = df['transform_input_tokens'] + df['transform_output_tokens']

    # Update metrics list
    metrics = [
        ('survey_input_tokens', 'Survey Input Tokens'),
        ('survey_output_tokens', 'Survey Output Tokens'),
        ('transform_input_tokens', 'Transform Input Tokens'),
        ('transform_output_tokens', 'Transform Output Tokens'),
        ('survey_total_tokens', 'Survey Total Tokens'),
        ('transform_total_tokens', 'Transform Total Tokens'),
        ('total_tokens', 'Total Tokens'),
        ('total_cost_usd', 'Cost (USD)')
    ]

    for metric_col, metric_label in metrics:
        logger.info(f"\nCreating {metric_label} plot...")
        output_file = plot_dir / f"phase2_{metric_col}_boxplot.png"

        create_cost_boxplot(
            df=df,
            metric=metric_col,
            ylabel=metric_label,
            output_path=output_file,
            problem_type=problem_type
        )

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COST PLOTS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {plot_dir}")
    logger.info("Files created:")
    for metric_col, _ in metrics:
        logger.info(f"  - phase2_{metric_col}_boxplot.png")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create Phase 2 cost comparison plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 4 models for forward problem
  python plot_scripts/plot_phase2_costs.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 \\
      --problem forward
        """
    )

    parser.add_argument('--models', nargs='+', required=True, help='List of models')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--costs-dir', default='output_analysis/costs/phase2')
    parser.add_argument('--output-dir', default='output_analysis/plots/cross_model_comparison')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        create_phase2_cost_plots(
            models=args.models,
            problem_type=args.problem,
            costs_dir=args.costs_dir,
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
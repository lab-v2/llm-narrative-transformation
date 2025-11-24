"""
Network Centrality Cross-Model Comparison Script

Creates box plots comparing network centrality scores across different LLM models.
Shows distribution of centrality values for each model.

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

# Colors for different models
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

def load_centrality_for_model(
        model: str,
        problem_type: str,
        centrality_dir: str = "output_analysis/causal_graphs"
) -> pd.DataFrame:
    """
    Load network centrality CSV for a specific model.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        centrality_dir: Base directory

    Returns:
        DataFrame with centrality scores
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")
    csv_file = Path(centrality_dir) / sanitized_model / problem_type / "network_centrality.csv"

    if not csv_file.exists():
        logger.warning(f"Centrality file not found for {model}: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    df['model'] = model  # Add model column
    return df


def load_all_models(
        models: list,
        problem_type: str,
        centrality_dir: str = "output_analysis/causal_graphs"
) -> pd.DataFrame:
    """
    Load centrality data for all models.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        centrality_dir: Base directory

    Returns:
        Combined DataFrame with all models
    """
    all_dfs = []

    for model in models:
        df = load_centrality_for_model(model, problem_type, centrality_dir)
        if df is not None:
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} stories for {model}")

    if not all_dfs:
        raise ValueError("No centrality data found for any model")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\nTotal: {len(combined_df)} rows across {len(all_dfs)} models")

    return combined_df


# ==========================================================
# Box Plot Creation
# ==========================================================

def create_centrality_boxplot(
        df: pd.DataFrame,
        centrality_type: str,
        problem_type: str,
        output_path: Path
):
    """
    Create box plot comparing centrality across models.

    Args:
        df: Combined DataFrame with all models
        centrality_type: "segment", "feature", or "overall"
        problem_type: "forward" or "inverse"
        output_path: Path to save plot
    """
    # Get unique models (in order provided)
    models = df['model'].unique().tolist()

    # Column name based on centrality type
    col_name = f"{centrality_type}_centrality"

    # Prepare data for box plot
    data_to_plot = []
    labels = []
    colors_list = []

    for model in models:
        model_data = df[df['model'] == model][col_name].dropna().tolist()
        data_to_plot.append(model_data)

        # Clean model name for label
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
            label = model[:15]  # Truncate long names

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

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

    # Customize plot
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    centrality_label = centrality_type.capitalize()

    ax.set_title(
        f'{centrality_label} Centrality Distribution Across Models\n{problem_desc}',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Model', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Y-axis label depends on centrality type
    if centrality_type == "segment":
        ylabel = f'{centrality_label} Centrality\n(higher = few segments cover many features)'
    elif centrality_type == "feature":
        ylabel = f'{centrality_label} Centrality\n(higher = few features use many segments)'
    else:  # overall
        ylabel = f'{centrality_label} Centrality\n(higher = more centralized graph)'

    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    # Log statistics
    logger.info(f"\n{centrality_label} Centrality Statistics by Model:")
    for label, data in zip(labels, data_to_plot):
        logger.info(f"  {label}:")
        logger.info(f"    Median: {np.median(data):.4f}")
        logger.info(f"    Mean: {np.mean(data):.4f}")
        logger.info(f"    Std: {np.std(data):.4f}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_cross_model_centrality_plots(
        models: list,
        problem_type: str,
        centrality_dir: str = "output_analysis/causal_graphs",
        output_dir: str = "output_analysis/plots/cross_model_comparison"
):
    """
    Create centrality comparison plots across models.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        centrality_dir: Directory with centrality CSVs
        output_dir: Output directory for plots
    """
    logger.info("=" * 60)
    logger.info("CREATING CROSS-MODEL CENTRALITY PLOTS")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Problem: {problem_type}")

    # Load data for all models
    df_all = load_all_models(models, problem_type, centrality_dir)

    # Create output directory
    plot_dir = Path(output_dir) / problem_type
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create box plots for each centrality type
    centrality_types = ['segment', 'feature', 'overall']

    for cent_type in centrality_types:
        logger.info(f"\nCreating {cent_type} centrality plot...")
        output_file = plot_dir / f"{cent_type}_centrality_comparison.png"

        create_centrality_boxplot(
            df=df_all,
            centrality_type=cent_type,
            problem_type=problem_type,
            output_path=output_file
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PLOTS CREATED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {plot_dir}")
    logger.info("Files created:")
    for cent_type in centrality_types:
        logger.info(f"  - {cent_type}_centrality_comparison.png")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare network centrality across models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all models for forward problem
  python plot_scripts/compare_centrality_across_models.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning \\
      --problem forward

  # Compare for inverse problem
  python plot_scripts/compare_centrality_across_models.py \\
      --models gpt-4o claude-sonnet-4-5 \\
      --problem inverse
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of model names to compare (space-separated)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type: forward or inverse'
    )

    parser.add_argument(
        '--centrality-dir',
        type=str,
        default='output_analysis/causal_graphs',
        help='Centrality directory (default: output_analysis/causal_graphs)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/plots/cross_model_comparison',
        help='Output directory (default: output_analysis/plots/cross_model_comparison)'
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

    # Create plots
    try:
        create_cross_model_centrality_plots(
            models=args.models,
            problem_type=args.problem,
            centrality_dir=args.centrality_dir,
            output_dir=args.output_dir
        )

        logger.info("\n✓ Plotting complete!")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
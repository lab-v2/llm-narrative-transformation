"""
Comparison Plotting Script

Creates publication-quality bar plots comparing baseline vs abductive approaches.
Generates plots for Mean, Median, and Mode aggregate ratings across stories.

Output: PNG files in output_analysis/plots/{model}/{problem}/
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ==========================================================
# Plot Configuration (Publication Quality)
# ==========================================================

# Font sizes for publication
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# Colors (distinguishable in both color and B&W)
COLORS = {
    'original': '#2E86AB',      # Blue
    'baseline': '#2ECC71',      # Green
    'abductive_1': '#E74C3C',   # Red (lighter)
    'abductive_2': '#C0392B'    # Red (darker)
}

# Hatch patterns for B&W printing
HATCHES = {
    'original': '',  # Solid
    'baseline': '///',  # Diagonal lines
    'abductive_1': '...',  # Dots
    'abductive_2': 'xxx'  # Cross-hatch
}

# Bar configuration
BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Data Loading
# ==========================================================

def load_comparison_data(
        model: str,
        problem_type: str,
        baseline_dir: str = "output_analysis/baseline",
        phase2_dir: str = "output_analysis/phase2"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files needed for comparison.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        baseline_dir: Base directory for baseline results
        phase2_dir: Base directory for phase2 results

    Returns:
        Tuple of (original_df, baseline_df, abductive_iter1_df, abductive_iter2_df)
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Baseline files
    baseline_path = Path(baseline_dir) / sanitized_model / problem_type
    original_file = baseline_path / "original_ratings.csv"
    baseline_file = baseline_path / "baseline_transformed_ratings.csv"

    # Phase 2 files
    phase2_path = Path(phase2_dir) / sanitized_model / problem_type
    abductive_iter1_file = phase2_path / "transformed_ratings_iteration_0.csv"
    abductive_iter2_file = phase2_path / "transformed_ratings_iteration_1.csv"

    # Check all files exist
    files_to_check = {
        "Original ratings": original_file,
        "Baseline transformed": baseline_file,
        "Abductive iteration 1": abductive_iter1_file,
        "Abductive iteration 2": abductive_iter2_file
    }

    for name, path in files_to_check.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    # Load all files
    logger.info("Loading CSV files...")
    df_original = pd.read_csv(original_file)
    df_baseline = pd.read_csv(baseline_file)
    df_abductive_1 = pd.read_csv(abductive_iter1_file)
    df_abductive_2 = pd.read_csv(abductive_iter2_file)

    logger.info(f"✓ Loaded {len(df_original)} stories")

    return df_original, df_baseline, df_abductive_1, df_abductive_2


# ==========================================================
# Plot Creation
# ==========================================================
def create_comparison_plot(
        df_original: pd.DataFrame,
        df_baseline: pd.DataFrame,
        df_abductive_1: pd.DataFrame,
        df_abductive_2: pd.DataFrame,
        stat_type: str,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create comparison bar plot for a specific statistic.

    Args:
        df_original: Original ratings DataFrame
        df_baseline: Baseline transformed DataFrame
        df_abductive_1: Abductive iteration 1 DataFrame
        df_abductive_2: Abductive iteration 2 DataFrame
        stat_type: "Mean", "Median", or "Mode"
        model: Model name (for title)
        problem_type: "forward" or "inverse"
        output_path: Path to save plot
    """
    # Extract story names (ensure consistent order - alphabetical)
    stories = sorted(df_original['story_name'].tolist())
    n_stories = len(stories)

    # Extract statistic values for each method
    original_vals = []
    baseline_vals = []
    abductive_1_vals = []
    abductive_2_vals = []

    for story in stories:
        original_vals.append(df_original[df_original['story_name'] == story][stat_type].values[0])
        baseline_vals.append(df_baseline[df_baseline['story_name'] == story][stat_type].values[0])
        abductive_1_vals.append(df_abductive_1[df_abductive_1['story_name'] == story][stat_type].values[0])
        abductive_2_vals.append(df_abductive_2[df_abductive_2['story_name'] == story][stat_type].values[0])

    # Calculate figure size based on number of stories
    fig_width = max(16, n_stories * 0.6)  # Wider for more stories
    fig_height = 7

    # Create figure with extra space on right for legend
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate bar positions
    bar_width = 0.2  # Wider bars
    x = np.arange(n_stories)

    # Create bars for each method
    bars1 = ax.bar(x - 1.5 * bar_width, original_vals, bar_width,
                   label='Original',
                   color=COLORS['original'],
                   hatch=HATCHES['original'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    bars2 = ax.bar(x - 0.5 * bar_width, baseline_vals, bar_width,
                   label='Baseline',
                   color=COLORS['baseline'],
                   hatch=HATCHES['baseline'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    bars3 = ax.bar(x + 0.5 * bar_width, abductive_1_vals, bar_width,
                   label='Abductive-Guided (Iter 1)',
                   color=COLORS['abductive_1'],
                   hatch=HATCHES['abductive_1'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    bars4 = ax.bar(x + 1.5 * bar_width, abductive_2_vals, bar_width,
                   label='Abductive-Guided (Iter 2)',
                   color=COLORS['abductive_2'],
                   hatch=HATCHES['abductive_2'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    # Customize plot
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(
        f'{stat_type} Ratings Comparison: {problem_desc}\n({model})',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{stat_type} Rating (1-5 scale)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis ticks with better spacing
    ax.set_xticks(x)
    # Clean story names (remove underscores, make readable)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Set y-axis
    ax.set_ylim(0, 5.5)  # Rating scale is 1-5, give some margin
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid for easier reading (light gray, behind bars)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend - positioned outside plot area on the right
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1),  # Position outside on right
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=1,
        fancybox=False
    )

    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)  # Make room on right for legend

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()

# ==========================================================
# Main Plotting Function
# ==========================================================

def create_comparison_plots(
        model: str,
        problem_type: str,
        stat_type: str = "all",
        baseline_dir: str = "output_analysis/baseline",
        phase2_dir: str = "output_analysis/phase2",
        output_dir: str = "output_analysis/plots"
):
    """
    Create comparison plots.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        stat_type: "mean", "median", "mode", or "all"
        baseline_dir: Baseline results directory
        phase2_dir: Phase 2 results directory
        output_dir: Output directory for plots
    """
    logger.info("=" * 60)
    logger.info("CREATING COMPARISON PLOTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Statistics: {stat_type}")

    # Load data
    logger.info("\nLoading data...")
    df_original, df_baseline, df_abductive_1, df_abductive_2 = load_comparison_data(
        model=model,
        problem_type=problem_type,
        baseline_dir=baseline_dir,
        phase2_dir=phase2_dir
    )

    # Sanitize model name for output path
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type

    # Determine which plots to create
    if stat_type.lower() == "all":
        stats_to_plot = ["Mean", "Median", "Mode"]
    else:
        stats_to_plot = [stat_type.capitalize()]

    # Create plots
    logger.info("\nCreating plots...")
    for stat in stats_to_plot:
        output_file = plot_dir / f"{stat.lower()}_comparison.png"

        create_comparison_plot(
            df_original=df_original,
            df_baseline=df_baseline,
            df_abductive_1=df_abductive_1,
            df_abductive_2=df_abductive_2,
            stat_type=stat,
            model=model,
            problem_type=problem_type,
            output_path=output_file
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PLOTS CREATED")
    logger.info("=" * 60)
    logger.info(f"Plots saved to: {plot_dir}")
    logger.info(f"Number of plots: {len(stats_to_plot)}")
    for stat in stats_to_plot:
        logger.info(f"  - {stat.lower()}_comparison.png")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create comparison plots for baseline vs abductive approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all 3 plots (mean, median, mode) for gpt-4o forward
  python plot_scripts/create_comparison_plots.py --model gpt-4o --problem forward --stat all

  # Create only mean plot for Grok
  python plot_scripts/create_comparison_plots.py --model xai/grok-4-fast-reasoning --problem forward --stat mean

  # Create for inverse problem
  python plot_scripts/create_comparison_plots.py --model gpt-4o --problem inverse --stat all
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
        '--stat',
        type=str,
        choices=['mean', 'median', 'mode', 'all'],
        default='all',
        help='Which statistic to plot (default: all)'
    )

    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='output_analysis/baseline',
        help='Baseline results directory (default: output_analysis/baseline)'
    )

    parser.add_argument(
        '--phase2-dir',
        type=str,
        default='output_analysis/phase2',
        help='Phase 2 results directory (default: output_analysis/phase2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/plots',
        help='Output directory for plots (default: output_analysis/plots)'
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
        create_comparison_plots(
            model=args.model,
            problem_type=args.problem,
            stat_type=args.stat,
            baseline_dir=args.baseline_dir,
            phase2_dir=args.phase2_dir,
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
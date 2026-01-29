"""
Diagnosis Aggregates Plotting Script

Creates publication-quality bar plots comparing baseline vs optimal abductive iteration.
Uses correct statistics (Mean, Median, Mode) from optimal iteration.
Can exclude specific stories from plots.

Output: PNG files in output_analysis/plots/{model}/{problem}/
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple
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
LEGEND_FONT_SIZE = 12

# Colors
COLORS = {
    'original': '#2E86AB',        # Blue
    'baseline': '#2ECC71',        # Green
    'abductive_optimal': '#E74C3C'  # Red
}

# Hatches
HATCHES = {
    'original': '',
    'baseline': '///',
    'abductive_optimal': 'xxx'
}

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
    phase2_dir: str = "output_analysis/phase2",
    exclude_stories: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data for comparison plot.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        baseline_dir: Baseline analysis directory
        phase2_dir: Phase 2 analysis directory
        exclude_stories: List of story names to exclude

    Returns:
        Tuple of (original_df, baseline_df, optimal_df)
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load baseline data
    baseline_path = Path(baseline_dir) / sanitized_model / problem_type
    original_file = baseline_path / "original_ratings.csv"
    baseline_file = baseline_path / "baseline_transformed_ratings.csv"

    # Load Phase 2 optimal data
    phase2_path = Path(phase2_dir) / sanitized_model / problem_type
    optimal_file = phase2_path / "optimal_iterations.csv"

    # Check files exist
    if not original_file.exists():
        raise FileNotFoundError(f"Original ratings not found: {original_file}")
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline ratings not found: {baseline_file}")
    if not optimal_file.exists():
        raise FileNotFoundError(f"Optimal iterations not found: {optimal_file}")

    # Load files
    df_original = pd.read_csv(original_file)
    df_baseline = pd.read_csv(baseline_file)
    df_optimal = pd.read_csv(optimal_file)

    logger.info(f"Loaded data for {len(df_original)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df_original = df_original[~df_original['story_name'].isin(exclude_stories)]
        df_baseline = df_baseline[~df_baseline['story_name'].isin(exclude_stories)]
        df_optimal = df_optimal[~df_optimal['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df_original)}")

    return df_original, df_baseline, df_optimal


# ==========================================================
# Plot Creation
# ==========================================================

def create_comparison_plot(
    df_original: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_optimal: pd.DataFrame,
    stat_type: str,
    model: str,
    problem_type: str,
    output_path: Path
):
    """
    Create comparison bar plot using optimal iteration.

    Args:
        df_original: Original ratings
        df_baseline: Baseline ratings
        df_optimal: Optimal iteration ratings
        stat_type: "Mean", "Median", or "Mode"
        model: Model name
        problem_type: Problem type
        output_path: Output path
    """
    # Get stories (alphabetically ordered)
    stories = sorted(df_original['story_name'].tolist())
    n_stories = len(stories)

    # Determine column names based on stat_type
    original_col = stat_type  # "Mean", "Median", or "Mode"
    baseline_col = stat_type
    optimal_col = f'optimal_{stat_type.lower()}'  # "optimal_mean", "optimal_median", or "optimal_mode"

    # Extract values
    original_vals = []
    baseline_vals = []
    optimal_vals = []

    for story in stories:
        original_vals.append(df_original[df_original['story_name'] == story][original_col].values[0])
        baseline_vals.append(df_baseline[df_baseline['story_name'] == story][baseline_col].values[0])
        optimal_vals.append(df_optimal[df_optimal['story_name'] == story][optimal_col].values[0])

    # Figure size
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Bar positions
    bar_width = 0.25
    x = np.arange(n_stories)

    # Create bars
    ax.bar(x - bar_width, original_vals, bar_width,
           label='Original',
           color=COLORS['original'],
           hatch=HATCHES['original'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    ax.bar(x, baseline_vals, bar_width,
           label='Baseline',
           color=COLORS['baseline'],
           hatch=HATCHES['baseline'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    ax.bar(x + bar_width, optimal_vals, bar_width,
           label='Abductive-Guided (Optimal)',
           color=COLORS['abductive_optimal'],
           hatch=HATCHES['abductive_optimal'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    # ax.set_title(
    #     f'{stat_type} Ratings Comparison: {problem_desc}\n({model})',
    #     fontsize=TITLE_FONT_SIZE,
    #     fontweight='bold',
    #     pad=20
    # )

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{stat_type} Rating (1-5 scale)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # X-axis
    ax.set_xticks(x)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Y-axis
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    # ax.legend(
    #     loc='upper left',
    #     bbox_to_anchor=(1.01, 1),
    #     fontsize=LEGEND_FONT_SIZE,
    #     framealpha=1.0,
    #     edgecolor='black',
    #     borderpad=1,
    #     fancybox=False
    # )


    # plt.subplots_adjust(right=0.85)

    # Legend
    ax.legend(
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=1,
        fancybox=False
    )

    plt.subplots_adjust(right=0.95)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_optimal_comparison_plots(
    model: str,
    problem_type: str,
    stat_type: str = "all",
    baseline_dir: str = "output_analysis/baseline",
    phase2_dir: str = "output_analysis/phase2",
    output_dir: str = "output_analysis/plots",
    exclude_stories: list = None
):
    """
    Create comparison plots using optimal iteration.

    Args:
        model: Model name
        problem_type: Problem type
        stat_type: "mean", "median", "mode", or "all"
        baseline_dir: Baseline directory
        phase2_dir: Phase 2 directory
        output_dir: Output directory
        exclude_stories: Stories to exclude
    """
    logger.info("=" * 60)
    logger.info("CREATING OPTIMAL COMPARISON PLOTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df_original, df_baseline, df_optimal = load_comparison_data(
        model, problem_type, baseline_dir, phase2_dir, exclude_stories
    )

    # Determine which stats to plot
    if stat_type.lower() == "all":
        stats_to_plot = ["Mean", "Median", "Mode"]
    else:
        stats_to_plot = [stat_type.capitalize()]

    # Create plots
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type

    for stat in stats_to_plot:
        output_file = plot_dir / f"{stat.lower()}_comparison_optimal.png"

        create_comparison_plot(
            df_original, df_baseline, df_optimal,
            stat, model, problem_type, output_file
        )

    logger.info("\n" + "=" * 60)
    logger.info("PLOTS CREATED")
    logger.info("=" * 60)
    logger.info(f"Output: {plot_dir}")
    for stat in stats_to_plot:
        logger.info(f"  - {stat.lower()}_comparison_optimal.png")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create comparison plots with optimal iteration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all plots with optimal iteration
  python plot_scripts/plot_diagnosis_aggregates.py --model gpt-4o --problem forward --stat all
  
  # Exclude specific stories
  python plot_scripts/plot_diagnosis_aggregates.py \\
      --model gpt-4o --problem forward --stat all \\
      --exclude Back_To_The_Wall Community_Time
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--stat', default='all', choices=['mean', 'median', 'mode', 'all'])
    parser.add_argument('--baseline-dir', default='output_analysis/baseline')
    parser.add_argument('--phase2-dir', default='output_analysis/phase2')
    parser.add_argument('--output-dir', default='output_analysis/plots')
    parser.add_argument('--exclude', nargs='*', dest='exclude_stories', help='Stories to exclude')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        create_optimal_comparison_plots(
            model=args.model,
            problem_type=args.problem,
            stat_type=args.stat,
            baseline_dir=args.baseline_dir,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir,
            exclude_stories=args.exclude_stories
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
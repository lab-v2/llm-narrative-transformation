"""
Hyperparameter Comparison Plotting Script

Creates publication-quality line plots comparing baseline vs optimal abductive iteration
across different hyperparameter values (1-20).

Uses median of median ratings across all stories for each hyperparameter.
Separate plots for forward and inverse problems.

Output: PNG files in output_analysis/plots/{model}/{problem}/
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Plot Configuration
# ==========================================================

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22

# Colors
COLORS = {
    'original': '#2E86AB',        # Blue
    'baseline': '#2ECC71',        # Green
    'abductive_optimal': '#E74C3C'  # Red
}

BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Data Loading
# ==========================================================

def load_baseline_data(
    model: str,
    problem_type: str,
    baseline_dir: str = "output_analysis/baseline",
    exclude_stories: list = None
) -> Tuple[float, float]:
    """
    Load baseline data and compute median of medians and original median.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        baseline_dir: Baseline analysis directory
        exclude_stories: List of story names to exclude

    Returns:
        Tuple of (original_median, baseline_median)
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    baseline_path = Path(baseline_dir) / sanitized_model / problem_type
    original_file = baseline_path / "original_ratings.csv"
    baseline_file = baseline_path / "baseline_transformed_ratings.csv"

    # Check files exist
    if not original_file.exists():
        raise FileNotFoundError(f"Original ratings not found: {original_file}")
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline ratings not found: {baseline_file}")

    # Load files
    df_original = pd.read_csv(original_file)
    df_baseline = pd.read_csv(baseline_file)

    # Exclude stories if specified
    if exclude_stories:
        df_original = df_original[~df_original['story_name'].isin(exclude_stories)]
        df_baseline = df_baseline[~df_baseline['story_name'].isin(exclude_stories)]

    # Compute median of medians
    original_median = df_original['Median'].median()
    baseline_median = df_baseline['Median'].median()

    logger.info(f"Baseline - Original median: {original_median:.3f}, Baseline median: {baseline_median:.3f}")

    return original_median, baseline_median


def load_optimal_data_for_hyperparameter(
    model: str,
    problem_type: str,
    hyperparameter: int,
    phase2_base_dir: str = "output_analysis",
    exclude_stories: list = None
) -> float:
    """
    Load optimal iteration data for a specific hyperparameter and compute median of medians.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        hyperparameter: Hyperparameter value (1, 2, 5, 10, 20)
        phase2_base_dir: Base directory for hyperparameter folders
        exclude_stories: List of story names to exclude

    Returns:
        Median of medians for optimal iteration
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    phase2_path = Path(phase2_base_dir) / f"hyperparameter-{hyperparameter}/phase2" / sanitized_model / problem_type
    optimal_file = phase2_path / "optimal_iterations.csv"

    # Check file exists
    if not optimal_file.exists():
        raise FileNotFoundError(f"Optimal iterations not found: {optimal_file}")

    # Load file
    df_optimal = pd.read_csv(optimal_file)

    # Exclude stories if specified
    if exclude_stories:
        df_optimal = df_optimal[~df_optimal['story_name'].isin(exclude_stories)]

    # Compute median of medians
    optimal_median = df_optimal['optimal_median'].median()

    logger.info(f"Hyperparameter {hyperparameter} - Optimal median: {optimal_median:.3f}")

    return optimal_median


# ==========================================================
# Plot Creation
# ==========================================================

def create_hyperparameter_comparison_plot(
    hyperparameters: list,
    original_median: float,
    baseline_median: float,
    optimal_medians: Dict[int, float],
    model: str,
    problem_type: str,
    output_path: Path
):
    """
    Create hyperparameter comparison line plot.

    Args:
        hyperparameters: List of hyperparameter values (1, 2, 5, 10, 20)
        original_median: Original median rating
        baseline_median: Baseline median rating
        optimal_medians: Dict mapping hyperparameter -> optimal median
        model: Model name
        problem_type: Problem type ("forward" or "inverse")
        output_path: Output path
    """
    # Flip ratings for forward (C->I) so 5 is always most collectivistic/individualistic
    def flip_rating(val):
        if problem_type == "forward":
            return 6 - val
        return val

    original_median = flip_rating(original_median)
    baseline_median = flip_rating(baseline_median)
    optimal_medians = {hp: flip_rating(val) for hp, val in optimal_medians.items()}

    # Prepare data for x-axis 1-20
    x_full = list(range(1, 21))

    # Original and baseline are constant across all x values
    original_vals = [original_median] * len(x_full)
    baseline_vals = [baseline_median] * len(x_full)

    # Abductive: actual values for 1-5, then flat line (value at k=5) from 6-20
    optimal_vals = []
    for x in x_full:
        if x in optimal_medians:
            optimal_vals.append(optimal_medians[x])
        else:
            # For x > 5, use the value at k=5
            optimal_vals.append(optimal_medians.get(5, optimal_medians[max(optimal_medians.keys())]))

    # Figure size
    fig_width = 12
    fig_height = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Line width
    line_width = 4.0

    # Plot lines with different styles
    ax.plot(x_full, original_vals,
            label='Original',
            color=COLORS['original'],
            linewidth=line_width,
            linestyle='-',  # solid
            alpha=0.8)

    ax.plot(x_full, baseline_vals,
            label='Baseline',
            color=COLORS['baseline'],
            linewidth=line_width,
            linestyle='--',  # dashed
            alpha=0.8)

    ax.plot(x_full, optimal_vals,
            label='Abduction-Guided',
            color=COLORS['abductive_optimal'],
            linewidth=line_width,
            linestyle=':',  # dotted
            alpha=0.8)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"

    ax.set_xlabel('Number of features', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Median Rating', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # X-axis from 1 to 20
    ax.set_xticks(range(1, 21))
    ax.set_xlim(1, 20)
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)

    # Y-axis
    ax.set_ylim(0, 6.0)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='upper center',
        ncol=6,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=0.4,
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

def create_hyperparameter_comparison_plots(
    model: str,
    problem_type: str,
    hyperparameters: list = None,
    baseline_dir: str = "output_analysis/baseline",
    phase2_base_dir: str = "output_analysis",
    output_dir: str = "output_analysis/plots",
    exclude_stories: list = None
):
    """
    Create hyperparameter comparison plots using optimal iteration.

    Args:
        model: Model name
        problem_type: Problem type ("forward" or "inverse")
        hyperparameters: List of hyperparameter values to compare
        baseline_dir: Baseline directory
        phase2_base_dir: Base directory for hyperparameter folders
        output_dir: Output directory
        exclude_stories: Stories to exclude
    """
    if hyperparameters is None:
        hyperparameters = [1, 2, 5, 10, 20]

    logger.info("=" * 60)
    logger.info("CREATING HYPERPARAMETER COMPARISON PLOTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Hyperparameters: {hyperparameters}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load baseline data (original and baseline)
    original_median, baseline_median = load_baseline_data(
        model, problem_type, baseline_dir, exclude_stories
    )

    # Load optimal data for each hyperparameter
    optimal_medians = {}
    for hp in hyperparameters:
        try:
            optimal_median = load_optimal_data_for_hyperparameter(
                model, problem_type, hp, phase2_base_dir, exclude_stories
            )
            optimal_medians[hp] = optimal_median
        except FileNotFoundError as e:
            logger.warning(f"Skipping hyperparameter {hp}: {e}")
            continue

    if not optimal_medians:
        logger.error("No optimal data found for any hyperparameter!")
        return

    # Create plot
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type
    output_file = plot_dir / "hyperparameter_median_comparison.png"

    create_hyperparameter_comparison_plot(
        hyperparameters, original_median, baseline_median, optimal_medians,
        model, problem_type, output_file
    )

    logger.info("\n" + "=" * 60)
    logger.info("PLOT CREATED")
    logger.info("=" * 60)
    logger.info(f"Output: {output_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create hyperparameter comparison plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plot with default hyperparameters (1, 2, 5, 10, 20)
  python plot_scripts/plot_hyperparameter_comparison.py --model gpt-4o --problem forward
  
  # Custom hyperparameter values
  python plot_scripts/plot_hyperparameter_comparison.py \\
      --model gpt-4o --problem forward \\
      --hyperparameters 1 2 5 10 20
  
  # Short syntax with -k
  python plot_scripts/plot_hyperparameter_comparison.py \\
      --model gpt-4o --problem forward \\
      -k 1 2 5 10
  
  # Exclude specific stories
  python plot_scripts/plot_hyperparameter_comparison.py \\
      --model gpt-4o --problem forward \\
      --exclude Back_To_The_Wall Community_Time
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--hyperparameters', '-k', nargs='+', type=int, default=[1, 2, 5, 10, 20],
                        help='Hyperparameter values to compare (default: 1 2 5 10 20)')
    parser.add_argument('--baseline-dir', default='output_analysis/baseline')
    parser.add_argument('--phase2-base-dir', default='output_analysis')
    parser.add_argument('--output-dir', default='output_analysis/plots')
    parser.add_argument('--exclude', nargs='*', dest='exclude_stories', help='Stories to exclude')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        create_hyperparameter_comparison_plots(
            model=args.model,
            problem_type=args.problem,
            hyperparameters=args.hyperparameters,
            baseline_dir=args.baseline_dir,
            phase2_base_dir=args.phase2_base_dir,
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
"""
JS Divergence Plot Script (Optimal Iteration Version)

Creates publication-quality plots showing JS divergence between
original and transformed stories using optimal iteration.

Can exclude specific stories from plots.

Lower JS divergence = better coherence with original (less drastic change).

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
COLOR_BASELINE = '#2ECC71'  # Green
COLOR_ABDUCTIVE = '#E74C3C'  # Red

# Hatches
HATCH_BASELINE = '///'
HATCH_ABDUCTIVE = 'xxx'

BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Data Loading
# ==========================================================

def load_js_data(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2",
        exclude_stories: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load JS divergence data.

    Args:
        model: Model name
        problem_type: Problem type
        divergence_dir: Divergence directory
        phase2_dir: Phase 2 directory (for optimal iterations)
        exclude_stories: Stories to exclude

    Returns:
        Tuple of (js_df, optimal_df)
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load JS divergence
    js_file = Path(divergence_dir) / sanitized_model / problem_type / "js_divergence.csv"
    if not js_file.exists():
        raise FileNotFoundError(f"JS divergence not found: {js_file}")

    # Load optimal iterations
    optimal_file = Path(phase2_dir) / sanitized_model / problem_type / "optimal_iterations.csv"
    if not optimal_file.exists():
        raise FileNotFoundError(f"Optimal iterations not found: {optimal_file}")

    df_js = pd.read_csv(js_file)
    df_optimal = pd.read_csv(optimal_file)

    logger.info(f"Loaded data for {len(df_js)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df_js = df_js[~df_js['story_name'].isin(exclude_stories)]
        df_optimal = df_optimal[~df_optimal['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df_js)}")

    return df_js, df_optimal


# ==========================================================
# Plot Creation
# ==========================================================

def create_optimal_js_plot(
        df_js: pd.DataFrame,
        df_optimal: pd.DataFrame,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create JS divergence plot using optimal iteration.

    Args:
        df_js: JS divergence DataFrame
        df_optimal: Optimal iterations DataFrame
        model: Model name
        problem_type: Problem type
        output_path: Output path
    """
    # Get stories (alphabetically ordered)
    stories = sorted(df_js['story_name'].tolist())
    n_stories = len(stories)

    # Extract values
    baseline_vals = []
    optimal_vals = []

    for story in stories:
        # Baseline JS
        baseline_js = df_js[df_js['story_name'] == story]['baseline_js'].values[0]
        baseline_vals.append(baseline_js)

        # Optimal iteration JS
        optimal_iter = df_optimal[df_optimal['story_name'] == story]['optimal_iteration'].values[0]

        if pd.notna(optimal_iter):
            optimal_iter = int(optimal_iter)
            col_name = f'abductive_iter_{optimal_iter}_js'

            if col_name in df_js.columns:
                optimal_js = df_js[df_js['story_name'] == story][col_name].values[0]
                optimal_vals.append(optimal_js)
            else:
                logger.warning(f"Column {col_name} not found for {story}")
                optimal_vals.append(np.nan)
        else:
            optimal_vals.append(np.nan)

    # Figure size
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Bar positions
    bar_width = 0.35
    x = np.arange(n_stories)

    # Create bars
    ax.bar(x - bar_width / 2, baseline_vals, bar_width,
           label='Baseline',
           color=COLOR_BASELINE,
           hatch=HATCH_BASELINE,
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    ax.bar(x + bar_width / 2, optimal_vals, bar_width,
           label='Abductive-Guided (Optimal)',
           color=COLOR_ABDUCTIVE,
           hatch=HATCH_ABDUCTIVE,
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(
        f'JS Divergence: {problem_desc}\n({model})',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('JS Divergence (0.0 - 1.0, lower = better)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # X-axis
    ax.set_xticks(x)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Y-axis
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=1,
        fancybox=False
    )

    plt.subplots_adjust(right=0.85)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_optimal_js_plot_main(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2",
        output_dir: str = "output_analysis/plots",
        exclude_stories: list = None
):
    """
    Create optimal JS divergence plot.
    """
    logger.info("=" * 60)
    logger.info("CREATING OPTIMAL JS DIVERGENCE PLOT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df_js, df_optimal = load_js_data(
        model, problem_type, divergence_dir, phase2_dir, exclude_stories
    )

    # Create plot
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type
    output_file = plot_dir / "js_divergence_optimal.png"

    create_optimal_js_plot(
        df_js, df_optimal, model, problem_type, output_file
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
        description='Create optimal JS divergence plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plot with optimal iteration
  python plot_scripts/plot_js_divergence_optimal.py --model gpt-4o --problem forward

  # Exclude specific stories
  python plot_scripts/plot_js_divergence_optimal.py \\
      --model gpt-4o --problem forward \\
      --exclude Back_To_The_Wall Fleabags
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--divergence-dir', default='output_analysis/divergences')
    parser.add_argument('--phase2-dir', default='output_analysis/phase2')
    parser.add_argument('--output-dir', default='output_analysis/plots')
    parser.add_argument('--exclude', nargs='*', dest='exclude_stories', help='Stories to exclude')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        create_optimal_js_plot_main(
            model=args.model,
            problem_type=args.problem,
            divergence_dir=args.divergence_dir,
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
"""
KL Divergence Box Plot Script (Optimal Iteration Version)

Creates publication-quality box plots showing KL(transformed || original) divergence
distribution using optimal iteration.

Can exclude specific stories from plots.

Lower KL divergence = better coherence with original.

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

BOX_ALPHA = 0.7
MEDIAN_COLOR = 'black'
MEDIAN_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_kl_data(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2",
        exclude_stories: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load KL divergence data.

    Args:
        model: Model name
        problem_type: Problem type
        divergence_dir: Divergence directory
        phase2_dir: Phase 2 directory (for optimal iterations)
        exclude_stories: Stories to exclude

    Returns:
        Tuple of (kl_df, optimal_df)
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load KL divergence
    kl_file = Path(divergence_dir) / sanitized_model / problem_type / "kl_divergence_transformed_original.csv"
    if not kl_file.exists():
        raise FileNotFoundError(f"KL divergence not found: {kl_file}")

    # Load optimal iterations
    optimal_file = Path(phase2_dir) / sanitized_model / problem_type / "optimal_iterations.csv"
    if not optimal_file.exists():
        raise FileNotFoundError(f"Optimal iterations not found: {optimal_file}")

    df_kl = pd.read_csv(kl_file)
    df_optimal = pd.read_csv(optimal_file)

    logger.info(f"Loaded data for {len(df_kl)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df_kl = df_kl[~df_kl['story_name'].isin(exclude_stories)]
        df_optimal = df_optimal[~df_optimal['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df_kl)}")

    return df_kl, df_optimal


# ==========================================================
# Plot Creation
# ==========================================================

def create_optimal_kl_boxplot(
        df_kl: pd.DataFrame,
        df_optimal: pd.DataFrame,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create KL divergence box plot using optimal iteration.

    Args:
        df_kl: KL divergence DataFrame
        df_optimal: Optimal iterations DataFrame
        model: Model name
        problem_type: Problem type
        output_path: Output path
    """
    # Collect baseline values
    baseline_vals = df_kl['baseline_kl'].dropna().tolist()

    # Collect optimal iteration values
    optimal_vals = []
    for _, row in df_kl.iterrows():
        story_name = row['story_name']

        # Find optimal iteration for this story
        optimal_row = df_optimal[df_optimal['story_name'] == story_name]

        if len(optimal_row) > 0:
            optimal_iter = optimal_row['optimal_iteration'].values[0]

            if pd.notna(optimal_iter):
                optimal_iter = int(optimal_iter)
                col_name = f'abductive_iter_{optimal_iter}_kl'

                if col_name in row.index and pd.notna(row[col_name]):
                    optimal_vals.append(row[col_name])

    # Prepare data for box plot
    data_to_plot = [baseline_vals, optimal_vals]
    labels = ['Baseline', 'Abductive-Guided\n(Optimal)']
    colors = [COLOR_BASELINE, COLOR_ABDUCTIVE]

    logger.info(f"Baseline: {len(baseline_vals)} values")
    logger.info(f"Optimal: {len(optimal_vals)} values")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

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
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(
        f'KL Divergence Distribution: {problem_desc}\n({model})',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Method', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('KL(transformed || original)\n(lower = better)', fontsize=LABEL_FONT_SIZE, fontweight='bold',
                  labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    # Log statistics
    logger.info("\nBox Plot Statistics:")
    for label, data in zip(labels, data_to_plot):
        logger.info(f"  {label}:")
        logger.info(f"    Median: {np.median(data):.4f}")
        logger.info(f"    Mean: {np.mean(data):.4f}")
        logger.info(f"    Std: {np.std(data):.4f}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_optimal_kl_boxplot_main(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        phase2_dir: str = "output_analysis/phase2",
        output_dir: str = "output_analysis/plots",
        exclude_stories: list = None
):
    """
    Create optimal KL divergence box plot.
    """
    logger.info("=" * 60)
    logger.info("CREATING OPTIMAL KL DIVERGENCE BOX PLOT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df_kl, df_optimal = load_kl_data(
        model, problem_type, divergence_dir, phase2_dir, exclude_stories
    )

    # Create plot
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type
    output_file = plot_dir / "kl_divergence_boxplot_optimal.png"

    create_optimal_kl_boxplot(
        df_kl, df_optimal, model, problem_type, output_file
    )

    logger.info("\n" + "=" * 60)
    logger.info("BOX PLOT CREATED")
    logger.info("=" * 60)
    logger.info(f"Output: {output_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create optimal KL divergence box plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create box plot with optimal iteration
  python plot_scripts/plot_kl_boxplot_optimal.py --model gpt-4o --problem forward

  # Exclude specific stories
  python plot_scripts/plot_kl_boxplot_optimal.py \\
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
        create_optimal_kl_boxplot_main(
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
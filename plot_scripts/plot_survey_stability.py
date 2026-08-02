"""
Survey Stability Plot Script

Creates error bar plot showing mean and standard deviation of survey ratings
across multiple runs for each story.

Input: CSV file from survey_stability_experiment.py with columns:
       story_id, run_1, run_2, ..., run_N

Output: PNG files in output_analysis/
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

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 20

# Color
COLOR_BARS = '#3498DB'  # Blue
ERROR_COLOR = '#2C3E50'  # Dark gray
BAR_ALPHA = 0.7
ERROR_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_stability_data(csv_file: str) -> pd.DataFrame:
    """
    Load survey stability results CSV.

    Args:
        csv_file: Path to survey_stability_results.csv

    Returns:
        DataFrame with columns: story_id, run_1, run_2, ..., run_N
    """
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded stability data: {len(df)} stories, {len(df.columns) - 1} runs")
    return df


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, and other statistics for each story.

    Args:
        df: DataFrame with story_id and run columns

    Returns:
        DataFrame with statistics
    """
    stats = []

    for _, row in df.iterrows():
        story_id = row['story_id']

        # Extract all run values (all columns except story_id)
        runs = [v for k, v in row.items() if k != 'story_id']
        runs = [r for r in runs if pd.notna(r)]  # Remove NaN values

        if len(runs) == 0:
            logger.warning(f"No valid runs for {story_id}")
            continue

        mean = np.mean(runs)
        std = np.std(runs)
        min_val = np.min(runs)
        max_val = np.max(runs)
        num_runs = len(runs)

        stats.append({
            'story_id': story_id,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'num_runs': num_runs
        })

    df_stats = pd.DataFrame(stats)
    logger.info(f"\nComputed statistics:")
    logger.info(df_stats.to_string(index=False))

    return df_stats


# ==========================================================
# Plot Creation
# ==========================================================

def create_stability_plot(
        csv_file: str,
        output_file: str
):
    """
    Create survey stability error bar plot.

    Args:
        csv_file: Path to survey_stability_results.csv
        output_file: Path to save PNG
    """
    # Load data
    df = load_stability_data(csv_file)

    # Compute statistics
    df_stats = compute_statistics(df)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for plotting
    story_ids = df_stats['story_id'].tolist()
    means = df_stats['mean'].tolist()
    stds = df_stats['std'].tolist()

    # Create bar plot with error bars
    x_pos = np.arange(len(story_ids))
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        color=COLOR_BARS,
        alpha=BAR_ALPHA,
        capsize=8,
        error_kw={'elinewidth': ERROR_WIDTH, 'ecolor': ERROR_COLOR, 'alpha': 0.8},
        edgecolor='black',
        linewidth=1.5,
        width=0.6
    )

    # Customize axes
    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Rating (Mean ± Std)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis - convert S_1 to S₁ format
    formatted_labels = [s.replace('S_', '$S_{') + '}$' if 'S_' in s else s for s in story_ids]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(formatted_labels, fontsize=TICK_FONT_SIZE)

    # Set y-axis
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # # Add value labels on bars (optional)
    # for i, (mean, std) in enumerate(zip(means, stds)):
    #     ax.text(
    #         i, mean + std + 0.15,
    #         f'{mean:.2f}',
    #         ha='center', va='bottom',
    #         fontsize=10,
    #         fontweight='bold'
    #     )

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create survey stability error bar plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plot from stability results
  python plot_scripts/plot_survey_stability.py \\
      --input output_analysis/survey_stability_results.csv \\
      --output output_analysis/survey_stability_plot.png

  # With verbose logging
  python plot_scripts/plot_survey_stability.py \\
      --input output_analysis/survey_stability_results.csv \\
      --output output_analysis/survey_stability_plot.png \\
      --verbose
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file (survey_stability_results.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output_analysis/survey_stability_plot.png',
        help='Output PNG file (default: output_analysis/survey_stability_plot.png)'
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

    logger.info("=" * 60)
    logger.info("CREATING SURVEY STABILITY PLOT")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    try:
        create_stability_plot(
            csv_file=args.input,
            output_file=args.output
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

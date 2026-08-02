"""
N-Segments vs Runtime Plot Script

Creates line plot showing relationship between n_segments and mean runtime
across all stories.

Input: CSV file from extract_runtime_metrics.py with columns:
       hyperparameter, story_name, n_segments, k_features, total_input_tokens,
       total_output_tokens, runtime_seconds

Output: PNG file in output_analysis_runtime/
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
LINE_COLOR = '#3498DB'  # Blue
LINE_WIDTH = 2.5
MARKER_SIZE = 8
MARKER_ALPHA = 0.8


# ==========================================================
# Data Loading and Processing
# ==========================================================

def load_runtime_data(csv_file: str) -> pd.DataFrame:
    """
    Load runtime metrics results CSV.

    Args:
        csv_file: Path to runtime_metrics.csv

    Returns:
        DataFrame with runtime metrics
    """
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded runtime data: {len(df)} stories")
    return df


def compute_mean_runtime_by_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by n_segments and compute mean runtime for each segment value.

    Args:
        df: DataFrame with n_segments and runtime_seconds columns

    Returns:
        DataFrame with n_segments and mean_runtime
    """
    grouped = df.groupby('n_segments')['runtime_seconds'].mean().reset_index()
    grouped.columns = ['n_segments', 'mean_runtime']
    grouped = grouped.sort_values('n_segments')

    logger.info(f"\nGrouped statistics by n_segments:")
    logger.info(grouped.to_string(index=False))

    return grouped


# ==========================================================
# Plot Creation
# ==========================================================

def create_n_segments_runtime_plot(
        csv_file: str,
        output_file: str
):
    """
    Create n_segments vs runtime line plot.

    Args:
        csv_file: Path to runtime_metrics.csv
        output_file: Path to save PNG
    """
    # Load data
    df = load_runtime_data(csv_file)

    # Compute mean runtime by n_segments
    df_grouped = compute_mean_runtime_by_segments(df)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for plotting
    n_segments = df_grouped['n_segments'].tolist()
    mean_runtime = df_grouped['mean_runtime'].tolist()

    # Create line plot
    ax.plot(
        n_segments,
        mean_runtime,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        marker='o',
        markersize=MARKER_SIZE,
        alpha=MARKER_ALPHA,
        markerfacecolor=LINE_COLOR,
        markeredgecolor='black',
        markeredgewidth=1.5
    )

    # Customize axes
    ax.set_xlabel('Number of Chunks', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mean Runtime (seconds)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis with constant gap of 2, up to 30
    x_ticks = list(range(0, 40, 2))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=TICK_FONT_SIZE)
    ax.set_xlim(-1, 41)

    # Set y-axis
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

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
        description='Create n_segments vs runtime line plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plot from runtime metrics
  python plot_scripts/plot_n_segments_runtime.py \\
      --input output_analysis_runtime/runtime_metrics.csv \\
      --output output_analysis_runtime/n_segments_vs_runtime.png

  # With verbose logging
  python plot_scripts/plot_n_segments_runtime.py \\
      --input output_analysis_runtime/runtime_metrics.csv \\
      --output output_analysis_runtime/n_segments_vs_runtime.png \\
      --verbose
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file (runtime_metrics.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output_analysis_runtime/n_segments_vs_runtime.png',
        help='Output PNG file (default: output_analysis_runtime/n_segments_vs_runtime.png)'
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
    logger.info("CREATING N_SEGMENTS VS RUNTIME PLOT")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    try:
        create_n_segments_runtime_plot(
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
"""
Divergence and Similarity Plotting Script

Creates publication-quality bar plots comparing baseline vs abductive approaches
using cosine similarity and JS divergence metrics.

Output: PNG files in plots/
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
# Plot Configuration (Publication Quality)
# ==========================================================

# Font sizes for publication
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# Colors (distinguishable in both color and B&W)
COLORS = {
    'baseline': '#2ECC71',  # Green
    'abduction': '#E74C3C',  # Red
}

# Hatch patterns for B&W printing
HATCHES = {
    'baseline': '///',  # Diagonal lines
    'abduction': '...',  # Dots
}

# Bar configuration
BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Data Loading
# ==========================================================

def load_cosine_data(cosine_file: str) -> pd.DataFrame:
    """Load cosine similarity CSV file."""
    if not Path(cosine_file).exists():
        raise FileNotFoundError(f"Cosine similarity file not found: {cosine_file}")

    logger.info(f"Loading cosine similarity: {cosine_file}")
    df = pd.read_csv(cosine_file)
    logger.info(f"✓ Loaded {len(df)} stories")
    return df


def load_divergence_data(divergence_file: str) -> pd.DataFrame:
    """Load JS divergence CSV file."""
    if not Path(divergence_file).exists():
        raise FileNotFoundError(f"JS divergence file not found: {divergence_file}")

    logger.info(f"Loading JS divergence: {divergence_file}")
    df = pd.read_csv(divergence_file)
    logger.info(f"✓ Loaded {len(df)} stories")
    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_cosine_similarity_plot(
        df: pd.DataFrame,
        output_path: Path,
        title_suffix: str = ""
):
    """
    Create cosine similarity comparison bar plot.

    Args:
        df: DataFrame with columns: story_name, cosine_sim_original_baseline, cosine_similarity_original_abduction
        output_path: Path to save plot
        title_suffix: Additional text for title
    """
    # Extract story names (alphabetical order)
    stories = sorted(df['story_name'].tolist())
    n_stories = len(stories)

    # Extract similarity values
    baseline_vals = []
    abduction_vals = []

    for story in stories:
        row = df[df['story_name'] == story].iloc[0]
        baseline_vals.append(row['cosine_sim_original_baseline'])
        abduction_vals.append(row['cosine_similarity_original_abduction'])

    # Calculate figure size based on number of stories
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate bar positions
    bar_width = 0.35
    x = np.arange(n_stories)

    # Create bars
    bars1 = ax.bar(x - bar_width / 2, baseline_vals, bar_width,
                   label='Baseline vs Original',
                   color=COLORS['baseline'],
                   hatch=HATCHES['baseline'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    bars2 = ax.bar(x + bar_width / 2, abduction_vals, bar_width,
                   label='Abduction vs Original',
                   color=COLORS['abduction'],
                   hatch=HATCHES['abduction'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    # Customize plot
    title = f'Cosine Similarity: Transformed Stories vs Original{title_suffix}'
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=20)

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cosine Similarity (0-1 scale, higher = more similar)',
                  fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis ticks
    ax.set_xticks(x)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Set y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Add reference line at 1.0 (perfect similarity)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Similarity')

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

    # Adjust layout
    plt.subplots_adjust(right=0.85)

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


def create_js_divergence_plot(
        df: pd.DataFrame,
        output_path: Path,
        title_suffix: str = ""
):
    """
    Create JS divergence comparison bar plot.

    Args:
        df: DataFrame with columns: story_name, js_divergence_baseline, js_divergence_abduction
        output_path: Path to save plot
        title_suffix: Additional text for title
    """
    # Extract story names (alphabetical order)
    stories = sorted(df['story_name'].tolist())
    n_stories = len(stories)

    # Extract divergence values
    baseline_vals = []
    abduction_vals = []

    for story in stories:
        row = df[df['story_name'] == story].iloc[0]
        baseline_vals.append(row['js_divergence_baseline'])
        abduction_vals.append(row['js_divergence_abduction'])

    # Calculate figure size based on number of stories
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate bar positions
    bar_width = 0.35
    x = np.arange(n_stories)

    # Create bars
    bars1 = ax.bar(x - bar_width / 2, baseline_vals, bar_width,
                   label='Baseline vs Original',
                   color=COLORS['baseline'],
                   hatch=HATCHES['baseline'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    bars2 = ax.bar(x + bar_width / 2, abduction_vals, bar_width,
                   label='Abduction vs Original',
                   color=COLORS['abduction'],
                   hatch=HATCHES['abduction'],
                   alpha=BAR_ALPHA,
                   edgecolor=EDGE_COLOR,
                   linewidth=EDGE_WIDTH)

    # Customize plot
    title = f'JS Divergence: Transformed Stories vs Original{title_suffix}'
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=20)

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('JS Divergence (0-1 scale, higher = more different)',
                  fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis ticks
    ax.set_xticks(x)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Set y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Add reference line at 0.0 (no divergence)
    ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No Divergence')

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

    # Adjust layout
    plt.subplots_adjust(right=0.85)

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Plotting Function
# ==========================================================

def create_plots(
        metric_type: str,
        input_file: str,
        output_dir: str = "plots",
        title_suffix: str = ""
):
    """
    Create comparison plots for cosine similarity or JS divergence.

    Args:
        metric_type: "cosine" or "divergence"
        input_file: Path to CSV file
        output_dir: Output directory for plots
        title_suffix: Additional text for title (e.g., model name)
    """
    logger.info("=" * 60)
    logger.info("CREATING COMPARISON PLOTS")
    logger.info("=" * 60)
    logger.info(f"Metric: {metric_type}")
    logger.info(f"Input file: {input_file}")

    # Load data
    logger.info("\nLoading data...")
    if metric_type == "cosine":
        df = load_cosine_data(input_file)
    else:
        df = load_divergence_data(input_file)

    # Create output directory
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create plot
    logger.info("\nCreating plot...")
    if metric_type == "cosine":
        output_file = plot_dir / "cosine_similarity_comparison.png"
        create_cosine_similarity_plot(df, output_file, title_suffix)
    else:
        output_file = plot_dir / "js_divergence_comparison.png"
        create_js_divergence_plot(df, output_file, title_suffix)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PLOT CREATED")
    logger.info("=" * 60)
    logger.info(f"Plot saved to: {output_file}")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create comparison plots for cosine similarity or JS divergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot cosine similarity
  python plot_divergence.py --metric cosine --input cosine_similarities.csv

  # Plot JS divergence
  python plot_divergence.py --metric divergence --input js_divergence_token.csv

  # With custom output directory and title
  python plot_divergence.py --metric cosine --input cosine_similarities.csv \\
      --output plots/gpt-4o/forward --title " (GPT-4o, Forward)"
        """
    )

    parser.add_argument(
        '--metric',
        type=str,
        choices=['cosine', 'divergence'],
        required=True,
        help='Metric type: cosine similarity or JS divergence'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )

    parser.add_argument(
        '--title',
        type=str,
        default='',
        help='Additional text to append to plot title (e.g., " (GPT-4o)")'
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
        create_plots(
            metric_type=args.metric,
            input_file=args.input,
            output_dir=args.output,
            title_suffix=args.title
        )

        logger.info("\n✓ Plotting complete!")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
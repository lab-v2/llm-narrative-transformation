"""
Cosine Similarity Plotting Script

Creates publication-quality plots comparing cosine similarities
between original and transformed stories.

Higher similarity indicates better coherence with original narrative structure
while still achieving the transformation goal.

Output: PNG files in output_analysis/plots/{model}/{problem}/
"""

import argparse
import logging
from pathlib import Path
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

# Colors (green for baseline, shades of red for abductive)
COLOR_BASELINE = '#2ECC71'  # Green
COLOR_ABDUCTIVE_BASE = '#E74C3C'  # Red base

# Hatch patterns for B&W printing
HATCH_BASELINE = '///'  # Diagonal lines
HATCH_ABDUCTIVE = ['...', 'xxx', '\\\\\\', '|||', '+++']  # Different patterns for iterations

# Bar configuration
BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Data Loading
# ==========================================================

def load_cosine_similarities(
        model: str,
        problem_type: str,
        embeddings_dir: str = "output_analysis/embeddings"
) -> pd.DataFrame:
    """
    Load cosine similarities CSV.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        embeddings_dir: Base directory for embeddings

    Returns:
        DataFrame with cosine similarities
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    csv_file = Path(embeddings_dir) / sanitized_model / problem_type / "cosine_similarities.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Cosine similarities CSV not found: {csv_file}")

    logger.info(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"✓ Loaded {len(df)} stories")

    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_cosine_similarity_plot(
        df: pd.DataFrame,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create cosine similarity comparison plot.

    Args:
        df: DataFrame with cosine similarities
        model: Model name (for title)
        problem_type: "forward" or "inverse"
        output_path: Path to save plot
    """
    # Extract story names (alphabetically ordered)
    stories = sorted(df['story_name'].tolist())
    n_stories = len(stories)

    # Identify all similarity columns
    all_cols = df.columns.tolist()
    baseline_col = 'baseline_similarity'
    abductive_cols = [col for col in all_cols if col.startswith('abductive_iter_') and col.endswith('_similarity')]
    abductive_cols = sorted(abductive_cols, key=lambda x: int(x.split('_')[2]))  # Sort by iteration number

    n_methods = 1 + len(abductive_cols)  # baseline + all abductive iterations

    logger.info(f"Methods to plot: baseline + {len(abductive_cols)} abductive iterations")

    # Calculate figure size based on number of stories
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate bar positions
    bar_width = 0.8 / n_methods
    x = np.arange(n_stories)

    # Colors for abductive iterations (gradient from light to dark red)
    def get_abductive_color(idx, total):
        # Create gradient from light to dark red
        if total == 1:
            return COLOR_ABDUCTIVE_BASE
        # Interpolate between light and dark
        factor = idx / (total - 1)
        base_rgb = np.array([231, 76, 60]) / 255  # #E74C3C
        dark_rgb = np.array([192, 57, 43]) / 255  # #C0392B
        color_rgb = base_rgb * (1 - factor) + dark_rgb * factor
        return f'#{int(color_rgb[0] * 255):02x}{int(color_rgb[1] * 255):02x}{int(color_rgb[2] * 255):02x}'

    # Collect all bars for legend
    bars_list = []
    labels_list = []

    # Plot baseline
    baseline_vals = []
    for story in stories:
        val = df[df['story_name'] == story][baseline_col].values[0]
        baseline_vals.append(val)

    bars_baseline = ax.bar(
        x - (n_methods - 1) * bar_width / 2,
        baseline_vals,
        bar_width,
        label='Baseline',
        color=COLOR_BASELINE,
        hatch=HATCH_BASELINE,
        alpha=BAR_ALPHA,
        edgecolor=EDGE_COLOR,
        linewidth=EDGE_WIDTH
    )
    bars_list.append(bars_baseline)
    labels_list.append('Baseline')

    # Plot abductive iterations
    for i, col in enumerate(abductive_cols):
        iter_num = int(col.split('_')[2])

        abductive_vals = []
        for story in stories:
            val = df[df['story_name'] == story][col].values[0]
            abductive_vals.append(val)

        position = x - (n_methods - 1) * bar_width / 2 + (i + 1) * bar_width

        bars_abd = ax.bar(
            position,
            abductive_vals,
            bar_width,
            label=f'Abductive-Guided (Iter {iter_num})',
            color=get_abductive_color(i, len(abductive_cols)),
            hatch=HATCH_ABDUCTIVE[i % len(HATCH_ABDUCTIVE)],
            alpha=BAR_ALPHA,
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH
        )
        bars_list.append(bars_abd)
        labels_list.append(f'Abductive-Guided (Iter {iter_num})')

    # Customize plot
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(
        f'Cosine Similarity: {problem_desc}\n({model})',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cosine Similarity (0.0 - 1.0)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set x-axis ticks
    ax.set_xticks(x)
    clean_names = [s.replace('_', ' ') for s in stories]
    ax.set_xticklabels(clean_names, rotation=90, ha='right', fontsize=TICK_FONT_SIZE)

    # Set y-axis
    ax.set_ylim(0, 1.05)  # Similarity is 0-1
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend - positioned outside plot area on the right
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=1,
        fancybox=False
    )

    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_cosine_plots(
        model: str,
        problem_type: str,
        embeddings_dir: str = "output_analysis/embeddings",
        output_dir: str = "output_analysis/plots"
):
    """
    Create cosine similarity plots.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        embeddings_dir: Base directory for embeddings/CSV
        output_dir: Output directory for plots
    """
    logger.info("=" * 60)
    logger.info("CREATING COSINE SIMILARITY PLOT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")

    # Load data
    df = load_cosine_similarities(model, problem_type, embeddings_dir)

    # Sanitize model name for output path
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type

    # Create plot
    output_file = plot_dir / "cosine_similarity.png"

    create_cosine_similarity_plot(
        df=df,
        model=model,
        problem_type=problem_type,
        output_path=output_file
    )

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
        description='Create cosine similarity plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plot for gpt-4o forward
  python plot_scripts/create_cosine_similarity_plot.py --model gpt-4o --problem forward

  # Create for Claude inverse
  python plot_scripts/create_cosine_similarity_plot.py --model claude-sonnet-4-5 --problem inverse
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4o, claude-sonnet-4-5)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type: forward or inverse'
    )

    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='output_analysis/embeddings',
        help='Embeddings directory (default: output_analysis/embeddings)'
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

    # Create plot
    try:
        create_cosine_plots(
            model=args.model,
            problem_type=args.problem,
            embeddings_dir=args.embeddings_dir,
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
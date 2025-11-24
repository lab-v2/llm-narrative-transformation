"""
KL Divergence Box Plot Script

Creates publication-quality box plots showing KL(transformed || original) divergence
distribution across all stories.

Lower KL divergence indicates better coherence with original narrative
(less drastic change while still transforming).

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

# Font sizes
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# Colors
COLOR_BASELINE = '#2ECC71'  # Green
COLOR_ABDUCTIVE_BASE = '#E74C3C'  # Red

# Box plot styling
BOX_ALPHA = 0.7
MEDIAN_COLOR = 'black'
MEDIAN_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_kl_divergence(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences"
) -> pd.DataFrame:
    """
    Load KL divergence CSV.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        divergence_dir: Base directory for divergences

    Returns:
        DataFrame with KL(transformed || original) divergences
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")

    csv_file = Path(divergence_dir) / sanitized_model / problem_type / "kl_divergence_transformed_original.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"KL divergence CSV not found: {csv_file}")

    logger.info(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"✓ Loaded {len(df)} stories")

    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_kl_boxplot(
        df: pd.DataFrame,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create KL divergence box plot.

    Args:
        df: DataFrame with KL divergences
        model: Model name (for title)
        problem_type: "forward" or "inverse"
        output_path: Path to save plot
    """
    # Identify all KL columns
    all_cols = df.columns.tolist()
    baseline_col = 'baseline_kl'
    abductive_cols = [col for col in all_cols if col.startswith('abductive_iter_') and col.endswith('_kl')]
    abductive_cols = sorted(abductive_cols, key=lambda x: int(x.split('_')[2]))

    logger.info(f"Methods to plot: baseline + {len(abductive_cols)} abductive iterations")

    # Prepare data for box plot
    data_to_plot = []
    labels = []
    colors = []

    # Baseline
    baseline_vals = df[baseline_col].dropna().tolist()
    data_to_plot.append(baseline_vals)
    labels.append('Baseline')
    colors.append(COLOR_BASELINE)

    # Abductive iterations
    n_abductive = len(abductive_cols)
    for i, col in enumerate(abductive_cols):
        iter_num = int(col.split('_')[2])
        abductive_vals = df[col].dropna().tolist()
        data_to_plot.append(abductive_vals)
        labels.append(f'Abductive\n(Iter {iter_num})')

        # Color gradient from light to dark red
        if n_abductive == 1:
            colors.append(COLOR_ABDUCTIVE_BASE)
        else:
            factor = i / (n_abductive - 1)
            base_rgb = np.array([231, 76, 60]) / 255
            dark_rgb = np.array([192, 57, 43]) / 255
            color_rgb = base_rgb * (1 - factor) + dark_rgb * factor
            colors.append(f'#{int(color_rgb[0] * 255):02x}{int(color_rgb[1] * 255):02x}{int(color_rgb[2] * 255):02x}')

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
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Customize plot
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

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    # Log statistics
    logger.info("\nBox Plot Statistics:")
    for i, (label, data) in enumerate(zip(labels, data_to_plot)):
        logger.info(f"  {label}:")
        logger.info(f"    Median: {np.median(data):.4f}")
        logger.info(f"    Mean: {np.mean(data):.4f}")
        logger.info(f"    Std: {np.std(data):.4f}")
        logger.info(f"    Min-Max: {np.min(data):.4f} - {np.max(data):.4f}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_kl_boxplots(
        model: str,
        problem_type: str,
        divergence_dir: str = "output_analysis/divergences",
        output_dir: str = "output_analysis/plots"
):
    """
    Create KL divergence box plots.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        divergence_dir: Base directory for divergence CSVs
        output_dir: Output directory for plots
    """
    logger.info("=" * 60)
    logger.info("CREATING KL DIVERGENCE BOX PLOT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")

    # Load data
    df = load_kl_divergence(model, problem_type, divergence_dir)

    # Sanitize model name
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type

    # Create plot
    output_file = plot_dir / "kl_divergence_boxplot.png"

    create_kl_boxplot(
        df=df,
        model=model,
        problem_type=problem_type,
        output_path=output_file
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BOX PLOT CREATED")
    logger.info("=" * 60)
    logger.info(f"Plot saved to: {output_file}")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create KL divergence box plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create box plot for gpt-4o forward
  python plot_scripts/create_kl_boxplot.py --model gpt-4o --problem forward

  # Create for Claude inverse
  python plot_scripts/create_kl_boxplot.py --model claude-sonnet-4-5 --problem inverse
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
        '--divergence-dir',
        type=str,
        default='output_analysis/divergences',
        help='Divergence directory (default: output_analysis/divergences)'
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
        create_kl_boxplots(
            model=args.model,
            problem_type=args.problem,
            divergence_dir=args.divergence_dir,
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
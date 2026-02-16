"""
BERTScore F1 Box Plot Script

Creates publication-quality box plots showing BERTScore F1 distribution
comparing baseline and abductive-guided story transformations.

Can exclude specific stories from plots.

Higher F1 score = better semantic preservation with original.

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

def load_bertscore_data(
        model: str,
        problem_type: str,
        bertscore_dir: str = "output_analysis/bert_scores_csv",
        exclude_stories: list = None
) -> pd.DataFrame:
    """
    Load BERTScore data.

    Args:
        model: Model name
        problem_type: Problem type ('forward' or 'inverse')
        bertscore_dir: BERTScore CSV directory
        exclude_stories: Stories to exclude

    Returns:
        DataFrame with BERTScore F1 values
    """
    # Map problem type to direction names
    if problem_type == "forward":
        direction = "forward"
    elif problem_type == "inverse":
        direction = "inverse"
    else:
        raise ValueError("problem_type must be 'forward' or 'inverse'")

    # Sanitize model name for file paths
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load BERTScore CSV
    bertscore_file = Path(bertscore_dir) / f"bertscore_results_{sanitized_model}_{direction}.csv"
    
    if not bertscore_file.exists():
        raise FileNotFoundError(f"BERTScore CSV not found: {bertscore_file}")

    df = pd.read_csv(bertscore_file)

    logger.info(f"Loaded data for {len(df)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df = df[~df['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df)}")

    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_bertscore_f1_boxplot(
        df: pd.DataFrame,
        model: str,
        problem_type: str,
        output_path: Path
):
    """
    Create BERTScore F1 box plot.

    Args:
        df: BERTScore DataFrame
        model: Model name
        problem_type: Problem type
        output_path: Output path
    """
    # Extract F1 scores
    baseline_vals = df['baseline_bertscore_f1'].dropna().tolist()
    abductive_vals = df['abduction_bertscore_f1'].dropna().tolist()

    # Prepare data for box plot
    data_to_plot = [baseline_vals, abductive_vals]
    labels = ['Baseline', 'Abduction-Guided']
    colors = [COLOR_BASELINE, COLOR_ABDUCTIVE]

    logger.info(f"Baseline: {len(baseline_vals)} values")
    logger.info(f"Abduction-Guided: {len(abductive_vals)} values")

    # Create figure (taller for better box visibility)
    fig, ax = plt.subplots(figsize=(4, 5))

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
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        # Color outliers to match their box color
        bp['fliers'][i].set_markerfacecolor(color)
        bp['fliers'][i].set_markeredgecolor(color)

    # Customize
    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    # ax.set_title(
    #     f'BERTScore F1 Distribution: {problem_desc}\n({model})',
    #     fontsize=TITLE_FONT_SIZE,
    #     fontweight='bold',
    #     pad=20
    # )

    # ax.set_xlabel('Method', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('BERTScore F1', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # Set y-axis limits zoomed to BERTScore F1 range (typically 0.75-1.0)
    ax.set_ylim(0.75, 1.01)

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
        logger.info(f"    Min: {np.min(data):.4f}")
        logger.info(f"    Max: {np.max(data):.4f}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_bertscore_f1_boxplot_main(
        model: str,
        problem_type: str,
        bertscore_dir: str = "output_analysis/bert_scores_csv",
        output_dir: str = "output_analysis/plots",
        exclude_stories: list = None
):
    """
    Create BERTScore F1 box plot.
    """
    logger.info("=" * 60)
    logger.info("CREATING BERTSCORE F1 BOX PLOT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df = load_bertscore_data(
        model, problem_type, bertscore_dir, exclude_stories
    )

    # Create plot
    sanitized_model = model.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_model / problem_type
    output_file = plot_dir / "bertscore_f1_boxplot.png"

    create_bertscore_f1_boxplot(
        df, model, problem_type, output_file
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
        description='Create BERTScore F1 box plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create box plot for forward problem
  python plot_scripts/bertscore_f1_boxplot.py --model gpt-4o --problem forward

  # Exclude specific stories
  python plot_scripts/bertscore_f1_boxplot.py \\
      --model gpt-4o --problem forward \\
      --exclude Back_To_The_Wall a_plain_case
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--bertscore-dir', default='output_analysis/bert_scores_csv')
    parser.add_argument('--output-dir', default='output_analysis/plots')
    parser.add_argument('--exclude', nargs='*', dest='exclude_stories', help='Stories to exclude')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        create_bertscore_f1_boxplot_main(
            model=args.model,
            problem_type=args.problem,
            bertscore_dir=args.bertscore_dir,
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
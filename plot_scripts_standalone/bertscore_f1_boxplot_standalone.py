#!/usr/bin/env python3
"""
BERTScore F1 Box Plot Script (Standalone)

Creates publication-quality box plots showing BERTScore F1 distribution
comparing baseline and abduction-guided story transformations.

Can exclude specific stories from plots.

Higher F1 score = better semantic preservation with original.

Output: PNG files in output_analysis_standalone/plots/{tuned_model}/{survey_model}/{story_type}/
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

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# Colors
COLOR_BASELINE = '#2ECC71'  # Green
COLOR_ABDUCTION = '#E74C3C'  # Red

BOX_ALPHA = 0.7
MEDIAN_COLOR = 'black'
MEDIAN_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_bertscore_data(
        bertscore_dir: str,
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        exclude_stories: list = None
) -> pd.DataFrame:
    """
    Load BERTScore data.

    Args:
        bertscore_dir: Base BERTScore CSV directory
        tuned_model_name: Name of tuned model
        model_name_for_survey: Name of survey model
        original_stories_type: Type of original stories (individualistic/collectivistic)
        exclude_stories: Stories to exclude

    Returns:
        DataFrame with BERTScore F1 values
    """
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Load BERTScore CSV
    bertscore_file = Path(bertscore_dir) / f"bertscore_results_{sanitized_tuned}_{sanitized_survey}_{original_stories_type}.csv"
    
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
        output_path: Path
):
    """
    Create BERTScore F1 box plot.

    Args:
        df: BERTScore DataFrame
        output_path: Output path
    """
    # Extract F1 scores
    baseline_vals = df['baseline_bertscore_f1'].dropna().tolist()
    abduction_vals = df['abduction_bertscore_f1'].dropna().tolist()

    # Prepare data for box plot
    data_to_plot = [baseline_vals, abduction_vals]
    labels = ['Baseline', 'Abduction-Guided']
    colors = [COLOR_BASELINE, COLOR_ABDUCTION]

    logger.info(f"Baseline: {len(baseline_vals)} values")
    logger.info(f"Abduction-Guided: {len(abduction_vals)} values")

    # Create figure (taller for better box visibility)
    fig, ax = plt.subplots(figsize=(8, 9))

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
    ax.set_xlabel('Method', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('BERTScore F1', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

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
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        bertscore_dir: str = "output_analysis_standalone/bert_scores_csv",
        output_dir: str = "output_analysis_standalone/plots",
        exclude_stories: list = None
):
    """
    Create BERTScore F1 box plot.
    """
    logger.info("=" * 70)
    logger.info("CREATING BERTSCORE F1 BOX PLOT (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df = load_bertscore_data(
        bertscore_dir, tuned_model_name, model_name_for_survey,
        original_stories_type, exclude_stories
    )

    # Create plot
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    output_file = plot_dir / "bertscore_f1_boxplot.png"

    create_bertscore_f1_boxplot(df, output_file)

    logger.info("\n" + "=" * 70)
    logger.info("BOX PLOT CREATED")
    logger.info("=" * 70)
    logger.info(f"Output: {output_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create BERTScore F1 box plots from survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create box plot
  python plot_scripts_standalone/bertscore_f1_boxplot_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"

  # Exclude specific stories
  python plot_scripts_standalone/bertscore_f1_boxplot_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic" \\
    --exclude Story1 Story2
        """
    )

    parser.add_argument(
        '--tuned-model-name',
        type=str,
        required=True,
        help='Name of the fine-tuned model that generated transformed stories'
    )

    parser.add_argument(
        '--model-name-for-survey',
        type=str,
        required=True,
        help='Name of the model used for diagnosis/evaluation survey'
    )

    parser.add_argument(
        '--original-stories-type',
        type=str,
        choices=['individualistic', 'collectivistic'],
        required=True,
        help='Type of original stories'
    )

    parser.add_argument(
        '--bertscore-dir',
        type=str,
        default='output_analysis_standalone/bert_scores_csv',
        help='Base BERTScore CSV directory (default: output_analysis_standalone/bert_scores_csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/plots',
        help='Output directory for plots (default: output_analysis_standalone/plots)'
    )

    parser.add_argument(
        '--exclude',
        nargs='*',
        dest='exclude_stories',
        help='Stories to exclude from plots'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    logger.info(f"Arguments:")
    logger.info(f"  --tuned-model-name: {args.tuned_model_name}")
    logger.info(f"  --model-name-for-survey: {args.model_name_for_survey}")
    logger.info(f"  --original-stories-type: {args.original_stories_type}")
    logger.info(f"  --bertscore-dir: {args.bertscore_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --exclude: {args.exclude_stories}")
    logger.info(f"  --verbose: {args.verbose}")

    try:
        create_bertscore_f1_boxplot_main(
            tuned_model_name=args.tuned_model_name,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
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

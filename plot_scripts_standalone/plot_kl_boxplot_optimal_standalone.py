#!/usr/bin/env python3
"""
KL Divergence Box Plot Script (Standalone)

Creates publication-quality box plots showing KL(transformed || original) divergence
distribution comparing baseline vs abduction-guided transformations.

Can exclude specific stories from plots.

Lower KL divergence = better coherence with original.

Output: PNG files in output_analysis_standalone/plots/{tuned_model}/{survey_model}/{story_type}/
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
COLOR_ABDUCTION = '#E74C3C'  # Red

BOX_ALPHA = 0.7
MEDIAN_COLOR = 'black'
MEDIAN_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_kl_data(
        divergence_dir: str,
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        exclude_stories: list = None
) -> pd.DataFrame:
    """
    Load KL divergence data.

    Args:
        divergence_dir: Base divergence directory
        tuned_model_name: Name of tuned model
        model_name_for_survey: Name of survey model
        original_stories_type: Type of original stories (individualistic/collectivistic)
        exclude_stories: Stories to exclude

    Returns:
        KL divergence DataFrame
    """
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Load KL divergence
    kl_file = Path(divergence_dir) / sanitized_tuned / sanitized_survey / original_stories_type / "kl_divergence_transformed_original.csv"
    
    if not kl_file.exists():
        raise FileNotFoundError(f"KL divergence not found: {kl_file}")

    df_kl = pd.read_csv(kl_file)

    logger.info(f"Loaded data for {len(df_kl)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df_kl = df_kl[~df_kl['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df_kl)}")

    return df_kl


# ==========================================================
# Plot Creation
# ==========================================================

def create_kl_boxplot(
        df_kl: pd.DataFrame,
        tuned_model_name: str,
        original_stories_type: str,
        y_max: float,
        output_path: Path
):
    """
    Create KL divergence box plot.

    Args:
        df_kl: KL divergence DataFrame
        tuned_model_name: Name of tuned model
        original_stories_type: Type of original stories
        y_max: Maximum y-axis value
        output_path: Output path
    """
    # Collect baseline values
    baseline_vals = df_kl['baseline_kl'].dropna().tolist()

    # Collect abduction values
    abduction_vals = df_kl['abduction_kl'].dropna().tolist()

    # Prepare data for box plot
    data_to_plot = [baseline_vals, abduction_vals]
    labels = ['Baseline', 'Abduction-Guided']
    colors = [COLOR_BASELINE, COLOR_ABDUCTION]

    logger.info(f"Baseline: {len(baseline_vals)} values")
    logger.info(f"Abduction: {len(abduction_vals)} values")

    # Create figure
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
    ax.set_ylabel('KL(transformed || original)', fontsize=LABEL_FONT_SIZE, fontweight='bold',
                  labelpad=10)

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # Set y-axis limits
    ax.set_ylim(0, y_max)

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

def create_kl_boxplot_main(
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        y_max: float = 4.0,
        divergence_dir: str = "output_analysis_standalone/divergences",
        output_dir: str = "output_analysis_standalone/plots",
        exclude_stories: list = None
):
    """
    Create KL divergence box plot.
    """
    logger.info("=" * 70)
    logger.info("CREATING KL DIVERGENCE BOX PLOT (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")
    logger.info(f"Y-axis max: {y_max}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load data
    df_kl = load_kl_data(
        divergence_dir, tuned_model_name, model_name_for_survey, 
        original_stories_type, exclude_stories
    )

    # Create plot
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    output_file = plot_dir / "kl_divergence_boxplot_optimal.png"

    create_kl_boxplot(
        df_kl, tuned_model_name, original_stories_type, y_max, output_file
    )

    logger.info("\n" + "=" * 70)
    logger.info("BOX PLOT CREATED")
    logger.info("=" * 70)
    logger.info(f"Output: {output_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create KL divergence box plots from survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create box plot
  python plot_scripts_standalone/plot_kl_boxplot_optimal_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"

  # With custom y-axis limit
  python plot_scripts_standalone/plot_kl_boxplot_optimal_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic" \\
    --y-max 3.5

  # Exclude specific stories
  python plot_scripts_standalone/plot_kl_boxplot_optimal_standalone.py \\
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
        '--y-max',
        type=float,
        default=4.0,
        help='Maximum y-axis value (default: 4.0)'
    )

    parser.add_argument(
        '--divergence-dir',
        type=str,
        default='output_analysis_standalone/divergences',
        help='Base divergence directory (default: output_analysis_standalone/divergences)'
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
    logger.info(f"  --y-max: {args.y_max}")
    logger.info(f"  --divergence-dir: {args.divergence_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --exclude: {args.exclude_stories}")
    logger.info(f"  --verbose: {args.verbose}")

    try:
        create_kl_boxplot_main(
            tuned_model_name=args.tuned_model_name,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            y_max=args.y_max,
            divergence_dir=args.divergence_dir,
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

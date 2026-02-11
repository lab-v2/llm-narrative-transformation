#!/usr/bin/env python3
"""
Diagnosis Aggregates Plotting Script (Standalone)

Creates publication-quality bar plots comparing original vs baseline vs abduction-guided transformations.
Uses survey results from the survey CSVs.

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

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 20

# Colors
COLORS = {
    'original': '#2E86AB',  # Blue
    'baseline': '#2ECC71',  # Green
    'abduction': '#E74C3C'  # Red
}

# Hatches
HATCHES = {
    'original': '',
    'baseline': '///',
    'abduction': 'xxx'
}

BAR_ALPHA = 0.8
EDGE_COLOR = 'black'
EDGE_WIDTH = 1.2


# ==========================================================
# Story Name Mapping
# ==========================================================

def load_story_mapping(
        mapping_file: str = "output_analysis/story_name_mapping.csv"
) -> dict:
    """
    Load story name mapping.

    Args:
        mapping_file: Path to mapping CSV

    Returns:
        Dict mapping original_name -> abbreviated_name
    """
    mapping_path = Path(mapping_file)

    if not mapping_path.exists():
        logger.warning(f"Story mapping file not found: {mapping_file}")
        logger.warning("Run: python plot_scripts/create_story_name_mapping.py")
        return {}

    df_mapping = pd.read_csv(mapping_path)
    mapping = dict(zip(df_mapping['original_name'], df_mapping['abbreviated_name']))

    logger.info(f"Loaded story name mapping: {len(mapping)} stories")

    return mapping


# ==========================================================
# Data Loading
# ==========================================================

def load_comparison_data(
        analysis_dir: str,
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        exclude_stories: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data for comparison plot.

    Args:
        analysis_dir: Base analysis directory (output_analysis_standalone/survey)
        tuned_model_name: Name of tuned model
        model_name_for_survey: Name of survey model
        original_stories_type: Type of original stories (individualistic/collectivistic)
        exclude_stories: List of story names to exclude

    Returns:
        Tuple of (original_df, baseline_df, abduction_df)
    """
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Load survey data from analysis directory
    survey_path = Path(analysis_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    original_file = survey_path / "original_ratings.csv"
    baseline_file = survey_path / "baseline_transformed_ratings.csv"
    abduction_file = survey_path / "abduction_transformed_ratings.csv"

    # Check files exist
    if not original_file.exists():
        raise FileNotFoundError(f"Original ratings not found: {original_file}")
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline ratings not found: {baseline_file}")
    if not abduction_file.exists():
        raise FileNotFoundError(f"Abduction ratings not found: {abduction_file}")

    # Load files
    df_original = pd.read_csv(original_file)
    df_baseline = pd.read_csv(baseline_file)
    df_abduction = pd.read_csv(abduction_file)

    logger.info(f"Loaded data for {len(df_original)} stories")

    # Exclude stories if specified
    if exclude_stories:
        logger.info(f"Excluding {len(exclude_stories)} stories: {exclude_stories}")
        df_original = df_original[~df_original['story_name'].isin(exclude_stories)]
        df_baseline = df_baseline[~df_baseline['story_name'].isin(exclude_stories)]
        df_abduction = df_abduction[~df_abduction['story_name'].isin(exclude_stories)]
        logger.info(f"Remaining stories: {len(df_original)}")

    return df_original, df_baseline, df_abduction


# ==========================================================
# Plot Creation
# ==========================================================

def create_comparison_plot(
        df_original: pd.DataFrame,
        df_baseline: pd.DataFrame,
        df_abduction: pd.DataFrame,
        stat_type: str,
        original_stories_type: str,
        output_path: Path,
        story_mapping: dict = None
):
    """
    Create comparison bar plot.

    Args:
        df_original: Original ratings
        df_baseline: Baseline transformed ratings
        df_abduction: Abduction transformed ratings
        stat_type: "Mean", "Median", or "Mode"
        original_stories_type: Type of original stories (individualistic/collectivistic)
        output_path: Output path
        story_mapping: Dict mapping original names to abbreviated names
    """
    # Get stories (alphabetically ordered)
    stories = sorted(df_original['story_name'].tolist())
    n_stories = len(stories)

    # Determine column names based on stat_type
    stat_col = stat_type  # "Mean", "Median", or "Mode"

    # Extract values
    original_vals = []
    baseline_vals = []
    abduction_vals = []

    for story in stories:
        original_vals.append(df_original[df_original['story_name'] == story][stat_col].values[0])
        baseline_vals.append(df_baseline[df_baseline['story_name'] == story][stat_col].values[0])
        abduction_vals.append(df_abduction[df_abduction['story_name'] == story][stat_col].values[0])

    # Flip ratings for collectivistic original stories (so higher is always better visually)
    if original_stories_type == "collectivistic":
        original_vals = [6 - v for v in original_vals]
        baseline_vals = [6 - v for v in baseline_vals]
        abduction_vals = [6 - v for v in abduction_vals]

    # Figure size
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Bar positions
    bar_width = 0.25
    x = np.arange(n_stories)

    # Create bars
    ax.bar(x - bar_width, original_vals, bar_width,
           label='Original',
           color=COLORS['original'],
           hatch=HATCHES['original'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    ax.bar(x, baseline_vals, bar_width,
           label='Baseline',
           color=COLORS['baseline'],
           hatch=HATCHES['baseline'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    ax.bar(x + bar_width, abduction_vals, bar_width,
           label='Abduction-Guided',
           color=COLORS['abduction'],
           hatch=HATCHES['abduction'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    # Customize
    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{stat_type} Rating', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # X-axis labels with LaTeX subscripts
    ax.set_xticks(x)

    if story_mapping:
        # Use abbreviated names with LaTeX subscripts
        abbreviated_labels = []
        for story in stories:
            if story in story_mapping:
                # Convert S_1 to $S_1$ for LaTeX rendering
                abbrev = story_mapping[story]
                # Extract number from S_X format
                if abbrev.startswith('S_'):
                    num = abbrev.split('_')[1]
                    latex_label = f'$S_{{{num}}}$'
                    abbreviated_labels.append(latex_label)
                else:
                    abbreviated_labels.append(abbrev)
            else:
                # Fallback to cleaned original name
                abbreviated_labels.append(story.replace('_', ' '))

        ax.set_xticklabels(abbreviated_labels, rotation=0, ha='right', fontsize=TICK_FONT_SIZE)
    else:
        # No mapping, use original names
        clean_names = [s.replace('_', ' ') for s in stories]
        ax.set_xticklabels(clean_names, rotation=0, ha='right', fontsize=TICK_FONT_SIZE)

    # Y-axis
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        borderpad=0.2,
        fancybox=False
    )

    plt.subplots_adjust(right=0.95)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_comparison_plots_standalone(
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        stat_type: str = "all",
        analysis_dir: str = "output_analysis_standalone/survey",
        output_dir: str = "output_analysis_standalone/plots",
        exclude_stories: list = None,
        mapping_file: str = "output_analysis/story_name_mapping.csv"
):
    """
    Create comparison plots from survey data.

    Args:
        tuned_model_name: Name of tuned model
        model_name_for_survey: Name of survey model
        original_stories_type: Type of original stories (individualistic/collectivistic)
        stat_type: "mean", "median", "mode", or "all"
        analysis_dir: Base analysis directory (output_analysis_standalone/survey)
        output_dir: Output directory
        exclude_stories: Stories to exclude
        mapping_file: Story name mapping file
    """
    logger.info("=" * 70)
    logger.info("CREATING COMPARISON PLOTS (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")
    if exclude_stories:
        logger.info(f"Excluding: {exclude_stories}")

    # Load story mapping
    story_mapping = load_story_mapping(mapping_file)

    # Load data
    df_original, df_baseline, df_abduction = load_comparison_data(
        analysis_dir, tuned_model_name, model_name_for_survey, 
        original_stories_type, exclude_stories
    )

    # Determine which stats to plot
    if stat_type.lower() == "all":
        stats_to_plot = ["Mean", "Median", "Mode"]
    else:
        stats_to_plot = [stat_type.capitalize()]

    # Create plots
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")
    plot_dir = Path(output_dir) / sanitized_tuned / sanitized_survey / original_stories_type

    for stat in stats_to_plot:
        output_file = plot_dir / f"{stat.lower()}_comparison.png"

        create_comparison_plot(
            df_original, df_baseline, df_abduction,
            stat, original_stories_type, output_file,
            story_mapping
        )

    logger.info("\n" + "=" * 70)
    logger.info("PLOTS CREATED")
    logger.info("=" * 70)
    logger.info(f"Output: {plot_dir}")
    for stat in stats_to_plot:
        logger.info(f"  - {stat.lower()}_comparison.png")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create comparison plots from survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all plots
  python plot_scripts_standalone/plot_diagnosis_aggregates_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic" \\
    --stat all

  # Exclude specific stories
  python plot_scripts_standalone/plot_diagnosis_aggregates_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic" \\
    --stat all \\
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
        '--stat',
        type=str,
        default='all',
        choices=['mean', 'median', 'mode', 'all'],
        help='Which statistics to plot (default: all)'
    )

    parser.add_argument(
        '--analysis-dir',
        type=str,
        default='output_analysis_standalone/survey',
        help='Base analysis directory with extracted CSV ratings (default: output_analysis_standalone/survey)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/plots',
        help='Output directory for plots (default: output_analysis_standalone/plots)'
    )

    parser.add_argument(
        '--mapping-file',
        type=str,
        default='output_analysis/story_name_mapping.csv',
        help='Story name mapping file (default: output_analysis/story_name_mapping.csv)'
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
    logger.info(f"  --stat: {args.stat}")
    logger.info(f"  --analysis-dir: {args.analysis_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --mapping-file: {args.mapping_file}")
    logger.info(f"  --exclude: {args.exclude_stories}")
    logger.info(f"  --verbose: {args.verbose}")

    try:
        create_comparison_plots_standalone(
            tuned_model_name=args.tuned_model_name,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            stat_type=args.stat,
            analysis_dir=args.analysis_dir,
            output_dir=args.output_dir,
            exclude_stories=args.exclude_stories,
            mapping_file=args.mapping_file
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

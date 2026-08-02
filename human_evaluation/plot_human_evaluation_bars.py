"""
Human Evaluation Bar Plots Script

Creates publication-quality grouped bar plots comparing original vs baseline vs abduction ratings.
Matches the style of plot_diagnosis_aggregates.py exactly.
Generates separate plots for forward and inverse problem types.

Usage:
  python plot_human_evaluation_bars.py

Output: PNG files in the same directory
- human_evaluation_forward_bars.png
- human_evaluation_inverse_bars.png
"""

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ==========================================================
# Plot Configuration (MATCHING plot_diagnosis_aggregates.py)
# ==========================================================

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 20

# Colors
COLORS = {
    'original': '#2E86AB',      # Blue
    'baseline': '#2ECC71',      # Green
    'abduction': '#E74C3C'      # Red
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
# Data Loading
# ==========================================================

def load_evaluation_data(csv_file: str) -> pd.DataFrame:
    """
    Load human evaluation CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        DataFrame with columns: Story_id, original, baseline, abduction
    """
    csv_path = Path(csv_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    logger.info(f"Loaded {len(df)} stories from {csv_path.name}")
    logger.info(f"Stories: {', '.join(df['Story_id'].tolist())}")

    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_bar_plot(
        df: pd.DataFrame,
        problem_type: str,
        output_path: Path
):
    """
    Create grouped bar plot for human evaluation data.
    Matches the style of plot_diagnosis_aggregates.py.

    Args:
        df: DataFrame with Story_id, original, baseline, abduction columns
        problem_type: "forward" or "inverse"
        output_path: Output PNG path
    """
    n_stories = len(df)

    # Figure size
    fig_width = max(16, n_stories * 0.6)
    fig_height = 7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Extract data
    stories = df['Story_id'].tolist()
    original_vals = df['original'].values
    baseline_vals = df['baseline'].values
    abduction_vals = df['abduction'].values

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
           label='NARRATE',
           color=COLORS['abduction'],
           hatch=HATCHES['abduction'],
           alpha=BAR_ALPHA,
           edgecolor=EDGE_COLOR,
           linewidth=EDGE_WIDTH)

    # Customize axes
    ax.set_xlabel('Stories', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Rating (1-5)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # X-axis labels with LaTeX subscripts
    ax.set_xticks(x)
    latex_labels = []
    
    for story_id in stories:
        # Convert S_5 to $S_5$ for LaTeX rendering
        if story_id.startswith('S_'):
            num = story_id.split('_')[1]
            latex_label = f'$S_{{{num}}}$'
            latex_labels.append(latex_label)
        else:
            latex_labels.append(story_id)

    ax.set_xticklabels(latex_labels, rotation=0, ha='center', fontsize=TICK_FONT_SIZE)

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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {output_path}")

    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def main():
    """
    Create bar plots for forward and inverse problems.
    Looks for CSV files in the same directory as this script.
    """
    logger.info("=" * 60)
    logger.info("CREATING HUMAN EVALUATION BAR PLOTS")
    logger.info("=" * 60)

    script_dir = Path(__file__).parent

    # Look for CSV files - try both with and without timestamp prefix
    forward_files = list(script_dir.glob("*human_evaluation_forward.csv"))
    inverse_files = list(script_dir.glob("*human_evaluation_inverse.csv"))

    if not forward_files:
        raise FileNotFoundError("Could not find *human_evaluation_forward.csv in script directory")
    if not inverse_files:
        raise FileNotFoundError("Could not find *human_evaluation_inverse.csv in script directory")

    forward_csv = forward_files[0]
    inverse_csv = inverse_files[0]

    logger.info(f"Found CSV files:")
    logger.info(f"  Forward: {forward_csv.name}")
    logger.info(f"  Inverse: {inverse_csv.name}")

    # Forward plot
    logger.info("\nProcessing FORWARD problem...")
    df_forward = load_evaluation_data(str(forward_csv))
    forward_output = script_dir / "human_evaluation_forward_bars.png"
    create_bar_plot(df_forward, "forward", forward_output)

    # Inverse plot
    logger.info("\nProcessing INVERSE problem...")
    df_inverse = load_evaluation_data(str(inverse_csv))
    inverse_output = script_dir / "human_evaluation_inverse_bars.png"
    create_bar_plot(df_inverse, "inverse", inverse_output)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {script_dir}")
    logger.info(f"  - human_evaluation_forward_bars.png")
    logger.info(f"  - human_evaluation_inverse_bars.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    try:
        main()
        print("\n✓ Success!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)

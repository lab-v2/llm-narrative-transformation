"""
Single Story Runtime Plot Script

Creates line plot showing runtime for each transformation step (iteration)
for a single story.

Input: Direction (forward/inverse), story name
Output: PNG file in output_analysis_runtime/
"""

import argparse
import logging
from pathlib import Path
import json
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
BAR_ALPHA = 0.7


# ==========================================================
# Data Loading and Processing
# ==========================================================

def find_story_data_all_hp(output_dir: Path, direction: str, story_name: str) -> dict:
    """
    Find the story directory paths for all hyperparameters.

    Args:
        output_dir: Path to output directory
        direction: 'forward' or 'inverse'
        story_name: Name of the story folder

    Returns:
        Dictionary with hyperparameter as key and story path as value
    """
    hyperparameters = [1, 2, 3, 4, 5, 10,20]
    story_paths = {}

    for hp in hyperparameters:
        story_path = output_dir / f'hyperparameter-{hp}' / 'phase2' / 'gpt-4o' / direction / story_name

        if story_path.exists():
            story_paths[hp] = story_path
        else:
            logger.warning(f"Story path does not exist for hyperparameter-{hp}: {story_path}")

    if not story_paths:
        raise FileNotFoundError(f"No valid story paths found for {story_name} in {direction}")

    return story_paths


def extract_iteration_0_runtime(story_path: Path) -> tuple:
    """
    Extract runtime and n_segments from iteration_0 of a story.

    Args:
        story_path: Path to story directory

    Returns:
        Tuple of (n_segments, runtime_seconds)
    """
    json_path = story_path / 'iteration_0' / 'cost_tracking_transformation.json'

    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        n_segments = data.get('n_segments', 0)
        per_story_breakdown = data.get('per_story_breakdown', [])

        if not per_story_breakdown:
            raise ValueError(f"Empty per_story_breakdown in {json_path}")

        # Calculate runtime in seconds
        from datetime import datetime
        first_timestamp = datetime.fromisoformat(per_story_breakdown[0]['timestamp'])
        last_timestamp = datetime.fromisoformat(per_story_breakdown[-1]['timestamp'])
        runtime_seconds = (last_timestamp - first_timestamp).total_seconds()

        logger.info(f"  iteration_0: n_segments={n_segments}, runtime={runtime_seconds:.2f} seconds")

        return n_segments, runtime_seconds

    except Exception as e:
        raise ValueError(f"Error processing {json_path}: {e}")


# ==========================================================
# Plot Creation
# ==========================================================

def create_story_runtime_plot(
        output_dir: Path,
        direction: str,
        story_name: str,
        output_file: str
):
    """
    Create single story iteration_0 runtime bar plot across all hyperparameters.

    Args:
        output_dir: Path to output directory
        direction: 'forward' or 'inverse'
        story_name: Story name
        output_file: Path to save PNG
    """
    # Find story paths for all hyperparameters
    story_paths = find_story_data_all_hp(output_dir, direction, story_name)
    logger.info(f"Processing story: {story_name}")
    logger.info(f"Found {len(story_paths)} hyperparameter configurations")

    # Extract iteration_0 runtime for each hyperparameter
    logger.info("Extracting iteration_0 runtimes...")
    n_segments_runtimes = {}

    for hp, story_path in sorted(story_paths.items()):
        logger.info(f"\nHyperparameter-{hp}:")
        try:
            n_segments, runtime = extract_iteration_0_runtime(story_path)
            n_segments_runtimes[n_segments] = runtime
        except Exception as e:
            logger.warning(f"Failed to extract iteration_0 for hyperparameter-{hp}: {e}")
            continue

    if not n_segments_runtimes:
        raise ValueError(f"No valid iteration_0 data found for {story_name}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for plotting
    n_segments_list = sorted(n_segments_runtimes.keys())
    runtimes = [n_segments_runtimes[ns] for ns in n_segments_list]

    # Create line plot
    ax.plot(
        n_segments_list,
        runtimes,
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
    ax.set_xlabel('Number of Segments (n_segments)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Runtime (seconds)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # Set title
    formatted_story_name = story_name.replace('_', ' ')
    ax.set_title(
        f'{formatted_story_name} - {direction.capitalize()} (iteration_0)',
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold',
        pad=20
    )

    # Set x-axis
    ax.set_xticks(n_segments_list)
    ax.set_xticklabels(n_segments_list, fontsize=TICK_FONT_SIZE)

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
        description='Create single story iteration runtime line plot across all hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot forward transformations for a story (all hyperparameters)
  python plot_scripts/plot_story_runtime.py \\
      --direction forward \\
      --story "A_Piece_of_Yellow_Soap"

  # Plot inverse transformations with custom output
  python plot_scripts/plot_story_runtime.py \\
      --direction inverse \\
      --story "About_a_Hum" \\
      --output output_analysis_runtime/about_a_hum_inverse.png

  # With verbose logging
  python plot_scripts/plot_story_runtime.py \\
      --direction forward \\
      --story "A_Piece_of_Yellow_Soap" \\
      --verbose
        """
    )

    parser.add_argument(
        '--direction',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Direction: forward or inverse'
    )

    parser.add_argument(
        '--story',
        type=str,
        required=True,
        help='Story name (folder name, e.g., A_Piece_of_Yellow_Soap)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PNG file (default: output_analysis_runtime/{story}_{direction}.png)'
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
    logger.info("CREATING SINGLE STORY RUNTIME PLOT")
    logger.info("=" * 60)
    logger.info(f"Direction: {args.direction}")
    logger.info(f"Story: {args.story}")

    # Default output file if not specified
    if args.output is None:
        output_file = f'output_analysis_runtime/{args.story}_{args.direction}.png'
    else:
        output_file = args.output

    logger.info(f"Output: {output_file}")

    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'output'

    try:
        create_story_runtime_plot(
            output_dir=output_dir,
            direction=args.direction,
            story_name=args.story,
            output_file=output_file
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
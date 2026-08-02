"""
Story Name Mapping CSV Generation Script

Creates a mapping from original story names to abbreviated names (S_1, S_2, etc.)
for use in publication plots.

Forward stories (collectivistic): S_1 to S_28
Inverse stories (individualistic): S_29 onwards

Stories are numbered alphabetically within each problem type.

Output: CSV file in output_analysis/story_name_mapping.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Story Discovery
# ==========================================================

def get_stories_from_directory(
        data_dir: str,
        problem_type: str
) -> list:
    """
    Get list of stories from data directory.

    Args:
        data_dir: Data directory
        problem_type: "forward" or "inverse"

    Returns:
        Sorted list of story names (without .txt extension)
    """
    if problem_type == "forward":
        # Forward = collectivistic stories
        story_dir = Path(data_dir) / "collectivistic-stories-all"
    else:  # inverse
        # Inverse = individualistic stories
        story_dir = Path(data_dir) / "individualistic-rags-to-riches-stories-subset"

    if not story_dir.exists():
        raise FileNotFoundError(f"Story directory not found: {story_dir}")

    # Get all .txt files
    story_files = sorted(story_dir.glob("*.txt"))
    story_names = [f.stem for f in story_files]

    logger.info(f"Found {len(story_names)} {problem_type} stories")

    return story_names


# ==========================================================
# Mapping Creation
# ==========================================================

def create_story_mapping(
        data_dir: str = "data"
) -> pd.DataFrame:
    """
    Create story name mapping DataFrame.

    Args:
        data_dir: Data directory

    Returns:
        DataFrame with mapping
    """
    logger.info("=" * 60)
    logger.info("CREATING STORY NAME MAPPING")
    logger.info("=" * 60)

    # Get forward stories (collectivistic)
    forward_stories = get_stories_from_directory(data_dir, "forward")

    # Get inverse stories (individualistic)
    inverse_stories = get_stories_from_directory(data_dir, "inverse")

    # Create mapping
    rows = []
    story_counter = 1

    # Forward stories: S_1 to S_N
    for story_name in sorted(forward_stories):
        rows.append({
            'original_name': story_name,
            'abbreviated_name': f'S_{story_counter}',
            'problem_type': 'forward'
        })
        story_counter += 1

    logger.info(f"\nForward stories: S_1 to S_{story_counter - 1}")

    # Inverse stories: S_(N+1) onwards
    inverse_start = story_counter
    for story_name in sorted(inverse_stories):
        rows.append({
            'original_name': story_name,
            'abbreviated_name': f'S_{story_counter}',
            'problem_type': 'inverse'
        })
        story_counter += 1

    logger.info(f"Inverse stories: S_{inverse_start} to S_{story_counter - 1}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df


# ==========================================================
# Main Function
# ==========================================================

def generate_story_mapping_csv(
        data_dir: str = "data",
        output_dir: str = "output_analysis"
):
    """
    Generate story name mapping CSV.

    Args:
        data_dir: Data directory
        output_dir: Output directory
    """
    # Create mapping
    df = create_story_mapping(data_dir)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "story_name_mapping.csv"
    df.to_csv(csv_file, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("STORY NAME MAPPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Saved: {csv_file}")
    logger.info(f"  Total stories: {len(df)}")
    logger.info(f"  Forward: {len(df[df['problem_type'] == 'forward'])}")
    logger.info(f"  Inverse: {len(df[df['problem_type'] == 'inverse'])}")

    # Show preview
    logger.info("\nPreview (first 5 and last 5):")
    print("\nFirst 5:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5:")
    print(df.tail(5).to_string(index=False))


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate story name mapping CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate mapping
  python plot_scripts/create_story_name_mapping.py

  # Custom data directory
  python plot_scripts/create_story_name_mapping.py --data-dir /path/to/data
        """
    )

    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='output_analysis', help='Output directory')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_story_mapping_csv(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
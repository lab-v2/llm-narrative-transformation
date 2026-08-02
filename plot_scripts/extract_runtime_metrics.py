import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


def parse_timestamp(timestamp_str):
    """Parse ISO format timestamp string to datetime object."""
    return datetime.fromisoformat(timestamp_str)


def extract_metrics_from_story(json_path):
    """
    Extract metrics from cost_tracking_transformation.json file.
    Returns None if n_segments == 0, otherwise returns metrics dict.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

    n_segments = data.get('n_segments', 0)

    # Skip if n_segments is 0
    if n_segments == 0:
        return None

    # Extract metrics
    k_features = data.get('k_features', 0)
    per_story_breakdown = data.get('per_story_breakdown', [])

    if not per_story_breakdown:
        return None

    # Sum input and output tokens
    total_input_tokens = sum(item.get('input_tokens', 0) for item in per_story_breakdown)
    total_output_tokens = sum(item.get('output_tokens', 0) for item in per_story_breakdown)

    # Calculate runtime in seconds
    first_timestamp = parse_timestamp(per_story_breakdown[0]['timestamp'])
    last_timestamp = parse_timestamp(per_story_breakdown[-1]['timestamp'])
    runtime_seconds = (last_timestamp - first_timestamp).total_seconds()

    return {
        'n_segments': n_segments,
        'k_features': k_features,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'runtime_seconds': runtime_seconds
    }


def process_hyperparameters(output_dir, direction='forward'):
    """
    Process all hyperparameter configurations and collect metrics.
    """
    records = []
    hyperparameters = [1, 2, 3, 4, 5, 10, 20]

    for hp in hyperparameters:
        hp_folder = output_dir / f'hyperparameter-{hp}' / 'phase2' / 'gpt-4o' / direction

        if not hp_folder.exists():
            print(f"Warning: {hp_folder} does not exist, skipping hyperparameter-{hp}")
            continue

        # Iterate through story folders
        for story_folder in hp_folder.iterdir():
            if not story_folder.is_dir():
                continue

            story_name = story_folder.name
            json_path = story_folder / 'iteration_0' / 'cost_tracking_transformation.json'

            if not json_path.exists():
                print(f"Warning: {json_path} does not exist, skipping {story_name}")
                continue

            metrics = extract_metrics_from_story(json_path)

            if metrics is not None:
                record = {
                    'hyperparameter': hp,
                    'story_name': story_name,
                    'n_segments': metrics['n_segments'],
                    'k_features': metrics['k_features'],
                    'total_input_tokens': metrics['total_input_tokens'],
                    'total_output_tokens': metrics['total_output_tokens'],
                    'runtime_seconds': metrics['runtime_seconds']
                }
                records.append(record)
                print(f"Processed: hyperparameter-{hp}/{story_name}")
            else:
                print(f"Skipped: hyperparameter-{hp}/{story_name} (n_segments == 0 or empty)")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description='Extract runtime metrics from cost_tracking_transformation.json files'
    )
    parser.add_argument(
        '--direction',
        choices=['forward', 'inverse'],
        default='forward',
        help='Process forward or inverse transformations (default: forward)'
    )

    args = parser.parse_args()

    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'output'
    output_analysis_dir = project_root.parent / 'llm-narrative-transformation'/'output_analysis_runtime'

    # Create output directory if it doesn't exist
    output_analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.direction} transformations...")
    print(f"Output directory: {output_dir}")
    print(f"Analysis will be saved to: {output_analysis_dir}")
    print()

    # Process hyperparameters
    df = process_hyperparameters(output_dir, direction=args.direction)

    if df.empty:
        print("No valid data found to process.")
        return

    # Save to CSV
    csv_path = output_analysis_dir / 'runtime_metrics.csv'
    df.to_csv(csv_path, index=False)

    print()
    print(f"✓ Successfully saved {len(df)} records to {csv_path}")
    print()
    print("Summary Statistics:")
    print(df.describe())
    print()
    print("Column names:")
    print(df.columns.tolist())


if __name__ == '__main__':
    main()
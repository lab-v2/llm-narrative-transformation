#!/usr/bin/env python3
"""
BERTScore analysis script (Standalone) for comparing abduction vs baseline story transformations.
Compares transformed stories against the original story to measure semantic preservation.
"""

import os
import csv
import argparse
import logging
from pathlib import Path
from bert_score import score
import pandas as pd

logger = logging.getLogger(__name__)


def read_story_file(filepath):
    """Read a story file and return its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def compute_bertscore(original_text, candidate_text):
    """
    Compute BERTScore between original and candidate texts.
    Returns precision, recall, f1 scores.
    """
    if not original_text or not candidate_text:
        return None, None, None

    try:
        P, R, F1 = score(
            [candidate_text],
            [original_text],
            lang="en",
            verbose=False
        )
        return float(P[0]), float(R[0]), float(F1[0])
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return None, None, None


def process_stories(
        data_root,
        tuned_model_name,
        model_name_for_survey,
        original_stories_type,
        output_csv
):
    """
    Process all stories in the evaluation_data directory.

    Args:
        data_root: Path to evaluation_data root directory
        tuned_model_name: Name of the tuned model that generated transformed stories
        model_name_for_survey: Name of the model used for survey/diagnosis
        original_stories_type: Type of original stories (individualistic/collectivistic)
        output_csv: Path to output CSV file
    """
    logger.info("=" * 70)
    logger.info("BERTSCORE ANALYSIS (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output CSV: {output_csv}\n")

    # Sanitize model names
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")

    # Build paths
    data_root = Path(data_root)
    model_folder = data_root / sanitized_tuned / original_stories_type

    if not model_folder.exists():
        logger.error(f"Model folder not found: {model_folder}")
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    # Prepare output
    os.makedirs(Path(output_csv).parent, exist_ok=True)

    results = []

    # Iterate through story folders
    story_folders = sorted([d for d in model_folder.iterdir() if d.is_dir()])
    logger.info(f"Found {len(story_folders)} stories to process\n")

    for idx, story_folder in enumerate(story_folders, 1):
        story_name = story_folder.name
        logger.info(f"[{idx}/{len(story_folders)}] Processing: {story_name}")

        # Read original story
        original_story_path = story_folder / "original_story.txt"
        original_text = read_story_file(original_story_path)

        if not original_text:
            logger.warning(f"  Could not read original story, skipping")
            continue

        # Read abduction transformed story
        abduction_path = story_folder / "abduction_transformed.txt"
        abduction_text = read_story_file(abduction_path)

        # Read baseline transformed story
        baseline_path = story_folder / "baseline_transformed.txt"
        baseline_text = read_story_file(baseline_path)

        if not abduction_text and not baseline_text:
            logger.warning(f"  Could not read any transformed stories, skipping")
            continue

        # Compute BERTScores
        ab_precision, ab_recall, ab_f1 = None, None, None
        bs_precision, bs_recall, bs_f1 = None, None, None

        if abduction_text:
            ab_precision, ab_recall, ab_f1 = compute_bertscore(original_text, abduction_text)
            if ab_f1 is not None:
                logger.info(f"  Abduction - P: {ab_precision:.4f}, R: {ab_recall:.4f}, F1: {ab_f1:.4f}")

        if baseline_text:
            bs_precision, bs_recall, bs_f1 = compute_bertscore(original_text, baseline_text)
            if bs_f1 is not None:
                logger.info(f"  Baseline  - P: {bs_precision:.4f}, R: {bs_recall:.4f}, F1: {bs_f1:.4f}")

        # Store results
        results.append({
            'story_name': story_name,
            'original_stories_type': original_stories_type,
            'abduction_bertscore_precision': ab_precision if ab_precision is not None else '',
            'abduction_bertscore_recall': ab_recall if ab_recall is not None else '',
            'abduction_bertscore_f1': ab_f1 if ab_f1 is not None else '',
            'baseline_bertscore_precision': bs_precision if bs_precision is not None else '',
            'baseline_bertscore_recall': bs_recall if bs_recall is not None else '',
            'baseline_bertscore_f1': bs_f1 if bs_f1 is not None else ''
        })

    # Write CSV
    if results:
        fieldnames = [
            'story_name',
            'original_stories_type',
            'abduction_bertscore_precision',
            'abduction_bertscore_recall',
            'abduction_bertscore_f1',
            'baseline_bertscore_precision',
            'baseline_bertscore_recall',
            'baseline_bertscore_f1'
        ]

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"\n✓ Results saved to: {output_csv}")
        logger.info(f"✓ Processed {len(results)} stories")

        # Print summary statistics
        df = pd.read_csv(output_csv)
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Abduction F1 - Mean: {df['abduction_bertscore_f1'].mean():.4f}, Std: {df['abduction_bertscore_f1'].std():.4f}")
        logger.info(f"Baseline F1  - Mean: {df['baseline_bertscore_f1'].mean():.4f}, Std: {df['baseline_bertscore_f1'].std():.4f}")
        logger.info("=" * 70)
    else:
        logger.error("No results to save.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute BERTScores for story transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline model
  python plot_scripts_standalone/bertscore_analysis_standalone.py \\
    --tuned-model-name "gpt-4o-mini-2024-07-18" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"

  # Fine-tuned model
  python plot_scripts_standalone/bertscore_analysis_standalone.py \\
    --tuned-model-name "ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"
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
        '--data-root',
        type=str,
        default='evaluation_data',
        help='Path to evaluation_data root directory (default: evaluation_data)'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Path to output CSV file (if not provided, auto-generated)'
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

    # Generate filename if not provided
    if args.output_csv is None:
        sanitized_tuned = args.tuned_model_name.replace("/", "-").replace(":", "-")
        sanitized_survey = args.model_name_for_survey.replace("/", "-").replace(":", "-")
        filename = f"bertscore_results_{sanitized_tuned}_{sanitized_survey}_{args.original_stories_type}.csv"
        args.output_csv = f"output_analysis_standalone/bert_scores_csv/{filename}"

    logger.info(f"Arguments:")
    logger.info(f"  --tuned-model-name: {args.tuned_model_name}")
    logger.info(f"  --model-name-for-survey: {args.model_name_for_survey}")
    logger.info(f"  --original-stories-type: {args.original_stories_type}")
    logger.info(f"  --data-root: {args.data_root}")
    logger.info(f"  --output-csv: {args.output_csv}")
    logger.info(f"  --verbose: {args.verbose}\n")

    process_stories(
        args.data_root,
        args.tuned_model_name,
        args.model_name_for_survey,
        args.original_stories_type,
        args.output_csv
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Token-Based Divergence Computation Script (Standalone)

Computes KL divergence and JS divergence between original and transformed stories
using token-based probability distributions (word frequencies).

This is a free, lexical-based metric (no API calls needed).

Output: CSV files saved to output_analysis_standalone/divergences/{tuned_model}/{survey_model}/{story_type}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ==========================================================
# Tokenization and Distribution
# ==========================================================

def read_file(filepath):
    """Read text file and return content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def tokenize(text):
    """Simple tokenization: lowercase and split on non-alphanumeric"""
    if text is None:
        return []
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def get_token_distribution(tokens, smoothing=1e-5):
    """
    Get probability distribution from tokens with Laplace smoothing.

    Args:
        tokens: List of tokens
        smoothing: Smoothing parameter (default 1e-5)

    Returns:
        Dict: {token: probability}
    """
    if not tokens:
        return {}

    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    vocab_size = len(token_counts)
    distribution = {}

    for token, count in token_counts.items():
        distribution[token] = (count + smoothing) / (total_tokens + smoothing * vocab_size)

    return distribution


# ==========================================================
# Divergence Computation
# ==========================================================

def compute_kl_divergence(dist_p, dist_q, smoothing=1e-5):
    """
    Compute KL(P || Q) from token distributions.
    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Args:
        dist_p: Probability distribution P (dict)
        dist_q: Probability distribution Q (dict)
        smoothing: Smoothing for unseen tokens

    Returns:
        Float: KL divergence value
    """
    if not dist_p or not dist_q:
        return None

    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    kl_sum = 0.0

    for token in all_tokens:
        p = dist_p.get(token, smoothing)
        q = dist_q.get(token, smoothing)
        kl_sum += p * np.log(p / q)

    return float(kl_sum)


def compute_js_divergence(dist_p, dist_q, smoothing=1e-5):
    """
    Compute Jensen-Shannon Divergence (symmetric, bounded 0-1).
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = (P + Q) / 2

    Args:
        dist_p: Probability distribution P
        dist_q: Probability distribution Q
        smoothing: Smoothing parameter

    Returns:
        Float: JS divergence normalized to 0-1
    """
    if not dist_p or not dist_q:
        return None

    all_tokens = set(dist_p.keys()) | set(dist_q.keys())

    # Compute midpoint distribution M
    dist_m = {}
    for token in all_tokens:
        p = dist_p.get(token, smoothing)
        q = dist_q.get(token, smoothing)
        dist_m[token] = (p + q) / 2

    # Compute KL(P || M) and KL(Q || M)
    kl_pm = compute_kl_divergence(dist_p, dist_m, smoothing)
    kl_qm = compute_kl_divergence(dist_q, dist_m, smoothing)

    # Compute JSD and normalize to 0-1
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    jsd_normalized = jsd / np.log(2)

    return float(jsd_normalized)


# ==========================================================
# Main Processing
# ==========================================================

def process_stories_standalone(
        tuned_model_name: str,
        model_name_for_survey: str,
        original_stories_type: str,
        evaluation_data_dir: str = 'evaluation_data',
        output_dir: str = 'output_analysis_standalone/divergences'
):
    """
    Process all stories and compute token-based divergences.

    Args:
        tuned_model_name: Name of tuned model that generated transformed stories
        model_name_for_survey: Name of model used for survey/diagnosis
        original_stories_type: Type of original stories (individualistic/collectivistic)
        evaluation_data_dir: Base evaluation data directory
        output_dir: Base output directory
    """
    logger.info("=" * 70)
    logger.info("TOKEN-BASED DIVERGENCE COMPUTATION (STANDALONE)")
    logger.info("=" * 70)
    logger.info(f"Tuned Model: {tuned_model_name}")
    logger.info(f"Survey Model: {model_name_for_survey}")
    logger.info(f"Original Stories Type: {original_stories_type}")

    # Sanitize model names
    sanitized_tuned = tuned_model_name.replace("/", "-").replace(":", "-")
    sanitized_survey = model_name_for_survey.replace("/", "-").replace(":", "-")

    # Base paths
    eval_base = Path(evaluation_data_dir) / sanitized_tuned / original_stories_type

    if not eval_base.exists():
        logger.error(f"Evaluation data directory not found: {eval_base}")
        return

    # Get all story directories
    story_dirs = [d for d in eval_base.iterdir() if d.is_dir()]

    if not story_dirs:
        logger.error(f"No story directories found in {eval_base}")
        return

    logger.info(f"Found {len(story_dirs)} stories to process\n")

    # Results storage
    results_kl_trans_orig = []  # KL(transformed || original)
    results_kl_orig_trans = []  # KL(original || transformed)
    results_js = []  # JS divergence

    for idx, story_dir in enumerate(sorted(story_dirs), 1):
        story_name = story_dir.name
        logger.info(f"[{idx}/{len(story_dirs)}] Processing: {story_name}")

        # Read and tokenize original
        original_path = story_dir / "original_story.txt"
        original_text = read_file(original_path)

        if not original_text:
            logger.warning(f"  Original story not found, skipping")
            continue

        original_tokens = tokenize(original_text)
        original_dist = get_token_distribution(original_tokens)

        logger.info(f"  Original: {len(original_tokens)} tokens")

        # Initialize result rows
        row_kl_to = {'story_name': story_name}
        row_kl_ot = {'story_name': story_name}
        row_js = {'story_name': story_name}

        # Process baseline transformed
        baseline_path = story_dir / "baseline_transformed.txt"
        baseline_text = read_file(baseline_path)

        if baseline_text:
            baseline_tokens = tokenize(baseline_text)
            baseline_dist = get_token_distribution(baseline_tokens)

            logger.info(f"  Baseline: {len(baseline_tokens)} tokens")

            # Compute divergences
            kl_baseline_orig = compute_kl_divergence(baseline_dist, original_dist)
            kl_orig_baseline = compute_kl_divergence(original_dist, baseline_dist)
            js_baseline = compute_js_divergence(baseline_dist, original_dist)

            row_kl_to['baseline_kl'] = kl_baseline_orig
            row_kl_ot['baseline_kl'] = kl_orig_baseline
            row_js['baseline_js'] = js_baseline

            logger.info(f"    KL(baseline||orig): {kl_baseline_orig:.4f}, JS: {js_baseline:.4f}")
        else:
            row_kl_to['baseline_kl'] = None
            row_kl_ot['baseline_kl'] = None
            row_js['baseline_js'] = None
            logger.warning("  Baseline not found")

        # Process abduction transformed
        abduction_path = story_dir / "abduction_transformed.txt"
        abduction_text = read_file(abduction_path)

        if abduction_text:
            abduction_tokens = tokenize(abduction_text)
            abduction_dist = get_token_distribution(abduction_tokens)

            logger.info(f"  Abduction: {len(abduction_tokens)} tokens")

            # Compute divergences
            kl_abd_orig = compute_kl_divergence(abduction_dist, original_dist)
            kl_orig_abd = compute_kl_divergence(original_dist, abduction_dist)
            js_abd = compute_js_divergence(abduction_dist, original_dist)

            row_kl_to['abduction_kl'] = kl_abd_orig
            row_kl_ot['abduction_kl'] = kl_orig_abd
            row_js['abduction_js'] = js_abd

            logger.info(f"    KL(abduction||orig): {kl_abd_orig:.4f}, JS: {js_abd:.4f}")
        else:
            row_kl_to['abduction_kl'] = None
            row_kl_ot['abduction_kl'] = None
            row_js['abduction_js'] = None
            logger.warning("  Abduction not found")

        results_kl_trans_orig.append(row_kl_to)
        results_kl_orig_trans.append(row_kl_ot)
        results_js.append(row_js)
        logger.info("")

    if not results_js:
        logger.error("No results to save")
        return

    # Create output directory
    output_path = Path(output_dir) / sanitized_tuned / sanitized_survey / original_stories_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to CSV files
    df_kl_to = pd.DataFrame(results_kl_trans_orig)
    df_kl_ot = pd.DataFrame(results_kl_orig_trans)
    df_js = pd.DataFrame(results_js)

    kl_to_file = output_path / 'kl_divergence_transformed_original.csv'
    kl_ot_file = output_path / 'kl_divergence_original_transformed.csv'
    js_file = output_path / 'js_divergence.csv'

    df_kl_to.to_csv(kl_to_file, index=False)
    df_kl_ot.to_csv(kl_ot_file, index=False)
    df_js.to_csv(js_file, index=False)

    logger.info(f"✓ KL(transformed||original) saved to: {kl_to_file}")
    logger.info(f"✓ KL(original||transformed) saved to: {kl_ot_file}")
    logger.info(f"✓ JS divergence saved to: {js_file}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Stories processed: {len(results_js)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 70)

    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info("\nKL(transformed || original):")
    numeric_cols = [col for col in df_kl_to.columns if col != 'story_name']
    if numeric_cols:
        logger.info("\n" + df_kl_to[numeric_cols].describe().to_string())

    logger.info("\nJS Divergence:")
    numeric_cols = [col for col in df_js.columns if col != 'story_name']
    if numeric_cols:
        logger.info("\n" + df_js[numeric_cols].describe().to_string())


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute token-based KL and JS divergences for survey data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for baseline model
  python plot_scripts_standalone/create_divergence_csv_standalone.py \\
    --tuned-model-name "gpt-4o-mini-2024-07-18" \\
    --model-name-for-survey "gpt-4o-mini-2024-07-18" \\
    --original-stories-type "individualistic"

  # Compute for fine-tuned model
  python plot_scripts_standalone/create_divergence_csv_standalone.py \\
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
        '--evaluation-data-dir',
        type=str,
        default='evaluation_data',
        help='Base evaluation data directory (default: evaluation_data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_standalone/divergences',
        help='Output directory (default: output_analysis_standalone/divergences)'
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

    logger.info(f"Arguments:")
    logger.info(f"  --tuned-model-name: {args.tuned_model_name}")
    logger.info(f"  --model-name-for-survey: {args.model_name_for_survey}")
    logger.info(f"  --original-stories-type: {args.original_stories_type}")
    logger.info(f"  --evaluation-data-dir: {args.evaluation_data_dir}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --verbose: {args.verbose}")

    # Run processing
    try:
        process_stories_standalone(
            tuned_model_name=args.tuned_model_name,
            model_name_for_survey=args.model_name_for_survey,
            original_stories_type=args.original_stories_type,
            evaluation_data_dir=args.evaluation_data_dir,
            output_dir=args.output_dir
        )

        logger.info("\n✓ Processing complete!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()

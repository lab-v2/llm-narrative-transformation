"""
Abstract Metrics Computation Script with BertScore

Computes the improvement percentages for the paper abstract:
- x%: Forward rating improvement (median)
- y%: Forward KL divergence improvement (Q2)
- z%: Inverse rating improvement (median)
- w%: Inverse KL divergence improvement (Q2)
- p%: Forward BertScore F1 improvement (median)
- q%: Inverse BertScore F1 improvement (median)

Output: CSV file with metrics for all models
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Metric Computation
# ==========================================================

def compute_abstract_metrics(
        model_name: str,
        summary_dir: str = "output_analysis/cross_model_summary",
        bert_dir: str = "output_analysis/bert_scores_csv",
        model_name_ratings: str = None
) -> dict:
    """
    Compute abstract metrics for a specific model.

    Args:
        model_name: Model name for ratings/KL files (with forward slashes and dots)
                   Examples: 'bedrock/us.deepseek.r1-v1-0', 'xai/grok-4-fast-reasoning', 'claude-sonnet-4-5'
        summary_dir: Cross-model summary directory
        bert_dir: BertScore CSV directory
        model_name_ratings: Model name for ratings/KL files. If None, uses model_name as-is.
                           Automatically converts "/" → "-" for BertScore file lookups.

    Returns:
        Dict with x, y, z, w, p, q percentages
    """
    logger.info(f"\nComputing metrics for {model_name}...")

    results = {'model': model_name}
    
    # Convert model name for BertScore lookups:
    # ratings/KL: bedrock/us.deepseek.r1-v1-0 → BertScore: bedrock-us.deepseek.r1-v1-0
    # ratings/KL: xai/grok-4-fast-reasoning → BertScore: xai-grok-4-fast-reasoning
    model_name_bert = model_name.replace('/', '-')
    
    # For ratings/KL, use model_name as-is (with forward slashes)
    if model_name_ratings is None:
        model_name_ratings = model_name

    # ==========================================
    # FORWARD PROBLEM
    # ==========================================

    # Load forward median ratings
    forward_median_file = Path(summary_dir) / "forward" / "model_comparison_median.csv"

    if forward_median_file.exists():
        df_forward = pd.read_csv(forward_median_file)

        # Get baseline and abductive medians (averaged across all stories)
        baseline_col = f'{model_name_ratings}_baseline'
        abd_col = f'{model_name_ratings}_transformed'

        if baseline_col in df_forward.columns and abd_col in df_forward.columns:
            baseline_median = df_forward[baseline_col].mean()  # Mean of medians across stories
            abd_median = df_forward[abd_col].mean()

            # x% = rating improvement (lower is better)
            if baseline_median > 0:
                x_pct = ((baseline_median - abd_median) / baseline_median) * 100
                results['x_rating_improvement_forward'] = round(x_pct, 2)
                logger.info(f"  Forward rating: Baseline={baseline_median:.2f}, Abd={abd_median:.2f}, x={x_pct:.2f}%")
            else:
                results['x_rating_improvement_forward'] = None
        else:
            logger.warning(f"  Columns not found for forward ratings (tried: {baseline_col}, {abd_col})")
            results['x_rating_improvement_forward'] = None
    else:
        logger.warning(f"  Forward median file not found")
        results['x_rating_improvement_forward'] = None

    # Load forward KL quartiles
    forward_kl_file = Path(summary_dir) / "forward" / "kl_divergence_quartiles.csv"

    if forward_kl_file.exists():
        df_kl_forward = pd.read_csv(forward_kl_file)
        model_row = df_kl_forward[df_kl_forward['model'] == model_name_ratings]

        if len(model_row) > 0:
            baseline_q2 = model_row['baseline_q2'].values[0]
            abd_q2 = model_row['abd_q2'].values[0]

            # y% = KL improvement (lower is better)
            if pd.notna(baseline_q2) and baseline_q2 > 0:
                y_pct = ((baseline_q2 - abd_q2) / baseline_q2) * 100
                results['y_kl_improvement_forward'] = round(y_pct, 2)
                logger.info(f"  Forward KL: Baseline_Q2={baseline_q2:.4f}, Abd_Q2={abd_q2:.4f}, y={y_pct:.2f}%")
            else:
                results['y_kl_improvement_forward'] = None
        else:
            logger.warning(f"  Model not found in forward KL quartiles (tried: {model_name_ratings})")
            results['y_kl_improvement_forward'] = None
    else:
        logger.warning(f"  Forward KL file not found")
        results['y_kl_improvement_forward'] = None

    # Load forward BertScore F1
    forward_bert_file = Path(bert_dir) / f"bertscore_results_{model_name_bert}_forward.csv"

    if forward_bert_file.exists():
        df_bert_forward = pd.read_csv(forward_bert_file)
        
        # Extract F1 scores for baseline and abduction
        baseline_f1_scores = df_bert_forward['baseline_bertscore_f1'].values
        abduction_f1_scores = df_bert_forward['abduction_bertscore_f1'].values
        
        # Compute improvements for each story
        improvements = []
        for baseline_f1, abd_f1 in zip(baseline_f1_scores, abduction_f1_scores):
            if pd.notna(baseline_f1) and pd.notna(abd_f1) and baseline_f1 > 0:
                improvement = ((abd_f1 - baseline_f1) / baseline_f1) * 100
                improvements.append(improvement)
        
        if improvements:
            # p% = BertScore F1 improvement (higher is better, so abd - baseline)
            p_pct = np.median(improvements)
            results['p_bertscore_f1_improvement_forward'] = round(p_pct, 2)
            baseline_f1_median = np.median(baseline_f1_scores)
            abd_f1_median = np.median(abduction_f1_scores)
            logger.info(f"  Forward BertScore F1: Baseline_Median={baseline_f1_median:.4f}, Abd_Median={abd_f1_median:.4f}, p={p_pct:.2f}%")
        else:
            logger.warning(f"  Could not compute forward BertScore F1 improvements")
            results['p_bertscore_f1_improvement_forward'] = None
    else:
        logger.warning(f"  Forward BertScore file not found: {forward_bert_file}")
        results['p_bertscore_f1_improvement_forward'] = None

    # ==========================================
    # INVERSE PROBLEM
    # ==========================================

    # Load inverse median ratings
    inverse_median_file = Path(summary_dir) / "inverse" / "model_comparison_median.csv"

    if inverse_median_file.exists():
        df_inverse = pd.read_csv(inverse_median_file)

        # Get baseline and abductive medians
        baseline_col = f'{model_name_ratings}_baseline'
        abd_col = f'{model_name_ratings}_transformed'

        if baseline_col in df_inverse.columns and abd_col in df_inverse.columns:
            baseline_median = df_inverse[baseline_col].mean()
            abd_median = df_inverse[abd_col].mean()

            # z% = rating improvement (higher is better for inverse)
            if baseline_median > 0:
                z_pct = ((abd_median - baseline_median) / baseline_median) * 100
                results['z_rating_improvement_inverse'] = round(z_pct, 2)
                logger.info(f"  Inverse rating: Baseline={baseline_median:.2f}, Abd={abd_median:.2f}, z={z_pct:.2f}%")
            else:
                results['z_rating_improvement_inverse'] = None
        else:
            logger.warning(f"  Columns not found for inverse ratings (tried: {baseline_col}, {abd_col})")
            results['z_rating_improvement_inverse'] = None
    else:
        logger.warning(f"  Inverse median file not found")
        results['z_rating_improvement_inverse'] = None

    # Load inverse KL quartiles
    inverse_kl_file = Path(summary_dir) / "inverse" / "kl_divergence_quartiles.csv"

    if inverse_kl_file.exists():
        df_kl_inverse = pd.read_csv(inverse_kl_file)
        model_row = df_kl_inverse[df_kl_inverse['model'] == model_name_ratings]

        if len(model_row) > 0:
            baseline_q2 = model_row['baseline_q2'].values[0]
            abd_q2 = model_row['abd_q2'].values[0]

            # w% = KL improvement (lower is better)
            if pd.notna(baseline_q2) and baseline_q2 > 0:
                w_pct = ((baseline_q2 - abd_q2) / baseline_q2) * 100
                results['w_kl_improvement_inverse'] = round(w_pct, 2)
                logger.info(f"  Inverse KL: Baseline_Q2={baseline_q2:.4f}, Abd_Q2={abd_q2:.4f}, w={w_pct:.2f}%")
            else:
                results['w_kl_improvement_inverse'] = None
        else:
            logger.warning(f"  Model not found in inverse KL quartiles (tried: {model_name_ratings})")
            results['w_kl_improvement_inverse'] = None
    else:
        logger.warning(f"  Inverse KL file not found")
        results['w_kl_improvement_inverse'] = None

    # Load inverse BertScore F1
    inverse_bert_file = Path(bert_dir) / f"bertscore_results_{model_name_bert}_inverse.csv"

    if inverse_bert_file.exists():
        df_bert_inverse = pd.read_csv(inverse_bert_file)
        
        # Extract F1 scores for baseline and abduction
        baseline_f1_scores = df_bert_inverse['baseline_bertscore_f1'].values
        abduction_f1_scores = df_bert_inverse['abduction_bertscore_f1'].values
        
        # Compute improvements for each story
        improvements = []
        for baseline_f1, abd_f1 in zip(baseline_f1_scores, abduction_f1_scores):
            if pd.notna(baseline_f1) and pd.notna(abd_f1) and baseline_f1 > 0:
                improvement = ((abd_f1 - baseline_f1) / baseline_f1) * 100
                improvements.append(improvement)
        
        if improvements:
            # q% = BertScore F1 improvement (higher is better, so abd - baseline)
            q_pct = np.median(improvements)
            results['q_bertscore_f1_improvement_inverse'] = round(q_pct, 2)
            baseline_f1_median = np.median(baseline_f1_scores)
            abd_f1_median = np.median(abduction_f1_scores)
            logger.info(f"  Inverse BertScore F1: Baseline_Median={baseline_f1_median:.4f}, Abd_Median={abd_f1_median:.4f}, q={q_pct:.2f}%")
        else:
            logger.warning(f"  Could not compute inverse BertScore F1 improvements")
            results['q_bertscore_f1_improvement_inverse'] = None
    else:
        logger.warning(f"  Inverse BertScore file not found: {inverse_bert_file}")
        results['q_bertscore_f1_improvement_inverse'] = None

    return results


# ==========================================================
# Main Function
# ==========================================================

def generate_abstract_metrics(
        models: list,
        summary_dir: str = "output_analysis/cross_model_summary",
        bert_dir: str = "output_analysis/bert_scores_csv",
        output_dir: str = "output_analysis/cross_model_summary",
        model_ratings_map: dict = None
):
    """
    Generate abstract metrics CSV with BertScore.

    Args:
        models: List of model names (for BertScore files)
        summary_dir: Summary directory
        bert_dir: BertScore CSV directory
        output_dir: Output directory
        model_ratings_map: Dict mapping model names to their ratings/KL names. 
                          If None, uses automatic conversion (dots to underscores)
    """
    logger.info("=" * 60)
    logger.info("COMPUTING ABSTRACT METRICS WITH BERTSCORE")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")

    results = []

    for model_name in models:
        # Get ratings model name from map, or use automatic conversion
        model_name_ratings = None
        if model_ratings_map and model_name in model_ratings_map:
            model_name_ratings = model_ratings_map[model_name]
        
        metrics = compute_abstract_metrics(model_name, summary_dir, bert_dir, model_name_ratings)
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns: x, y, z, w (original), then p, q (new BertScore)
    cols = ['model',
            'x_rating_improvement_forward', 'y_kl_improvement_forward',
            'p_bertscore_f1_improvement_forward',
            'z_rating_improvement_inverse', 'w_kl_improvement_inverse',
            'q_bertscore_f1_improvement_inverse']
    df = df[cols]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "abstract_metrics_with_bert.csv"
    df.to_csv(csv_file, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("ABSTRACT METRICS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Saved: {csv_file}")
    logger.info("\nSummary:")
    print("\n" + df.to_string(index=False))


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute abstract metrics with BertScore (x, y, z, w, p, q percentages)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-converts "/" to "-" for BertScore filenames)
  # Ratings/KL: bedrock/us.deepseek.r1-v1-0 → BertScore: bedrock-us.deepseek.r1-v1-0
  python plot_scripts/compute_abstract_metrics_with_bert.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning \\
      --models bedrock/us.deepseek.r1-v1-0 bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 gpt-5.2
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='List of model names (for BertScore files)'
    )
    parser.add_argument('--summary-dir', default='output_analysis/cross_model_summary',
                        help='Summary directory')
    parser.add_argument('--bert-dir', default='output_analysis/bert_scores_csv',
                        help='BertScore CSV directory')
    parser.add_argument('--output-dir', default='output_analysis/cross_model_summary',
                        help='Output directory')
    parser.add_argument('--model-ratings-map', nargs='+', default=[],
                        help='Model name mapping for ratings/KL (format: bert_name=ratings_name)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    # Parse model ratings map
    model_ratings_map = {}
    if args.model_ratings_map:
        for mapping in args.model_ratings_map:
            if '=' in mapping:
                bert_name, ratings_name = mapping.split('=', 1)
                model_ratings_map[bert_name] = ratings_name

    try:
        generate_abstract_metrics(
            models=args.models,
            summary_dir=args.summary_dir,
            bert_dir=args.bert_dir,
            output_dir=args.output_dir,
            model_ratings_map=model_ratings_map if model_ratings_map else None
        )
        logger.info("\n✓ Complete!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

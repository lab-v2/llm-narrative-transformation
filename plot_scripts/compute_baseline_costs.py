"""
Baseline LLM Cost Computation Script

Computes token usage and costs for baseline transformation approach.
Separates transformation costs from post-transformation survey costs.

Output: CSV file in output_analysis/costs/baseline/{model}/{problem}/
"""

import json
import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Baseline Cost Computation
# ==========================================================

def compute_baseline_costs(
        model: str,
        problem_type: str,
        baseline_dir: str = "output/baseline"
) -> list:
    """
    Compute baseline costs per story.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        baseline_dir: Baseline output directory

    Returns:
        List of dicts with per-story costs
    """
    logger.info("Computing baseline costs...")

    sanitized_model = model.replace("/", "-").replace(":", "-")
    results_dir = Path(baseline_dir) / sanitized_model / problem_type

    if not results_dir.exists():
        raise FileNotFoundError(f"Baseline results not found: {results_dir}")

    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    logger.info(f"  Found {len(story_dirs)} stories")

    results = []

    for story_dir in sorted(story_dirs):
        story_name = story_dir.name
        cost_file = story_dir / "cost_tracking.json"

        if not cost_file.exists():
            logger.warning(f"  {story_name}: cost file not found, skipping")
            continue

        with open(cost_file, 'r') as f:
            cost_data = json.load(f)

        # Extract per_story_breakdown
        breakdown = cost_data.get('per_story_breakdown', [])

        # Find transformation entry (has "_transformation" suffix)
        transform_entry = None
        survey_entry = None

        for entry in breakdown:
            entry_name = entry.get('story_name', '')
            if entry_name.endswith('_transformation'):
                transform_entry = entry
            elif entry_name == story_name:
                survey_entry = entry

        # Extract transformation costs
        if transform_entry:
            transform_input = transform_entry.get('input_tokens', 0)
            transform_output = transform_entry.get('output_tokens', 0)
            transform_cost = transform_entry.get('cost_usd', 0.0)
        else:
            logger.warning(f"  {story_name}: transformation entry not found")
            transform_input = 0
            transform_output = 0
            transform_cost = 0.0

        # Extract post-survey costs
        if survey_entry:
            survey_input = survey_entry.get('input_tokens', 0)
            survey_output = survey_entry.get('output_tokens', 0)
            survey_cost = survey_entry.get('cost_usd', 0.0)
        else:
            logger.warning(f"  {story_name}: survey entry not found")
            survey_input = 0
            survey_output = 0
            survey_cost = 0.0

        # Combine
        total_input = transform_input + survey_input
        total_output = transform_output + survey_output
        total_tokens = total_input + total_output
        total_cost = transform_cost + survey_cost

        results.append({
            'story_name': story_name,
            'transformation_input_tokens': transform_input,
            'transformation_output_tokens': transform_output,
            'transformation_cost_usd': round(transform_cost, 4),
            'post_survey_input_tokens': survey_input,
            'post_survey_output_tokens': survey_output,
            'post_survey_cost_usd': round(survey_cost, 4),
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost, 4)
        })

        logger.info(f"  {story_name}: {total_tokens:,} tokens, ${total_cost:.2f}")

    return results


# ==========================================================
# Main Function
# ==========================================================

def generate_baseline_cost_csv(
        model: str,
        problem_type: str,
        baseline_dir: str = "output/baseline",
        output_dir: str = "output_analysis/costs/baseline"
):
    """
    Generate baseline cost CSV.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        baseline_dir: Baseline directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("GENERATING BASELINE LLM COSTS CSV")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")

    # Compute costs
    baseline_results = compute_baseline_costs(model, problem_type, baseline_dir)

    if not baseline_results:
        logger.warning("No baseline costs computed")
        return

    # Save CSV
    sanitized_model = model.replace("/", "-").replace(":", "-")
    output_path = Path(output_dir) / sanitized_model / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.DataFrame(baseline_results)
    csv_file = output_path / "llm_costs.csv"
    df_baseline.to_csv(csv_file, index=False)

    logger.info(f"\n✓ Baseline CSV saved to: {csv_file}")

    # Summary
    logger.info("\nBaseline Summary:")
    logger.info(f"  Stories: {len(baseline_results)}")
    logger.info(f"  Total tokens: {df_baseline['total_tokens'].sum():,}")
    logger.info(f"  Mean tokens/story: {df_baseline['total_tokens'].mean():.0f}")
    logger.info(f"  Total cost: ${df_baseline['total_cost_usd'].sum():.2f}")
    logger.info(f"  Mean cost/story: ${df_baseline['total_cost_usd'].mean():.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COST CSV COMPLETE")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate baseline LLM cost CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for GPT-4o forward
  python plot_scripts/compute_baseline_costs.py --model gpt-4o --problem forward

  # Generate for Claude inverse
  python plot_scripts/compute_baseline_costs.py --model claude-sonnet-4-5 --problem inverse
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--baseline-dir', default='output/baseline')
    parser.add_argument('--output-dir', default='output_analysis/costs/baseline')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_baseline_cost_csv(
            model=args.model,
            problem_type=args.problem,
            baseline_dir=args.baseline_dir,
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
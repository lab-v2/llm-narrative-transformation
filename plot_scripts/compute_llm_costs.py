"""
LLM Cost Computation Script

Computes token usage and costs for Phase 1 and Phase 2.

Phase 1: Total survey costs across all training stories
Phase 2: Per-story costs for iteration 0 (survey + transformation)

Output: CSV files in output_analysis/costs/phase1/ and output_analysis/costs/phase2/
"""

import json
import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Phase 1 Cost Computation
# ==========================================================

def compute_phase1_costs(
        model: str,
        problem_type: str,
        phase1_dir: str = "output/phase1"
) -> list:
    """
    Compute Phase 1 costs per story.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        phase1_dir: Phase 1 output directory

    Returns:
        List of dicts with per-story costs
    """
    logger.info("Computing Phase 1 costs...")

    sanitized_model = model.replace("/", "-").replace(":", "-")
    cost_file = Path(phase1_dir) / sanitized_model / problem_type / "cost_tracking.json"

    if not cost_file.exists():
        raise FileNotFoundError(f"Phase 1 cost file not found: {cost_file}")

    with open(cost_file, 'r') as f:
        cost_data = json.load(f)

    # Extract per-story breakdown
    per_story = cost_data.get('per_story_breakdown', [])

    logger.info(f"  Found {len(per_story)} stories")

    results = []
    for story_data in per_story:
        results.append({
            'story_name': story_data.get('story_name'),
            'input_tokens': story_data.get('input_tokens', 0),
            'output_tokens': story_data.get('output_tokens', 0),
            'total_tokens': story_data.get('input_tokens', 0) + story_data.get('output_tokens', 0),
            'cost_usd': story_data.get('cost_usd', 0.0)
        })

    total_tokens = sum(r['total_tokens'] for r in results)
    total_cost = sum(r['cost_usd'] for r in results)

    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Total cost: ${total_cost:.2f}")

    return results


# ==========================================================
# Phase 2 Cost Computation
# ==========================================================

def compute_phase2_costs(
        model: str,
        problem_type: str,
        iteration: int = 0,
        phase2_dir: str = "output/phase2"
) -> list:
    """
    Compute Phase 2 costs per story for specified iteration.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        iteration: Which iteration to use (default: 0)
        phase2_dir: Phase 2 output directory

    Returns:
        List of dicts with per-story costs
    """
    logger.info(f"Computing Phase 2 costs (iteration {iteration})...")

    sanitized_model = model.replace("/", "-").replace(":", "-")
    results_dir = Path(phase2_dir) / sanitized_model / problem_type

    if not results_dir.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {results_dir}")

    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    logger.info(f"  Found {len(story_dirs)} stories")

    results = []

    for story_dir in sorted(story_dirs):
        story_name = story_dir.name
        iter_dir = story_dir / f"iteration_{iteration}"

        if not iter_dir.exists():
            logger.warning(f"  {story_name}: iteration {iteration} not found, skipping")
            continue

        # Load survey costs
        survey_cost_file = iter_dir / "cost_tracking.json"
        if survey_cost_file.exists():
            with open(survey_cost_file, 'r') as f:
                survey_data = json.load(f)
            survey_input = survey_data.get('total_tokens', {}).get('input', 0)  # Changed
            survey_output = survey_data.get('total_tokens', {}).get('output', 0)  # Changed
            survey_cost = survey_data.get('total_cost_usd', 0.0)  # Changed
        else:
            logger.warning(f"  {story_name}: survey cost file not found")
            survey_input = 0
            survey_output = 0
            survey_cost = 0.0

        # Load transformation costs
        transform_cost_file = iter_dir / "cost_tracking_transformation.json"
        if transform_cost_file.exists():
            with open(transform_cost_file, 'r') as f:
                transform_data = json.load(f)
            transform_input = transform_data.get('total_tokens', {}).get('input', 0)  # Changed
            transform_output = transform_data.get('total_tokens', {}).get('output', 0)  # Changed
            transform_cost = transform_data.get('total_cost_usd', 0.0)  # Changed
        else:
            logger.warning(f"  {story_name}: transformation cost file not found")
            transform_input = 0
            transform_output = 0
            transform_cost = 0.0

        # Combine
        total_input = survey_input + transform_input
        total_output = survey_output + transform_output
        total_tokens = total_input + total_output
        total_cost = survey_cost + transform_cost

        results.append({
            'story_name': story_name,
            'iteration': iteration,
            'survey_input_tokens': survey_input,
            'survey_output_tokens': survey_output,
            'transform_input_tokens': transform_input,
            'transform_output_tokens': transform_output,
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost, 4)
        })

        logger.info(f"  {story_name}: {total_tokens:,} tokens, ${total_cost:.2f}")

    return results


# ==========================================================
# Main Function
# ==========================================================

def generate_cost_csvs(
        model: str,
        problem_type: str,
        phase: str,
        iteration: int = 0,
        phase1_dir: str = "output/phase1",
        phase2_dir: str = "output/phase2",
        output_dir: str = "output_analysis/costs"
):
    """
    Generate cost CSV files.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        phase: "phase1", "phase2", or "both"
        iteration: Iteration for Phase 2 (default: 0)
        phase1_dir: Phase 1 directory
        phase2_dir: Phase 2 directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("GENERATING LLM COST CSVs")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Phase: {phase}")

    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Phase 1
    if phase in ["phase1", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1")
        logger.info("=" * 60)

        phase1_results = compute_phase1_costs(model, problem_type, phase1_dir)

        # Save Phase 1 CSV
        output_path = Path(output_dir) / "phase1" / sanitized_model / problem_type
        output_path.mkdir(parents=True, exist_ok=True)

        df_phase1 = pd.DataFrame(phase1_results)  # Changed from [phase1_result]
        csv_file = output_path / "llm_costs.csv"
        df_phase1.to_csv(csv_file, index=False)
        logger.info(f"\n✓ Phase 1 CSV saved to: {csv_file}")

        # Summary
        logger.info(f"  Stories: {len(phase1_results)}")
        logger.info(f"  Total tokens: {df_phase1['total_tokens'].sum():,}")
        logger.info(f"  Total cost: ${df_phase1['cost_usd'].sum():.2f}")

    # Phase 2
    if phase in ["phase2", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2")
        logger.info("=" * 60)

        phase2_results = compute_phase2_costs(model, problem_type, iteration, phase2_dir)

        if not phase2_results:
            logger.warning("No Phase 2 costs computed")
            return

        # Save Phase 2 CSV
        output_path = Path(output_dir) / "phase2" / sanitized_model / problem_type
        output_path.mkdir(parents=True, exist_ok=True)

        df_phase2 = pd.DataFrame(phase2_results)
        csv_file = output_path / "llm_costs.csv"
        df_phase2.to_csv(csv_file, index=False)
        logger.info(f"\n✓ Phase 2 CSV saved to: {csv_file}")

        # Summary statistics
        logger.info("\nPhase 2 Summary:")
        logger.info(f"  Stories: {len(phase2_results)}")
        logger.info(f"  Total tokens: {df_phase2['total_tokens'].sum():,}")
        logger.info(f"  Mean tokens/story: {df_phase2['total_tokens'].mean():.0f}")
        logger.info(f"  Total cost: ${df_phase2['total_cost_usd'].sum():.2f}")
        logger.info(f"  Mean cost/story: ${df_phase2['total_cost_usd'].mean():.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("COST CSV GENERATION COMPLETE")
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate LLM cost CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for both phases
  python plot_scripts/compute_llm_costs.py --model gpt-4o --problem forward --phase both

  # Just Phase 1
  python plot_scripts/compute_llm_costs.py --model gpt-4o --problem forward --phase phase1

  # Phase 2 with specific iteration
  python plot_scripts/compute_llm_costs.py --model gpt-4o --problem forward --phase phase2 --iteration 1
        """
    )

    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--phase', choices=['phase1', 'phase2', 'both'], default='both')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration for Phase 2 (default: 0)')
    parser.add_argument('--phase1-dir', default='output/phase1')
    parser.add_argument('--phase2-dir', default='output/phase2')
    parser.add_argument('--output-dir', default='output_analysis/costs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_cost_csvs(
            model=args.model,
            problem_type=args.problem,
            phase=args.phase,
            iteration=args.iteration,
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir,
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
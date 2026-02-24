"""
Cross-Model Comparison CSV Generation Script

Creates comparison CSV files showing results across multiple models.
Generates 3 CSV files: one for Mean, one for Median, one for Mode.

Each CSV has columns for each model's:
- Original rating
- Optimal iteration number
- Transformed rating (at optimal iteration)
- Target rating (at optimal iteration)

Output: CSV files in output_analysis/cross_model_summary/{problem}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Data Loading and Processing
# ==========================================================

def load_optimal_data_for_model(
    model: str,
    problem_type: str,
    phase2_dir: str = "output_analysis/phase2"
) -> pd.DataFrame:
    """
    Load optimal iterations data for a specific model.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        phase2_dir: Phase 2 analysis directory

    Returns:
        DataFrame with optimal iteration data
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")
    optimal_file = Path(phase2_dir) / sanitized_model / problem_type / "optimal_iterations.csv"

    if not optimal_file.exists():
        logger.warning(f"Optimal iterations file not found for {model}: {optimal_file}")
        return None

    df = pd.read_csv(optimal_file)
    return df


def create_model_comparison_df(
    models: list,
    problem_type: str,
    stat_type: str,
    phase2_dir: str = "output_analysis/phase2"
) -> pd.DataFrame:
    """
    Create comparison DataFrame for a specific statistic.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        stat_type: "mean", "median", or "mode"
        phase2_dir: Phase 2 directory

    Returns:
        Combined DataFrame with all models
    """
    logger.info(f"\nCreating {stat_type.upper()} comparison CSV...")

    # Load optimal data for all models
    model_optimal_dfs = {}
    model_baseline_dfs = {}  # Add baseline data

    for model in models:
        # Load optimal iterations (Phase 2)
        df_optimal = load_optimal_data_for_model(model, problem_type, phase2_dir)
        if df_optimal is not None:
            model_optimal_dfs[model] = df_optimal
            logger.info(f"  Loaded {len(df_optimal)} stories for {model} (Phase 2)")

        # Load baseline ratings
        sanitized_model = model.replace("/", "-").replace(":", "-")
        baseline_file = Path("output_analysis/baseline") / sanitized_model / problem_type / "baseline_transformed_ratings.csv"

        if baseline_file.exists():
            df_baseline = pd.read_csv(baseline_file)
            model_baseline_dfs[model] = df_baseline
            logger.info(f"    Loaded baseline for {model}")
        else:
            logger.warning(f"    Baseline not found for {model}")

    if not model_optimal_dfs:
        raise ValueError("No data loaded for any model")

    # Get all unique stories (use first model as reference)
    first_model = list(model_optimal_dfs.keys())[0]
    all_stories = sorted(model_optimal_dfs[first_model]['story_name'].tolist())

    # Build comparison DataFrame
    rows = []

    for story in all_stories:
        row = {'story_name': story}

        for model in models:
            model_short = model
            # # Create short model name for columns
            # if model.lower()=='gpt-4o':
            #     model_short = 'gpt4o'
            # elif 'gpt-5o' in model.lower():
            #     model_short = 'gpt5o'
            # # elif 'gpt' in model.lower():
            # #     # Generic GPT (gpt-3.5, gpt-4, etc)
            # #     model_short = model.lower().replace('gpt-', 'gpt').replace('.', '')[:8]
            # elif 'claude' in model.lower():
            #     if 'sonnet' in model.lower():
            #         if '4-5' in model or '4.5' in model:
            #             model_short = 'claude45'
            #         elif '3-5' in model or '3.5' in model:
            #             model_short = 'claude35'
            #         else:
            #             model_short = 'claude_sonnet'
            #     elif 'opus' in model.lower():
            #         model_short = 'claude_opus'
            #     elif 'haiku' in model.lower():
            #         model_short = 'claude_haiku'
            #     else:
            #         model_short = 'claude'
            # elif 'grok' in model.lower():
            #     model_short = 'grok'
            # elif 'llama' in model.lower():
            #     if 'llama4' in model.lower() or 'llama-4' in model.lower():
            #         model_short = 'llama4'
            #     elif 'llama3' in model.lower() or 'llama-3' in model.lower():
            #         model_short = 'llama3'
            #     else:
            #         model_short = 'llama'
            # elif 'deepseek' in model.lower():
            #     if 'r1' in model.lower():
            #         model_short = 'deepseek_r1'
            #     else:
            #         model_short = 'deepseek'
            # else:
            #     # Fallback: sanitize model name
            #     model_short = model

            # Get data for this model-story combination
            if model in model_optimal_dfs:
                df_optimal = model_optimal_dfs[model]
                story_data = df_optimal[df_optimal['story_name'] == story]

                if len(story_data) > 0:
                    # Original rating
                    original_col = f'original_{stat_type}'
                    row[f'{model_short}_original'] = story_data[original_col].values[0] if original_col in story_data.columns else None

                    # Baseline rating
                    if model in model_baseline_dfs:
                        df_baseline = model_baseline_dfs[model]
                        baseline_story = df_baseline[df_baseline['story_name'] == story]

                        if len(baseline_story) > 0:
                            stat_col = stat_type.capitalize()  # "Mean", "Median", or "Mode"
                            row[f'{model_short}_baseline'] = baseline_story[stat_col].values[0] if stat_col in baseline_story.columns else None
                        else:
                            row[f'{model_short}_baseline'] = None
                    else:
                        row[f'{model_short}_baseline'] = None

                    # Optimal iteration number
                    optimal_iter = story_data['optimal_iteration'].values[0] if 'optimal_iteration' in story_data.columns else None
                    row[f'{model_short}_optimal_iter'] = optimal_iter

                    # Transformed rating (at optimal iteration)
                    transformed_col = f'optimal_{stat_type}'
                    row[f'{model_short}_transformed'] = story_data[transformed_col].values[0] if transformed_col in story_data.columns else None
                else:
                    # Story not found
                    row[f'{model_short}_original'] = None
                    row[f'{model_short}_baseline'] = None
                    row[f'{model_short}_optimal_iter'] = None
                    row[f'{model_short}_transformed'] = None
            else:
                # Model not available
                row[f'{model_short}_original'] = None
                row[f'{model_short}_baseline'] = None
                row[f'{model_short}_optimal_iter'] = None
                row[f'{model_short}_transformed'] = None

        rows.append(row)

    df_comparison = pd.DataFrame(rows)
    return df_comparison


# ==========================================================
# Main Function
# ==========================================================

def generate_model_comparison_csvs(
    models: list,
    problem_type: str,
    phase2_dir: str = "output_analysis/phase2",
    output_dir: str = "output_analysis/cross_model_summary"
):
    """
    Generate model comparison CSV files.

    Args:
        models: List of model names
        problem_type: "forward" or "inverse"
        phase2_dir: Phase 2 directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("GENERATING MODEL COMPARISON CSVs")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Problem: {problem_type}")

    # Create output directory
    output_path = Path(output_dir) / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate CSV for each statistic
    for stat in ['mean', 'median', 'mode']:
        df = create_model_comparison_df(models, problem_type, stat, phase2_dir)

        csv_file = output_path / f"model_comparison_{stat}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"\n✓ Saved: {csv_file}")
        logger.info(f"  Stories: {len(df)}")
        logger.info(f"  Columns: {len(df.columns)}")

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON CSVs COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path}")
    logger.info("Files created:")
    logger.info("  - model_comparison_mean.csv")
    logger.info("  - model_comparison_median.csv")
    logger.info("  - model_comparison_mode.csv")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate cross-model comparison CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comparison CSVs for all models
  python plot_scripts/create_model_comparison_csv.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 \\
      --problem forward
        """
    )

    parser.add_argument('--models', nargs='+', required=True, help='List of model names')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--phase2-dir', default='output_analysis/phase2')
    parser.add_argument('--output-dir', default='output_analysis/cross_model_summary')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')

    try:
        generate_model_comparison_csvs(
            models=args.models,
            problem_type=args.problem,
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
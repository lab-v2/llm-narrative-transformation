"""
Runtime Analysis Plotting Script

Creates separate scatter plots with linear regression for each model-problem combination.
Shows relationship between N_c (tokens in segments) and number of LLM calls.

Each plot includes:
- Scatter points (one per story)
- Linear regression line
- R² value annotation
- Both axes start at 0

Output: PNG files in output_analysis/plots/runtime/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ==========================================================
# Plot Configuration
# ==========================================================

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20

# Colors for models
MODEL_COLORS = {
    'gpt-4o': '#1B4965',
    'claude-sonnet-4-5': '#9B59B6',
    'xai-grok-4-fast-reasoning': '#E67E22',
    'bedrock-us.meta.llama4-maverick-17b-instruct-v1-0': '#27AE60',
    'bedrock-us.deepseek.r1-v1-0': '#E74C3C'
}

MODEL_LABELS = {
    'gpt-4o': 'GPT-4o',
    'claude-sonnet-4-5': 'Claude',
    'xai-grok-4-fast-reasoning': 'Grok',
    'bedrock-us.meta.llama4-maverick-17b-instruct-v1-0': 'Llama 4',
    'bedrock-us.deepseek.r1-v1-0': 'DeepSeek R1'
}

SCATTER_ALPHA = 0.6
MARKER_SIZE = 50
LINE_WIDTH = 2


# ==========================================================
# Data Loading
# ==========================================================

def load_gamma_runtime_for_model(
    model: str,
    problem_type: str,
    runtime_dir: str = "output_analysis/runtime"
) -> pd.DataFrame:
    """
    Load gamma/runtime data for a specific model.
    
    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        runtime_dir: Runtime directory
        
    Returns:
        DataFrame or None
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")
    csv_file = Path(runtime_dir) / sanitized_model / problem_type / "gamma_runtime_iter0.csv"
    
    if not csv_file.exists():
        logger.warning(f"Data not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    return df


# ==========================================================
# Plot Creation
# ==========================================================

def create_model_runtime_plot(
    df: pd.DataFrame,
    model: str,
    problem_type: str,
    output_path: Path
):
    """
    Create runtime plot for a single model.
    
    Args:
        df: DataFrame with gamma/runtime data
        model: Model name (original)
        problem_type: "forward" or "inverse"
        output_path: Output path
    """
    sanitized_model = model.replace("/", "-").replace(":", "-")
    
    # Get style
    color = MODEL_COLORS.get(sanitized_model, '#95A5A6')
    label = MODEL_LABELS.get(sanitized_model, model[:15])
    
    # Extract data
    x = df['words_in_segments'].values
    y = df['n_segments'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Scatter plot
    ax.scatter(
        x, y,
        color=color,
        s=MARKER_SIZE,
        alpha=SCATTER_ALPHA,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Linear regression
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2
        
        # Plot regression line (extend to start from 0)
        x_max = max(x.max() * 1.05, 100)  # Add some padding
        x_line = np.linspace(0, x_max, 100)
        y_line = slope * x_line + intercept
        
        ax.plot(
            x_line, y_line,
            color=color,
            linestyle='-',
            linewidth=LINE_WIDTH,
            alpha=0.8,
            label=f'$R^2 = {r_squared:.3f}$'
        )
        
        logger.info(f"  R² = {r_squared:.4f}")
        logger.info(f"  Slope = {slope:.6f}")
        logger.info(f"  Intercept = {intercept:.2f}")
    else:
        logger.warning("  Not enough data points for regression")
    
    # Customize
    ax.set_xlabel(r'Tokens in Abduced Segments($N_c$)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of LLM Calls', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)
    
    # Set axes to start at 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(
        loc='upper left',
        fontsize=LEGEND_FONT_SIZE,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False
    )
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved: {output_path.name}\n")
    
    plt.close()


# ==========================================================
# Main Function
# ==========================================================

def create_all_runtime_plots(
    models: list,
    runtime_dir: str = "output_analysis/runtime",
    output_dir: str = "output_analysis/plots/runtime"
):
    """
    Create runtime plots for all model-problem combinations.
    
    Args:
        models: List of model names
        runtime_dir: Runtime directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("CREATING RUNTIME ANALYSIS PLOTS")
    logger.info("=" * 60)
    logger.info(f"Models: {models}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each model and problem
    for problem_type in ["forward", "inverse"]:
        logger.info("=" * 60)
        logger.info(f"{problem_type.upper()} PROBLEM")
        logger.info("=" * 60)
        
        for model in models:
            logger.info(f"\n{model}:")
            
            # Load data
            df = load_gamma_runtime_for_model(model, problem_type, runtime_dir)
            
            if df is None or len(df) == 0:
                logger.warning(f"  No data, skipping\n")
                continue
            
            logger.info(f"  Loaded {len(df)} stories")
            
            # Create plot filename
            if 'gpt' in model.lower():
                model_short = 'gpt4o'
            elif 'claude' in model.lower():
                model_short = 'claude'
            elif 'grok' in model.lower():
                model_short = 'grok'
            elif 'llama' in model.lower():
                model_short = 'llama4'
            elif 'deepseek' in model.lower():
                model_short = 'deepseek_r1'
            else:
                model_short = model[:15].replace("/", "_").replace(":", "_")
            
            output_file = output_path / f"runtime_{model_short}_{problem_type}.png"
            
            # Create plot
            create_model_runtime_plot(df, model, problem_type, output_file)
    
    logger.info("=" * 60)
    logger.info("ALL RUNTIME PLOTS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info("\nFiles created:")
    for problem in ["forward", "inverse"]:
        for model in models:
            if 'gpt' in model.lower():
                m = 'gpt4o'
            elif 'claude' in model.lower():
                m = 'claude'
            elif 'grok' in model.lower():
                m = 'grok'
            elif 'llama' in model.lower():
                m = 'llama4'
            elif 'deepseek' in model.lower():
                m = 'deepseek_r1'
            else:
                continue
            logger.info(f"  - runtime_{m}_{problem}.png")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create runtime analysis plots for all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plots for all models
  python plot_scripts/plot_runtime_analysis.py \\
      --models gpt-4o claude-sonnet-4-5 xai/grok-4-fast-reasoning bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 bedrock/us.deepseek.r1-v1:0
        """
    )
    
    parser.add_argument('--models', nargs='+', required=True, help='List of model names')
    parser.add_argument('--runtime-dir', default='output_analysis/runtime')
    parser.add_argument('--output-dir', default='output_analysis/plots/runtime')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    
    try:
        create_all_runtime_plots(
            models=args.models,
            runtime_dir=args.runtime_dir,
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

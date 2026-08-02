"""
Gamma and Runtime Computation Script

Computes gamma (fraction of story requiring transformation) and number of successful
LLM calls for runtime analysis.

Only counts transformations where status != "skipped_reversal"

Formula:
- n_segments = count of successful transformations (Y-axis for runtime plot)
- gamma = (words in successful segments) / (total story words)

Output: CSV file in output_analysis/runtime/{model}/{problem}/
"""

import json
import argparse
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ==========================================================
# Helper Functions
# ==========================================================

def clean_text(text):
    """Clean text for consistent word counting."""
    if not isinstance(text, str):
        return text
    
    # Replace smart quotes and special characters
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


# ==========================================================
# Gamma Computation
# ==========================================================

def compute_gamma_for_story(
    story_dir: Path,
    iteration: int = 0
) -> dict:
    """
    Compute gamma for a single story.
    
    Args:
        story_dir: Path to story directory
        iteration: Which iteration (default: 0)
        
    Returns:
        Dict with gamma and segment info
    """
    story_name = story_dir.name
    iter_dir = story_dir / f"iteration_{iteration}"
    
    # Load story
    story_file = iter_dir / "story.txt"
    if not story_file.exists():
        logger.warning(f"  {story_name}: story.txt not found")
        return None
    
    with open(story_file, 'r', encoding='utf-8') as f:
        story_text = f.read()
    
    story_text = clean_text(story_text)
    total_words = len(story_text.split())
    
    # Load transformation log
    log_file = iter_dir / "transformation_log.json"
    if not log_file.exists():
        logger.warning(f"  {story_name}: transformation_log.json not found")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # Count successful transformations (status != "skipped_reversal")
    transformations = log_data.get('transformations', [])
    
    successful_segments = []
    n_total = len(transformations)
    n_skipped = 0
    
    for trans in transformations:
        status = trans.get('status')
        
        # If status key exists and is "skipped_reversal", skip it
        if status == "skipped_reversal":
            n_skipped += 1
            continue
        
        # Otherwise, it's a successful transformation
        segment = trans.get('segment', '')
        if segment:
            successful_segments.append(segment)
    
    n_successful = len(successful_segments)
    
    # Count words in successful segments
    words_in_segments = sum(len(clean_text(seg).split()) for seg in successful_segments)
    
    # Compute gamma
    gamma = (words_in_segments / total_words) if total_words > 0 else 0.0
    
    logger.info(f"  {story_name}: {n_successful}/{n_total} transforms, γ={gamma:.4f} ({gamma*100:.2f}%)")
    
    return {
        'story_name': story_name,
        'n_segments': n_successful,
        'n_total_attempts': n_total,
        'n_skipped': n_skipped,
        'words_in_segments': words_in_segments,
        'total_story_words': total_words,
        'gamma': round(gamma, 4),
        'gamma_percent': round(gamma * 100, 2)
    }


# ==========================================================
# Main Processing
# ==========================================================

def compute_gamma_and_runtime(
    model: str,
    problem_type: str,
    iteration: int = 0,
    phase2_dir: str = "output/phase2",
    output_dir: str = "output_analysis/runtime"
):
    """
    Compute gamma and runtime metrics for all stories.
    
    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        iteration: Which iteration (default: 0)
        phase2_dir: Phase 2 directory
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info("COMPUTING GAMMA AND RUNTIME METRICS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Iteration: {iteration}")
    
    # Find all story directories
    sanitized_model = model.replace("/", "-").replace(":", "-")
    results_dir = Path(phase2_dir) / sanitized_model / problem_type
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(story_dirs)} stories\n")
    
    # Process each story
    results = []
    
    for story_dir in sorted(story_dirs):
        result = compute_gamma_for_story(story_dir, iteration)
        if result:
            results.append(result)
    
    if not results:
        logger.error("No results computed")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    output_path = Path(output_dir) / sanitized_model / problem_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / f"gamma_runtime_iter{iteration}.csv"
    df.to_csv(csv_file, index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Saved: {csv_file}")
    logger.info(f"  Stories processed: {len(results)}")
    
    # Summary statistics
    logger.info("\nGamma Statistics:")
    logger.info(f"  Median γ: {df['gamma_percent'].median():.2f}%")
    logger.info(f"  Mean γ: {df['gamma_percent'].mean():.2f}%")
    logger.info(f"  Std Dev: {df['gamma_percent'].std():.2f}%")
    logger.info(f"  Min γ: {df['gamma_percent'].min():.2f}%")
    logger.info(f"  Max γ: {df['gamma_percent'].max():.2f}%")
    logger.info(f"  Q1: {df['gamma_percent'].quantile(0.25):.2f}%")
    logger.info(f"  Q3: {df['gamma_percent'].quantile(0.75):.2f}%")
    
    logger.info("\nSegment Count Statistics:")
    logger.info(f"  Median n_segments: {df['n_segments'].median():.0f}")
    logger.info(f"  Mean n_segments: {df['n_segments'].mean():.0f}")
    logger.info(f"  Min: {df['n_segments'].min()}")
    logger.info(f"  Max: {df['n_segments'].max()}")
    
    logger.info("=" * 60)


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute gamma and runtime metrics from transformation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for GPT-4o forward
  python plot_scripts/compute_gamma_runtime.py --model gpt-4o --problem forward
  
  # Compute for Claude inverse
  python plot_scripts/compute_gamma_runtime.py --model claude-sonnet-4-5 --problem inverse
        """
    )
    
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    parser.add_argument('--iteration', type=int, default=0, help='Iteration number (default: 0)')
    parser.add_argument('--phase2-dir', default='output/phase2')
    parser.add_argument('--output-dir', default='output_analysis/runtime')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    
    try:
        compute_gamma_and_runtime(
            model=args.model,
            problem_type=args.problem,
            iteration=args.iteration,
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

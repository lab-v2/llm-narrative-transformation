"""
Cosine Similarity Computation Script

Computes cosine similarity between original and transformed stories
using OpenAI embeddings. Measures how much stories changed.

Processes all available iterations and saves embeddings for reuse.

Output: CSV and embeddings saved to output_analysis/embeddings/{model}/{problem}/
"""

import os
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai required. Install with: pip install openai")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ==========================================================
# Utility Functions
# ==========================================================
def truncate_text_for_embedding(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate text to fit within embedding model's token limit.

    Args:
        text: Input text
        max_tokens: Maximum tokens (default 8000, safe for 8192 limit)

    Returns:
        Truncated text
    """
    if text is None:
        return None

    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    # Truncate and add marker
    truncated = text[:max_chars]
    logger.warning(f"Text truncated from {len(text)} to {len(truncated)} chars (~{max_tokens} tokens)")

    return truncated

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


def get_embeddings(text, client):
    """Generate embeddings for given text using OpenAI"""
    if text is None or text.strip() == "":
        return None

    # Truncate if too long
    text = truncate_text_for_embedding(text, max_tokens=8000)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def compute_cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    if emb1 is None or emb2 is None:
        return None
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


# ==========================================================
# Main Processing
# ==========================================================

def process_stories(model_name, problem_type, output_dir='output_analysis/embeddings'):
    """
    Process all stories and compute cosine similarities for all iterations.

    Args:
        model_name: e.g., 'gpt-4o', 'claude-sonnet-4-5'
        problem_type: 'forward' or 'inverse'
        output_dir: Base output directory
    """
    logger.info("=" * 60)
    logger.info("COSINE SIMILARITY COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Problem: {problem_type}")

    # Initialize OpenAI client
    logger.info("\nInitializing OpenAI client...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Determine original story directory based on problem type
    if problem_type == "forward":
        original_dir = Path('data/collectivistic-stories-all')
    else:  # inverse
        original_dir = Path('data/individualistic-rags-to-riches-stories')

    logger.info(f"Original stories directory: {original_dir}")

    # Sanitize model name
    sanitized_model = model_name.replace("/", "-").replace(":", "-")

    # Create output directory and check for existing embeddings
    output_path = Path(output_dir) / sanitized_model / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    embeddings_file = output_path / 'embeddings.json'

    # Load existing embeddings if available
    if embeddings_file.exists():
        logger.info(f"\nLoading existing embeddings from: {embeddings_file}")
        with open(embeddings_file, 'r') as f:
            all_embeddings = json.load(f)
        logger.info(f"✓ Loaded cached embeddings for {len(all_embeddings)} stories")
    else:
        all_embeddings = {}
        logger.info("\nNo cached embeddings found, will generate fresh")

    # Base paths
    abduction_base = Path(f'output/phase2/{sanitized_model}/{problem_type}')
    baseline_base = Path(f'output/baseline/{sanitized_model}/{problem_type}')

    # Get all story directories from abduction
    story_dirs = [d for d in abduction_base.iterdir() if d.is_dir()]

    if not story_dirs:
        logger.error(f"No story directories found in {abduction_base}")
        return

    logger.info(f"Found {len(story_dirs)} stories to process\n")

    results = []
    # all_embeddings = {}

    for idx, story_dir in enumerate(sorted(story_dirs), 1):
        story_name = story_dir.name
        logger.info(f"[{idx}/{len(story_dirs)}] Processing: {story_name}")

        # Find original story
        original_path = original_dir / f"{story_name}.txt"
        if not original_path.exists():
            logger.warning(f"  Original story not found, skipping")
            continue

        # # Read and embed original
        # original_text = read_file(original_path)
        # logger.info("  Generating original embedding...")
        # original_emb = get_embeddings(original_text, client)

        # Read original
        original_text = read_file(original_path)

        # Check if we have cached embedding for original
        if story_name in all_embeddings and 'original' in all_embeddings[story_name]:
            logger.info("  Using cached original embedding")
            original_emb = np.array(all_embeddings[story_name]['original'])
        else:
            logger.info("  Generating original embedding...")
            original_emb = get_embeddings(original_text, client)

            if original_emb is not None:
                # Initialize if needed
                if story_name not in all_embeddings:
                    all_embeddings[story_name] = {}
                all_embeddings[story_name]['original'] = original_emb.tolist()

        if original_emb is None:
            logger.warning(f"  Failed to generate embedding, skipping")
            continue

        # Initialize embeddings storage for this story
        all_embeddings[story_name] = {
            'original': original_emb.tolist()
        }

        # Initialize result row
        result_row = {'story_name': story_name}

        # Process baseline
        baseline_path = baseline_base / story_name / 'transformed_story.txt'
        baseline_text = read_file(baseline_path)

        # if baseline_text:
        #     logger.info("  Generating baseline embedding...")
        #     baseline_emb = get_embeddings(baseline_text, client)

        if baseline_text:
            # Check cache
            if story_name in all_embeddings and 'baseline' in all_embeddings[story_name]:
                logger.info("  Using cached baseline embedding")
                baseline_emb = np.array(all_embeddings[story_name]['baseline'])
            else:
                logger.info("  Generating baseline embedding...")
                baseline_emb = get_embeddings(baseline_text, client)

                if baseline_emb is not None:
                    all_embeddings[story_name]['baseline'] = baseline_emb.tolist()

            if baseline_emb is not None:
                # all_embeddings[story_name]['baseline'] = baseline_emb.tolist()
                sim_baseline = compute_cosine_sim(original_emb, baseline_emb)
                result_row['baseline_similarity'] = sim_baseline
                logger.info(f"    Baseline similarity: {sim_baseline:.4f}")
            else:
                result_row['baseline_similarity'] = None
        else:
            result_row['baseline_similarity'] = None
            logger.warning("  Baseline story not found")

        # Process all available abductive iterations
        iteration_dirs = sorted([
            d for d in story_dir.iterdir()
            if d.is_dir() and d.name.startswith('iteration_')
        ])

        logger.info(f"  Found {len(iteration_dirs)} iterations")

        for iter_dir in iteration_dirs:
            iter_num = int(iter_dir.name.split('_')[1])
            transformed_path = iter_dir / 'story_transformed.txt'

            if transformed_path.exists():
                abduction_text = read_file(transformed_path)

                # if abduction_text:
                #     logger.info(f"  Generating abductive iter_{iter_num} embedding...")
                #     abduction_emb = get_embeddings(abduction_text, client)

                if abduction_text:
                    cache_key = f'abductive_iter_{iter_num}'

                    # Check cache
                    if story_name in all_embeddings and cache_key in all_embeddings[story_name]:
                        logger.info(f"  Using cached abductive iter_{iter_num} embedding")
                        abduction_emb = np.array(all_embeddings[story_name][cache_key])
                    else:
                        logger.info(f"  Generating abductive iter_{iter_num} embedding...")
                        abduction_emb = get_embeddings(abduction_text, client)

                        if abduction_emb is not None:
                            all_embeddings[story_name][cache_key] = abduction_emb.tolist()

                    if abduction_emb is not None:
                        # all_embeddings[story_name][f'abductive_iter_{iter_num}'] = abduction_emb.tolist()
                        sim_abduction = compute_cosine_sim(original_emb, abduction_emb)
                        result_row[f'abductive_iter_{iter_num}_similarity'] = sim_abduction
                        logger.info(f"    Similarity: {sim_abduction:.4f}")
                    else:
                        result_row[f'abductive_iter_{iter_num}_similarity'] = None
                else:
                    result_row[f'abductive_iter_{iter_num}_similarity'] = None
            else:
                logger.warning(f"  Iteration {iter_num} transformed story not found")

        results.append(result_row)
        logger.info("")

    # Create output directory
    output_path = Path(output_dir) / sanitized_model / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings to JSON
    embeddings_file = output_path / 'embeddings.json'
    logger.info(f"Saving embeddings to: {embeddings_file}")
    with open(embeddings_file, 'w') as f:
        json.dump(all_embeddings, f, indent=2)
    logger.info(f"✓ Embeddings saved ({len(all_embeddings)} stories)")

    # Save similarities to CSV
    df = pd.DataFrame(results)
    csv_file = output_path / 'cosine_similarities.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"✓ Cosine similarities saved to: {csv_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stories processed: {len(results)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 60)

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    numeric_cols = [col for col in df.columns if col != 'story_name']
    if numeric_cols:
        logger.info("\n" + df[numeric_cols].describe().to_string())


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute cosine similarities using OpenAI embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for gpt-4o forward problem
  python plot_scripts/create_cosine_similarity_csv.py --model gpt-4o --problem forward

  # Compute for Claude inverse problem
  python plot_scripts/create_cosine_similarity_csv.py --model claude-sonnet-4-5 --problem inverse
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4o, claude-sonnet-4-5)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type: forward or inverse'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/embeddings',
        help='Output directory (default: output_analysis/embeddings)'
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

    # Run processing
    try:
        process_stories(
            model_name=args.model,
            problem_type=args.problem,
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
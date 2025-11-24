import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re
from dotenv import load_dotenv

load_dotenv('.env')


def read_file(filepath):
    """Read text file and return content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def tokenize(text):
    """Simple tokenization: lowercase and split on non-alphanumeric"""
    if text is None:
        return []
    # Convert to lowercase and split on non-word characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def get_token_distribution(tokens, smoothing=1e-5):
    """
    Get probability distribution from tokens with Laplace smoothing

    Args:
        tokens: list of tokens
        smoothing: smoothing parameter (default 1e-5)

    Returns:
        dict: {token: probability}
    """
    if not tokens:
        return {}

    # Count token frequencies
    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    # Apply Laplace smoothing and compute probabilities
    vocab_size = len(token_counts)
    distribution = {}

    for token, count in token_counts.items():
        # Add smoothing to avoid zero probabilities
        distribution[token] = (count + smoothing) / (total_tokens + smoothing * vocab_size)

    return distribution


def compute_kl_divergence(dist_p, dist_q, smoothing=1e-5):
    """
    Compute KL(P || Q) from token distributions
    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Args:
        dist_p: probability distribution P (dict)
        dist_q: probability distribution Q (dict)
        smoothing: smoothing for unseen tokens

    Returns:
        float: KL divergence value
    """
    if not dist_p or not dist_q:
        return None

    # Get union of all tokens
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())

    kl_sum = 0.0
    for token in all_tokens:
        # Get probabilities with smoothing for unseen tokens
        p = dist_p.get(token, smoothing)
        q = dist_q.get(token, smoothing)

        # Add to KL divergence sum
        kl_sum += p * np.log(p / q)

    return float(kl_sum)


def compute_js_divergence(dist_p, dist_q, smoothing=1e-5):
    """
    Compute Jensen-Shannon Divergence (symmetric, bounded 0-1)
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = (P + Q) / 2
    """
    if not dist_p or not dist_q:
        return None

    # Get union of all tokens
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

    # Compute JSD
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Normalize to 0-1 range (divide by log(2))
    jsd_normalized = jsd / np.log(2)

    return float(jsd_normalized)


def process_stories(phase, model_name, problem_type):
    """
    Process all stories and compute token-based KL divergences

    Args:
        phase: e.g., 'phase2'
        model_name: e.g., 'gpt-4o'
        problem_type: e.g., 'forward'
    """
    print("Starting token-based KL-divergence computation...\n")

    # Base paths
    original_dir = Path('data/individualistic-rags-to-riches-stories-subset')
    abduction_base = Path(f'output/{phase}/{model_name}/{problem_type}')
    baseline_base = Path(f'output/baseline/{model_name}/{problem_type}')

    # Get all story files from original directory
    story_files = list(original_dir.glob('*.txt'))

    if not story_files:
        print(f"No story files found in {original_dir}")
        return

    print(f"Found {len(story_files)} stories to process\n")

    results_transformed_original = []
    results_original_transformed = []
    results_js_divergence = []

    for idx, original_path in enumerate(story_files, 1):
        story_name = original_path.stem
        print(f"[{idx}/{len(story_files)}] Processing: {story_name}")

        # Construct file paths
        abduction_path = abduction_base / story_name / 'iteration_1' / 'story_transformed.txt'
        baseline_path = baseline_base / story_name / 'transformed_story.txt'

        # Read files
        original_text = read_file(original_path)
        abduction_text = read_file(abduction_path)
        baseline_text = read_file(baseline_path)

        # Tokenize
        print("  - Tokenizing texts...")
        original_tokens = tokenize(original_text)
        abduction_tokens = tokenize(abduction_text)
        baseline_tokens = tokenize(baseline_text)

        print(f"    Original: {len(original_tokens)} tokens")
        print(f"    Baseline: {len(baseline_tokens)} tokens")
        print(f"    Abduction: {len(abduction_tokens)} tokens")

        # Get token distributions
        print("  - Computing token distributions...")
        original_dist = get_token_distribution(original_tokens)
        abduction_dist = get_token_distribution(abduction_tokens)
        baseline_dist = get_token_distribution(baseline_tokens)

        # Compute KL divergences - Direction 1: KL(transformed || original)
        print("  - Computing KL(transformed || original)...")
        kl_baseline_original = compute_kl_divergence(baseline_dist, original_dist)
        kl_abduction_original = compute_kl_divergence(abduction_dist, original_dist)

        # Compute KL divergences - Direction 2: KL(original || transformed)
        print("  - Computing KL(original || transformed)...")
        kl_original_baseline = compute_kl_divergence(original_dist, baseline_dist)
        kl_original_abduction = compute_kl_divergence(original_dist, abduction_dist)

        # Compute JS divergences (symmetric)
        print("  - Computing JS divergences...")
        js_baseline = compute_js_divergence(baseline_dist, original_dist)
        js_abduction = compute_js_divergence(abduction_dist, original_dist)

        # Store results
        results_transformed_original.append({
            'story_name': story_name,
            'kl_baseline_original': kl_baseline_original,
            'kl_abduction_original': kl_abduction_original
        })

        results_original_transformed.append({
            'story_name': story_name,
            'kl_original_baseline': kl_original_baseline,
            'kl_original_abduction': kl_original_abduction
        })

        results_js_divergence.append({
            'story_name': story_name,
            'js_divergence_baseline': js_baseline,
            'js_divergence_abduction': js_abduction
        })

        print(f"  ✓ KL(baseline||original): {kl_baseline_original}")
        print(f"  ✓ KL(abduction||original): {kl_abduction_original}")
        print(f"  ✓ JS(baseline, original): {js_baseline}")
        print(f"  ✓ JS(abduction, original): {js_abduction}\n")

    # Create DataFrames and save to CSV
    df1 = pd.DataFrame(results_transformed_original)
    df2 = pd.DataFrame(results_original_transformed)
    df3 = pd.DataFrame(results_js_divergence)

    output1 = 'kl_divergence_token_transformed_original.csv'
    output2 = 'kl_divergence_token_original_transformed.csv'
    output3 = 'js_divergence_token.csv'

    df1.to_csv(output1, index=False)
    df2.to_csv(output2, index=False)
    df3.to_csv(output3, index=False)

    print(f"\n✓ Results saved to:")
    print(f"  - {output1}")
    print(f"  - {output2}")
    print(f"  - {output3}")
    print(f"✓ Processed {len(results_transformed_original)} stories")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\n1. KL(transformed || original):")
    print(df1[['kl_baseline_original', 'kl_abduction_original']].describe())

    print("\n2. KL(original || transformed):")
    print(df2[['kl_original_baseline', 'kl_original_abduction']].describe())

    print("\n3. JS Divergence (normalized 0-1):")
    print(df3[['js_divergence_baseline', 'js_divergence_abduction']].describe())


def main():
    parser = argparse.ArgumentParser(
        description='Compute token-based KL divergences and JS divergences'
    )
    parser.add_argument('--phase', type=str, required=True,
                        help='Phase name (e.g., phase2)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., gpt-4o)')
    parser.add_argument('--problem_type', type=str, required=True,
                        help='Problem type (e.g., forward)')

    args = parser.parse_args()

    print("=" * 70)
    print("Token-Based KL-Divergence & JS-Divergence Computation")
    print("=" * 70)
    print(f"Phase: {args.phase}")
    print(f"Model: {args.model}")
    print(f"Problem Type: {args.problem_type}")
    print("=" * 70 + "\n")

    process_stories(args.phase, args.model, args.problem_type)


if __name__ == '__main__':
    main()
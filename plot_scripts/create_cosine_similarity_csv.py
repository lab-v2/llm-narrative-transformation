import os
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv
load_dotenv('.env')
print(f"DEBUG: API key loaded: {os.getenv('OPENAI_API_KEY')[:10]}...")  # Print first 10 chars
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


def get_embeddings(text, client):
    """Generate embeddings for given text using OpenAI"""
    if text is None or text.strip() == "":
        return None
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # or "text-embedding-3-large" for better quality
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def compute_cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    if emb1 is None or emb2 is None:
        return None
    # Reshape for sklearn cosine_similarity
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


def process_stories(phase, model_name, problem_type, output_csv='cosine_similarities.csv'):
    """
    Process all stories and compute cosine similarities

    Args:
        phase: e.g., 'phase2'
        model_name: e.g., 'gpt-4o'
        problem_type: e.g., 'forward'
        output_csv: output CSV filename
    """
    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("Client initialized successfully!\n")

    # Base paths
    original_dir = Path('data/individualistic-rags-to-riches-stories-subset')
    abduction_base = Path(f'output/{phase}/{model_name}/{problem_type}')
    baseline_base = Path(f'output/baseline/{model_name}/{problem_type}')

    print(original_dir)
    # Get all story files from original directory
    story_files = list(original_dir.glob('*.txt'))

    if not story_files:
        print(f"No story files found in {original_dir}")
        return

    print(f"Found {len(story_files)} stories to process\n")

    results = []

    for idx, original_path in enumerate(story_files, 1):
        story_name = original_path.stem  # filename without extension
        print(f"[{idx}/{len(story_files)}] Processing: {story_name}")

        # Construct file paths
        abduction_path = abduction_base / story_name / 'iteration_1' / 'story_transformed.txt'
        baseline_path = baseline_base / story_name / 'transformed_story.txt'

        # Read files
        original_text = read_file(original_path)
        abduction_text = read_file(abduction_path)
        baseline_text = read_file(baseline_path)

        # Generate embeddings
        print("  - Generating embeddings...")
        original_emb = get_embeddings(original_text, client)
        abduction_emb = get_embeddings(abduction_text, client)
        baseline_emb = get_embeddings(baseline_text, client)

        # Compute cosine similarities
        sim_baseline = compute_cosine_sim(original_emb, baseline_emb)
        sim_abduction = compute_cosine_sim(original_emb, abduction_emb)

        # Store results
        results.append({
            'story_name': story_name,
            'cosine_sim_original_baseline': sim_baseline,
            'cosine_similarity_original_abduction': sim_abduction
        })

        print(f"  - Baseline similarity: {sim_baseline}")
        print(f"  - Abduction similarity: {sim_abduction}\n")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")
    print(f"✓ Processed {len(results)} stories")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(df[['cosine_sim_original_baseline', 'cosine_similarity_original_abduction']].describe())


def main():
    parser = argparse.ArgumentParser(
        description='Compute cosine similarities between original, baseline, and abduction stories using OpenAI embeddings'
    )
    parser.add_argument('--phase', type=str, required=True,
                        help='Phase name (e.g., phase2)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., gpt-4o)')
    parser.add_argument('--problem_type', type=str, required=True,
                        help='Problem type (e.g., forward)')
    parser.add_argument('--output', type=str, default='cosine_similarities.csv',
                        help='Output CSV filename (default: cosine_similarities.csv)')

    args = parser.parse_args()

    print("=" * 60)
    print("Cosine Similarity Computation (OpenAI Embeddings)")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Model: {args.model}")
    print(f"Problem Type: {args.problem_type}")
    print(f"Output: {args.output}")
    print("=" * 60 + "\n")

    process_stories(args.phase, args.model, args.problem_type, args.output)


if __name__ == '__main__':
    main()
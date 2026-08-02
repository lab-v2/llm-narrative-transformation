# extract_evaluation_stories.py
import os
import pandas as pd
import shutil
from pathlib import Path
import argparse


def extract_evaluation_stories(model_name):
    """
    Extract all original and transformed stories for evaluation
    - Individualistic: original individualistic + abduction/baseline transformed to collectivistic (inverse)
    - Collectivistic: original collectivistic + abduction/baseline transformed to individualistic (forward)

    Args:
        model_name: The model name (e.g., "gpt-4o", "claude-sonnet-4-5")
    """

    print(f"🚀 Extracting evaluation stories for model: {model_name}")

    # Paths
    output_dir = f"evaluation_data/{model_name}"
    individualistic_dir = f"{output_dir}/individualistic"
    collectivistic_dir = f"{output_dir}/collectivistic"

    # Source directories
    original_individualistic_dir = "data/individualistic-rags-to-riches-stories"
    original_collectivistic_dir = "data/collectivistic-stories-all"

    # Optimal iterations CSVs
    inverse_optimal_csv = f"output_analysis/phase2/{model_name}/inverse/optimal_iterations.csv"
    forward_optimal_csv = f"output_analysis/phase2/{model_name}/forward/optimal_iterations.csv"

    # Check if source directories exist
    if not os.path.exists(original_individualistic_dir):
        raise FileNotFoundError(f"Original individualistic stories not found: {original_individualistic_dir}")

    if not os.path.exists(original_collectivistic_dir):
        raise FileNotFoundError(f"Original collectivistic stories not found: {original_collectivistic_dir}")

    # Handle output directory
    if os.path.exists(output_dir):
        print(f"⚠️  Output directory {output_dir} already exists!")
        response = input("Do you want to recreate it? This will delete existing data (y/n): ").lower()
        if response == 'y':
            print(f"🗑️  Removing existing directory...")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"📁 Recreated directory: {output_dir}")
        else:
            print("❌ Aborted to avoid overwriting existing data")
            return
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created directory: {output_dir}")

    # Create subdirectories
    os.makedirs(individualistic_dir, exist_ok=True)
    os.makedirs(collectivistic_dir, exist_ok=True)

    # Process individualistic stories (inverse transformation)
    print(f"\n📚 Processing individualistic stories (inverse transformation)...")
    process_individualistic_stories(model_name, individualistic_dir, original_individualistic_dir, inverse_optimal_csv)

    # Process collectivistic stories (forward transformation)
    print(f"\n📚 Processing collectivistic stories (forward transformation)...")
    process_collectivistic_stories(model_name, collectivistic_dir, original_collectivistic_dir, forward_optimal_csv)

    print(f"\n🎉 Successfully extracted all evaluation stories!")
    print(f"📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── individualistic/ (original individualistic + abduction/baseline → collectivistic)")
    print(f"   └── collectivistic/ (original collectivistic + abduction/baseline → individualistic)")


def process_individualistic_stories(model_name, output_dir, original_dir, optimal_csv):
    """Process individualistic stories with inverse transformation"""

    # Check if optimal iterations CSV exists
    if not os.path.exists(optimal_csv):
        print(f"⚠️  Inverse optimal iterations CSV not found: {optimal_csv}")
        print(f"   Skipping individualistic stories...")
        return

    # Load optimal iterations
    print(f"📈 Loading inverse optimal iterations from: {optimal_csv}")
    optimal_df = pd.read_csv(optimal_csv)

    # Create dictionary of story_name -> optimal_iteration
    optimal_iterations = {}
    for _, row in optimal_df.iterrows():
        optimal_iterations[row['story_name']] = row['optimal_iteration']

    print(f"📋 Found optimal iterations for {len(optimal_iterations)} individualistic stories")

    processed_count = 0

    for story_name, optimal_iter in optimal_iterations.items():
        print(f"\n📖 Processing individualistic story: {story_name}")
        print(f"   🎯 Optimal iteration: {optimal_iter}")

        # Create story folder
        story_folder = os.path.join(output_dir, story_name)
        os.makedirs(story_folder, exist_ok=True)

        # Path to original individualistic story
        original_story_path = os.path.join(original_dir, f"{story_name}.txt")
        if not os.path.exists(original_story_path):
            print(f"   ⚠️  Original individualistic story not found: {original_story_path}")
            continue

        # Path to abduction transformed story (individualistic -> collectivistic)
        abduction_story_path = f"output/phase2/{model_name}/inverse/{story_name}/iteration_{optimal_iter}/story_transformed.txt"
        if not os.path.exists(abduction_story_path):
            print(f"   ⚠️  Abduction transformed story not found: {abduction_story_path}")
            continue

        # Path to baseline transformed story (individualistic -> collectivistic)
        baseline_story_path = f"output/baseline/{model_name}/inverse/{story_name}/transformed_story.txt"
        if not os.path.exists(baseline_story_path):
            print(f"   ⚠️  Baseline transformed story not found: {baseline_story_path}")
            # continue
        else:
            # Copy baseline transformed collectivistic story
            baseline_dest = os.path.join(story_folder, "baseline_transformed.txt")
            shutil.copy2(baseline_story_path, baseline_dest)
            print(f"   ✅ Copied baseline transformed collectivistic story")

        # Copy original individualistic story
        original_dest = os.path.join(story_folder, "original_story.txt")
        shutil.copy2(original_story_path, original_dest)
        print(f"   ✅ Copied original individualistic story")

        # Copy abduction transformed collectivistic story
        abduction_dest = os.path.join(story_folder, "abduction_transformed.txt")
        shutil.copy2(abduction_story_path, abduction_dest)
        print(f"   ✅ Copied abduction transformed collectivistic story")



        processed_count += 1

    print(f"\n✅ Processed {processed_count} individualistic stories")


def process_collectivistic_stories(model_name, output_dir, original_dir, optimal_csv):
    """Process collectivistic stories with forward transformation"""

    # Check if optimal iterations CSV exists
    if not os.path.exists(optimal_csv):
        print(f"⚠️  Forward optimal iterations CSV not found: {optimal_csv}")
        print(f"   Skipping collectivistic stories...")
        return

    # Load optimal iterations
    print(f"📈 Loading forward optimal iterations from: {optimal_csv}")
    optimal_df = pd.read_csv(optimal_csv)

    # Create dictionary of story_name -> optimal_iteration
    optimal_iterations = {}
    for _, row in optimal_df.iterrows():
        optimal_iterations[row['story_name']] = row['optimal_iteration']

    print(f"📋 Found optimal iterations for {len(optimal_iterations)} collectivistic stories")

    processed_count = 0

    for story_name, optimal_iter in optimal_iterations.items():
        print(f"\n📖 Processing collectivistic story: {story_name}")
        print(f"   🎯 Optimal iteration: {optimal_iter}")

        # Create story folder
        story_folder = os.path.join(output_dir, story_name)
        os.makedirs(story_folder, exist_ok=True)

        # Path to original collectivistic story
        original_story_path = os.path.join(original_dir, f"{story_name}.txt")
        if not os.path.exists(original_story_path):
            print(f"   ⚠️  Original collectivistic story not found: {original_story_path}")
            continue

        # Path to abduction transformed story (collectivistic -> individualistic)
        abduction_story_path = f"output/phase2/{model_name}/forward/{story_name}/iteration_{optimal_iter}/story_transformed.txt"
        if not os.path.exists(abduction_story_path):
            print(f"   ⚠️  Abduction transformed story not found: {abduction_story_path}")
            continue

        # Path to baseline transformed story (collectivistic -> individualistic)
        baseline_story_path = f"output/baseline/{model_name}/forward/{story_name}/transformed_story.txt"
        if not os.path.exists(baseline_story_path):
            print(f"   ⚠️  Baseline transformed story not found: {baseline_story_path}")
            continue

        # Copy original collectivistic story
        original_dest = os.path.join(story_folder, "original_story.txt")
        shutil.copy2(original_story_path, original_dest)
        print(f"   ✅ Copied original collectivistic story")

        # Copy abduction transformed individualistic story
        abduction_dest = os.path.join(story_folder, "abduction_transformed.txt")
        shutil.copy2(abduction_story_path, abduction_dest)
        print(f"   ✅ Copied abduction transformed individualistic story")

        # Copy baseline transformed individualistic story
        baseline_dest = os.path.join(story_folder, "baseline_transformed.txt")
        shutil.copy2(baseline_story_path, baseline_dest)
        print(f"   ✅ Copied baseline transformed individualistic story")

        processed_count += 1

    print(f"\n✅ Processed {processed_count} collectivistic stories")


def show_evaluation_info(model_name):
    """Show information about extracted evaluation stories"""
    output_dir = f"evaluation_data/{model_name}"
    individualistic_dir = f"{output_dir}/individualistic"
    collectivistic_dir = f"{output_dir}/collectivistic"

    if not os.path.exists(output_dir):
        print(f"❌ No evaluation data found for model: {model_name}")
        print(f"   Directory {output_dir} does not exist")
        print(f"   Run: python extract_evaluation_stories.py {model_name}")
        return

    print(f"\n📊 Evaluation Data for {model_name}:")
    print(f"   - Location: {output_dir}")

    # Check individualistic stories
    if os.path.exists(individualistic_dir):
        individualistic_stories = [f for f in os.listdir(individualistic_dir) if
                                   os.path.isdir(os.path.join(individualistic_dir, f))]
        print(f"   - Individualistic stories: {len(individualistic_stories)}")

        for i, story_name in enumerate(sorted(individualistic_stories), 1):
            story_path = os.path.join(individualistic_dir, story_name)
            original_file = os.path.join(story_path, "original_story.txt")
            abduction_file = os.path.join(story_path, "abduction_transformed.txt")
            baseline_file = os.path.join(story_path, "baseline_transformed.txt")

            original_exists = "✅" if os.path.exists(original_file) else "❌"
            abduction_exists = "✅" if os.path.exists(abduction_file) else "❌"
            baseline_exists = "✅" if os.path.exists(baseline_file) else "❌"

            print(
                f"      {i:2d}. {story_name}: Orig {original_exists} | Abduct {abduction_exists} | Base {baseline_exists}")
    else:
        print(f"   - Individualistic stories: 0 (directory not found)")

    # Check collectivistic stories
    if os.path.exists(collectivistic_dir):
        collectivistic_stories = [f for f in os.listdir(collectivistic_dir) if
                                  os.path.isdir(os.path.join(collectivistic_dir, f))]
        print(f"   - Collectivistic stories: {len(collectivistic_stories)}")

        for i, story_name in enumerate(sorted(collectivistic_stories), 1):
            story_path = os.path.join(collectivistic_dir, story_name)
            original_file = os.path.join(story_path, "original_story.txt")
            abduction_file = os.path.join(story_path, "abduction_transformed.txt")
            baseline_file = os.path.join(story_path, "baseline_transformed.txt")

            original_exists = "✅" if os.path.exists(original_file) else "❌"
            abduction_exists = "✅" if os.path.exists(abduction_file) else "❌"
            baseline_exists = "✅" if os.path.exists(baseline_file) else "❌"

            print(
                f"      {i:2d}. {story_name}: Orig {original_exists} | Abduct {abduction_exists} | Base {baseline_exists}")
    else:
        print(f"   - Collectivistic stories: 0 (directory not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all stories for evaluation with abduction and baseline transformations")
    parser.add_argument("--model_name", help="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
    parser.add_argument("--info", action="store_true", help="Show info about extracted stories")

    args = parser.parse_args()

    if args.info:
        show_evaluation_info(args.model_name)
    else:
        try:
            extract_evaluation_stories(args.model_name)
            show_evaluation_info(args.model_name)

            print(f"\n🚀 Evaluation data ready!")
            print(f"   Each story now has 3 files:")
            print(f"   - original_story.txt")
            print(f"   - abduction_transformed.txt")
            print(f"   - baseline_transformed.txt")

        except Exception as e:
            print(f"❌ Error: {e}")
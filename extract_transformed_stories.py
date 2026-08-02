# extract_transformed_stories.py
import os
import pandas as pd
import shutil
from pathlib import Path
import argparse


def extract_transformed_stories(model_name):
    """
    Extract original and transformed stories for training stories based on optimal iterations

    Args:
        model_name: The model name (e.g., "gpt-4o")
    """

    print(f"🚀 Extracting transformed stories for model: {model_name}")

    # Paths
    train_split_csv = "data/train_test_split.csv"
    optimal_iterations_csv = f"output_analysis/phase2/{model_name}/inverse/optimal_iterations.csv"
    output_dir = f"data/train_{model_name}"
    original_stories_dir = "data/individualistic-rags-to-riches-stories-subset-subset"

    # Check if required files exist
    if not os.path.exists(train_split_csv):
        raise FileNotFoundError(f"Train/test split file not found: {train_split_csv}")

    if not os.path.exists(optimal_iterations_csv):
        raise FileNotFoundError(f"Optimal iterations CSV not found: {optimal_iterations_csv}")

    if not os.path.exists(original_stories_dir):
        raise FileNotFoundError(f"Original stories directory not found: {original_stories_dir}")

    # Load train/test split
    print("📊 Loading train/test split...")
    split_df = pd.read_csv(train_split_csv)
    train_stories = split_df[split_df['split'] == 'train']['story_name'].tolist()

    print(f"🎯 Found {len(train_stories)} training stories")
    for story in train_stories:
        print(f"   - {story}")

    # Load optimal iterations
    print("\n📈 Loading optimal iterations...")
    optimal_df = pd.read_csv(optimal_iterations_csv)

    # Create dictionary of story_name -> optimal_iteration
    optimal_iterations = {}
    for _, row in optimal_df.iterrows():
        optimal_iterations[row['story_name']] = row['optimal_iteration']

    print(f"📋 Loaded optimal iterations for {len(optimal_iterations)} stories")

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
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created new directory: {output_dir}")

    # Process each training story
    print(f"\n🔄 Processing {len(train_stories)} training stories...")

    for story_name in train_stories:
        print(f"\n📖 Processing: {story_name}")

        # Check if story has optimal iteration data
        if story_name not in optimal_iterations:
            raise ValueError(f"Story '{story_name}' not found in optimal iterations CSV!")

        optimal_iter = optimal_iterations[story_name]
        print(f"   🎯 Optimal iteration: {optimal_iter}")

        # Create story folder
        story_folder = os.path.join(output_dir, story_name)
        os.makedirs(story_folder, exist_ok=True)

        # Path to original story
        original_story_path = os.path.join(original_stories_dir, f"{story_name}.txt")
        if not os.path.exists(original_story_path):
            raise FileNotFoundError(f"Original story not found: {original_story_path}")

        # Path to transformed story
        transformed_story_path = f"output/phase2/{model_name}/inverse/{story_name}/iteration_{optimal_iter}/story_transformed.txt"
        if not os.path.exists(transformed_story_path):
            raise FileNotFoundError(f"Transformed story not found: {transformed_story_path}")

        # Copy original story
        original_dest = os.path.join(story_folder, "original_story.txt")
        shutil.copy2(original_story_path, original_dest)
        print(f"   ✅ Copied original story to: {original_dest}")

        # Copy transformed story
        transformed_dest = os.path.join(story_folder, "transformed_story.txt")
        shutil.copy2(transformed_story_path, transformed_dest)
        print(f"   ✅ Copied transformed story to: {transformed_dest}")

        print(f"   📁 Story folder created: {story_folder}")

    print(f"\n🎉 Successfully processed all {len(train_stories)} training stories!")
    print(f"📁 Output directory structure:")
    print(f"   {output_dir}/")
    for story_name in train_stories:
        print(f"   ├── {story_name}/")
        print(f"   │   ├── original_story.txt")
        print(f"   │   └── transformed_story.txt")

    print(f"\n🚀 Next step: Create CSV for LLM3 training from {output_dir}")


def show_extracted_info(model_name):
    """Show information about extracted stories"""
    output_dir = f"data/train_{model_name}"

    if not os.path.exists(output_dir):
        print(f"📁 No extracted data found for model: {model_name}")
        print(f"   Directory {output_dir} does not exist")
        print(f"   Run: python extract_transformed_stories.py {model_name}")
        return

    story_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]

    if len(story_folders) == 0:
        print(f"📁 Directory {output_dir} exists but is empty")
        print(f"   Run: python extract_transformed_stories.py {model_name}")
        return

    print(f"\n📊 Extracted Stories for {model_name}:")
    print(f"   - Total stories: {len(story_folders)}")
    print(f"   - Location: {output_dir}")

    for i, story_name in enumerate(sorted(story_folders), 1):
        story_path = os.path.join(output_dir, story_name)
        original_file = os.path.join(story_path, "original_story.txt")
        transformed_file = os.path.join(story_path, "transformed_story.txt")

        original_exists = "✅" if os.path.exists(original_file) else "❌"
        transformed_exists = "✅" if os.path.exists(transformed_file) else "❌"

        print(f"   {i:2d}. {story_name}: Original {original_exists} | Transformed {transformed_exists}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract transformed stories for LLM3 training")
    parser.add_argument("model_name", help="Model name (e.g., gpt-4o)")
    parser.add_argument("--info", action="store_true", help="Show info about extracted stories")

    args = parser.parse_args()

    if args.info:
        show_extracted_info(args.model_name)
    else:
        try:
            extract_transformed_stories(args.model_name)
            show_extracted_info(args.model_name)
        except Exception as e:
            print(f"❌ Error: {e}")
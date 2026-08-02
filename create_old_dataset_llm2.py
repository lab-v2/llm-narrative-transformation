# create_dataset_llm2.py
import os
import pandas as pd
import json
from pathlib import Path


def create_dataset_from_stories(stories_folder="data/collectivistic-stories-all",
                                output_file="data/collectivistic_dataset_llm2.jsonl"):
    """
    Create a dataset JSONL from story text files
    - Uses ALL stories for training (no train/test split)
    - Creates JSONL with fields: filename, story_text, prompt
    """

    stories_path = Path(stories_folder)

    # Check if JSONL already exists
    if os.path.exists(output_file):
        print(f"⚠️  Dataset JSONL '{output_file}' already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("📊 Loading existing dataset...")
            # Load JSONL
            data = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
            print(f"✅ Dataset loaded: {len(df)} total stories")
            return output_file

    # Get all .txt files
    txt_files = list(stories_path.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in {stories_folder}")

    print(f"📁 Found {len(txt_files)} story files")
    print(f"🎯 Using ALL {len(txt_files)} stories for training")

    # Create dataset
    dataset_rows = []
    prompt = "Write a collectivistic story"

    # Process all files
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                story_text = f.read().strip()

            if story_text:  # Only add if file has content
                dataset_rows.append({
                    'filename': txt_file.name,
                    'story_text': story_text,
                    'prompt': prompt
                })
                print(f"✅ Added story: {txt_file.name}")
            else:
                print(f"⚠️  Skipping empty file: {txt_file.name}")

        except Exception as e:
            print(f"❌ Error reading {txt_file.name}: {e}")

    # Create DataFrame and save as JSONL
    df = pd.DataFrame(dataset_rows)

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')

    print(f"\n📊 Dataset created successfully!")
    print(f"   - Total stories: {len(df)}")
    print(f"   - All stories will be used for training")
    print(f"   - Saved to: {output_file}")

    # Show sample
    if len(df) > 0:
        print(f"\n📖 Sample entry:")
        sample = df.iloc[0]
        print(f"   Filename: {sample['filename']}")
        print(f"   Prompt: {sample['prompt']}")
        print(f"   Story preview: {sample['story_text'][:150]}...")

    return output_file


def show_dataset_info(file_path="data/collectivistic_dataset_llm2.jsonl"):
    """Show information about the created dataset"""
    if not os.path.exists(file_path):
        print(f"❌ Dataset file {file_path} not found!")
        return

    # Load JSONL
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"\n📊 Dataset Information:")
    print(f"   - Total entries: {len(df)}")
    print(f"   - Average story length: {df['story_text'].str.len().mean():.0f} characters")
    print(f"   - Longest story: {df['story_text'].str.len().max()} characters")
    print(f"   - Shortest story: {df['story_text'].str.len().min()} characters")

    print(f"\n📝 All story files:")
    for i, filename in enumerate(df['filename'].tolist(), 1):
        story_length = df.iloc[i - 1]['story_text']
        print(f"   {i:2d}. {filename} ({len(story_length)} chars)")


if __name__ == "__main__":
    # Create the dataset
    output_file = create_dataset_from_stories()

    # Show dataset information
    show_dataset_info(output_file)

    print(f"\n🚀 Dataset ready for training!")

    # Load to get count
    data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"   All {len(data)} stories will be used for fine-tuning")
    print(f"   Next step: Update train_llm2.py to handle JSONL format")
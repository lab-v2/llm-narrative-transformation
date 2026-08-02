# create_dataset_llm2.py
import os
import json
from pathlib import Path


def create_dataset_from_stories(stories_folder="data/collectivistic-stories-all",
                                output_file="data/collectivistic_dataset_llm2.jsonl"):
    """
    Create a dataset JSONL from story text files in OpenAI messages format
    - Uses ALL stories for training (no train/test split)
    - Output format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    stories_path = Path(stories_folder)

    # Check if JSONL already exists
    if os.path.exists(output_file):
        print(f"⚠️  Dataset JSONL '{output_file}' already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("📊 Loading existing dataset...")
            data = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            print(f"✅ Dataset loaded: {len(data)} total stories")
            return output_file

    # Get all .txt files
    txt_files = list(stories_path.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in {stories_folder}")

    print(f"📁 Found {len(txt_files)} story files")
    print(f"🎯 Using ALL {len(txt_files)} stories for training")

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    prompt = "Write a collectivistic story"
    count = 0

    # Process all files and write directly to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as story_file:
                    story_text = story_file.read().strip()

                if story_text:  # Only add if file has content
                    # Create message format for fine-tuning
                    message_obj = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": story_text}
                        ]
                    }
                    json.dump(message_obj, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
                    print(f"✅ Added story: {txt_file.name}")
                else:
                    print(f"⚠️  Skipping empty file: {txt_file.name}")

            except Exception as e:
                print(f"❌ Error reading {txt_file.name}: {e}")

    print(f"\n📊 Dataset created successfully!")
    print(f"   - Total stories: {count}")
    print(f"   - All stories will be used for training")
    print(f"   - Saved to: {output_file}")

    # Show sample
    if count > 0:
        print(f"\n📖 Sample entry:")
        with open(output_file, 'r', encoding='utf-8') as f:
            sample_line = f.readline()
            sample = json.loads(sample_line)
            print(f"   User prompt: {sample['messages'][0]['content']}")
            print(f"   Story preview: {sample['messages'][1]['content'][:150]}...")

    return output_file


def show_dataset_info(file_path="data/collectivistic_dataset_llm2.jsonl"):
    """Show information about the created dataset"""
    if not os.path.exists(file_path):
        print(f"❌ Dataset file {file_path} not found!")
        return

    # Load and analyze JSONL
    data = []
    story_lengths = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                data.append(entry)
                story_text = entry['messages'][1]['content']
                story_lengths.append(len(story_text))
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"\n📊 Dataset Information:")
    print(f"   - Total entries: {len(data)}")
    if story_lengths:
        print(f"   - Average story length: {sum(story_lengths) / len(story_lengths):.0f} characters")
        print(f"   - Longest story: {max(story_lengths)} characters")
        print(f"   - Shortest story: {min(story_lengths)} characters")


if __name__ == "__main__":
    # Create the dataset
    output_file = create_dataset_from_stories()

    # Show dataset information
    show_dataset_info(output_file)

    print(f"\n🚀 Dataset ready for fine-tuning!")
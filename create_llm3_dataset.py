# create_llm3_dataset.py
import os
import pandas as pd
import argparse
from pathlib import Path


def create_llm3_dataset(model_name):
    """
    Create LLM3 training dataset from extracted original and transformed stories

    Args:
        model_name: The model name (e.g., "gpt-4o", "claude-sonnet-4-5")
    """

    print(f"🚀 Creating LLM3 training dataset for model: {model_name}")

    # Paths
    input_dir = f"data/train_{model_name}"
    output_csv = f"data/llm3_training_dataset_{model_name}.csv"

    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Training data directory not found: {input_dir}")
        print(f"   Please run: python extract_transformed_stories.py {model_name}")

    # Check if CSV already exists
    if os.path.exists(output_csv):
        print(f"⚠️  Dataset CSV '{output_csv}' already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("📊 Loading existing dataset...")
            df = pd.read_csv(output_csv)
            print(f"✅ Dataset loaded: {len(df)} training examples")
            return output_csv

    # Get all story folders
    story_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    if len(story_folders) == 0:
        raise ValueError(f"No story folders found in {input_dir}")

    print(f"📁 Found {len(story_folders)} story folders")
    for story in sorted(story_folders):
        print(f"   - {story}")

    # Create dataset
    dataset_rows = []

    print(f"\n📚 Processing stories...")

    for story_name in story_folders:
        print(f"\n📖 Processing: {story_name}")

        story_path = os.path.join(input_dir, story_name)
        original_file = os.path.join(story_path, "original_story.txt")
        transformed_file = os.path.join(story_path, "transformed_story.txt")

        # Check if both files exist
        if not os.path.exists(original_file):
            print(f"   ❌ Missing original story: {original_file}")
            continue

        if not os.path.exists(transformed_file):
            print(f"   ❌ Missing transformed story: {transformed_file}")
            continue

        # Read original story
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_story = f.read().strip()
        except Exception as e:
            print(f"   ❌ Error reading original story: {e}")
            continue

        # Read transformed story
        try:
            with open(transformed_file, 'r', encoding='utf-8') as f:
                transformed_story = f.read().strip()
        except Exception as e:
            print(f"   ❌ Error reading transformed story: {e}")
            continue

        # Check if stories have content
        if not original_story:
            print(f"   ⚠️  Original story is empty")
            continue

        if not transformed_story:
            print(f"   ⚠️  Transformed story is empty")
            continue

        # Create prompt and response
        user_prompt = f"Transform this individualistic story to collectivistic: {original_story}"
        assistant_output = transformed_story

        dataset_rows.append({
            'user_prompt': user_prompt,
            'assistant_output': assistant_output
        })

        print(f"   ✅ Added training example")
        print(f"      - Original length: {len(original_story)} chars")
        print(f"      - Transformed length: {len(transformed_story)} chars")
        print(f"      - Prompt length: {len(user_prompt)} chars")

    if len(dataset_rows) == 0:
        raise ValueError("No valid training examples created!")

    # Create DataFrame and save
    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_csv, index=False)

    print(f"\n📊 LLM3 training dataset created successfully!")
    print(f"   - Total examples: {len(df)}")
    print(f"   - Saved to: {output_csv}")

    # Show statistics
    prompt_lengths = df['user_prompt'].str.len()
    output_lengths = df['assistant_output'].str.len()

    print(f"\n📈 Dataset Statistics:")
    print(f"   - Average prompt length: {prompt_lengths.mean():.0f} chars")
    print(f"   - Average output length: {output_lengths.mean():.0f} chars")
    print(f"   - Max prompt length: {prompt_lengths.max()} chars")
    print(f"   - Max output length: {output_lengths.max()} chars")
    print(f"   - Min prompt length: {prompt_lengths.min()} chars")
    print(f"   - Min output length: {output_lengths.min()} chars")

    return output_csv


def show_dataset_info(model_name):
    """Show information about the created LLM3 dataset"""
    csv_path = f"data/llm3_training_dataset_{model_name}.csv"

    if not os.path.exists(csv_path):
        print(f"❌ LLM3 dataset not found for model: {model_name}")
        print(f"   File not found: {csv_path}")
        print(f"   Run: python create_llm3_dataset.py {model_name}")
        return

    df = pd.read_csv(csv_path)

    print(f"\n📊 LLM3 Dataset Information for {model_name}:")
    print(f"   - Total training examples: {len(df)}")
    print(f"   - Location: {csv_path}")

    # Show sample
    if len(df) > 0:
        print(f"\n📖 Sample training example:")
        sample = df.iloc[0]
        print(f"   User Prompt (first 150 chars): {sample['user_prompt'][:150]}...")
        print(f"   Assistant Output (first 150 chars): {sample['assistant_output'][:150]}...")

    # Show all examples briefly
    print(f"\n📝 All training examples:")
    for i in range(len(df)):
        prompt = df.iloc[i]['user_prompt']
        output = df.iloc[i]['assistant_output']

        # Extract story name from prompt (after "collectivistic: ")
        story_start = prompt.find("collectivistic: ") + len("collectivistic: ")
        story_preview = prompt[story_start:story_start + 50] + "..."

        print(f"   {i + 1:2d}. Story preview: {story_preview} ({len(output)} chars output)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LLM3 training dataset from transformed stories")
    parser.add_argument("model_name", help="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
    parser.add_argument("--info", action="store_true", help="Show info about created dataset")

    args = parser.parse_args()

    if args.info:
        show_dataset_info(args.model_name)
    else:
        try:
            csv_file = create_llm3_dataset(args.model_name)
            show_dataset_info(args.model_name)

            print(f"\n🚀 LLM3 dataset ready for fine-tuning!")
            print(f"   Next step: Use {csv_file} to fine-tune your model")

        except Exception as e:
            print(f"❌ Error: {e}")
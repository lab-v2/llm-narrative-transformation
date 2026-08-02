# create_llm3_dataset.py (Updated for JSONL format)
import os
import pandas as pd
import json
import argparse
from pathlib import Path


def create_llm3_dataset(model_name):
    """
    Create LLM3 training dataset from extracted original and transformed stories
    Output format: JSONL with conversation structure

    Args:
        model_name: The model name (e.g., "gpt-4o", "claude-sonnet-4-5")
    """

    print(f"🚀 Creating LLM3 training dataset for model: {model_name}")

    # Paths
    input_dir = f"data/train_{model_name}"
    output_jsonl = f"data/llm3_training_dataset_{model_name}.jsonl"

    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Training data directory not found: {input_dir}")
        print(f"   Please run: python extract_transformed_stories.py {model_name}")

    # Check if JSONL already exists
    if os.path.exists(output_jsonl):
        print(f"⚠️  Dataset JSONL '{output_jsonl}' already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("📊 Loading existing dataset...")
            # Count lines in JSONL
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"✅ Dataset loaded: {count} training examples")
            return output_jsonl

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

        # Create conversation in the format you showed
        conversation = {
            "schemaVersion": "bedrock-conversation-2024",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": f"Transform this individualistic story to collectivistic: {original_story}"}]
                },
                {
                    "role": "assistant",
                    "content": [{"text": transformed_story}]
                }
            ]
        }

        dataset_rows.append(conversation)

        print(f"   ✅ Added training example")
        print(f"      - Original length: {len(original_story)} chars")
        print(f"      - Transformed length: {len(transformed_story)} chars")

    if len(dataset_rows) == 0:
        raise ValueError("No valid training examples created!")

    # Save as JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for conversation in dataset_rows:
            json.dump(conversation, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n📊 LLM3 training dataset created successfully!")
    print(f"   - Total examples: {len(dataset_rows)}")
    print(f"   - Saved to: {output_jsonl}")

    # Show statistics
    user_lengths = []
    assistant_lengths = []

    for conv in dataset_rows:
        user_text = conv['messages'][0]['content'][0]['text']
        assistant_text = conv['messages'][1]['content'][0]['text']
        user_lengths.append(len(user_text))
        assistant_lengths.append(len(assistant_text))

    print(f"\n📈 Dataset Statistics:")
    print(f"   - Average user prompt length: {sum(user_lengths) / len(user_lengths):.0f} chars")
    print(f"   - Average assistant response length: {sum(assistant_lengths) / len(assistant_lengths):.0f} chars")
    print(f"   - Max user prompt length: {max(user_lengths)} chars")
    print(f"   - Max assistant response length: {max(assistant_lengths)} chars")
    print(f"   - Min user prompt length: {min(user_lengths)} chars")
    print(f"   - Min assistant response length: {min(assistant_lengths)} chars")

    return output_jsonl


def show_dataset_info(model_name):
    """Show information about the created LLM3 dataset"""
    jsonl_path = f"data/llm3_training_dataset_{model_name}.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"❌ LLM3 dataset not found for model: {model_name}")
        print(f"   File not found: {jsonl_path}")
        print(f"   Run: python create_llm3_dataset.py {model_name}")
        return

    # Load JSONL
    conversations = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line.strip()))

    print(f"\n📊 LLM3 Dataset Information for {model_name}:")
    print(f"   - Total training conversations: {len(conversations)}")
    print(f"   - Location: {jsonl_path}")

    # Show sample
    if len(conversations) > 0:
        print(f"\n📖 Sample training conversation:")
        sample = conversations[0]
        user_text = sample['messages'][0]['content'][0]['text']
        assistant_text = sample['messages'][1]['content'][0]['text']
        print(f"   User: {user_text[:100]}...")
        print(f"   Assistant: {assistant_text[:100]}...")

    # Show all examples briefly
    print(f"\n📝 All training conversations:")
    for i, conv in enumerate(conversations, 1):
        user_text = conv['messages'][0]['content'][0]['text']
        assistant_text = conv['messages'][1]['content'][0]['text']

        # Extract story preview from user message
        if "collectivistic: " in user_text:
            story_start = user_text.find("collectivistic: ") + len("collectivistic: ")
            story_preview = user_text[story_start:story_start + 50] + "..."
        else:
            story_preview = user_text[:50] + "..."

        print(f"   {i:2d}. Story: {story_preview} ({len(assistant_text)} chars response)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LLM3 training dataset from transformed stories (JSONL format)")
    parser.add_argument("model_name", help="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
    parser.add_argument("--info", action="store_true", help="Show info about created dataset")

    args = parser.parse_args()

    if args.info:
        show_dataset_info(args.model_name)
    else:
        try:
            jsonl_file = create_llm3_dataset(args.model_name)
            show_dataset_info(args.model_name)

            print(f"\n🚀 LLM3 dataset ready for fine-tuning!")
            print(f"   Dataset format: JSONL with conversation structure")
            print(f"   Next step: Use {jsonl_file} to fine-tune your transformation model")

        except Exception as e:
            print(f"❌ Error: {e}")
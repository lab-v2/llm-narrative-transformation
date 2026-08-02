# create_train_test_split.py
import os
import pandas as pd
import random
from pathlib import Path


def create_train_test_split(stories_folder="data/individualistic-rags-to-riches-stories-subset/",
                            output_csv="data/train_test_split.csv",
                            train_ratio=0.5,
                            random_seed=42):
    """
    Create a train/test split CSV for individualistic stories
    - Creates CSV with columns: story_name, split
    - Randomly assigns stories to 'train' or 'test' groups
    """

    stories_path = Path(stories_folder)

    # Check if CSV already exists
    if os.path.exists(output_csv):
        print(f"⚠️  Split CSV '{output_csv}' already exists!")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("📊 Loading existing split...")
            df = pd.read_csv(output_csv)
            print(f"✅ Split loaded: {len(df)} stories")
            train_count = len(df[df['split'] == 'train'])
            test_count = len(df[df['split'] == 'test'])
            print(f"   - Train: {train_count} stories")
            print(f"   - Test: {test_count} stories")
            return output_csv

    # Get all .txt files
    txt_files = list(stories_path.glob("*.txt"))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in {stories_folder}")

    print(f"📁 Found {len(txt_files)} individualistic story files")

    # Extract story names (without .txt extension)
    story_names = [txt_file.stem for txt_file in txt_files]

    print(f"📚 Story names preview:")
    for i, name in enumerate(story_names[:5]):
        print(f"   {i + 1}. {name}")
    if len(story_names) > 5:
        print(f"   ... and {len(story_names) - 5} more")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Randomly shuffle the story names
    shuffled_names = story_names.copy()
    random.shuffle(shuffled_names)

    # Calculate split point
    train_count = int(len(shuffled_names) * train_ratio)

    # Split into train and test
    train_stories = shuffled_names[:train_count]
    test_stories = shuffled_names[train_count:]

    print(f"\n🎯 Split configuration:")
    print(f"   - Train ratio: {train_ratio}")
    print(f"   - Random seed: {random_seed}")
    print(f"   - Train stories: {len(train_stories)}")
    print(f"   - Test stories: {len(test_stories)}")

    # Create dataset
    dataset_rows = []

    # Add train stories
    for story_name in train_stories:
        dataset_rows.append({
            'story_name': story_name,
            'split': 'train'
        })

    # Add test stories
    for story_name in test_stories:
        dataset_rows.append({
            'story_name': story_name,
            'split': 'test'
        })

    # Create DataFrame and save
    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_csv, index=False)

    print(f"\n📊 Train/Test split created successfully!")
    print(f"   - Total stories: {len(df)}")
    print(f"   - Train stories: {len(df[df['split'] == 'train'])}")
    print(f"   - Test stories: {len(df[df['split'] == 'test'])}")
    print(f"   - Saved to: {output_csv}")

    return output_csv


def show_split_info(csv_path="data/train_test_split.csv"):
    """Show detailed information about the train/test split"""
    if not os.path.exists(csv_path):
        print(f"❌ Split file {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)

    train_stories = df[df['split'] == 'train']['story_name'].tolist()
    test_stories = df[df['split'] == 'test']['story_name'].tolist()

    print(f"\n📊 Split Information:")
    print(f"   - Total stories: {len(df)}")
    print(f"   - Train: {len(train_stories)} stories")
    print(f"   - Test: {len(test_stories)} stories")
    print(f"   - Split ratio: {len(train_stories) / len(df):.1%} train, {len(test_stories) / len(df):.1%} test")

    print(f"\n🚂 Train stories (Group 1 - for LLM3 training):")
    for i, story in enumerate(train_stories, 1):
        print(f"   {i:2d}. {story}")

    print(f"\n🧪 Test stories (Group 2 - for evaluation):")
    for i, story in enumerate(test_stories, 1):
        print(f"   {i:2d}. {story}")


if __name__ == "__main__":
    # Configuration
    STORIES_FOLDER = "data/individualistic-rags-to-riches-stories-subset/"
    OUTPUT_CSV = "data/train_test_split.csv"
    TRAIN_RATIO = 0.5  # 50-50 split
    RANDOM_SEED = 42  # For reproducibility

    print(f"🔀 Creating train/test split for individualistic stories...")
    print(f"📁 Source folder: {STORIES_FOLDER}")
    print(f"💾 Output file: {OUTPUT_CSV}")
    print(f"🎲 Train ratio: {TRAIN_RATIO}")

    try:
        # Create the split
        csv_file = create_train_test_split(
            stories_folder=STORIES_FOLDER,
            output_csv=OUTPUT_CSV,
            train_ratio=TRAIN_RATIO,
            random_seed=RANDOM_SEED
        )

        # Show detailed information
        show_split_info(csv_file)

        print(f"\n🚀 Train/test split ready!")

    except Exception as e:
        print(f"❌ Error creating split: {e}")
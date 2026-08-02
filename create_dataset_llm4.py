# create_dataset_llm4.py
import json
import os


def create_recipe_dataset(json_file_path="data/recipe_corpus_finetune.json", output_file="data/recipe_dataset_llm4.jsonl"):
    """
    Create a recipe dataset JSONL from JSON file in OpenAI messages format
    - Output format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - Prompt format: "Write a recipe for [dish]"
    - Recipe: numbered instructions from recipeInstructions
    """

    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

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
            print(f"✅ Dataset loaded: {len(data)} recipes")
            return output_file

    print(f"📁 Reading JSON file: {json_file_path}")

    # Read JSON lines file
    recipes = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                recipe_data = json.loads(line.strip())
                recipes.append(recipe_data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {line_num}: {e}")
                continue

    print(f"📚 Found {len(recipes)} recipes in JSON file")

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0

    # Process recipes and write directly to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for recipe in recipes:
            try:
                # Get dish name - handle "Unknown" case
                dish = recipe.get('dish', 'Unknown')
                if dish == 'Unknown' or not dish.strip():
                    # Fall back to name if dish is Unknown or empty
                    dish = recipe.get('name', 'Unknown Dish')

                # Create prompt
                prompt = f"Write a recipe for {dish}"

                # Get recipe instructions
                instructions = recipe.get('recipeInstructions', [])
                if isinstance(instructions, list):
                    # Join instructions with numbered steps
                    recipe_text = ""
                    for i, instruction in enumerate(instructions, 1):
                        recipe_text += f"{i}. {instruction}\n"
                    recipe_text = recipe_text.strip()
                else:
                    # If instructions is not a list, convert to string
                    recipe_text = str(instructions)

                if recipe_text:  # Only add if we have instructions
                    # Create message format for fine-tuning
                    message_obj = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": recipe_text}
                        ]
                    }
                    json.dump(message_obj, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
                    print(f"✅ Added recipe: {dish}")
                else:
                    print(f"⚠️  Skipping recipe with no instructions: {dish}")

            except Exception as e:
                print(f"❌ Error processing recipe {recipe.get('name', 'Unknown')}: {e}")

    print(f"\n📊 Recipe dataset created successfully!")
    print(f"   - Total recipes: {count}")
    print(f"   - Saved to: {output_file}")

    # Show sample
    if count > 0:
        print(f"\n📖 Sample entry:")
        with open(output_file, 'r', encoding='utf-8') as f:
            sample_line = f.readline()
            sample = json.loads(sample_line)
            print(f"   User prompt: {sample['messages'][0]['content']}")
            print(f"   Recipe preview: {sample['messages'][1]['content'][:200]}...")

    return output_file


def show_recipe_dataset_info(file_path="data/recipe_dataset_llm4.jsonl"):
    """Show information about the created recipe dataset"""
    if not os.path.exists(file_path):
        print(f"❌ Dataset file {file_path} not found!")
        return

    data = []
    recipe_lengths = []
    dish_names = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                data.append(entry)
                recipe_text = entry['messages'][1]['content']
                recipe_lengths.append(len(recipe_text))
                prompt = entry['messages'][0]['content']
                dish = prompt.replace("Write a recipe for ", "")
                dish_names.append(dish)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"\n📊 Recipe Dataset Information:")
    print(f"   - Total recipes: {len(data)}")
    if recipe_lengths:
        print(f"   - Average recipe length: {sum(recipe_lengths) / len(recipe_lengths):.0f} characters")
        print(f"   - Longest recipe: {max(recipe_lengths)} characters")
        print(f"   - Shortest recipe: {min(recipe_lengths)} characters")

    print(f"   - Unique dishes: {len(set(dish_names))}")

    print(f"\n📝 Sample recipes:")
    for i in range(min(5, len(data))):
        prompt = data[i]['messages'][0]['content']
        recipe_length = len(data[i]['messages'][1]['content'])
        print(f"   {i + 1}. {prompt} ({recipe_length} chars)")

    if len(data) > 5:
        print(f"   ... and {len(data) - 5} more recipes")


if __name__ == "__main__":
    # Use the specific JSON file
    json_file_path = "data/recipe_corpus_finetune.json"

    print(f"📁 Processing recipe corpus: {json_file_path}")

    # Create the dataset
    try:
        jsonl_file = create_recipe_dataset(json_file_path)

        # Show dataset information
        show_recipe_dataset_info(jsonl_file)

        print(f"\n🚀 Recipe dataset ready for fine-tuning!")
        print(f"   Dataset saved as: {jsonl_file}")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        print(f"   Make sure {json_file_path} exists and is accessible")
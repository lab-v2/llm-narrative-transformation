"""
Create LLM4 OpenAI Fine-tuning Dataset (Recipe Control)

This script creates an OpenAI fine-tuning dataset from Chinese recipe data
for training LLM4 (recipe generation control model).

Input: data/recipe_corpus_finetune.json (recipe data)
Output: llm4_ft_openai.jsonl (OpenAI fine-tuning format)
"""

import json
import os
import argparse
from pathlib import Path

def create_recipe_openai_dataset(
    json_file_path: str = "data/recipe_corpus_finetune.json", 
    output_file: str = "llm4_ft_openai.jsonl",
    max_samples: int = 1000
):
    """
    Create a recipe dataset in OpenAI format from JSON file
    
    Args:
        json_file_path: Path to recipe JSON file
        output_file: Output JSONL file path  
        max_samples: Maximum number of recipes to include
    """
    
    # Prompt variations for recipes
    prompt_variations = [
        "Write a recipe for {dish}",
        "Create a recipe for {dish}", 
        "Share the recipe for {dish}",
        "How do you make {dish}?",
        "Provide cooking instructions for {dish}",
        "Show me how to cook {dish}",
        "Give me the recipe for {dish}",
        "What's the recipe for {dish}?"
    ]
    
    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    print(f"📁 Reading recipe data: {json_file_path}")
    print(f"🎯 Target samples: {max_samples}")

    # Read JSON lines file
    recipes = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                recipe_data = json.loads(line.strip())
                recipes.append(recipe_data)
                
                # Stop reading if we have enough recipes
                if len(recipes) >= max_samples:
                    print(f"📊 Reached target of {max_samples} recipes")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {line_num}: {e}")
                continue

    print(f"📚 Loaded {len(recipes)} recipes from JSON file")

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    # Process recipes and write to OpenAI format
    with open(output_path, 'w', encoding='utf-8') as f:
        
        for recipe in recipes:
            try:
                # Get dish name - handle "Unknown" case
                dish = recipe.get('dish', 'Unknown')
                if dish == 'Unknown' or not dish.strip():
                    # Fall back to name if dish is Unknown or empty
                    dish = recipe.get('name', 'Unknown Dish')

                # Use rotating prompts for variety
                prompt_template = prompt_variations[converted_count % len(prompt_variations)]
                prompt = prompt_template.format(dish=dish)

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
                    # Create OpenAI format entry
                    entry = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": recipe_text}
                        ]
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                    # Progress feedback
                    if converted_count % 100 == 0:
                        print(f"✅ Processed {converted_count} recipes...")
                else:
                    print(f"⚠️  Skipping recipe with no instructions: {dish}")
                    skipped_count += 1

            except Exception as e:
                print(f"❌ Error processing recipe {recipe.get('name', 'Unknown')}: {e}")
                skipped_count += 1
                continue

    print(f"\n🎉 OpenAI recipe dataset created successfully!")
    print(f"   📁 Input file: {json_file_path}")
    print(f"   📄 Output file: {output_file}")
    print(f"   ✅ Converted recipes: {converted_count}")
    if skipped_count > 0:
        print(f"   ❌ Skipped recipes: {skipped_count}")
    print(f"   📊 Total processed: {converted_count + skipped_count}")

    return converted_count

def validate_openai_format(file_path: str, sample_size: int = 3):
    """Validate and show sample of OpenAI format file."""
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return 0
    
    print(f"\n🔍 Validating OpenAI format: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries = []
            total_entries = 0
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    
                    # Validate structure
                    if "messages" not in entry:
                        print(f"❌ Line {line_num}: Missing 'messages' key")
                        continue
                        
                    messages = entry["messages"]
                    if len(messages) != 2:
                        print(f"❌ Line {line_num}: Expected 2 messages, got {len(messages)}")
                        continue
                        
                    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                        print(f"❌ Line {line_num}: Invalid roles")
                        continue
                    
                    # Collect samples
                    if len(entries) < sample_size:
                        entries.append(entry)
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Line {line_num}: JSON decode error: {e}")
                    continue
        
        print(f"✅ Validation complete: {total_entries} valid entries found")
        
        if entries:
            print(f"\n📋 Sample entries:")
            for i, entry in enumerate(entries, 1):
                user_msg = entry['messages'][0]['content']
                assistant_msg = entry['messages'][1]['content']
                
                print(f"\n--- Sample {i} ---")
                print(f"👤 User: {user_msg}")
                print(f"🍜 Recipe: {assistant_msg[:150]}{'...' if len(assistant_msg) > 150 else ''}")
        
        return total_entries
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Create LLM4 OpenAI fine-tuning dataset from Chinese recipes"
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='data/recipe_corpus_finetune.json',
        help='Input recipe JSON file (default: data/recipe_corpus_finetune.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str, 
        default='llm4_ft_openai.jsonl',
        help='Output file in OpenAI format (default: llm4_ft_openai.jsonl)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Maximum number of recipes to include (default: 1000)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate and show sample of output file after creation'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true', 
        help='Only validate existing output file (skip creation)'
    )
    
    args = parser.parse_args()
    
    try:
        if not args.validate_only:
            # Create the dataset
            count = create_recipe_openai_dataset(args.input, args.output, args.samples)
            
            if count == 0:
                print("❌ No recipes were converted!")
                return 1
        
        if args.validate or args.validate_only:
            # Validate the output
            total_entries = validate_openai_format(args.output)
            
            if total_entries == 0:
                print("❌ No valid entries found!")
                return 1
            
            print(f"\n🎯 Recipe dataset ready for OpenAI fine-tuning!")
            print(f"   📄 File: {args.output}")
            print(f"   📊 Entries: {total_entries}")
            print(f"\n💡 Next steps:")
            print(f"   1. Upload to OpenAI: openai api files.create -f {args.output} -p fine-tune")
            print(f"   2. Start fine-tuning job with the uploaded file ID")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

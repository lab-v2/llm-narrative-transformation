"""
Create LLM3 OpenAI Fine-tuning Dataset (Story Transformation)

This script creates an OpenAI fine-tuning dataset from story transformation pairs
for training LLM3 (individualistic to collectivistic transformation model).

Input: evaluation_data/gpt-4o-mini-2024-07-18/individualistic/ (story folders)
Output: llm3_ft_gpt-4o-mini_openai.jsonl (OpenAI fine-tuning format)
"""

import json
import argparse
from pathlib import Path

def create_transformation_openai_dataset(
    data_dir: str = "evaluation_data/gpt-4o-mini-2024-07-18/individualistic/", 
    output_file: str = "llm3_ft_gpt-4o-mini_openai.jsonl",
    max_samples: int = 1000
):
    """
    Create a story transformation dataset in OpenAI format
    
    Args:
        data_dir: Directory containing story folders
        output_file: Output JSONL file path  
        max_samples: Maximum number of story pairs to include
    """
    
    # Transformation prompt variations
    prompt_variations = [
        "Transform this individualistic story to collectivistic:\n\n{original}",
        "Rewrite this story to emphasize community values:\n\n{original}",
        "Make this story more collectivistic:\n\n{original}",
        "Change this individual-focused story to group-focused:\n\n{original}",
        "Convert this story from individualistic to collectivistic:\n\n{original}",
        "Transform this story to focus on community rather than individual:\n\n{original}",
        "Rewrite this story with collectivistic values:\n\n{original}",
        "Make this story emphasize group harmony over individual goals:\n\n{original}"
    ]
    
    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"📁 Reading transformation data: {data_dir}")
    print(f"🎯 Target samples: {max_samples}")

    # Find all story folders
    story_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not story_folders:
        raise ValueError(f"No story folders found in {data_dir}")
    
    print(f"📚 Found {len(story_folders)} story folders")

    converted_count = 0
    skipped_count = 0

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process story transformation pairs
    with open(output_path, 'w', encoding='utf-8') as outfile:
        
        for story_folder in sorted(story_folders):  # Sort for consistent order
            
            # Stop if we've reached max samples
            if converted_count >= max_samples:
                print(f"📊 Reached target of {max_samples} transformations")
                break
                
            try:
                story_name = story_folder.name
                
                # Look for original and transformed story files
                original_file = story_folder / "original_story.txt"
                transformed_file = story_folder / "abduction_transformed.txt"
                
                # Check if both files exist
                if not original_file.exists():
                    print(f"⚠️  Missing original_story.txt in {story_name}")
                    skipped_count += 1
                    continue
                    
                if not transformed_file.exists():
                    print(f"⚠️  Missing abduction_transformed.txt in {story_name}")
                    skipped_count += 1
                    continue
                
                # Read both story files
                original_story = original_file.read_text(encoding='utf-8').strip()
                transformed_story = transformed_file.read_text(encoding='utf-8').strip()
                
                # Skip if either file is empty
                if not original_story or not transformed_story:
                    print(f"⚠️  Empty story files in {story_name}")
                    skipped_count += 1
                    continue
                
                # Use rotating prompts for variety
                prompt_template = prompt_variations[converted_count % len(prompt_variations)]
                user_prompt = prompt_template.format(original=original_story)
                
                # Create OpenAI format entry
                entry = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": transformed_story}
                    ]
                }
                
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Progress feedback
                if converted_count % 10 == 0:
                    print(f"✅ Processed {converted_count} transformations...")
                    
            except Exception as e:
                print(f"❌ Error processing {story_folder.name}: {e}")
                skipped_count += 1
                continue

    print(f"\n🎉 OpenAI transformation dataset created successfully!")
    print(f"   📁 Input directory: {data_dir}")
    print(f"   📄 Output file: {output_file}")
    print(f"   ✅ Converted transformations: {converted_count}")
    if skipped_count > 0:
        print(f"   ❌ Skipped folders: {skipped_count}")
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
                print(f"👤 User: {user_msg[:100]}...")
                print(f"🔄 Transformed: {assistant_msg[:150]}{'...' if len(assistant_msg) > 150 else ''}")
        
        return total_entries
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Create LLM3 OpenAI fine-tuning dataset from story transformations"
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='evaluation_data/gpt-4o-mini-2024-07-18/individualistic/',
        help='Input directory containing story folders (default: evaluation_data/gpt-4o-mini-2024-07-18/individualistic/)'
    )
    
    parser.add_argument(
        '--output',
        type=str, 
        default='llm3_ft_gpt-4o-mini_openai.jsonl',
        help='Output file in OpenAI format (default: llm3_ft_gpt-4o-mini_openai.jsonl)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Maximum number of transformations to include (default: 1000)'
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
            count = create_transformation_openai_dataset(args.input, args.output, args.samples)
            
            if count == 0:
                print("❌ No transformations were converted!")
                return 1
        
        if args.validate or args.validate_only:
            # Validate the output
            total_entries = validate_openai_format(args.output)
            
            if total_entries == 0:
                print("❌ No valid entries found!")
                return 1
            
            print(f"\n🎯 Transformation dataset ready for OpenAI fine-tuning!")
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

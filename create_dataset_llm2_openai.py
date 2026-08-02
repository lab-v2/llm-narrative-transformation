"""
Create LLM2 OpenAI Fine-tuning Dataset

This script creates an OpenAI fine-tuning dataset from raw collectivistic story files
for training LLM2 (collectivistic story generation model).

Input: data/collectivistic-stories-all/*.txt (raw story files)
Output: llm2_ft_openai.jsonl (OpenAI fine-tuning format)
"""

import json
import argparse
from pathlib import Path

def create_openai_dataset(stories_dir: str, output_file: str, prompt_variations: list = None):
    """
    Create OpenAI dataset directly from raw collectivistic story files.
    
    Args:
        stories_dir: Directory containing .txt story files
        output_file: Output JSONL file path
        prompt_variations: List of different prompts to use
    """
    if prompt_variations is None:
        prompt_variations = [
            "Write a collectivistic story",
            "Create a story that emphasizes community values", 
            "Tell a story about group cooperation and shared identity",
            "Write a narrative focused on collective well-being",
            "Create a story where community comes before individual needs",
            "Write a story about interdependence and social harmony",
            "Tell a story that celebrates collective achievement",
            "Create a narrative about family loyalty and group solidarity"
        ]
    
    stories_path = Path(stories_dir)
    output_path = Path(output_file)
    
    if not stories_path.exists():
        raise FileNotFoundError(f"Stories directory not found: {stories_dir}")
    
    story_files = list(stories_path.glob("*.txt"))
    if not story_files:
        raise ValueError(f"No .txt files found in {stories_dir}")
    
    print(f"📚 Found {len(story_files)} story files in {stories_dir}")
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        
        for story_file in sorted(story_files):  # Sort for consistent order
            try:
                # Read story content
                story_content = story_file.read_text(encoding='utf-8').strip()
                
                # Use rotating prompts for variety
                prompt = prompt_variations[converted_count % len(prompt_variations)]
                
                # Create OpenAI format entry
                entry = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": story_content}
                    ]
                }
                
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Progress feedback
                if converted_count % 5 == 0:
                    print(f"✅ Processed {converted_count} stories...")
                
            except Exception as e:
                print(f"❌ Error processing {story_file.name}: {e}")
                skipped_count += 1
                continue
    
    print(f"\n🎉 OpenAI dataset created successfully!")
    print(f"   📁 Stories directory: {stories_dir}")
    print(f"   📄 Output file: {output_file}")
    print(f"   ✅ Converted stories: {converted_count}")
    if skipped_count > 0:
        print(f"   ❌ Failed stories: {skipped_count}")
    print(f"   📊 Total files processed: {converted_count + skipped_count}")
    
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
                print(f"🤖 Assistant: {assistant_msg[:150]}{'...' if len(assistant_msg) > 150 else ''}")
        
        return total_entries
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Create LLM2 OpenAI fine-tuning dataset from collectivistic stories"
    )
    
    parser.add_argument(
        '--stories-dir', 
        type=str, 
        default='data/collectivistic-stories-all',
        help='Directory containing raw story .txt files (default: data/collectivistic-stories-all)'
    )
    
    parser.add_argument(
        '--output',
        type=str, 
        default='llm2_ft_openai.jsonl',
        help='Output file in OpenAI format (default: llm2_ft_openai.jsonl)'
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
            count = create_openai_dataset(args.stories_dir, args.output)
            
            if count == 0:
                print("❌ No stories were converted!")
                return 1
        
        if args.validate or args.validate_only:
            # Validate the output
            total_entries = validate_openai_format(args.output)
            
            if total_entries == 0:
                print("❌ No valid entries found!")
                return 1
            
            print(f"\n🎯 Dataset ready for OpenAI fine-tuning!")
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

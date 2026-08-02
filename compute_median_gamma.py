import json
import os
from pathlib import Path
import statistics
import csv
import re


def clean_text(text):
    """
    Replace all problematic characters with standard ASCII equivalents
    """
    if not isinstance(text, str):
        return text

    # Replace smart quotes and apostrophes with standard versions
    text = text.replace(''', "'")        # left single quote
    text = text.replace(''', "'")  # right single quote
    text = text.replace('`', "'")  # backtick
    text = text.replace('"', '"')  # left double quote
    text = text.replace('"', '"')  # right double quote
    text = text.replace('„', '"')  # double low-9 quote
    text = text.replace('"', '"')  # other variants

    # Replace dashes
    text = text.replace('–', '-')  # en dash
    text = text.replace('—', '-')  # em dash
    text = text.replace('−', '-')  # minus sign

    # Replace ellipsis
    text = text.replace('…', '...')  # ellipsis

    # Handle mangled UTF-8 encoding (like â€™ for ')
    text = text.replace('â€™', "'")
    text = text.replace('â€˜', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€\x9d', '"')
    text = text.replace('â€"', '-')

    # Remove any other non-ASCII characters that might cause issues
    # Keep letters, numbers, spaces, and basic punctuation
    text = ''.join(c if ord(c) < 128 or c.isalpha() else '' for c in text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def calculate_gamma_by_words(story_file, prescriptions_file):
    """
    Calculate gamma based on word count:
    γ = (words_in_excerpts / total_story_words) × 100%
    """

    # Step 1: Read story and count total words
    try:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_text = f.read()
        story_text = clean_text(story_text)
        total_words = len(story_text.split())
    except Exception as e:
        print(f"  Error reading story: {e}")
        return 0, 0, 0.0, 0

    # Step 2: Make a temp copy
    story_temp = story_text

    # Step 3: Read prescriptions and extract unique excerpts
    try:
        with open(prescriptions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prescriptions = data.get('prescriptions', [])
    except Exception as e:
        print(f"  Error reading prescriptions: {e}")
        return 0, total_words, 0.0, 0

    # Extract and clean unique excerpts
    unique_excerpts = set()
    for prescription in prescriptions:
        segment_text = prescription.get('segment_text', '').strip()
        if segment_text:
            segment_cleaned = clean_text(segment_text)
            if segment_cleaned:  # Only add if not empty after cleaning
                unique_excerpts.add(segment_cleaned)

    # Step 4: Remove each unique excerpt from temp story
    words_removed = 0

    for excerpt in unique_excerpts:
        # Simple exact match - if excerpt is in story
        if excerpt in story_temp:
            # Count words in this excerpt
            excerpt_word_count = len(excerpt.split())

            # Remove it (only first occurrence)
            story_temp = story_temp.replace(excerpt, "", 1)

            words_removed += excerpt_word_count

    # Step 5: Count remaining words
    remaining_words = len(story_temp.split())

    # Calculate actual words removed
    actual_words_removed = total_words - remaining_words

    # Calculate gamma
    gamma = (actual_words_removed / total_words * 100.0) if total_words > 0 else 0.0

    return actual_words_removed, total_words, gamma, len(unique_excerpts)


def main():
    base_path = input("Enter full path to your phase2/gpt-4o/forward/ directory: ").strip()

    if not os.path.exists(base_path):
        print(f"Error: {base_path} does not exist")
        return

    results = []

    print("\nProcessing stories...")
    print("=" * 90)

    for story_dir in sorted(Path(base_path).iterdir()):
        if not story_dir.is_dir():
            continue

        story_name = story_dir.name
        story_file = story_dir / "iteration_0" / "story.txt"
        prescriptions_file = story_dir / "iteration_0" / "ranked_prescriptions.json"

        if not story_file.exists() or not prescriptions_file.exists():
            print(f"✗ {story_name:45s} | Files not found")
            continue

        try:
            words_removed, total_words, gamma, unique_excerpts = calculate_gamma_by_words(story_file,
                                                                                          prescriptions_file)

            results.append({
                'story_name': story_name,
                'words_in_excerpts': words_removed,
                'total_story_words': total_words,
                'gamma_percent': gamma,
                'unique_excerpts': unique_excerpts
            })

            print(f"✓ {story_name:45s} | {words_removed:4d}/{total_words:4d} words | γ = {gamma:6.2f}%")

        except Exception as e:
            print(f"✗ {story_name:45s} | Error: {e}")

    if not results:
        print("No stories processed!")
        return

    # Save to CSV
    output_file = "gamma_results_wordcount.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['story_name', 'words_in_excerpts', 'total_story_words', 'gamma_percent',
                                               'unique_excerpts'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Saved to {output_file}")

    # Compute statistics
    gamma_values = [r['gamma_percent'] for r in results]

    print("\n" + "=" * 90)
    print("GAMMA STATISTICS")
    print("=" * 90)
    print(f"Total stories:        {len(results)}")
    print(f"Median γ:             {statistics.median(gamma_values):.2f}%")
    print(f"Mean γ:               {statistics.mean(gamma_values):.2f}%")
    if len(results) > 1:
        print(f"Std Dev:              {statistics.stdev(gamma_values):.2f}%")
    print(f"Min γ:                {min(gamma_values):.2f}%")
    print(f"Max γ:                {max(gamma_values):.2f}%")
    print(f"Q1 (25th percentile): {sorted(gamma_values)[len(gamma_values) // 4]:.2f}%")
    print(f"Q3 (75th percentile): {sorted(gamma_values)[3 * len(gamma_values) // 4]:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
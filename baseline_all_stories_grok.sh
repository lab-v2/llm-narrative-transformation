#!/bin/bash
# run_baseline_all_stories.sh
#
# Run baseline transformation for all test stories.
# Skips stories that already have baseline results.

# ==========================================================
# Configuration
# ==========================================================

PROBLEM="inverse"
MODEL="ft:gpt-4o-mini-2024-07-18:syracuse-university:llm2:D5JJuHZi"
TEMPERATURE=0.7
DATA_DIR="data"
PHASE2_DIR="output/phase2"
OUTPUT_DIR="output/baseline"

# Story directory (based on problem type)
if [ "$PROBLEM" == "forward" ]; then
    STORIES_DIR="data/collectivistic-stories-all"
else
    STORIES_DIR="data/individualistic-rags-to-riches-stories-subset-subset"
fi

# ==========================================================
# Script
# ==========================================================

echo "========================================"
echo "Baseline Transformation: All Stories"
echo "========================================"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Stories directory: $STORIES_DIR"
echo "========================================"
echo ""

# Check if directory exists
if [ ! -d "$STORIES_DIR" ]; then
    echo "Error: Stories directory not found: $STORIES_DIR"
    exit 1
fi

# Count total stories
TOTAL_STORIES=$(find "$STORIES_DIR" -name "*.txt" | wc -l)
echo "Found $TOTAL_STORIES stories"
echo ""

# Process each story
CURRENT=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

for story_file in "$STORIES_DIR"/*.txt; do
    CURRENT=$((CURRENT + 1))
    story_name=$(basename "$story_file" .txt)

    # Check if this story already has baseline results
    sanitized_model=$(echo "$MODEL" | sed 's/\//-/g' | sed 's/:/-/g')
    story_output_dir="$OUTPUT_DIR/$sanitized_model/$PROBLEM/$story_name"

    if [ -d "$story_output_dir" ] && [ -f "$story_output_dir/transformed_survey.json" ]; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL_STORIES] SKIPPING (already processed): $story_name"
        echo "========================================"
        echo ""
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL_STORIES] Processing: $story_name"
    echo "========================================"

    # Run baseline transformation
    python baseline_transform.py \
        --story "$story_file" \
        --problem $PROBLEM \
        --model $MODEL \
        --temperature $TEMPERATURE \
        --data-dir $DATA_DIR \
        --phase2-dir $PHASE2_DIR \
        --output-dir $OUTPUT_DIR

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ Success: $story_name"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed: $story_name"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total stories: $TOTAL_STORIES"
echo "Successful: $SUCCESSFUL"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo "========================================"
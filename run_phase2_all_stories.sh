#!/bin/bash
# run_phase2_all_stories.sh
#
# Run Phase 2 transformation for all test stories in a directory.
# This script processes stories sequentially (one after another).
# For parallel execution on HTCondor, see run_phase2_parallel.sh

# ==========================================================
# Configuration
# ==========================================================

# Phase 2 parameters
PHASE=2
PROBLEM="forward"
MODEL="gpt-5.2"
MAX_ITERATIONS=3
TOP_K=20
TEMPERATURE=0.7
DATA_DIR="data"
OUTPUT_DIR="output"

# Story directory (change based on problem type)
# For forward: use collectivistic stories
# For inverse: use individualistic stories
if [ "$PROBLEM" == "forward" ]; then
    STORIES_DIR="data/collectivistic-stories-all"
else
    STORIES_DIR="data/individualistic-rags-to-riches-stories-subset"
fi

# ==========================================================
# Script
# ==========================================================

echo "========================================"
echo "Phase 2: Running All Test Stories"
echo "========================================"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "Max iterations: $MAX_ITERATIONS"
echo "Top-k features: $TOP_K"
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
echo "Found $TOTAL_STORIES stories to process"
echo ""

# Process each story
CURRENT=0
SUCCESSFUL=0
FAILED=0

for story_file in "$STORIES_DIR"/*.txt; do
    CURRENT=$((CURRENT + 1))
    story_name=$(basename "$story_file" .txt)  # Remove .txt extension

    # Check if this story already has results
    story_output_dir="$OUTPUT_DIR/phase2/${MODEL//\//-}/$PROBLEM/$story_name"

    if [ -d "$story_output_dir" ] && [ -f "$story_output_dir/iteration_0/survey.json" ]; then
        echo "========================================"
        echo "[$CURRENT/$TOTAL_STORIES] SKIPPING (already processed): $story_name"
        echo "========================================"
        echo ""
        SUCCESSFUL=$((SUCCESSFUL + 1))
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL_STORIES] Processing: $story_name"
    echo "========================================"
    # Run Phase 2 for this story
    python main.py \
        --phase $PHASE \
        --problem $PROBLEM \
        --model $MODEL \
        --story "$story_file" \
        --max-iterations $MAX_ITERATIONS \
        --top-k $TOP_K \
        --temperature $TEMPERATURE \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --verbose

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
echo "Failed: $FAILED"
echo "========================================"
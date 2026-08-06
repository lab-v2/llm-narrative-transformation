#!/bin/bash
# baseline_all_stories_cot_claude.sh
#
# Run the CoT (chain-of-thought) baseline transformation for all test stories
# for a single model. Skips stories that already have CoT baseline results.

# ==========================================================
# Configuration
# ==========================================================

# Problem type: pass as first arg (forward|inverse); defaults to forward.
#   bash baseline_all_stories_cot_claude.sh            -> forward
#   bash baseline_all_stories_cot_claude.sh inverse    -> inverse
PROBLEM="${1:-forward}"
MODEL="claude-sonnet-4-5"
TEMPERATURE=0.7
MAX_TOKENS=8096
DATA_DIR="data"
PHASE2_DIR="output/phase2"
OUTPUT_DIR="output/baseline_cot"

# Story directory (based on problem type)
if [ "$PROBLEM" == "forward" ]; then
    STORIES_DIR="data/collectivistic-stories-all"
else
    STORIES_DIR="data/individualistic-rags-to-riches-stories"
fi

# ==========================================================
# Script
# ==========================================================

echo "========================================"
echo "CoT Baseline Transformation: All Stories"
echo "========================================"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Stories directory: $STORIES_DIR"
echo "========================================"
echo ""

if [ ! -d "$STORIES_DIR" ]; then
    echo "Error: Stories directory not found: $STORIES_DIR"
    exit 1
fi

TOTAL_STORIES=$(find "$STORIES_DIR" -name "*.txt" | wc -l | tr -d ' ')
echo "Found $TOTAL_STORIES stories"
echo ""

CURRENT=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

for story_file in "$STORIES_DIR"/*.txt; do
    CURRENT=$((CURRENT + 1))
    story_name=$(basename "$story_file" .txt)

    sanitized_model=$(echo "$MODEL" | sed 's/\//-/g' | sed 's/:/-/g')
    story_output_dir="$OUTPUT_DIR/$sanitized_model/$PROBLEM/$story_name"

    if [ -d "$story_output_dir" ] && [ -f "$story_output_dir/transformed_survey.json" ]; then
        echo "[$CURRENT/$TOTAL_STORIES] SKIPPING (already processed): $story_name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL_STORIES] Processing: $story_name"
    echo "========================================"

    python baseline_transform_cot.py \
        --story "$story_file" \
        --problem $PROBLEM \
        --model "$MODEL" \
        --temperature $TEMPERATURE \
        --max-tokens $MAX_TOKENS \
        --data-dir $DATA_DIR \
        --phase2-dir $PHASE2_DIR \
        --output-dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo "✓ Success: $story_name"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed: $story_name"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Model: $MODEL"
echo "Total stories: $TOTAL_STORIES"
echo "Successful: $SUCCESSFUL"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo "========================================"
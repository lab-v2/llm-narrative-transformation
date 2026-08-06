#!/bin/bash
# baseline_all_stories_fewshot_gpt52.sh
#
# Run the few-shot baseline transformation for all test stories for a single
# model. Skips stories that already have few-shot baseline results.

# ==========================================================
# Configuration
# ==========================================================

# Problem type: pass as first arg (forward|inverse); defaults to forward.
#   bash baseline_all_stories_fewshot_gpt52.sh            -> forward
#   bash baseline_all_stories_fewshot_gpt52.sh inverse    -> inverse
PROBLEM="${1:-forward}"
MODEL="gpt-5.2"
TEMPERATURE=0.7
MAX_TOKENS=8096
DATA_DIR="data"
EXAMPLES_DIR="data/fewshot_examples"
PHASE2_DIR="output/phase2"
OUTPUT_DIR="output/baseline_fewshot"

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
echo "Few-Shot Baseline Transformation: All Stories"
echo "========================================"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Examples dir: $EXAMPLES_DIR/$PROBLEM"
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

    # Skip stories not in this model's test set (no Phase-2 iteration_0 survey to copy).
    # This keeps the run to the actual test stories (e.g. 28), not every file in the dir.
    if [ ! -f "$PHASE2_DIR/$sanitized_model/$PROBLEM/$story_name/iteration_0/survey.json" ]; then
        echo "[$CURRENT/$TOTAL_STORIES] SKIPPING (no Phase-2 survey / not in test set): $story_name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "========================================"
    echo "[$CURRENT/$TOTAL_STORIES] Processing: $story_name"
    echo "========================================"

    python baseline_transform_fewshot.py \
        --story "$story_file" \
        --problem $PROBLEM \
        --model "$MODEL" \
        --temperature $TEMPERATURE \
        --max-tokens $MAX_TOKENS \
        --data-dir $DATA_DIR \
        --examples-dir $EXAMPLES_DIR \
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
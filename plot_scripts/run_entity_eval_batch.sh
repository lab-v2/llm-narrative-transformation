#!/usr/bin/env bash
set -euo pipefail

BASE="/home/dbavikad/leibniz/llm-narrative-transformation/output/evaluation_data/evaluation_data"
GRID_BASE="/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_grids/evaluation_data"
TRANS_BASE="/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_transitions/evaluation_data"

# Ensure spaCy is available (connect env)
if [ -f "/home/dbavikad/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/dbavikad/miniconda3/etc/profile.d/conda.sh"
  conda activate connect
fi

# MODELS=(bedrock-us.deepseek.r1-v1-0 bedrock-us.meta.llama4-maverick-17b-instruct-v1-0 claude-sonnet-4-5 gpt-4o gpt-5.2 xai-grok-4-fast-reasoning)

unset MODELS
unset GROUPS
MODELS=(claude-sonnet-4-5)
GROUPS=(collectivistic individualistic)

TMP_DIR="/tmp/entity_eval_$$"
mkdir -p "$TMP_DIR"
ABDUCTION_SIMS="$TMP_DIR/abduction_sims.txt"
BASELINE_SIMS="$TMP_DIR/baseline_sims.txt"
ERRORS="$TMP_DIR/errors.log"
DEBUG="${DEBUG:-0}"
> "$ABDUCTION_SIMS"
> "$BASELINE_SIMS"
> "$ERRORS"

for model in "${MODELS[@]}"; do
  for group in "${GROUPS[@]}"; do
    group_dir="$BASE/$model/$group"
    if [ ! -d "$group_dir" ]; then
      continue
    fi
    for story_dir in "$group_dir"/*; do
      [ -d "$story_dir" ] || continue
      original="$story_dir/original_story.txt"
      abduced="$story_dir/abduction_transformed.txt"
      baseline="$story_dir/baseline_transformed.txt"
      if [ ! -f "$original" ] || [ ! -f "$abduced" ] || [ ! -f "$baseline" ]; then
        continue
      fi

      story="$(basename "$story_dir")"
      out_grid_abd="$GRID_BASE/$model/$group/$story/abduction"
      out_grid_base="$GRID_BASE/$model/$group/$story/baseline"
      out_trans_abd="$TRANS_BASE/$model/$group/$story/abduction"
      out_trans_base="$TRANS_BASE/$model/$group/$story/baseline"

      if [ "$DEBUG" = "1" ]; then
        echo "processing $model $group $story" >> "$ERRORS"
      fi

      if [ -f "$out_trans_abd/transition_similarity.txt" ] && [ -f "$out_trans_base/transition_similarity.txt" ]; then
        if [ "$DEBUG" = "1" ]; then
          echo "skip_existing $model $group $story" >> "$ERRORS"
        fi
      else
        echo "processing $model $group $story"

        python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_grid.py \
          --original "$original" \
          --transformed "$abduced" \
          --shared-entities \
          --output-dir "$out_grid_abd" \
          >> "$ERRORS" 2>&1 || {
            echo "grid_abduction_failed $model $group $story" >> "$ERRORS"
            continue
          }

        python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_grid.py \
          --original "$original" \
          --transformed "$baseline" \
          --shared-entities \
          --output-dir "$out_grid_base" \
          >> "$ERRORS" 2>&1 || {
            echo "grid_baseline_failed $model $group $story" >> "$ERRORS"
            continue
          }

        python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_transition_similarity.py \
          --original "$original" \
          --transformed "$abduced" \
          --shared-entities \
          --output-dir "$out_trans_abd" \
          >> "$ERRORS" 2>&1 || {
            echo "transition_abduction_failed $model $group $story" >> "$ERRORS"
            continue
          }

        python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_transition_similarity.py \
          --original "$original" \
          --transformed "$baseline" \
          --shared-entities \
          --output-dir "$out_trans_base" \
          >> "$ERRORS" 2>&1 || {
            echo "transition_baseline_failed $model $group $story" >> "$ERRORS"
            continue
          }
      fi

      abd_sim=$(awk -F= '/cosine_similarity/ {print $2}' "$out_trans_abd/transition_similarity.txt")
      base_sim=$(awk -F= '/cosine_similarity/ {print $2}' "$out_trans_base/transition_similarity.txt")

      if [ -n "${abd_sim:-}" ]; then
        echo -e "${model}\t${group}\tabduction\t${abd_sim}" >> "$ABDUCTION_SIMS"
      else
        echo "missing_abduction_sim $model $group $story $out_trans_abd" >> "$ERRORS"
      fi
      if [ -n "${base_sim:-}" ]; then
        echo -e "${model}\t${group}\tbaseline\t${base_sim}" >> "$BASELINE_SIMS"
      else
        echo "missing_baseline_sim $model $group $story $out_trans_base" >> "$ERRORS"
      fi
    done
  done
done

# echo -e "Model\tGroup\tKind\tAvgCosine\tN"
# cat "$ABDUCTION_SIMS" "$BASELINE_SIMS" | awk '
# {
#   key=$1"\t"$2"\t"$3
#   sum[key]+=$4
#   cnt[key]+=1
# }
# END {
#   for (k in sum) {
#     printf "%s\t%.6f\t%d\n", k, (sum[k]/cnt[k]), cnt[k]
#   }
# }
# ' | sort

if [ -s "$ERRORS" ]; then
  echo ""
  echo "Errors (see full log): $ERRORS"
  tail -n 20 "$ERRORS"
fi

if [ "${KEEP_LOG:-0}" = "1" ]; then
  LOG_OUT="/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_transitions/evaluation_data/run_entity_eval_batch_errors.log"
  cp "$ERRORS" "$LOG_OUT"
  echo ""
  echo "Saved full error log to: $LOG_OUT"
else
  rm -rf "$TMP_DIR"
fi

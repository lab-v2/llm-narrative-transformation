#!/usr/bin/env bash
set -euo pipefail

BASE="/home/dbavikad/leibniz/llm-narrative-transformation/output/phase2/bedrock-us.meta.llama3-1-8b-instruct-v1-0/inverse/gpt-4o/final"
GRID_OUT="/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_grids/batch_gpt4o"
TRANS_OUT="/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_transitions/batch_gpt4o"

sum=0
count=0

for d in "$BASE"/*; do
  story="$d/story.txt"
  transformed="$d/story_transformed.txt"
  if [ -f "$story" ] && [ -f "$transformed" ]; then
    name="$(basename "$d")"

    python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_grid.py \
      --original "$story" \
      --transformed "$transformed" \
      --shared-entities \
      --output-dir "$GRID_OUT/$name"

    python /home/dbavikad/leibniz/llm-narrative-transformation/plot_scripts/create_entity_transition_similarity.py \
      --original "$story" \
      --transformed "$transformed" \
      --shared-entities \
      --output-dir "$TRANS_OUT/$name"

    sim=$(awk -F= '/cosine_similarity/ {print $2}' "$TRANS_OUT/$name/transition_similarity.txt")
    if [ -n "$sim" ]; then
      sum=$(python - <<PY
s=float("$sum"); v=float("$sim"); print(s+v)
PY
)
      count=$((count+1))
    fi
  fi
done

python - <<PY
s=float("$sum"); c=int("$count")
print(f"average_cosine_similarity={s/c if c else 0:.6f} (n={c})")
PY

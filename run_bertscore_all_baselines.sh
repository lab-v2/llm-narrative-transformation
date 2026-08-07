#!/bin/bash
# Compute BERTScore CSVs for all 4 baselines x 6 models x 2 directions.
# Skips any CSV already present (resumable).
MODELS=(gpt-4o gpt-5.2 xai-grok-4-fast-reasoning claude-sonnet-4-5 bedrock-us.meta.llama4-maverick-17b-instruct-v1-0 bedrock-us.deepseek.r1-v1-0)
for tech in zeroshot cot fewshot iterative; do
  for m in "${MODELS[@]}"; do
    for pair in "collectivistic:forward" "individualistic:inverse"; do
      dir="${pair%%:*}"; prob="${pair##*:}"
      out="output_analysis/bert_scores_csv_$tech/bertscore_results_${m}_${prob}.csv"
      [ -f "$out" ] && { echo "SKIP $out"; continue; }
      echo ">>> $tech $m $prob"
      python plot_scripts/bertscore_analysis.py "$dir" "$m" \
        --data-root "evaluation_data_$tech" --output-csv "$out"
    done
  done
done
echo "ALL BERTSCORE DONE"

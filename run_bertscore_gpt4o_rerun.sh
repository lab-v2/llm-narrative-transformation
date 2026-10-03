#!/bin/bash
declare -A SRC=( [zeroshot]=output/baseline_zeroshot [cot]=output/baseline_cot [fewshot]=output/baseline_fewshot [iterative]=output/baseline_iterative )
for tech in zeroshot cot fewshot iterative; do
  echo ">>> extract $tech"
  python extract_evaluation_stories.py --model_name gpt-4o \
    --baseline-dir "${SRC[$tech]}" --output-dir "evaluation_data_$tech" --force
  for pair in "collectivistic:forward" "individualistic:inverse"; do
    dir="${pair%%:*}"; prob="${pair##*:}"
    echo ">>> bertscore $tech gpt-4o $prob"
    python plot_scripts/bertscore_analysis.py "$dir" gpt-4o \
      --data-root "evaluation_data_$tech" \
      --output-csv "output_analysis/bert_scores_csv_$tech/bertscore_results_gpt-4o_${prob}.csv"
  done
done
echo "GPT4O BERTSCORE RERUN DONE"

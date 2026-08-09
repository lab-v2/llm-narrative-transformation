#!/bin/bash
set -e
declare -a JOBS=(
  "forward:data/collectivistic-stories-all/The_Christmas_Monks.txt"
  "forward:data/collectivistic-stories-all/The_Strangers_That_Came_to_Town_Ambrose_Flack.txt"
  "inverse:data/individualistic-rags-to-riches-stories/A_Lodging_for_the_Night.txt"
  "inverse:data/individualistic-rags-to-riches-stories/a_plain_case.txt"
)
for j in "${JOBS[@]}"; do
  P="${j%%:*}"; story="${j#*:}"; name=$(basename "$story" .txt)
  echo ">>> $P $name : deleting old dir + re-running zero-shot"
  rm -rf "output/baseline_zeroshot/gpt-5.2/$P/$name"
  python baseline_transform.py --story "$story" --problem "$P" --model gpt-5.2 \
    --temperature 0.7 --max-tokens 8096 --data-dir data \
    --phase2-dir output/phase2 --output-dir output/baseline_zeroshot
done
echo "GPT52 ZEROSHOT 4-STORY RERUN DONE"

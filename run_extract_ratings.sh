##!/bin/bash
#
## Run extract_ratings.py with hyperparameters 1, 2, 5, 10 and both forward/inverse problems
#
#HYPERPARAMETERS=(1 2 5 10)
#PROBLEMS=("forward" "inverse")
#
#for hp in "${HYPERPARAMETERS[@]}"; do
#    for problem in "${PROBLEMS[@]}"; do
#        echo "Running: hyperparameter-${hp}, problem: ${problem}"
#
#        python plot_scripts/extract_ratings.py \
#            --model gpt-4o \
#            --problem "$problem" \
#            --max-iterations 3 \
#            --base-dir "output/hyperparameter-${hp}/phase2" \
#            --output-dir "output_analysis/hyperparameter-${hp}/phase2"
#
#        echo "Completed: hyperparameter-${hp}, problem: ${problem}"
#        echo "---"
#    done
#done
#
#echo "All runs completed!"

#!/bin/bash

# Run find_optimal_iteration.py with hyperparameters 1, 2, 5, 10 and both forward/inverse problems

HYPERPARAMETERS=(1 2 5 10)
PROBLEMS=("forward" "inverse")

for hp in "${HYPERPARAMETERS[@]}"; do
    for problem in "${PROBLEMS[@]}"; do
        echo "Running: hyperparameter-${hp}, problem: ${problem}"

        python plot_scripts/find_optimal_iteration.py \
            --model gpt-4o \
            --problem "$problem" \
            --max-iterations 3 \
            --phase2-analysis-dir "output_analysis/hyperparameter-${hp}/phase2"

        echo "Completed: hyperparameter-${hp}, problem: ${problem}"
        echo "---"
    done
done

echo "All runs completed!"
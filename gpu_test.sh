#!/bin/bash

# Wrapper script for GPU Test on OrangeGrid

echo "=== GPU Test Job ==="
echo "Job ID: $CONDOR_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo

# Set up CUDA environment
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Change to the project directory
cd /home/dbavikad/leibniz/llm-narrative-transformation

set -euo pipefail

source /home/dbavikad/miniconda3/etc/profile.d/conda.sh
conda activate connect
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "CONDA_PREFIX=$CONDA_PREFIX"
# python fine_tune.py --use-lora --data-csv data/fine_tuning_data/llm4_recipe_dataset_sub.jsonl --output-dir output/local_llama4_maverick_lora_llm4 --base-model meta-llama/Llama-3.1-8B-Instruct
# python train_llm2.py
python fine_tune.py --use-lora --data-csv data/fine_tuning_data/llm3_llama4.jsonl --output-dir output/local_llama3_lora_llm3 --base-model meta-llama/Llama-3.1-8B-Instruct
# python test_gpu.py
echo
echo "Job completed at: $(date)"

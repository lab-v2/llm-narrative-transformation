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

source /home/dbavikad/miniconda3/etc/profile.d/conda.sh
conda activate connect
python test_gpu.py

echo
echo "Job completed at: $(date)"
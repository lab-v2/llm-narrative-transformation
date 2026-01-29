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

# Run from the execute scratch dir if provided
if [ -n "$_CONDOR_SCRATCH_DIR" ] && [ -d "$_CONDOR_SCRATCH_DIR" ]; then
  cd "$_CONDOR_SCRATCH_DIR" || exit 1
fi

python3 test_gpu.py

echo
echo "Job completed at: $(date)"

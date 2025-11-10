#!/bin/bash

# --- Environment Setup Script ---

# 1. Load the 'conda' environment module
echo "Loading Conda module..."
ml conda

# Check if the 'conda' module loaded successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to load the 'conda' module. Exiting."
    exit 1
fi

# 2. Activate the 'psf' Conda environment
echo "Activating 'py312' Conda environment..."
conda activate py312

# Check if the Conda environment activation was successful
# (Conda activation sometimes exits with 0 even on failure, but this is standard practice)
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the 'psf' environment. Check if it exists. Exiting."
    exit 1
fi

# 3. Load the necessary system modules (GCC, CUDA, PyTorch)
echo "Loading system module pytorch..."
ml pytorch
ml gcc

# Check if the modules loaded successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to load one or more system modules. Exiting."
    # Attempt to deactivate the Conda environment before exiting gracefully
    conda deactivate
    exit 1
fi

echo "Environment successfully set up."
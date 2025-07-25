#!/bin/bash
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: run_with_proper_env.sh.
Description:
    Wrapper script that ensures proper conda environment activation before
    running the nuclei segmentation pipeline. This fixes the environment
    issues causing segmentation timeouts and failures.

Dependencies:
    • Conda/Miniconda installation.
    • iri310_cellpose3 environment.

Usage:
    wsl ./run_with_proper_env.sh

Arguments:
    None (automatically runs the segmentation pipeline).

Key Features:
    • Automatic conda environment activation.
    • Environment validation and error checking.
    • Proper PyTorch/Cellpose3 setup verification.
    • Detailed logging of environment status.

Notes:
    • This script fixes the main cause of segmentation failures.
    • Run this instead of the regular pipeline script.
    • Make sure the script is executable: chmod +x run_with_proper_env.sh.
"""

set -e  # Exit on any error.

echo "============================================================"
echo "I/R INJURY NUCLEI SEGMENTATION - ENVIRONMENT WRAPPER"
echo "============================================================"

# Source conda initialization.
echo "Initializing conda..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the specific environment.
echo "Activating iri310_cellpose3 environment..."
conda activate iri310_cellpose3

# Verify environment activation.
if [[ "$CONDA_DEFAULT_ENV" != "iri310_cellpose3" ]]; then
    echo "ERROR: Failed to activate iri310_cellpose3 environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "✓ Successfully activated environment: $CONDA_DEFAULT_ENV"

# Verify Python and packages.
echo "Verifying Python installation..."
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

echo "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Verifying Cellpose..."
python -c "from cellpose import models; print('Cellpose imported successfully')"

echo "============================================================"
echo "ENVIRONMENT VERIFICATION COMPLETE - STARTING PIPELINE"
echo "============================================================"

# Change to the correct directory.
cd /mnt/c/Projects/I-R-Injury-Spatial-Multiomics-Analysis

# Run the segmentation pipeline.
python code/nuclei_segmentation/run_this.py

#!/bin/bash
#
# Author: Christos Botos
# Affiliation: Leiden University Medical Center
# Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos
#
# Script Name: run_with_proper_env.sh
# Description:
#     Environment wrapper for I/R injury nuclei segmentation pipeline.
#     Ensures proper conda environment activation and validates all dependencies
#     before running the segmentation pipeline. This script fixes the most common
#     cause of segmentation failures: incorrect environment setup.
#
# Dependencies:
#     • Conda/Miniconda installation (in ~/miniconda3 or system-wide)
#     • venv310_cellpose3 environment (created from cellpose3_environment.yml)
#     • PyTorch and Cellpose3 packages
#
# Usage:
#     Local machine:     ./run_with_proper_env.sh
#     WSL/Windows:       wsl ./run_with_proper_env.sh
#     Server/HPC:        bash run_with_proper_env.sh
#
# Arguments:
#     None (automatically runs the nuclei segmentation pipeline)
#
# Key Features:
#     • Automatic conda environment detection and activation
#     • Comprehensive environment validation and error checking
#     • PyTorch/Cellpose3 compatibility verification
#     • Detailed logging of environment status and issues
#     • Support for multiple conda installation locations
#     • Server and HPC cluster compatibility
#
# Troubleshooting:
#     • If "conda not found": Install miniconda in ~/miniconda3
#     • If "environment not found": Run mamba env create -f cellpose3_environment.yml
#     • If "permission denied": Run chmod +x run_with_proper_env.sh
#     • If still failing: Check the troubleshooting section in README.md
#

set -e  # Exit on any error.

echo "============================================================"
echo "I/R INJURY NUCLEI SEGMENTATION - ENVIRONMENT WRAPPER"
echo "============================================================"

# Function to find and source conda initialization.
find_and_init_conda() {
    local conda_paths=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
    )

    echo "Searching for conda installation..."
    for conda_path in "${conda_paths[@]}"; do
        if [[ -f "$conda_path" ]]; then
            echo "✓ Found conda at: $conda_path"
            source "$conda_path"
            return 0
        fi
    done

    echo "❌ ERROR: Conda not found in standard locations"
    echo "Searched paths:"
    for path in "${conda_paths[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo "SOLUTION: Install miniconda in your home directory:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
    echo "  source ~/miniconda3/etc/profile.d/conda.sh"
    echo "  conda init bash"
    exit 1
}

# Initialize conda.
find_and_init_conda

# Check if conda command is available.
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda command not found after initialization"
    echo "Try restarting your terminal or running: source ~/.bashrc"
    exit 1
fi

echo "✓ Conda initialized successfully"

# Check if environment exists.
echo "Checking for venv310_cellpose3 environment..."
if ! conda env list | grep -q "venv310_cellpose3"; then
    echo "❌ ERROR: venv310_cellpose3 environment not found"
    echo "Available environments:"
    conda env list
    echo ""
    echo "SOLUTION: Create the environment:"
    echo "  mamba env create -f cellpose3_environment.yml"
    echo "  # or if mamba not available:"
    echo "  conda env create -f cellpose3_environment.yml"
    exit 1
fi

echo "✓ Environment venv310_cellpose3 found"

# Activate the specific environment.
echo "Activating venv310_cellpose3 environment..."
conda activate venv310_cellpose3

# Verify environment activation.
if [[ "$CONDA_DEFAULT_ENV" != "venv310_cellpose3" ]]; then
    echo "❌ ERROR: Failed to activate venv310_cellpose3 environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo "This might be a shell configuration issue."
    echo ""
    echo "SOLUTION: Try manual activation:"
    echo "  conda activate venv310_cellpose3"
    echo "  python code/nuclei_segmentation/run_this.py"
    exit 1
fi

echo "✓ Successfully activated environment: $CONDA_DEFAULT_ENV"

# Verify Python and packages.
echo "Verifying Python installation..."
python_path=$(which python)
echo "Python executable: $python_path"

# Check if we're using the right Python.
if [[ "$python_path" != *"venv310_cellpose3"* ]]; then
    echo "⚠️  WARNING: Python path doesn't contain venv310_cellpose3"
    echo "This might indicate environment activation issues"
fi

echo "Python version: $(python --version)"

# Verify critical packages.
echo "Verifying critical packages..."

# Test PyTorch.
echo "Testing PyTorch..."
if python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "✓ PyTorch working correctly"
else
    echo "❌ ERROR: PyTorch not available or broken"
    echo "SOLUTION: Recreate environment:"
    echo "  conda env remove -n venv310_cellpose3"
    echo "  mamba env create -f cellpose3_environment.yml"
    exit 1
fi

# Test Cellpose.
echo "Testing Cellpose..."
if python -c "from cellpose import models; print('✓ Cellpose imported successfully')" 2>/dev/null; then
    echo "✓ Cellpose working correctly"
else
    echo "❌ ERROR: Cellpose not available or broken"
    echo "SOLUTION: Install Cellpose:"
    echo "  pip install cellpose==3.0.10"
    exit 1
fi

# Test other critical packages.
echo "Testing other packages..."
critical_packages=("numpy" "scipy" "matplotlib" "PIL" "skimage")
for package in "${critical_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package available"
    else
        echo "⚠️  WARNING: $package not available (might cause issues)"
    fi
done

echo "============================================================"
echo "ENVIRONMENT VERIFICATION COMPLETE - STARTING PIPELINE"
echo "============================================================"

# Change to the correct directory.
cd /mnt/c/Projects/Nuclei-Segmentation-with-Cellpose

# Run the segmentation pipeline.
python code/nuclei_segmentation/run_this.py

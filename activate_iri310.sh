#!/bin/bash
#
# Activation script for the iri310 conda environment.
# This script activates the environment and provides helpful information.
#

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate iri310

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" == "iri310" ]]; then
    echo "✅ Successfully activated iri310 environment!"
    echo ""
    echo "Environment details:"
    echo "- Python: $(python --version)"
    echo "- Conda environment: $CONDA_DEFAULT_ENV"
    echo "- Environment path: $CONDA_PREFIX"
    echo ""
    echo "Available packages include:"
    echo "- PyTorch 2.2.0 with CUDA 12.1 support"
    echo "- Cellpose 4.0.6 for cell segmentation"
    echo "- Scanpy for single-cell analysis"
    echo "- NumPy, SciPy, Pandas for data processing"
    echo "- scikit-image for image processing"
    echo "- Transformers for NLP tasks"
    echo ""
    echo "To test the environment, run:"
    echo "  python test_environment.py"
    echo ""
    echo "To deactivate the environment, run:"
    echo "  conda deactivate"
else
    echo "❌ Failed to activate iri310 environment!"
    echo "Make sure the environment exists. To create it, run:"
    echo "  mamba env create -f environment.yml"
    exit 1
fi

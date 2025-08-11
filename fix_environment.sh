#!/bin/bash
"""
COMPREHENSIVE ENVIRONMENT SETUP FOR NUCLEI SEGMENTATION PIPELINE

Author: Christos Botos
Description: 
    Creates a robust conda environment with all required dependencies for the
    nuclei segmentation pipeline. Uses a combination of conda and pip packages
    to ensure maximum compatibility while minimizing conflicts.

Dependencies:
    • Conda/Miniconda installed and accessible
    • CUDA 11.8 compatible GPU (optional but recommended)
    • WSL or Linux environment

Usage:
    bash fix_environment.sh

Key Features:
    • Removes existing broken environments
    • Creates new environment with fixed package versions
    • Tests all critical imports
    • Provides detailed error reporting and solutions
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🧬 NUCLEI SEGMENTATION ENVIRONMENT SETUP 🧬${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Function to print colored messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

print_info "Conda found: $(conda --version)"

# Remove existing environment if it exists
ENV_NAME="venv310_cellpose3_fixed"
print_info "Checking for existing environment: $ENV_NAME"

if conda env list | grep -q "$ENV_NAME"; then
    print_warning "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
    print_status "Environment removed successfully"
fi

# Create new environment with Python 3.10
print_info "Creating new environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

# Activate the environment
print_info "Activating environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    print_error "Failed to activate environment"
    exit 1
fi

print_status "Environment activated successfully"

# Install core scientific packages via conda
print_info "Installing core scientific packages via conda..."
conda install -c conda-forge -y \
    numpy=1.26.4 \
    scipy=1.11.4 \
    pandas=2.1.4 \
    matplotlib=3.8.2 \
    seaborn=0.13.0 \
    pillow=10.1.0 \
    scikit-learn=1.3.2 \
    scikit-image=0.22.0 \
    joblib=1.3.2 \
    tqdm=4.66.1 \
    psutil=5.9.6 \
    imageio=2.31.6 \
    networkx=3.2.1 \
    h5py=3.10.0

print_status "Core packages installed"

# Install PyTorch with CUDA support
print_info "Installing PyTorch with CUDA 11.8 support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

print_status "PyTorch installed"

# Install additional conda packages
print_info "Installing additional conda packages..."
conda install -c conda-forge -y \
    opencv=4.8.1 \
    rich=13.6.0 \
    click=8.1.7 \
    pytest=7.4.3 \
    packaging=23.2 \
    python-dateutil=2.8.2 \
    pytz=2023.3 \
    six=1.16.0 \
    typing_extensions=4.8.0 \
    filelock=3.13.1 \
    jinja2=3.1.2 \
    markupsafe=2.1.3

print_status "Additional conda packages installed"

# Install pip packages that are not available or better via pip
print_info "Installing specialized packages via pip..."
pip install --no-deps cellpose==3.0.10
pip install --no-deps fastremap==1.14.0
pip install --no-deps roifile==2023.8.12
pip install --no-deps typer==0.9.0
pip install --no-deps shellingham==1.5.4
pip install --no-deps imagecodecs==2023.9.18
pip install --no-deps opencv-python-headless==4.8.1.78

print_status "Specialized packages installed"

# Install bioinformatics packages
print_info "Installing bioinformatics packages..."
pip install anndata==0.10.3
pip install scanpy==1.9.6
pip install pyarrow==13.0.0
pip install fastparquet==2023.10.1

print_status "Bioinformatics packages installed"

# Test critical imports
print_info "Testing critical package imports..."

python -c "
import sys
import traceback

def test_import(module_name, description=''):
    try:
        __import__(module_name)
        print(f'✓ {module_name} - {description}')
        return True
    except ImportError as e:
        print(f'✗ {module_name} - FAILED: {e}')
        return False

print('Testing core packages...')
success = True
success &= test_import('numpy', 'Scientific computing')
success &= test_import('scipy', 'Scientific algorithms')
success &= test_import('pandas', 'Data manipulation')
success &= test_import('matplotlib', 'Plotting')
success &= test_import('PIL', 'Image processing')
success &= test_import('skimage', 'Image analysis')
success &= test_import('sklearn', 'Machine learning')
success &= test_import('cv2', 'Computer vision')
success &= test_import('imageio', 'Image I/O')
success &= test_import('rich', 'Rich console output')
success &= test_import('typer', 'CLI framework')
success &= test_import('tqdm', 'Progress bars')

print('\\nTesting deep learning packages...')
success &= test_import('torch', 'PyTorch')
success &= test_import('torchvision', 'PyTorch vision')

print('\\nTesting specialized packages...')
success &= test_import('cellpose', 'Cell segmentation')
success &= test_import('anndata', 'Annotated data')
success &= test_import('scanpy', 'Single-cell analysis')

if success:
    print('\\n🎉 All critical packages imported successfully!')
    sys.exit(0)
else:
    print('\\n❌ Some packages failed to import')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "All package imports successful"
else
    print_error "Some package imports failed"
    exit 1
fi

# Test the actual pipeline
print_info "Testing the nuclei segmentation pipeline..."
cd code/nuclei_segmentation

python -c "
try:
    from utils.logging_utils import setup_logging
    from utils.project_setup import load_config
    from utils.debug_utils import setup_debug
    from pipeline import run_segmentation_pipeline
    print('✓ All pipeline modules imported successfully')
except Exception as e:
    print(f'✗ Pipeline import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Pipeline modules imported successfully"
    
    echo ""
    echo -e "${GREEN}🎉 ENVIRONMENT SETUP COMPLETE! 🎉${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    echo -e "${CYAN}Environment Details:${NC}"
    echo -e "  • Name: $ENV_NAME"
    echo -e "  • Python: $(python --version)"
    echo -e "  • Location: $CONDA_PREFIX"
    echo ""
    echo -e "${CYAN}To use this environment:${NC}"
    echo -e "  1. conda activate $ENV_NAME"
    echo -e "  2. cd code/nuclei_segmentation"
    echo -e "  3. python run_this.py"
    echo ""
    echo -e "${CYAN}Key Packages Installed:${NC}"
    echo -e "  • PyTorch $(python -c 'import torch; print(torch.__version__)')"
    echo -e "  • Cellpose $(python -c 'import cellpose; print(cellpose.__version__)')"
    echo -e "  • NumPy $(python -c 'import numpy; print(numpy.__version__)')"
    echo -e "  • scikit-image $(python -c 'import skimage; print(skimage.__version__)')"
    echo ""
    
else
    print_error "Pipeline import test failed"
    exit 1
fi

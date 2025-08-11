#!/bin/bash

# Environment Setup Script for Nuclei Segmentation with Cellpose.
#
# Author: Christos Botos.
# Affiliation: Leiden University Medical Center
# Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.
#
# Description:
#     Creates a properly configured conda environment for nuclei segmentation pipeline.
#     Handles both Linux and WSL2 environments with fixed package versions.
#
# Usage:
#     bash setup_cellpose_env.sh
#
# Key Features:
#     • Fixed package versions for reproducibility.
#     • GPU support with CUDA 11.8.
#     • Comprehensive dependency management.
#     • Import testing for validation.

set -e  # Exit on any error

# Color codes for output.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

# Environment name with timestamp to avoid conflicts.
ENV_NAME="cellpose_env_$(date +%Y%m%d_%H%M%S)"

print_header "Setting up Cellpose Environment: $ENV_NAME"

# Remove any existing environment with similar name (if accessible).
print_info "Checking for existing environments..."
if conda env list | grep -q "cellpose_env"; then
    print_warning "Found existing cellpose environments, creating new one with unique name"
fi

# Create base environment with Python and core packages.
print_info "Creating conda environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

print_status "Base environment created"

# Activate environment.
print_info "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

print_status "Environment activated"

# Install PyTorch with CUDA support first.
print_info "Installing PyTorch with CUDA 11.8 support..."
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

print_status "PyTorch with CUDA installed"

# Install core scientific packages.
print_info "Installing core scientific packages..."
conda install -c conda-forge \
    numpy=1.26.4 \
    scipy=1.11.4 \
    pandas=2.1.4 \
    matplotlib=3.8.2 \
    seaborn=0.13.0 \
    pillow=10.1.0 \
    scikit-learn=1.3.2 \
    scikit-image=0.22.0 \
    joblib=1.3.2 \
    -y

print_status "Core scientific packages installed"

# Install image processing packages.
print_info "Installing image processing packages..."
conda install -c conda-forge \
    imageio=2.31.6 \
    tifffile=2023.9.26 \
    opencv=4.8.1 \
    h5py=3.10.0 \
    -y

print_status "Image processing packages installed"

# Install utility packages.
print_info "Installing utility packages..."
conda install -c conda-forge \
    rich=13.6.0 \
    click=8.1.7 \
    tqdm=4.66.1 \
    colorama=0.4.6 \
    pytest=7.4.3 \
    psutil=5.9.6 \
    packaging=23.2 \
    networkx=3.2.1 \
    -y

print_status "Utility packages installed"

# Install data handling packages.
print_info "Installing data handling packages..."
conda install -c conda-forge \
    pyarrow=13.0.0 \
    fastparquet=2023.10.1 \
    numba=0.58.1 \
    -y

print_status "Data handling packages installed"

# Install pip packages that are not available or better via pip.
print_info "Installing specialized packages via pip..."
pip install --no-deps cellpose==3.0.10
pip install --no-deps fastremap==1.14.0
pip install --no-deps roifile==2023.8.12
pip install --no-deps typer==0.9.0
pip install --no-deps shellingham==1.5.4
pip install --no-deps imagecodecs==2023.9.18

print_status "Specialized packages installed"

# Install bioinformatics packages.
print_info "Installing bioinformatics packages..."
pip install anndata==0.10.3
pip install scanpy==1.9.6

print_status "Bioinformatics packages installed"

# Test critical imports.
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

print('\nTesting deep learning packages...')
success &= test_import('torch', 'PyTorch')
success &= test_import('torchvision', 'PyTorch vision')

print('\nTesting specialized packages...')
success &= test_import('cellpose', 'Cell segmentation')
success &= test_import('anndata', 'Annotated data')
success &= test_import('scanpy', 'Single-cell analysis')

if success:
    print('\n🎉 All critical packages imported successfully!')
    sys.exit(0)
else:
    print('\n❌ Some packages failed to import')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "All package imports successful"
else
    print_error "Some package imports failed"
    exit 1
fi

# Test PyTorch CUDA availability.
print_info "Testing PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - will use CPU')
"

print_status "PyTorch CUDA test completed"

# Final success message.
print_header "Environment Setup Complete"
print_status "Environment name: $ENV_NAME"
print_info "To activate this environment, run:"
echo -e "    ${CYAN}conda activate $ENV_NAME${NC}"
print_info "To test the pipeline, run:"
echo -e "    ${CYAN}cd code/nuclei_segmentation && python run_this.py${NC}"

print_status "Setup completed successfully!"

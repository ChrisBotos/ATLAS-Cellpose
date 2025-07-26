#!/bin/bash
#
# Author: Christos Botos
# Script: setup_server_environment.sh
# Description: Automated server environment setup with disk space management
#
# This script helps set up the I/R injury nuclei segmentation environment on
# servers with limited disk space and permissions. It includes disk space
# checking, cleanup, and fallback installation methods.
#

set -e  # Exit on any error

echo "============================================================"
echo "I/R INJURY NUCLEI SEGMENTATION - SERVER ENVIRONMENT SETUP"
echo "============================================================"

# Function to check disk space
check_disk_space() {
    echo "--- Checking Disk Space ---"
    
    # Check home directory space
    home_space=$(df -BG ~ | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "Available space in home directory: ${home_space}GB"
    
    # Check conda directory if it exists
    if [[ -d ~/miniconda3 ]]; then
        conda_size=$(du -sh ~/miniconda3 2>/dev/null | cut -f1)
        echo "Current conda installation size: $conda_size"
        
        # Check package cache
        if [[ -d ~/miniconda3/pkgs ]]; then
            cache_size=$(du -sh ~/miniconda3/pkgs 2>/dev/null | cut -f1)
            echo "Conda package cache size: $cache_size"
        fi
    fi
    
    # Recommend minimum space
    if [[ $home_space -lt 5 ]]; then
        echo "⚠️  WARNING: Less than 5GB available"
        echo "   Recommended: Clean up space or use minimal environment"
        return 1
    else
        echo "✅ Sufficient disk space available"
        return 0
    fi
}

# Function to clean conda cache
clean_conda_cache() {
    echo "--- Cleaning Conda Cache ---"
    
    if command -v conda &> /dev/null; then
        echo "Cleaning conda package cache..."
        conda clean --all -y
        echo "✅ Conda cache cleaned"
    else
        echo "Conda not found, skipping cache cleanup"
    fi
}

# Function to install miniconda
install_miniconda() {
    echo "--- Installing Miniconda ---"
    
    if [[ -d ~/miniconda3 ]]; then
        echo "Miniconda already installed at ~/miniconda3"
        return 0
    fi
    
    echo "Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    echo "Installing Miniconda in ~/miniconda3..."
    bash /tmp/miniconda.sh -b -p ~/miniconda3
    
    echo "Initializing conda..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    
    echo "✅ Miniconda installed successfully"
    rm /tmp/miniconda.sh
}

# Function to install mamba
install_mamba() {
    echo "--- Installing Mamba ---"
    
    source ~/miniconda3/etc/profile.d/conda.sh
    
    if command -v mamba &> /dev/null; then
        echo "Mamba already installed"
        return 0
    fi
    
    echo "Installing mamba for faster dependency resolution..."
    conda install -n base mamba -c conda-forge -y
    echo "✅ Mamba installed successfully"
}

# Function to create environment with fallback options
create_environment() {
    echo "--- Creating Environment ---"
    
    source ~/miniconda3/etc/profile.d/conda.sh
    
    # Try full environment first
    if [[ -f "cellpose3_environment.yml" ]]; then
        echo "Attempting to create full environment..."
        if mamba env create -f cellpose3_environment.yml; then
            echo "✅ Full environment created successfully"
            return 0
        else
            echo "❌ Full environment creation failed"
        fi
    fi
    
    # Try minimal environment
    if [[ -f "cellpose3_minimal_environment.yml" ]]; then
        echo "Attempting to create minimal environment..."
        if mamba env create -f cellpose3_minimal_environment.yml; then
            echo "✅ Minimal environment created successfully"
            echo "Environment name: iri310_cellpose3_minimal"
            return 0
        else
            echo "❌ Minimal environment creation failed"
        fi
    fi
    
    # Manual pip installation as last resort
    echo "Attempting manual pip installation..."
    conda create -n iri310_cellpose3_manual python=3.10 pip -y
    conda activate iri310_cellpose3_manual
    
    echo "Installing core packages via conda..."
    conda install numpy scipy matplotlib pillow tqdm psutil -c conda-forge -y
    
    echo "Installing PyTorch via pip (CPU-only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    echo "Installing Cellpose and other packages..."
    pip install cellpose==3.0.10 scikit-image opencv-python-headless imagecodecs pandas joblib pytest
    
    echo "✅ Manual environment created successfully"
    echo "Environment name: iri310_cellpose3_manual"
}

# Function to test environment
test_environment() {
    echo "--- Testing Environment ---"
    
    source ~/miniconda3/etc/profile.d/conda.sh
    
    # Find the created environment
    env_name=""
    if conda env list | grep -q "iri310_cellpose3_minimal"; then
        env_name="iri310_cellpose3_minimal"
    elif conda env list | grep -q "iri310_cellpose3_manual"; then
        env_name="iri310_cellpose3_manual"
    elif conda env list | grep -q "iri310_cellpose3"; then
        env_name="iri310_cellpose3"
    else
        echo "❌ No environment found"
        return 1
    fi
    
    echo "Testing environment: $env_name"
    conda activate $env_name
    
    # Test critical imports
    if python -c "import torch, cellpose, numpy, scipy, matplotlib; print('✅ All critical packages working')"; then
        echo "✅ Environment test passed"
        echo ""
        echo "Environment ready! To use:"
        echo "  conda activate $env_name"
        echo "  python code/nuclei_segmentation/run_this.py"
        return 0
    else
        echo "❌ Environment test failed"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting server environment setup..."
    echo "Current user: $(whoami)"
    echo "Current directory: $(pwd)"
    echo ""
    
    # Check if we have the environment files
    if [[ ! -f "cellpose3_environment.yml" && ! -f "cellpose3_minimal_environment.yml" ]]; then
        echo "❌ Environment YAML files not found"
        echo "Please ensure you're in the project directory with the environment files"
        exit 1
    fi
    
    # Step 1: Check disk space
    if ! check_disk_space; then
        echo ""
        echo "⚠️  Limited disk space detected. Recommendations:"
        echo "1. Clean up unnecessary files"
        echo "2. Use the minimal environment"
        echo "3. Continue anyway (may fail)"
        echo ""
        read -p "Continue with setup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled"
            exit 1
        fi
    fi
    
    # Step 2: Clean conda cache
    clean_conda_cache
    
    # Step 3: Install miniconda if needed
    install_miniconda
    
    # Step 4: Install mamba
    install_mamba
    
    # Step 5: Create environment
    create_environment
    
    # Step 6: Test environment
    test_environment
    
    echo ""
    echo "============================================================"
    echo "SETUP COMPLETED SUCCESSFULLY!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Activate the environment: conda activate [environment_name]"
    echo "3. Test the pipeline: python test_environment_setup.py"
    echo "4. Run the pipeline: ./run_with_proper_env.sh"
    echo ""
}

# Run main function
main "$@"

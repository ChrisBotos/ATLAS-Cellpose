# Cellpose 3.0 Environment Setup Guide

## Overview

This guide provides instructions for setting up a Python 3.10 environment with Cellpose 3.0 and all necessary dependencies for nuclei segmentation and spatial multiomics analysis.

## Environment Details

- **Environment Name**: `iri310_cellpose3`
- **Python Version**: 3.10.18
- **Cellpose Version**: 3.0.10
- **PyTorch Version**: 2.5.1+cu121 (CUDA 12.1 support)
- **CUDA Support**: Yes (if NVIDIA GPU available)

## Key Features

- ✅ Cellpose 3.0 with latest models
- ✅ CUDA-enabled PyTorch for GPU acceleration
- ✅ Complete scientific Python stack (NumPy, SciPy, Pandas, Matplotlib)
- ✅ Single-cell analysis tools (Scanpy, AnnData)
- ✅ Image processing libraries (scikit-image, OpenCV, ImageIO)
- ✅ Data format support (Parquet, HDF5, TIFF)
- ✅ Rich console output and progress bars
- ✅ Testing framework (pytest)

## Quick Setup

### Option 1: Automated Setup (Recommended)

**Windows Batch Script:**
```batch
setup_environment.bat
```

**PowerShell Script:**
```powershell
.\setup_environment.ps1
```

### Option 2: Manual Setup

1. **Create conda environment:**
```bash
conda create -n iri310_cellpose3 python=3.10 pip -y
```

2. **Activate environment:**
```bash
conda activate iri310_cellpose3
```

3. **Install packages:**
```bash
pip install -r requirements.txt
```

### Option 3: From YAML file

```bash
conda env create -f cellpose3_environment.yml
```

## Environment Verification

After setup, test the environment:

```python
import cellpose
import numpy
import torch
import pandas
import matplotlib
import scanpy

print(f"NumPy: {numpy.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Cellpose: {cellpose.__version__}")
print(f"Scanpy: {scanpy.__version__}")
```

Expected output:
```
NumPy: 1.26.4
PyTorch: 2.5.1+cu121
CUDA available: True
Cellpose: 3.0.10
Scanpy: 1.11.4
```

## Package Versions

### Core Scientific Computing
- numpy==1.26.4
- scipy==1.15.3
- pandas==2.3.1

### Visualization
- matplotlib==3.10.5
- seaborn==0.13.2

### Machine Learning
- scikit-learn==1.7.1
- scikit-image==0.25.2

### Deep Learning
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- torchaudio==2.5.1+cu121

### Image Processing
- cellpose==3.0.10
- opencv-python-headless==4.11.0.86
- imagecodecs==2025.3.30
- imageio==2.37.0
- pillow==11.3.0

### Single-cell Analysis
- anndata==0.11.4
- scanpy==1.11.4

### Data Formats
- pyarrow==21.0.0
- fastparquet==2024.11.0

### Utilities
- rich==13.9.4
- typer==0.16.0
- tqdm==4.67.1
- psutil==7.0.0

## GPU Support

The environment includes CUDA-enabled PyTorch (2.5.1+cu121) for GPU acceleration. To verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")
```

## Troubleshooting

### Common Issues

1. **Conda not found**: Install Anaconda or Miniconda
2. **CUDA not available**: Install NVIDIA drivers and CUDA toolkit
3. **Package conflicts**: Use the pinned versions in requirements_clean.txt
4. **Memory issues**: Ensure sufficient RAM (8GB+ recommended)

### Environment Recreation

If you need to recreate the environment:

```bash
conda env remove -n iri310_cellpose3
conda env create -f cellpose3_environment.yml
```

## Usage

Activate the environment before running any scripts:

```bash
conda activate iri310_cellpose3
python your_script.py
```

## Files Included

- `cellpose3_environment_pinned.yml` - Complete environment specification
- `requirements.txt` - Full pip freeze output
- `requirements_clean.txt` - Essential packages only
- `setup_environment.bat` - Windows batch setup script
- `setup_environment.ps1` - PowerShell setup script
- `ENVIRONMENT_SETUP.md` - This documentation

## Notes

- The environment uses Python 3.10.18 for compatibility with all packages
- PyTorch includes CUDA 12.1 support for GPU acceleration
- All package versions are pinned for reproducibility
- The environment is tested and verified to work correctly

## Support

For issues or questions, refer to:
- Cellpose documentation: https://cellpose.readthedocs.io/
- PyTorch documentation: https://pytorch.org/docs/
- Scanpy documentation: https://scanpy.readthedocs.io/

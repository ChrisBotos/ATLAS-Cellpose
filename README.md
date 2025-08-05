# Nuclei Segmentation Pipeline for Tissue Analysis

A computational pipeline for analyzing nuclear morphology and spatial organization in tissue sections, with specific optimization for ischemia-reperfusion (I/R) injury studies.

**Authors**: Christos Botos and Benedetta Manzato
**Affiliation**: Human Genetics Department, Leiden University Medical Center
**PI**: Ahmed Mahfouz

## Overview

This pipeline combines Cellpose3-based deep learning segmentation with advanced image processing to extract quantitative features from DAPI-stained nuclei. It addresses the computational challenges of processing large tissue sections while maintaining biological accuracy.

### Research Applications

The pipeline is designed for studying ischemia-reperfusion injury in kidney tissue, enabling quantitative analysis of:
- Nuclear morphology changes across time points
- Spatial organization of cellular responses
- Cell death pathway markers (apoptosis, pyroptosis, necroptosis, ferroptosis)
- Regenerative processes (Wnt signaling, cell migration, angiogenesis)

## Key Features

- **Cellpose3 Integration**: Optimized nuclear segmentation with adaptive diameter detection
- **Tiled Processing**: Memory-efficient handling of large tissue sections
- **Advanced Merging**: Four-step algorithm for resolving tile overlaps while preserving cross-boundary nuclei
- **CLAHE Parameter Testing**: Systematic contrast enhancement optimization with 63 parameter combinations
- **Quality Control**: Comprehensive visualization and validation tools
- **Performance Optimization**: GPU acceleration and intelligent memory management
- **Server Compatibility**: Designed for HPC clusters with limited permissions

## Table of Contents

- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scientific Applications](#scientific-applications)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 (installed automatically with environment)
- **CUDA**: ≥ 12.1 (optional, for GPU acceleration)
- **Memory**: ≥ 8 GB RAM (≥ 16 GB for large images)
- **Storage**: ≥ 5 GB free space for conda environment

### Quick Start

```bash
# 1. Create the environment
mamba env create -f cellpose3_environment.yml

# 2. Activate the environment
conda activate iri310_cellpose3

# 3. Test the installation
python test_environment_setup.py

# 4. Run the pipeline
./run_with_proper_env.sh
```

### Server Installation

For HPC clusters or servers with limited permissions:

```bash
# Use the automated setup script
bash setup_server_environment.sh
```

The automated setup script handles:
- Miniconda installation in user directory
- Environment creation with fallback options
- Dependency resolution and testing

### Manual Server Setup

If you prefer manual installation:

```bash
# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

# 2. Install mamba and create environment
conda install -n base mamba -c conda-forge
mamba env create -f cellpose3_environment.yml
conda activate iri310_cellpose3

# 3. Verify installation
python -c "import torch, cellpose; print('Environment ready')"
```

### Environment Activation

**Important**: The conda environment must be activated before running the pipeline.

**Recommended approach:**
```bash
./run_with_proper_env.sh  # Automatically activates environment
```

**Manual activation:**
```bash
conda activate iri310_cellpose3
python code/nuclei_segmentation/run_this.py
```

### Testing Installation

Validate your environment setup:

```bash
conda activate iri310_cellpose3
python test_environment_setup.py
```

Expected output: "🎉 ENVIRONMENT READY!"

## CLAHE Parameter Testing

The pipeline includes a specialized tool for optimizing CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters:

### Quick CLAHE Testing

```bash
conda activate iri310_cellpose3
python temp.py
```

This generates 63 different contrast enhancement combinations in the `temp_results/` directory:

- **Clip Limits**: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0
- **Grid Sizes**: 4×4, 8×8, 12×12, 16×16, 20×20, 24×24, 32×32

### Parameter Selection Guidelines

**For Nuclear Imaging:**
- **Conservative Enhancement**: clip_limit=2.0, grid=8×8
- **Balanced Enhancement**: clip_limit=3.0, grid=16×16
- **Aggressive Enhancement**: clip_limit=5.0, grid=4×4

**Grid Size Effects:**
- **Small grids (4×4, 8×8)**: Local enhancement, preserves fine details
- **Large grids (24×24, 32×32)**: Global enhancement, uniform contrast

### Output Files

Files are systematically named for easy comparison:
```
temp_results/clahe_clip2.0_grid8x8.tif
temp_results/clahe_clip3.0_grid16x16.tif
temp_results/clahe_clip5.0_grid4x4.tif
```

## Pipeline Architecture

The pipeline follows a modular workflow optimized for large tissue sections:

```
Input Image (DAPI) → Preprocessing → Tiling → Cellpose Segmentation → Merging → Quality Control → Feature Extraction
```

### Core Components

1. **Preprocessing**: CLAHE contrast enhancement, gamma correction, ROI cropping
2. **Tiling**: Adaptive tiling with overlap for memory-efficient processing
3. **Segmentation**: Cellpose3 with adaptive diameter detection
4. **Merging**: Four-step algorithm for resolving tile overlaps
5. **Quality Control**: Before/after visualizations and validation metrics
6. **Feature Extraction**: Morphological and spatial feature computation

### Tiled Processing

For large tissue sections, the pipeline automatically uses tiled processing:

- **Tile Size**: 512×512 pixels (configurable)
- **Overlap**: 20% between adjacent tiles
- **Memory Management**: Processes tiles in batches to avoid memory overflow
- **Merge Algorithm**: Four-step process to resolve overlapping segmentations

### Cellpose Integration

The pipeline uses Cellpose3 for nuclear segmentation with optimized parameters:

- **Model**: `nuclei` (pre-trained for nuclear morphology)
- **Diameter**: Auto-detection (adaptive to tissue regions)
- **Thresholds**: Optimized for DAPI-stained tissue sections
- **GPU Support**: Automatic GPU acceleration when available

## Configuration

The pipeline is configured through `configs/nuclei_segmentation_config.ini`:

### Key Parameters

```ini
[general]
image_path = kidney_section.tif     # Input DAPI image
output_dir = results_timestamp      # Output directory
enhance_contrast = True             # Apply CLAHE preprocessing
crop_image = False                  # Enable ROI cropping

[cellpose]
model_type = nuclei                 # Use nuclear model
gpu = False                         # Enable GPU acceleration
use_cellpose4 = False               # Use Cellpose3 (recommended)
diameter = None                     # Auto-detection
flow_threshold = 0.9                # Boundary detection sensitivity
cellprob_threshold = -12            # Cell detection sensitivity

[tiling]
use_tiling = True                   # Enable for large images
tile_side_length = 512              # Tile size (pixels)
tile_overlap = 0.2                  # 20% overlap between tiles
qc_overlays = True                  # Generate QC images
```

### Parameter Optimization

**For healthy tissue:**
- `cellprob_threshold = -9` (standard sensitivity)
- `flow_threshold = 0.9` (standard boundaries)

**For injured/inflamed tissue:**
- `cellprob_threshold = -12` (high sensitivity for dim nuclei)
- `flow_threshold = 0.8` (more sensitive boundaries)

## Usage

### Basic Usage

```bash
# Recommended: Use the wrapper script
./run_with_proper_env.sh

# Manual execution
conda activate iri310_cellpose3
python code/nuclei_segmentation/run_this.py
```

### Batch Processing

```python
import glob
from pathlib import Path
from code.nuclei_segmentation.utils.project_setup import load_config
from code.nuclei_segmentation.pipeline import run_segmentation_pipeline
from code.nuclei_segmentation.utils.logging_utils import setup_logging

# Process all TIFF files in directory
image_dir = Path("kidney_images/")
for image_path in image_dir.glob("*.tif"):
    settings, cellpose_params, project_dirs = load_config()
    logger = setup_logging(settings["output_dir"], settings.get("debug_mode", False))

    settings['image_path'] = str(image_path)
    settings['job_name'] = f"batch_{image_path.stem}"

    exit_code = run_segmentation_pipeline(settings, cellpose_params, project_dirs, logger)
```

### Output Files

The pipeline generates:

- **Segmentation masks**: `segmentation_masks.npy` (labeled nuclei)
- **Quality control images**: Before/after merge visualizations
- **Feature data**: Morphological and spatial features (CSV format)
- **Configuration snapshot**: Reproducible parameter settings
- **Log files**: Detailed processing information

### Feature Extraction

Optional nuclear feature extraction includes:

```bash
# Extract morphological features
python code/engineered_feature_extraction/extract_engineered_features.py

# Cluster analysis
python code/engineered_feature_extraction/cluster_engineered_features.py
```

Features include:
- **Morphological**: Area, perimeter, eccentricity, solidity
- **Intensity**: Mean, standard deviation, skewness, kurtosis
- **Spatial**: Nearest neighbor distances, local density
- **Texture**: GLCM and Haralick features (optional)

## Scientific Applications

### Ischemia-Reperfusion Injury Analysis

The pipeline enables quantitative analysis of nuclear changes in kidney I/R injury:

- **Time-course studies**: Compare nuclear morphology across recovery time points
- **Spatial analysis**: Map cellular responses within tissue architecture
- **Cell death pathways**: Quantify apoptosis, pyroptosis, necroptosis markers
- **Regeneration studies**: Track Wnt signaling, migration, angiogenesis

### Typical Workflow

1. **Image acquisition**: DAPI-stained tissue sections
2. **Preprocessing**: Contrast enhancement and ROI selection
3. **Segmentation**: Automated nuclear detection with Cellpose3
4. **Feature extraction**: Morphological and spatial measurements
5. **Analysis**: Statistical comparison across conditions/time points

## Performance

### System Requirements

**Recommended:**
- **RAM**: 16GB+ for large tissue sections
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **CPU**: 8+ cores for parallel processing
- **Storage**: SSD for faster I/O operations

**Minimum:**
- **RAM**: 8GB
- **CPU**: 4+ cores
- **Python**: 3.10+

### Performance Features

- **GPU Acceleration**: Automatic GPU acceleration when available (3-5x speedup)
- **Memory Management**: Intelligent batching prevents memory overflow
- **Parallel Processing**: Multi-core CPU utilization for feature extraction
- **Caching**: Optimized algorithms with intelligent caching for repeated operations

### Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage |
|-------------|----------------|--------------|
| Small (512×512, 50 nuclei) | 0.8s | 2GB |
| Medium (1024×1024, 200 nuclei) | 3.2s | 4GB |
| Large (2048×2048, 800 nuclei) | 11.4s | 8GB |

### GPU Acceleration

Install CuPy for GPU acceleration:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# Verify GPU acceleration
python -c "import cupy; print('GPU acceleration available')"
```

## Troubleshooting

### Common Issues

**Environment activation fails:**
```bash
# Ensure conda is properly initialized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate iri310_cellpose3
```

**CUDA compatibility problems:**
```bash
# Check CUDA version
nvidia-smi

# Update PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Memory issues during processing:**
- Reduce `tile_side_length` in configuration
- Set `gpu = False` to use CPU-only processing
- Increase system swap space

**Package import failures:**
```bash
# Clean and recreate environment
conda env remove -n iri310_cellpose3
mamba env create -f cellpose3_environment.yml
```

### Getting Help

For technical issues:
1. Check the log files in the `logs/` directory
2. Run `python test_environment_setup.py` to validate setup
3. Review configuration parameters in `configs/nuclei_segmentation_config.ini`

---

**Repository**: [github.com/ChrisBotos/Nuclei-Segmentation-with-Cellpose](https://github.com/ChrisBotos/Nuclei-Segmentation-with-Cellpose)
**Contact**: botoschristos@gmail.com

























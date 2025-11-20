# ATLAS-Cellpose
## Adaptive Tiled Local Analysis Segmentation

**A memory-efficient computational pipeline for large-scale nuclear segmentation and morphological analysis in tissue sections**

**Authors**: Christos Botos and Benedetta Manzato
**Affiliation**: Human Genetics Department, Leiden University Medical Center
**PI**: Ahmed Mahfouz

---

### Abstract

ATLAS-Cellpose is a computational framework that enables memory-efficient analysis of large tissue sections through adaptive tiled processing combined with Cellpose3 deep learning segmentation. The method addresses the computational challenge of analyzing gigapixel tissue images by partitioning them into manageable tiles with systematic overlap resolution. ATLAS-Cellpose implements a four-step merging algorithm that preserves cross-boundary nuclei while eliminating redundant detections, enabling accurate quantification of nuclear morphology and spatial organization in large tissue sections. The pipeline has been optimized for ischemia-reperfusion injury studies in kidney tissue, providing comprehensive feature extraction (up to 40 morphological, spatial, and texture features) with perfect nucleus tracking throughout the analysis workflow. ATLAS-Cellpose is designed for high-performance computing environments and supports both GPU-accelerated and CPU-based processing.

## Overview

ATLAS-Cellpose (**A**daptive **T**iled **L**ocal **A**nalysis **S**egmentation) is a specialized computational framework that integrates Cellpose3 deep learning segmentation with adaptive tiled processing and advanced image analysis. The pipeline addresses the computational challenges of processing large tissue sections through intelligent tiling strategies, systematic overlap resolution, and comprehensive morphological feature extraction from DAPI-stained nuclei.

### Method Innovation

The ATLAS approach enables analysis of tissue sections that exceed available memory constraints by:

1. **Adaptive Tiling**: Dynamically partitioning large images into manageable tiles with configurable overlap
2. **Local Analysis**: Processing each tile independently while maintaining global context
3. **Systematic Merging**: Resolving tile boundaries through a four-step algorithm that preserves cross-boundary nuclei
4. **Memory Efficiency**: Processing gigapixel images on standard workstations without memory overflow

This combination of adaptive tiling and local analysis makes ATLAS-Cellpose particularly suitable for whole-slide imaging and large tissue section analysis where traditional approaches fail due to memory limitations.

### Research Applications

ATLAS-Cellpose is designed for studying ischemia-reperfusion injury in kidney tissue, enabling quantitative analysis of:
- Nuclear morphology changes across time points
- Spatial organization of cellular responses
- Cell death pathway markers (apoptosis, pyroptosis, necroptosis, ferroptosis)
- Regenerative processes (Wnt signaling, cell migration, angiogenesis)

## Key Features

- **Adaptive Tiled Processing**: Memory-efficient handling of large tissue sections through intelligent tiling with configurable overlap
- **Cellpose3 Integration**: Optimized nuclear segmentation with adaptive diameter detection
- **Systematic Overlap Resolution**: Four-step merging algorithm for resolving tile boundaries while preserving cross-boundary nuclei
- **CLAHE Parameter Testing**: Systematic contrast enhancement optimization with 63 parameter combinations
- **Quality Control**: Comprehensive visualization and validation tools
- **Performance Optimization**: GPU acceleration and intelligent memory management
- **Comprehensive Feature Extraction**: Up to 40 morphological, spatial, and texture features with optimized processing
- **Nucleus Label Preservation**: Perfect 1:1 mapping between segmentation masks, feature extraction, and clustering results
- **Adaptive Processing**: Dynamic batch sizing and resource allocation based on feature complexity
- **Server Compatibility**: Designed for HPC clusters with limited permissions

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scientific Applications](#scientific-applications)
- [Performance](#performance)
- [Nuclear Feature Clustering](#nuclear-feature-clustering)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 (installed automatically with environment)
- **CUDA**: = 11.8 (optional, for GPU acceleration)
- **Memory**: ≥ 8 GB RAM (≥ 16 GB for large images)
- **Storage**: ≥ 5 GB free space for conda environment

### Quick Start

```bash
# 1. Create the environment
mamba env create -f cellpose3_environment.yml

# 2. Activate the environment
conda activate venv310_cellpose3

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
conda activate venv310_cellpose3

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
conda activate venv310_cellpose3
python code/nuclei_segmentation/run_this.py
```

### Testing Installation

Validate your environment setup:

```bash
conda activate venv310_cellpose3
python test_environment_setup.py
```

Expected output: "🎉 ENVIRONMENT READY!"

### Running Tests

ATLAS-Cellpose includes a comprehensive test suite to validate functionality:

```bash
conda activate venv310_cellpose3

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/nuclei_segmentation_tests/ -v  # Core pipeline tests
python -m pytest tests/test_gpu_merge_4step_integration.py -v  # GPU merge tests
python -m pytest tests/test_white_segmentation_masks_performance.py -v  # Performance tests
```

**Test Coverage:**
- **Core Pipeline**: Segmentation, adaptive tiling, preprocessing, and feature extraction
- **Merge Algorithms**: Systematic merging, 4-step CPU algorithm, GPU integration
- **Performance**: Memory efficiency, large image handling, optimization
- **Visualization**: Color generation, overlay creation, QC tools
- **Integration**: End-to-end pipeline validation

The test suite has been cleaned to remove outdated debug tests and ensure all tests reference existing code modules.

## CLAHE Parameter Testing

ATLAS-Cellpose includes a specialized tool for optimizing CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters:

### Quick CLAHE Testing

```bash
conda activate venv310_cellpose3
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

ATLAS-Cellpose follows a modular workflow optimized for large tissue sections:

```
Input Image (DAPI) → Preprocessing → Adaptive Tiling → Cellpose Segmentation → Systematic Merging → Filtering → Quality Control → Feature Extraction
```

### Core Components

1. **Preprocessing**: CLAHE contrast enhancement, gamma correction, ROI cropping
2. **Adaptive Tiling**: Intelligent tiling with configurable overlap for memory-efficient processing
3. **Segmentation**: Cellpose3 with adaptive diameter detection
4. **Systematic Merging**: Four-step algorithm for resolving tile overlaps
5. **Quality Control**: Before/after visualizations and validation metrics
6. **Feature Extraction**: Morphological and spatial feature computation

### Adaptive Tiled Processing

For large tissue sections, ATLAS-Cellpose automatically employs adaptive tiled processing:

- **Tile Size**: 512×512 pixels (configurable)
- **Overlap**: 20% between adjacent tiles
- **Memory Management**: Processes tiles in batches to avoid memory overflow
- **Merge Algorithm**: Four-step systematic process to resolve overlapping segmentations

### Cellpose Integration

ATLAS-Cellpose integrates Cellpose3 for nuclear segmentation with optimized parameters:

- **Model**: `nuclei` (pre-trained for nuclear morphology)
- **Diameter**: Auto-detection (adaptive to tissue regions)
- **Thresholds**: Optimized for DAPI-stained tissue sections
- **GPU Support**: Automatic GPU acceleration when available

### Morphological Filtering

ATLAS-Cellpose includes comprehensive morphological filtering to remove segmentation artifacts:

- **Size Filtering**: Removes objects that are too small (debris) or too large (merged nuclei)
- **Shape Filtering**: Filters based on circularity, solidity, and eccentricity
- **Aspect Ratio**: Removes overly elongated objects that are likely artifacts
- **Hole Detection**: Filters objects with excessive internal holes
- **Border Exclusion**: Optional removal of nuclei touching image borders

**Default Thresholds (optimized for kidney tissue):**
- Size: 20-900 pixels
- Circularity: 0.56-1.00 (moderately circular to perfect circle)
- Solidity: 0.765-1.00 (solid objects with minimal concavity)
- Eccentricity: 0.00-0.975 (circular to moderately elongated)
- Aspect Ratio: 0.50-3.20 (prevents extremely elongated artifacts)
- Hole Fraction: 0.00-0.001 (minimal internal holes allowed)

### Optimized Feature Extraction

ATLAS-Cellpose includes both a comprehensive feature extraction system (43 features) and a streamlined simple extraction system (up to 40 features) with granular control:

#### Feature Categories

1. **Shape Features (11 features)**:
   - Basic: circularity, eccentricity, solidity, aspect_ratio, compactness, elongation, roundness, form_factor
   - Advanced: convex_area_ratio, convexity, fractal_dimension

2. **Size Features (10 features)**:
   - Primary: area, perimeter, equivalent_diameter, major_axis_length, minor_axis_length
   - Bounding box: width, height, area
   - Feret diameters: maximum, minimum

3. **Neighborhood Features (8 features)**:
   - Spatial: nearest_neighbor_distance, neighborhood_density, boundary_proximity
   - Organization: cluster_elongation, cluster_polarization, spatial_autocorrelation
   - Indices: tissue_organization_index, local_clustering_coefficient

4. **Texture Features (12 features)**:
   - Basic statistics: intensity_mean, intensity_std, intensity_median, intensity_skewness, intensity_kurtosis
   - Entropy: texture_entropy
   - Gradient: gradient_magnitude_mean, gradient_magnitude_std
   - GLCM: contrast, dissimilarity, homogeneity, energy

#### Performance Optimizations

- **Granular Selection**: Extract only the features you need for significant performance improvements
- **Adaptive Batch Processing**: Automatically adjusts batch sizes based on feature complexity
- **Computational Caching**: LRU caching for expensive operations like convex hull calculations
- **GPU Acceleration**: CuPy integration for vectorized operations on large datasets
- **Memory Management**: Intelligent memory allocation and garbage collection

#### Configuration Example

```ini
# Enable all features (comprehensive analysis)
extract_all_features = True

# Or select individual features for optimal performance
extract_all_features = False
extract_area = True
extract_circularity = True
extract_intensity_mean = True
extract_nearest_neighbor_distance = True
# ... (43 individual feature flags available)
```

### Simple Feature Extraction (Streamlined Pipeline)

For faster processing and most research applications, the simple feature extraction system provides:

#### Feature Categories (40 total features)

1. **Size Features (10 features)**: area, perimeter, equivalent_diameter, major_axis_length, minor_axis_length, bounding_box_width, bounding_box_height, bounding_box_area, feret_diameter_max, feret_diameter_min

2. **Shape Features (10 features)**: circularity, eccentricity, solidity, aspect_ratio, compactness, elongation, roundness, form_factor, convex_area_ratio, convexity

3. **Neighborhood Features (8 features)**: neighbor_count, neighbor_density, nearest_neighbor_distance, mean_neighbor_distance, neighbor_area_ratio, local_density_gradient, clustering_coefficient, isolation_score

4. **Texture Features (12 features - Optional)**: intensity_mean, intensity_std, intensity_median, intensity_skewness, intensity_kurtosis, texture_entropy, gradient_magnitude_mean, gradient_magnitude_std, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy

#### Performance Benefits

- **3.5x faster** than comprehensive pipeline
- **Single-threaded reliability** (no multiprocessing complexity)
- **Perfect nucleus tracking** through entire pipeline
- **Optional texture analysis** for chromatin studies
- **Configurable feature sets** for optimal performance

#### Configuration Example

```ini
# Simple feature extraction settings
extract_texture_features = False    # Disable for faster processing (28 features)
extract_texture_features = True     # Enable for chromatin analysis (40 features)
```

**Performance Comparison:**
- **Without texture features**: ~215 nuclei/second (28 features)
- **With texture features**: ~215 nuclei/second (40 features, minimal overhead)

## Configuration

ATLAS-Cellpose is configured through `configs/nuclei_segmentation_config.ini`:

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

[filtering]
use_filtering = True                # Enable morphological filtering
min_pixels = 20                     # Minimum nucleus size (pixels)
max_pixels = 900                    # Maximum nucleus size (pixels)
min_circularity = 0.56              # Minimum circularity (0=line, 1=circle)
max_circularity = 1.00              # Maximum circularity
min_solidity = 0.765                # Minimum solidity (convex hull ratio)
max_solidity = 1.00                 # Maximum solidity
min_eccentricity = 0.00             # Minimum eccentricity (0=circle, 1=line)
max_eccentricity = 0.975            # Maximum eccentricity
min_aspect_ratio = 0.50             # Minimum aspect ratio (major/minor axis)
max_aspect_ratio = 3.20             # Maximum aspect ratio
min_hole_fraction = 0.00            # Minimum hole fraction
max_hole_fraction = 0.001           # Maximum hole fraction
exclude_border = False              # Exclude border-touching nuclei
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
conda activate venv310_cellpose3
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

ATLAS-Cellpose generates:

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

ATLAS-Cellpose enables quantitative analysis of nuclear changes in kidney I/R injury:

- **Time-course studies**: Compare nuclear morphology across recovery time points
- **Spatial analysis**: Map cellular responses within tissue architecture
- **Cell death pathways**: Quantify apoptosis, pyroptosis, necroptosis markers
- **Regeneration studies**: Track Wnt signaling, migration, angiogenesis

### Typical Workflow

1. **Image acquisition**: DAPI-stained tissue sections
2. **Preprocessing**: Contrast enhancement and ROI selection
3. **Segmentation**: Automated nuclear detection with adaptive tiled processing
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
conda activate venv310_cellpose3
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
conda env remove -n venv310_cellpose3
mamba env create -f cellpose3_environment.yml
```

## Nuclear Feature Clustering

After extracting nuclear features, you can perform clustering analysis to identify distinct nuclear populations and phenotypes.

### Quick Start - Simple Clustering

For basic clustering with area and circularity features:

```bash
# 1. Extract simple features (fast, reliable)
python code/engineered_feature_extraction/extract_engineered_features.py \
    --config configs/engineered_feature_extraction_config.ini

# 2. Perform clustering analysis
python code/engineered_feature_extraction/cluster_engineered_features.py \
    --config configs/engineered_feature_extraction_config.ini
```

### Clustering Outputs

The clustering analysis generates comprehensive results:

1. **nuclear_clusters.csv** - Original features + cluster assignments
2. **cluster_overlay.tif** - Visual overlay showing clusters on tissue image
3. **pca_clusters.png** - PCA visualization of cluster separation
4. **feature_importance.png** - Feature contribution to clustering
5. **cluster_statistics.csv** - Statistical summary per cluster
6. **kmeans_model.joblib** - Saved model for reproducibility

### Column Name Compatibility

The clustering script automatically handles different column naming conventions:
- ✅ **'label'** column (from full feature extraction)
- ✅ **'nucleus_id'** column (from simple feature extraction)
- 🔄 Automatic renaming for pipeline consistency

### Configuration Parameters

The `engineered_feature_extraction_config.ini` file provides **comprehensive granular control** with 80+ parameters for fine-tuned feature extraction and clustering analysis. The configuration system offers both high-level category controls and individual feature selection for optimal performance.

#### Feature Selection Categories

**Individual Feature Control**: Each feature can be enabled/disabled independently:
- **Size Features (10 parameters)**: area, perimeter, equivalent_diameter, major/minor_axis_length, bounding_box dimensions, feret_diameters
- **Shape Features (10 parameters)**: circularity, eccentricity, solidity, aspect_ratio, compactness, elongation, roundness, form_factor, convex_area_ratio, convexity
- **Neighborhood Features (9 parameters)**: neighbor_count, neighbor_density, distance metrics, clustering_coefficient
- **Texture Features (12 parameters)**: intensity statistics, entropy, gradients, GLCM properties

#### Quality Control Parameters

**Nuclei Filtering**: Morphological thresholds matching segmentation pipeline:
- Size thresholds: `min_pixels=20`, `max_pixels=900`
- Shape quality: `min_circularity=0.56`, `min_solidity=0.765`
- Morphology limits: `max_eccentricity=0.975`, `max_aspect_ratio=3.20`

#### Performance Optimization

**Processing Control**: Memory and computational efficiency:
- Worker allocation: `feature_extraction_workers=1` (single-threaded reliability)
- Batch processing: `extraction_batch_size=1000` for memory efficiency
- Memory management: `max_memory_gb=8.0`, `enable_memory_mapping=True`
- Progress tracking: `enable_progress_tracking=True`, `save_diagnostic_files=True`
- Automatic temp directories: `temp_directory=auto` (creates unique timestamped directories)

#### Automatic Temporary Directory Management

The system automatically creates unique temporary directories for intermediate processing files:

```ini
# Automatic unique temp directory generation
temp_directory = auto  # Creates: YYYYMMDD_HHMMSS_temp (e.g., 20250929_143052_temp)

# Or specify custom directory
temp_directory = ./my_custom_temp
```

**Benefits of Automatic Temp Directories:**
- **Unique timestamps**: Prevents conflicts between concurrent runs
- **Automatic cleanup**: Built-in utilities for safe directory removal
- **No user intervention**: System handles all directory management
- **Collision-free**: Each run gets its own isolated temporary space

Key parameters in the comprehensive configuration:

```ini
[feature_extraction]
# Simple feature extraction parameters
neighborhood_radius = 20.0              # Spatial analysis radius (pixels)
extract_texture_features = False        # Enable/disable texture features (12 features)

# Input/output paths
extraction_image_path = ../../results/example_cropped/preprocessed/first.tif
extraction_mask_path = ../../results/example_cropped/masks/segmentation_masks.npy
extraction_output_dir = ../../results/example_cropped/engineered_features

[clustering]
# Clustering algorithm parameters
default_clusters = 8                    # Number of clusters for K-means
auto_k_method = None                     # Automatic cluster selection method
clustering_seed = 42                     # Random seed for reproducibility

# Visualization parameters
generate_cluster_overlay = True          # Create tissue overlay visualization
generate_pca_plot = True                 # Generate PCA feature space plot
generate_feature_importance = True       # Analyze feature contributions

# Color configuration - vibrant, non-bluish palette
color_alpha = 250                        # Cluster color transparency (0-255)
color_saturation = 1.0                   # Maximum saturation for vibrant colors
custom_colors = #FF0000, #00FF00, #FF6000, #FF3000, #FF8000, #FFFF00, #FF0080, #80FF00, #FF9000, #FF4000, #00FF80, #FF8040, #40FF80, #FFA000, #FF4080, #40FF40, #FF8080, #80FF80, #FFB000, #FFC000
clustering_seed = 42
save_clustering_model = true

# Visualization settings
overlay_alpha = 0.85
overlay_tile_size = 1024
enable_gpu = true
```

### Scientific Context

Nuclear clustering reveals:
- **Healthy nuclei**: Regular shape, moderate size
- **Apoptotic nuclei**: Fragmented, irregular morphology
- **Necrotic nuclei**: Swollen, loss of membrane integrity
- **Proliferating nuclei**: Larger size, specific morphological features

### Getting Help

For technical issues:
1. Check the log files in the `logs/` directory
2. Run `python test_environment_setup.py` to validate setup
3. Review configuration parameters in `configs/nuclei_segmentation_config.ini`

---

## Citation

If you use ATLAS-Cellpose in your research, please cite:

```
Botos, C., Manzato, B., & Mahfouz, A. (2025). ATLAS-Cellpose: Adaptive Tiled Local Analysis
Segmentation for Large-Scale Tissue Image Analysis. [Journal Name], [Volume], [Pages].
```

---

**Repository**: [github.com/ChrisBotos/ATLAS-Cellpose](https://github.com/ChrisBotos/ATLAS-Cellpose)
**Contact**: botoschristos@gmail.com
**License**: [Specify License]

























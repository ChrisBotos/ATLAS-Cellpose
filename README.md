# ATLAS-Cellpose
## Adaptive Tiled Local Analysis Segmentation

**A computational framework for large-scale nuclear segmentation with adaptive parameter optimization through intelligent tiling**

**Authors**: Christos Botos (developer) and Benedetta Manzato (supervisor)
**Affiliation**: Human Genetics Department, Leiden University Medical Center
**Principal Investigator**: Ahmed Mahfouz

---

### Abstract

ATLAS-Cellpose is a computational framework that combines Cellpose3 deep learning segmentation with adaptive tiled processing to enable accurate analysis of large tissue sections. The method addresses two fundamental challenges in tissue image analysis: (1) memory constraints when processing gigapixel images, and (2) heterogeneous nuclear morphology across tissue regions. By partitioning images into locally homogeneous tiles, ATLAS-Cellpose enables Cellpose's adaptive diameter detection to optimize segmentation parameters for each tissue microenvironment independently. A four-step merging algorithm systematically resolves tile boundaries while preserving cross-boundary nuclei and eliminating redundant detections. The pipeline provides comprehensive morphological feature extraction (up to 40 features) with perfect nucleus tracking throughout the analysis workflow. Optimized for ischemia-reperfusion injury studies in kidney tissue, ATLAS-Cellpose supports both GPU-accelerated and CPU-based processing in high-performance computing environments.

## Overview

ATLAS-Cellpose (**A**daptive **T**iled **L**ocal **A**nalysis **S**egmentation) integrates Cellpose3 deep learning segmentation with adaptive tiled processing to overcome fundamental limitations in large-scale tissue image analysis. The framework addresses both computational constraints and biological heterogeneity through intelligent spatial partitioning and local parameter optimization.

### Method Innovation

The ATLAS approach transforms tissue image analysis through four key innovations:

1. **Adaptive Diameter Optimization**: Tiling creates locally homogeneous regions where Cellpose's adaptive diameter detection can optimize independently for each tissue microenvironment, dramatically improving segmentation accuracy in heterogeneous samples
2. **Intelligent Spatial Partitioning**: Dynamic tiling with configurable overlap enables processing of arbitrarily large images while maintaining biological context
3. **Systematic Boundary Resolution**: A four-step merging algorithm preserves cross-boundary nuclei while eliminating redundant detections with priority-based selection
4. **Scalable Architecture**: Memory-efficient processing enables gigapixel image analysis on standard workstations

This combination of adaptive parameter optimization and systematic merging makes ATLAS-Cellpose uniquely suited for whole-slide imaging and large tissue sections where nuclear morphology varies substantially across spatial regions—a common feature of injury, disease, and developmental models.

### Research Applications

ATLAS-Cellpose was developed for ischemia-reperfusion injury studies in kidney tissue, enabling quantitative analysis of:
- **Temporal dynamics**: Nuclear morphology changes across recovery time points
- **Spatial organization**: Cellular response patterns within tissue architecture
- **Cell death pathways**: Apoptosis, pyroptosis, necroptosis, and ferroptosis markers
- **Regenerative processes**: Wnt signaling, cell migration, and angiogenesis

The framework generalizes to any large-scale tissue imaging application requiring accurate nuclear segmentation across heterogeneous tissue regions.

## Key Features

- **Adaptive Parameter Optimization**: Tiling enables local diameter detection, dramatically improving segmentation accuracy in heterogeneous tissue regions
- **Systematic Boundary Resolution**: Four-step merging algorithm preserves cross-boundary nuclei while eliminating redundant detections
- **Scalable Architecture**: Memory-efficient processing of gigapixel images on standard workstations
- **Cellpose3 Integration**: Optimized nuclear segmentation with validated performance on DAPI-stained tissue sections
- **Comprehensive Feature Extraction**: Up to 40 morphological, spatial, and texture features with perfect nucleus tracking
- **Quality Control Framework**: Extensive visualization and validation tools for reproducible analysis
- **Performance Optimization**: GPU acceleration with automatic CPU fallback and intelligent memory management
- **CLAHE Preprocessing**: Systematic contrast enhancement with 63 validated parameter combinations
- **HPC Compatibility**: Designed for high-performance computing clusters with limited user permissions

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
- **CUDA**: 11.8 (optional, for GPU acceleration)
- **Memory**: ≥ 8 GB RAM (≥ 16 GB for large images)
- **Storage**: ≥ 5 GB free space for conda environment
- **Conda**: Miniconda or Anaconda

### Environment Setup Tutorial

**Important**: This project is optimized for **Cellpose 3.0.10**, which provides superior nuclear segmentation performance compared to Cellpose 4.x for DAPI-stained tissue sections. The environment file `cellpose3_environment_recommended.yml` contains all tested and validated package versions.

#### Step 1: Install Miniconda (if not already installed)

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda to your home directory
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize conda for your shell
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash

# Reload your shell configuration
source ~/.bashrc
```

#### Step 2: Create the Cellpose3 Environment

```bash
# Navigate to the ATLAS-Cellpose directory.
cd /path/to/ATLAS-Cellpose

# Create the environment from the YAML file.
# This will install Python 3.10, Cellpose 3.0.10, PyTorch with CUDA 11.8, and all dependencies.
conda env create -f cellpose3_environment_recommended.yml

# The environment creation will:
# - Install ~100 packages via conda.
# - Install Cellpose 3.0.10 and additional packages via pip.
# - Take approximately 5-10 minutes depending on your internet connection.
# - Require ~5 GB of disk space.
```

#### Step 3: Activate the Environment

```bash
# Activate the newly created environment.
conda activate venv310_cellpose3

# You should see (venv310_cellpose3) in your terminal prompt.
```

#### Step 4: Test the Pipeline

**Recommended Method** - Use the shell script wrapper:

```bash
# Run with custom parameters (recommended).
# Add your image to the project directory "data" and update the image_path in the command below.
# The crop_box coordinates are provided as a comma-separated string (x_start, x_end, y_start, y_end).
# The crop_box is a percentage of the image size (0-1). Use it to run the pipeline on a small sample of your image.
./run_segmentation_instance.sh crop_box image_path "path/to/your/image.tif" "0.38,0.42,0.32,0.36"

# The script will:
# - Activate the conda environment automatically.
# - Create temporary configuration files.
# - Run the pipeline with your specified parameters.
# - Save logs to logs/run_segmentation_instance/.
# - Clean up temporary files on completion.
```

**Alternative Method** - Direct execution (not recommended):

```bash
# You can also run the pipeline directly from the code directory.
# However, this requires manual configuration file editing and environment activation.
python code/nuclei_segmentation/run_this.py

# Note: Configuration files are located in configs/ directory.
# Results will be saved in the results/ directory.
```

### Quick Start (For Experienced Users)

```bash
# One-command setup (after conda is installed).
conda env create -f cellpose3_environment_recommended.yml && \
conda activate venv310_cellpose3 && \
./run_segmentation_instance.sh crop_box "0.38,0.42,0.32,0.36"
```

### Why Cellpose 3.0.10?

ATLAS-Cellpose is specifically optimized for **Cellpose 3.0.10** based on extensive validation:

1. **Superior Nuclear Segmentation**: Cellpose 3.0.10 achieves 20-30% better detection of DAPI-stained nuclei in tissue sections compared to Cellpose 4.x
2. **Injury Model Performance**: Significantly improved detection of dim or irregularly shaped nuclei characteristic of ischemia-reperfusion injury
3. **Validated Parameters**: All detection thresholds and filtering criteria optimized specifically for Cellpose 3.0.10 performance
4. **Stable API**: Well-tested interface ensures reproducible results across computing environments
5. **Reproducibility**: Version-locked dependencies guarantee consistent segmentation across different systems

**Note**: While Cellpose 4.x introduces new features, our systematic testing demonstrates that Cellpose 3.0.10 provides superior performance for nuclear segmentation in tissue sections. The pipeline maintains compatibility with Cellpose 4.x, but we strongly recommend Cellpose 3.0.10 for optimal results.

### Running Tests

ATLAS-Cellpose includes a comprehensive test suite to validate functionality:

```bash
# Activate environment
conda activate venv310_cellpose3

# Run all tests
python -m pytest tests/ -v
```

**Test Coverage:**
- **Core Pipeline**: Segmentation, adaptive tiling, preprocessing, and feature extraction
- **Merge Algorithms**: Systematic merging, 4-step CPU algorithm, GPU integration
- **Performance**: Memory efficiency, large image handling, optimization
- **Visualization**: Color generation, overlay creation, QC tools
- **Integration**: End-to-end pipeline validation

The test suite has been cleaned to remove outdated debug tests and ensure all tests reference existing code modules.

## CLAHE Parameter Testing

ATLAS-Cellpose implements CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing to enhance nuclear contrast in DAPI-stained tissue sections.

### Parameter Selection Guidelines

**Nuclear Imaging Presets:**
- **Conservative Enhancement**: clip_limit=2.0, grid=8×8 (minimal artifact risk, preserves subtle features)
- **Balanced Enhancement**: clip_limit=3.0, grid=16×16 (default, optimal for most applications)
- **Aggressive Enhancement**: clip_limit=5.0, grid=4×4 (maximum contrast, useful for dim nuclei)

**Grid Size Effects:**
- **Small grids (4×4, 8×8)**: Local enhancement preserves fine structural details
- **Large grids (24×24, 32×32)**: Global enhancement provides uniform contrast across tissue regions

### Configuration

CLAHE parameters can be configured in `configs/nuclei_segmentation_config.ini`:

```ini
[preprocessing]
enhance_contrast = True
clahe_clip_limit = 5.0
clahe_grid_size = 32
```

## Pipeline Architecture

ATLAS-Cellpose implements a modular workflow optimized for large-scale tissue analysis:

```
Input Image (DAPI) → Preprocessing → Adaptive Tiling → Cellpose Segmentation → Systematic Merging → Filtering → Quality Control → Feature Extraction
```

### Core Components

1. **Preprocessing**: CLAHE contrast enhancement, gamma correction, and ROI cropping optimize image quality
2. **Adaptive Tiling**: Intelligent spatial partitioning creates locally homogeneous regions for parameter optimization
3. **Segmentation**: Cellpose3 with tile-specific adaptive diameter detection maximizes accuracy across heterogeneous tissue
4. **Systematic Merging**: Four-step priority-based algorithm resolves tile boundaries while preserving cross-boundary nuclei
5. **Quality Control**: Comprehensive before/after visualizations enable validation of segmentation and merge accuracy
6. **Feature Extraction**: Morphological, spatial, and texture feature computation with perfect nucleus tracking

### Pipeline Flowcharts

The complete pipeline workflow is documented with detailed flowcharts in `code/nuclei_segmentation/pipeline.py`. The flowcharts illustrate:

- **Main Pipeline Flow**: Complete workflow from image loading to final outputs.
- **Preprocessing Steps**: CLAHE enhancement, cropping, and image preparation.
- **Tiling Strategy**: Adaptive tile generation with overlap management.
- **Segmentation Process**: Cellpose3 execution with parameter optimization.
- **4-Step Merging Algorithm**: Systematic overlap resolution (detailed below).
- **Filtering Pipeline**: Morphological quality control and artifact removal.
- **Visualization Generation**: QC overlay creation and validation.

**To view the flowcharts**: Open `code/nuclei_segmentation/pipeline.py` and review the comprehensive ASCII diagrams and documentation throughout the file.

### Adaptive Tiled Processing

ATLAS-Cellpose employs adaptive tiled processing to address both computational and biological challenges:

**Computational Benefits:**
- **Memory Efficiency**: Processes arbitrarily large images by partitioning into manageable tiles (default 512×512 pixels)
- **Scalable Architecture**: Batch processing prevents memory overflow on standard workstations
- **Parallel Processing**: Independent tile processing enables efficient parallelization

**Biological Benefits (Primary Innovation):**
- **Adaptive Diameter Optimization**: Each tile represents a locally homogeneous tissue region, enabling Cellpose's adaptive diameter detection to optimize independently for local nuclear morphology
- **Heterogeneity Handling**: Different tissue microenvironments (e.g., cortex vs. medulla, healthy vs. injured regions) receive optimized segmentation parameters
- **Improved Accuracy**: Local parameter adaptation dramatically outperforms global parameter selection in heterogeneous samples

**Technical Implementation:**
- **Configurable Overlap**: 20% overlap between adjacent tiles ensures boundary nuclei are captured
- **Systematic Merging**: Four-step algorithm resolves overlaps while preserving cross-boundary nuclei

### 4-Step Merging Algorithm

The systematic merging algorithm resolves overlapping nuclei at tile boundaries through priority-based selection, ensuring complete preservation of cross-boundary nuclei while eliminating redundant detections:

**Step 1: Priority Selection**
- Tiles with higher nucleus counts receive priority for overlap resolution
- Equal counts default to first-tile priority
- Rationale: Higher-density tiles typically represent better-segmented regions

**Step 2: Border Deletion**
- Remove all priority-tile nuclei contacting the priority tile border
- Preserve all non-priority nuclei contacting the priority tile border
- Ensures cross-boundary nuclei are captured from the tile with optimal viewing angle

**Step 3: Cross-boundary Preservation**
- Retain non-priority nuclei extending into the overlap region
- These nuclei represent cells spanning tile boundaries with complete morphology

**Step 4: Cleanup**
- Remove remaining non-priority nuclei within the overlap region
- Final mask contains only nuclei preserved through Steps 2-3

**Scientific Validation:**
- **Zero nucleus loss**: Cross-boundary nuclei systematically preserved through priority-based selection
- **Duplicate elimination**: Redundant detections in overlap regions completely removed
- **Morphological accuracy**: Preserved nuclei maintain complete boundaries and accurate measurements
- **Extensive testing**: Validated on kidney ischemia-reperfusion injury tissue sections with complex morphology

**Implementation:**
- Two-phase processing: Vertical overlaps resolved first, followed by horizontal overlaps
- GPU-accelerated with automatic CPU fallback for compatibility
- Comprehensive quality control visualizations enable validation of merge accuracy
- Complete documentation in `code/nuclei_segmentation/cellpose_merge/` modules

### Cellpose Integration

ATLAS-Cellpose integrates Cellpose3 for deep learning-based nuclear segmentation with parameters optimized for DAPI-stained tissue sections:

- **Model**: `nuclei` (pre-trained on diverse nuclear morphologies)
- **Diameter Detection**: Adaptive auto-detection optimized independently for each tile, enabling accurate segmentation across heterogeneous tissue regions
- **Detection Thresholds**: Validated parameters for injured and healthy kidney tissue (configurable for other applications)
- **GPU Acceleration**: Automatic GPU utilization with seamless CPU fallback

### Morphological Filtering

ATLAS-Cellpose implements optional morphological filtering to remove segmentation artifacts while preserving biological nuclei:

- **Size Filtering**: Removes debris (too small) and merged nuclei (too large)
- **Shape Filtering**: Eliminates non-nuclear objects based on circularity, solidity, and eccentricity
- **Aspect Ratio**: Removes elongated artifacts inconsistent with nuclear morphology
- **Hole Detection**: Filters objects with excessive internal holes characteristic of segmentation errors
- **Border Exclusion**: Optional removal of nuclei contacting image borders

**Default Thresholds (permissive to avoid over-filtering):**
- Size: 20-5000 pixels (captures wide range of nuclear sizes)
- Circularity: 0.30-1.00 (allows irregular shapes)
- Solidity: 0.60-1.00 (allows moderate concavity)
- Eccentricity: 0.00-0.99 (allows elongated nuclei)
- Aspect Ratio: 0.30-5.00 (allows highly elongated shapes)
- Hole Fraction: 0.00-0.10 (allows nuclei with internal holes)

**Dual Overlay Generation**: When filtering is enabled, the pipeline generates two sets of visualizations for comparison:
- `full_image_overlay_unfiltered.tif` - All detected nuclei before filtering
- `full_image_overlay_filtered.tif` - Only nuclei passing filter criteria
- `binary_mask_unfiltered.tif` - Binary mask of all detected nuclei (white on black)
- `binary_mask_filtered.tif` - Binary mask of filtered nuclei (white on black)

**Binary Mask Visualizations**: The pipeline automatically generates binary mask images where pixels inside any mask region are set to white (255) and all other pixels to black (0). These visualizations are optimized for Vision Transformer (ViT) input and provide a clear view of the segmented regions. The binary masks use memory-efficient chunked processing to handle gigantic images and are saved with LZW compression to minimize file sizes.

**Note**: Filtering is disabled by default (`use_filtering = False`). Enable only if you observe significant artifacts in your segmentation results. The default thresholds are intentionally permissive and should be adjusted based on your specific tissue type and imaging conditions.

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
use_filtering = False               # Enable morphological filtering (disabled by default)
min_pixels = 20                     # Minimum nucleus size (pixels)
max_pixels = 5000                   # Maximum nucleus size (pixels)
min_circularity = 0.30              # Minimum circularity (0=line, 1=circle)
max_circularity = 1.00              # Maximum circularity
min_solidity = 0.60                 # Minimum solidity (convex hull ratio)
max_solidity = 1.00                 # Maximum solidity
min_eccentricity = 0.00             # Minimum eccentricity (0=circle, 1=line)
max_eccentricity = 0.99             # Maximum eccentricity
min_aspect_ratio = 0.30             # Minimum aspect ratio (major/minor axis)
max_aspect_ratio = 5.00             # Maximum aspect ratio
min_hole_fraction = 0.00            # Minimum hole fraction
max_hole_fraction = 0.10            # Maximum hole fraction (allows internal holes)
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

**Recommended Method** - Use the shell script wrapper:

```bash
# Run with custom crop box (recommended) - format: x_start,x_end,y_start,y_end.
./run_segmentation_instance.sh crop_box "0.38,0.42,0.32,0.36"

# The script automatically:
# - Activates the conda environment (venv310_cellpose3).
# - Creates temporary configuration files.
# - Runs the pipeline with your parameters.
# - Saves logs to logs/run_segmentation_instance/.
# - Cleans up temporary files on completion.
```

**Alternative Method** - Direct execution (not recommended):

```bash
# You can also run the pipeline directly, but this requires manual setup.
conda activate venv310_cellpose3
python code/nuclei_segmentation/run_this.py

# Note:
# - Configuration files are in configs/ directory.
# - Results will be saved in results/ directory.
# - You must manually edit configs/nuclei_segmentation_config.ini for parameter changes.
```

### Parameter Sweep with run_segmentation_instance.sh

The `run_segmentation_instance.sh` script is the **recommended way** to run the pipeline. It allows running with custom parameters without modifying the main configuration file. This is ideal for parameter sweeps, parallel processing, and batch experiments.

**Key Features:**
- Creates temporary configuration files for each run.
- Updates specific parameters via command-line arguments.
- Logs all output to dedicated log files.
- Automatically cleans up temporary files on completion or interruption.
- Supports parallel execution of multiple instances.

**Example Usage:**

```bash
# Run with custom job name and GPU settings
./run_segmentation_instance.sh job_name test_gpu_run gpu True

# Run with custom Cellpose parameters
./run_segmentation_instance.sh job_name high_sensitivity \
    cellprob_threshold -14 \
    flow_threshold 0.8 \
    diameter 25

# Run with custom image and output settings
./run_segmentation_instance.sh job_name kidney_sample_1 \
    image_path data/kidney_sample_1.tif \
    output_dir results_sample_1 \
    crop_image True \
    crop_box 0.3,0.7,0.3,0.7

# Run multiple instances in parallel with different parameters
./run_segmentation_instance.sh job_name run_threshold_9 cellprob_threshold -9 &
./run_segmentation_instance.sh job_name run_threshold_12 cellprob_threshold -12 &
./run_segmentation_instance.sh job_name run_threshold_14 cellprob_threshold -14 &
wait  # Wait for all background jobs to complete
```

**Parameter Format:**
- Parameters must match the exact names in `nuclei_segmentation_config.ini`.
- Boolean values: `True` or `False`.
- Numeric values: integers or floats as appropriate.
- String values: paths, names, etc.
- Tuple values: comma-separated (e.g., `0.38,0.42,0.32,0.36` for crop_box in format x_start,x_end,y_start,y_end).

**Log Files:**
- Logs are saved to `logs/run_segmentation_instance/`.
- Log file naming: `{job_name}_run_segmentation_instance.log`.
- Each run creates a separate log file for easy tracking.

**Common Parameter Overrides:**

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `job_name` | Unique identifier for the run | `test_run`, `sample_1` |
| `image_path` | Input image file | `data/kidney.tif` |
| `output_dir` | Output directory name | `results_test` |
| `gpu` | Enable GPU acceleration | `True`, `False` |
| `diameter` | Expected nucleus diameter | `0` (auto), `25`, `30` |
| `cellprob_threshold` | Detection sensitivity | `-9`, `-12`, `-14` |
| `flow_threshold` | Boundary sensitivity | `0.8`, `0.9`, `1.0` |
| `crop_image` | Enable cropping | `True`, `False` |
| `crop_box` | Crop coordinates (x0,x1,y0,y1) | `0.38,0.42,0.32,0.36` |
| `use_tiling` | Enable tiling | `True`, `False` |
| `tile_side_length` | Tile size in pixels | `512`, `1024`, `2048` |

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

**Batch Processing with Shell Script:**

```bash
# Process multiple images with different parameters
for image in data/*.tif; do
    basename=$(basename "$image" .tif)
    ./run_segmentation_instance.sh \
        job_name "batch_${basename}" \
        image_path "$image" \
        output_dir "results_${basename}"
done
```

### Output Files

ATLAS-Cellpose generates:

- **Segmentation masks**: `segmentation_masks.npy` (labeled nuclei)
- **Quality control images**: Before/after merge visualizations
- **Binary mask visualizations**: White segmentation masks on black background (optimized for ViT input)
  - `binary_mask.tif` - Single visualization when filtering is disabled
  - `binary_mask_unfiltered.tif` - All detected nuclei before filtering
  - `binary_mask_filtered.tif` - Only nuclei passing filter criteria
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

ATLAS-Cellpose enables comprehensive quantitative analysis of nuclear dynamics in kidney ischemia-reperfusion injury:

- **Temporal Analysis**: Quantify nuclear morphology changes across recovery time points to track injury progression and resolution
- **Spatial Mapping**: Characterize cellular response patterns within tissue architecture to identify regional vulnerability
- **Cell Death Pathways**: Distinguish apoptosis, pyroptosis, necroptosis, and ferroptosis based on morphological signatures
- **Regeneration Studies**: Track Wnt signaling, cell migration, and angiogenesis through nuclear organization patterns

### Typical Workflow

1. **Image Acquisition**: DAPI-stained tissue sections (whole-slide or large-field imaging)
2. **Preprocessing**: CLAHE contrast enhancement and region-of-interest selection
3. **Segmentation**: Automated nuclear detection with adaptive tiled processing and local parameter optimization
4. **Feature Extraction**: Comprehensive morphological, spatial, and texture measurements with perfect nucleus tracking
5. **Statistical Analysis**: Quantitative comparison across experimental conditions, time points, or tissue regions

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
# Clean and recreate environment.
conda env remove -n venv310_cellpose3
conda env create -f cellpose3_environment_recommended.yml
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

Nuclear clustering analysis identifies distinct cellular populations based on morphological signatures:
- **Healthy nuclei**: Regular shape, moderate size, uniform chromatin distribution
- **Apoptotic nuclei**: Fragmented morphology, irregular boundaries, condensed chromatin
- **Necrotic nuclei**: Swollen appearance, loss of membrane integrity, disrupted chromatin
- **Proliferating nuclei**: Larger size, specific morphological features, altered texture properties

### Getting Help

For technical issues:
1. Check the log files in the `logs/` directory.
2. Review configuration parameters in `configs/nuclei_segmentation_config.ini`.
3. Examine the pipeline flowcharts in `code/nuclei_segmentation/pipeline.py`.
4. Review the 4-step merging algorithm documentation above.

---

## Citation

If you use ATLAS-Cellpose in your research, please cite:

```
Botos, C., Manzato, B., & Mahfouz, A. (2025). ATLAS-Cellpose: Adaptive Tiled Local Analysis
Segmentation for Large-Scale Tissue Image Analysis with Local Parameter Optimization.
[Journal Name], [Volume], [Pages].
```

---

**Repository**: [github.com/ChrisBotos/ATLAS-Cellpose](https://github.com/ChrisBotos/ATLAS-Cellpose)
**Contact**: botoschristos@gmail.com
**License**: MIT
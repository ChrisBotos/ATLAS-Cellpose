# ATLAS-Cellpose
## Adaptive Tiled Local Analysis Segmentation

ATLAS-Cellpose is a tiled nuclear segmentation pipeline for large DAPI-stained tissue images. It wraps Cellpose 3.0.10 with adaptive tiling, a 4-step tile merge algorithm, CLAHE preprocessing, and morphological filtering. It was built for ischemia-reperfusion injury studies in kidney tissue.

**Authors**: Christos Botos (lead developer) and Benedetta Manzato (supervisor)
**Affiliation**: Human Genetics Department, Leiden University Medical Center
**Principal Investigator**: Ahmed Mahfouz

---

## Overview

ATLAS-Cellpose (**A**daptive **T**iled **L**ocal **A**nalysis **S**egmentation) partitions large images into overlapping tiles, runs Cellpose on each tile independently, and merges the results with a 4-step priority-based algorithm. Tiling lets Cellpose's adaptive diameter detection optimize per-tile, which helps when nuclear morphology varies across the tissue.

### What It Does

1. **Adaptive Diameter Optimization**: Tiling creates locally homogeneous regions where Cellpose's diameter detection can optimize independently for each region.
2. **Intelligent Spatial Partitioning**: Dynamic tiling with configurable overlap enables processing of large images while maintaining biological context.
3. **Systematic Boundary Resolution**: A 4-step merging algorithm resolves tile boundaries using priority-based selection.
4. **Scalable Architecture**: Memory-efficient tiled processing enables large image analysis on standard workstations.

### Research Applications

ATLAS-Cellpose was developed for ischemia-reperfusion injury studies in kidney tissue, enabling quantitative analysis of:
- **Temporal dynamics**: Nuclear morphology changes across recovery time points.
- **Spatial organization**: Cellular response patterns within tissue architecture.
- **Cell death pathways**: Apoptosis, pyroptosis, necroptosis, and ferroptosis markers.
- **Regenerative processes**: Wnt signaling, cell migration, and angiogenesis.

The framework can be applied to other large-scale tissue imaging tasks requiring nuclear segmentation.

## Key Features

- **Adaptive Parameter Optimization**: Per-tile diameter detection for heterogeneous tissue regions.
- **4-Step Tile Merging**: Priority-based boundary resolution that aims to preserve cross-boundary nuclei while eliminating duplicates.
- **Scalable Architecture**: Memory-efficient tiled processing of large images.
- **Cellpose 3.0.10 Integration**: Optimized for DAPI-stained tissue sections.
- **Quality Control**: Visualization overlays for segmentation and merge validation.
- **GPU-Accelerated Segmentation**: GPU support for Cellpose inference with automatic CPU fallback. The merge step is CPU-only.
- **CLAHE Preprocessing**: Configurable contrast enhancement.
- **Morphological Filtering**: Optional artifact removal based on size, shape, and solidity thresholds.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scientific Applications](#scientific-applications)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 (installed automatically with environment)
- **CUDA**: 11.8 (optional, for GPU acceleration)
- **Memory**: >= 8 GB RAM (>= 16 GB for large images)
- **Storage**: >= 5 GB free space for conda environment
- **Conda**: Miniconda or Anaconda

### Environment Setup

**Important**: This project is optimized for **Cellpose 3.0.10**. The environment file `cellpose3_environment_recommended.yml` contains all tested package versions.

#### Step 1: Install Miniconda (if not already installed)

```bash
# Download Miniconda installer.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda to your home directory.
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize conda for your shell.
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash

# Reload your shell configuration.
source ~/.bashrc
```

#### Step 2: Create the Cellpose3 Environment

```bash
# Navigate to the ATLAS-Cellpose directory.
cd /path/to/ATLAS-Cellpose

# Create the environment from the YAML file.
conda env create -f cellpose3_environment_recommended.yml
```

**Note**: The yml file's pip section installs torch, cellpose, and opencv. If the pip step fails (e.g. due to CUDA version mismatch), install those packages manually after creating the environment:

```bash
conda activate venv310_cellpose3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install cellpose==3.0.10
pip install opencv-python-headless
```

#### Step 3: Install the Package

```bash
# Activate the environment.
conda activate venv310_cellpose3

# Install ATLAS-Cellpose in editable mode.
pip install -e . --no-deps
```

#### Step 4: Test the Pipeline

**Recommended Method** - Use the shell script wrapper:

```bash
# Run with custom parameters.
# Add your image to the data/ directory and update image_path below.
# crop_box coordinates are fractions of image size (x_start,x_end,y_start,y_end).
./tasks_and_tools/run_segmentation_instance.sh image_path "data/your_image.tif" crop_box "0.38,0.42,0.32,0.36"
```

**Alternative Method** - Direct execution:

```bash
conda activate venv310_cellpose3
python src/atlas_cellpose/run_this.py

# Note: Requires manual editing of configs/nuclei_segmentation_config.ini.
```

### Quick Start (For Experienced Users)

```bash
conda env create -f cellpose3_environment_recommended.yml && \
conda activate venv310_cellpose3 && \
pip install -e . --no-deps && \
./tasks_and_tools/run_segmentation_instance.sh image_path "data/your_image.tif" crop_box "0.38,0.42,0.32,0.36"
```

### Why Cellpose 3.0.10?

In our testing on DAPI-stained kidney tissue sections, Cellpose 3.0.10 produced better nuclear segmentation results than Cellpose 4.x, particularly for dim or irregularly shaped nuclei in injured tissue. All detection thresholds and filtering criteria were optimized for this version. The pipeline has experimental Cellpose 4 support (`use_cellpose4 = True`) but this is less tested.

## CLAHE Parameter Testing

ATLAS-Cellpose implements CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing to enhance nuclear contrast.

### Parameter Selection Guidelines

**Nuclear Imaging Presets:**
- **Conservative Enhancement**: clip_limit=2.0, grid=8x8 (minimal artifact risk)
- **Balanced Enhancement**: clip_limit=3.0, grid=16x16
- **Aggressive Enhancement**: clip_limit=5.0, grid=32x32 (maximum contrast, useful for dim nuclei)

### Configuration

CLAHE parameters are configured in `configs/nuclei_segmentation_config.ini`:

```ini
[general]
enhance_contrast = True

[clahe]
cliplimit = 5.0
tile_grid_size = 32,32
```

## Pipeline Architecture

ATLAS-Cellpose implements a modular workflow:

```
Input Image (DAPI) -> Preprocessing -> Adaptive Tiling -> Cellpose Segmentation -> 4-Step Merging -> Filtering -> Quality Control
```

### Core Components

1. **Preprocessing**: CLAHE contrast enhancement, gamma correction, and ROI cropping.
2. **Adaptive Tiling**: Spatial partitioning into overlapping tiles (default 400x400 pixels, 20% overlap).
3. **Segmentation**: Cellpose 3.0.10 with per-tile adaptive diameter detection.
4. **Systematic Merging**: 4-step priority-based algorithm resolves tile boundaries (CPU-only).
5. **Filtering**: Optional morphological filtering to remove artifacts.
6. **Quality Control**: Before/after visualizations for validation.

### Pipeline Flowcharts

The complete pipeline workflow is documented with detailed flowcharts in `src/atlas_cellpose/pipeline.py`.

### Adaptive Tiled Processing

**Computational Benefits:**
- **Memory Efficiency**: Processes large images by partitioning into tiles (default 400x400 pixels).
- **Scalable Architecture**: Batch processing prevents memory overflow.
- **Parallel Processing**: Independent tile processing enables parallelization.

**Biological Benefits:**
- **Adaptive Diameter Optimization**: Each tile is a locally homogeneous region, enabling per-tile diameter detection.
- **Heterogeneity Handling**: Different tissue microenvironments receive optimized segmentation parameters.

**Technical Implementation:**
- **Configurable Overlap**: 20% overlap between adjacent tiles ensures boundary nuclei are captured.
- **Systematic Merging**: 4-step algorithm resolves overlaps.

### 4-Step Merging Algorithm

The merging algorithm resolves overlapping nuclei at tile boundaries through priority-based selection:

**Step 1: Priority Selection**
- Tiles with higher nucleus counts receive priority for overlap resolution.
- Equal counts default to first-tile priority.

**Step 2: Border Deletion**
- Remove priority-tile nuclei contacting the priority tile border.
- Preserve non-priority nuclei contacting the priority tile border.

**Step 3: Cross-boundary Preservation**
- Retain non-priority nuclei extending into the overlap region.

**Step 4: Cleanup**
- Remove remaining non-priority nuclei within the overlap region.

**Implementation Notes:**
- Two-phase processing: vertical overlaps resolved first, then horizontal overlaps.
- The merge step is CPU-only. GPU acceleration applies to Cellpose segmentation, not merging.
- The algorithm intentionally deletes some nuclei (border-touching priority nuclei in Step 2) to resolve duplicates. It aims to preserve cross-boundary nuclei but does not guarantee zero loss.
- QC visualizations in `src/atlas_cellpose/cellpose_merge/` modules help validate merge quality.
- Cellpose's built-in `stitch_threshold` parameter requires a uniform diameter across all tiles. Because ATLAS-Cellpose runs each tile with `diameter=0` for adaptive per-tile diameter detection, the built-in stitching is incompatible and a custom merge algorithm is used instead.

### Cellpose Integration

- **Model**: `nuclei` (pre-trained on diverse nuclear morphologies).
- **Diameter Detection**: Adaptive auto-detection optimized independently for each tile.
- **Detection Thresholds**: Validated parameters for injured and healthy kidney tissue.
- **GPU Acceleration**: Automatic GPU utilization for Cellpose inference with CPU fallback.

### Morphological Filtering

Optional morphological filtering removes segmentation artifacts while preserving biological nuclei:

- **Size Filtering**: Removes debris (too small) and merged nuclei (too large).
- **Shape Filtering**: Eliminates non-nuclear objects based on circularity, solidity, and eccentricity.
- **Aspect Ratio**: Removes elongated artifacts.
- **Hole Detection**: Filters objects with excessive internal holes.
- **Border Exclusion**: Optional removal of nuclei contacting image borders.

**Default Thresholds** (from `configs/nuclei_segmentation_config.ini`):
- Size: 20-3000 pixels
- Circularity: 0.40-1.00
- Solidity: 0.60-1.00
- Eccentricity: 0.00-0.99
- Aspect Ratio: 0.50-4.00
- Hole Fraction: 0.00-0.05

**Dual Overlay Generation**: When filtering is enabled, the pipeline generates two sets of visualizations:
- `full_image_overlay_unfiltered.tif` - All detected nuclei before filtering.
- `full_image_overlay_filtered.tif` - Only nuclei passing filter criteria.
- `binary_mask_unfiltered.tif` - Binary mask of all detected nuclei (white on black).
- `binary_mask_filtered.tif` - Binary mask of filtered nuclei (white on black).

**Binary Mask Visualizations**: The pipeline generates binary mask images where pixels inside any mask region are set to white (255) and all other pixels to black (0). These are useful as input for Vision Transformers. The binary masks use memory-efficient chunked processing and LZW compression.

**Note**: Filtering is enabled by default (`use_filtering = True`). The default thresholds are intentionally permissive and should be adjusted based on your specific tissue type and imaging conditions.

## Configuration

ATLAS-Cellpose is configured through `configs/nuclei_segmentation_config.ini`:

### Key Parameters

```ini
[general]
image_path = kidney_section.tif     # Input DAPI image
output_dir = results_timestamp      # Output directory
enhance_contrast = True             # Apply CLAHE preprocessing
crop_image = True                   # Enable ROI cropping

[cellpose]
model_type = nuclei                 # Use nuclear model
gpu = True                          # Enable GPU acceleration
use_cellpose4 = False               # Use Cellpose3 (recommended)
diameter = 0                        # Auto-detection
flow_threshold = 0.8                # Boundary detection sensitivity
cellprob_threshold = -14            # Cell detection sensitivity

[tiling]
use_tiling = True                   # Enable for large images
tile_side_length = 400              # Tile size (pixels)
tile_overlap = 0.2                  # 20% overlap between tiles
qc_overlays = True                  # Generate QC images

[filtering]
use_filtering = True                # Enable morphological filtering
min_pixels = 20                     # Minimum nucleus size (pixels)
max_pixels = 3000                   # Maximum nucleus size (pixels)
min_circularity = 0.40              # Minimum circularity (0=line, 1=circle)
max_circularity = 1.00              # Maximum circularity
min_solidity = 0.60                 # Minimum solidity (convex hull ratio)
max_solidity = 1.00                 # Maximum solidity
min_eccentricity = 0.00             # Minimum eccentricity (0=circle, 1=line)
max_eccentricity = 0.99             # Maximum eccentricity
min_aspect_ratio = 0.50             # Minimum aspect ratio (major/minor axis)
max_aspect_ratio = 4.00             # Maximum aspect ratio
min_hole_fraction = 0.00            # Minimum hole fraction
max_hole_fraction = 0.05            # Maximum hole fraction
exclude_border = False              # Exclude border-touching nuclei
```

### Parameter Optimization

**For healthy tissue:**
- `cellprob_threshold = -9` (standard sensitivity)
- `flow_threshold = 0.9` (standard boundaries)

**For injured/inflamed tissue:**
- `cellprob_threshold = -14` (high sensitivity for dim nuclei)
- `flow_threshold = 0.8` (more sensitive boundaries)

## Usage

### Basic Usage

**Recommended Method** - Use the shell script wrapper:

```bash
# Run with custom crop box - format: x_start,x_end,y_start,y_end.
./tasks_and_tools/run_segmentation_instance.sh crop_box "0.38,0.42,0.32,0.36"

# The script automatically:
# - Activates the conda environment (venv310_cellpose3).
# - Creates temporary configuration files.
# - Runs the pipeline with your parameters.
# - Saves logs to logs/run_segmentation_instance/.
# - Cleans up temporary files on completion.
```

**Alternative Method** - Direct execution:

```bash
conda activate venv310_cellpose3
python src/atlas_cellpose/run_this.py

# Note:
# - Configuration files are in configs/ directory.
# - Results will be saved in results/ directory.
# - You must manually edit configs/nuclei_segmentation_config.ini for parameter changes.
```

### Parameter Sweep with run_segmentation_instance.sh

The `run_segmentation_instance.sh` script is the **recommended way** to run the pipeline. It allows running with custom parameters without modifying the main configuration file.

**Key Features:**
- Creates temporary configuration files for each run.
- Updates specific parameters via command-line arguments.
- Logs all output to dedicated log files.
- Automatically cleans up temporary files on completion or interruption.
- Supports parallel execution of multiple instances.

**Example Usage:**

```bash
# Run with custom job name and GPU settings.
./tasks_and_tools/run_segmentation_instance.sh job_name test_gpu_run gpu True

# Run with custom Cellpose parameters.
./tasks_and_tools/run_segmentation_instance.sh job_name high_sensitivity \
    cellprob_threshold -14 \
    flow_threshold 0.8 \
    diameter 25

# Run with custom image and output settings.
./tasks_and_tools/run_segmentation_instance.sh job_name kidney_sample_1 \
    image_path data/kidney_sample_1.tif \
    output_dir results_sample_1 \
    crop_image True \
    crop_box 0.3,0.7,0.3,0.7

# Run multiple instances in parallel with different parameters.
./tasks_and_tools/run_segmentation_instance.sh job_name run_threshold_9 cellprob_threshold -9 &
./tasks_and_tools/run_segmentation_instance.sh job_name run_threshold_12 cellprob_threshold -12 &
./tasks_and_tools/run_segmentation_instance.sh job_name run_threshold_14 cellprob_threshold -14 &
wait  # Wait for all background jobs to complete.
```

**Parameter Format:**
- Parameters must match the exact names in `nuclei_segmentation_config.ini`.
- Boolean values: `True` or `False`.
- Numeric values: integers or floats as appropriate.
- String values: paths, names, etc.
- Tuple values: comma-separated (e.g., `0.38,0.42,0.32,0.36` for crop_box).

**Log Files:**
- Logs are saved to `logs/run_segmentation_instance/`.
- Log file naming: `{job_name}_run_segmentation_instance.log`.

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
| `tile_side_length` | Tile size in pixels | `400`, `512`, `1024` |

### Batch Processing

```python
from pathlib import Path
from atlas_cellpose.utils.project_setup import load_config
from atlas_cellpose.pipeline import run_segmentation_pipeline
from atlas_cellpose.utils.logging_utils import setup_logging

# Process all TIFF files in directory.
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
# Process multiple images with different parameters.
for image in data/*.tif; do
    basename=$(basename "$image" .tif)
    ./tasks_and_tools/run_segmentation_instance.sh \
        job_name "batch_${basename}" \
        image_path "$image" \
        output_dir "results_${basename}"
done
```

### Output Files

ATLAS-Cellpose generates:

- **Segmentation masks**: `segmentation_masks.npy` (labeled nuclei).
- **Quality control images**: Before/after merge visualizations.
- **Binary mask visualizations**: White segmentation masks on black background.
  - `binary_mask.tif` - Single visualization when filtering is disabled.
  - `binary_mask_unfiltered.tif` / `binary_mask_filtered.tif` - When filtering is enabled.
- **Configuration snapshot**: Reproducible parameter settings.
- **Log files**: Detailed processing information.

## Scientific Applications

### Ischemia-Reperfusion Injury Analysis

ATLAS-Cellpose enables quantitative analysis of nuclear dynamics in kidney ischemia-reperfusion injury:

- **Temporal Analysis**: Quantify nuclear morphology changes across recovery time points.
- **Spatial Mapping**: Characterize cellular response patterns within tissue architecture.
- **Cell Death Pathways**: Distinguish apoptosis, pyroptosis, necroptosis, and ferroptosis based on morphological signatures.
- **Regeneration Studies**: Track Wnt signaling, cell migration, and angiogenesis through nuclear organization patterns.

### Typical Workflow

1. **Image Acquisition**: DAPI-stained tissue sections (whole-slide or large-field imaging).
2. **Preprocessing**: CLAHE contrast enhancement and region-of-interest selection.
3. **Segmentation**: Automated nuclear detection with adaptive tiled processing.
4. **Quality Control**: Visual validation using overlay visualizations.
5. **Statistical Analysis**: Quantitative comparison across experimental conditions.

## Performance

### System Requirements

**Recommended:**
- **RAM**: 16GB+ for large tissue sections
- **GPU**: NVIDIA GPU with CUDA 11.8 support (optional, speeds up Cellpose inference)
- **CPU**: 8+ cores for parallel tile processing
- **Storage**: SSD for faster I/O operations

**Minimum:**
- **RAM**: 8GB
- **CPU**: 4+ cores
- **Python**: 3.10

### Performance Notes

- **GPU Acceleration**: Applies to Cellpose segmentation inference. The merge step runs on CPU only.
- **Memory Management**: Tiled processing with configurable tile size prevents memory overflow.
- **Parallel Processing**: Multi-core CPU utilization for tile segmentation and feature extraction.

### GPU Acceleration

GPU acceleration is automatically enabled when CUDA-compatible hardware is detected. The pipeline falls back to CPU processing if GPU is unavailable. To force CPU mode, set `gpu = False` in the config or pass it via the shell script.

## Troubleshooting

### Common Issues

**Environment activation fails:**
```bash
# Ensure conda is properly initialized.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv310_cellpose3
```

**CUDA compatibility problems:**
```bash
# Check CUDA version.
nvidia-smi

# Reinstall PyTorch for your CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Memory issues during processing:**
- Reduce `tile_side_length` in configuration.
- Set `gpu = False` to use CPU-only processing.
- Increase system swap space.

**Package import failures:**
```bash
# Clean and recreate environment.
conda env remove -n venv310_cellpose3
conda env create -f cellpose3_environment_recommended.yml
conda activate venv310_cellpose3
pip install -e . --no-deps
```

### Getting Help

For technical issues:
1. Check the log files in the `logs/` directory.
2. Review configuration parameters in `configs/nuclei_segmentation_config.ini`.
3. Examine the pipeline flowcharts in `src/atlas_cellpose/pipeline.py`.
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

# Nuclear Feature Clustering for Kidney I/R Injury Analysis

## Overview

This module provides comprehensive clustering analysis of nuclear morphological features extracted from DAPI-stained kidney tissue sections. The clustering pipeline is specifically designed for analyzing tissue organization patterns and cellular responses during ischemia/reperfusion (I/R) injury progression.

## Key Features

### 🧬 Comprehensive Feature Analysis
- **Morphological Features**: Area, perimeter, circularity, eccentricity, solidity, aspect ratio
- **Intensity Features**: Mean, standard deviation, median, skewness, kurtosis, entropy
- **Texture Features**: Local binary patterns, GLCM properties, gradient features
- **Neighborhood Features**: Spatial clustering, nearest neighbor analysis, tissue organization

### 🎨 Enhanced Color Visualization
- **35+ Distinct Colors**: Extended predefined palette for large cluster numbers
- **Scientific Optimization**: High contrast ratios (WCAG compliant) for microscopy
- **Custom Palettes**: Support for user-defined color schemes
- **Background Adaptation**: Automatic optimization for light/dark backgrounds

### ⚡ Memory-Efficient Processing
- **Streaming Algorithms**: MiniBatchKMeans with configurable batch sizes
- **Scalable Architecture**: Handles datasets with 10,000+ nuclei efficiently
- **Memory Monitoring**: Automatic memory management and optimization

### 📊 Publication-Quality Outputs
- **Cluster Overlays**: High-resolution tissue overlays with cluster colors
- **PCA Visualizations**: Feature space analysis with cluster separation
- **Feature Importance**: Statistical analysis of discriminative features
- **Comprehensive Statistics**: Per-cluster feature summaries and distributions

## Installation and Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy pandas scikit-learn matplotlib seaborn pillow rich typer joblib scipy
```

### Configuration
The clustering system uses a dedicated configuration file (`configs/engineered_feature_extraction_config.ini`) with comprehensive parameter control. Key sections include:

```ini
[general]
# Enable major pipeline components
enable_feature_extraction = True
enable_clustering = True
enable_visualizations = True

[feature_extraction]
# Feature category selection
shape_features = True
size_features = True
neighborhood_features = True
texture_features = True
neighborhood_radius = 50.0
feature_extraction_workers = -1

[clustering]
# Clustering parameters
default_clusters = 12
auto_k_method = silhouette  # 'none', 'silhouette', 'dbi'
max_clusters_test = 25
clustering_batch_size = 5000
clustering_seed = 42

# Visualization settings
generate_cluster_overlay = True
generate_pca_plot = True
generate_feature_importance = True
overlay_crop_region = 0.1, 0.9, 0.1, 0.9

# Color palette configuration
color_background = dark
color_alpha = 200
color_saturation = 0.95
color_contrast_ratio = 4.5

[visualization]
# Publication-quality outputs
figure_dpi = 300
figure_format = png
enable_statistical_testing = True
timepoint_colors = #FF6B6B, #4ECDC4, #45B7D1

[performance]
# Memory and processing optimization
max_memory_gb = 8.0
enable_parallel_processing = True
enable_progress_tracking = True
```

## Usage Examples

### Basic Clustering Analysis
```bash
# Cluster nuclear features with default settings
python cluster_engineered_features.py \
    --features nuclear_features.csv \
    --image tissue_dapi.tif \
    --mask segmentation_masks.npy \
    --outdir results/clustering
```

### Advanced Configuration
```bash
# Use dedicated feature extraction configuration file
python cluster_engineered_features.py \
    --config configs/engineered_feature_extraction_config.ini \
    --features nuclear_features.csv \
    --image tissue_dapi.tif \
    --mask segmentation_masks.npy \
    --clusters 20 \
    --auto-k silhouette \
    --outdir results/clustering_analysis
```

### Custom Region Analysis
```bash
# Analyze specific tissue region with custom parameters
python cluster_engineered_features.py \
    --features nuclear_features.csv \
    --image tissue_dapi.tif \
    --mask segmentation_masks.npy \
    --clusters 15 \
    --region 0.2 0.8 0.2 0.8 \
    --downsample 2 \
    --seed 42
```

### Demonstration Workflow
```bash
# Run complete demo with synthetic data
python example_clustering_workflow.py --demo
```

## Output Files

### Core Results
- `nuclear_clusters.csv`: Original features with cluster assignments
- `kmeans_model.joblib`: Trained clustering model for reuse
- `scaler.joblib`: Feature normalization parameters

### Visualizations
- `cluster_overlay.tif`: Tissue overlay with cluster colors
- `pca_clusters.png`: PCA scatter plot with cluster separation
- `feature_importance.png`: Bar plot of discriminative features

### Analysis Reports
- `cluster_statistics.csv`: Per-cluster feature summaries
- `feature_importance.csv`: Ranked feature importance scores
- `cluster_selection_scores.csv`: K-selection optimization results (if used)

## Scientific Interpretation

### Cluster Analysis for I/R Injury
1. **Healthy Tissue Clusters**: High circularity, uniform size, regular spacing
2. **Early Damage Clusters**: Moderate morphological changes, increased heterogeneity
3. **Severe Damage Clusters**: Low circularity, variable size, disrupted organization

### Feature Importance Interpretation
- **Morphological Features**: Primary indicators of cellular damage
- **Intensity Features**: Reflect chromatin condensation and nuclear integrity
- **Neighborhood Features**: Indicate tissue architecture disruption
- **Texture Features**: Capture subtle chromatin organization changes

### Temporal Analysis (10h, 2d, 14d timepoints)
- **10h**: Early apoptotic changes, moderate clustering
- **2d**: Peak damage with maximum cluster separation
- **14d**: Recovery patterns with intermediate clustering

## Advanced Features

### Custom Color Palettes
```python
# Define custom colors for specific biological contexts
custom_colors = [
    "#FF0000",  # Severe damage - red
    "#FF8C00",  # Moderate damage - orange  
    "#FFFF00",  # Mild damage - yellow
    "#00FF00",  # Healthy - green
    "#0080FF",  # Recovery - blue
]
```

### Batch Processing
```python
# Process multiple tissue sections
for tissue_file in tissue_files:
    python cluster_engineered_features.py \
        --features f"{tissue_file}_features.csv" \
        --image f"{tissue_file}.tif" \
        --mask f"{tissue_file}_mask.npy" \
        --config shared_config.ini
```

### Integration with Feature Extraction

```python
# Complete pipeline from segmentation to clustering
from code.engineered_feature_extraction.extract_engineered_features import process_image_with_config
from code.engineered_feature_extraction.cluster_engineered_features import main as cluster_main
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config

# Load configuration
config = load_feature_extraction_config('configs/engineered_feature_extraction_config.ini')

# Extract features
features_df = process_image_with_config(image_path, mask_path, features_path,
                                        config_path='configs/engineered_feature_extraction_config.ini')

# Cluster features
cluster_main()  # Uses dedicated configuration system automatically
```

## Performance Optimization

### Memory Management
- **Large Datasets (>10K nuclei)**: Use batch_size=2000-5000
- **Limited RAM (<16GB)**: Reduce batch_size to 1000-2000
- **High-resolution images**: Increase downsample_factor to 2-4

### Processing Speed
- **CPU Optimization**: Set feature_extraction_workers=-1 for auto-detection
- **I/O Optimization**: Use SSD storage for temporary files
- **Memory Mapping**: Automatic for large feature matrices

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch_size or increase downsample_factor
2. **Color Visibility**: Adjust color_contrast_ratio or color_alpha
3. **Cluster Quality**: Try different auto_k_method or adjust max_clusters_test
4. **Processing Speed**: Reduce pca_sample_size for faster visualization

### Validation
```bash
# Run comprehensive tests
python -m pytest tests/test_cluster_engineered_features.py -v

# Test color palette generation
python code/engineered_feature_extraction/utils/generate_contrast_colors.py
```

## Citation and References

When using this clustering module in publications, please cite:

```bibtex
@software{nuclear_clustering_2024,
  title={Nuclear Feature Clustering for Kidney I/R Injury Analysis},
  author={Christos Botos},
  institution={Leiden University Medical Center},
  year={2024},
  url={https://github.com/ChrisBotos/Nuclei-Segmentation-with-Cellpose}
}
```

## Support and Development

- **Issues**: Report bugs and feature requests via GitHub issues
- **Documentation**: Comprehensive docstrings in all modules
- **Testing**: Full test suite with synthetic data validation
- **Configuration**: Flexible parameter system for different research contexts

---

*This module is part of the comprehensive I/R injury spatial multiomics analysis pipeline, designed for publication-quality research in kidney pathology and regenerative medicine.*

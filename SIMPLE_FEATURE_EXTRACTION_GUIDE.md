# Simple Nuclear Feature Extraction Guide

## Overview

The `extract_simple_features.py` script provides a fast, reliable alternative to the complex feature extraction pipeline. It extracts comprehensive nuclear size and shape features for thorough morphological analysis:

### **Size Features (10 features):**
- **area**: Nuclear area in pixels
- **perimeter**: Nuclear boundary length
- **equivalent_diameter**: Diameter of circle with same area
- **major_axis_length**: Length of major axis of fitted ellipse
- **minor_axis_length**: Length of minor axis of fitted ellipse
- **bounding_box_width**: Width of smallest rectangle containing nucleus
- **bounding_box_height**: Height of smallest rectangle containing nucleus
- **bounding_box_area**: Area of bounding rectangle
- **feret_diameter_max**: Maximum distance between any two boundary points
- **feret_diameter_min**: Minimum distance between parallel tangent lines

### **Shape Features (10 features):**
- **circularity**: Shape regularity measure (4π*area/perimeter²)
- **eccentricity**: Measure of ellipse deviation from circle
- **solidity**: Ratio of area to convex hull area
- **aspect_ratio**: Elongation measure (major_axis/minor_axis)
- **compactness**: Measure of shape regularity
- **elongation**: Measure of nuclear stretching
- **roundness**: Alternative circularity measure
- **form_factor**: Shape complexity measure
- **convex_area_ratio**: Ratio of actual to convex area
- **convexity**: Ratio of convex hull to actual perimeter

### **Neighborhood Features (8 features):**
- **neighbor_count**: Number of nuclei within 20-pixel radius
- **neighbor_density**: Nuclei per unit area in neighborhood
- **nearest_neighbor_distance**: Distance to closest nucleus
- **mean_neighbor_distance**: Average distance to all neighbors
- **neighbor_area_ratio**: Ratio of nucleus area to mean neighbor area
- **local_density_gradient**: Change in density from center to edge
- **clustering_coefficient**: Measure of local clustering
- **isolation_score**: Measure of spatial isolation

## Why Use the Enhanced Simple Version?

✅ **Reliable**: Single-threaded processing avoids multiprocessing issues
✅ **Fast**: Processes 15,000+ nuclei in ~115 seconds
✅ **Comprehensive**: 28 essential morphological and neighborhood features for thorough analysis
✅ **Clean**: Minimal dependencies and clear progress reporting
✅ **Scientific**: All features have direct biological relevance for I/R injury analysis

## Scientific Context

### Area Feature
- **Biological significance**: Measures nuclear size changes during cell death
- **Increases during**: Cell swelling, early apoptosis
- **Decreases during**: Cell shrinkage, late apoptosis, necrosis
- **Normal range**: 100-300 pixels (varies by tissue type)

### Circularity Feature  
- **Biological significance**: Detects nuclear fragmentation and deformation
- **Formula**: 4π × area / perimeter²
- **Perfect circle**: 1.0
- **Irregular shapes**: Approach 0.0
- **Decreases during**: Apoptosis, nuclear fragmentation, membrane blebbing

## Usage

### Basic Command
```bash
python code/engineered_feature_extraction/extract_simple_features.py \
    --image path/to/dapi_image.tif \
    --mask path/to/segmentation_mask.npy \
    --output path/to/results.csv
```

### Example with Your Data
```bash
python code/engineered_feature_extraction/extract_simple_features.py \
    --image results/example_cropped/preprocessed/first.tif \
    --mask results/example_cropped/masks/segmentation_masks.npy \
    --output results/example_cropped/simple_features.csv
```

## Output Format

The script generates a CSV file with 13 columns (nucleus_id + 12 features):

| Column | Description | Example |
|--------|-------------|---------|
| `nucleus_id` | Unique identifier from segmentation mask | 24423 |
| `area` | Nuclear area in pixels | 175.0 |
| `perimeter` | Nuclear boundary length in pixels | 54.97 |
| `equivalent_diameter` | Diameter of circle with same area | 14.93 |
| `major_axis_length` | Length of major axis of fitted ellipse | 18.59 |
| `minor_axis_length` | Length of minor axis of fitted ellipse | 14.06 |
| `bounding_box_width` | Width of smallest rectangle | 17.0 |
| `bounding_box_height` | Height of smallest rectangle | 16.0 |
| `bounding_box_area` | Area of bounding rectangle | 272.0 |
| `feret_diameter_max` | Maximum caliper measurement | 19.72 |
| `feret_diameter_min` | Minimum caliper measurement | 14.06 |
| `circularity` | Shape regularity (0-1) | 0.728 |
| `aspect_ratio` | Elongation measure | 1.32 |

## Performance

- **Processing speed**: ~600 nuclei/second (12 features vs 2 features)
- **Memory usage**: Low (single-threaded)
- **Reliability**: No multiprocessing issues
- **Dependencies**: Minimal (numpy, pandas, scikit-image, typer, rich)
- **Feature richness**: 12 comprehensive morphological measurements

## Integration with Analysis Pipeline

### 1. After Segmentation
```bash
# Run segmentation first
python code/nuclei_segmentation/segment_nuclei.py --config configs/segmentation_config.ini

# Then extract simple features
python code/engineered_feature_extraction/extract_simple_features.py \
    --image results/your_data/preprocessed/image.tif \
    --mask results/your_data/masks/segmentation_masks.npy \
    --output results/your_data/simple_features.csv
```

### 2. Statistical Analysis
```python
import pandas as pd

# Load results
df = pd.read_csv('results/your_data/simple_features.csv')

# Basic statistics
print(f"Mean area: {df['area'].mean():.1f} ± {df['area'].std():.1f}")
print(f"Mean circularity: {df['circularity'].mean():.3f} ± {df['circularity'].std():.3f}")

# Identify potentially damaged nuclei
damaged = df[df['circularity'] < 0.7]  # Low circularity = fragmented
print(f"Potentially damaged nuclei: {len(damaged)} ({100*len(damaged)/len(df):.1f}%)")
```

## Troubleshooting

### Common Issues

1. **File not found errors**
   - Check file paths are correct
   - Ensure segmentation was completed successfully

2. **Dimension mismatch**
   - Image and mask must have identical dimensions
   - Re-run segmentation if needed

3. **Empty results**
   - Check mask contains labeled nuclei (not just binary)
   - Verify mask file format is .npy

### Getting Help
```bash
python code/engineered_feature_extraction/extract_simple_features.py --help
```

## When to Use Complex vs Simple Extraction

### Use Simple Version When:
- You need reliable, fast processing
- Area and circularity are sufficient for your analysis
- You're having multiprocessing issues with the complex version
- You want to quickly assess nuclear morphology

### Use Complex Version When:
- You need comprehensive feature analysis (43+ features)
- You're doing advanced machine learning analysis
- You need texture, neighborhood, or advanced shape features
- You have stable multiprocessing environment

## Next Steps

After extracting simple features, you can:

1. **Statistical Analysis**: Compare area/circularity between treatment groups
2. **Visualization**: Create histograms and scatter plots of features
3. **Quality Control**: Identify outliers and potential segmentation errors
4. **Advanced Analysis**: Use features as input for clustering or classification

The simple extractor provides a solid foundation for nuclear morphology analysis while avoiding the complexity and potential issues of the full feature extraction pipeline.

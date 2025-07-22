# Memory-Efficient Tile Overlay Functionality

## Overview

This document describes the refactored memory-efficient tile overlay functionality for kidney I/R injury tissue analysis. The system creates two distinct types of visualizations with proper alpha transparency blending on tissue backgrounds:

1. **Before Merging**: Tile-based colors where each tile gets a unique deterministic color
2. **After Merging**: Nucleus-based colors where each individual nucleus gets a unique random color

## Key Features

- **Distinct Color Schemes**:
  - Before merging: Tile-based deterministic colors for boundary identification
  - After merging: Nucleus-based random colors for individual cell visualization
- **Alpha Transparency Blending**: Proper overlay on tissue background for scientific analysis
- **Memory-Efficient Processing**: Handles thousands of tiles using batch processing
- **Flexible Input Handling**: Supports multiple tile naming conventions
- **Comprehensive Error Handling**: Robust processing with detailed logging
- **Integration Ready**: Works seamlessly with existing QC workflows

## Quick Start

### Simple Usage

```python
from qc import create_before_after_overlays, _load_rgb_image

# Load tissue image
full_image = _load_rgb_image(Path("data/IRI_regist_cropped.tif"))

# Create both before and after overlays from a results directory
before_overlay, after_overlay = create_before_after_overlays(
    results_dir=Path("results/20250722_044324_test_new_cellpose4_diameter0_large_crop"),
    full_image=full_image,
    crop_size=1300,
    batch_size=100,
    output_dir=Path("overlay_outputs")
)
```

### Direct Directory Processing

```python
from qc import create_tile_overlay_from_directory, _load_rgb_image

# Load tissue image
full_image = _load_rgb_image(Path("data/IRI_regist_cropped.tif"))

# Create overlay from tile directory
overlay = create_tile_overlay_from_directory(
    tiles_dir=Path("results/run_name/masks/tile_masks_npz"),
    full_image=full_image,
    batch_size=100,
    alpha=0.6,
    crop_size=1300,
    output_path=Path("before_overlay.tif"),
    overlay_type="before"
)
```

## Function Reference

### High-Level Interface

#### `create_before_after_overlays()`
Creates both before and after overlays from a results directory.

**Parameters:**
- `results_dir`: Path to results directory
- `full_image`: RGB tissue image array
- `tile_h`, `tile_w`: Tile dimensions (default: 512)
- `overlap`: Tile overlap (default: 64)
- `batch_size`: Processing batch size (default: 100)
- `alpha`: Overlay transparency (default: 0.6)
- `crop_size`: Crop size for visualization (default: 1300)

### Low-Level Interface

#### `create_tile_overlay_from_directory()`
Creates overlay directly from a tile directory.

**Parameters:**
- `tiles_dir`: Directory containing .npz tile files
- `full_image`: RGB tissue image array
- `tile_h`, `tile_w`: Tile dimensions (default: 512)
- `overlap`: Tile overlap (default: 64)
- `batch_size`: Processing batch size (default: 100)
- `alpha`: Overlay transparency (default: 0.6)
- `crop_size`: Crop size (None for full image)
- `output_path`: Path to save overlay (optional)
- `overlay_type`: "before" or "after" for color scheme

## Directory Structure

The system expects the following directory structure:

```
results/
└── run_name/
    ├── preprocessed/
    │   └── final.tif                # Tissue background image
    └── masks/
        ├── tile_masks_npz/          # Before merging tiles
        │   ├── 0_0.npz
        │   ├── 410_0.npz
        │   └── ...
        └── merged_tile_masks_npz/   # After merging tiles
            ├── 0_0.npz
            ├── 0_1.npz
            └── ...
```

## Tissue Background Integration

The system automatically loads the tissue background image for overlays:

1. **Primary**: `results/run_name/preprocessed/final.tif` (created during preprocessing)
2. **Fallback**: Uses provided `image_loader` function if available
3. **Last Resort**: Creates neutral gray background

This ensures that overlays always show the tissue context for scientific analysis.

## Tile Naming Conventions

The system supports multiple tile naming conventions:

- **Pixel coordinates**: `410_820.npz`, `0_0.npz`
- **Tile indices**: `row1_col2.npz`, `0_1.npz`
- **Space-separated**: `410 820.npz`

## Memory Management

### Batch Processing
- Tiles are processed in configurable batches to manage memory usage
- Default batch size: 100 tiles
- Reduce batch size for systems with limited RAM
- Increase batch size for better performance on high-memory systems

### Memory-Efficient Settings
For large datasets (>1000 tiles):
```python
# Using the main script
python create_tile_overlays.py \
    --results-dir "results/large_run" \
    --image-path "data/image.tif" \
    --batch-size 20 \
    --crop-size 800

# Or programmatically
overlay = create_tile_overlay_from_directory(
    tiles_dir=tiles_dir,
    full_image=full_image,
    batch_size=20,      # Smaller batches
    crop_size=800,      # Smaller crop
    alpha=0.6
)
```

## Color Schemes

### Before Merging (Tile-Based Colors)
- **Purpose**: Identify tile boundaries and overlapping regions
- **Method**: Deterministic colors based on tile coordinates
- **Colors**: Bright palette (reds, greens, blues, yellows, magentas, cyans)
- **Alpha transparency**: 0.6 (default)
- **Reproducible**: Same tile always gets same color across runs

### After Merging (Nucleus-Based Colors)
- **Purpose**: Visualize individual nuclei in final merged segmentation
- **Method**: Random colors assigned to each unique nucleus label
- **Colors**: High-saturation HSV-generated colors for maximum distinction
- **Alpha transparency**: 0.7 (default, more opaque for final results)
- **Unique**: Each nucleus gets its own color regardless of tile origin

### Alpha Transparency Blending
- **Tissue Background**: Uses `final.tif` from preprocessed directory as background
- **Automatic Loading**: System automatically finds and loads the tissue image
- **Scientific Visualization**: Maintains tissue context for analysis
- **Configurable**: Alpha values can be adjusted (0.0 = transparent, 1.0 = opaque)
- **Quality**: 16-bit composition prevents color banding artifacts
- **Fallback**: If `final.tif` not available, uses image_loader or neutral background

## Performance Optimization

### Recommended Settings

**Small datasets (<100 tiles):**
```python
batch_size=50
crop_size=1300
alpha=0.6
```

**Medium datasets (100-1000 tiles):**
```python
batch_size=100
crop_size=1000
alpha=0.6
```

**Large datasets (>1000 tiles):**
```python
batch_size=20
crop_size=800
alpha=0.5
```

### Performance Tips
1. Use smaller crop sizes for faster processing
2. Reduce batch size if running out of memory
3. Use SSD storage for better I/O performance
4. Process during off-peak hours for large datasets

## Error Handling

The system includes comprehensive error handling:

- **Missing directories**: Clear error messages with suggestions
- **Invalid tile files**: Skips corrupted files, continues processing
- **Memory errors**: Automatic fallback to smaller batch sizes
- **File I/O errors**: Detailed logging for troubleshooting

## Integration with Existing Workflow

### QC Pipeline Integration
The new functions integrate seamlessly with the existing QC pipeline:

```python
# In write_overlays function
if original_tiles_path and coords:
    before_overlay = _create_before_merging_overlay_from_files(
        original_tiles_path, coords, crop_info, tile_h, tile_w, overlap, tissue_background
    )
```

### Legacy Compatibility
All existing QC functions continue to work unchanged. The new functionality is accessed through:
- Enhanced `write_overlays()` function
- New high-level interface functions
- Direct directory processing functions

## Examples and Testing

### Main Script
Create overlays using the main script:
```bash
python create_tile_overlays.py --results-dir "results/run_name" --image-path "data/image.tif"
```

### Test Suite
Run comprehensive tests:
```bash
cd tests/nuclei_segmentation_tests
python -m pytest test_tile_overlay_functions.py -v
```

## Troubleshooting

### Common Issues

**"No valid tile files found"**
- Check tile directory path
- Verify .npz files exist
- Check file naming convention

**"Memory error during processing"**
- Reduce batch_size parameter
- Use smaller crop_size
- Close other applications

**"Image file not found"**
- Verify image path is correct
- Check file permissions
- Ensure image is in supported format (TIFF, PNG)

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Scientific Context

This functionality is specifically designed for kidney I/R injury tissue analysis:

- **Tile Identification**: Unique colors help identify tile boundaries and overlaps
- **Quality Assessment**: Visual validation of segmentation merge quality
- **Reproducible**: Deterministic color generation ensures consistent results

## Future Enhancements

Planned improvements:
- GPU acceleration for large datasets
- Interactive visualization tools
- Advanced color scheme customization
- Integration with spatial analysis pipelines
- Support for additional image formats

# Coordinate Mapping Fix Documentation

## Problem Analysis

### Root Cause
The two-phase merge pipeline was failing due to a critical mismatch between the tile coordinate naming conventions used during segmentation and merging:

- **Segmentation Process**: Saves tiles using pixel coordinates (e.g., `0_410.npz`, `0_820.npz`)
- **Merge Process**: Expected simple tile indices (e.g., `0_0.npz`, `0_1.npz`, `1_0.npz`)

### Technical Details
1. **Segmentation Naming**: Uses `y_start` and `x_start` pixel positions from tiling function
   - Files named as `{y_slice.start}_{x_slice.start}.npz`
   - Example: `0_410.npz` where 410 is the x-coordinate pixel position

2. **Merge Expectation**: Assumed tile index coordinates
   - Expected files like `{row}_{col}.npz` where row/col are tile indices
   - Example: `0_1.npz` for tile at row 0, column 1

## Solution Implementation

### Approach Selected
**Option B**: Update the merging process to correctly map tile coordinates to the actual file naming scheme used by segmentation.

**Rationale**:
- Segmentation process is already working and producing files
- Pixel-based naming is more descriptive and allows for variable tile sizes
- Safer to update merge process than change segmentation output format

### Key Functions Added

#### 1. Coordinate Conversion Functions

```python
def _tile_coord_to_pixel_coord(coord: TileCoord, tile_h: int, tile_w: int, overlap: int) -> TileCoord:
    """Convert tile index coordinates to pixel coordinates used in file naming."""
    row, col = coord
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    y_start = row * stride_h
    x_start = col * stride_w
    
    return (y_start, x_start)

def _pixel_coord_to_tile_coord(pixel_coord: TileCoord, tile_h: int, tile_w: int, overlap: int) -> TileCoord:
    """Convert pixel coordinates to tile index coordinates."""
    y_start, x_start = pixel_coord
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    row = y_start // stride_h
    col = x_start // stride_w
    
    return (row, col)
```

#### 2. File Discovery Function

```python
def discover_tile_coordinates_from_files(
    tile_masks_dir: Path,
    tile_h: int,
    tile_w: int,
    overlap: int
) -> List[TileCoord]:
    """Discover tile coordinates by examining existing segmentation output files."""
```

**Features**:
- Scans directory for existing `.npz` files
- Parses pixel coordinate filenames
- Converts to tile index coordinates
- Returns sorted list for consistent processing

#### 3. Updated Storage Functions

```python
def _load_tile_from_storage(coord: TileCoord, storage_dir: Path, tile_h: int, tile_w: int, overlap: int) -> NDArray[np.uint32]:
    """Load a tile mask from persistent storage using pixel coordinate file naming."""

def _save_tile_to_storage(coord: TileCoord, tile_mask: NDArray[np.uint32], storage_dir: Path, tile_h: int, tile_w: int, overlap: int) -> None:
    """Save a tile mask to persistent storage using pixel coordinate file naming."""
```

### Updated Function Signatures

#### File Management Functions
- `copy_tile_masks_to_merged_directory()`: Added `tile_h`, `tile_w`, `overlap` parameters
- `_load_tile_from_storage()`: Added `tile_h`, `tile_w`, `overlap` parameters  
- `_save_tile_to_storage()`: Added `tile_h`, `tile_w`, `overlap` parameters

#### Merge Functions
- `_merge_two_tiles()`: Added `tile_h`, `tile_w` parameters
- All function calls updated to pass required parameters

## Integration Points Fixed

### 1. File Path Management
- **Before**: Expected `{row}_{col}.npz` files
- **After**: Handles `{y_start}_{x_start}.npz` files correctly
- **Conversion**: Automatic mapping between coordinate systems

### 2. Tile Relationship Determination
- **Before**: Used pixel coordinates for relationship logic (incorrect)
- **After**: Uses tile index coordinates for relationship determination (correct)
- **Logic**: Maintains proper `tile1_left_of_tile2` etc. relationships

### 3. Enhanced 3-Step Merging Integration
- **Compatibility**: Full compatibility with enhanced `merge_tiles_cpu_3step` function
- **File Paths**: Correct file paths passed to merging algorithm
- **Mapping**: Proper coordinate mapping throughout pipeline

## Coordinate Conversion Examples

### Example 1: Standard Tiling
```
Tile dimensions: 1024x1024, overlap: 64
Stride: 960x960

Tile Index -> Pixel Coordinate
(0, 0) -> (0, 0)        # First tile
(0, 1) -> (0, 960)      # Second tile in first row  
(1, 0) -> (960, 0)      # First tile in second row
(1, 1) -> (960, 960)    # Second tile in second row
```

### Example 2: Smaller Tiles
```
Tile dimensions: 512x512, overlap: 32
Stride: 480x480

Tile Index -> Pixel Coordinate -> Filename
(0, 0) -> (0, 0) -> "0_0.npz"
(0, 1) -> (0, 480) -> "0_480.npz"
(1, 0) -> (480, 0) -> "480_0.npz"
(1, 1) -> (480, 480) -> "480_480.npz"
```

## Usage Examples

### Discovering Existing Tiles
```python
from code.nuclei_segmentation.cellpose_merge.two_phase_merge import discover_tile_coordinates_from_files

# Discover tiles from segmentation output
coords = discover_tile_coordinates_from_files(
    tile_masks_dir=Path("results/masks/tile_masks_npz"),
    tile_h=1024,
    tile_w=1024,
    overlap=64
)
```

### Running Enhanced Merge
```python
from code.nuclei_segmentation.cellpose_merge.two_phase_merge import merge_tiles_two_phase

# Run merge with discovered coordinates
merged_mask = merge_tiles_two_phase(
    coords=coords,
    height=image_height,
    width=image_width,
    tile_h=1024,
    tile_w=1024,
    overlap=64,
    debug_mode=True,
    output_dir=Path("results")
)
```

## Testing Results

All functionality thoroughly tested and verified:

### ✅ **Coordinate Conversion**
- Bidirectional conversion between tile indices and pixel coordinates
- Handles various tile sizes and overlap values
- Round-trip conversion accuracy verified

### ✅ **File Discovery**
- Correctly parses segmentation output filenames
- Handles realistic pixel coordinate naming
- Returns proper tile index coordinates

### ✅ **File Management**
- File copying with coordinate mapping working
- Proper error handling for missing files
- Maintains file content integrity

### ✅ **Complete Integration**
- Full pipeline integration with enhanced 3-step merging
- Proper tile relationship determination
- Successful merge completion with realistic data

## Performance Impact

- **Minimal Overhead**: Coordinate conversion adds negligible computation time
- **Memory Efficient**: No additional memory usage for coordinate mapping
- **Scalable**: Handles any number of tiles efficiently
- **Robust**: Comprehensive error handling and validation

## Breaking Changes

### Function Signatures
Several functions now require additional parameters:
- `tile_h`, `tile_w`, `overlap` parameters added to storage functions
- All calls to these functions must be updated

### File Discovery
New workflow for discovering existing tiles:
```python
# OLD: Manual coordinate specification
coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

# NEW: Automatic discovery from files
coords = discover_tile_coordinates_from_files(tile_dir, tile_h, tile_w, overlap)
```

## Scientific Benefits

### Accurate Pipeline Integration
- **Seamless Integration**: Merge pipeline now works with actual segmentation output
- **No Data Loss**: All segmentation results properly processed
- **Consistent Results**: Reliable merging regardless of tiling parameters

### Kidney I/R Injury Analysis
- **Complete Workflow**: End-to-end pipeline from segmentation to merged results
- **Flexible Tiling**: Supports various tile sizes and overlap configurations
- **Robust Processing**: Handles real-world segmentation output reliably

## Critical Pipeline Fix

🔧 **PIPELINE INTEGRATION FIXED!**

The coordinate mapping fix resolves the critical integration issue between segmentation and merging processes. The enhanced two-phase merge pipeline now works seamlessly with actual segmentation output files, enabling complete end-to-end processing for kidney I/R injury spatial multiomics analysis.

---

**Author**: Christos Botos  
**Date**: 2025-07-22  
**Context**: Kidney I/R Injury Spatial Multiomics Analysis Project

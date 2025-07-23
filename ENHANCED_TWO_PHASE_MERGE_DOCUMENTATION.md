# Enhanced Two-Phase Merge Implementation Documentation

## Overview

The `code/nuclei_segmentation/cellpose_merge/two_phase_merge.py` script has been comprehensively updated to implement proper file path management and nucleus ID conflict resolution for the enhanced 3-step merging algorithm with Step 3 (Cleanup).

## Key Enhancements

### 1. File Path Management

#### Directory Structure
- **Source**: `results/masks/tile_masks_npz/` (original tile masks)
- **Target**: `results/masks/merged_tile_masks_npz/` (working copies for merging)

#### Implementation
```python
def copy_tile_masks_to_merged_directory(
    source_dir: Path, 
    target_dir: Path, 
    coords: List[TileCoord]
) -> None:
```

**Features:**
- Copies all tile masks from source to target directory before merging
- Creates target directory if it doesn't exist
- Preserves original tile masks unchanged
- Handles file copying errors gracefully
- Progress tracking with tqdm

### 2. Nucleus ID Conflict Resolution

#### Problem Addressed
When tiles are processed independently, they may contain conflicting nucleus IDs (e.g., multiple tiles with nucleus ID=1), leading to incorrect merging decisions.

#### Solution
```python
def reassign_nucleus_ids(
    tile1_mask: NDArray[np.uint32], 
    tile2_mask: NDArray[np.uint32], 
    starting_id: int
) -> int:
```

**Features:**
- Reassigns all nucleus IDs to sequential integers starting from given number
- Processes both tiles in-place to apply new ID assignments
- Maintains spatial structure while ensuring unique IDs
- Returns next available ID for subsequent tile pairs
- Preserves background pixels (ID=0) unchanged

### 3. Enhanced Function Integration

#### Updated `_merge_two_tiles` Function

**New Signature:**
```python
def _merge_two_tiles(
    coord1: TileCoord,
    coord2: TileCoord,
    overlap_slices: Tuple[slice, slice, slice, slice],
    storage_dir: Path,
    overlap_length: int,
    use_gpu: bool = True,
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]:
```

**Key Changes:**
- Works with tile coordinates instead of loaded masks
- Uses file paths from `merged_tile_masks_npz/` directory
- Integrates with enhanced `merge_tiles_cpu_3step` function
- Automatic tile relationship determination
- Proper error handling and logging

#### Tile Relationship Mapping
```python
# Automatic relationship determination
if r1 == r2:  # Same row - horizontal relationship
    if c1 < c2:
        tile_relationship = "tile1_left_of_tile2"
    else:
        tile_relationship = "tile1_right_of_tile2"
elif c1 == c2:  # Same column - vertical relationship
    if r1 < r2:
        tile_relationship = "tile1_above_tile2"
    else:
        tile_relationship = "tile1_below_tile2"
```

### 4. Enhanced Two-Phase Workflow

#### Phase 0: File Management and ID Reassignment (NEW)
1. **File Copying**: Copy all tiles from `tile_masks_npz/` to `merged_tile_masks_npz/`
2. **ID Reassignment**: Reassign nucleus IDs to prevent conflicts across all tiles

#### Phase 1: Vertical Overlaps (Enhanced)
- Process horizontally adjacent tiles
- Use enhanced `merge_tiles_cpu_3step` with proper cleanup
- Save intermediate results to `merged_tile_masks_npz/`

#### Phase 2: Horizontal Overlaps (Enhanced)
- Process vertically adjacent tiles
- Load from `merged_tile_masks_npz/` (includes Phase 1 updates)
- Apply enhanced merging with Step 3 cleanup

#### Phase 3: Final Assembly (Updated)
- Combine all tiles into final merged mask
- Use priority-based assembly to handle conflicts
- Load efficiency calculation from original source files

## Enhanced 3-Step Algorithm Integration

### Step 1: Priority Selection
- Tile with more nuclei gets priority
- Automatic determination based on nucleus counts

### Step 2: Border Deletion
- Remove priority tile nuclei touching specific tile borders
- Uses `overlap_length=0` with appropriate direction

### Step 3: Cross-boundary Preservation
- Preserve non-priority nuclei extending into overlap region
- These nuclei touch the internal boundary line

### Step 4: Cleanup (NEW)
- Delete non-priority nuclei completely in overlap region
- Preserve cross-boundary nuclei to prevent fragmentation
- Proper cleanup of overlap region artifacts

## Function Signature Changes

### Updated `merge_tiles_two_phase`
```python
def merge_tiles_two_phase(
    coords: List[TileCoord],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    use_gpu: bool = True,
    merge_batch_size: int = 4,
    debug_mode: bool = False,
    output_dir: Optional[Path] = None,
) -> NDArray[np.uint32]:
```

**Removed Parameters:**
- `loader`: No longer needed as function works with stored files

**Enhanced Features:**
- Automatic file path management
- Nucleus ID conflict resolution
- Enhanced 3-step merging with cleanup
- Comprehensive logging and debugging

## Usage Examples

### Basic Usage
```python
from code.nuclei_segmentation.cellpose_merge.two_phase_merge import merge_tiles_two_phase

# Enhanced two-phase merge with automatic file management
merged_mask = merge_tiles_two_phase(
    coords=[(0, 0), (0, 1), (1, 0), (1, 1)],
    height=2000,
    width=2000,
    tile_h=1024,
    tile_w=1024,
    overlap=64,
    debug_mode=True,
    output_dir=Path("results")
)
```

### Directory Structure
```
results/
├── masks/
│   ├── tile_masks_npz/          # Original tiles (preserved)
│   │   ├── 0_0.npz
│   │   ├── 0_1.npz
│   │   └── ...
│   └── merged_tile_masks_npz/   # Working copies (modified during merge)
│       ├── 0_0.npz
│       ├── 0_1.npz
│       └── ...
```

## Scientific Benefits

### Accurate Nucleus Merging
- **Prevents ID conflicts** between independently processed tiles
- **Eliminates overlap region artifacts** through proper cleanup
- **Maintains biological structure integrity** during merging

### Kidney I/R Injury Analysis
- **Reliable cell counting** across tile boundaries
- **Consistent spatial relationships** preserved
- **Accurate damage assessment** regardless of tiling strategy
- **Reproducible results** across different processing runs

## Testing Results

All functionality has been thoroughly tested and verified:

### ✅ **File Management**
- File copying from source to target directory working correctly
- Directory creation and error handling functional
- Progress tracking and logging operational

### ✅ **Nucleus ID Conflict Resolution**
- ID reassignment prevents conflicts between tiles
- Sequential ID assignment working correctly
- Background preservation maintained

### ✅ **Enhanced 3-Step Merging**
- Integration with `merge_tiles_cpu_3step` functional
- Step 3 (Cleanup) properly implemented
- Cross-boundary nucleus preservation working

### ✅ **Complete Workflow**
- Two-phase merge with all enhancements operational
- Final assembly producing correct results
- Debug logging and efficiency calculation working

## Performance Impact

- **Minimal Overhead**: File copying adds negligible time compared to merging operations
- **Memory Efficient**: Works with stored files rather than keeping all tiles in memory
- **Scalable**: Handles large numbers of tiles efficiently
- **Robust**: Comprehensive error handling prevents pipeline failures

## Breaking Changes

### Function Signature
- `merge_tiles_two_phase` no longer requires `loader` parameter
- Function now expects tiles to exist in `tile_masks_npz/` directory

### Directory Requirements
- Requires `results/masks/tile_masks_npz/` directory with original tiles
- Creates `results/masks/merged_tile_masks_npz/` for working copies

---

**Author**: Christos Botos  
**Date**: 2025-07-22  
**Context**: Kidney I/R Injury Spatial Multiomics Analysis Project

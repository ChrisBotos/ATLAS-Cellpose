# Enhanced 3-Step Merging Function Documentation

## Overview

The `merge_tiles_cpu_3step` function implements a complete 3-step merging algorithm that properly utilizes the enhanced `_find_border_touching_nuclei` function for accurate nucleus merging in kidney I/R injury spatial analysis.

## Function Signature

```python
def merge_tiles_cpu_3step(
    tile1_path: Union[str, Path],
    tile2_path: Union[str, Path],
    overlap_length: int,
    tile_relationship: str,
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]:
```

## Parameters

### Input Parameters

- **`tile1_path`** (Union[str, Path]): Path to the first whole tile mask .npz file
- **`tile2_path`** (Union[str, Path]): Path to the second whole tile mask .npz file  
- **`overlap_length`** (int): Overlap distance in pixels between the tiles
- **`tile_relationship`** (str): Spatial relationship between tiles

### Tile Relationships

The `tile_relationship` parameter must be one of:

- **`"tile1_above_tile2"`**: tile1 is positioned above tile2 (vertical overlap)
- **`"tile1_left_of_tile2"`**: tile1 is positioned to the left of tile2 (horizontal overlap)
- **`"tile1_below_tile2"`**: tile1 is positioned below tile2 (vertical overlap)
- **`"tile1_right_of_tile2"`**: tile1 is positioned to the right of tile2 (horizontal overlap)

## Return Values

Returns a tuple containing:

1. **`updated_tile1_mask`** (NDArray[np.uint32]): Updated tile1 mask after merging
2. **`updated_tile2_mask`** (NDArray[np.uint32]): Updated tile2 mask after merging
3. **`mapping`** (Dict[int, int]): Mapping of preserved cross-boundary nucleus IDs

## Implementation Steps

### 1. Load Complete Tile Masks
- Loads full tile masks from .npz files (not just overlap patches)
- Validates file existence and format
- Converts masks to uint32 format

### 2. Priority Selection
- Counts nuclei in each tile using `_count_nuclei_in_tile()`
- Assigns priority to tile with higher nucleus count
- Logs priority decision for scientific tracking

### 3. Direction Mapping
Maps tile relationships to boundary directions:

| Tile Relationship | Tile1 Direction | Tile2 Direction |
|-------------------|-----------------|-----------------|
| tile1_above_tile2 | down | up |
| tile1_left_of_tile2 | right | left |
| tile1_below_tile2 | up | down |
| tile1_right_of_tile2 | left | right |

### 4. Enhanced Border Detection

#### Priority Tile (Original Mode)
```python
priority_border_nuclei = _find_border_touching_nuclei(priority_tile_mask)
```
- Uses original mode (no overlap_length parameter)
- Detects nuclei touching tile's outer borders
- These nuclei will be deleted (Step 2: Border Deletion)

#### Non-Priority Tile (Internal Boundary Mode)
```python
non_priority_overlap_nuclei = _find_border_touching_nuclei(
    non_priority_tile_mask, 
    overlap_length, 
    non_priority_direction
)
```
- Uses enhanced mode with overlap_length and direction
- Detects nuclei extending into overlap region
- These nuclei will be preserved (Step 3: Cross-boundary Preservation)

### 5. Apply 3-Step Merging Rules

#### Step 2: Border Deletion
- Removes all priority tile nuclei that touch the tile's outer borders
- Prevents border artifacts in merged results
- Logs deletion count for scientific tracking

#### Step 3: Cross-boundary Preservation
- Preserves all non-priority nuclei extending into overlap region
- These are considered cross-boundary nuclei
- Maintains original nucleus IDs in mapping dictionary

#### Step 4: Cleanup
- Handled by higher-level merging logic
- Function works with complete tiles rather than overlap patches

## Usage Examples

### Basic Usage
```python
from code.nuclei_segmentation.cellpose_merge.rules import merge_tiles_cpu_3step

# Merge two horizontally adjacent tiles
updated_tile1, updated_tile2, mapping = merge_tiles_cpu_3step(
    tile1_path="path/to/tile1.npz",
    tile2_path="path/to/tile2.npz", 
    overlap_length=64,
    tile_relationship="tile1_left_of_tile2"
)
```

### Vertical Overlap
```python
# Merge two vertically adjacent tiles
updated_tile1, updated_tile2, mapping = merge_tiles_cpu_3step(
    tile1_path="path/to/top_tile.npz",
    tile2_path="path/to/bottom_tile.npz",
    overlap_length=32,
    tile_relationship="tile1_above_tile2"
)
```

### Processing Results
```python
# Analyze merging results
print(f"Cross-boundary nuclei preserved: {len(mapping)}")
for original_id, preserved_id in mapping.items():
    print(f"Nucleus {original_id} preserved as {preserved_id}")

# Count final nuclei
final_tile1_nuclei = len(np.unique(updated_tile1[updated_tile1 > 0]))
final_tile2_nuclei = len(np.unique(updated_tile2[updated_tile2 > 0]))
```

## Key Features

### ✅ **Complete Workflow**
- Handles entire tile loading and merging process
- No need for manual overlap region extraction
- Automatic priority selection and direction mapping

### ✅ **Enhanced Boundary Detection**
- Uses original mode for priority tile border detection
- Uses internal boundary mode for non-priority overlap detection
- Proper ±1 pixel buffer zone handling

### ✅ **Scientific Accuracy**
- Preserves cross-boundary nuclei to prevent fragmentation
- Removes border artifacts systematically
- Maintains spatial integrity of biological structures

### ✅ **Robust Error Handling**
- Validates file existence and format
- Checks parameter validity
- Provides descriptive error messages

### ✅ **Comprehensive Logging**
- Info-level logging for scientific relevance
- Debug-level logging for detailed tracking
- Step-by-step algorithm progress

## Error Handling

The function validates inputs and provides clear error messages:

- **FileNotFoundError**: When tile mask files don't exist
- **ValueError**: For invalid tile relationships or negative overlap lengths
- **Format errors**: For invalid .npz file formats

## Performance Considerations

- **Memory Efficient**: Works with complete tiles without creating large intermediate arrays
- **Fast Processing**: Direct numpy operations for nucleus detection and deletion
- **Scalable**: Handles tiles of any reasonable size

## Integration with Existing Codebase

The function integrates seamlessly with the existing merging pipeline:
- Uses established `_find_border_touching_nuclei` function
- Compatible with existing .npz tile format
- Follows established logging and error handling patterns
- Maintains backward compatibility with existing workflows

## Scientific Context

This enhanced merging function is essential for accurate kidney I/R injury spatial analysis:
- **Prevents nucleus fragmentation** across tile boundaries
- **Maintains biological structure integrity** during merging
- **Supports priority-based merging decisions** for optimal results
- **Enables accurate spatial analysis** of tissue damage and repair

---

**Author**: Christos Botos  
**Date**: 2025-07-22  
**Context**: Kidney I/R Injury Spatial Multiomics Analysis Project

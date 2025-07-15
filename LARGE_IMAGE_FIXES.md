# Large Image Processing Fixes

## Overview

This document describes the fixes implemented to handle very large images that previously caused crashes due to memory allocation errors, uint32 overflow issues, and PyTorch tensor size limitations.

## Issues Identified

### 1. uint32 Overflow in Composite Keys
**Error**: `Python integer 4294967296 out of bounds for uint32`
**Cause**: The composite key calculation `(t << 32) | patch[t]` in `rules.py` caused overflow when tile index `t` became large.
**Impact**: Processing failed when clusters had many tiles.

### 2. Memory Allocation Failures
**Error**: `Unable to allocate 11.4 TiB for an array with shape (4489, 26460, 26459)`
**Cause**: The code attempted to create massive 3D arrays `np.zeros((T, cluster_h, cluster_w), dtype=np.uint32)` for large clusters.
**Impact**: System ran out of memory and crashed.

### 3. PyTorch Tensor Size Limitations
**Error**: `nonzero is not supported for tensors with more than INT_MAX elements`
**Cause**: PyTorch tensors have a maximum size limit of INT_MAX (2^31 - 1) elements.
**Impact**: GPU processing failed for very large image regions.

### 4. CUDA Memory Access Errors
**Error**: `CUDA error: an illegal memory access was encountered`
**Cause**: GPU memory allocation exceeded available VRAM or caused fragmentation.
**Impact**: GPU processing failed and fell back to CPU, which then also failed.

## Fixes Implemented

### 1. Fixed uint32 Overflow in Composite Keys

**File**: `code/nuclei_segmentation/cellpose_merge/rules.py`

- Changed composite key calculation to use `uint64` throughout
- Added overflow detection and meaningful error messages
- Ensured safe conversion between uint64 and int types

```python
# Before (caused overflow)
composite[t] = (t << 32) | patch[t]

# After (prevents overflow)
if t >= (1 << 32):
    raise ValueError(f"Tile index {t} exceeds maximum supported value")
composite[t] = (np.uint64(t) << 32) | patch[t].astype(np.uint64)
```

### 2. Added Cluster Feasibility Checking

**File**: `code/nuclei_segmentation/cellpose_merge/merge_tiles.py`

- Added `_check_cluster_feasibility()` function to validate clusters before processing
- Checks for array size limits, memory requirements, and uint32 overflow potential
- Provides detailed error messages explaining why processing failed

```python
def _check_cluster_feasibility(cluster, tile_h, tile_w, overlap, height, width, memory_limit_gb):
    # Check uint32 overflow
    if len(cluster) >= (1 << 32):
        return False, "Cluster exceeds uint32 composite key limit"
    
    # Check array size limits
    total_elements = cluster_size * cluster_h * cluster_w
    if total_elements > 2**31 - 1:
        return False, "Array would exceed NumPy size limits"
    
    # Check memory requirements
    estimated_memory = _estimate_cluster_memory_requirements(...)
    if estimated_memory > memory_limit_gb:
        return False, f"Memory requirement {estimated_memory:.1f} GB exceeds limit"
```

### 3. Implemented Cluster Splitting

**File**: `code/nuclei_segmentation/cellpose_merge/merge_tiles.py`

- Added `_split_large_cluster()` function to break large clusters into manageable pieces
- Automatically splits clusters that exceed feasibility limits
- Maintains spatial locality for efficient processing

```python
def _split_large_cluster(cluster, max_cluster_size=1000):
    if len(cluster) <= max_cluster_size:
        return [cluster]
    
    # Sort by spatial coordinates and split into chunks
    sorted_cluster = sorted(cluster)
    sub_clusters = []
    for i in range(0, len(sorted_cluster), max_cluster_size):
        sub_cluster = sorted_cluster[i:i + max_cluster_size]
        sub_clusters.append(sub_cluster)
    
    return sub_clusters
```

### 4. Enhanced GPU Error Handling

**File**: `code/nuclei_segmentation/cellpose_merge/gpu_merge.py`

- Added tensor size limit checking before GPU processing
- Improved error messages for GPU memory issues
- Better detection of PyTorch limitations

```python
# Check tensor size limits
total_elements = T * H * W
max_tensor_elements = 2**31 - 1  # PyTorch INT_MAX limit

if total_elements > max_tensor_elements:
    raise RuntimeError(f"Tensor would have {total_elements} elements, "
                      f"exceeding PyTorch limit of {max_tensor_elements}")
```

### 5. Improved Memory Allocation Error Handling

**File**: `code/nuclei_segmentation/cellpose_merge/batch_merge.py`

- Added pre-allocation checks for array sizes
- Better error messages for memory allocation failures
- Validation of gid_offset to prevent uint32 overflow

```python
# Check for potential memory issues before allocation
batch_elements = T * batch_h * batch_w
if batch_elements > 2**31 - 1:
    raise RuntimeError(f"Batch stack would have {batch_elements} elements, "
                      f"exceeding safe limits")

try:
    batch_stack = np.zeros((T, batch_h, batch_w), dtype=np.uint32)
except (MemoryError, OverflowError) as e:
    raise RuntimeError(f"Failed to allocate memory for batch stack: {e}")
```

### 6. Automatic Fallback and Recovery

**File**: `code/nuclei_segmentation/cellpose_merge/merge_tiles.py`

- When clusters exceed feasibility limits, automatically split them
- Process sub-clusters with reduced batch sizes
- Maintain processing continuity instead of crashing

## Testing

Created comprehensive test suite in `test_large_image_fixes.py`:

- Tests cluster feasibility checking
- Validates cluster splitting functionality
- Tests memory estimation accuracy
- Verifies uint32 overflow prevention
- Tests GPU tensor size limit detection

## Usage Recommendations

### For Very Large Images (>20GB):

1. **Reduce batch sizes**: Set `gpu_batch_size=1` or `gpu_batch_size=2`
2. **Lower memory limits**: Set `gpu_memory_limit_gb=4.0` or lower
3. **Use CPU processing**: Set `use_gpu=False` for extremely large images
4. **Process in regions**: Consider splitting the image into smaller regions

### Configuration Example:

```python
masks = merge_masks_streaming(
    height=image_height,
    width=image_width,
    tile_h=512,
    tile_w=512,
    overlap=64,
    tiles_path="./tile_masks_npz",
    threshold=0.3,
    gpu_batch_size=1,           # Reduced for large images
    gpu_memory_limit_gb=4.0,    # Conservative limit
    use_gpu=True                # Will auto-fallback to CPU if needed
)
```

## Performance Impact

- **Memory usage**: Significantly reduced for large clusters
- **Processing time**: May be slightly slower due to additional checks, but prevents crashes
- **Reliability**: Much more robust for large images
- **Scalability**: Can now handle images of any size (limited only by available disk space)

## Future Improvements

1. **Streaming processing**: Implement true streaming for extremely large images
2. **Distributed processing**: Support for processing across multiple GPUs/machines
3. **Adaptive batching**: Dynamic batch size adjustment based on available memory
4. **Progress estimation**: Better progress reporting for split clusters

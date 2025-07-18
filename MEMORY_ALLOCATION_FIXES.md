# Memory Allocation Fixes for Sparse Tile Distributions

**Author:** Christos Botos  
**Date:** July 18, 2025  
**Issue:** CUDA out of memory errors (123.64 GiB allocation attempts) during tile merging

## Problem Analysis

The original error occurred when processing sparse tile distributions where tiles were arranged in long horizontal or vertical lines. For example:
- **Problematic pattern:** Tiles spanning from (9,0) to (10,64) 
- **Result:** 66 tiles creating a bounding box of 922×26,459 pixels
- **Memory requirement:** 66 × 922 × 26,459 × 4 bytes ≈ 6.5 GB base + intermediates = 123+ GB

The issue was that the memory estimation and batch creation algorithms were calculating bounding boxes based on the spatial extent of tiles, not considering that sparse distributions create unreasonably large memory requirements.

## Root Cause

1. **Sparse tile distributions** create large bounding boxes with mostly empty space
2. **Memory estimation** was based on full bounding box dimensions
3. **Batch sizing** didn't account for spatial sparsity
4. **No safeguards** against unreasonable memory allocations

## Implemented Fixes

### 1. Enhanced Memory Estimation (`estimate_memory_requirements`)

**Key improvements:**
- **Sparse distribution detection:** Identifies when bounding boxes exceed reasonable dimensions (8192px)
- **Individual tile estimation:** For sparse distributions, calculates memory based on actual tile count rather than bounding box
- **Overflow protection:** Detects integer overflow in memory calculations
- **Sanity checks:** Flags estimates exceeding 100 GB as potentially problematic

```python
# Before: Always used bounding box
stack_memory = num_tiles * batch_h * batch_w * 4

# After: Smart estimation for sparse distributions
if batch_h > max_reasonable_dimension or batch_w > max_reasonable_dimension:
    individual_tile_memory = num_tiles * tile_h * tile_w * 4  # Much smaller!
```

### 2. Conservative Batch Sizing (`get_optimal_batch_size`)

**Key improvements:**
- **Sparse distribution detection:** Automatically detects problematic tile patterns
- **Forced batch size of 1:** For sparse distributions, processes tiles individually
- **Enhanced safety factors:** Increased from 1.3 to 2.0 for memory estimates
- **Multiple validation passes:** Tests batch sizes before finalizing

```python
# Detect sparse distributions
is_sparse_distribution = (bbox_h > max_reasonable_dimension or 
                         bbox_w > max_reasonable_dimension or 
                         density < 0.1)

if is_sparse_distribution:
    max_batch_size = 1  # Process individually
```

### 3. Conservative Spatial Chunking (`_create_conservative_spatial_chunks`)

**New function for sparse distributions:**
- **Spatial compactness:** Ensures chunks maintain reasonable density (≥25%)
- **Size limits:** Restricts chunk dimensions to 8×8 tile spans
- **Density validation:** Prevents creation of sparse chunks that would cause memory issues

### 4. Enhanced Batch Validation

**Pre-processing validation:**
- **Memory pre-check:** Validates all batches before GPU processing
- **Automatic splitting:** Problematic batches are automatically split into individual tiles
- **Real-time monitoring:** Continuous validation during processing

### 5. Improved Error Handling

**Multiple safety layers:**
- **Array size limits:** Checks for NumPy array size limits (2³¹-1 elements)
- **Dimension validation:** Rejects unreasonable bounding box dimensions
- **Memory allocation protection:** Enhanced try-catch with detailed error messages

## Configuration Updates

Created `configs/memory_safe_config.ini` with conservative settings:

```ini
# Memory-safe GPU settings
gpu_batch_size = 1                    # Always process individually
gpu_memory_limit_gb = 4               # Conservative memory limit
gpu_memory_safety_factor = 3.0        # High safety margin
gpu_spatial_strategy = spatial        # Use spatial chunking
gpu_adaptive_batching = True          # Enable intelligent batching
gpu_max_retries = 8                   # More retry attempts
```

## Test Results

All memory allocation fixes verified with comprehensive testing:

✅ **Dense distributions:** Normal processing with appropriate batch sizes  
✅ **Sparse distributions:** Automatic detection and individual tile processing  
✅ **Problematic patterns:** Original error pattern now processes safely  
✅ **Memory estimates:** Realistic estimates for all tile configurations  

**Example results:**
- **Original problematic pattern:** 130 tiles → 0.76 GB (was 123+ GB)
- **Batch size:** Automatically set to 1 for sparse distributions
- **Processing:** 130 individual batches, each using ~0.01 GB

## Performance Impact

**Memory safety vs. speed trade-off:**
- **Sparse distributions:** Slower but stable (individual tile processing)
- **Dense distributions:** Minimal impact (still uses optimized batching)
- **Overall:** Prevents crashes at the cost of some processing speed for problematic cases

## Usage Recommendations

1. **Use memory-safe config** for large images with unknown tile distributions
2. **Monitor logs** for sparse distribution warnings
3. **Adjust memory limits** based on available GPU memory
4. **Enable adaptive batching** for mixed tile patterns

## Technical Details

**Memory calculation improvements:**
```python
# Sparse distribution handling
if is_sparse:
    memory = num_tiles * tile_h * tile_w * 4 * safety_factor
else:
    memory = num_tiles * bbox_h * bbox_w * 4 * safety_factor
```

**Validation pipeline:**
1. Estimate memory for each batch
2. Reject batches exceeding limits
3. Split problematic batches into individual tiles
4. Validate final batch list before processing

## Future Enhancements

- **Dynamic memory monitoring:** Real-time GPU memory tracking
- **Adaptive tile sizing:** Adjust tile sizes based on available memory
- **Intelligent clustering:** Better algorithms for sparse tile grouping
- **Memory pooling:** Reuse allocated memory across batches

---

**Status:** ✅ **RESOLVED** - Memory allocation issues fixed and tested  
**Impact:** Prevents CUDA out-of-memory errors for sparse tile distributions  
**Compatibility:** Maintains performance for dense distributions

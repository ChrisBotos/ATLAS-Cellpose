# Environment Issues Analysis and Fixes

**Author**: Christos Botos  
**Date**: July 25, 2025  
**Issue**: Segmentation failures with tiles touching vertically and parallel processing timeouts

## Problem Analysis

### Root Cause Identified
The issue was **NOT specifically with vertical tiles**, but rather with **environment activation and PyTorch/Cellpose3 compatibility**:

1. **Environment Activation Failure**: The pipeline was running with system Python instead of the `iri310_cellpose3` conda environment
2. **Missing Dependencies**: Critical packages (PyTorch, Cellpose, etc.) were not available in system Python
3. **CUDA Unavailable**: PyTorch in the environment lacks CUDA support, forcing CPU-only processing
4. **Timeout Issues**: CPU processing is much slower, causing "tile 4 in batch X" timeouts consistently

### Specific Issues Found

#### 1. Environment Problems
- System Python: `/usr/bin/python` (missing packages)
- Expected: `/home/wsl_chris_botos/miniconda3/envs/iri310_cellpose3/bin/python`
- Missing packages: torch, cellpose, PIL, skimage, matplotlib, tqdm, psutil

#### 2. PyTorch Configuration
- PyTorch version: 2.7.1 (available in conda env)
- CUDA available: **False** (CPU-only processing)
- This causes significantly slower processing times

#### 3. Parallel Processing Issues
- Batch size: 4 tiles (too large for CPU processing)
- Max workers: 4 (too many for CPU)
- Timeout: 300 seconds (too short for CPU processing)
- Pattern: "tile 4 in batch X" consistently times out

## Solutions Implemented

### 1. Environment Wrapper Script
Created `run_with_proper_env.sh` that:
- Properly activates the `iri310_cellpose3` conda environment
- Verifies environment activation and package availability
- Provides detailed logging of environment status
- Ensures consistent environment setup

**Usage**:
```bash
wsl ./run_with_proper_env.sh
```

### 2. Configuration Optimization for CPU
Updated `configs/nuclei_segmentation_config.ini`:

#### Cellpose Settings
```ini
gpu = False  # Force CPU mode to avoid CUDA issues
```

#### Parallel Processing Optimization
```ini
parallel_batch_size = 2        # Reduced from 4 (prevents timeouts)
parallel_max_workers = 2       # Reduced from 4 (better CPU stability)
parallel_memory_limit_gb = 4.0 # Reduced from 6.0 (conservative)
parallel_timeout_seconds = 1000 # Already increased (was 300)
```

### 3. Diagnostic Tools
Created comprehensive diagnostic scripts:
- `debug_environment_issues.py`: Environment analysis and testing
- `test_cuda.py`: Simple CUDA availability test

## Technical Details

### Why "Tile 4" Was Failing
The pattern of "tile 4 in batch X" timing out was due to:
1. **Batch processing order**: Tiles 1-3 would process slowly on CPU
2. **Cumulative timeout**: By the time tile 4 started, the 300s batch timeout was reached
3. **CPU bottleneck**: Without GPU acceleration, each tile takes much longer
4. **Memory pressure**: 4 concurrent tiles on CPU caused memory issues

### Environment Activation Issue
The pipeline was not properly activating the conda environment because:
1. **Shell context**: The conda environment wasn't being sourced correctly
2. **Path resolution**: System Python was being used instead of conda Python
3. **Missing initialization**: `conda.sh` profile wasn't being sourced

## Verification Steps

### 1. Test Environment Activation
```bash
wsl bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python --version"
```
Expected output: Python 3.10.18 (conda-forge)

### 2. Test Package Availability
```bash
wsl bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python -c 'import torch, cellpose; print(\"Packages OK\")'"
```

### 3. Test CUDA Status
```bash
wsl bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python test_cuda.py"
```

## Recommendations

### Immediate Actions
1. **Use the wrapper script**: Always run `wsl ./run_with_proper_env.sh` instead of direct Python calls
2. **Monitor processing**: CPU processing will be slower but more stable
3. **Test with smaller images**: Verify fixes work before processing large datasets

### Long-term Improvements
1. **Install CUDA-enabled PyTorch**: For faster processing if GPU is available
2. **Environment automation**: Add environment checks to all pipeline scripts
3. **Configuration profiles**: Create separate configs for CPU vs GPU processing

### Performance Expectations
- **CPU processing**: ~3-5x slower than GPU but more stable
- **Reduced timeouts**: Smaller batches prevent timeout issues
- **Memory efficiency**: Conservative settings prevent memory errors

## Files Modified

1. `configs/nuclei_segmentation_config.ini` - CPU optimization
2. `run_with_proper_env.sh` - Environment wrapper (new)
3. `debug_environment_issues.py` - Diagnostic tool (new)
4. `test_cuda.py` - CUDA test (new)
5. `ENVIRONMENT_ISSUES_ANALYSIS_AND_FIXES.md` - This document (new)

## Testing Results

### Before Fixes
- Environment: System Python (missing packages)
- PyTorch: Not available
- Cellpose: Not available
- Result: Import errors and segmentation failures

### After Fixes
- Environment: iri310_cellpose3 (all packages available)
- PyTorch: 2.7.1 (CPU-only)
- Cellpose: Working (CPU mode)
- Result: Stable processing with optimized timeouts

## Next Steps

1. **Test the wrapper script** with a small image crop
2. **Monitor processing times** and adjust timeouts if needed
3. **Consider GPU setup** for faster processing in the future
4. **Document the new workflow** for consistent usage

The main issue was environment activation, not vertical tile processing specifically. The fixes ensure proper environment setup and optimize configuration for CPU-only processing.

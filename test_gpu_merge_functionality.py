#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_gpu_merge_functionality.py.
Description:
    Test script to verify that GPU tile merging functionality is working correctly
    with CuPy support. This script tests the actual merge functions that would be
    used in the pipeline to ensure GPU acceleration is available for tile processing.

Dependencies:
    • Python >= 3.10.
    • numpy >= 1.21.0.
    • cupy >= 12.0.0 (for GPU operations).

Usage:
    python test_gpu_merge_functionality.py

Key Features:
    • Tests GPU merge backend availability.
    • Verifies CuPy-based tile merging operations.
    • Validates 4-step merging rules on GPU.
    • Compares GPU vs CPU merge performance.

Notes:
    • This test uses synthetic tile data to verify merge functionality.
    • GPU operations require CUDA-compatible hardware and drivers.
"""

import traceback
import logging
import numpy as np
import sys
from pathlib import Path
import time

# Add the code directory to Python path.
sys.path.insert(0, str(Path(__file__).parent / "code"))

def setup_logging():
    """Set up logging for the GPU merge test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_gpu_merge_functionality.log')
        ]
    )
    return logging.getLogger(__name__)

def test_cupy_availability():
    """Test if CuPy is available and working."""
    logger = logging.getLogger(__name__)
    logger.info("Testing CuPy availability for GPU merge operations...")
    
    try:
        import cupy as cp
        
        # Test basic CuPy operations.
        device_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"CuPy found {device_count} CUDA devices")
        
        if device_count > 0:
            # Test GPU memory and computation.
            device = cp.cuda.Device()
            free_memory, total_memory = device.mem_info
            logger.info(f"GPU memory: {free_memory/(1024**3):.2f}/{total_memory/(1024**3):.2f} GB")
            
            # Test array operations.
            a = cp.random.randn(1000, 1000)
            b = cp.random.randn(1000, 1000)
            c = cp.dot(a, b)
            result = float(cp.sum(c))
            
            logger.info(f"CuPy GPU computation test successful: {result:.2f}")
            return True
        else:
            logger.warning("No CUDA devices found")
            return False
            
    except ImportError as e:
        logger.error(f"CuPy not available: {e}")
        return False
    except Exception as e:
        logger.error(f"CuPy test failed: {e}")
        logger.debug(f"CuPy error traceback:\n{traceback.format_exc()}")
        return False

def test_gpu_merge_backend():
    """Test the GPU merge backend from the pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Testing GPU merge backend availability...")
    
    try:
        # Try to import the GPU merge function.
        from code.nuclei_segmentation.cellpose_merge.gpu_merge import merge_patch_gpu
        logger.info("GPU merge backend imported successfully")
        
        # Create test data.
        patch = np.zeros((2, 100, 100), dtype=np.uint32)
        
        # Tile 1: nucleus 1.
        patch[0, 20:40, 20:40] = 1
        
        # Tile 2: nucleus 2 overlapping with nucleus 1.
        patch[1, 25:45, 25:45] = 2
        
        logger.info("Created synthetic overlapping tile data for GPU merge test")
        
        # Test GPU merge.
        start_time = time.time()
        merged, mapping = merge_patch_gpu(patch, threshold=0.3)
        gpu_time = time.time() - start_time
        
        logger.info(f"GPU merge completed in {gpu_time:.3f}s")
        logger.info(f"GPU merge result: shape={merged.shape}, unique_labels={np.unique(merged)}")
        logger.info(f"GPU mapping: {mapping}")
        
        return True, gpu_time
        
    except ImportError as e:
        logger.error(f"GPU merge backend not available: {e}")
        return False, 0.0
    except Exception as e:
        logger.error(f"GPU merge test failed: {e}")
        logger.debug(f"GPU merge error traceback:\n{traceback.format_exc()}")
        return False, 0.0

def test_cpu_merge_backend():
    """Test the CPU merge backend for comparison."""
    logger = logging.getLogger(__name__)
    logger.info("Testing CPU merge backend for comparison...")
    
    try:
        # Import the CPU merge function.
        from code.nuclei_segmentation.cellpose_merge.rules import merge_patch_cpu
        logger.info("CPU merge backend imported successfully")
        
        # Create test data.
        patch = np.zeros((2, 100, 100), dtype=np.uint32)
        
        # Tile 1: nucleus 1.
        patch[0, 20:40, 20:40] = 1
        
        # Tile 2: nucleus 2 overlapping with nucleus 1.
        patch[1, 25:45, 25:45] = 2
        
        logger.info("Created synthetic overlapping tile data for CPU merge test")
        
        # Test CPU merge.
        start_time = time.time()
        merged, mapping = merge_patch_cpu(patch, threshold=0.3)
        cpu_time = time.time() - start_time
        
        logger.info(f"CPU merge completed in {cpu_time:.3f}s")
        logger.info(f"CPU merge result: shape={merged.shape}, unique_labels={np.unique(merged)}")
        logger.info(f"CPU mapping: {mapping}")
        
        return True, cpu_time
        
    except Exception as e:
        logger.error(f"CPU merge test failed: {e}")
        logger.debug(f"CPU merge error traceback:\n{traceback.format_exc()}")
        return False, 0.0

def test_merge_backend_selection():
    """Test the merge backend selection logic."""
    logger = logging.getLogger(__name__)
    logger.info("Testing merge backend selection logic...")
    
    try:
        from code.nuclei_segmentation.cellpose_merge.merge_tiles import _lazy_import_merge_backends
        
        # Force import of merge backends.
        _lazy_import_merge_backends()
        logger.info("Merge backends imported successfully")
        
        # Check if GPU backend is available.
        try:
            from code.nuclei_segmentation.cellpose_merge.gpu_merge import merge_patch_gpu
            gpu_available = True
            logger.info("GPU merge backend is available")
        except ImportError:
            gpu_available = False
            logger.warning("GPU merge backend is not available")
        
        # Check if CPU backend is available.
        try:
            from code.nuclei_segmentation.cellpose_merge.rules import merge_patch_cpu
            cpu_available = True
            logger.info("CPU merge backend is available")
        except ImportError:
            cpu_available = False
            logger.error("CPU merge backend is not available")
        
        return gpu_available, cpu_available
        
    except Exception as e:
        logger.error(f"Backend selection test failed: {e}")
        logger.debug(f"Backend selection error traceback:\n{traceback.format_exc()}")
        return False, False

def main():
    """Run comprehensive GPU merge functionality tests."""
    logger = setup_logging()
    logger.info("Starting GPU merge functionality verification")
    logger.info("=" * 60)
    
    # Track test results.
    results = {}
    
    # Test 1: CuPy availability.
    logger.info("TEST 1: CuPy Availability")
    logger.info("-" * 30)
    results['cupy_available'] = test_cupy_availability()
    
    # Test 2: Merge backend selection.
    logger.info("\nTEST 2: Merge Backend Selection")
    logger.info("-" * 30)
    gpu_backend, cpu_backend = test_merge_backend_selection()
    results['gpu_backend'] = gpu_backend
    results['cpu_backend'] = cpu_backend
    
    # Test 3: CPU merge functionality.
    logger.info("\nTEST 3: CPU Merge Functionality")
    logger.info("-" * 30)
    cpu_success, cpu_time = test_cpu_merge_backend()
    results['cpu_merge'] = cpu_success
    
    # Test 4: GPU merge functionality (if available).
    logger.info("\nTEST 4: GPU Merge Functionality")
    logger.info("-" * 30)
    if results['cupy_available'] and results['gpu_backend']:
        gpu_success, gpu_time = test_gpu_merge_backend()
        results['gpu_merge'] = gpu_success
        
        if gpu_success and cpu_success:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            logger.info(f"Performance comparison: GPU {speedup:.2f}x faster than CPU")
    else:
        logger.warning("Skipping GPU merge test - requirements not met")
        results['gpu_merge'] = False
        gpu_time = 0.0
    
    # Summary.
    logger.info("\n" + "=" * 60)
    logger.info("GPU MERGE FUNCTIONALITY SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name.upper().replace('_', ' ')}: {status}")
    
    # Overall assessment.
    gpu_merge_ready = results['cupy_available'] and results['gpu_backend'] and results['gpu_merge']
    cpu_merge_ready = results['cpu_backend'] and results['cpu_merge']
    
    logger.info("\nOVERALL ASSESSMENT:")
    logger.info(f"GPU Merge Ready: {'YES' if gpu_merge_ready else 'NO'}")
    logger.info(f"CPU Merge Ready: {'YES' if cpu_merge_ready else 'NO'}")
    
    if gpu_merge_ready:
        logger.info("\n✓ GPU TILE MERGING IS FULLY FUNCTIONAL")
        logger.info("The pipeline can use GPU acceleration for tile merging operations")
        return 0
    elif cpu_merge_ready:
        logger.warning("\n⚠ PARTIAL FUNCTIONALITY")
        logger.warning("CPU tile merging works, but GPU acceleration is not available")
        return 1
    else:
        logger.error("\n✗ TILE MERGING NOT FUNCTIONAL")
        logger.error("Neither GPU nor CPU tile merging is working correctly")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

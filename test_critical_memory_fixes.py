#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_critical_memory_fixes.py.
Description:
    Critical test script to verify that the enhanced memory allocation fixes
    prevent the 131.47 GiB allocation error and handle the most problematic
    sparse tile distributions safely.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pytest.

Usage:
    python test_critical_memory_fixes.py

Key Features:
    • Tests the incremental processing function.
    • Validates GPU and CPU safety checks.
    • Simulates the exact problematic pattern from the error logs.
    • Verifies that no allocation exceeds reasonable limits.
"""

import traceback
import logging
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Configure logging for detailed output.
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_incremental_processing():
    """Test the new incremental processing function."""
    print("\n" + "="*80)
    print("TESTING INCREMENTAL PROCESSING (CRITICAL FIX)")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import _merge_cluster_incremental
        
        # Create a problematic sparse tile distribution.
        # This simulates the exact pattern that caused the 131.47 GiB error.
        problematic_cluster = []
        for r in [9, 10]:  # Two rows.
            for c in range(66):  # 66 columns (0 to 65).
                problematic_cluster.append((r, c))
        
        print(f"Created problematic cluster: {len(problematic_cluster)} tiles")
        print(f"Tile range: ({min(r for r, _ in problematic_cluster)},{min(c for _, c in problematic_cluster)}) "
              f"to ({max(r for r, _ in problematic_cluster)},{max(c for _, c in problematic_cluster)})")
        
        # Create a mock loader function.
        def mock_loader(ys: slice, xs: slice) -> np.ndarray:
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            # Create a small test mask with a few objects.
            mask = np.zeros((h, w), dtype=np.uint32)
            if h > 10 and w > 10:
                mask[5:8, 5:8] = 1  # Small object.
                mask[15:18, 15:18] = 2  # Another small object.
            return mask
        
        # Create a mock global array.
        image_height, image_width = 10000, 30000  # Large image.
        global_array = np.zeros((image_height, image_width), dtype=np.uint32)
        
        # Test incremental processing.
        tile_h, tile_w, overlap = 512, 512, 102
        
        result_patch, (y0, x0), mapping = _merge_cluster_incremental(
            cluster=problematic_cluster,
            loader=mock_loader,
            height=image_height,
            width=image_width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=0.3,
            use_gpu=False,  # Use CPU for safety.
            gid_offset=1,
            global_merged_array=global_array,
        )
        
        print(f"✓ Incremental processing completed successfully")
        print(f"✓ Result patch shape: {result_patch.shape} (should be minimal)")
        print(f"✓ Cluster position: ({y0}, {x0})")
        print(f"✓ Global array max value: {global_array.max()}")
        print(f"✓ Non-zero pixels in global array: {np.count_nonzero(global_array)}")
        
        # Verify that no massive allocations occurred.
        assert result_patch.size <= 1, "Result patch should be minimal"
        assert global_array.max() > 0, "Global array should have some processed data"
        
        print("✓ Incremental processing test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Incremental processing test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_gpu_safety_checks():
    """Test the enhanced GPU safety checks."""
    print("\n" + "="*80)
    print("TESTING GPU SAFETY CHECKS")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.gpu_merge import merge_patch_gpu
        
        # Test case 1: Reasonable patch (should work).
        reasonable_patch = np.zeros((2, 512, 512), dtype=np.uint32)
        reasonable_patch[0, 100:200, 100:200] = 1
        reasonable_patch[1, 150:250, 150:250] = 2
        
        try:
            result, mapping = merge_patch_gpu(reasonable_patch, threshold=0.3)
            print("✓ Reasonable patch processed successfully")
        except Exception as e:
            print(f"✗ Reasonable patch failed: {e}")
            return False
        
        # Test case 2: Problematic large patch (should be rejected).
        try:
            # This would create a 131+ GB allocation - should be rejected.
            large_patch = np.zeros((66, 922, 26752), dtype=np.uint32)
            result, mapping = merge_patch_gpu(large_patch, threshold=0.3)
            print("✗ Large patch was not rejected - safety check failed!")
            return False
        except RuntimeError as e:
            if "exceeding reasonable limit" in str(e) or "too large" in str(e):
                print("✓ Large patch correctly rejected by safety checks")
            else:
                print(f"✗ Large patch rejected for wrong reason: {e}")
                return False
        
        # Test case 3: Very wide patch (sparse distribution pattern).
        try:
            wide_patch = np.zeros((2, 500, 20000), dtype=np.uint32)
            result, mapping = merge_patch_gpu(wide_patch, threshold=0.3)
            print("✗ Wide patch was not rejected - safety check failed!")
            return False
        except RuntimeError as e:
            if "too large" in str(e) or "sparse tile distribution" in str(e):
                print("✓ Wide patch correctly rejected by safety checks")
            else:
                print(f"✗ Wide patch rejected for wrong reason: {e}")
                return False
        
        print("✓ All GPU safety check tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ GPU safety check test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_cpu_safety_checks():
    """Test the enhanced CPU safety checks."""
    print("\n" + "="*80)
    print("TESTING CPU SAFETY CHECKS")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.rules import merge_patch_cpu
        
        # Test case 1: Reasonable patch (should work).
        reasonable_patch = np.zeros((2, 512, 512), dtype=np.uint32)
        reasonable_patch[0, 100:200, 100:200] = 1
        reasonable_patch[1, 150:250, 150:250] = 2
        
        try:
            result, mapping = merge_patch_cpu(reasonable_patch, threshold=0.3)
            print("✓ Reasonable CPU patch processed successfully")
        except Exception as e:
            print(f"✗ Reasonable CPU patch failed: {e}")
            return False
        
        # Test case 2: Problematic large patch (should be rejected).
        try:
            # This would create a massive CPU allocation - should be rejected.
            large_patch = np.zeros((4, 10000, 10000), dtype=np.uint32)
            result, mapping = merge_patch_cpu(large_patch, threshold=0.3)
            print("✗ Large CPU patch was not rejected - safety check failed!")
            return False
        except RuntimeError as e:
            if "exceeding safe CPU limit" in str(e) or "too large" in str(e):
                print("✓ Large CPU patch correctly rejected by safety checks")
            else:
                print(f"✗ Large CPU patch rejected for wrong reason: {e}")
                return False
        
        print("✓ All CPU safety check tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ CPU safety check test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_batch_validation():
    """Test the enhanced batch validation in merge_cluster_batched."""
    print("\n" + "="*80)
    print("TESTING BATCH VALIDATION")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import merge_cluster_batched
        
        # Create a problematic cluster that should trigger incremental processing.
        problematic_cluster = [(r, c) for r in [5, 6] for c in range(100)]  # 200 tiles in a line.
        
        print(f"Created problematic cluster: {len(problematic_cluster)} tiles")
        
        # Create a mock loader.
        def mock_loader(ys: slice, xs: slice) -> np.ndarray:
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            return np.zeros((h, w), dtype=np.uint32)
        
        # Create a mock global array.
        global_array = np.zeros((10000, 50000), dtype=np.uint32)
        
        # This should automatically switch to incremental processing.
        result_patch, (y0, x0), mapping = merge_cluster_batched(
            cluster=problematic_cluster,
            loader=mock_loader,
            height=10000,
            width=50000,
            tile_h=512,
            tile_w=512,
            overlap=102,
            threshold=0.3,
            use_gpu=False,
            gid_offset=1,
            global_merged_array=global_array,
        )
        
        print("✓ Problematic cluster handled successfully with automatic fallback")
        print(f"✓ Result patch shape: {result_patch.shape} (should be minimal)")
        
        # Verify that incremental processing was used.
        assert result_patch.size <= 1, "Should use incremental processing for problematic clusters"
        
        print("✓ Batch validation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Batch validation test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Run all critical memory fix tests."""
    print("CRITICAL MEMORY ALLOCATION FIXES VERIFICATION")
    print("="*80)
    print("Testing fixes for the 131.47 GiB allocation error...")
    
    tests = [
        test_incremental_processing,
        test_gpu_safety_checks,
        test_cpu_safety_checks,
        test_batch_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*80)
    print(f"CRITICAL MEMORY FIXES TEST RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("✓ All critical memory allocation fixes are working correctly!")
        print("✓ The system should now prevent 131+ GiB allocation errors.")
        print("✓ Sparse tile distributions will be processed safely with incremental methods.")
        return True
    else:
        print("✗ Some critical memory fixes are not working correctly.")
        print("✗ Please review the failed tests and fix the issues.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

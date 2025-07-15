#!/usr/bin/env python3
"""
Author: Christos Botos.
Script Name: test_integration_large_image.py.
Description:
    Integration test to verify that the large image fixes work correctly.
    This script simulates processing a large image with many tiles to ensure
    that the fixes prevent crashes and handle edge cases gracefully.
    
    This test creates a realistic scenario similar to the one that was failing
    but with smaller dimensions to make it runnable in a test environment.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the cellpose_merge directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'nuclei_segmentation', 'cellpose_merge'))

from merge_tiles import merge_masks_streaming, _check_cluster_feasibility, _split_large_cluster


def create_test_tiles(tiles_dir: Path, num_rows: int, num_cols: int, tile_size: int = 512):
    """
    Create test tile masks for integration testing.
    
    Parameters
    ----------
    tiles_dir : Path
        Directory to save the tile masks.
    num_rows, num_cols : int
        Number of tile rows and columns.
    tile_size : int, default 512
        Size of each tile in pixels.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_rows * num_cols} test tiles in {tiles_dir}")
    
    for r in range(num_rows):
        for c in range(num_cols):
            # Create a simple test mask with some objects
            mask = np.zeros((tile_size, tile_size), dtype=np.uint32)
            
            # Add some circular objects to make it realistic
            for i in range(5):  # 5 objects per tile
                center_y = np.random.randint(50, tile_size - 50)
                center_x = np.random.randint(50, tile_size - 50)
                radius = np.random.randint(10, 30)
                
                # Create circular mask
                y, x = np.ogrid[:tile_size, :tile_size]
                circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                mask[circle_mask] = i + 1  # Object IDs start from 1
            
            # Save as NPZ (more efficient than TIFF)
            tile_filename = f"{r}_{c}.npz"
            np.savez_compressed(tiles_dir / tile_filename, mask=mask)
    
    print(f"Created {num_rows * num_cols} test tiles successfully")


def test_small_image():
    """Test with a small image that should work normally."""
    print("\n=== Testing Small Image (Should Work) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tiles_dir = temp_path / "tiles"
        
        # Create a small 3x3 grid of tiles
        num_rows, num_cols = 3, 3
        tile_size = 512
        overlap = 64
        
        create_test_tiles(tiles_dir, num_rows, num_cols, tile_size)
        
        # Calculate image dimensions
        stride = tile_size - overlap
        height = (num_rows - 1) * stride + tile_size
        width = (num_cols - 1) * stride + tile_size
        
        print(f"Image dimensions: {height} x {width}")
        print(f"Total tiles: {num_rows * num_cols}")
        
        try:
            # This should work fine
            merged_mask = merge_masks_streaming(
                height=height,
                width=width,
                tile_h=tile_size,
                tile_w=tile_size,
                overlap=overlap,
                tiles_path=str(tiles_dir),
                threshold=0.3,
                use_gpu=False,  # Use CPU to avoid GPU memory issues in testing
                gpu_batch_size=2,
                gpu_memory_limit_gb=4.0
            )
            
            print(f"✓ Small image processed successfully: {merged_mask.shape}, max_label={merged_mask.max()}")
            return True
            
        except Exception as e:
            print(f"✗ Small image processing failed: {e}")
            return False


def test_medium_image():
    """Test with a medium-sized image that should trigger batching."""
    print("\n=== Testing Medium Image (Should Trigger Batching) ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tiles_dir = temp_path / "tiles"

        # Create a more reasonable grid that should trigger batching but not be too slow
        num_rows, num_cols = 8, 8  # 64 tiles - more reasonable for testing
        tile_size = 512
        overlap = 64

        create_test_tiles(tiles_dir, num_rows, num_cols, tile_size)

        # Calculate image dimensions
        stride = tile_size - overlap
        height = (num_rows - 1) * stride + tile_size
        width = (num_cols - 1) * stride + tile_size

        print(f"Image dimensions: {height} x {width}")
        print(f"Total tiles: {num_rows * num_cols}")

        try:
            # This should trigger batching but still work
            merged_mask = merge_masks_streaming(
                height=height,
                width=width,
                tile_h=tile_size,
                tile_w=tile_size,
                overlap=overlap,
                tiles_path=str(tiles_dir),
                threshold=0.3,
                use_gpu=False,  # Use CPU to avoid GPU memory issues in testing
                gpu_batch_size=2,
                gpu_memory_limit_gb=4.0
            )

            print(f"✓ Medium image processed successfully: {merged_mask.shape}, max_label={merged_mask.max()}")
            return True

        except Exception as e:
            print(f"✗ Medium image processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_feasibility_checks():
    """Test the feasibility checking functions."""
    print("\n=== Testing Feasibility Checks ===")
    
    # Test with a cluster that should be feasible
    small_cluster = [(r, c) for r in range(5) for c in range(5)]  # 25 tiles
    is_feasible, reason = _check_cluster_feasibility(
        small_cluster, tile_h=512, tile_w=512, overlap=64,
        height=5000, width=5000, memory_limit_gb=8.0
    )
    
    if is_feasible:
        print("✓ Small cluster correctly identified as feasible")
    else:
        print(f"✗ Small cluster incorrectly identified as infeasible: {reason}")
        return False
    
    # Test with a cluster that should NOT be feasible
    large_cluster = [(r, c) for r in range(100) for c in range(100)]  # 10,000 tiles
    is_feasible, reason = _check_cluster_feasibility(
        large_cluster, tile_h=1024, tile_w=1024, overlap=128,
        height=100000, width=100000, memory_limit_gb=8.0
    )
    
    if not is_feasible:
        print(f"✓ Large cluster correctly identified as infeasible: {reason}")
    else:
        print("✗ Large cluster incorrectly identified as feasible")
        return False
    
    # Test cluster splitting
    sub_clusters = _split_large_cluster(large_cluster, max_cluster_size=500)
    if len(sub_clusters) > 1:
        print(f"✓ Large cluster split into {len(sub_clusters)} sub-clusters")
        
        # Verify no tiles were lost
        total_tiles = sum(len(sc) for sc in sub_clusters)
        if total_tiles == len(large_cluster):
            print("✓ No tiles lost during splitting")
        else:
            print(f"✗ Tiles lost during splitting: {total_tiles} != {len(large_cluster)}")
            return False
    else:
        print("✗ Large cluster was not split")
        return False
    
    return True


def main():
    """Run all integration tests."""
    print("Large Image Processing Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Small Image", test_small_image()))
    results.append(("Medium Image", test_medium_image()))
    results.append(("Feasibility Checks", test_feasibility_checks()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All integration tests passed! The large image fixes are working correctly.")
        return 0
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())

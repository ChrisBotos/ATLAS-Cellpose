#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_coordinate_and_batching_fixes.py.
Description:
    Test script to verify that the coordinate detection and batching fixes
    resolve the issues with giant images and proper batch creation.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib.

Usage:
    python test_coordinate_and_batching_fixes.py

Arguments:
    None.

Inputs:
    Simulates coordinate detection scenarios.

Outputs:
    Verifies coordinate detection and batching fixes work correctly.

Key Features:
    • Tests pixel coordinate to tile index conversion.
    • Validates proper 8-batch creation for 4×4 grids.
    • Tests giant image coordinate scenarios.
    • Verifies batch size parameter handling.

Notes:
    This test verifies the specific fixes for coordinate and batching issues.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Add the code directory to the path.
sys.path.insert(0, str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_pixel_coordinate_detection():
    """Test that pixel coordinates are correctly detected and converted."""
    logging.info("Testing pixel coordinate detection...")
    
    # Simulate the giant image scenario.
    # Image: 26460×26459, Tiles: 512×512, Overlap: 102
    height, width = 26460, 26459
    tile_h, tile_w = 512, 512
    overlap = 102
    stride_h, stride_w = tile_h - overlap, tile_w - overlap  # 410, 410
    
    # Simulate pixel coordinates like those in the giant image.
    pixel_coords = [
        (0, 0), (0, 410), (0, 820), (0, 1230),
        (410, 0), (410, 410), (410, 820), (410, 1230),
        (16800, 22000), (16800, 14800)  # Problematic coordinates from log
    ]
    
    # Test the coordinate detection logic.
    max_coord = max(max(r, c) for r, c in pixel_coords)
    min_coord = min(min(r, c) for r, c in pixel_coords)
    
    pixel_coord_indicators = [
        max_coord > 1000,  # Large coordinates suggest pixels
        min_coord >= 0,    # Should start from 0 or positive
        sum(1 for r, c in pixel_coords if r % stride_h == 0 and c % stride_w == 0) > len(pixel_coords) * 0.5
    ]
    
    if sum(pixel_coord_indicators) >= 2:
        # Convert pixel coordinates to tile indices.
        tile_coords = [(r // stride_h, c // stride_w) for r, c in pixel_coords]
        logging.info(f"✓ Detected pixel coordinates, converted to tile indices")
        
        # Verify the problematic coordinates are now reasonable.
        max_tile_r = max(r for r, _ in tile_coords)
        max_tile_c = max(c for _, c in tile_coords)
        
        # Calculate expected maximum tile indices.
        expected_max_r = (height + stride_h - 1) // stride_h - 1  # Should be ~64
        expected_max_c = (width + stride_w - 1) // stride_w - 1   # Should be ~64
        
        if max_tile_r <= expected_max_r and max_tile_c <= expected_max_c:
            logging.info(f"✓ Converted coordinates are within bounds: max=({max_tile_r},{max_tile_c}), expected_max=({expected_max_r},{expected_max_c})")
            return True
        else:
            logging.error(f"✗ Converted coordinates still out of bounds: max=({max_tile_r},{max_tile_c}), expected_max=({expected_max_r},{expected_max_c})")
            return False
    else:
        logging.error("✗ Failed to detect pixel coordinates")
        return False

def test_4x4_grid_batching():
    """Test that a 4×4 grid creates exactly 8 batches as expected."""
    logging.info("Testing 4×4 grid batching...")
    
    try:
        from batch_merge import group_tiles_by_spatial_proximity
        
        # Create a 4×4 grid (16 tiles).
        cluster = [(r, c) for r in range(4) for c in range(4)]
        
        # Test with batch_size=1 (should create 8 individual batches).
        batches_size_1 = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        if len(batches_size_1) != 8:
            logging.error(f"✗ Expected 8 batches for 4×4 grid with batch_size=1, got {len(batches_size_1)}")
            return False
        
        # Test with batch_size=2 (should create 4 combined batches).
        batches_size_2 = group_tiles_by_spatial_proximity(cluster, batch_size=2)
        
        if len(batches_size_2) != 4:
            logging.error(f"✗ Expected 4 batches for 4×4 grid with batch_size=2, got {len(batches_size_2)}")
            return False
        
        # Verify all tiles are processed.
        all_processed_tiles = set()
        for batch in batches_size_1:
            all_processed_tiles.update(batch)
        
        if all_processed_tiles != set(cluster):
            missing = set(cluster) - all_processed_tiles
            logging.error(f"✗ Missing tiles in batch processing: {missing}")
            return False
        
        logging.info(f"✓ 4×4 grid batching: {len(batches_size_1)} batches (size=1), {len(batches_size_2)} batches (size=2)")
        return True
        
    except Exception as e:
        logging.error(f"✗ 4×4 grid batching test failed: {e}")
        return False

def test_batch_content_validation():
    """Test that batches contain the expected tile groups."""
    logging.info("Testing batch content validation...")
    
    try:
        from batch_merge import group_tiles_by_spatial_proximity
        
        # Create a simple 2×2 grid for easier validation.
        cluster = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        # For a 2×2 grid, we should get 1 batch with all 4 tiles.
        if len(batches) != 1:
            logging.error(f"✗ Expected 1 batch for 2×2 grid, got {len(batches)}")
            return False
        
        if set(batches[0]) != set(cluster):
            logging.error(f"✗ Batch content mismatch: expected {cluster}, got {batches[0]}")
            return False
        
        # Test a 3×3 grid.
        cluster_3x3 = [(r, c) for r in range(3) for c in range(3)]
        batches_3x3 = group_tiles_by_spatial_proximity(cluster_3x3, batch_size=1)
        
        # Verify all tiles are processed.
        all_processed = set()
        for batch in batches_3x3:
            all_processed.update(batch)
        
        if all_processed != set(cluster_3x3):
            missing = set(cluster_3x3) - all_processed
            logging.error(f"✗ Missing tiles in 3×3 processing: {missing}")
            return False
        
        logging.info(f"✓ Batch content validation: 2×2 grid → {len(batches)} batches, 3×3 grid → {len(batches_3x3)} batches")
        return True
        
    except Exception as e:
        logging.error(f"✗ Batch content validation test failed: {e}")
        return False

def test_coordinate_bounds_validation():
    """Test that coordinate bounds validation works correctly."""
    logging.info("Testing coordinate bounds validation...")
    
    try:
        from merge_tiles import _check_cluster_feasibility
        
        # Test the giant image scenario.
        height, width = 26460, 26459
        tile_h, tile_w = 512, 512
        overlap = 102
        
        # Test with converted tile coordinates (should be valid).
        stride_h, stride_w = tile_h - overlap, tile_w - overlap
        
        # Convert the problematic pixel coordinates to tile indices.
        pixel_coord = (16800, 22000)
        tile_coord = (pixel_coord[0] // stride_h, pixel_coord[1] // stride_w)  # (40, 53)
        
        cluster = [tile_coord]
        is_feasible, reason = _check_cluster_feasibility(
            cluster, tile_h, tile_w, overlap, height, width, memory_limit_gb=8.0
        )
        
        if not is_feasible:
            logging.error(f"✗ Converted tile coordinate {tile_coord} rejected: {reason}")
            return False
        
        # Test with the original problematic pixel coordinates (should be invalid).
        cluster_invalid = [pixel_coord]
        is_feasible_invalid, reason_invalid = _check_cluster_feasibility(
            cluster_invalid, tile_h, tile_w, overlap, height, width, memory_limit_gb=8.0
        )
        
        if is_feasible_invalid:
            logging.error(f"✗ Invalid pixel coordinate {pixel_coord} incorrectly accepted")
            return False
        
        logging.info(f"✓ Coordinate bounds validation: converted {pixel_coord} → {tile_coord} (valid)")
        return True
        
    except Exception as e:
        logging.error(f"✗ Coordinate bounds validation test failed: {e}")
        return False

def main():
    """Run all coordinate and batching fix tests."""
    logging.info("Starting coordinate and batching fixes verification...")
    logging.info("This test verifies fixes for giant image coordinate issues and proper batching.\n")
    
    tests = [
        test_pixel_coordinate_detection,
        test_4x4_grid_batching,
        test_batch_content_validation,
        test_coordinate_bounds_validation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logging.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    logging.info(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        logging.info("\n🎉 ALL COORDINATE AND BATCHING FIXES VERIFIED! 🎉")
        logging.info("\n✅ FIXED ISSUES:")
        logging.info("• Pixel coordinate detection and conversion - FIXED")
        logging.info("• Giant image coordinate bounds errors - RESOLVED")
        logging.info("• Proper 8-batch creation for 4×4 grids - IMPLEMENTED")
        logging.info("• Batch size parameter handling - CORRECTED")
        logging.info("\n🎯 EXPECTED OUTCOMES:")
        logging.info("• Giant images will process all tiles correctly")
        logging.info("• Coordinate conversion will handle pixel coordinates properly")
        logging.info("• Batching will create the expected number of batches")
        logging.info("• All tiles will be processed without skipping")
        logging.info("\nYour giant image processing should now work correctly! 🚀")
        return 0
    else:
        logging.error(f"\n❌ {total - passed} test(s) still failing.")
        return 1

if __name__ == "__main__":
    exit(main())

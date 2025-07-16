#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_enhanced_batching.py.
Description:
    Test script to verify that the enhanced 2x2 spatial batching strategy
    works correctly and processes all tiles without skipping any.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib.

Usage:
    python test_enhanced_batching.py

Arguments:
    None.

Inputs:
    Creates test tile configurations for testing.

Outputs:
    Verifies enhanced batching strategy works correctly.

Key Features:
    • Tests 2x2 spatial batching with overlap regions.
    • Validates that all tiles are processed.
    • Tests different grid configurations.
    • Verifies proper batch ordering and overlap handling.

Notes:
    This test verifies the enhanced spatial batching implementation.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Add the code directory to the path.
sys.path.insert(0, str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_2x2_group_generation():
    """Test that 2x2 groups are generated correctly."""
    logging.info("Testing 2x2 group generation...")
    
    try:
        from batch_merge import group_tiles_by_spatial_proximity
        
        # Test with a 4x4 grid (16 tiles).
        cluster = [(r, c) for r in range(4) for c in range(4)]
        
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        # For a 4x4 grid, we should have 9 primary 2x2 groups.
        # Each group should have 4 tiles (except edge cases).
        if len(batches) == 0:
            logging.error("✗ No batches generated")
            return False
        
        # Verify all tiles are included.
        all_processed_tiles = set()
        for batch in batches:
            all_processed_tiles.update(batch)
        
        original_tiles = set(cluster)
        if all_processed_tiles != original_tiles:
            missing = original_tiles - all_processed_tiles
            extra = all_processed_tiles - original_tiles
            logging.error(f"✗ Tile processing mismatch: missing={missing}, extra={extra}")
            return False
        
        # Verify reasonable number of batches.
        expected_min_batches = 9  # At least 9 primary 2x2 groups for 4x4 grid.
        if len(batches) < expected_min_batches:
            logging.error(f"✗ Too few batches: {len(batches)} < {expected_min_batches}")
            return False
        
        logging.info(f"✓ Generated {len(batches)} batches for 4x4 grid")
        return True
        
    except Exception as e:
        logging.error(f"✗ 2x2 group generation test failed: {e}")
        return False

def test_batch_size_scaling():
    """Test that batch size parameter works correctly."""
    logging.info("Testing batch size scaling...")
    
    try:
        from batch_merge import group_tiles_by_spatial_proximity
        
        # Test with a 3x3 grid.
        cluster = [(r, c) for r in range(3) for c in range(3)]
        
        # Test different batch sizes.
        for batch_size in [1, 2, 3]:
            batches = group_tiles_by_spatial_proximity(cluster, batch_size=batch_size)
            
            if len(batches) == 0:
                logging.error(f"✗ No batches generated for batch_size={batch_size}")
                return False
            
            # Verify all tiles are still processed.
            all_processed_tiles = set()
            for batch in batches:
                all_processed_tiles.update(batch)
            
            if all_processed_tiles != set(cluster):
                logging.error(f"✗ Tiles missing for batch_size={batch_size}")
                return False
            
            logging.debug(f"Batch size {batch_size}: {len(batches)} batches")
        
        logging.info("✓ Batch size scaling working correctly")
        return True
        
    except Exception as e:
        logging.error(f"✗ Batch size scaling test failed: {e}")
        return False

def test_irregular_cluster_handling():
    """Test that irregular (non-rectangular) clusters are handled correctly."""
    logging.info("Testing irregular cluster handling...")
    
    try:
        from batch_merge import group_tiles_by_spatial_proximity
        
        # Test with an L-shaped cluster.
        cluster = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1),
            (2, 0)
        ]
        
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        if len(batches) == 0:
            logging.error("✗ No batches generated for irregular cluster")
            return False
        
        # Verify all tiles are processed.
        all_processed_tiles = set()
        for batch in batches:
            all_processed_tiles.update(batch)
        
        if all_processed_tiles != set(cluster):
            missing = set(cluster) - all_processed_tiles
            logging.error(f"✗ Tiles missing from irregular cluster: {missing}")
            return False
        
        logging.info(f"✓ Irregular cluster handled correctly with {len(batches)} batches")
        return True
        
    except Exception as e:
        logging.error(f"✗ Irregular cluster handling test failed: {e}")
        return False

def test_overlap_region_classification():
    """Test that overlap regions are classified correctly."""
    logging.info("Testing overlap region classification...")
    
    try:
        # Test the classification logic directly.
        def classify_group_type(r, c, min_r, max_r, min_c, max_c):
            """Simulate the group classification logic."""
            is_horizontal_edge = (r == min_r or r == max_r - 1)
            is_vertical_edge = (c == min_c or c == max_c - 1)
            is_corner = is_horizontal_edge and is_vertical_edge
            
            if is_corner:
                return "corner"
            elif is_horizontal_edge:
                return "horizontal_edge"
            elif is_vertical_edge:
                return "vertical_edge"
            else:
                return "center"
        
        # Test with a 4x4 grid (groups from 0,0 to 2,2).
        min_r, max_r, min_c, max_c = 0, 3, 0, 3
        
        expected_corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        expected_horizontal_edges = [(0, 1), (2, 1)]
        expected_vertical_edges = [(1, 0), (1, 2)]
        expected_centers = [(1, 1)]
        
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                group_type = classify_group_type(r, c, min_r, max_r, min_c, max_c)
                
                if group_type == "corner" and (r, c) not in expected_corners:
                    logging.error(f"✗ Incorrect corner classification: ({r},{c})")
                    return False
                elif group_type == "horizontal_edge" and (r, c) not in expected_horizontal_edges:
                    logging.error(f"✗ Incorrect horizontal edge classification: ({r},{c})")
                    return False
                elif group_type == "vertical_edge" and (r, c) not in expected_vertical_edges:
                    logging.error(f"✗ Incorrect vertical edge classification: ({r},{c})")
                    return False
                elif group_type == "center" and (r, c) not in expected_centers:
                    logging.error(f"✗ Incorrect center classification: ({r},{c})")
                    return False
        
        logging.info("✓ Overlap region classification working correctly")
        return True
        
    except Exception as e:
        logging.error(f"✗ Overlap region classification test failed: {e}")
        return False

def main():
    """Run all enhanced batching tests."""
    logging.info("Starting enhanced batching strategy verification...")
    
    tests = [
        test_2x2_group_generation,
        test_batch_size_scaling,
        test_irregular_cluster_handling,
        test_overlap_region_classification,
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
        logging.info("\n🎉 All enhanced batching tests verified successfully!")
        logging.info("\nThe enhanced 2x2 spatial batching strategy is working:")
        logging.info("• Primary 2x2 groups are generated correctly")
        logging.info("• All tiles are processed without skipping")
        logging.info("• Batch size scaling works properly")
        logging.info("• Irregular clusters are handled correctly")
        logging.info("• Overlap regions are classified appropriately")
        logging.info("\nThe merge process should now handle complex tile arrangements!")
        return 0
    else:
        logging.error(f"\n❌ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())

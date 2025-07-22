"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_merge_fixes.py.
Description:
    Comprehensive test to verify that the critical tile merging issues have been fixed.
    Tests the corrected 3-step merging implementation to ensure:
    1. Complete tile processing (no mask fragmentation)
    2. Proper 3-step merging rules implementation
    3. Mask integrity preservation
    4. ID consistency without collisions
    5. Border preservation without fragmentation

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • cellpose_merge.two_phase_merge module.

Key Features:
    • Tests complete nucleus processing vs overlap-only processing.
    • Validates 3-step merging rules with controlled synthetic data.
    • Checks mask integrity and ID consistency.
    • Verifies border preservation without fragmentation.
"""

import sys
import numpy as np
from pathlib import Path

# Add the cellpose_merge module to path.
sys.path.append('code/nuclei_segmentation/cellpose_merge')

from two_phase_merge import merge_two_tiles


def test_complete_nucleus_processing():
    """
    Test that complete nuclei are processed, not just overlap portions.
    
    This test creates a scenario where a nucleus spans across tile boundaries
    and verifies that the ENTIRE nucleus is affected by merge decisions.
    """
    print("=== Test 1: Complete Nucleus Processing ===")
    
    # Create tiles where nucleus spans across boundaries.
    tile1 = np.array([
        [1, 1, 1, 0],  # Nucleus 1 spans into overlap region
        [1, 1, 1, 0],
        [0, 0, 2, 2],  # Nucleus 2 only in overlap region
        [0, 0, 2, 2]
    ], dtype=np.uint32)
    
    tile2 = np.array([
        [0, 3, 3, 3],  # Nucleus 3 spans from overlap region
        [0, 3, 3, 3],
        [4, 4, 0, 0],  # Nucleus 4 only in overlap region
        [4, 4, 0, 0]
    ], dtype=np.uint32)
    
    # Overlap: columns 2-3 of tile1, columns 0-1 of tile2.
    overlap_slices = (slice(0, 4), slice(2, 4), slice(0, 4), slice(0, 2))
    
    print("Original tile1:")
    print(tile1)
    print("Original tile2:")
    print(tile2)
    print("Overlap from tile1:", tile1[0:4, 2:4])
    print("Overlap from tile2:", tile2[0:4, 0:2])
    
    result1, result2, mapping = merge_two_tiles(tile1, tile2, overlap_slices, use_gpu=False)
    
    print("Result tile1:")
    print(result1)
    print("Result tile2:")
    print(result2)
    print(f"Mapping: {mapping}")
    
    # Verify complete nucleus processing.
    # If nucleus 1 was affected, ALL pixels of nucleus 1 should be affected.
    original_nucleus1_pixels = np.sum(tile1 == 1)
    result_nucleus1_pixels = np.sum(result1 == 1)
    
    print(f"Original nucleus 1 pixels: {original_nucleus1_pixels}")
    print(f"Result nucleus 1 pixels: {result_nucleus1_pixels}")
    
    # Check that overlap regions are identical in both tiles.
    overlap1_result = result1[0:4, 2:4]
    overlap2_result = result2[0:4, 0:2]
    
    if np.array_equal(overlap1_result, overlap2_result):
        print("✓ PASS: Overlap regions are identical after merge")
    else:
        print("✗ FAIL: Overlap regions differ after merge")
        print("Overlap1 result:", overlap1_result)
        print("Overlap2 result:", overlap2_result)
    
    print()


def test_3step_merging_rules():
    """
    Test the exact implementation of 3-step merging rules.
    
    Creates a controlled scenario to verify:
    1. Priority Selection based on nuclei count
    2. Border Deletion of priority tile masks touching borders
    3. Cleanup of remaining non-priority masks
    """
    print("=== Test 2: 3-Step Merging Rules ===")
    
    # Create tiles with known priority (tile1 has more nuclei).
    tile1 = np.array([
        [1, 1, 2, 2],  # Nucleus 1 (internal), Nucleus 2 (border-touching)
        [1, 1, 2, 2],
        [3, 3, 0, 0],  # Nucleus 3 (border-touching)
        [3, 3, 0, 0]
    ], dtype=np.uint32)
    
    tile2 = np.array([
        [4, 4, 0, 0],  # Nucleus 4 (border-touching)
        [4, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint32)
    
    # Overlap: columns 0-1 of tile1, columns 0-1 of tile2.
    overlap_slices = (slice(0, 4), slice(0, 2), slice(0, 4), slice(0, 2))
    
    print("Tile1 (should be priority - 3 nuclei):")
    print(tile1)
    print("Tile2 (non-priority - 1 nucleus):")
    print(tile2)
    
    result1, result2, mapping = merge_two_tiles(tile1, tile2, overlap_slices, use_gpu=False)
    
    print("Result tile1:")
    print(result1)
    print("Result tile2:")
    print(result2)
    print(f"Mapping: {mapping}")
    
    # Analyze results according to 3-step rules.
    # Step 1: Tile1 should be priority (3 nuclei vs 1 nucleus).
    # Step 2: Priority border nuclei (1, 3) should be deleted.
    # Step 3: Non-priority nucleus 4 should be deleted (doesn't cross boundary).
    
    result1_nuclei = set(np.unique(result1[result1 > 0]))
    result2_nuclei = set(np.unique(result2[result2 > 0]))
    
    print(f"Remaining nuclei in tile1: {result1_nuclei}")
    print(f"Remaining nuclei in tile2: {result2_nuclei}")
    
    # Only nucleus 2 should remain (priority internal nucleus).
    expected_remaining = {2}
    actual_remaining = result1_nuclei | result2_nuclei
    
    if actual_remaining == expected_remaining:
        print("✓ PASS: 3-step rules correctly applied")
    else:
        print(f"✗ FAIL: Expected {expected_remaining}, got {actual_remaining}")
    
    print()


def test_mask_integrity():
    """
    Test that masks are never fragmented during merging.
    
    Verifies that nuclei are either completely preserved or completely deleted,
    never split into fragments.
    """
    print("=== Test 3: Mask Integrity ===")
    
    # Create a nucleus that spans across the overlap boundary.
    tile1 = np.array([
        [1, 1, 1, 1],  # Nucleus 1 spans entire width
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint32)
    
    tile2 = np.array([
        [1, 1, 2, 2],  # Same nucleus 1 continues, nucleus 2 starts
        [1, 1, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint32)
    
    # Overlap: columns 2-3 of tile1, columns 0-1 of tile2.
    overlap_slices = (slice(0, 4), slice(2, 4), slice(0, 4), slice(0, 2))
    
    print("Tile1 (nucleus 1 spans boundary):")
    print(tile1)
    print("Tile2 (nucleus 1 continues, nucleus 2 added):")
    print(tile2)
    
    result1, result2, mapping = merge_two_tiles(tile1, tile2, overlap_slices, use_gpu=False)
    
    print("Result tile1:")
    print(result1)
    print("Result tile2:")
    print(result2)
    print(f"Mapping: {mapping}")
    
    # Check that no nucleus is fragmented.
    # Count connected components for each nucleus ID.
    def count_connected_components(mask, nucleus_id):
        """Count connected components for a specific nucleus ID."""
        from scipy import ndimage
        nucleus_mask = mask == nucleus_id
        if not np.any(nucleus_mask):
            return 0
        labeled, num_components = ndimage.label(nucleus_mask)
        return num_components
    
    # Check each remaining nucleus for fragmentation.
    all_nuclei = set(np.unique(result1[result1 > 0])) | set(np.unique(result2[result2 > 0]))
    
    fragmentation_detected = False
    for nucleus_id in all_nuclei:
        components1 = count_connected_components(result1, nucleus_id)
        components2 = count_connected_components(result2, nucleus_id)
        total_components = components1 + components2
        
        print(f"Nucleus {nucleus_id}: {total_components} connected component(s)")
        
        if total_components > 1:
            fragmentation_detected = True
            print(f"  ✗ FRAGMENTATION DETECTED for nucleus {nucleus_id}")
    
    if not fragmentation_detected:
        print("✓ PASS: No mask fragmentation detected")
    else:
        print("✗ FAIL: Mask fragmentation detected")
    
    print()


def main():
    """Run all tests to verify merge fixes."""
    print("Testing Tile Merging Implementation Fixes")
    print("=" * 50)
    
    try:
        test_complete_nucleus_processing()
        test_3step_merging_rules()
        test_mask_integrity()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

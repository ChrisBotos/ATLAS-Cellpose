"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_enhanced_3step_cleanup.py.
Description:
    Test script for the enhanced _find_border_touching_nuclei function that now
    returns two sets of nuclei and the improved merge_tiles_cpu_3step function
    that properly implements Step 3 (Cleanup) of the 3-step merging algorithm.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • sys for path management.

Usage:
    python test_enhanced_3step_cleanup.py

Key Features:
    • Tests the enhanced _find_border_touching_nuclei function with two return sets.
    • Tests proper Step 3 (Cleanup) implementation in merge_tiles_cpu_3step.
    • Validates cross-boundary nucleus preservation.
    • Tests overlap region cleanup.

Notes:
    • Creates temporary .npz files for testing.
    • Tests realistic tile overlap scenarios with proper cleanup.
"""

import traceback
import numpy as np
import sys
import os
from pathlib import Path
import tempfile

# Add the cellpose_merge directory to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'nuclei_segmentation', 'cellpose_merge'))

from rules import _find_border_touching_nuclei, merge_tiles_cpu_3step


def test_enhanced_border_detection():
    """Test the enhanced _find_border_touching_nuclei function with two return sets."""
    print("=== Testing Enhanced Border Detection (Two Sets) ===")
    
    # Create a 20x20 tile with overlap_length=5.
    tile = np.zeros((20, 20), dtype=np.uint32)
    overlap_length = 5

    # For direction='right', boundary line is at column (20-5) = 15.
    # Buffer zone checks columns 14, 15, 16.
    # Overlap region starts at column 17 (15 + 2).

    # Nucleus 1: touches boundary line (column 15).
    tile[5:7, 14:16] = 1  # Spans columns 14-15, touches boundary.

    # Nucleus 2: completely beyond boundary line (in overlap region).
    tile[2:4, 17:19] = 2  # Spans columns 17-18, completely in overlap region.

    # Nucleus 3: before boundary line (not in overlap).
    tile[8:10, 10:12] = 3  # Spans columns 10-11, before boundary.

    # Nucleus 4: crosses boundary line.
    tile[11:13, 14:17] = 4  # Spans columns 14-16, crosses boundary.

    boundary_nuclei, overlap_nuclei = _find_border_touching_nuclei(tile, overlap_length, 'right')

    print(f"Boundary line at column: {20 - overlap_length}")
    print(f"Buffer zone columns: {[20 - overlap_length - 1, 20 - overlap_length, 20 - overlap_length + 1]}")
    print(f"Overlap region starts at column: {20 - overlap_length + 2}")
    print(f"Boundary-touching nuclei: {boundary_nuclei}")
    print(f"Overlap region nuclei: {overlap_nuclei}")

    # Expected results:
    # Boundary-touching: nuclei 1 and 4 (touch or cross the boundary line at column 15).
    # Overlap region: nucleus 2 (completely beyond column 16, in overlap region starting at 17).
    expected_boundary = {1, 4}
    expected_overlap = {2}
    
    assert boundary_nuclei == expected_boundary, f"Boundary: Expected {expected_boundary}, got {boundary_nuclei}"
    assert overlap_nuclei == expected_overlap, f"Overlap: Expected {expected_overlap}, got {overlap_nuclei}"
    
    print("✓ Enhanced border detection works correctly")
    return True


def create_cleanup_test_tiles():
    """
    Create test tiles that demonstrate proper Step 3 (Cleanup) implementation.
    
    Returns
    -------
    tuple
        (tile1_path, tile2_path, overlap_length, temp_dir) for testing.
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create 20x20 tiles with 5-pixel overlap.
    tile1_mask = np.zeros((20, 20), dtype=np.uint32)
    tile2_mask = np.zeros((20, 20), dtype=np.uint32)
    overlap_length = 5
    
    print(f"Creating test tiles with {overlap_length}-pixel overlap")
    print(f"Tile relationship: tile1_left_of_tile2")
    print(f"Overlap boundary at column {20 - overlap_length} = 15")
    
    # Tile1 nuclei (left tile):
    # Nucleus 1: internal nucleus (should be preserved).
    tile1_mask[3:6, 3:6] = 1
    
    # Nucleus 2: touches tile1 border (will be deleted if tile1 has priority).
    tile1_mask[8:11, 0:3] = 2
    
    # Nucleus 3: cross-boundary nucleus (extends into overlap region).
    tile1_mask[12:15, 14:18] = 3  # Spans columns 14-17, crosses boundary at 15.
    
    # Nucleus 4: touches bottom border.
    tile1_mask[17:20, 8:11] = 4
    
    # Tile2 nuclei (right tile):
    # Nucleus 5: internal nucleus (should be preserved).
    tile2_mask[2:5, 10:13] = 5
    
    # Nucleus 6: cross-boundary nucleus (extends from overlap region).
    tile2_mask[7:10, 2:7] = 6  # Spans columns 2-6, crosses boundary at 5 (overlap_length).
    
    # Nucleus 7: completely in overlap region (should be deleted in cleanup).
    tile2_mask[14:17, 1:4] = 7  # Completely in columns 1-3, within overlap region.
    
    # Nucleus 8: touches tile2 border.
    tile2_mask[0:3, 15:18] = 8
    
    # Save tiles as .npz files.
    tile1_path = temp_dir / "tile1_cleanup_test.npz"
    tile2_path = temp_dir / "tile2_cleanup_test.npz"
    
    np.savez(tile1_path, mask=tile1_mask)
    np.savez(tile2_path, mask=tile2_mask)
    
    print(f"Tile1 nuclei: {len(np.unique(tile1_mask[tile1_mask > 0]))}")
    print(f"Tile2 nuclei: {len(np.unique(tile2_mask[tile2_mask > 0]))}")
    
    return tile1_path, tile2_path, overlap_length, temp_dir


def test_step3_cleanup():
    """Test proper Step 3 (Cleanup) implementation in merge_tiles_cpu_3step."""
    print("\n=== Testing Step 3 (Cleanup) Implementation ===")
    
    tile1_path, tile2_path, overlap_length, temp_dir = create_cleanup_test_tiles()
    
    try:
        # Load original tiles for analysis.
        tile1_data = np.load(tile1_path)
        tile2_data = np.load(tile2_path)
        original_tile1 = tile1_data["mask"]
        original_tile2 = tile2_data["mask"]
        
        print("\nBEFORE MERGING:")
        print(f"  Tile1 nuclei: {len(np.unique(original_tile1[original_tile1 > 0]))}")
        print(f"  Tile2 nuclei: {len(np.unique(original_tile2[original_tile2 > 0]))}")
        
        # Apply enhanced merging with cleanup.
        updated_tile1, updated_tile2, mapping = merge_tiles_cpu_3step(
            tile1_path, 
            tile2_path, 
            overlap_length, 
            "tile1_left_of_tile2"
        )
        
        print("\nAFTER MERGING:")
        print(f"  Tile1 nuclei: {len(np.unique(updated_tile1[updated_tile1 > 0]))}")
        print(f"  Tile2 nuclei: {len(np.unique(updated_tile2[updated_tile2 > 0]))}")
        print(f"  Cross-boundary preserved: {len(mapping)}")
        
        # Analyze specific nuclei:
        # Check that nucleus 7 (completely in overlap region) was deleted.
        nucleus_7_remaining = np.any(updated_tile2 == 7)
        print(f"  Nucleus 7 (overlap region) remaining: {nucleus_7_remaining}")
        
        # Check that nucleus 6 (cross-boundary) was preserved.
        nucleus_6_remaining = np.any(updated_tile2 == 6)
        print(f"  Nucleus 6 (cross-boundary) remaining: {nucleus_6_remaining}")
        
        # Verify cleanup worked correctly.
        assert not nucleus_7_remaining, "Nucleus 7 should have been deleted (overlap region cleanup)"
        assert nucleus_6_remaining, "Nucleus 6 should have been preserved (cross-boundary)"
        
        print("✓ Step 3 (Cleanup) implementation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Step 3 cleanup test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_cross_boundary_preservation():
    """Test that cross-boundary nuclei are properly preserved."""
    print("\n=== Testing Cross-boundary Nucleus Preservation ===")

    # Create a simple test case.
    tile = np.zeros((15, 15), dtype=np.uint32)
    overlap_length = 5

    # For direction='left', boundary line is at column 5.
    # Buffer zone checks columns 4, 5, 6.
    # Overlap region is columns 0-2 (before column 3 = 5-2).

    # Nucleus 1: cross-boundary (touches boundary line).
    tile[4:6, 4:7] = 1  # Spans columns 4-6, crosses boundary at 5.

    # Nucleus 2: completely in overlap region.
    tile[7:9, 0:2] = 2  # Spans columns 0-1, completely in overlap region.

    boundary_nuclei, overlap_nuclei = _find_border_touching_nuclei(tile, overlap_length, 'left')

    print(f"Boundary line at column: {overlap_length}")
    print(f"Buffer zone columns: {[overlap_length - 1, overlap_length, overlap_length + 1]}")
    print(f"Overlap region ends at column: {overlap_length - 2}")
    print(f"Cross-boundary nuclei (touching boundary): {boundary_nuclei}")
    print(f"Overlap region nuclei (completely beyond): {overlap_nuclei}")

    # Expected: nucleus 1 is cross-boundary, nucleus 2 is in overlap region.
    expected_boundary = {1}
    expected_overlap = {2}
    
    assert boundary_nuclei == expected_boundary, f"Expected {expected_boundary}, got {boundary_nuclei}"
    assert overlap_nuclei == expected_overlap, f"Expected {expected_overlap}, got {overlap_nuclei}"
    
    print("✓ Cross-boundary preservation logic works correctly")
    return True


def main():
    """Run all tests for the enhanced 3-step cleanup implementation."""
    print("Testing Enhanced 3-Step Cleanup Implementation")
    print("=" * 60)
    
    tests = [
        test_enhanced_border_detection,
        test_cross_boundary_preservation,
        test_step3_cleanup,
    ]
    
    passed_tests = 0
    
    try:
        for test_func in tests:
            if test_func():
                passed_tests += 1
        
        print("\n" + "=" * 60)
        print(f"TEST RESULTS: {passed_tests}/{len(tests)} tests passed")
        
        if passed_tests == len(tests):
            print("🎉 ALL TESTS PASSED! Enhanced 3-step cleanup works correctly.")
            print("✓ Enhanced border detection returns two sets")
            print("✓ Cross-boundary nuclei properly preserved")
            print("✓ Step 3 (Cleanup) properly implemented")
            print("✓ Overlap region nuclei correctly deleted")
            return True
        else:
            print(f"❌ {len(tests) - passed_tests} tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

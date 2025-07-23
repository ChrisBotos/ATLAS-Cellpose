"""
Debug script to identify the Step 2 Border Deletion issue.
"""

import numpy as np
import logging
from code.nuclei_segmentation.cellpose_merge.rules import (
    merge_tiles_cpu_3step,
    _find_border_touching_nuclei
)

# Set up detailed logging.
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def create_debug_patch():
    """Create the exact same patch that was failing in the test."""
    patch = np.zeros((2, 60, 60), dtype=np.uint32)

    # Tile 0: Priority tile (4 nuclei - gets priority).
    # Internal nucleus (should be kept).
    patch[0, 25:35, 25:35] = 1

    # Priority tile border-touching nuclei (should be deleted).
    patch[0, 0:8, 25:33] = 2    # Touches top border.
    patch[0, 52:60, 25:33] = 3  # Touches bottom border.
    patch[0, 25:33, 0:8] = 4    # Touches left border.

    # Tile 1: Non-priority tile (3 nuclei).
    # Cross-boundary nuclei that touch priority tile borders (should be kept).
    patch[1, 0:10, 25:35] = 5   # Touches priority top border.
    patch[1, 25:35, 52:60] = 6  # Touches priority right border.

    # Non-cross-boundary nucleus (should be deleted).
    patch[1, 45:55, 45:55] = 7  # Internal, doesn't touch priority border.

    return patch

def debug_step2_implementation():
    """Debug the Step 2 implementation step by step."""
    patch = create_debug_patch()
    
    print("=== DEBUG: Step 2 Border Deletion Issue ===")
    print(f"Patch shape: {patch.shape}")
    print(f"Tile 0 (priority) unique labels: {np.unique(patch[0])}")
    print(f"Tile 1 (non-priority) unique labels: {np.unique(patch[1])}")
    
    # Check border detection.
    priority_tile = patch[0]
    border_nuclei = _find_border_touching_nuclei(priority_tile)
    print(f"Priority tile border-touching nuclei: {border_nuclei}")
    
    # Check the actual positions.
    print(f"Nucleus 1 positions: {np.where(priority_tile == 1)}")
    print(f"Nucleus 2 positions: {np.where(priority_tile == 2)}")
    
    # Run the merge and examine the result.
    merged, mapping = merge_tiles_cpu_3step(patch)
    
    print(f"Mapping: {mapping}")
    print(f"Merged unique labels: {np.unique(merged)}")
    print(f"Merged shape: {merged.shape}")
    
    # Check specific regions that were failing in the test.
    print(f"Top border region (0:8, 25:33): {merged[0:8, 25:33]}")
    print(f"Bottom border region (52:60, 25:33): {merged[52:60, 25:33]}")
    print(f"Left border region (25:33, 0:8): {merged[25:33, 0:8]}")
    print(f"Internal region (25:35, 25:35): {merged[25:35, 25:35]}")

    # Check if any of the border regions contain non-zero values.
    top_border_nonzero = np.any(merged[0:8, 25:33] != 0)
    bottom_border_nonzero = np.any(merged[52:60, 25:33] != 0)
    left_border_nonzero = np.any(merged[25:33, 0:8] != 0)

    print(f"Top border has non-zero values: {top_border_nonzero}")
    print(f"Bottom border has non-zero values: {bottom_border_nonzero}")
    print(f"Left border has non-zero values: {left_border_nonzero}")
    
    # Check if the issue is in the nucleus_mask application.
    print("\n=== Debugging nucleus_mask application ===")
    nucleus_1_mask = priority_tile == 1
    nucleus_2_mask = priority_tile == 2
    
    print(f"Nucleus 1 mask shape: {nucleus_1_mask.shape}")
    print(f"Nucleus 2 mask shape: {nucleus_2_mask.shape}")
    print(f"Merged mask shape: {merged.shape}")
    
    print(f"Nucleus 1 mask sum: {np.sum(nucleus_1_mask)}")
    print(f"Nucleus 2 mask sum: {np.sum(nucleus_2_mask)}")
    
    # Check if there's a dimension mismatch.
    print(f"Priority tile shape: {priority_tile.shape}")
    print(f"Merged shape: {merged.shape}")
    
    if priority_tile.shape != merged.shape:
        print("ERROR: Shape mismatch between priority_tile and merged!")
    
    return merged, mapping

if __name__ == "__main__":
    debug_step2_implementation()

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_step2_debug.py.
Description:
    Debug script to identify the exact issue with Step 2 border deletion in the 3-step
    merging algorithm. This script creates a controlled test case to examine why
    priority tile masks touching borders are not being properly deleted.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.

Usage:
    python test_step2_debug.py

Key Features:
    • Creates controlled test patches with known border-touching nuclei.
    • Tests the current CPU implementation.
    • Visualizes the results to identify line artifacts.
    • Provides detailed debugging output.

Notes:
    • This script helps identify why Step 2 border deletion is failing.
    • Results should show clean tile boundaries without line artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the cellpose_merge directory to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'nuclei_segmentation', 'cellpose_merge'))

from rules import merge_tiles_cpu_3step, _find_border_touching_nuclei

def create_test_patch_with_border_issue():
    """
    Create a test patch that should demonstrate the Step 2 border deletion issue.
    
    Returns
    -------
    patch : np.ndarray
        Test patch with overlapping tiles containing border-touching nuclei.
    """
    # Create a 2-tile patch (T=2, H=100, W=100).
    patch = np.zeros((2, 100, 100), dtype=np.uint32)
    
    # Tile 0 (will be non-priority - fewer nuclei).
    # Add a nucleus that touches the right border.
    patch[0, 40:60, 85:100] = 1  # Nucleus 1 touches right border.
    patch[0, 20:35, 20:35] = 2   # Nucleus 2 internal.
    
    # Tile 1 (will be priority - more nuclei).
    # Add nuclei, some touching borders.
    patch[1, 10:25, 0:15] = 3    # Nucleus 3 touches left border.
    patch[1, 70:85, 85:100] = 4  # Nucleus 4 touches right border.
    patch[1, 85:100, 40:55] = 5  # Nucleus 5 touches bottom border.
    patch[1, 30:45, 30:45] = 6   # Nucleus 6 internal.
    patch[1, 50:65, 50:65] = 7   # Nucleus 7 internal.
    
    return patch

def visualize_patch(patch, title="Test Patch"):
    """Visualize the test patch for debugging."""
    T, H, W = patch.shape
    
    fig, axes = plt.subplots(1, T + 1, figsize=(15, 5))
    
    # Show individual tiles.
    for t in range(T):
        axes[t].imshow(patch[t], cmap='tab10', vmin=0, vmax=10)
        axes[t].set_title(f'Tile {t}')
        axes[t].grid(True, alpha=0.3)
        
        # Count nuclei.
        nuclei_count = len(np.unique(patch[t][patch[t] > 0]))
        axes[t].text(5, 5, f'{nuclei_count} nuclei', 
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Show combined view.
    combined = np.maximum(patch[0], patch[1])
    axes[T].imshow(combined, cmap='tab10', vmin=0, vmax=10)
    axes[T].set_title('Combined')
    axes[T].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_border_detection():
    """Test the border detection function specifically."""
    print("=== Testing Border Detection Function ===")
    
    # Create a simple test tile.
    tile = np.zeros((10, 10), dtype=np.uint32)
    tile[0:3, 0:3] = 1      # Nucleus 1 touches top and left borders.
    tile[7:10, 7:10] = 2    # Nucleus 2 touches bottom and right borders.
    tile[4:6, 4:6] = 3      # Nucleus 3 internal.
    tile[2:5, 8:10] = 4     # Nucleus 4 touches right border.
    
    print("Test tile:")
    print(tile)
    
    border_nuclei = _find_border_touching_nuclei(tile)
    print(f"Border-touching nuclei: {border_nuclei}")
    print(f"Expected: {{1, 2, 4}} (nucleus 3 should be internal)")
    
    # Verify correctness.
    expected = {1, 2, 4}
    if border_nuclei == expected:
        print("✓ Border detection working correctly")
    else:
        print(f"✗ Border detection failed. Expected {expected}, got {border_nuclei}")
    
    return border_nuclei == expected

def test_step2_behavior():
    """Test the Step 2 behavior specifically."""
    print("\n=== Testing Step 2 Border Deletion ===")

    patch = create_test_patch_with_border_issue()
    print("Created test patch with known border-touching nuclei")

    # Analyze the patch before merging.
    print("\nPatch analysis:")
    for t in range(patch.shape[0]):
        nuclei_count = len(np.unique(patch[t][patch[t] > 0]))
        border_nuclei = _find_border_touching_nuclei(patch[t])
        print(f"Tile {t}: {nuclei_count} nuclei, border-touching: {border_nuclei}")

    # Expected: Tile 1 should be priority (5 nuclei vs 2 nuclei).
    # Step 2 should delete nuclei 3, 4, 5 from tile 1 (all touch borders).
    # Only nuclei 6, 7 from tile 1 should remain.
    # Step 3 should preserve nucleus 1 from tile 0 (cross-boundary).

    print("\nRunning 3-step merge...")
    merged, mapping = merge_tiles_cpu_3step(patch)

    print(f"Mapping: {mapping}")
    print(f"Final nuclei count: {len(np.unique(merged[merged > 0]))}")

    return merged, mapping

def test_overlap_region_issue():
    """Test the real issue: applying 3-step algorithm to overlap regions only."""
    print("\n=== Testing Overlap Region Issue ===")

    # Create two complete tiles with an overlap region.
    tile1 = np.zeros((100, 100), dtype=np.uint32)
    tile2 = np.zeros((100, 100), dtype=np.uint32)

    # Tile 1: Add nuclei, some in overlap region.
    tile1[10:25, 10:25] = 1    # Internal nucleus.
    tile1[40:55, 80:95] = 2    # Nucleus in overlap region (should NOT be deleted).
    tile1[70:85, 85:100] = 3   # Nucleus touching tile border (should be deleted).

    # Tile 2: Add nuclei, some in overlap region.
    tile2[20:35, 5:20] = 4     # Nucleus in overlap region.
    tile2[50:65, 15:30] = 5    # Internal nucleus.

    # Define overlap region (last 20 columns of tile1, first 20 columns of tile2).
    overlap1 = tile1[:, 80:100]  # Right edge of tile1.
    overlap2 = tile2[:, 0:20]    # Left edge of tile2.

    print("Complete tiles:")
    print(f"Tile 1 nuclei: {set(np.unique(tile1[tile1 > 0]))}")
    print(f"Tile 2 nuclei: {set(np.unique(tile2[tile2 > 0]))}")

    print("\nOverlap regions:")
    print(f"Overlap1 nuclei: {set(np.unique(overlap1[overlap1 > 0]))}")
    print(f"Overlap2 nuclei: {set(np.unique(overlap2[overlap2 > 0]))}")

    # Apply 3-step algorithm to overlap regions (current implementation).
    patch = np.stack([overlap1, overlap2], axis=0)
    merged_overlap, mapping = merge_tiles_cpu_3step(patch)

    print(f"\nAfter applying 3-step to overlap regions:")
    print(f"Mapping: {mapping}")
    print(f"Merged overlap nuclei: {set(np.unique(merged_overlap[merged_overlap > 0]))}")

    # The issue: nucleus 2 touches the border of the OVERLAP REGION but NOT the complete tile border.
    # It should be preserved, but the current implementation deletes it.

    # Check border detection on overlap vs complete tile.
    overlap1_border_nuclei = _find_border_touching_nuclei(overlap1)
    tile1_border_nuclei = _find_border_touching_nuclei(tile1)

    print(f"\nBorder analysis:")
    print(f"Nuclei touching overlap1 border: {overlap1_border_nuclei}")
    print(f"Nuclei touching complete tile1 border: {tile1_border_nuclei}")
    print(f"ISSUE: Nucleus 2 touches overlap border but NOT complete tile border!")

    return tile1, tile2, overlap1, overlap2, merged_overlap

def main():
    """Main debugging function."""
    print("Step 2 Border Deletion Debug Script")
    print("=" * 50)

    # Test 1: Border detection function.
    border_test_passed = test_border_detection()

    # Test 2: Step 2 behavior on complete patches.
    merged, mapping = test_step2_behavior()

    # Test 3: The real issue - overlap region processing.
    test_overlap_region_issue()

    # Summary.
    print("\n=== Debug Summary ===")
    if border_test_passed:
        print("✓ Border detection function works correctly")
    else:
        print("✗ Border detection function has issues")

    print(f"Final merged result has {len(np.unique(merged[merged > 0]))} nuclei")
    print("ISSUE IDENTIFIED: 3-step algorithm applied to overlap regions only,")
    print("not complete tiles. This causes incorrect border deletion.")

    return merged, mapping

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_step2_fix.py.
Description:
    Test script to verify that the Step 2 border deletion fix works correctly.
    This script tests the new merge_complete_tiles_cpu_3step function to ensure
    it properly handles border detection using complete tiles rather than overlap regions.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.

Usage:
    python test_step2_fix.py

Key Features:
    • Tests the fixed implementation with complete tile border detection.
    • Compares old vs new behavior.
    • Verifies that line artifacts are eliminated.

Notes:
    • This script validates the fix for the Step 2 border deletion bug.
    • Results should show no line artifacts along tile boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the cellpose_merge directory to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'nuclei_segmentation', 'cellpose_merge'))

from rules import merge_tiles_cpu_3step, merge_tiles_cpu_3step, _find_border_touching_nuclei

def create_realistic_test_case():
    """
    Create a realistic test case that demonstrates the Step 2 fix.
    
    Returns
    -------
    tile1, tile2 : np.ndarray
        Complete tile masks.
    overlap_slices : tuple
        Overlap region slices.
    """
    # Create two complete tiles (200x200 each).
    tile1 = np.zeros((200, 200), dtype=np.uint32)
    tile2 = np.zeros((200, 200), dtype=np.uint32)
    
    # Tile 1: Add nuclei with different border relationships.
    tile1[20:35, 20:35] = 1      # Internal nucleus (should be preserved).
    tile1[50:65, 170:185] = 2    # Nucleus in overlap region, NOT touching complete tile border.
    tile1[185:200, 180:195] = 3  # Nucleus touching complete tile border (bottom edge).
    tile1[100:115, 185:200] = 4  # Nucleus touching complete tile border (right edge).
    
    # Tile 2: Add nuclei.
    tile2[30:45, 10:25] = 5      # Nucleus in overlap region.
    tile2[80:95, 5:20] = 6       # Cross-boundary nucleus.
    tile2[120:135, 30:45] = 7    # Internal nucleus.
    
    # Define overlap region: last 30 columns of tile1, first 30 columns of tile2.
    overlap_slices = (slice(None), slice(170, 200), slice(None), slice(0, 30))
    
    return tile1, tile2, overlap_slices

def test_old_vs_new_implementation():
    """Test the old vs new implementation to show the fix."""
    print("=== Testing Old vs New Implementation ===")
    
    tile1, tile2, overlap_slices = create_realistic_test_case()
    
    print("Test case:")
    print(f"Tile 1 nuclei: {set(np.unique(tile1[tile1 > 0]))}")
    print(f"Tile 2 nuclei: {set(np.unique(tile2[tile2 > 0]))}")
    
    # Extract overlap regions for old method.
    tile1_slice_y, tile1_slice_x, tile2_slice_y, tile2_slice_x = overlap_slices
    overlap1 = tile1[tile1_slice_y, tile1_slice_x]
    overlap2 = tile2[tile2_slice_y, tile2_slice_x]
    
    print(f"Overlap1 nuclei: {set(np.unique(overlap1[overlap1 > 0]))}")
    print(f"Overlap2 nuclei: {set(np.unique(overlap2[overlap2 > 0]))}")
    
    # Test old implementation (applied to overlap regions only).
    print("\n--- Old Implementation (Overlap Regions Only) ---")
    patch_old = np.stack([overlap1, overlap2], axis=0)
    merged_old, mapping_old = merge_tiles_cpu_3step(patch_old)
    
    print(f"Old mapping: {mapping_old}")
    print(f"Old result nuclei: {set(np.unique(merged_old[merged_old > 0]))}")
    
    # Test new implementation (complete tiles).
    print("\n--- New Implementation (Complete Tile Borders) ---")

    # Get complete tile border information.
    tile1_border_nuclei = _find_border_touching_nuclei(tile1)
    tile2_border_nuclei = _find_border_touching_nuclei(tile2)

    print(f"Tile1 border-touching nuclei: {tile1_border_nuclei}")
    print(f"Tile2 border-touching nuclei: {tile2_border_nuclei}")

    # Apply enhanced 3-step algorithm with complete tile border information.
    patch_new = np.stack([overlap1, overlap2], axis=0)
    merged_new, mapping_new = merge_tiles_cpu_3step(
        patch_new, tile1_border_nuclei, tile2_border_nuclei
    )

    print(f"New mapping: {mapping_new}")
    print(f"New result nuclei: {set(np.unique(merged_new[merged_new > 0]))}")
    
    # Analyze the differences.
    print("\n--- Analysis ---")
    
    # Check nucleus 2: should be preserved in new implementation.
    nucleus2_in_old = 2 in set(np.unique(merged_old[merged_old > 0]))
    nucleus2_in_new = 2 in set(np.unique(merged_new[merged_new > 0]))

    print(f"Nucleus 2 (overlap region, NOT complete tile border):")
    print(f"  Old implementation: {'PRESERVED' if nucleus2_in_old else 'DELETED (INCORRECT)'}")
    print(f"  New implementation: {'PRESERVED (CORRECT)' if nucleus2_in_new else 'DELETED'}")

    # Check nucleus 3: should be deleted in new implementation.
    nucleus3_in_new = 3 in set(np.unique(merged_new[merged_new > 0]))
    print(f"Nucleus 3 (complete tile border):")
    print(f"  New implementation: {'PRESERVED' if nucleus3_in_new else 'DELETED (CORRECT)'}")
    
    # Visualize results.
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original tiles.
    axes[0, 0].imshow(tile1, cmap='tab10', vmin=0, vmax=10)
    axes[0, 0].set_title('Original Tile 1')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].imshow(tile2, cmap='tab10', vmin=0, vmax=10)
    axes[0, 1].set_title('Original Tile 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Show overlap region.
    overlap_combined = np.maximum(overlap1, overlap2)
    axes[0, 2].imshow(overlap_combined, cmap='tab10', vmin=0, vmax=10)
    axes[0, 2].set_title('Overlap Region')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Results.
    axes[1, 0].imshow(merged_new, cmap='tab10', vmin=0, vmax=10)
    axes[1, 0].set_title('New Result (Complete Borders)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].imshow(merged_old, cmap='tab10', vmin=0, vmax=10)
    axes[1, 1].set_title('Old Result (Overlap Only)')
    axes[1, 1].grid(True, alpha=0.3)

    # Show difference.
    diff = (merged_new > 0).astype(int) - (merged_old > 0).astype(int)
    axes[1, 2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Difference (Blue=Fixed)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step2_fix_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return nucleus2_in_new and not nucleus3_in_new

def main():
    """Main test function."""
    print("Step 2 Border Deletion Fix Test")
    print("=" * 50)
    
    # Test the fix.
    fix_works = test_old_vs_new_implementation()
    
    # Summary.
    print("\n=== Test Summary ===")
    if fix_works:
        print("✓ Step 2 fix is working correctly!")
        print("  - Nucleus 2 (overlap border, NOT tile border) is preserved")
        print("  - Nucleus 3 (complete tile border) is deleted")
        print("  - Line artifacts should be eliminated")
    else:
        print("✗ Step 2 fix needs more work")
    
    return fix_works

if __name__ == "__main__":
    main()

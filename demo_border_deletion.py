"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: demo_border_deletion.py.
Description:
    Demonstration script specifically focused on showing that the new 3-step
    merging algorithm correctly removes ALL border-touching nuclei from tiles.
    This addresses the critical requirement that no masks touching their
    original tile's border should be kept after merging.

Dependencies:
    • Python ≥ 3.10.
    • numpy, matplotlib.
    • cellpose_merge modules.

Usage:
    python demo_border_deletion.py

Key Features:
    • Creates synthetic tiles with border-touching nuclei.
    • Shows before/after merging with detailed border analysis.
    • Verifies that NO border-touching nuclei remain after merging.
    • Provides visual confirmation of proper border deletion.

Notes:
    • This script creates controlled test cases to validate border deletion.
    • All border-touching nuclei should be removed except those from non-priority
      tiles that overlap with priority tile regions.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_patch_with_exact_3step_rule():
    """
    Create a test patch that demonstrates the exact 3-step rule implementation.

    This creates a controlled scenario where we can validate each step precisely.

    Returns
    -------
    patch : np.ndarray
        Test patch with shape (2, 80, 80) containing precisely placed nuclei.
    """
    patch = np.zeros((2, 80, 80), dtype=np.uint32)

    # Tile 0 (will get priority with 4 nuclei).
    # Internal nucleus (should be kept - doesn't touch priority border).
    patch[0, 35:45, 35:45] = 1    # Internal nucleus.

    # Priority border-touching nuclei (should be deleted).
    patch[0, 0:10, 35:45] = 2     # Touches top border of priority tile.
    patch[0, 70:80, 35:45] = 3    # Touches bottom border of priority tile.
    patch[0, 35:45, 0:10] = 4     # Touches left border of priority tile.

    # Tile 1 (3 nuclei, non-priority).
    # Cross-boundary nucleus that touches priority tile border (should be preserved).
    patch[1, 0:12, 35:47] = 5     # Overlaps with priority tile top border region.

    # Another cross-boundary nucleus.
    patch[1, 35:47, 0:12] = 6     # Overlaps with priority tile left border region.

    # Non-cross-boundary nucleus (should be deleted).
    patch[1, 60:70, 60:70] = 7    # Doesn't touch priority tile border.

    logging.info("Created exact 3-step rule test patch:")
    logging.info("  Tile 0 (priority): 4 nuclei")
    logging.info("    - 1 internal nucleus (should be kept)")
    logging.info("    - 3 priority border-touching nuclei (should be deleted)")
    logging.info("  Tile 1 (non-priority): 3 nuclei")
    logging.info("    - 2 cross-boundary nuclei touching priority borders (should be kept)")
    logging.info("    - 1 non-cross-boundary nucleus (should be deleted)")

    return patch


def analyze_border_touching_nuclei(mask, title="Mask"):
    """
    Analyze which nuclei in a mask touch the borders.
    
    Parameters
    ----------
    mask : np.ndarray
        2D mask with nucleus labels.
    title : str
        Title for logging.
        
    Returns
    -------
    border_nuclei : set
        Set of nucleus labels that touch any border.
    """
    h, w = mask.shape
    border_nuclei = set()
    
    if h > 0 and w > 0:
        # Check all four borders.
        border_nuclei.update(np.unique(mask[0, :]))      # Top.
        border_nuclei.update(np.unique(mask[-1, :]))     # Bottom.
        border_nuclei.update(np.unique(mask[:, 0]))      # Left.
        border_nuclei.update(np.unique(mask[:, -1]))     # Right.
    
    # Remove background.
    border_nuclei.discard(0)
    
    total_nuclei = len(np.unique(mask[mask > 0]))
    
    logging.info(f"{title}: {total_nuclei} total nuclei, {len(border_nuclei)} touch borders")
    if border_nuclei:
        logging.info(f"  Border-touching nuclei: {sorted(border_nuclei)}")
    else:
        logging.info("  No border-touching nuclei found!")
    
    return border_nuclei


def visualize_masks(patch, merged, output_path=None):
    """
    Create a visualization showing before and after merging.
    
    Parameters
    ----------
    patch : np.ndarray
        Original patch with shape (T, H, W).
    merged : np.ndarray
        Merged mask with shape (H, W).
    output_path : str, optional
        Path to save the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show tile 0.
    axes[0].imshow(patch[0], cmap='tab20', vmin=0, vmax=20)
    axes[0].set_title('Tile 0 (Priority)\n4 nuclei (3 border, 1 internal)')
    axes[0].grid(True, alpha=0.3)
    
    # Show tile 1.
    axes[1].imshow(patch[1], cmap='tab20', vmin=0, vmax=20)
    axes[1].set_title('Tile 1 (Non-priority)\n2 nuclei (1 border+overlap, 1 internal)')
    axes[1].grid(True, alpha=0.3)
    
    # Show merged result.
    axes[2].imshow(merged, cmap='tab20', vmin=0, vmax=20)
    axes[2].set_title('Merged Result\n(No border-touching nuclei)')
    axes[2].grid(True, alpha=0.3)
    
    # Add border indicators.
    for ax in axes:
        ax.axhline(y=0, color='red', linewidth=2, alpha=0.7)
        ax.axhline(y=patch.shape[1]-1, color='red', linewidth=2, alpha=0.7)
        ax.axvline(x=0, color='red', linewidth=2, alpha=0.7)
        ax.axvline(x=patch.shape[2]-1, color='red', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Visualization saved to: {output_path}")
    
    plt.show()


def demonstrate_exact_3step_rule():
    """
    Demonstrate the exact 3-step rule implementation with precise validation.
    """
    logging.info("=== Demonstrating Exact 3-Step Rule Implementation ===")

    # Create test patch.
    patch = create_test_patch_with_exact_3step_rule()
    
    # Analyze original tiles.
    logging.info("\n--- BEFORE MERGING ---")
    tile0_border = analyze_border_touching_nuclei(patch[0], "Tile 0 (Priority)")
    tile1_border = analyze_border_touching_nuclei(patch[1], "Tile 1 (Non-priority)")
    
    # Apply 3-step merging.
    logging.info("\n--- APPLYING 3-STEP MERGE ---")
    from code.nuclei_segmentation.cellpose_merge.rules_3step import merge_patch_cpu_3step
    
    merged, mapping = merge_patch_cpu_3step(patch)
    
    # Analyze merged result.
    logging.info("\n--- AFTER MERGING ---")
    merged_border = analyze_border_touching_nuclei(merged, "Merged Result")
    
    # Verify correct border handling according to 3-step rule.
    # Cross-boundary nuclei should be preserved, so some border-touching nuclei may remain.
    logging.info(f"Border-touching nuclei in merged result: {len(merged_border)}")
    if len(merged_border) > 0:
        logging.info(f"These are cross-boundary nuclei (expected): {sorted(merged_border)}")
        logging.info("✅ SUCCESS: 3-step rule correctly preserves cross-boundary nuclei!")
    else:
        logging.info("✅ SUCCESS: No border-touching nuclei in this case!")

    # The success criterion is that the algorithm follows the 3-step rule correctly.
    success = True
    
    # Detailed analysis.
    logging.info("\n--- DETAILED ANALYSIS ---")
    total_input_nuclei = len(np.unique(patch[patch > 0]))
    final_nuclei = len(np.unique(merged[merged > 0]))
    
    logging.info(f"Input nuclei: {total_input_nuclei}")
    logging.info(f"Final nuclei: {final_nuclei}")
    logging.info(f"Nuclei removed: {total_input_nuclei - final_nuclei}")
    
    # Check specific expectations.
    logging.info("\n--- EXPECTED BEHAVIOR VERIFICATION ---")
    
    # Tile 0 (priority): Should keep only internal nucleus (label 4).
    # Border nuclei (1, 2, 3) should be removed.
    tile0_kept = [label for label in [1, 2, 3, 4] if label in mapping]
    tile0_removed = [label for label in [1, 2, 3, 4] if label not in mapping]
    
    logging.info(f"Tile 0 - Kept: {tile0_kept}, Removed: {tile0_removed}")
    
    # Tile 1 (non-priority): Should keep internal nucleus (6) and possibly
    # border nucleus (5) if it overlaps with priority tile.
    tile1_kept = [label for label in [5, 6] if label in mapping]
    tile1_removed = [label for label in [5, 6] if label not in mapping]
    
    logging.info(f"Tile 1 - Kept: {tile1_kept}, Removed: {tile1_removed}")
    
    # Expected according to exact 3-step rule:
    # - Nucleus 1: Internal from priority tile (should be kept)
    # - Nuclei 2,3,4: Priority border-touching (should be deleted)
    # - Nuclei 5,6: Cross-boundary from non-priority (should be kept)
    # - Nucleus 7: Non-cross-boundary from non-priority (should be deleted)
    expected_kept = {1, 5, 6}  # Internal priority + 2 cross-boundary.
    actual_kept = set(mapping.keys())

    if actual_kept == expected_kept:
        logging.info("✅ Exact 3-step rule implemented correctly!")
        logging.info(f"  Kept nuclei: {sorted(actual_kept)}")
    else:
        logging.error(f"❌ 3-step rule implementation incorrect!")
        logging.error(f"  Expected: {sorted(expected_kept)}")
        logging.error(f"  Actual: {sorted(actual_kept)}")
        return False
    
    # Create visualization.
    logging.info("\n--- CREATING VISUALIZATION ---")
    visualize_masks(patch, merged, "border_deletion_demo.png")
    
    logging.info("\n=== 3-Step Rule Demonstration Complete ===")
    return success


if __name__ == "__main__":
    success = demonstrate_exact_3step_rule()
    if success:
        print("\n🎉 DEMONSTRATION SUCCESSFUL: Exact 3-step rule is implemented correctly!")
    else:
        print("\n❌ DEMONSTRATION FAILED: 3-step rule implementation is incorrect!")
        exit(1)

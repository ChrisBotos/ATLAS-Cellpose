"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_step2_border_deletion_validation.py.
Description:
    Comprehensive validation test for Step 2 Border Deletion rule in the 4-step
    merge algorithm. This test specifically focuses on verifying that priority
    tile nuclei touching priority tile borders are correctly deleted, while
    non-priority nuclei touching priority tile borders are preserved.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy.
    • cellpose_merge.cpu_merge module.

Usage:
    python -m pytest tests/test_step2_border_deletion_validation.py -v -s

Arguments:
    None (pytest handles test discovery and execution).

Inputs:
    Synthetic tile masks with controlled border-touching scenarios.

Outputs:
    Test results validating Step 2 Border Deletion implementation.

Key Features:
    • Detailed Step 2 Border Deletion rule validation.
    • Synthetic test cases with known expected outcomes.
    • Comprehensive logging of merge decisions for debugging.
    • Edge case testing for corner and multi-border scenarios.
    • Integration with existing 4-step merge implementation.

Notes:
    This test is designed to identify and fix issues where Step 2 Border
    Deletion is not working correctly in the user's results.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Tuple

import numpy as np
import pytest

# Adjust path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.nuclei_segmentation.cellpose_merge.cpu_merge import (
    merge_tiles_cpu_4step,
    _find_border_touching_nuclei,
    _find_nuclei_touching_priority_border
)


class TestStep2BorderDeletionValidation:
    """Comprehensive validation of Step 2 Border Deletion rule."""
    
    def setup_method(self):
        """Set up logging for detailed test output."""
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    def create_step2_validation_patch(self) -> np.ndarray:
        """
        Create a controlled test patch specifically for Step 2 validation.
        
        This patch is designed to test all aspects of Step 2 Border Deletion:
        - Priority tile nuclei touching borders (should be deleted)
        - Priority tile internal nuclei (should be kept)
        - Non-priority nuclei touching priority borders (should be kept)
        - Non-priority nuclei not touching priority borders (should be deleted)
        """
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
    
    def test_step2_priority_border_deletion(self):
        """Test that Step 2 correctly deletes priority tile nuclei touching borders."""
        patch = self.create_step2_validation_patch()
        
        print("\n=== Step 2 Border Deletion Validation ===")
        print("Testing priority tile border-touching nuclei deletion...")
        
        merged, mapping = merge_tiles_cpu_4step(patch)
        
        # Step 2 validation: Priority border nuclei should be deleted.
        assert 2 not in mapping, "Priority top border nucleus (2) should be deleted"
        assert 3 not in mapping, "Priority bottom border nucleus (3) should be deleted"
        assert 4 not in mapping, "Priority left border nucleus (4) should be deleted"
        
        # Priority internal nucleus should be kept.
        assert 1 in mapping, "Priority internal nucleus (1) should be kept"
        
        print("✓ Step 2 Border Deletion working correctly for priority tile")
        
        # Verify the border regions are handled correctly in the merged mask.
        # Top border region may contain cross-boundary nuclei (nucleus 5).
        # We need to check that the priority border nucleus (2) is gone, but cross-boundary nuclei are preserved.

        # Bottom border region should be empty (no cross-boundary nuclei there).
        assert np.all(merged[52:60, 25:33] == 0), "Priority bottom border region should be cleared"

        # Left border region should be empty (no cross-boundary nuclei there).
        assert np.all(merged[25:33, 0:8] == 0), "Priority left border region should be cleared"

        # Top border region should contain cross-boundary nucleus 5 (now with global ID).
        cross_boundary_5_id = mapping[5]
        assert np.any(merged[0:8, 25:33] == cross_boundary_5_id), "Top border should contain cross-boundary nucleus 5"
        
        # Internal region should contain the preserved nucleus.
        assert np.any(merged[25:35, 25:35] > 0), "Priority internal region should contain preserved nucleus"
        
        print("✓ Border regions correctly cleared in merged mask")
    
    def test_step2_cross_boundary_preservation(self):
        """Test that Step 2 preserves non-priority nuclei touching priority borders."""
        patch = self.create_step2_validation_patch()
        
        print("\n=== Cross-boundary Nuclei Preservation ===")
        print("Testing non-priority cross-boundary nuclei preservation...")
        
        merged, mapping = merge_tiles_cpu_4step(patch)
        
        # Cross-boundary nuclei should be preserved.
        assert 5 in mapping, "Cross-boundary nucleus (5) touching priority top border should be kept"
        assert 6 in mapping, "Cross-boundary nucleus (6) touching priority right border should be kept"
        
        # Non-cross-boundary nucleus should be deleted.
        assert 7 not in mapping, "Non-cross-boundary nucleus (7) should be deleted"
        
        print("✓ Cross-boundary nuclei correctly preserved")
        
        # Verify cross-boundary nuclei are present in merged mask.
        cross_boundary_5_id = mapping[5]
        cross_boundary_6_id = mapping[6]
        
        assert np.any(merged == cross_boundary_5_id), "Cross-boundary nucleus 5 should be in merged mask"
        assert np.any(merged == cross_boundary_6_id), "Cross-boundary nucleus 6 should be in merged mask"
        
        # Verify they are in the expected regions.
        assert np.any(merged[0:10, 25:35] == cross_boundary_5_id), "Cross-boundary nucleus 5 should be in top region"
        assert np.any(merged[25:35, 52:60] == cross_boundary_6_id), "Cross-boundary nucleus 6 should be in right region"
        
        print("✓ Cross-boundary nuclei correctly positioned in merged mask")
    
    def test_step2_complete_rule_validation(self):
        """Complete validation of Step 2 rule implementation."""
        patch = self.create_step2_validation_patch()
        
        print("\n=== Complete Step 2 Rule Validation ===")
        
        merged, mapping = merge_tiles_cpu_4step(patch)
        
        # Expected final nuclei: {1, 5, 6} (internal priority + 2 cross-boundary).
        expected_nuclei = {1, 5, 6}
        actual_nuclei = set(mapping.keys())
        
        print(f"Expected nuclei: {expected_nuclei}")
        print(f"Actual nuclei: {actual_nuclei}")
        
        assert actual_nuclei == expected_nuclei, f"Expected {expected_nuclei}, got {actual_nuclei}"
        
        # Verify merged mask contains exactly these nuclei.
        merged_labels = set(np.unique(merged[merged > 0]))
        expected_merged_labels = {mapping[k] for k in expected_nuclei}
        
        print(f"Expected merged labels: {expected_merged_labels}")
        print(f"Actual merged labels: {merged_labels}")
        
        assert merged_labels == expected_merged_labels, "Merged mask should contain exactly the expected nuclei"
        
        print("✓ Complete Step 2 rule validation passed")
    
    def test_border_detection_functions(self):
        """Test the border detection utility functions."""
        print("\n=== Border Detection Functions Validation ===")
        
        # Create a simple test tile.
        tile = np.zeros((20, 20), dtype=np.uint32)
        tile[0:5, 8:12] = 1    # Top border nucleus.
        tile[15:20, 8:12] = 2  # Bottom border nucleus.
        tile[8:12, 0:5] = 3    # Left border nucleus.
        tile[8:12, 15:20] = 4  # Right border nucleus.
        tile[8:12, 8:12] = 5   # Internal nucleus.
        
        border_nuclei = _find_border_touching_nuclei(tile)
        expected_border_nuclei = {1, 2, 3, 4}
        
        print(f"Expected border nuclei: {expected_border_nuclei}")
        print(f"Detected border nuclei: {border_nuclei}")
        
        assert border_nuclei == expected_border_nuclei, "Border detection should identify all border-touching nuclei"
        assert 5 not in border_nuclei, "Internal nucleus should not be detected as border-touching"
        
        print("✓ Border detection functions working correctly")
    
    def test_priority_border_cross_detection(self):
        """Test detection of non-priority nuclei touching priority borders."""
        print("\n=== Priority Border Cross Detection ===")
        
        # Create priority and non-priority tiles.
        priority_tile = np.zeros((20, 20), dtype=np.uint32)
        priority_tile[8:12, 8:12] = 1  # Internal nucleus.
        
        non_priority_tile = np.zeros((20, 20), dtype=np.uint32)
        non_priority_tile[0:6, 8:14] = 2   # Touches priority top border.
        non_priority_tile[14:20, 8:14] = 3 # Touches priority bottom border.
        non_priority_tile[8:14, 0:6] = 4   # Touches priority left border.
        non_priority_tile[8:14, 14:20] = 5 # Touches priority right border.
        non_priority_tile[2:6, 2:6] = 6    # Doesn't touch priority border.
        
        cross_boundary_nuclei = _find_nuclei_touching_priority_border(non_priority_tile, priority_tile)
        expected_cross_boundary = {2, 3, 4, 5}
        
        print(f"Expected cross-boundary nuclei: {expected_cross_boundary}")
        print(f"Detected cross-boundary nuclei: {cross_boundary_nuclei}")
        
        assert cross_boundary_nuclei == expected_cross_boundary, "Should detect all nuclei touching priority borders"
        assert 6 not in cross_boundary_nuclei, "Internal non-priority nucleus should not be cross-boundary"
        
        print("✓ Priority border cross detection working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

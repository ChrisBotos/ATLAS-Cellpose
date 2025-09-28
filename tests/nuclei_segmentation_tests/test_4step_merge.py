"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_4step_merge.py.
Description:
    Comprehensive test suite for the new 4-step tile merging implementation.
    Tests the simplified priority-based merging algorithm that replaces the
    previous 4-step approach with better performance and scientific accuracy.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy, torch (optional).
    • cellpose_merge.cpu_merge module.

Key Features:
    • Priority selection tests for various tile configurations.
    • Border deletion tests with synthetic overlapping nuclei.
    • Integration tests comparing 4-step vs expected results.
    • Performance validation for simplified algorithm.
    • Edge case handling for irregular tile patterns.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Adjust path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.nuclei_segmentation.cellpose_merge.cpu_merge import (
    merge_tiles_cpu_4step,
    _count_nuclei_in_tile,
    _find_border_touching_nuclei
)


class TestUtilityFunctions:
    """Test utility functions for the 4-step algorithm."""
    
    def test_count_nuclei_in_tile(self):
        """Test nucleus counting in tile masks."""
        # Empty tile.
        empty_tile = np.zeros((100, 100), dtype=np.uint32)
        assert _count_nuclei_in_tile(empty_tile) == 0
        
        # Tile with 3 nuclei.
        tile_with_nuclei = np.zeros((100, 100), dtype=np.uint32)
        tile_with_nuclei[10:20, 10:20] = 1
        tile_with_nuclei[30:40, 30:40] = 2
        tile_with_nuclei[50:60, 50:60] = 3
        assert _count_nuclei_in_tile(tile_with_nuclei) == 3
        
        # Tile with background and nuclei.
        mixed_tile = np.zeros((50, 50), dtype=np.uint32)
        mixed_tile[5:15, 5:15] = 1
        mixed_tile[20:30, 20:30] = 5  # Non-consecutive label.
        assert _count_nuclei_in_tile(mixed_tile) == 2
    
    def test_find_border_touching_nuclei(self):
        """Test identification of border-touching nuclei."""
        # Empty tile.
        empty_tile = np.zeros((50, 50), dtype=np.uint32)
        assert _find_border_touching_nuclei(empty_tile) == set()
        
        # Nucleus touching top border.
        tile = np.zeros((50, 50), dtype=np.uint32)
        tile[0:10, 20:30] = 1  # Touches top border.
        tile[25:35, 25:35] = 2  # Internal nucleus.
        border_nuclei = _find_border_touching_nuclei(tile)
        assert border_nuclei == {1}
        
        # Multiple nuclei touching different borders.
        tile = np.zeros((50, 50), dtype=np.uint32)
        tile[0:5, 10:20] = 1    # Top border.
        tile[45:50, 10:20] = 2  # Bottom border.
        tile[10:20, 0:5] = 3    # Left border.
        tile[10:20, 45:50] = 4  # Right border.
        tile[25:35, 25:35] = 5  # Internal.
        border_nuclei = _find_border_touching_nuclei(tile)
        assert border_nuclei == {1, 2, 3, 4}


class TestThreeStepMerging:
    """Test the complete 4-step merge algorithm with exact rule validation."""

    def create_controlled_test_patch(self) -> np.ndarray:
        """
        Create a controlled test patch where we know exactly which nuclei
        touch which borders and can validate the 4-step rule precisely.
        """
        patch = np.zeros((2, 50, 50), dtype=np.uint32)

        # Tile 0 (will be priority with 3 nuclei):
        # Internal nucleus (should be kept).
        patch[0, 20:30, 20:30] = 1

        # Priority tile border-touching nuclei (should be deleted).
        patch[0, 0:8, 10:18] = 2    # Touches top border of priority tile.
        patch[0, 42:50, 10:18] = 3  # Touches bottom border of priority tile.

        # Tile 1 (non-priority with 2 nuclei):
        # Cross-boundary nucleus that touches priority tile border (should be kept).
        patch[1, 0:12, 20:32] = 4   # Overlaps with priority tile top border region.

        # Non-cross-boundary nucleus (should be deleted).
        patch[1, 35:45, 35:45] = 5  # Internal to non-priority, doesn't touch priority border.

        return patch
    
    def test_single_tile_merge(self):
        """Test merging with a single tile (should return unchanged)."""
        single_tile = np.zeros((1, 50, 50), dtype=np.uint32)
        single_tile[0, 10:20, 10:20] = 1
        single_tile[0, 30:40, 30:40] = 2
        
        merged, mapping = merge_tiles_cpu_4step(single_tile)
        
        # Should preserve both nuclei.
        assert np.array_equal(merged, single_tile[0])
        assert len(mapping) == 2
        assert 1 in mapping and 2 in mapping
    
    def test_exact_4step_rule_implementation(self):
        """Test the exact 4-step rule with controlled synthetic data."""
        patch = self.create_controlled_test_patch()

        merged, mapping = merge_tiles_cpu_4step(patch)

        # Validate Step 1: Priority Selection
        # Tile 0 has 3 nuclei, Tile 1 has 2 nuclei -> Tile 0 should be priority.

        # Validate Step 2: Priority Border Deletion
        # Priority tile nuclei touching priority borders should be deleted.
        assert 2 not in mapping, "Priority border nucleus 2 should be deleted"
        assert 3 not in mapping, "Priority border nucleus 3 should be deleted"

        # Priority tile internal nucleus should be kept.
        assert 1 in mapping, "Priority internal nucleus 1 should be kept"

        # Validate Step 3: Non-Priority Preservation Rule
        # Cross-boundary nucleus should be kept.
        assert 4 in mapping, "Cross-boundary nucleus 4 should be kept"

        # Non-cross-boundary nucleus should be deleted.
        assert 5 not in mapping, "Non-cross-boundary nucleus 5 should be deleted"

        # Final validation: only expected nuclei remain.
        expected_remaining = {1, 4}  # Internal priority + cross-boundary.
        actual_remaining = set(mapping.keys())
        assert actual_remaining == expected_remaining, f"Expected {expected_remaining}, got {actual_remaining}"
    
    def test_priority_border_deletion_only(self):
        """Test that ONLY priority tile nuclei touching priority borders are deleted."""
        patch = np.zeros((2, 40, 40), dtype=np.uint32)

        # Tile 0: Priority tile (2 nuclei).
        patch[0, 15:25, 15:25] = 1  # Internal nucleus (should be kept).
        patch[0, 0:8, 15:23] = 2    # Touches top border of priority tile (should be deleted).

        # Tile 1: Non-priority tile (1 nucleus).
        patch[1, 30:38, 30:38] = 3  # Internal to non-priority (should be deleted - not cross-boundary).

        merged, mapping = merge_tiles_cpu_4step(patch)

        # Validate exact rule implementation.
        assert 1 in mapping, "Priority internal nucleus should be kept"
        assert 2 not in mapping, "Priority border nucleus should be deleted"
        assert 3 not in mapping, "Non-priority non-cross-boundary nucleus should be deleted"

        # Verify border region is cleared.
        assert np.all(merged[0:8, 15:23] == 0), "Priority border region should be cleared"

        # Verify internal nucleus is preserved.
        assert np.any(merged[15:25, 15:25] > 0), "Priority internal nucleus should be present"
    
    def test_cross_boundary_nuclei_preservation(self):
        """Test that non-priority nuclei touching PRIORITY TILE borders are preserved."""
        patch = np.zeros((2, 30, 30), dtype=np.uint32)

        # Tile 0: Priority tile (2 nuclei).
        patch[0, 10:20, 10:20] = 1  # Internal (should be kept).
        patch[0, 0:6, 10:16] = 2    # Touches priority top border (should be deleted).

        # Tile 1: Non-priority tile (2 nuclei).
        # Cross-boundary nucleus that touches priority tile border.
        patch[1, 0:8, 10:18] = 3    # Overlaps with priority tile top border (should be kept).

        # Non-cross-boundary nucleus.
        patch[1, 20:28, 20:28] = 4  # Doesn't touch priority border (should be deleted).

        merged, mapping = merge_tiles_cpu_4step(patch)

        # Validate exact preservation rule.
        assert 1 in mapping, "Priority internal nucleus should be kept"
        assert 2 not in mapping, "Priority border nucleus should be deleted"
        assert 3 in mapping, "Cross-boundary nucleus should be kept"
        assert 4 not in mapping, "Non-cross-boundary nucleus should be deleted"

        # Verify cross-boundary nucleus is present.
        cross_boundary_global_id = mapping[3]
        assert np.any(merged == cross_boundary_global_id), "Cross-boundary nucleus should be in merged mask"

        # Verify it actually touches the priority border region.
        cross_boundary_mask = merged == cross_boundary_global_id
        assert np.any(cross_boundary_mask[0:8, 10:18]), "Cross-boundary nucleus should be in expected region"
    
    def test_empty_tiles(self):
        """Test handling of empty tiles."""
        patch = np.zeros((2, 50, 50), dtype=np.uint32)
        # Both tiles are empty.
        
        merged, mapping = merge_tiles_cpu_4step(patch)
        
        # Should return empty merged mask.
        assert np.all(merged == 0)
        assert len(mapping) == 0
    
    def test_memory_safety_check(self):
        """Test that large patches trigger safety checks."""
        # Create a patch that would exceed memory limits.
        large_patch = np.zeros((2, 20000, 20000), dtype=np.uint32)
        
        with pytest.raises(RuntimeError, match="exceeding safe CPU limit"):
            merge_tiles_cpu_4step(large_patch)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_three_tile_merge(self):
        """Test merging with three tiles."""
        patch = np.zeros((3, 50, 50), dtype=np.uint32)
        
        # Tile 0: 1 nucleus.
        patch[0, 10:20, 10:20] = 1
        
        # Tile 1: 2 nuclei (should get priority).
        patch[1, 30:40, 30:40] = 2
        patch[1, 0:10, 0:10] = 3  # Border-touching.
        
        # Tile 2: 1 nucleus.
        patch[2, 40:50, 40:50] = 4
        
        merged, mapping = merge_tiles_cpu_4step(patch)
        
        # Should handle multiple tiles correctly.
        assert merged.shape == (50, 50)
        assert len(np.unique(merged[merged > 0])) >= 1
    
    def test_identical_nucleus_counts(self):
        """Test behavior when tiles have identical nucleus counts."""
        patch = np.zeros((2, 50, 50), dtype=np.uint32)

        # Both tiles have 2 nuclei each.
        patch[0, 10:20, 10:20] = 1
        patch[0, 30:40, 30:40] = 2

        patch[1, 15:25, 15:25] = 3
        patch[1, 35:45, 35:45] = 4

        merged, mapping = merge_tiles_cpu_4step(patch)

        # Should handle tie-breaking (first tile gets priority by default).
        assert merged.shape == (50, 50)
        assert len(mapping) >= 1

    def test_comprehensive_4step_rule_validation(self):
        """Comprehensive test validating all aspects of the 4-step rule."""
        patch = np.zeros((2, 60, 60), dtype=np.uint32)

        # Tile 0: Priority tile (4 nuclei - gets priority).
        patch[0, 25:35, 25:35] = 1  # Internal (should be kept).
        patch[0, 0:8, 25:33] = 2    # Touches top border (should be deleted).
        patch[0, 52:60, 25:33] = 3  # Touches bottom border (should be deleted).
        patch[0, 25:33, 0:8] = 4    # Touches left border (should be deleted).

        # Tile 1: Non-priority tile (3 nuclei).
        patch[1, 0:10, 25:35] = 5   # Touches priority top border (cross-boundary - should be kept).
        patch[1, 45:55, 45:55] = 6  # Internal, doesn't touch priority border (should be deleted).
        patch[1, 25:35, 52:60] = 7  # Touches priority right border (cross-boundary - should be kept).

        merged, mapping = merge_tiles_cpu_4step(patch)

        # Step 1 validation: Priority selection.
        # Tile 0 has 4 nuclei, Tile 1 has 3 nuclei -> Tile 0 is priority.

        # Step 2 validation: Priority border deletion.
        assert 1 in mapping, "Priority internal nucleus should be kept"
        assert 2 not in mapping, "Priority top border nucleus should be deleted"
        assert 3 not in mapping, "Priority bottom border nucleus should be deleted"
        assert 4 not in mapping, "Priority left border nucleus should be deleted"

        # Step 3 validation: Non-priority preservation rule.
        assert 5 in mapping, "Cross-boundary nucleus (top) should be kept"
        assert 6 not in mapping, "Non-cross-boundary nucleus should be deleted"
        assert 7 in mapping, "Cross-boundary nucleus (right) should be kept"

        # Final validation.
        expected_nuclei = {1, 5, 7}  # Internal priority + 2 cross-boundary.
        actual_nuclei = set(mapping.keys())
        assert actual_nuclei == expected_nuclei, f"Expected {expected_nuclei}, got {actual_nuclei}"

        # Verify merged mask contains exactly these nuclei.
        merged_labels = set(np.unique(merged[merged > 0]))
        expected_merged_labels = {mapping[k] for k in expected_nuclei}
        assert merged_labels == expected_merged_labels, "Merged mask should contain exactly the expected nuclei"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

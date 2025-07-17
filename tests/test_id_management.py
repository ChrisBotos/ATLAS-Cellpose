"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Test Name: test_id_management.py.
Description:
    Comprehensive tests for the enhanced global ID management system in merge_tiles.py.
    Tests the segmented ID allocation strategy that prevents uint32 overflow issues
    and minimizes ID conflicts during counter resets.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy.

Usage:
    pytest tests/test_id_management.py -v

Arguments:
    None.

Inputs:
    Test data simulating various ID counter scenarios.

Outputs:
    Test results validating ID management functionality.

Key Features:
    • Tests segmented ID allocation strategy.
    • Validates overflow prevention mechanisms.
    • Checks for ID conflict minimization.
    • Tests edge cases and boundary conditions.

Notes:
    • Tests both normal operation and overflow scenarios.
    • Validates that resets use different segments to avoid conflicts.
    • Ensures proper handling of large patch values.
"""

import traceback
import pytest
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to import the module.
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.merge_tiles import _get_next_safe_gid_range


class TestIDManagement:
    """Test suite for enhanced global ID management system."""
    
    def test_normal_operation(self):
        """Test normal ID allocation without overflow."""
        current_gid = 1000
        patch_max = 500
        max_safe_gid = 2**31 - 1
        reset_count = 0
        segment_size = max_safe_gid // 10
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )
        
        assert new_gid == current_gid + patch_max
        assert gid_offset == current_gid
        assert was_reset is False
        
    def test_overflow_prevention(self):
        """Test that overflow is properly detected and handled."""
        current_gid = 2**31 - 100  # Close to limit.
        patch_max = 200  # Would cause overflow.
        max_safe_gid = 2**31 - 1
        reset_count = 0
        segment_size = max_safe_gid // 10
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )
        
        assert was_reset is True
        assert gid_offset == segment_size + 1  # First reset goes to segment 1.
        assert new_gid == gid_offset + patch_max
        
    def test_multiple_resets(self):
        """Test that multiple resets use different segments."""
        patch_max = 1000
        max_safe_gid = 2**31 - 1
        segment_size = max_safe_gid // 10
        
        # First reset.
        current_gid = max_safe_gid - 500
        new_gid1, gid_offset1, was_reset1 = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, 0, segment_size
        )
        
        assert was_reset1 is True
        assert gid_offset1 == segment_size + 1  # Segment 1.
        
        # Second reset.
        current_gid = max_safe_gid - 500
        new_gid2, gid_offset2, was_reset2 = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, 1, segment_size
        )
        
        assert was_reset2 is True
        assert gid_offset2 == 2 * segment_size + 1  # Segment 2.
        assert gid_offset2 != gid_offset1  # Different segments.
        
    def test_segment_boundaries(self):
        """Test behavior at segment boundaries."""
        max_safe_gid = 2**31 - 1
        segment_size = max_safe_gid // 10
        
        # Test at the boundary of the first segment.
        current_gid = segment_size - 100
        patch_max = 200  # Would exceed first segment.
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, 0, segment_size
        )
        
        # Should not reset because we're still within max_safe_gid.
        assert was_reset is False
        assert new_gid == current_gid + patch_max
        
    def test_large_patch_values(self):
        """Test handling of very large patch values."""
        current_gid = 1000
        patch_max = 1000000  # Large patch.
        max_safe_gid = 2**31 - 1
        reset_count = 0
        segment_size = max_safe_gid // 10
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )
        
        # Should handle large patches normally if within limits.
        assert new_gid == current_gid + patch_max
        assert was_reset is False
        
    def test_zero_patch_max(self):
        """Test handling of empty patches (patch_max = 0)."""
        current_gid = 1000
        patch_max = 0
        max_safe_gid = 2**31 - 1
        reset_count = 0
        segment_size = max_safe_gid // 10
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )
        
        assert new_gid == current_gid  # No change for empty patch.
        assert gid_offset == current_gid
        assert was_reset is False
        
    def test_extreme_reset_count(self):
        """Test behavior with many resets (edge case)."""
        current_gid = 2**31 - 100
        patch_max = 200
        max_safe_gid = 2**31 - 1
        reset_count = 50  # Many resets already.
        segment_size = max_safe_gid // 10

        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )

        assert was_reset is True
        # With high reset count, should fall back to simple reset.
        expected_segment_start = (reset_count + 1) * segment_size + 1
        if expected_segment_start + patch_max > 2**32 - 1:
            # Should fall back to simple reset.
            assert gid_offset == 1
        else:
            assert gid_offset == expected_segment_start
        
    def test_uint32_absolute_limit(self):
        """Test fallback when approaching absolute uint32 limit."""
        current_gid = 2**31 - 100
        patch_max = 200
        max_safe_gid = 2**31 - 1
        reset_count = 100  # Would exceed uint32 space.
        segment_size = max_safe_gid // 10
        
        new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
            current_gid, patch_max, max_safe_gid, reset_count, segment_size
        )
        
        assert was_reset is True
        # Should fall back to simple reset when segments are exhausted.
        assert gid_offset == 1
        assert new_gid == 1 + patch_max


class TestIDManagementIntegration:
    """Integration tests for ID management in realistic scenarios."""
    
    def test_realistic_kidney_analysis_scenario(self):
        """Test ID management in a realistic kidney tissue analysis scenario."""
        # Simulate processing a large kidney slice with many nuclei.
        max_safe_gid = 2**31 - 1
        segment_size = max_safe_gid // 10
        
        current_gid = 1
        reset_count = 0
        total_nuclei_processed = 0
        
        # Simulate processing 1000 patches with varying nucleus counts.
        patch_sizes = np.random.randint(1000, 50000, 1000)  # Realistic nucleus counts per patch.
        
        for i, patch_max in enumerate(patch_sizes):
            new_gid, gid_offset, was_reset = _get_next_safe_gid_range(
                current_gid, patch_max, max_safe_gid, reset_count, segment_size
            )
            
            if was_reset:
                reset_count += 1
                print(f"Reset #{reset_count} occurred at patch {i+1}")
                
            current_gid = new_gid
            total_nuclei_processed += patch_max
            
            # Verify IDs are within valid range.
            assert gid_offset > 0
            assert gid_offset < 2**32
            assert new_gid > gid_offset
            
        print(f"Processed {total_nuclei_processed:,} nuclei across {len(patch_sizes)} patches")
        print(f"Required {reset_count} counter resets")
        
        # Verify we handled the processing without errors.
        assert total_nuclei_processed > 0
        assert current_gid > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

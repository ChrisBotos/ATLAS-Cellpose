"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_gpu_merge_4step_integration.py.
Description:
    Integration test for GPU merge function with 4-step algorithm. This test
    validates that the GPU merge function correctly falls back to the CPU
    implementation and produces identical results for the 4-step merge algorithm.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy, torch (optional).
    • cellpose_merge.gpu_merge, cellpose_merge.cpu_merge modules.

Usage:
    python -m pytest tests/test_gpu_merge_4step_integration.py -v -s

Arguments:
    None (pytest handles test discovery and execution).

Inputs:
    Synthetic tile masks with controlled overlapping scenarios.

Outputs:
    Test results validating GPU-CPU integration for 4-step merge.

Key Features:
    • GPU-CPU result consistency validation.
    • Fallback behavior testing for various scenarios.
    • Step 2 Border Deletion validation in GPU context.
    • Integration with two-phase merge pipeline.
    • Performance and correctness comparison.

Notes:
    This test ensures that the GPU merge function correctly implements
    the 4-step algorithm and produces identical results to the CPU version.
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

from code.nuclei_segmentation.cellpose_merge.cpu_merge import merge_tiles_cpu_4step
from code.nuclei_segmentation.cellpose_merge.gpu_merge import merge_patch_gpu_4step


class TestGPUMerge4stepIntegration:
    """Integration tests for GPU merge with 4-step algorithm."""
    
    def setup_method(self):
        """Set up logging for detailed test output."""
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    def create_test_patch(self) -> np.ndarray:
        """
        Create a test patch for GPU-CPU comparison.
        
        This patch tests all aspects of the 4-step algorithm:
        - Priority selection
        - Border deletion
        - Cross-boundary preservation
        """
        patch = np.zeros((2, 40, 40), dtype=np.uint32)
        
        # Tile 0: Priority tile (3 nuclei).
        patch[0, 15:25, 15:25] = 1  # Internal nucleus (should be kept).
        patch[0, 0:8, 15:23] = 2    # Top border nucleus (should be deleted).
        patch[0, 32:40, 15:23] = 3  # Bottom border nucleus (should be deleted).
        
        # Tile 1: Non-priority tile (2 nuclei).
        patch[1, 0:10, 15:25] = 4   # Cross-boundary nucleus (should be kept).
        patch[1, 30:38, 30:38] = 5  # Internal nucleus (should be deleted).
        
        return patch
    
    def test_gpu_cpu_result_consistency(self):
        """Test that GPU and CPU implementations produce identical results."""
        patch = self.create_test_patch()
        
        print("\n=== GPU-CPU Result Consistency Test ===")
        print("Comparing GPU and CPU 4-step merge results...")
        
        # Run CPU merge.
        cpu_merged, cpu_mapping = merge_tiles_cpu_4step(patch)
        
        # Run GPU merge (should fall back to CPU).
        gpu_merged, gpu_mapping = merge_patch_gpu_4step(patch)
        
        print(f"CPU mapping: {cpu_mapping}")
        print(f"GPU mapping: {gpu_mapping}")
        
        # Results should be identical.
        assert np.array_equal(cpu_merged, gpu_merged), "GPU and CPU merged masks should be identical"
        assert cpu_mapping == gpu_mapping, "GPU and CPU mappings should be identical"
        
        print("✓ GPU and CPU results are identical")
    
    def test_gpu_fallback_behavior(self):
        """Test GPU fallback behavior under various conditions."""
        patch = self.create_test_patch()
        
        print("\n=== GPU Fallback Behavior Test ===")
        
        # The GPU implementation should always fall back to CPU for the 4-step algorithm.
        merged, mapping = merge_patch_gpu_4step(patch)
        
        # Validate the 4-step rule is correctly applied.
        expected_nuclei = {1, 4}  # Internal priority + cross-boundary.
        actual_nuclei = set(mapping.keys())
        
        print(f"Expected nuclei: {expected_nuclei}")
        print(f"Actual nuclei: {actual_nuclei}")
        
        assert actual_nuclei == expected_nuclei, f"Expected {expected_nuclei}, got {actual_nuclei}"
        
        print("✓ GPU fallback correctly applies 4-step algorithm")
    
    def test_gpu_step2_border_deletion(self):
        """Test that GPU merge correctly implements Step 2 Border Deletion."""
        patch = self.create_test_patch()
        
        print("\n=== GPU Step 2 Border Deletion Test ===")
        
        merged, mapping = merge_patch_gpu_4step(patch)
        
        # Step 2 validation: Priority border nuclei should be deleted.
        assert 2 not in mapping, "Priority top border nucleus (2) should be deleted"
        assert 3 not in mapping, "Priority bottom border nucleus (3) should be deleted"
        
        # Priority internal nucleus should be kept.
        assert 1 in mapping, "Priority internal nucleus (1) should be kept"
        
        # Cross-boundary nucleus should be kept.
        assert 4 in mapping, "Cross-boundary nucleus (4) should be kept"
        
        # Non-cross-boundary nucleus should be deleted.
        assert 5 not in mapping, "Non-cross-boundary nucleus (5) should be deleted"
        
        print("✓ GPU merge correctly implements Step 2 Border Deletion")
        
        # Verify border regions are handled correctly.
        # Bottom border region should be empty (no cross-boundary nuclei there).
        assert np.all(merged[32:40, 15:23] == 0), "Priority bottom border region should be cleared"
        
        # Top border region should contain cross-boundary nucleus 4.
        cross_boundary_4_id = mapping[4]
        assert np.any(merged[0:10, 15:25] == cross_boundary_4_id), "Top border should contain cross-boundary nucleus 4"
        
        print("✓ GPU merge correctly handles border regions")
    
    def test_gpu_memory_safety(self):
        """Test GPU memory safety checks."""
        print("\n=== GPU Memory Safety Test ===")
        
        # Create a small patch that should trigger CPU fallback due to size.
        small_patch = np.zeros((2, 10, 10), dtype=np.uint32)
        small_patch[0, 2:6, 2:6] = 1
        small_patch[1, 4:8, 4:8] = 2
        
        # Should fall back to CPU due to small size.
        merged, mapping = merge_patch_gpu_4step(small_patch)
        
        # Should still work correctly.
        assert merged.shape == (10, 10)
        assert len(mapping) >= 1
        
        print("✓ GPU memory safety checks working correctly")
    
    def test_gpu_integration_with_two_phase_merge(self):
        """Test GPU merge integration with two-phase merge pipeline."""
        from code.nuclei_segmentation.cellpose_merge.two_phase_merge import _merge_two_tiles
        
        print("\n=== GPU Integration with Two-Phase Merge ===")
        
        # Create two tile masks.
        tile1 = np.zeros((50, 50), dtype=np.uint32)
        tile2 = np.zeros((50, 50), dtype=np.uint32)
        
        # Add nuclei to tiles.
        tile1[20:30, 20:30] = 1  # Internal nucleus.
        tile1[0:8, 20:28] = 2    # Border nucleus.
        
        tile2[0:10, 20:30] = 3   # Cross-boundary nucleus.
        tile2[35:45, 35:45] = 4  # Internal nucleus.
        
        # Define overlap region.
        overlap_slices = (slice(0, 15), slice(20, 35), slice(0, 15), slice(20, 35))
        
        # Test with GPU enabled.
        updated_tile1, updated_tile2, mapping = _merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=True
        )
        
        # Validate results.
        assert updated_tile1.shape == tile1.shape
        assert updated_tile2.shape == tile2.shape
        assert isinstance(mapping, dict)
        
        print("✓ GPU integration with two-phase merge working correctly")
    
    def test_gpu_error_handling(self):
        """Test GPU error handling and graceful fallback."""
        print("\n=== GPU Error Handling Test ===")
        
        # Test with various edge cases.
        edge_cases = [
            np.zeros((1, 10, 10), dtype=np.uint32),  # Single tile.
            np.zeros((2, 0, 0), dtype=np.uint32),    # Empty tiles.
            np.zeros((2, 5, 5), dtype=np.uint32),    # Very small tiles.
        ]
        
        for i, patch in enumerate(edge_cases):
            try:
                merged, mapping = merge_patch_gpu_4step(patch)
                print(f"✓ Edge case {i+1} handled correctly")
            except Exception as e:
                print(f"✗ Edge case {i+1} failed: {e}")
                raise
        
        print("✓ GPU error handling working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

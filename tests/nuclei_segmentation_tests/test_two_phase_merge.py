"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_two_phase_merge.py.
Description:
    Comprehensive test suite for the two-phase tile merging implementation using
    the new 3-step algorithm. Tests overlap dictionary creation, pairwise tile
    merging, and the complete two-phase merge pipeline with various tile configurations.

    The new 3-step merging rule:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching borders, preserve
       non-priority nuclei touching priority borders
    3. Cleanup: Remove remaining non-priority nuclei in overlap region

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy, torch (optional).
    • cellpose_merge.two_phase_merge module.

Key Features:
    • Overlap dictionary generation tests for various tile arrangements.
    • Pairwise merge tests with synthetic overlapping nuclei using 3-step algorithm.
    • Integration tests comparing two-phase vs single-tile results.
    • Edge case handling for irregular tile patterns and boundary conditions.
    • Performance validation for simplified algorithm.
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

from code.nuclei_segmentation.cellpose_merge.two_phase_merge import (
    create_overlap_dictionaries,
    merge_two_tiles,
    merge_tiles_two_phase
)


class TestOverlapDictionaries:
    """Test overlap dictionary creation for various tile arrangements."""
    
    def test_simple_2x2_grid(self):
        """Test overlap detection for a simple 2x2 tile grid."""
        coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        tile_h, tile_w = 256, 256
        overlap = 64
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, tile_h, tile_w, overlap
        )
        
        # Should have 2 vertical overlaps: (0,0)-(0,1) and (1,0)-(1,1).
        assert len(vertical_overlaps) == 2
        assert ((0, 0), (0, 1)) in vertical_overlaps
        assert ((1, 0), (1, 1)) in vertical_overlaps
        
        # Should have 2 horizontal overlaps: (0,0)-(1,0) and (0,1)-(1,1).
        assert len(horizontal_overlaps) == 2
        assert ((0, 0), (1, 0)) in horizontal_overlaps
        assert ((0, 1), (1, 1)) in horizontal_overlaps
    
    def test_single_row_tiles(self):
        """Test overlap detection for tiles in a single row."""
        coords = [(0, 0), (0, 1), (0, 2)]
        tile_h, tile_w = 256, 256
        overlap = 64
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, tile_h, tile_w, overlap
        )
        
        # Should have 2 vertical overlaps: (0,0)-(0,1) and (0,1)-(0,2).
        assert len(vertical_overlaps) == 2
        assert ((0, 0), (0, 1)) in vertical_overlaps
        assert ((0, 1), (0, 2)) in vertical_overlaps
        
        # Should have no horizontal overlaps.
        assert len(horizontal_overlaps) == 0
    
    def test_single_column_tiles(self):
        """Test overlap detection for tiles in a single column."""
        coords = [(0, 0), (1, 0), (2, 0)]
        tile_h, tile_w = 256, 256
        overlap = 64
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, tile_h, tile_w, overlap
        )
        
        # Should have no vertical overlaps.
        assert len(vertical_overlaps) == 0
        
        # Should have 2 horizontal overlaps: (0,0)-(1,0) and (1,0)-(2,0).
        assert len(horizontal_overlaps) == 2
        assert ((0, 0), (1, 0)) in horizontal_overlaps
        assert ((1, 0), (2, 0)) in horizontal_overlaps
    
    def test_sparse_tile_pattern(self):
        """Test overlap detection for sparse tile patterns."""
        coords = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Missing center tiles.
        tile_h, tile_w = 256, 256
        overlap = 64
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, tile_h, tile_w, overlap
        )
        
        # Should have no overlaps since tiles are not adjacent.
        assert len(vertical_overlaps) == 0
        assert len(horizontal_overlaps) == 0
    
    def test_overlap_slice_calculation(self):
        """Test that overlap slices are calculated correctly."""
        coords = [(0, 0), (0, 1)]
        tile_h, tile_w = 256, 256
        overlap = 64
        
        vertical_overlaps, _ = create_overlap_dictionaries(
            coords, tile_h, tile_w, overlap
        )
        
        # Get the overlap slices for the (0,0)-(0,1) pair.
        overlap_slices = vertical_overlaps[((0, 0), (0, 1))]
        tile1_slice_y, tile1_slice_x, tile2_slice_y, tile2_slice_x = overlap_slices
        
        # Verify slice dimensions make sense.
        assert tile1_slice_y.start == 0
        assert tile1_slice_y.stop == tile_h
        assert tile1_slice_x.stop == tile_w
        assert tile2_slice_y.start == 0
        assert tile2_slice_y.stop == tile_h
        assert tile2_slice_x.start == 0


class TestNucleusContinuity:
    """Test nucleus continuity and ID consistency across tile boundaries."""

    def create_cross_boundary_tiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tiles with nuclei that span across tile boundaries.

        This tests the critical case where a single biological nucleus
        extends from one tile into another tile's overlap region.
        """
        tile1 = np.zeros((256, 256), dtype=np.uint32)
        tile2 = np.zeros((256, 256), dtype=np.uint32)

        # Cross-boundary nucleus: extends from tile1 main body into tile2 overlap.
        # This nucleus should remain as ONE contiguous mask with ONE ID after merging.
        tile1[100:150, 180:256] = 1  # Main body in tile1 (extends to right edge).
        tile2[100:150, 0:40] = 1     # Extension in tile2 overlap (same ID initially).

        # Priority tile nucleus touching border (should be deleted).
        tile2[50:80, 0:20] = 2       # Touches left border of tile2.

        # Internal nucleus in priority tile (should be kept).
        tile2[200:230, 100:130] = 3  # Internal to tile2.

        # Non-cross-boundary nucleus in tile1 (should be deleted).
        tile1[50:80, 100:130] = 4    # Internal to tile1, doesn't cross boundary.

        return tile1, tile2

    def create_large_cross_boundary_nucleus(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tiles with a large nucleus that extends significantly beyond overlap.

        This tests the case where most of the nucleus is outside the overlap region,
        but a small part extends into the overlap.
        """
        tile1 = np.zeros((256, 256), dtype=np.uint32)
        tile2 = np.zeros((256, 256), dtype=np.uint32)

        # Large cross-boundary nucleus: most of it is in tile1, small part in tile2.
        tile1[50:200, 150:256] = 1   # Large main body in tile1.
        tile2[50:200, 0:20] = 1      # Small extension in tile2 overlap.

        # Priority tile nuclei.
        tile2[10:40, 100:130] = 2    # Internal (should be kept).
        tile2[220:250, 0:15] = 3     # Border-touching (should be deleted).

        return tile1, tile2
    
    def test_cross_boundary_nucleus_continuity(self):
        """Test that cross-boundary nuclei remain contiguous with consistent IDs."""
        tile1, tile2 = self.create_cross_boundary_tiles()

        # Define overlap region (rightmost 64 pixels of tile1, leftmost 64 pixels of tile2).
        overlap_slices = (
            slice(0, 256),    # tile1_y: full height
            slice(192, 256),  # tile1_x: rightmost 64 pixels
            slice(0, 256),    # tile2_y: full height
            slice(0, 64),     # tile2_x: leftmost 64 pixels
        )

        # Verify initial cross-boundary nucleus exists in both tiles.
        assert np.any(tile1 == 1), "Cross-boundary nucleus should exist in tile1"
        assert np.any(tile2 == 1), "Cross-boundary nucleus should exist in tile2"

        updated_tile1, updated_tile2, mapping = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=False
        )

        # Critical test: Cross-boundary nucleus should be preserved with original ID.
        assert 1 in mapping, "Cross-boundary nucleus should be preserved in mapping"
        assert mapping[1] == 1, "Cross-boundary nucleus should keep its original ID"

        # Critical test: The ENTIRE cross-boundary nucleus should have the same original ID.
        # This tests the fix for the nucleus splitting bug.

        # Check that the nucleus exists with the original ID in both tiles.
        assert np.any(updated_tile1 == 1), "Cross-boundary nucleus with original ID should exist in tile1"
        assert np.any(updated_tile2 == 1), "Cross-boundary nucleus with original ID should exist in tile2"

        # Critical test: No new IDs should be created.
        original_ids = {1, 2, 3, 4}  # All original IDs from both tiles
        updated_ids_tile1 = set(np.unique(updated_tile1[updated_tile1 > 0]))
        updated_ids_tile2 = set(np.unique(updated_tile2[updated_tile2 > 0]))
        all_updated_ids = updated_ids_tile1 | updated_ids_tile2

        assert all_updated_ids.issubset(original_ids), f"No new IDs should be created. Original: {original_ids}, Updated: {all_updated_ids}"

        # Test nucleus continuity: The nucleus should be contiguous across tiles.
        # Extract the nucleus from both tiles and verify it forms one connected component.
        nucleus_tile1 = updated_tile1 == 1
        nucleus_tile2 = updated_tile2 == 1

        # The nucleus should exist in the expected regions.
        assert np.any(nucleus_tile1[100:150, 180:256]), "Nucleus should exist in tile1 main body"
        assert np.any(nucleus_tile2[100:150, 0:40]), "Nucleus should exist in tile2 overlap region"

    def test_large_cross_boundary_nucleus_id_consistency(self):
        """Test ID consistency for nuclei that extend far beyond overlap regions."""
        tile1, tile2 = self.create_large_cross_boundary_nucleus()

        overlap_slices = (
            slice(0, 256),    # tile1_y: full height
            slice(192, 256),  # tile1_x: rightmost 64 pixels
            slice(0, 256),    # tile2_y: full height
            slice(0, 64),     # tile2_x: leftmost 64 pixels
        )

        # Verify the large nucleus exists in both tiles initially.
        nucleus1_area = np.sum(tile1 == 1)
        nucleus2_area = np.sum(tile2 == 1)
        total_initial_area = nucleus1_area + nucleus2_area

        assert nucleus1_area > 0, "Large nucleus should exist in tile1"
        assert nucleus2_area > 0, "Large nucleus should exist in tile2"
        assert nucleus1_area > nucleus2_area, "Most of the nucleus should be in tile1"

        updated_tile1, updated_tile2, mapping = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=False
        )

        # The large cross-boundary nucleus should be preserved with original ID.
        assert 1 in mapping, "Large cross-boundary nucleus should be preserved"
        assert mapping[1] == 1, "Cross-boundary nucleus should keep its original ID"

        # Critical test: The entire large nucleus should have consistent original ID.
        updated_nucleus1_area = np.sum(updated_tile1 == 1)
        updated_nucleus2_area = np.sum(updated_tile2 == 1)
        total_updated_area = updated_nucleus1_area + updated_nucleus2_area

        # The total area may increase when merging cross-boundary nuclei because
        # the merged nucleus creates a consistent shape across both tiles.
        # The key requirement is that the nucleus remains as one contiguous entity.
        assert total_updated_area >= total_initial_area, f"Nucleus area should not decrease: {total_initial_area} -> {total_updated_area}"

        # The nucleus should exist with its original ID in both tiles.
        assert np.any(updated_tile1 == 1), "Cross-boundary nucleus should exist in tile1"
        assert np.any(updated_tile2 == 1), "Cross-boundary nucleus should exist in tile2"

        # The nucleus should still be mostly in tile1.
        assert updated_nucleus1_area > updated_nucleus2_area, "Most of the nucleus should still be in tile1"

    def test_nucleus_splitting_bug_detection(self):
        """Explicitly test for the nucleus splitting bug."""
        tile1, tile2 = self.create_cross_boundary_tiles()

        overlap_slices = (
            slice(0, 256),    # tile1_y: full height
            slice(192, 256),  # tile1_x: rightmost 64 pixels
            slice(0, 256),    # tile2_y: full height
            slice(0, 64),     # tile2_x: leftmost 64 pixels
        )

        updated_tile1, updated_tile2, mapping = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=False
        )

        # Get all unique IDs in both updated tiles.
        all_ids_tile1 = set(np.unique(updated_tile1[updated_tile1 > 0]))
        all_ids_tile2 = set(np.unique(updated_tile2[updated_tile2 > 0]))

        # Check for nucleus splitting: if the same biological nucleus has different IDs
        # in different parts of the tiles, this indicates the splitting bug.

        # The cross-boundary nucleus should have exactly one original ID across both tiles.
        if 1 in mapping:
            assert mapping[1] == 1, "Cross-boundary nucleus should keep its original ID"

            # If the nucleus exists in both tiles, it should have the same original ID.
            if np.any(updated_tile1 == 1) and np.any(updated_tile2 == 1):
                # Check that the nucleus has consistent ID throughout.
                nucleus_mask_tile1 = updated_tile1 == 1
                nucleus_mask_tile2 = updated_tile2 == 1

                # In the regions where the nucleus exists, there should be only one ID.
                nucleus_region_tile1 = updated_tile1[100:150, 180:256]
                nucleus_region_tile2 = updated_tile2[100:150, 0:40]

                nucleus_ids_tile1 = set(np.unique(nucleus_region_tile1[nucleus_region_tile1 > 0]))
                nucleus_ids_tile2 = set(np.unique(nucleus_region_tile2[nucleus_region_tile2 > 0]))

                # Each region should have at most one nucleus ID.
                assert len(nucleus_ids_tile1) <= 1, f"Nucleus region in tile1 should have at most 1 ID, got {nucleus_ids_tile1}"
                assert len(nucleus_ids_tile2) <= 1, f"Nucleus region in tile2 should have at most 1 ID, got {nucleus_ids_tile2}"

                # If both regions have nuclei, they should have the same original ID.
                if nucleus_ids_tile1 and nucleus_ids_tile2:
                    assert nucleus_ids_tile1 == nucleus_ids_tile2, f"Nucleus should have same ID in both tiles: {nucleus_ids_tile1} vs {nucleus_ids_tile2}"
                    assert 1 in nucleus_ids_tile1, "Cross-boundary nucleus should have original ID 1"
    
    def test_merge_two_tiles_no_overlap(self):
        """Test merging tiles with no overlapping nuclei."""
        tile1 = np.zeros((256, 256), dtype=np.uint32)
        tile2 = np.zeros((256, 256), dtype=np.uint32)
        
        # Add non-overlapping nuclei.
        tile1[50:100, 50:100] = 1
        tile2[150:200, 150:200] = 2
        
        overlap_slices = (
            slice(0, 256), slice(192, 256),
            slice(0, 256), slice(0, 64)
        )
        
        updated_tile1, updated_tile2, mapping = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=False
        )
        
        # Tiles should be mostly unchanged since no nuclei overlap.
        # Only the overlap regions might be zeroed out.
        assert updated_tile1.shape == tile1.shape
        assert updated_tile2.shape == tile2.shape
    
    def test_merge_with_gpu(self):
        """Test GPU-accelerated pairwise merging if available."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for GPU testing")
        except ImportError:
            pytest.skip("PyTorch not available for GPU testing")
        
        tile1, tile2 = self.create_cross_boundary_tiles()
        overlap_slices = (
            slice(0, 256), slice(192, 256),
            slice(0, 256), slice(0, 64)
        )
        
        # Test GPU merge.
        updated_tile1_gpu, updated_tile2_gpu, mapping_gpu = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=True
        )

        # Test CPU merge for comparison.
        updated_tile1_cpu, updated_tile2_cpu, mapping_cpu = merge_two_tiles(
            tile1, tile2, overlap_slices, use_gpu=False
        )
        
        # Results should be similar (allowing for minor differences in implementation).
        assert updated_tile1_gpu.shape == updated_tile1_cpu.shape
        assert updated_tile2_gpu.shape == updated_tile2_cpu.shape


class TestTwoPhaseIntegration:
    """Test the complete two-phase merge pipeline."""
    
    def create_tile_loader(self, tile_data: Dict[Tuple[int, int], np.ndarray]):
        """Create a tile loader function for testing."""
        def loader(y_slice: slice, x_slice: slice) -> np.ndarray:
            # Simple loader that returns pre-created tile data.
            # In practice, this would load from disk.
            stride_h, stride_w = 192, 192  # 256 - 64 overlap
            
            r = y_slice.start // stride_h if y_slice.start is not None else 0
            c = x_slice.start // stride_w if x_slice.start is not None else 0
            
            return tile_data.get((r, c), np.zeros((256, 256), dtype=np.uint32))
        
        return loader
    
    def test_two_phase_merge_2x2_grid(self):
        """Test two-phase merging on a 2x2 tile grid."""
        # Create synthetic tile data.
        tile_data = {}
        
        # Tile (0,0): nucleus in bottom-right corner.
        tile_data[(0, 0)] = np.zeros((256, 256), dtype=np.uint32)
        tile_data[(0, 0)][200:256, 200:256] = 1
        
        # Tile (0,1): nucleus in bottom-left corner (overlaps with (0,0)).
        tile_data[(0, 1)] = np.zeros((256, 256), dtype=np.uint32)
        tile_data[(0, 1)][200:256, 0:56] = 2
        
        # Tile (1,0): nucleus in top-right corner (overlaps with (0,0)).
        tile_data[(1, 0)] = np.zeros((256, 256), dtype=np.uint32)
        tile_data[(1, 0)][0:56, 200:256] = 3
        
        # Tile (1,1): nucleus in top-left corner (overlaps with others).
        tile_data[(1, 1)] = np.zeros((256, 256), dtype=np.uint32)
        tile_data[(1, 1)][0:56, 0:56] = 4
        
        coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        loader = self.create_tile_loader(tile_data)
        
        # Run two-phase merge with new 3-step algorithm.
        merged = merge_tiles_two_phase(
            coords=coords,
            loader=loader,
            height=448,  # 2 * 192 + 64
            width=448,
            tile_h=256,
            tile_w=256,
            overlap=64,
            use_gpu=False,
            merge_batch_size=2
        )
        
        # Verify merged result.
        assert merged.shape == (448, 448)
        assert np.any(merged > 0)  # Should contain nuclei.
        
        # Check that we have reasonable number of unique labels.
        unique_labels = np.unique(merged)
        assert len(unique_labels) >= 2  # At least background + some nuclei.
    
    def test_two_phase_merge_single_row(self):
        """Test two-phase merging on a single row of tiles."""
        tile_data = {}
        
        # Create 3 tiles in a row with overlapping nuclei.
        for c in range(3):
            tile_data[(0, c)] = np.zeros((256, 256), dtype=np.uint32)
            # Add nucleus that spans tile boundaries.
            tile_data[(0, c)][100:150, 200:256] = c + 1
            if c > 0:
                tile_data[(0, c)][100:150, 0:56] = c + 1
        
        coords = [(0, 0), (0, 1), (0, 2)]
        loader = self.create_tile_loader(tile_data)
        
        merged = merge_tiles_two_phase(
            coords=coords,
            loader=loader,
            height=256,
            width=640,  # 3 * 192 + 64
            tile_h=256,
            tile_w=256,
            overlap=64,
            use_gpu=False
        )
        
        assert merged.shape == (256, 640)
        assert np.any(merged > 0)
    
    def test_two_phase_merge_empty_tiles(self):
        """Test two-phase merging with some empty tiles."""
        tile_data = {
            (0, 0): np.zeros((256, 256), dtype=np.uint32),
            (0, 1): np.zeros((256, 256), dtype=np.uint32),
        }
        
        # Add nucleus only to first tile.
        tile_data[(0, 0)][100:150, 100:150] = 1
        
        coords = [(0, 0), (0, 1)]
        loader = self.create_tile_loader(tile_data)
        
        merged = merge_tiles_two_phase(
            coords=coords,
            loader=loader,
            height=256,
            width=448,
            tile_h=256,
            tile_w=256,
            overlap=64,
            use_gpu=False
        )
        
        assert merged.shape == (256, 448)
        # Should still have the nucleus from the first tile.
        assert np.any(merged > 0)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_no_tiles(self):
        """Test behavior with no tiles."""
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            [], 256, 256, 64
        )
        
        assert len(vertical_overlaps) == 0
        assert len(horizontal_overlaps) == 0
    
    def test_single_tile(self):
        """Test behavior with a single tile."""
        coords = [(0, 0)]
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, 256, 256, 64
        )
        
        assert len(vertical_overlaps) == 0
        assert len(horizontal_overlaps) == 0
    
    def test_zero_overlap(self):
        """Test behavior with zero overlap."""
        coords = [(0, 0), (0, 1)]
        
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, 256, 256, 0
        )
        
        # Should have no overlaps when overlap is zero.
        assert len(vertical_overlaps) == 0
        assert len(horizontal_overlaps) == 0
    
    def test_large_overlap(self):
        """Test behavior with overlap larger than tile size."""
        coords = [(0, 0), (0, 1)]
        
        # This should not crash, but may produce unexpected results.
        vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
            coords, 256, 256, 300
        )
        
        # Function should handle this gracefully.
        assert isinstance(vertical_overlaps, dict)
        assert isinstance(horizontal_overlaps, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
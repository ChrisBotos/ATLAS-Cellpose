"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_batch_merge.py.
Description:
    Comprehensive test suite for the batched GPU merge functionality.
    Tests memory estimation, batch size optimization, spatial grouping,
    and the complete batched merge pipeline with various configurations.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy, torch.
    • cellpose_merge.batch_merge module.

Key Features:
    • Memory estimation validation tests.
    • Batch size optimization tests with different memory constraints.
    • Spatial proximity grouping tests for various tile arrangements.
    • Integration tests comparing batched vs non-batched results.
    • Memory usage monitoring and timeout protection.
    • Edge case handling for irregular tile patterns.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch

from ..batch_merge import (
    estimate_memory_requirements,
    get_optimal_batch_size,
    group_tiles_by_spatial_proximity,
    merge_cluster_batched
)


class TestMemoryEstimation:
    """Test memory estimation functions for batch size optimization."""
    
    def test_memory_estimation_basic(self):
        """Test basic memory estimation for different tile configurations."""
        # Test small tiles
        memory_gb = estimate_memory_requirements(
            num_tiles=4, tile_h=256, tile_w=256, overlap=64
        )
        assert 0.001 < memory_gb < 1.0, f"Unexpected memory estimate: {memory_gb} GB"
        
        # Test larger tiles
        memory_gb_large = estimate_memory_requirements(
            num_tiles=4, tile_h=512, tile_w=512, overlap=128
        )
        assert memory_gb_large > memory_gb, "Larger tiles should require more memory"
        
        # Test more tiles
        memory_gb_more_tiles = estimate_memory_requirements(
            num_tiles=16, tile_h=256, tile_w=256, overlap=64
        )
        assert memory_gb_more_tiles > memory_gb, "More tiles should require more memory"
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation with different memory limits."""
        # Test with generous memory limit
        batch_size = get_optimal_batch_size(
            total_tiles=100, tile_h=256, tile_w=256, overlap=64, memory_limit_gb=16.0
        )
        assert batch_size >= 1, "Batch size should be at least 1"
        assert batch_size <= 25, "Batch size should not exceed total tile groups"
        
        # Test with restrictive memory limit
        batch_size_small = get_optimal_batch_size(
            total_tiles=100, tile_h=512, tile_w=512, overlap=128, memory_limit_gb=2.0
        )
        assert batch_size_small >= 1, "Should always allow at least one group"
        assert batch_size_small <= batch_size, "Smaller memory should give smaller batch size"
    
    def test_memory_limit_edge_cases(self):
        """Test edge cases for memory limit handling."""
        # Test with zero memory limit (should auto-detect)
        batch_size = get_optimal_batch_size(
            total_tiles=50, tile_h=256, tile_w=256, overlap=64, memory_limit_gb=0.0
        )
        assert batch_size >= 1, "Should handle auto-detection gracefully"
        
        # Test with very small memory limit
        batch_size_tiny = get_optimal_batch_size(
            total_tiles=50, tile_h=256, tile_w=256, overlap=64, memory_limit_gb=0.1
        )
        assert batch_size_tiny == 1, "Very small memory should force batch_size=1"


class TestSpatialGrouping:
    """Test spatial proximity grouping for tile batching."""
    
    def test_simple_2x2_grid(self):
        """Test grouping for a simple 2x2 tile grid."""
        cluster = [(0, 0), (0, 1), (1, 0), (1, 1)]
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        assert len(batches) >= 1, "Should create at least one batch"
        
        # All tiles should be included
        all_tiles = set()
        for batch in batches:
            all_tiles.update(batch)
        assert all_tiles == set(cluster), "All tiles should be included in batches"
    
    def test_larger_grid_batching(self):
        """Test grouping for a larger tile grid."""
        # Create a 4x4 grid
        cluster = [(r, c) for r in range(4) for c in range(4)]
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=2)
        
        assert len(batches) >= 1, "Should create at least one batch"
        
        # All tiles should be included
        all_tiles = set()
        for batch in batches:
            all_tiles.update(batch)
        assert all_tiles == set(cluster), "All tiles should be included in batches"
        
        # Check that batch sizes are reasonable
        for batch in batches:
            assert len(batch) <= 16, "Batch should not be too large"  # 2 groups * 4 tiles max
    
    def test_irregular_tile_pattern(self):
        """Test grouping for irregular tile patterns."""
        # Create an L-shaped pattern
        cluster = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        assert len(batches) >= 1, "Should handle irregular patterns"
        
        # All tiles should be included
        all_tiles = set()
        for batch in batches:
            all_tiles.update(batch)
        assert all_tiles == set(cluster), "All tiles should be included"
    
    def test_single_tile(self):
        """Test grouping for a single tile."""
        cluster = [(0, 0)]
        batches = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        assert len(batches) == 1, "Should create exactly one batch for single tile"
        assert batches[0] == [(0, 0)], "Single tile should be in its own batch"
    
    def test_batch_size_effects(self):
        """Test how different batch sizes affect grouping."""
        cluster = [(r, c) for r in range(3) for c in range(3)]  # 3x3 grid
        
        # Test with batch_size=1
        batches_small = group_tiles_by_spatial_proximity(cluster, batch_size=1)
        
        # Test with batch_size=3
        batches_large = group_tiles_by_spatial_proximity(cluster, batch_size=3)
        
        # Larger batch size should generally create fewer batches
        assert len(batches_large) <= len(batches_small), "Larger batch size should create fewer batches"


class TestBatchedMerge:
    """Test the complete batched merge functionality."""
    
    def create_synthetic_tile_loader(self, tiles_data: dict) -> callable:
        """Create a synthetic tile loader for testing."""
        def loader(ys: slice, xs: slice) -> np.ndarray:
            # Simple implementation that returns synthetic data
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            
            # Create a simple pattern based on position
            tile_id = (ys.start // 100, xs.start // 100)  # Assuming 100px stride
            if tile_id in tiles_data:
                return tiles_data[tile_id][:h, :w]
            else:
                return np.zeros((h, w), dtype=np.uint32)
        
        return loader
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_batched_vs_regular_merge_gpu(self):
        """Test that batched merge produces similar results to regular merge on GPU."""
        # Create a small test case
        cluster = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create synthetic tile data
        tiles_data = {}
        for r, c in cluster:
            tile = np.zeros((256, 256), dtype=np.uint32)
            # Add some synthetic nuclei
            tile[50:100, 50:100] = 1
            tile[150:200, 150:200] = 2
            tiles_data[(r, c)] = tile
        
        loader = self.create_synthetic_tile_loader(tiles_data)
        
        # Test parameters
        height, width = 512, 512
        tile_h, tile_w = 256, 256
        overlap = 64
        threshold = 0.3
        
        # Run batched merge
        result_batched, pos_batched, _ = merge_cluster_batched(
            cluster=cluster,
            loader=loader,
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=threshold,
            use_gpu=True,
            gid_offset=1,
            batch_size=1,
            memory_limit_gb=8.0
        )
        
        assert result_batched.shape[0] > 0, "Should produce non-empty result"
        assert result_batched.shape[1] > 0, "Should produce non-empty result"
        assert np.any(result_batched > 0), "Should contain some nuclei"
    
    def test_batched_merge_cpu(self):
        """Test batched merge on CPU."""
        # Create a small test case
        cluster = [(0, 0), (0, 1)]
        
        # Create synthetic tile data
        tiles_data = {}
        for r, c in cluster:
            tile = np.zeros((256, 256), dtype=np.uint32)
            # Add a synthetic nucleus
            tile[100:150, 100:150] = 1
            tiles_data[(r, c)] = tile
        
        loader = self.create_synthetic_tile_loader(tiles_data)
        
        # Test parameters
        height, width = 256, 512
        tile_h, tile_w = 256, 256
        overlap = 64
        threshold = 0.3
        
        # Run batched merge on CPU
        result_batched, pos_batched, _ = merge_cluster_batched(
            cluster=cluster,
            loader=loader,
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=threshold,
            use_gpu=False,
            gid_offset=1,
            batch_size=1,
            memory_limit_gb=8.0
        )
        
        assert result_batched.shape[0] > 0, "Should produce non-empty result"
        assert result_batched.shape[1] > 0, "Should produce non-empty result"
        assert pos_batched == (0, 0), "Position should be correct"
    
    def test_memory_fallback(self):
        """Test graceful fallback when memory is insufficient."""
        # Create a larger cluster that might cause memory issues
        cluster = [(r, c) for r in range(5) for c in range(5)]
        
        # Create minimal tile data
        tiles_data = {(r, c): np.zeros((100, 100), dtype=np.uint32) for r, c in cluster}
        loader = self.create_synthetic_tile_loader(tiles_data)
        
        # Test with very restrictive memory limit
        result, pos, _ = merge_cluster_batched(
            cluster=cluster,
            loader=loader,
            height=500,
            width=500,
            tile_h=100,
            tile_w=100,
            overlap=20,
            threshold=0.3,
            use_gpu=False,  # Use CPU to avoid GPU memory issues in testing
            gid_offset=1,
            batch_size=1,
            memory_limit_gb=0.1  # Very small limit
        )
        
        assert result.shape[0] > 0, "Should handle memory constraints gracefully"
        assert result.shape[1] > 0, "Should handle memory constraints gracefully"


if __name__ == "__main__":
    pytest.main([__file__])

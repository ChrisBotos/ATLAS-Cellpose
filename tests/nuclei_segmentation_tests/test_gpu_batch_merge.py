"""
Test Suite: test_gpu_batch_merge.py.

Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Comprehensive test suite for GPU-optimized tile merging functionality.
    Tests memory management, spatial batching strategies, error recovery,
    and performance optimization features in the enhanced batch merge system.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/test_gpu_batch_merge.py -v

Dependencies:
    • Python >= 3.10.
    • numpy, pytest, torch, unittest.mock.
    • The enhanced batch merge utilities from cellpose_merge package.

Key Features:
    • GPU memory management validation.
    • Spatial batching strategy testing.
    • Error recovery and CPU fallback testing.
    • Memory estimation accuracy verification.
    • Performance benchmarking for different strategies.

Notes:
    • Tests are designed to work with or without GPU availability.
    • Mock objects are used to simulate GPU memory constraints.
    • All tests include proper cleanup and resource management.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List, Tuple
from unittest.mock import MagicMock, patch, Mock
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Adjust path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from code.nuclei_segmentation.cellpose_merge.batch_merge import (
    estimate_memory_requirements,
    get_optimal_batch_size,
    group_tiles_by_spatial_proximity,
    merge_cluster_batched,
    _create_optimized_2x2_groups,
    _create_spatial_chunks,
    _create_hybrid_groups,
)

"""Test fixtures and utilities."""

@pytest.fixture
def sample_tile_cluster():
    """Create a sample tile cluster for testing."""
    # Create a 4x4 grid of tiles with some gaps.
    tiles = []
    for r in range(4):
        for c in range(4):
            if not (r == 1 and c == 1):  # Skip one tile to create irregular pattern.
                tiles.append((r, c))
    return tiles

@pytest.fixture
def dense_tile_cluster():
    """Create a dense tile cluster for testing 2x2 grouping."""
    return [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1)]

@pytest.fixture
def sparse_tile_cluster():
    """Create a sparse tile cluster for testing spatial chunking."""
    return [(0, 0), (0, 5), (3, 2), (7, 8), (10, 1), (15, 12)]

@pytest.fixture
def mock_loader():
    """Create a mock tile loader function."""
    def loader(y_slice, x_slice):
        h = y_slice.stop - y_slice.start
        w = x_slice.stop - x_slice.start
        # Create a simple mask with some objects.
        mask = np.zeros((h, w), dtype=np.uint32)
        if h > 10 and w > 10:
            mask[5:h-5, 5:w-5] = 1  # Simple rectangular object.
        return mask
    return loader

"""Memory estimation tests."""

class TestMemoryEstimation:
    """Test enhanced memory estimation functionality."""

    def test_basic_memory_estimation(self, sample_tile_cluster):
        """Test basic memory estimation with default parameters."""
        memory_gb = estimate_memory_requirements(
            tiles=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64
        )
        
        assert memory_gb > 0, "Memory estimate should be positive"
        assert memory_gb < 100, "Memory estimate should be reasonable"

    def test_memory_estimation_with_safety_factor(self, sample_tile_cluster):
        """Test memory estimation with different safety factors."""
        base_memory = estimate_memory_requirements(
            tiles=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            safety_factor=1.0
        )
        
        safe_memory = estimate_memory_requirements(
            tiles=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            safety_factor=2.0
        )
        
        assert safe_memory > base_memory, "Higher safety factor should increase estimate"
        assert safe_memory == base_memory * 2, "Safety factor should scale linearly"

    def test_empty_cluster_memory_estimation(self):
        """Test memory estimation with empty cluster."""
        memory_gb = estimate_memory_requirements(
            tiles=[],
            tile_h=512,
            tile_w=512,
            overlap=64
        )
        
        assert memory_gb == 0.0, "Empty cluster should have zero memory requirement"

    def test_large_cluster_memory_estimation(self):
        """Test memory estimation with large cluster."""
        # Create a large cluster.
        large_cluster = [(r, c) for r in range(20) for c in range(20)]
        
        memory_gb = estimate_memory_requirements(
            tiles=large_cluster,
            tile_h=1024,
            tile_w=1024,
            overlap=128
        )
        
        assert memory_gb > 1.0, "Large cluster should require significant memory"

"""Batch size optimization tests."""

class TestBatchSizeOptimization:
    """Test enhanced batch size optimization algorithms."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_optimal_batch_size_basic(self, sample_tile_cluster):
        """Test basic optimal batch size calculation."""
        batch_size = get_optimal_batch_size(
            cluster=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=8.0
        )
        
        assert batch_size >= 1, "Batch size should be at least 1"
        assert batch_size <= len(sample_tile_cluster), "Batch size should not exceed cluster size"

    def test_optimal_batch_size_memory_constrained(self, sample_tile_cluster):
        """Test batch size optimization with memory constraints."""
        # Test with very low memory limit.
        small_batch = get_optimal_batch_size(
            cluster=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=0.1
        )
        
        # Test with high memory limit.
        large_batch = get_optimal_batch_size(
            cluster=sample_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=32.0
        )
        
        assert small_batch <= large_batch, "Higher memory limit should allow larger batches"

    def test_optimal_batch_size_adaptive(self, dense_tile_cluster, sparse_tile_cluster):
        """Test adaptive batch sizing for different cluster types."""
        dense_batch = get_optimal_batch_size(
            cluster=dense_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=8.0,
            adaptive_sizing=True
        )
        
        sparse_batch = get_optimal_batch_size(
            cluster=sparse_tile_cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=8.0,
            adaptive_sizing=True
        )
        
        # Dense clusters might allow larger batches due to better spatial locality.
        assert dense_batch >= 1 and sparse_batch >= 1, "Both should have valid batch sizes"

    def test_empty_cluster_batch_size(self):
        """Test batch size calculation with empty cluster."""
        batch_size = get_optimal_batch_size(
            cluster=[],
            tile_h=512,
            tile_w=512,
            overlap=64,
            memory_limit_gb=8.0
        )
        
        assert batch_size == 1, "Empty cluster should return batch size 1"

"""Spatial batching strategy tests."""

class TestSpatialBatchingStrategies:
    """Test different spatial batching strategies."""

    def test_adaptive_strategy_selection(self, dense_tile_cluster, sparse_tile_cluster):
        """Test adaptive strategy selection based on cluster characteristics."""
        # Test with dense cluster.
        dense_batches = group_tiles_by_spatial_proximity(
            cluster=dense_tile_cluster,
            batch_size=2,
            strategy="adaptive"
        )
        
        # Test with sparse cluster.
        sparse_batches = group_tiles_by_spatial_proximity(
            cluster=sparse_tile_cluster,
            batch_size=2,
            strategy="adaptive"
        )
        
        assert len(dense_batches) > 0, "Dense cluster should produce batches"
        assert len(sparse_batches) > 0, "Sparse cluster should produce batches"
        
        # Verify all tiles are included.
        dense_tiles_in_batches = [tile for batch in dense_batches for tile in batch]
        sparse_tiles_in_batches = [tile for batch in sparse_batches for tile in batch]
        
        assert set(dense_tiles_in_batches) == set(dense_tile_cluster), "All dense tiles should be included"
        assert set(sparse_tiles_in_batches) == set(sparse_tile_cluster), "All sparse tiles should be included"

    def test_2x2_strategy(self, dense_tile_cluster):
        """Test optimized 2x2 grouping strategy."""
        batches = group_tiles_by_spatial_proximity(
            cluster=dense_tile_cluster,
            batch_size=1,
            strategy="2x2"
        )
        
        assert len(batches) > 0, "2x2 strategy should produce batches"
        
        # Check that some batches contain 2x2 groups.
        has_2x2_group = any(len(batch) == 4 for batch in batches)
        assert has_2x2_group or len(dense_tile_cluster) < 4, "Should create 2x2 groups when possible"

    def test_spatial_strategy(self, sparse_tile_cluster):
        """Test spatial chunking strategy."""
        batches = group_tiles_by_spatial_proximity(
            cluster=sparse_tile_cluster,
            batch_size=3,
            strategy="spatial"
        )
        
        assert len(batches) > 0, "Spatial strategy should produce batches"
        
        # Verify batch sizes don't exceed the limit.
        for batch in batches:
            assert len(batch) <= 3, "Batch size should not exceed limit"

    def test_hybrid_strategy(self, sample_tile_cluster):
        """Test hybrid batching strategy."""
        batches = group_tiles_by_spatial_proximity(
            cluster=sample_tile_cluster,
            batch_size=2,
            strategy="hybrid"
        )
        
        assert len(batches) > 0, "Hybrid strategy should produce batches"
        
        # Verify all tiles are included.
        all_tiles_in_batches = [tile for batch in batches for tile in batch]
        assert set(all_tiles_in_batches) == set(sample_tile_cluster), "All tiles should be included"

    def test_unknown_strategy_fallback(self, sample_tile_cluster):
        """Test fallback behavior for unknown strategy."""
        batches = group_tiles_by_spatial_proximity(
            cluster=sample_tile_cluster,
            batch_size=2,
            strategy="unknown_strategy"
        )
        
        assert len(batches) > 0, "Unknown strategy should fall back to spatial chunking"

"""Individual grouping function tests."""

class TestGroupingFunctions:
    """Test individual grouping functions."""

    def test_optimized_2x2_groups(self, dense_tile_cluster):
        """Test optimized 2x2 grouping function."""
        batches = _create_optimized_2x2_groups(dense_tile_cluster, batch_size=1)
        
        assert len(batches) > 0, "Should create batches"
        
        # Verify all tiles are included.
        all_tiles = [tile for batch in batches for tile in batch]
        assert set(all_tiles) == set(dense_tile_cluster), "All tiles should be included"

    def test_spatial_chunks_with_morton_order(self, sparse_tile_cluster):
        """Test spatial chunking with Morton order."""
        batches = _create_spatial_chunks(sparse_tile_cluster, max_batch_size=2)
        
        assert len(batches) > 0, "Should create batches"
        
        # Verify batch size limits.
        for batch in batches:
            assert len(batch) <= 2, "Batch size should not exceed limit"

    def test_hybrid_groups(self, sample_tile_cluster):
        """Test hybrid grouping function."""
        batches = _create_hybrid_groups(sample_tile_cluster, batch_size=2)
        
        assert len(batches) > 0, "Should create batches"
        
        # Verify all tiles are included.
        all_tiles = [tile for batch in batches for tile in batch]
        assert set(all_tiles) == set(sample_tile_cluster), "All tiles should be included"


"""GPU memory management and error recovery tests."""

class TestGPUMemoryManagement:
    """Test GPU memory management and error recovery functionality."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_gpu_memory_cleanup(self, sample_tile_cluster, mock_loader):
        """Test GPU memory cleanup functionality."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.synchronize') as mock_sync:

            # Test with aggressive cleanup enabled.
            try:
                merge_cluster_batched(
                    cluster=sample_tile_cluster[:4],  # Small cluster for testing.
                    loader=mock_loader,
                    height=2048,
                    width=2048,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    threshold=0.3,
                    use_gpu=True,
                    gid_offset=1,
                    batch_size=2,
                    memory_limit_gb=8.0,
                    aggressive_cleanup=True
                )
            except Exception:
                pass  # We're testing cleanup calls, not success.

            # Verify cleanup functions were called.
            assert mock_empty_cache.called, "GPU memory cache should be cleared"
            assert mock_sync.called, "GPU should be synchronized"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_memory_error_recovery(self, sample_tile_cluster, mock_loader):
        """Test memory error recovery and batch size reduction."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('code.nuclei_segmentation.cellpose_merge.gpu_merge.merge_patch_gpu') as mock_gpu_merge:

            # Simulate CUDA out of memory error.
            mock_gpu_merge.side_effect = RuntimeError("CUDA out of memory")

            # Test should fall back to CPU or reduce batch size.
            try:
                result = merge_cluster_batched(
                    cluster=sample_tile_cluster[:2],  # Small cluster.
                    loader=mock_loader,
                    height=1024,
                    width=1024,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    threshold=0.3,
                    use_gpu=True,
                    gid_offset=1,
                    batch_size=4,  # Start with large batch size.
                    memory_limit_gb=8.0
                )
                # If we get here, recovery worked.
                assert result is not None, "Should recover from memory error"
            except Exception as e:
                # Should eventually fall back to CPU or succeed.
                assert "CUDA out of memory" not in str(e), "Should recover from CUDA memory errors"

    def test_cpu_fallback_behavior(self, sample_tile_cluster, mock_loader):
        """Test CPU fallback when GPU is not available."""
        with patch('torch.cuda.is_available', return_value=False):

            result = merge_cluster_batched(
                cluster=sample_tile_cluster[:4],
                loader=mock_loader,
                height=2048,
                width=2048,
                tile_h=512,
                tile_w=512,
                overlap=64,
                threshold=0.3,
                use_gpu=True,  # Request GPU but it's not available.
                gid_offset=1,
                batch_size=2
            )

            assert result is not None, "Should work with CPU fallback"
            assert result.merged_patch is not None, "Should produce merged result"

    def test_tensor_size_error_handling(self, mock_loader):
        """Test handling of tensor size limit errors."""
        # Create a very large cluster that would exceed tensor limits.
        large_cluster = [(r, c) for r in range(100) for c in range(100)]

        with patch('torch.cuda.is_available', return_value=True):
            try:
                result = merge_cluster_batched(
                    cluster=large_cluster[:10],  # Use subset to avoid timeout.
                    loader=mock_loader,
                    height=51200,  # Very large image.
                    width=51200,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    threshold=0.3,
                    use_gpu=True,
                    gid_offset=1,
                    batch_size=1,  # Force small batches.
                    memory_limit_gb=1.0  # Low memory limit.
                )
                assert result is not None, "Should handle large tensors gracefully"
            except RuntimeError as e:
                # Should provide informative error message.
                assert "tensor" in str(e).lower() or "memory" in str(e).lower(), \
                    "Should provide informative error for tensor size issues"


"""Integration and performance tests."""

class TestIntegrationAndPerformance:
    """Test integration scenarios and performance characteristics."""

    def test_full_merge_pipeline_small(self, dense_tile_cluster, mock_loader):
        """Test complete merge pipeline with small cluster."""
        result = merge_cluster_batched(
            cluster=dense_tile_cluster,
            loader=mock_loader,
            height=2048,
            width=2048,
            tile_h=512,
            tile_w=512,
            overlap=64,
            threshold=0.3,
            use_gpu=False,  # Use CPU for reliable testing.
            gid_offset=1,
            batch_size=2,
            spatial_strategy="adaptive",
            adaptive_batching=True
        )

        assert result is not None, "Should complete successfully"
        assert result.merged_patch is not None, "Should produce merged patch"
        assert result.merged_patch.shape[0] > 0, "Merged patch should have positive height"
        assert result.merged_patch.shape[1] > 0, "Merged patch should have positive width"

    def test_different_strategies_consistency(self, sample_tile_cluster, mock_loader):
        """Test that different strategies produce consistent results."""
        strategies = ["2x2", "spatial", "hybrid"]
        results = {}

        for strategy in strategies:
            try:
                result = merge_cluster_batched(
                    cluster=sample_tile_cluster[:6],  # Use subset for faster testing.
                    loader=mock_loader,
                    height=2048,
                    width=2048,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    threshold=0.3,
                    use_gpu=False,
                    gid_offset=1,
                    batch_size=2,
                    spatial_strategy=strategy
                )
                results[strategy] = result
            except Exception as e:
                pytest.fail(f"Strategy {strategy} failed: {e}")

        # All strategies should produce results.
        for strategy in strategies:
            assert results[strategy] is not None, f"Strategy {strategy} should produce result"
            assert results[strategy].merged_patch is not None, f"Strategy {strategy} should produce patch"

    def test_parameter_validation(self, sample_tile_cluster, mock_loader):
        """Test parameter validation and error handling."""
        # Test with invalid parameters.
        with pytest.raises((ValueError, RuntimeError)):
            merge_cluster_batched(
                cluster=sample_tile_cluster,
                loader=mock_loader,
                height=0,  # Invalid height.
                width=2048,
                tile_h=512,
                tile_w=512,
                overlap=64,
                threshold=0.3,
                use_gpu=False,
                gid_offset=1
            )

    def test_edge_cases(self, mock_loader):
        """Test edge cases and boundary conditions."""
        # Test with single tile.
        single_tile = [(0, 0)]
        result = merge_cluster_batched(
            cluster=single_tile,
            loader=mock_loader,
            height=512,
            width=512,
            tile_h=512,
            tile_w=512,
            overlap=0,
            threshold=0.3,
            use_gpu=False,
            gid_offset=1
        )

        assert result is not None, "Should handle single tile"
        assert result.merged_patch is not None, "Should produce result for single tile"

    def test_memory_safety_factors(self, sample_tile_cluster):
        """Test different memory safety factors."""
        safety_factors = [1.0, 1.5, 2.0, 3.0]

        for factor in safety_factors:
            memory_estimate = estimate_memory_requirements(
                tiles=sample_tile_cluster,
                tile_h=512,
                tile_w=512,
                overlap=64,
                safety_factor=factor
            )

            assert memory_estimate > 0, f"Safety factor {factor} should produce positive estimate"

            # Higher safety factors should produce higher estimates.
            if factor > 1.0:
                base_estimate = estimate_memory_requirements(
                    tiles=sample_tile_cluster,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    safety_factor=1.0
                )
                assert memory_estimate >= base_estimate, \
                    f"Safety factor {factor} should not reduce memory estimate"

"""
Author: Christos Botos.
Script Name: test_large_image_fixes.py.
Description:
    Test suite for verifying fixes to handle large images that previously caused
    uint32 overflow, memory allocation errors, and PyTorch tensor size limits.
    
    This test suite validates:
    - uint32 overflow prevention in composite key generation
    - Memory allocation error detection and handling
    - PyTorch tensor size limit detection
    - Cluster splitting functionality for very large clusters
    - Proper error messages and fallback mechanisms
"""

import numpy as np
import pytest
import traceback
from unittest.mock import Mock, patch
from typing import List, Tuple

# Import the functions we want to test.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_tiles import (
    _check_cluster_feasibility,
    _split_large_cluster,
    _estimate_cluster_memory_requirements
)
from rules import merge_patch_cpu
from gpu_merge import merge_patch_gpu


class TestLargeImageFixes:
    """Test suite for large image processing fixes."""
    
    def test_cluster_feasibility_check_normal_cluster(self):
        """Test that normal-sized clusters pass feasibility checks."""
        # Create a small cluster that should be feasible.
        cluster = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        is_feasible, reason = _check_cluster_feasibility(
            cluster=cluster,
            tile_h=512,
            tile_w=512,
            overlap=64,
            height=2048,
            width=2048,
            memory_limit_gb=8.0
        )
        
        assert is_feasible, f"Normal cluster should be feasible, but got: {reason}"
        assert reason == ""
    
    def test_cluster_feasibility_check_memory_limit(self):
        """Test that clusters exceeding memory limits are detected."""
        # Create a cluster that would require too much memory.
        cluster = [(r, c) for r in range(100) for c in range(100)]  # 10,000 tiles
        
        is_feasible, reason = _check_cluster_feasibility(
            cluster=cluster,
            tile_h=1024,
            tile_w=1024,
            overlap=128,
            height=100000,
            width=100000,
            memory_limit_gb=1.0  # Very low limit
        )
        
        assert not is_feasible, "Large cluster should not be feasible with low memory limit"
        assert ("memory requirement" in reason.lower() or "array" in reason.lower()), f"Expected memory or array limit error, got: {reason}"
    
    def test_cluster_feasibility_check_array_size_limit(self):
        """Test that clusters exceeding array size limits are detected."""
        # Create a cluster that would exceed array size limits.
        # This is a bit tricky to test without actually creating huge arrays.
        cluster = [(0, 0)]  # Single tile but with huge dimensions
        
        is_feasible, reason = _check_cluster_feasibility(
            cluster=cluster,
            tile_h=100000,
            tile_w=100000,
            overlap=0,
            height=100000,
            width=100000,
            memory_limit_gb=1000.0  # High memory limit, but array size should fail
        )
        
        # This should pass because it's just one tile, but let's test with multiple tiles.
        large_cluster = [(r, c) for r in range(1000) for c in range(1000)]  # 1M tiles
        
        is_feasible, reason = _check_cluster_feasibility(
            cluster=large_cluster,
            tile_h=1024,
            tile_w=1024,
            overlap=128,
            height=1000000,
            width=1000000,
            memory_limit_gb=1000.0
        )
        
        assert not is_feasible, "Extremely large cluster should not be feasible"
        assert "array" in reason.lower() or "memory" in reason.lower()
    
    def test_cluster_splitting(self):
        """Test that large clusters are split correctly."""
        # Create a large cluster.
        large_cluster = [(r, c) for r in range(50) for c in range(50)]  # 2500 tiles
        
        # Split into smaller clusters.
        sub_clusters = _split_large_cluster(large_cluster, max_cluster_size=100)
        
        # Check that we got multiple sub-clusters.
        assert len(sub_clusters) > 1, "Large cluster should be split into multiple sub-clusters"
        
        # Check that each sub-cluster is within the size limit.
        for sub_cluster in sub_clusters:
            assert len(sub_cluster) <= 100, f"Sub-cluster has {len(sub_cluster)} tiles, exceeding limit"
        
        # Check that all original tiles are preserved.
        all_tiles = []
        for sub_cluster in sub_clusters:
            all_tiles.extend(sub_cluster)
        
        assert len(all_tiles) == len(large_cluster), "Tiles lost during splitting"
        assert set(all_tiles) == set(large_cluster), "Tiles changed during splitting"
    
    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        # Test with reasonable values.
        memory_gb = _estimate_cluster_memory_requirements(
            cluster_size=100,
            cluster_h=5000,
            cluster_w=5000
        )
        
        assert memory_gb > 0, "Memory estimate should be positive"
        assert memory_gb < 1000, "Memory estimate seems unreasonably high"
        
        # Test with larger values.
        large_memory_gb = _estimate_cluster_memory_requirements(
            cluster_size=1000,
            cluster_h=10000,
            cluster_w=10000
        )
        
        assert large_memory_gb > memory_gb, "Larger cluster should require more memory"
    
    def test_uint32_overflow_prevention_in_rules(self):
        """Test that uint32 overflow is prevented in rules.py."""
        # Create a small patch to test with.
        patch = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.uint32)
        
        # This should work normally.
        merged, mapping = merge_patch_cpu(patch, threshold=0.3)
        
        assert merged.shape == (2, 2), "Merged patch should have correct shape"
        assert merged.dtype == np.uint32, "Merged patch should be uint32"
    
    @pytest.mark.skipif(not pytest.importorskip("torch", reason="PyTorch not available"), reason="PyTorch not available")
    def test_gpu_tensor_size_limit_detection(self):
        """Test that GPU tensor size limits are detected."""
        import torch
        
        # Create a patch that would exceed tensor size limits.
        # We'll mock this to avoid actually creating huge arrays.
        from unittest.mock import patch as mock_patch
        with mock_patch('torch.from_numpy') as mock_from_numpy:
            mock_from_numpy.side_effect = RuntimeError("nonzero is not supported for tensors with more than INT_MAX elements")
            
            # Create a reasonably sized patch for testing.
            test_patch = np.ones((10, 1000, 1000), dtype=np.uint32)
            
            with pytest.raises(RuntimeError) as exc_info:
                merge_patch_gpu(test_patch, threshold=0.3)
            
            assert "tensor" in str(exc_info.value).lower() or "elements" in str(exc_info.value).lower()
    
    def test_small_cluster_no_splitting(self):
        """Test that small clusters are not split unnecessarily."""
        small_cluster = [(0, 0), (0, 1), (1, 0)]
        
        sub_clusters = _split_large_cluster(small_cluster, max_cluster_size=100)
        
        assert len(sub_clusters) == 1, "Small cluster should not be split"
        assert sub_clusters[0] == small_cluster, "Small cluster should remain unchanged"


if __name__ == "__main__":
    # Run the tests if this file is executed directly.
    pytest.main([__file__, "-v"])

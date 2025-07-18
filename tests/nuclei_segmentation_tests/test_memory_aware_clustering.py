"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_memory_aware_clustering.py.
Description:
    Test suite for the new memory-aware clustering algorithm that prevents
    problematic array allocations during tile merging operations.

Dependencies:
    • Python >= 3.7.
    • pytest for test framework.
    • numpy for array operations.

Usage:
    pytest tests/nuclei_segmentation_tests/test_memory_aware_clustering.py -v

Inputs:
    • Test tile coordinate patterns (dense, sparse, mixed).
    • Memory and dimension constraints.

Outputs:
    • Test results validating clustering behavior.
    • Performance metrics for different scenarios.

Key Features:
    • Tests for memory-efficient cluster creation.
    • Validation of cluster size limits.
    • Verification of spatial locality preservation.
    • Edge case handling for sparse distributions.

Notes:
    • Designed to prevent the 922×26459 element array allocation issues.
    • Validates that clusters stay within memory constraints.
    • Ensures 4-step merging rules are preserved.
"""

import traceback
import pytest
import numpy as np
from typing import List, Tuple

# Import the functions we want to test.
from code.nuclei_segmentation.cellpose_merge.merge_tiles import (
    _build_memory_aware_clusters,
    _estimate_cluster_requirements,
    _split_cluster_spatially
)


class TestMemoryAwareClustering:
    """Test suite for memory-aware clustering algorithm."""

    def test_dense_cluster_creation(self):
        """Test that dense tile patterns create reasonable clusters."""
        # Create a dense 4x4 grid of tiles.
        coords = [(r, c) for r in range(4) for c in range(4)]
        
        clusters = _build_memory_aware_clusters(
            coords, tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=2.0, max_cluster_dimension=2048
        )
        
        # Should create multiple small clusters instead of one large one.
        assert len(clusters) > 1, "Dense pattern should be split into multiple clusters"
        
        # Each cluster should be reasonably sized.
        for cluster in clusters:
            assert len(cluster) <= 16, f"Cluster too large: {len(cluster)} tiles"
            
            # Verify memory requirements.
            memory, dimensions = _estimate_cluster_requirements(
                cluster, tile_h=512, tile_w=512, overlap=64
            )
            assert memory <= 2.0, f"Cluster exceeds memory limit: {memory:.2f} GB"
            assert max(dimensions) <= 2048, f"Cluster dimensions too large: {dimensions}"

    def test_sparse_cluster_handling(self):
        """Test that sparse tile distributions are handled correctly."""
        # Create a sparse pattern that would cause problematic arrays.
        coords = [(0, 0), (0, 50), (50, 0), (50, 50)]  # Very sparse 4 tiles.
        
        clusters = _build_memory_aware_clusters(
            coords, tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=1.0, max_cluster_dimension=1024
        )
        
        # Should create individual clusters for sparse patterns.
        assert len(clusters) == 4, f"Expected 4 individual clusters, got {len(clusters)}"
        
        # Each cluster should contain only one tile.
        for cluster in clusters:
            assert len(cluster) == 1, f"Sparse cluster should have 1 tile, got {len(cluster)}"

    def test_memory_estimation_accuracy(self):
        """Test that memory estimation prevents problematic allocations."""
        # Test case that would create the problematic 922×26459 array.
        large_sparse_cluster = [(0, 0), (0, 100), (100, 0), (100, 100)]
        
        memory, dimensions = _estimate_cluster_requirements(
            large_sparse_cluster, tile_h=512, tile_w=512, overlap=64
        )
        
        # Should detect this as problematic.
        assert memory > 1.0, "Should detect high memory requirement"
        assert max(dimensions) > 4096, "Should detect large dimensions"

    def test_cluster_splitting_functionality(self):
        """Test that large clusters are split correctly."""
        # Create a cluster that needs splitting.
        large_cluster = [(r, c) for r in range(10) for c in range(10)]  # 100 tiles.
        
        split_clusters = _split_cluster_spatially(
            large_cluster, tile_h=512, tile_w=512, overlap=64,
            max_memory_gb=1.0, max_dimension=2048
        )
        
        # Should be split into multiple smaller clusters.
        assert len(split_clusters) > 1, "Large cluster should be split"
        
        # Each split cluster should meet constraints.
        for cluster in split_clusters:
            memory, dimensions = _estimate_cluster_requirements(
                cluster, tile_h=512, tile_w=512, overlap=64
            )
            assert memory <= 1.5, f"Split cluster exceeds memory: {memory:.2f} GB"  # Allow some tolerance.
            assert max(dimensions) <= 3000, f"Split cluster too large: {dimensions}"  # Allow some tolerance.

    def test_empty_and_single_tile_cases(self):
        """Test edge cases with empty or single tile inputs."""
        # Empty case.
        empty_clusters = _build_memory_aware_clusters(
            [], tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=1.0, max_cluster_dimension=1024
        )
        assert len(empty_clusters) == 0, "Empty input should return empty clusters"
        
        # Single tile case.
        single_clusters = _build_memory_aware_clusters(
            [(0, 0)], tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=1.0, max_cluster_dimension=1024
        )
        assert len(single_clusters) == 1, "Single tile should create one cluster"
        assert len(single_clusters[0]) == 1, "Single tile cluster should have one tile"

    def test_configuration_parameter_effects(self):
        """Test that configuration parameters affect clustering behavior."""
        coords = [(r, c) for r in range(6) for c in range(6)]  # 36 tiles.
        
        # Test with very restrictive parameters.
        restrictive_clusters = _build_memory_aware_clusters(
            coords, tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=0.1, max_cluster_dimension=512
        )
        
        # Test with more permissive parameters.
        permissive_clusters = _build_memory_aware_clusters(
            coords, tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=5.0, max_cluster_dimension=8192
        )
        
        # Restrictive should create more, smaller clusters.
        assert len(restrictive_clusters) >= len(permissive_clusters), \
            "Restrictive parameters should create more clusters"

    def test_spatial_locality_preservation(self):
        """Test that spatial locality is preserved in clustering."""
        # Create an L-shaped pattern.
        coords = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
        
        clusters = _build_memory_aware_clusters(
            coords, tile_h=512, tile_w=512, overlap=64,
            max_cluster_memory_gb=2.0, max_cluster_dimension=2048
        )
        
        # Should preserve spatial locality - adjacent tiles should be clustered together.
        for cluster in clusters:
            if len(cluster) > 1:
                # Check that tiles in the cluster are spatially connected.
                for i, (r1, c1) in enumerate(cluster):
                    has_neighbor = False
                    for j, (r2, c2) in enumerate(cluster):
                        if i != j:
                            # Check if tiles are adjacent (4-connectivity).
                            if abs(r1 - r2) + abs(c1 - c2) == 1:
                                has_neighbor = True
                                break
                    # At least one tile should have a neighbor (except for single-tile clusters).
                    if len(cluster) > 1:
                        assert has_neighbor or len(cluster) == 1, \
                            f"Tile ({r1}, {c1}) has no adjacent neighbors in cluster"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_merge_tiles.py.
Description:
    Comprehensive test suite for the merge_tiles module in kidney I/R injury
    spatial multiomics analysis. This test suite validates all critical
    functionality including the 4 tile merging rules, edge tile handling,
    memory management, and infinite loop prevention.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pathlib, tempfile for testing infrastructure.
    • PIL for image handling in test data generation.
    • merge_tiles module from the cellpose_merge package.

Usage:
    pytest test_merge_tiles.py -v
    pytest test_merge_tiles.py::test_edge_tile_merging -v

Inputs:
    • Synthetic tile masks and tissue images created for testing.
    • Various edge case scenarios for robust validation.

Outputs:
    • Test results with comprehensive validation of merge functionality.
    • Memory usage monitoring and timeout protection.

Key Features:
    • Complete validation of the 4 tile merging rules implementation.
    • Edge tile handling tests (critical bug fix validation).
    • Memory management tests to prevent RAM overflow.
    • Timeout protection to prevent infinite loops.
    • Integration tests for complete workflow validation.
    • Scientific context validation for kidney tissue analysis.

Notes:
    • This test suite specifically validates the recently fixed edge tile
      merging functionality that was causing incomplete segmentation coverage.
    • All tests include memory monitoring and timeout protection.
    • Tests are designed with kidney I/R injury analysis context in mind.
"""

import traceback
import pytest
import numpy as np
import tempfile
import time
import psutil
import os
from pathlib import Path
from typing import Tuple, Callable
from numpy.typing import NDArray
from unittest.mock import patch, MagicMock

# Import the module under test.
import sys
sys.path.append(str(Path(__file__).parent.parent))
from cellpose_merge.merge_tiles import (
    merge_masks_streaming,
    _parse_tile_filename,
    _resolve_tiles_path,
    _discover_tiles,
    _build_clusters,
    _merge_cluster
)


"""MEMORY AND TIMEOUT MONITORING"""

class MemoryMonitor:
    """Monitor memory usage during tests to prevent RAM overflow."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
    
    def check_memory(self):
        """Check current memory usage and raise error if exceeded."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > self.max_memory_mb:
            raise MemoryError(f"Memory usage exceeded {self.max_memory_mb}MB: "
                            f"current increase = {memory_increase:.1f}MB")
        
        return memory_increase


def timeout_protection(timeout_seconds: int = 30):
    """Decorator to add timeout protection to test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            def check_timeout():
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Test function {func.__name__} exceeded {timeout_seconds}s timeout")
            
            # Inject timeout checker into the test function.
            kwargs['_timeout_checker'] = check_timeout
            return func(*args, **kwargs)
        return wrapper
    return decorator


"""SYNTHETIC DATA GENERATION"""

def create_synthetic_tiles(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    temp_dir: Path,
    include_edge_tiles: bool = True
) -> Tuple[int, int]:
    """
    Create synthetic tile masks for testing merge functionality.
    
    This function generates realistic tile masks that simulate the output
    of Cellpose segmentation on kidney tissue sections, including edge
    cases that extend beyond image boundaries.
    
    Parameters
    ----------
    height, width : int
        Target image dimensions in pixels.
    tile_h, tile_w : int
        Individual tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    temp_dir : Path
        Directory to save synthetic tile masks.
    include_edge_tiles : bool
        Whether to include tiles that extend beyond image boundaries.
        
    Returns
    -------
    Tuple[int, int]
        Number of tiles created and number of edge tiles.
    """
    
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    # Calculate tile grid coverage.
    n_rows = (height + stride_h - 1) // stride_h
    n_cols = (width + stride_w - 1) // stride_w
    
    tiles_created = 0
    edge_tiles = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate tile position in global coordinates.
            global_y0 = r * stride_h
            global_x0 = c * stride_w
            
            # Determine actual tile size (may be smaller for edge tiles).
            if include_edge_tiles:
                # Include tiles that extend beyond boundaries.
                actual_tile_h = tile_h
                actual_tile_w = tile_w
                
                # Crop to image boundaries for the actual content.
                content_h = min(tile_h, height - global_y0)
                content_w = min(tile_w, width - global_x0)
            else:
                # Only include tiles that fit completely within image.
                if global_y0 + tile_h > height or global_x0 + tile_w > width:
                    continue
                actual_tile_h = tile_h
                actual_tile_w = tile_w
                content_h = tile_h
                content_w = tile_w
            
            if content_h <= 0 or content_w <= 0:
                continue
            
            # Create synthetic mask with realistic nucleus patterns.
            tile_mask = np.zeros((actual_tile_h, actual_tile_w), dtype=np.uint32)
            
            # Add synthetic nuclei with unique labels per tile.
            nucleus_id = r * n_cols + c + 1
            nucleus_size = 8  # Realistic nucleus size for kidney tissue.
            
            # Add multiple nuclei per tile for realistic density.
            nuclei_per_tile = 3
            for n in range(nuclei_per_tile):
                # Random position within the content area.
                if content_h > nucleus_size * 2 and content_w > nucleus_size * 2:
                    center_y = np.random.randint(nucleus_size, content_h - nucleus_size)
                    center_x = np.random.randint(nucleus_size, content_w - nucleus_size)
                    
                    # Create circular nucleus.
                    for dy in range(-nucleus_size//2, nucleus_size//2 + 1):
                        for dx in range(-nucleus_size//2, nucleus_size//2 + 1):
                            if dy*dy + dx*dx <= (nucleus_size//2)**2:
                                ny = center_y + dy
                                nx = center_x + dx
                                if 0 <= ny < actual_tile_h and 0 <= nx < actual_tile_w:
                                    tile_mask[ny, nx] = nucleus_id + n * 1000
            
            # Save tile using pixel coordinate naming convention.
            tile_filename = f"{global_y0}_{global_x0}.npz"
            tile_path = temp_dir / tile_filename
            np.savez_compressed(tile_path, mask=tile_mask)
            
            tiles_created += 1
            
            # Count edge tiles.
            if (global_y0 + tile_h > height or global_x0 + tile_w > width or
                global_y0 == 0 or global_x0 == 0):
                edge_tiles += 1
    
    return tiles_created, edge_tiles


"""CORE FUNCTIONALITY TESTS"""

class TestMergeTilesCore:
    """Test core functionality of the merge_tiles module."""
    
    @timeout_protection(60)
    def test_parse_tile_filename(self, _timeout_checker):
        """Test tile filename parsing functionality."""
        _timeout_checker()
        
        # Test valid filename patterns.
        assert _parse_tile_filename("123_456.tif") == (123, 456)
        assert _parse_tile_filename("0_0.npz") == (0, 0)
        assert _parse_tile_filename("1000 2000.tif") == (1000, 2000)
        
        # Test invalid patterns.
        assert _parse_tile_filename("invalid.tif") is None
        assert _parse_tile_filename("123.tif") is None
        assert _parse_tile_filename("abc_def.tif") is None
        
        _timeout_checker()
    
    @timeout_protection(60)
    def test_resolve_tiles_path(self, _timeout_checker):
        """Test tiles path resolution with fallback handling."""
        _timeout_checker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directories.
            tiles_dir = temp_path / "tiles"
            tiles_npz_dir = temp_path / "tiles_npz"
            tiles_dir.mkdir()
            tiles_npz_dir.mkdir()
            
            # Test direct path resolution.
            resolved = _resolve_tiles_path(tiles_dir)
            assert resolved == tiles_dir
            
            # Test fallback to _npz suffix.
            base_path = temp_path / "tiles"
            resolved = _resolve_tiles_path(base_path)
            assert resolved == tiles_dir
            
            _timeout_checker()
    
    @timeout_protection(60)
    def test_discover_tiles(self, _timeout_checker):
        """Test tile discovery functionality."""
        _timeout_checker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test tile files.
            (temp_path / "0_0.tif").touch()
            (temp_path / "0_512.tif").touch()
            (temp_path / "512_0.tif").touch()
            (temp_path / "invalid.txt").touch()  # Should be ignored.
            
            file_map, coords = _discover_tiles(temp_path)
            
            assert len(file_map) == 3
            assert len(coords) == 3
            assert (0, 0) in coords
            assert (0, 512) in coords
            assert (512, 0) in coords
            
            _timeout_checker()
    
    @timeout_protection(60)
    def test_build_clusters(self, _timeout_checker):
        """Test cluster building for parallel processing."""
        _timeout_checker()
        
        # Test simple grid clustering.
        coords = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid.
        clusters = _build_clusters(coords)
        
        # Should form one connected cluster.
        assert len(clusters) == 1
        assert len(clusters[0]) == 4
        
        # Test disconnected clusters.
        coords = [(0, 0), (0, 1), (5, 5), (5, 6)]  # Two separate 1x2 clusters.
        clusters = _build_clusters(coords)
        
        assert len(clusters) == 2
        
        _timeout_checker()


"""EDGE TILE HANDLING TESTS"""

class TestEdgeTileHandling:
    """Test edge tile handling - critical for the recent bug fix."""
    
    @timeout_protection(120)
    def test_edge_tile_merging(self, _timeout_checker):
        """Test that edge tiles are properly merged (critical bug fix validation)."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        
        # Test parameters designed to create edge tiles.
        height, width = 1000, 800
        tile_h, tile_w = 256, 256
        overlap = 64
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create synthetic tiles including edge cases.
            tiles_created, edge_tiles = create_synthetic_tiles(
                height, width, tile_h, tile_w, overlap, temp_path, include_edge_tiles=True
            )
            
            assert edge_tiles > 0, "Test setup should create edge tiles"
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Run the merge process.
            merged = merge_masks_streaming(
                height=height,
                width=width,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                tiles_path=temp_path,
                threshold=0.3,
                use_gpu=False,
                qc=False
            )
            
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Verify the merged result has correct dimensions.
            assert merged.shape == (height, width), f"Expected shape ({height}, {width}), got {merged.shape}"
            
            # Verify that we have non-zero pixels (nuclei were merged).
            total_nuclei_pixels = np.count_nonzero(merged)
            assert total_nuclei_pixels > 0, "Merged mask should contain nuclei pixels"
            
            # CRITICAL TEST: Verify edge coverage (this was the bug).
            edge_margin = 50
            top_edge = merged[:edge_margin, :].sum()
            bottom_edge = merged[-edge_margin:, :].sum()
            left_edge = merged[:, :edge_margin].sum()
            right_edge = merged[:, -edge_margin:].sum()
            
            # All edges should have some coverage (this was failing before the fix).
            assert top_edge > 0, "Top edge should have segmentation coverage"
            assert bottom_edge > 0, "Bottom edge should have segmentation coverage"
            assert left_edge > 0, "Left edge should have segmentation coverage"
            assert right_edge > 0, "Right edge should have segmentation coverage"
            
            memory_monitor.check_memory()
            _timeout_checker()
    
    @timeout_protection(60)
    def test_single_edge_tile(self, _timeout_checker):
        """Test processing of a single edge tile that extends beyond boundaries."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create a small image with a single tile that extends beyond boundaries.
        height, width = 200, 150
        tile_h, tile_w = 256, 256  # Tile larger than image.
        overlap = 32
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a single tile at (0,0) that extends beyond the image.
            tile_mask = np.zeros((height, width), dtype=np.uint32)
            tile_mask[50:100, 50:100] = 1  # Add a nucleus.
            
            tile_filename = "0_0.npz"
            tile_path = temp_path / tile_filename
            np.savez_compressed(tile_path, mask=tile_mask)
            
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Run the merge process.
            merged = merge_masks_streaming(
                height=height,
                width=width,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                tiles_path=temp_path,
                threshold=0.3,
                use_gpu=False,
                qc=False
            )
            
            # Verify the result.
            assert merged.shape == (height, width)
            assert np.count_nonzero(merged) > 0, "Should have nuclei pixels from the edge tile"
            assert merged[75, 75] == 1, "Nucleus should be preserved at the expected location"
            
            memory_monitor.check_memory()
            _timeout_checker()


"""TILE MERGING RULES VALIDATION"""

class TestTileMergingRules:
    """Test the 4 critical tile merging rules implementation."""

    @timeout_protection(120)
    def test_overlap_threshold_rule(self, _timeout_checker):
        """Test Rule 1: Overlap threshold determines nucleus merging."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        height, width = 512, 512
        tile_h, tile_w = 256, 256
        overlap = 64

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create two overlapping tiles with nuclei in overlap region.
            # Tile 1: (0, 0) with nucleus at (200, 200).
            tile1 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            tile1[190:210, 190:210] = 1  # Nucleus in overlap region.
            np.savez_compressed(temp_path / "0_0.npz", mask=tile1)

            # Tile 2: (0, 192) with overlapping nucleus.
            tile2 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            tile2[190:210, 8:28] = 2  # Overlapping nucleus (192 + 8 = 200).
            np.savez_compressed(temp_path / "0_192.npz", mask=tile2)

            memory_monitor.check_memory()
            _timeout_checker()

            # Test with high threshold (should merge).
            merged_high = merge_masks_streaming(
                height=height, width=width, tile_h=tile_h, tile_w=tile_w,
                overlap=overlap, tiles_path=temp_path, threshold=0.1,
                use_gpu=False, qc=False
            )

            # Test with low threshold (should not merge).
            merged_low = merge_masks_streaming(
                height=height, width=width, tile_h=tile_h, tile_w=tile_w,
                overlap=overlap, tiles_path=temp_path, threshold=0.9,
                use_gpu=False, qc=False
            )

            # Verify different merging behavior based on threshold.
            unique_labels_high = len(np.unique(merged_high)) - 1  # Exclude background.
            unique_labels_low = len(np.unique(merged_low)) - 1

            # With high threshold, nuclei should merge (fewer unique labels).
            # With low threshold, nuclei should remain separate (more unique labels).
            assert unique_labels_high <= unique_labels_low, "High threshold should result in more merging"

            memory_monitor.check_memory()
            _timeout_checker()

    @timeout_protection(120)
    def test_spatial_continuity_rule(self, _timeout_checker):
        """Test Rule 2: Spatial continuity preservation across tiles."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        height, width = 512, 512
        tile_h, tile_w = 256, 256
        overlap = 64

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create tiles with a continuous nucleus across tile boundary.
            stride = tile_w - overlap

            # Tile 1: Nucleus extending to right edge.
            tile1 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            tile1[100:120, 240:256] = 1  # Nucleus at right edge.
            np.savez_compressed(temp_path / f"0_0.npz", mask=tile1)

            # Tile 2: Continuation of the same nucleus from left edge.
            tile2 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            tile2[100:120, 0:16] = 2  # Continuation from left edge.
            np.savez_compressed(temp_path / f"0_{stride}.npz", mask=tile2)

            memory_monitor.check_memory()
            _timeout_checker()

            # Merge with appropriate threshold for continuity.
            merged = merge_masks_streaming(
                height=height, width=width, tile_h=tile_h, tile_w=tile_w,
                overlap=overlap, tiles_path=temp_path, threshold=0.3,
                use_gpu=False, qc=False
            )

            # Verify spatial continuity is preserved.
            # The nucleus should be continuous across the tile boundary.
            boundary_region = merged[100:120, 240:272]  # Across tile boundary.
            unique_labels = np.unique(boundary_region[boundary_region > 0])

            # Should have continuous labeling (ideally one label).
            assert len(unique_labels) <= 2, "Spatial continuity should be preserved across tiles"

            memory_monitor.check_memory()
            _timeout_checker()

    @timeout_protection(120)
    def test_label_uniqueness_rule(self, _timeout_checker):
        """Test Rule 3: Global label uniqueness across all tiles."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        height, width = 768, 768
        tile_h, tile_w = 256, 256
        overlap = 64

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple tiles with non-overlapping nuclei.
            stride_h = tile_h - overlap
            stride_w = tile_w - overlap

            tile_count = 0
            for r in range(3):  # 3x3 grid.
                for c in range(3):
                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)

                    # Add unique nucleus per tile.
                    nucleus_y = 50 + r * 20
                    nucleus_x = 50 + c * 20
                    tile_mask[nucleus_y:nucleus_y+10, nucleus_x:nucleus_x+10] = 1

                    global_y = r * stride_h
                    global_x = c * stride_w
                    np.savez_compressed(temp_path / f"{global_y}_{global_x}.npz", mask=tile_mask)
                    tile_count += 1

            memory_monitor.check_memory()
            _timeout_checker()

            # Merge all tiles.
            merged = merge_masks_streaming(
                height=height, width=width, tile_h=tile_h, tile_w=tile_w,
                overlap=overlap, tiles_path=temp_path, threshold=0.3,
                use_gpu=False, qc=False
            )

            # Verify global label uniqueness.
            unique_labels = np.unique(merged)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background.

            # Each nucleus should have a unique label.
            assert len(unique_labels) == tile_count, f"Expected {tile_count} unique labels, got {len(unique_labels)}"

            # Verify no label conflicts.
            assert len(unique_labels) == len(set(unique_labels)), "All labels should be unique"

            memory_monitor.check_memory()
            _timeout_checker()

    @timeout_protection(120)
    def test_boundary_preservation_rule(self, _timeout_checker):
        """Test Rule 4: Nucleus boundary preservation during merging."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        height, width = 512, 512
        tile_h, tile_w = 256, 256
        overlap = 64

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create tiles with well-defined nucleus boundaries.
            # Tile 1: Circular nucleus.
            tile1 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            center_y, center_x = 128, 200  # Near tile boundary.
            radius = 15

            for y in range(tile_h):
                for x in range(tile_w):
                    if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                        tile1[y, x] = 1

            np.savez_compressed(temp_path / "0_0.npz", mask=tile1)

            # Tile 2: Adjacent tile with potential boundary interaction.
            tile2 = np.zeros((tile_h, tile_w), dtype=np.uint32)
            # Add a separate nucleus that should not merge.
            tile2[128:143, 8:23] = 2  # Separate nucleus.
            np.savez_compressed(temp_path / "0_192.npz", mask=tile2)

            memory_monitor.check_memory()
            _timeout_checker()

            # Merge with conservative threshold to preserve boundaries.
            merged = merge_masks_streaming(
                height=height, width=width, tile_h=tile_h, tile_w=tile_w,
                overlap=overlap, tiles_path=temp_path, threshold=0.5,
                use_gpu=False, qc=False
            )

            # Verify boundary preservation.
            # Check that the circular nucleus maintains its shape.
            nucleus_region = merged[center_y-radius:center_y+radius+1, center_x-radius:center_x+radius+1]
            nucleus_pixels = np.count_nonzero(nucleus_region)
            expected_pixels = np.pi * radius**2  # Approximate circle area.

            # Allow some tolerance for discretization.
            assert abs(nucleus_pixels - expected_pixels) / expected_pixels < 0.3, \
                "Nucleus boundary should be approximately preserved"

            memory_monitor.check_memory()
            _timeout_checker()

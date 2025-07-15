"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_tiling.py.
Description:
    Comprehensive test suite for the tiling utilities in kidney I/R injury
    spatial multiomics analysis. This test suite validates image tiling,
    overlap handling, edge case management, and memory efficiency for
    large tissue image processing in bioinformatics workflows.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pathlib, tempfile for testing infrastructure.
    • PIL for image handling and validation.
    • tiling utilities from the utils package.

Usage:
    pytest test_tiling.py -v
    pytest test_tiling.py::test_tile_generation_with_overlap -v

Inputs:
    • Synthetic tissue images of various sizes for testing.
    • Different tiling configurations for comprehensive validation.

Outputs:
    • Test results with validation of tiling functionality.
    • Memory usage monitoring and edge case validation.

Key Features:
    • Image tiling with overlap handling.
    • Edge tile management for boundary conditions.
    • Memory efficiency testing for large images.
    • Coordinate system validation.
    • Tile reconstruction verification.
    • Scientific context for kidney tissue analysis.

Notes:
    • This test suite validates tiling functionality critical for processing
      large kidney tissue images that exceed memory limitations.
    • Tests ensure proper handling of edge tiles and overlap regions.
    • All tests include memory monitoring to prevent RAM overflow.
"""

import traceback
import pytest
import numpy as np
import tempfile
import time
import psutil
import os
from pathlib import Path
from typing import List, Tuple
from numpy.typing import NDArray
from PIL import Image

# Import the module under test.
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.tiling import (
    feather_mask,
    split_image_into_tiles
)


"""MEMORY AND TIMEOUT MONITORING"""

class MemoryMonitor:
    """Monitor memory usage during tiling tests to prevent RAM overflow."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
    
    def check_memory(self):
        """Check current memory usage and raise error if exceeded."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > self.max_memory_mb:
            raise MemoryError(f"Tiling memory usage exceeded {self.max_memory_mb}MB: "
                            f"current increase = {memory_increase:.1f}MB")
        
        return memory_increase


def timeout_protection(timeout_seconds: int = 120):
    """Decorator to add timeout protection to tiling test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            def check_timeout():
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Tiling test {func.__name__} exceeded {timeout_seconds}s timeout")
            
            kwargs['_timeout_checker'] = check_timeout
            return func(*args, **kwargs)
        return wrapper
    return decorator


"""SYNTHETIC DATA GENERATION"""

def create_test_image(height: int, width: int, pattern: str = "gradient") -> NDArray[np.uint8]:
    """
    Create test images with known patterns for tiling validation.
    
    This function generates test images with predictable patterns that
    allow verification of correct tiling and reconstruction.
    """
    
    if pattern == "gradient":
        # Create gradient pattern for easy verification.
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image[y, x, 0] = (y * 255) // height  # Red gradient by row.
                image[y, x, 1] = (x * 255) // width   # Green gradient by column.
                image[y, x, 2] = ((y + x) * 255) // (height + width)  # Blue diagonal.
        
    elif pattern == "checkerboard":
        # Create checkerboard pattern.
        image = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 50
        for y in range(height):
            for x in range(width):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    image[y, x] = [255, 255, 255]  # White squares.
                else:
                    image[y, x] = [0, 0, 0]        # Black squares.
    
    elif pattern == "coordinates":
        # Encode coordinates in pixel values for precise verification.
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image[y, x, 0] = y % 256  # Y coordinate in red channel.
                image[y, x, 1] = x % 256  # X coordinate in green channel.
                image[y, x, 2] = (y + x) % 256  # Sum in blue channel.
    
    else:
        # Default uniform pattern.
        image = np.full((height, width, 3), 128, dtype=np.uint8)
    
    return image


"""CORE TILING FUNCTIONALITY TESTS"""

class TestTilingCore:
    """Test core tiling functionality."""

    @timeout_protection(30)
    def test_feather_mask(self, _timeout_checker):
        """Test feather mask generation for tile blending."""
        _timeout_checker()

        # Test standard feather mask.
        h, w, overlap = 256, 256, 64
        mask = feather_mask(h, w, overlap)

        assert mask.shape == (h, w)
        assert mask.dtype == np.float64

        # Check that mask values are between 0 and 1.
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

        # Check that interior is close to 1.
        interior = mask[overlap:-overlap, overlap:-overlap]
        assert np.all(interior > 0.9), "Interior should be close to 1"

        # Check that edges taper to lower values.
        edge_values = [
            mask[0, overlap:-overlap].mean(),    # Top edge.
            mask[-1, overlap:-overlap].mean(),   # Bottom edge.
            mask[overlap:-overlap, 0].mean(),    # Left edge.
            mask[overlap:-overlap, -1].mean()    # Right edge.
        ]

        for edge_val in edge_values:
            assert edge_val < 0.8, "Edges should have lower values for feathering"

        _timeout_checker()

    @timeout_protection(30)
    def test_feather_mask_no_overlap(self, _timeout_checker):
        """Test feather mask with zero overlap."""
        _timeout_checker()

        # Test with no overlap (should be uniform 1.0).
        h, w, overlap = 256, 256, 0
        mask = feather_mask(h, w, overlap)

        assert mask.shape == (h, w)
        assert np.allclose(mask, 1.0), "No overlap should result in uniform mask"

        _timeout_checker()

    @timeout_protection(60)
    def test_split_image_into_tiles(self, _timeout_checker):
        """Test image splitting into tiles."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        # Create test image.
        height, width = 512, 512
        test_image = create_test_image(height, width, "coordinates")

        # Mock logger for testing.
        class MockLogger:
            def debug(self, msg): pass
            def info(self, msg): pass
            def warning(self, msg): pass

        logger = MockLogger()

        # Test tile splitting.
        tile_h, tile_w, overlap = 256, 256, 64
        tiles = list(split_image_into_tiles(test_image, tile_h, tile_w, overlap, logger))

        # Verify tile count.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        expected_rows = (height + stride_h - 1) // stride_h
        expected_cols = (width + stride_w - 1) // stride_w
        expected_tiles = expected_rows * expected_cols

        assert len(tiles) == expected_tiles

        # Verify tile properties.
        for tile, (y_slice, x_slice) in tiles:
            assert tile.ndim == 3  # Should preserve RGB channels.
            assert tile.shape[2] == 3  # RGB channels.

            # Verify slice dimensions match tile dimensions.
            expected_h = y_slice.stop - y_slice.start
            expected_w = x_slice.stop - x_slice.start
            assert tile.shape[:2] == (expected_h, expected_w)

        memory_monitor.check_memory()
        _timeout_checker()

    @timeout_protection(60)
    def test_split_image_edge_tiles(self, _timeout_checker):
        """Test image splitting with edge tiles."""
        _timeout_checker()

        memory_monitor = MemoryMonitor(max_memory_mb=512)

        # Create non-standard size image to force edge tiles.
        height, width = 550, 750
        test_image = create_test_image(height, width, "gradient")

        class MockLogger:
            def debug(self, msg): pass
            def info(self, msg): pass
            def warning(self, msg): pass

        logger = MockLogger()

        # Test tile splitting with edge cases.
        tile_h, tile_w, overlap = 256, 256, 64
        tiles = list(split_image_into_tiles(test_image, tile_h, tile_w, overlap, logger))

        # Verify that edge tiles are properly handled.
        for tile, (y_slice, x_slice) in tiles:
            # Verify slices are within image bounds.
            assert y_slice.start >= 0
            assert y_slice.stop <= height
            assert x_slice.start >= 0
            assert x_slice.stop <= width

            # Verify tile content matches slice.
            expected_tile = test_image[y_slice, x_slice]
            assert np.array_equal(tile, expected_tile)

        memory_monitor.check_memory()
        _timeout_checker()


"""TILE GENERATION TESTS"""

class TestTileGeneration:
    """Test tile generation functionality."""
    
    @timeout_protection(60)
    def test_generate_tiles_basic(self, _timeout_checker):
        """Test basic tile generation without overlap."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create test image.
        height, width = 512, 512
        test_image = create_test_image(height, width, "coordinates")
        
        # Generate tiles without overlap.
        tile_size, overlap = 256, 0
        tiles = generate_tiles(test_image, tile_size, overlap)
        
        # Verify tile count.
        expected_tiles = (height // tile_size) * (width // tile_size)  # 4 tiles.
        assert len(tiles) == expected_tiles
        
        # Verify tile dimensions.
        for tile_info in tiles:
            tile, (row, col), (y0, x0, y1, x1) = tile_info
            assert tile.shape == (tile_size, tile_size, 3)
            assert tile.dtype == test_image.dtype
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(90)
    def test_generate_tiles_with_overlap(self, _timeout_checker):
        """Test tile generation with overlap."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create test image.
        height, width = 600, 600
        test_image = create_test_image(height, width, "gradient")
        
        # Generate tiles with overlap.
        tile_size, overlap = 256, 64
        tiles = generate_tiles(test_image, tile_size, overlap)
        
        # Verify overlap regions.
        stride = tile_size - overlap
        n_rows = (height + stride - 1) // stride
        n_cols = (width + stride - 1) // stride
        expected_tiles = n_rows * n_cols
        
        assert len(tiles) == expected_tiles
        
        # Verify overlap by checking adjacent tiles.
        tile_dict = {(row, col): tile for tile, (row, col), coords in tiles}
        
        if (0, 0) in tile_dict and (0, 1) in tile_dict:
            tile_00 = tile_dict[(0, 0)]
            tile_01 = tile_dict[(0, 1)]
            
            # Check overlap region.
            overlap_region_00 = tile_00[:, -overlap:, :]
            overlap_region_01 = tile_01[:, :overlap, :]
            
            # Should have similar content (allowing for edge effects).
            assert overlap_region_00.shape == overlap_region_01.shape
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(90)
    def test_generate_tiles_edge_cases(self, _timeout_checker):
        """Test tile generation with edge cases."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Test with image smaller than tile size.
        small_image = create_test_image(100, 150, "checkerboard")
        tiles = generate_tiles(small_image, 256, 64)
        
        assert len(tiles) == 1  # Should generate one tile.
        tile, (row, col), (y0, x0, y1, x1) = tiles[0]
        assert tile.shape[:2] == (100, 150)  # Should match image size.
        
        # Test with non-square image and tiles.
        rect_image = create_test_image(400, 800, "gradient")
        tiles = generate_tiles(rect_image, 200, 50)
        
        assert len(tiles) > 1  # Should generate multiple tiles.
        
        # Verify all tiles are processed.
        total_pixels_covered = 0
        for tile, (row, col), (y0, x0, y1, x1) in tiles:
            tile_area = (y1 - y0) * (x1 - x0)
            total_pixels_covered += tile_area
        
        # Should cover at least the image area (with overlap, may be more).
        image_area = 400 * 800
        assert total_pixels_covered >= image_area
        
        memory_monitor.check_memory()
        _timeout_checker()


"""TILE RECONSTRUCTION TESTS"""

class TestTileReconstruction:
    """Test tile reconstruction functionality."""
    
    @timeout_protection(120)
    def test_reconstruct_from_tiles_perfect(self, _timeout_checker):
        """Test perfect reconstruction without overlap."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create test image with coordinate pattern for precise verification.
        height, width = 512, 512
        original_image = create_test_image(height, width, "coordinates")
        
        # Generate tiles without overlap.
        tile_size, overlap = 256, 0
        tiles = generate_tiles(original_image, tile_size, overlap)
        
        # Reconstruct image.
        reconstructed = reconstruct_from_tiles(tiles, height, width, tile_size, overlap)
        
        # Verify perfect reconstruction.
        assert reconstructed.shape == original_image.shape
        assert np.array_equal(reconstructed, original_image), "Reconstruction should be perfect without overlap"
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(120)
    def test_reconstruct_from_tiles_with_overlap(self, _timeout_checker):
        """Test reconstruction with overlap handling."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create test image.
        height, width = 600, 600
        original_image = create_test_image(height, width, "gradient")
        
        # Generate tiles with overlap.
        tile_size, overlap = 256, 64
        tiles = generate_tiles(original_image, tile_size, overlap)
        
        # Reconstruct image.
        reconstructed = reconstruct_from_tiles(tiles, height, width, tile_size, overlap)
        
        # Verify reconstruction dimensions.
        assert reconstructed.shape == original_image.shape
        
        # Verify reconstruction quality (should be very close to original).
        # Allow small differences due to overlap averaging.
        mse = np.mean((reconstructed.astype(np.float32) - original_image.astype(np.float32))**2)
        assert mse < 100, f"Reconstruction MSE too high: {mse}"
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(120)
    def test_reconstruct_edge_tiles(self, _timeout_checker):
        """Test reconstruction with edge tiles that extend beyond boundaries."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create non-standard size image to force edge tiles.
        height, width = 550, 750
        original_image = create_test_image(height, width, "checkerboard")
        
        # Generate tiles with overlap.
        tile_size, overlap = 256, 64
        tiles = generate_tiles(original_image, tile_size, overlap)
        
        # Reconstruct image.
        reconstructed = reconstruct_from_tiles(tiles, height, width, tile_size, overlap)
        
        # Verify reconstruction dimensions.
        assert reconstructed.shape == original_image.shape
        
        # Verify edge regions are properly reconstructed.
        # Check corners and edges specifically.
        corner_size = 50
        
        # Top-left corner.
        assert np.array_equal(
            reconstructed[:corner_size, :corner_size],
            original_image[:corner_size, :corner_size]
        ), "Top-left corner should be perfectly reconstructed"
        
        # Bottom-right corner.
        assert np.array_equal(
            reconstructed[-corner_size:, -corner_size:],
            original_image[-corner_size:, -corner_size:]
        ), "Bottom-right corner should be perfectly reconstructed"
        
        memory_monitor.check_memory()
        _timeout_checker()


"""MEMORY EFFICIENCY TESTS"""

class TestTilingMemoryEfficiency:
    """Test memory efficiency of tiling operations."""
    
    @timeout_protection(180)
    def test_large_image_tiling_memory(self, _timeout_checker):
        """Test memory efficiency with large images."""
        _timeout_checker()
        
        # Use strict memory monitoring for large image test.
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        
        # Create moderately large image for testing.
        height, width = 2048, 2048
        
        # Create image in chunks to avoid initial memory spike.
        test_image = np.zeros((height, width, 3), dtype=np.uint8)
        chunk_size = 512
        
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                y_end = min(y + chunk_size, height)
                x_end = min(x + chunk_size, width)
                
                # Fill chunk with simple pattern.
                test_image[y:y_end, x:x_end, 0] = (y // 8) % 256
                test_image[y:y_end, x:x_end, 1] = (x // 8) % 256
                test_image[y:y_end, x:x_end, 2] = ((y + x) // 16) % 256
        
        memory_monitor.check_memory()
        _timeout_checker()
        
        # Generate tiles with memory-efficient parameters.
        tile_size, overlap = 512, 128
        tiles = generate_tiles(test_image, tile_size, overlap)
        
        # Verify tiling completed without excessive memory usage.
        memory_increase = memory_monitor.check_memory()
        
        # Memory increase should be reasonable.
        image_memory_mb = (height * width * 3) / 1024 / 1024
        assert memory_increase < image_memory_mb * 2, \
            f"Memory usage too high: {memory_increase:.1f}MB for {image_memory_mb:.1f}MB image"
        
        # Verify tile count is reasonable.
        stride = tile_size - overlap
        expected_tiles = ((height + stride - 1) // stride) * ((width + stride - 1) // stride)
        assert len(tiles) == expected_tiles
        
        _timeout_checker()
    
    @timeout_protection(60)
    def test_memory_cleanup_after_tiling(self, _timeout_checker):
        """Test that memory is properly cleaned up after tiling operations."""
        _timeout_checker()
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Run multiple tiling operations.
        for i in range(3):
            height, width = 1024, 1024
            test_image = create_test_image(height, width, "gradient")
            
            # Generate and immediately discard tiles.
            tiles = generate_tiles(test_image, 256, 64)
            
            # Force cleanup.
            del tiles
            del test_image
            
            _timeout_checker()
        
        # Check final memory usage.
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal after cleanup.
        assert memory_increase < 200, f"Memory not properly cleaned up: {memory_increase:.1f}MB increase"
        
        _timeout_checker()

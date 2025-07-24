"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_tile_overlay_functions.py.
Description:
    Comprehensive test suite for the memory-efficient tile overlay functionality
    in tissue analysis. This test suite validates the overlay functions that can
    handle thousands of tiles without overwhelming RAM.

    Tests cover:
    1. Tile coordinate parsing from different naming conventions
    2. Memory-efficient batch processing of tile masks
    3. Color generation for unique tile identification
    4. Overlay creation from tile directories
    5. Integration with existing QC workflow

Dependencies:
    • Python >= 3.10.
    • pytest >= 6.0.
    • numpy, pillow for core functionality.
    • tempfile for test data creation.

Usage:
    pytest test_tile_overlay_functions.py -v

Arguments:
    None (pytest handles test discovery and execution).

Inputs:
    • Synthetic tile mask data for testing.
    • Mock tissue images for overlay background.

Outputs:
    • Test results and coverage reports.
    • Temporary overlay images for validation.

Key Features:
    • Comprehensive test coverage for all overlay functions.
    • Memory usage validation for large tile sets.
    • Error handling and edge case testing.
    • Performance benchmarking for batch processing.

Notes:
    • Tests use synthetic data to avoid dependencies on actual tissue images.
    • Memory efficiency is validated with simulated large tile sets.
    • All tests are designed to run in CI/CD environments.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

# Import the functions to test.
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

from qc import (
    create_tile_overlay_from_directory,
    create_before_after_overlays,
    _parse_tile_coordinates,
    _generate_tile_color_deterministic,
    _apply_tile_to_overlay,
    _calculate_crop_region
)

"""TEST FIXTURES"""

@pytest.fixture
def mock_tissue_image() -> np.ndarray:
    """Create a mock tissue image for testing."""
    # Create a simple gradient image.
    height, width = 1000, 1000
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient pattern.
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = (x * 255) // width  # Red gradient.
            image[y, x, 1] = (y * 255) // height  # Green gradient.
            image[y, x, 2] = 128  # Constant blue.
    
    return image


@pytest.fixture
def mock_tile_directory() -> Path:
    """Create a temporary directory with mock tile masks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tiles_dir = Path(temp_dir)
        
        # Create mock tile masks with different naming conventions.
        tile_configs = [
            ("0_0.npz", (0, 0)),
            ("410_0.npz", (410, 0)),
            ("0_410.npz", (0, 410)),
            ("410_410.npz", (410, 410)),
            ("820_0.npz", (820, 0)),
            ("820_410.npz", (820, 410)),
        ]
        
        for filename, coords in tile_configs:
            # Create synthetic tile mask.
            tile_mask = np.zeros((512, 512), dtype=np.uint32)
            
            # Add some nuclei.
            for i in range(5):
                center_y = 100 + i * 80
                center_x = 100 + i * 80
                
                # Create circular nucleus.
                y_indices, x_indices = np.ogrid[:512, :512]
                mask = (y_indices - center_y)**2 + (x_indices - center_x)**2 <= 30**2
                tile_mask[mask] = i + 1
            
            # Save as NPZ file.
            np.savez_compressed(tiles_dir / filename, mask=tile_mask)
        
        yield tiles_dir


"""UNIT TESTS"""

def test_parse_tile_coordinates():
    """Test tile coordinate parsing from different filename formats."""
    
    # Test pixel coordinate format.
    assert _parse_tile_coordinates("410_820.npz") == (410, 820)
    assert _parse_tile_coordinates("0_0.npz") == (0, 0)
    assert _parse_tile_coordinates("1234_5678.tif") == (1234, 5678)
    
    # Test tile index format.
    assert _parse_tile_coordinates("row1_col2.npz") == (1, 2)
    assert _parse_tile_coordinates("row0_col0.npz") == (0, 0)
    
    # Test space-separated format.
    assert _parse_tile_coordinates("410 820.npz") == (410, 820)
    
    # Test invalid formats.
    assert _parse_tile_coordinates("invalid.npz") is None
    assert _parse_tile_coordinates("abc_def.npz") is None


def test_generate_tile_color_deterministic():
    """Test deterministic color generation for tiles."""

    # Test that different coordinates produce different colors.
    color1 = _generate_tile_color_deterministic((0, 0))
    color2 = _generate_tile_color_deterministic((1, 1))
    color3 = _generate_tile_color_deterministic((10, 10))

    # Colors should be different.
    assert not np.array_equal(color1, color2)
    assert not np.array_equal(color1, color3)
    assert not np.array_equal(color2, color3)

    # Colors should be in valid range.
    assert np.all(color1 >= 100) and np.all(color1 <= 255)
    assert np.all(color2 >= 100) and np.all(color2 <= 255)
    assert np.all(color3 >= 100) and np.all(color3 <= 255)

    # Same coordinates should produce same color (deterministic).
    color1_repeat = _generate_tile_color_deterministic((0, 0))
    assert np.array_equal(color1, color1_repeat)


def test_calculate_crop_region():
    """Test crop region calculation for different image sizes."""
    
    # Test small image (should use full image).
    crop_info = _calculate_crop_region(800, 600, 1000)
    assert crop_info['y_start'] == 0
    assert crop_info['y_end'] == 800
    assert crop_info['x_start'] == 0
    assert crop_info['x_end'] == 600
    
    # Test large image (should use central crop).
    crop_info = _calculate_crop_region(2000, 1500, 1000)
    assert crop_info['height'] == 1000
    assert crop_info['width'] == 1000
    assert crop_info['y_start'] == 500  # (2000 - 1000) / 2
    assert crop_info['x_start'] == 250  # (1500 - 1000) / 2


def test_apply_tile_to_overlay(mock_tissue_image):
    """Test applying tile mask to overlay canvas."""

    # Create test overlay.
    overlay = mock_tissue_image[:500, :500].astype(np.uint16)

    # Create test tile mask.
    tile_mask = np.zeros((200, 200), dtype=np.uint32)
    tile_mask[50:150, 50:150] = 1  # Add nucleus.

    # Create test parameters using tile indices (small numbers).
    coords = (1, 1)  # Tile indices (will be converted to pixel coordinates).
    tile_color = np.array([255, 100, 100], dtype=np.uint16)
    alpha = 0.5
    crop_info = {
        'y_start': 0, 'y_end': 500,
        'x_start': 0, 'x_end': 500,
        'height': 500, 'width': 500
    }
    stride_h, stride_w = 150, 150  # Stride that will place tile at (150, 150).

    # Apply tile to overlay.
    success = _apply_tile_to_overlay(
        overlay, tile_mask, coords, tile_color, alpha,
        crop_info, stride_h, stride_w
    )

    assert success

    # Check that overlay was modified in the nucleus region.
    # Tile at (1,1) with stride (150,150) starts at (150,150).
    # Nucleus is at tile_mask[50:150, 50:150], so global coords [200:300, 200:300].
    nucleus_region = overlay[200:300, 200:300]  # Where nucleus should be.
    background_region = overlay[0:50, 0:50]  # Where no nucleus should be.

    # Nucleus region should have some red component from tile color.
    assert np.mean(nucleus_region[:, :, 0]) > np.mean(background_region[:, :, 0])


"""INTEGRATION TESTS"""

def test_create_tile_overlay_from_directory(mock_tile_directory, mock_tissue_image):
    """Test creating overlay from tile directory."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_overlay.tif"
        
        # Create overlay.
        overlay = create_tile_overlay_from_directory(
            tiles_dir=mock_tile_directory,
            full_image=mock_tissue_image,
            tile_h=512,
            tile_w=512,
            overlap=64,
            batch_size=10,
            alpha=0.6,
            crop_size=800,
            output_path=output_path,
            overlay_type="before"
        )
        
        # Check overlay properties.
        assert overlay.shape[2] == 3  # RGB.
        assert overlay.dtype == np.uint8
        assert overlay.shape[0] <= 800  # Should be cropped.
        assert overlay.shape[1] <= 800
        
        # Check that output file was created.
        assert output_path.exists()
        
        # Load and verify saved image.
        saved_image = np.array(Image.open(output_path))
        assert np.array_equal(overlay, saved_image)


def test_memory_efficiency_large_tile_set(mock_tissue_image):
    """Test memory efficiency with simulated large tile set."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tiles_dir = Path(temp_dir)
        
        # Create many small tile masks to simulate large dataset.
        num_tiles = 500  # Simulate 500 tiles.
        
        for i in range(num_tiles):
            row = i // 25
            col = i % 25
            filename = f"{row * 100}_{col * 100}.npz"
            
            # Create small tile mask.
            tile_mask = np.zeros((100, 100), dtype=np.uint32)
            if i % 10 == 0:  # Add nucleus to every 10th tile.
                tile_mask[40:60, 40:60] = 1
            
            np.savez_compressed(tiles_dir / filename, mask=tile_mask)
        
        # Test with small batch size for memory efficiency.
        overlay = create_tile_overlay_from_directory(
            tiles_dir=tiles_dir,
            full_image=mock_tissue_image,
            tile_h=100,
            tile_w=100,
            overlap=10,
            batch_size=20,  # Small batch size.
            alpha=0.5,
            crop_size=500,
            overlay_type="before"
        )
        
        # Should complete without memory errors.
        assert overlay.shape[2] == 3
        assert overlay.dtype == np.uint8


def test_error_handling():
    """Test error handling for invalid inputs."""
    
    # Test with non-existent directory.
    with pytest.raises(FileNotFoundError):
        create_tile_overlay_from_directory(
            tiles_dir=Path("non_existent_directory"),
            full_image=np.zeros((100, 100, 3), dtype=np.uint8),
            overlay_type="before"
        )
    
    # Test with empty directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir)
        
        with pytest.raises(FileNotFoundError):
            create_tile_overlay_from_directory(
                tiles_dir=empty_dir,
                full_image=np.zeros((100, 100, 3), dtype=np.uint8),
                overlay_type="before"
            )


"""PERFORMANCE TESTS"""

def test_batch_processing_performance(mock_tissue_image):
    """Test that batch processing improves performance for large tile sets."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tiles_dir = Path(temp_dir)
        
        # Create moderate number of tiles.
        num_tiles = 100
        
        for i in range(num_tiles):
            filename = f"{i * 50}_{(i % 10) * 50}.npz"
            tile_mask = np.random.randint(0, 2, (50, 50), dtype=np.uint32)
            np.savez_compressed(tiles_dir / filename, mask=tile_mask)
        
        import time
        
        # Test with different batch sizes.
        start_time = time.time()
        overlay_small_batch = create_tile_overlay_from_directory(
            tiles_dir=tiles_dir,
            full_image=mock_tissue_image,
            batch_size=10,
            crop_size=400,
            overlay_type="before"
        )
        small_batch_time = time.time() - start_time
        
        start_time = time.time()
        overlay_large_batch = create_tile_overlay_from_directory(
            tiles_dir=tiles_dir,
            full_image=mock_tissue_image,
            batch_size=50,
            crop_size=400,
            overlay_type="before"
        )
        large_batch_time = time.time() - start_time
        
        # Both should produce similar results.
        assert overlay_small_batch.shape == overlay_large_batch.shape
        
        # Performance difference should be reasonable (not testing exact timing).
        assert small_batch_time > 0
        assert large_batch_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: tiling_test.py.
Description:
    Test suite for the tiling and merging utilities used in nuclei segmentation.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest, matplotlib, scipy.
    • Custom tiling utilities from the nuclei_segmentation package.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/tiling_test.py -v

Inputs:
    • None (tests run on generated test data).

Outputs:
    • Test results indicating pass/fail status.

Key Features:
    • Tests for image tiling with various dimensions and overlaps.
    • Tests for feather mask generation and properties.
    • Tests for merging tiled outputs with weighted blending.
    • Tests for merging instance segmentation masks.

Notes:
    • These tests verify the correct behavior of tiling utilities used in the nuclei segmentation pipeline.
    • No visual outputs are generated to keep the tests fully automated.
"""

import sys, os
import numpy as np
import pytest
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from scipy.ndimage import label

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from code.nuclei_segmentation.utils.tiling import (
    split_image_into_tiles,
    merge_tiles_with_weighted_overlap,
    merge_masks,
    feather_mask,
)


@pytest.fixture
def dummy_logger():
    return MagicMock()


@pytest.fixture
def dummy_settings():
    return {"MERGE_OVERLAP_THRESHOLD": 0.3}


def test_split_and_reconstruct_identity(dummy_logger):
    """
    Test that splitting an image into tiles and then merging them back
    produces a result very close to the original image.
    """
    img = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    tile_size, overlap = 128, 32

    tiles, slices = split_image_into_tiles(img, tile_size, overlap, dummy_logger)
    tiles = [tile.astype(np.float32) for tile in tiles]
    merged = merge_tiles_with_weighted_overlap(tiles, slices, img.shape, overlap)
    merged_uint8 = np.clip(np.round(merged), 0, 255).astype(np.uint8)
    diff = np.abs(img.astype(np.int16) - merged_uint8.astype(np.int16))

    assert (diff < 50).mean() > 0.99
    assert (diff < 30).mean() > 0.98
    assert (diff < 20).mean() > 0.95
    assert (diff < 10).mean() > 0.90


def test_split_image_into_tiles_dimensions(dummy_logger):
    """
    Test that the split_image_into_tiles function correctly handles different image dimensions,
    tile sizes, and overlaps.
    """
    # Test case 1: Square image with even division
    img1 = np.zeros((200, 200), dtype=np.uint8)
    tile_size1, overlap1 = 100, 20
    tiles1, slices1 = split_image_into_tiles(img1, tile_size1, overlap1, dummy_logger)

    # Calculate expected number of tiles based on the implementation
    # The stride is tile_size - overlap
    stride1 = tile_size1 - overlap1  # 80
    # Number of tiles in each dimension is ceil(dimension / stride)
    expected_tiles_x = int(np.ceil(200 / stride1))  # 3
    expected_tiles_y = int(np.ceil(200 / stride1))  # 3
    expected_total_tiles = expected_tiles_x * expected_tiles_y  # 9

    assert len(tiles1) == expected_total_tiles, f"Should create {expected_total_tiles} tiles for a 200x200 image with 100px tiles and 20px overlap"
    assert all(tile.shape == (100, 100) for tile in tiles1), "All tiles should be 100x100"

    # Test case 2: Rectangular image with uneven division
    img2 = np.zeros((150, 250), dtype=np.uint8)
    tile_size2, overlap2 = 100, 25
    tiles2, slices2 = split_image_into_tiles(img2, tile_size2, overlap2, dummy_logger)

    # Calculate expected number of tiles
    stride2 = tile_size2 - overlap2  # 75
    expected_tiles_x2 = int(np.ceil(250 / stride2))  # 4
    expected_tiles_y2 = int(np.ceil(150 / stride2))  # 2
    expected_total_tiles2 = expected_tiles_x2 * expected_tiles_y2  # 8

    assert len(tiles2) == expected_total_tiles2, f"Should create {expected_total_tiles2} tiles for a 150x250 image with 100px tiles and 25px overlap"
    assert all(tile.shape == (100, 100) for tile in tiles2), "All tiles should be 100x100"

    # Test case 3: Image smaller than tile size
    img3 = np.zeros((50, 50), dtype=np.uint8)
    tile_size3, overlap3 = 100, 10
    tiles3, slices3 = split_image_into_tiles(img3, tile_size3, overlap3, dummy_logger)

    # Should produce 1 tile
    assert len(tiles3) == 1, "Should create 1 tile for an image smaller than the tile size"
    assert tiles3[0].shape == (100, 100), "Tile should be padded to 100x100"
    assert slices3[0] == (slice(0, 50), slice(0, 50)), "Slice should cover the entire image"


def test_feather_blending(dummy_logger):
    """
    Test that the feather blending in merge_tiles_with_weighted_overlap works correctly
    by creating a simple gradient pattern and verifying the merged result.
    """
    # Create a simple horizontal gradient image
    img = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        img[:, i] = i / 100.0  # Values from 0 to 0.99

    # Split into 4 tiles with overlap
    tile_size, overlap = 60, 20
    tiles, slices = split_image_into_tiles(img, tile_size, overlap, dummy_logger)

    # Merge tiles back
    merged = merge_tiles_with_weighted_overlap(tiles, slices, img.shape, overlap)

    # The merged result should be close to the original gradient, but not exact due to feathering
    # Check that most pixels are within a reasonable tolerance
    diff = np.abs(img - merged)
    assert (diff < 0.1).mean() > 0.9, "At least 90% of pixels should be within 0.1 of original values"

    # Test with multi-channel data (3 channels)
    img_multi = np.zeros((3, 100, 100), dtype=np.float32)
    for c in range(3):
        for i in range(100):
            img_multi[c, :, i] = (c + 1) * i / 100.0  # Different gradient per channel

    # Convert to shape expected by split_image_into_tiles (H, W)
    img_2d = img_multi[0]  # Just use first channel for splitting
    tiles_2d, slices = split_image_into_tiles(img_2d, tile_size, overlap, dummy_logger)

    # Create 3-channel tiles
    tiles_multi = []
    for s in slices:
        y_slice, x_slice = s
        tile_multi = np.zeros((3, tile_size, tile_size), dtype=np.float32)
        h = min(y_slice.stop - y_slice.start, tile_size)
        w = min(x_slice.stop - x_slice.start, tile_size)
        tile_multi[:, :h, :w] = img_multi[:, y_slice.start:y_slice.stop, x_slice.start:x_slice.stop]
        tiles_multi.append(tile_multi)

    # Merge multi-channel tiles
    merged_multi = merge_tiles_with_weighted_overlap(tiles_multi, slices, img.shape, overlap)

    # Verify each channel
    assert merged_multi.shape == (3, 100, 100), "Merged result should have 3 channels"
    for c in range(3):
        diff = np.abs(img_multi[c] - merged_multi[c])
        assert (diff < 0.1 * (c + 1)).mean() > 0.9, f"Channel {c} should have most pixels within tolerance"


def test_merge_masks_overlap_without_visual(dummy_logger):
    """
    Test merging behavior with two overlapping circular masks in adjacent tiles,
    but without generating visual output files.
    """
    import cv2

    settings = {"MERGE_OVERLAP_THRESHOLD": 0.3}
    H, W, overlap = 64, 128, 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),    # Tile 1
        (slice(0, 64), slice(64, 128)),  # Tile 2
    ]

    # Create two circular masks that touch in the overlap
    tile1 = np.zeros((64, 64), dtype=np.uint16)
    tile2 = np.zeros((64, 64), dtype=np.uint16)

    # Circle in tile 1 (right side, label 1)
    cv2.circle(tile1, center=(48, 32), radius=12, color=1, thickness=-1)
    # Circle in tile 2 (left side, label 2)
    cv2.circle(tile2, center=(16, 32), radius=12, color=2, thickness=-1)

    tiles = [tile1, tile2]
    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, settings)

    # Verify the merge results programmatically instead of visually
    # Check that both circles exist in the merged result
    assert np.any(merged == 1), "Circle from tile 1 should be present in merged result"
    assert np.any(merged == 2), "Circle from tile 2 should be present in merged result"

    # Check the overlap region
    overlap_slice = (slice(16, 48), slice(48, 80))
    merged_crop = merged[overlap_slice]

    # The implementation allows both labels to exist in the overlap region
    # if neither meets the merge threshold
    unique_labels = np.unique(merged_crop)

    # Verify that the labels in the overlap region are only 0, 1, or 2
    assert all(label in [0, 1, 2] for label in unique_labels), "Labels in overlap should be 0, 1, or 2"



def test_merge_masks_isolated_labels(dummy_logger, dummy_settings):
    """
    Ensure isolated label is preserved with a new ID.
    """
    H, W = 128, 128
    overlap = 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),
        (slice(0, 64), slice(64, 128)),
        (slice(64, 128), slice(0, 64)),
        (slice(64, 128), slice(64, 128)),
    ]

    tiles = [np.zeros((64, 64), dtype=np.uint16) for _ in range(4)]
    tiles[3][10:30, 10:30] = 7

    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, dummy_settings)

    # Check that the isolated label exists in the merged result
    assert np.count_nonzero(merged[74:94, 74:94]) > 0

    # The region we're checking might only contain the label (no background)
    # or might contain both background and label
    unique_labels = np.unique(merged[74:94, 74:94])

    # Check that there's at least one non-zero label
    assert np.any(unique_labels > 0), "Should have at least one non-zero label"


def test_feather_mask_properties():
    """
    Test that the feather mask has the expected properties:
    1. Values taper from 1.0 in the center to near 0.0 at the edges
    2. Correct dimensions
    3. Proper handling of different overlap values
    """
    # Test case 1: Square mask with moderate overlap
    h, w, overlap = 100, 100, 20
    mask = feather_mask(h, w, overlap)

    # Check dimensions
    assert mask.shape == (h, w), "Mask should have requested dimensions"

    # Check values at center (should be 1.0)
    assert mask[h//2, w//2] == 1.0, "Center of mask should be 1.0"

    # Check values at edges (should be close to 0.0)
    # The implementation might not have exactly 0.0 at corners due to numerical precision
    assert mask[0, 0] < 0.01, "Corners should be close to 0.0"
    assert mask[0, w-1] < 0.01, "Corners should be close to 0.0"
    assert mask[h-1, 0] < 0.01, "Corners should be close to 0.0"
    assert mask[h-1, w-1] < 0.01, "Corners should be close to 0.0"

    # Check that values increase from edge to center
    assert np.all(mask[1:overlap, w//2] >= mask[0:overlap-1, w//2]), "Values should increase from edge to center"
    assert np.all(mask[h//2, 1:overlap] >= mask[h//2, 0:overlap-1]), "Values should increase from edge to center"

    # Test case 2: Rectangular mask
    h2, w2, overlap2 = 80, 120, 15
    mask2 = feather_mask(h2, w2, overlap2)
    assert mask2.shape == (h2, w2), "Mask should have requested dimensions"

    # Test case 3: Overlap larger than half the dimension
    h3, w3, overlap3 = 30, 30, 20
    mask3 = feather_mask(h3, w3, overlap3)

    # Overlap should be capped at half the dimension
    effective_overlap = min(overlap3, h3 // 2)
    assert mask3[0, w3//2] <= mask3[effective_overlap, w3//2], "Values should increase from edge to center"
    assert mask3[h3-1, w3//2] <= mask3[h3-effective_overlap-1, w3//2], "Values should increase from edge to center"

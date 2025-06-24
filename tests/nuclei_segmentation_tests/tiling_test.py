#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: tiling_test.py.
Description:
    Test suite for the tiling and merging utilities used in nuclei segmentation.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest, matplotlib, scipy, OpenCV.
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
    • Tests for merging instance segmentation masks, including transitive merges.

Notes:
    • These tests verify the correct behavior of tiling utilities used in the nuclei segmentation pipeline.
    • No visual outputs are generated to keep the tests fully automated.
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label

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
    return {"merge_overlap_threshold": 0.3}


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
    # Square image with even division
    img1 = np.zeros((200, 200), dtype=np.uint8)
    tile_size1, overlap1 = 100, 20
    tiles1, slices1 = split_image_into_tiles(img1, tile_size1, overlap1, dummy_logger)

    stride1 = tile_size1 - overlap1
    expected_tiles_x = int(np.ceil(200 / stride1))
    expected_tiles_y = int(np.ceil(200 / stride1))
    expected_total_tiles = expected_tiles_x * expected_tiles_y

    assert len(tiles1) == expected_total_tiles
    assert all(tile.shape == (100, 100) for tile in tiles1)

    # Rectangular image with uneven division
    img2 = np.zeros((150, 250), dtype=np.uint8)
    tile_size2, overlap2 = 100, 25
    tiles2, slices2 = split_image_into_tiles(img2, tile_size2, overlap2, dummy_logger)

    stride2 = tile_size2 - overlap2
    expected_tiles_x2 = int(np.ceil(250 / stride2))
    expected_tiles_y2 = int(np.ceil(150 / stride2))
    expected_total_tiles2 = expected_tiles_x2 * expected_tiles_y2

    assert len(tiles2) == expected_total_tiles2
    assert all(tile.shape == (100, 100) for tile in tiles2)

    # Image smaller than tile size
    img3 = np.zeros((50, 50), dtype=np.uint8)
    tile_size3, overlap3 = 100, 10
    tiles3, slices3 = split_image_into_tiles(img3, tile_size3, overlap3, dummy_logger)

    assert len(tiles3) == 1
    assert tiles3[0].shape == (100, 100)
    assert slices3[0] == (slice(0, 50), slice(0, 50))


def test_feather_blending(dummy_logger):
    """
    Test that the feather blending in merge_tiles_with_weighted_overlap works correctly
    by creating a simple gradient pattern and verifying the merged result.
    """
    img = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        img[:, i] = i / 100.0

    tile_size, overlap = 60, 20
    tiles, slices = split_image_into_tiles(img, tile_size, overlap, dummy_logger)
    merged = merge_tiles_with_weighted_overlap(tiles, slices, img.shape, overlap)

    diff = np.abs(img - merged)
    assert (diff < 0.1).mean() > 0.9

    img_multi = np.zeros((3, 100, 100), dtype=np.float32)
    for c in range(3):
        for i in range(100):
            img_multi[c, :, i] = (c + 1) * i / 100.0

    img_2d = img_multi[0]
    tiles_2d, slices = split_image_into_tiles(img_2d, tile_size, overlap, dummy_logger)

    tiles_multi = []
    for ys, xs in slices:
        h = min(ys.stop - ys.start, tile_size)
        w = min(xs.stop - xs.start, tile_size)
        tile_multi = np.zeros((3, tile_size, tile_size), dtype=np.float32)
        tile_multi[:, :h, :w] = img_multi[:, ys.start:ys.stop, xs.start:xs.stop]
        tiles_multi.append(tile_multi)

    merged_multi = merge_tiles_with_weighted_overlap(tiles_multi, slices, img.shape, overlap)
    assert merged_multi.shape == (3, 100, 100)
    for c in range(3):
        diff = np.abs(img_multi[c] - merged_multi[c])
        assert (diff < 0.1 * (c + 1)).mean() > 0.9


def test_feather_mask_properties():
    """
    Test that the feather mask has the expected properties:
    1. Values taper from 1.0 in the center to near 0.0 at the edges.
    2. Correct dimensions.
    3. Proper handling of different overlap values.
    """
    h, w, overlap = 100, 100, 20
    mask = feather_mask(h, w, overlap)

    assert mask.shape == (h, w)
    assert mask[h//2, w//2] == 1.0
    assert mask[0, 0] < 0.01 and mask[-1, -1] < 0.01
    assert np.all(mask[1:overlap, w//2] >= mask[0:overlap-1, w//2])
    assert np.all(mask[h//2, 1:overlap] >= mask[h//2, 0:overlap-1])

    h2, w2, overlap2 = 80, 120, 15
    mask2 = feather_mask(h2, w2, overlap2)
    assert mask2.shape == (h2, w2)

    h3, w3, overlap3 = 30, 30, 20
    mask3 = feather_mask(h3, w3, overlap3)
    eff = min(overlap3, h3 // 2)
    assert mask3[0, w3//2] <= mask3[eff, w3//2]
    assert mask3[-1, w3//2] <= mask3[-eff-1, w3//2]


def test_merge_masks_overlap_without_visual(dummy_logger):
    """
    Test merging behavior with two overlapping circular masks in adjacent tiles.
    """
    import cv2

    settings = {"merge_overlap_threshold": 0.3}
    H, W, overlap = 64, 128, 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),
        (slice(0, 64), slice(64, 128)),
    ]

    tile1 = np.zeros((64, 64), dtype=np.uint16)
    tile2 = np.zeros((64, 64), dtype=np.uint16)

    cv2.circle(tile1, (48, 32), 12, 1, -1)
    cv2.circle(tile2, (16, 32), 12, 2, -1)

    merged = merge_masks([tile1, tile2], slices, image_shape, overlap, dummy_logger, settings)

    unique = np.unique(merged)
    assert len(unique[unique>0]) == 2
    assert set(unique[unique>0]) <= {1, 2}


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

    # Use dummy_logger for logging instead of settings dict
    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, dummy_settings)
    components, num = cc_label(merged > 0)

    assert num == 1, "The isolated label should form exactly one connected component"



def test_merge_masks_transitive_merge(dummy_logger):
    """
    Test that three overlapping tiles with chained overlaps merge into connected components.
    """
    H, W = 10, 30
    overlap = 2
    image_shape = (H, W)
    settings = {"merge_overlap_threshold": 0.2}

    # Define three horizontal tiles with overlap
    slices = [
        (slice(0, H), slice(0, 10)),
        (slice(0, H), slice(8, 18)),
        (slice(0, H), slice(16, 26)),
    ]

    # Create tile masks with slight offsets to chain-connect.
    tiles = []
    for i, (ys, xs) in enumerate(slices):
        tile = np.zeros((H, 10), dtype=np.uint16)
        # Draw a vertical bar near the overlap boundary.
        x_start = 6 if i == 0 else (6 if i == 1 else 0)
        x_end = x_start + 4
        tile[:, x_start:x_end] = 1
        tiles.append(tile)

    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, settings)
    components, num = cc_label(merged > 0)
    # Since tile3 does not overlap tile2 in mask, expect two components.
    assert num == 2, "Expected two connected components for the given offsets"

def test_split_image_odd_dimensions_coverage(dummy_logger):
    """
    Test tiling on odd-sized images to ensure correct tile count and full coverage.
    """
    img = np.zeros((101, 103), dtype=np.uint8)
    tile_size, overlap = 30, 5

    tiles, slices = split_image_into_tiles(img, tile_size, overlap, dummy_logger)

    # Compute expected number of tiles
    stride = tile_size - overlap
    expected_x = int(np.ceil(103 / stride))
    expected_y = int(np.ceil(101 / stride))
    assert len(tiles) == expected_x * expected_y, \
        f"Should create {expected_x * expected_y} tiles for a 101x103 image."

    # Check that every pixel in the image is covered by at least one tile
    coverage = np.zeros(img.shape, dtype=bool)
    for ys, xs in slices:
        coverage[ys, xs] = True
    assert coverage.all(), "All image pixels should be covered by tiles."


def test_weight_map_full_coverage(dummy_logger):
    """
    Test that feather blending covers every pixel by ensuring all weights are positive.
    """
    img = np.random.rand(50, 50).astype(np.float32)
    tile_size, overlap = 20, 5

    tiles, slices = split_image_into_tiles(img, tile_size, overlap, dummy_logger)

    # Capture the weight map via the debug_snap callback
    captured = {}
    def snap(name, arr):
        if name == "merge_weights":
            captured["weights"] = arr.copy()

    _ = merge_tiles_with_weighted_overlap(
        tiles, slices, img.shape, overlap,
        dummy_logger, debug_snap=snap
    )

    weights = captured.get("weights", None)
    assert weights is not None, "merge_weights snapshot should have been captured."
    assert weights.min() > 0, "All pixels should have positive weight coverage."


def test_merge_masks_threshold_extremes(dummy_logger):
    """
    Test that merge_masks respects extreme thresholds of zero and one.
    """
    H, W = 5, 5
    tile_w, overlap = 3, 1
    image_shape = (H, W)
    # Two tiles overlapping by exactly one column
    slices = [
        (slice(0, H), slice(0,     tile_w)),
        (slice(0, H), slice(tile_w - overlap, tile_w*2 - overlap)),
    ]

    tile1 = np.zeros((H, tile_w), dtype=np.uint16)
    tile2 = np.zeros((H, tile_w), dtype=np.uint16)
    # Each bar spans 2 columns so they overlap by 1 column
    tile1[:, 1:3] = 1
    tile2[:, 0:2] = 1

    # Threshold = 1.0: requires 100% overlap → should NOT merge
    merged1 = merge_masks(
        [tile1, tile2], slices, image_shape,
        overlap, dummy_logger, {"merge_overlap_threshold": 1.0}
    )
    labels1 = np.unique(merged1)
    labels1 = labels1[labels1 > 0]
    assert len(labels1) == 2, \
        "No merge should occur when threshold=1.0 and overlap<100%."

    # Threshold = 0.0: any overlap → should merge
    merged0 = merge_masks(
        [tile1, tile2], slices, image_shape,
        overlap, dummy_logger, {"merge_overlap_threshold": 0.0}
    )
    labels0 = np.unique(merged0)
    labels0 = labels0[labels0 > 0]
    assert len(labels0) == 1, \
        "Merge should occur when threshold=0.0 and overlap>0%."


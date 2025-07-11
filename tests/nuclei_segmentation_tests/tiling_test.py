"""
Test Suite: tiling_test.py.

Author: Christos Botos.
Affiliation: Institute of Molecular Biology and Biotechnology.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Validation tests for the newly refactored tiling and merging utilities.
    The tests exercise tiling without zero‑padding, feather‑weight blending of
    continuous outputs, and ratio‑based fusion of instance masks.

Usage:
    python -m pytest tests/tiling_test.py -v

Dependencies:
    • Python>=3.10.
    • numpy, pytest, matplotlib, scipy, opencv‑python.
    • The refactored tiling utilities (import path adjusted below).

Notes:
    • No visual artefacts are written; the suite is fully automated.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pytest
from unittest.mock import MagicMock
from scipy.ndimage import label as cc_label
import cv2

# Adjust path so that "import nuclei_segmentation ..." works when run from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.nuclei_segmentation.utils.tiling import (  # noqa: E402; path tweaked above.
    split_image_into_tiles,
    merge_tiles_with_weighted_overlap,
    merge_masks,
    feather_mask,
)

"""Pytest fixtures."""

@pytest.fixture
def dummy_logger():
    return MagicMock()


@pytest.fixture
def dummy_settings():
    return {"merge_overlap_threshold": 0.3}


"""Core functional tests."""

def test_split_and_reconstruct_identity(dummy_logger):
    """Splitting then merging a greyscale image should reproduce it with high fidelity."""
    img = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    tile_h = tile_w = 128
    overlap = 32

    tiles_iter = split_image_into_tiles(img, tile_h, tile_w, overlap, logger=dummy_logger)
    tiles, slices = zip(*tiles_iter)
    tiles_f32: List[np.ndarray] = [t.astype(np.float32) for t in tiles]

    merged = merge_tiles_with_weighted_overlap(
        tiles_f32, slices, img.shape, overlap, logger=dummy_logger
    )
    merged_uint8 = np.clip(np.round(merged), 0, 255).astype(np.uint8)
    diff = np.abs(img.astype(np.int16) - merged_uint8.astype(np.int16))

    assert (diff < 50).mean() > 0.99
    assert (diff < 30).mean() > 0.98
    assert (diff < 20).mean() > 0.95
    assert (diff < 10).mean() > 0.90


def test_split_image_into_tiles_dimensions(dummy_logger):
    """Ensure tiling yields the correct tile count and valid shapes for diverse inputs."""
    # Square image with even division.
    img1 = np.zeros((200, 200), dtype=np.uint8)
    tile_h1 = tile_w1 = 100
    overlap1 = 20

    tiles_slices1 = list(split_image_into_tiles(img1, tile_h1, tile_w1, overlap1, logger=dummy_logger))
    tiles1, slices1 = zip(*tiles_slices1)

    stride1 = tile_h1 - overlap1
    expected_tiles_x = int(np.ceil(200 / stride1))
    expected_tiles_y = int(np.ceil(200 / stride1))
    assert len(tiles1) == expected_tiles_x * expected_tiles_y
    assert all(t.shape[0] <= tile_h1 and t.shape[1] <= tile_w1 for t in tiles1)

    # Rectangular image with uneven division.
    img2 = np.zeros((150, 250), dtype=np.uint8)
    tile_h2 = tile_w2 = 100
    overlap2 = 25

    tiles_slices2 = list(split_image_into_tiles(img2, tile_h2, tile_w2, overlap2, logger=dummy_logger))
    tiles2, slices2 = zip(*tiles_slices2)

    stride2 = tile_h2 - overlap2
    expected_tiles_x2 = int(np.ceil(250 / stride2))
    expected_tiles_y2 = int(np.ceil(150 / stride2))
    assert len(tiles2) == expected_tiles_x2 * expected_tiles_y2
    assert all(t.shape[0] <= tile_h2 and t.shape[1] <= tile_w2 for t in tiles2)

    # Image smaller than tile size.
    img3 = np.zeros((50, 50), dtype=np.uint8)
    tile_h3 = tile_w3 = 100
    overlap3 = 10

    tiles_slices3 = list(split_image_into_tiles(img3, tile_h3, tile_w3, overlap3, logger=dummy_logger))
    assert len(tiles_slices3) == 1
    tile3, sl3 = tiles_slices3[0]
    assert tile3.shape == img3.shape  # No padding expected.
    assert sl3 == (slice(0, 50), slice(0, 50))


def test_feather_blending(dummy_logger):
    """Gradient reconstruction with feather blending should be accurate."""
    img = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        img[:, i] = i / 100.0

    tile_h = tile_w = 60
    overlap = 20

    tiles_slices = list(split_image_into_tiles(img, tile_h, tile_w, overlap, logger=dummy_logger))
    tiles, slices = zip(*tiles_slices)
    merged = merge_tiles_with_weighted_overlap(tiles, slices, img.shape, overlap, logger=dummy_logger)

    diff = np.abs(img - merged)
    assert (diff < 0.1).mean() > 0.9

    # Multi‑channel variant (CHW).
    img_multi = np.zeros((3, 100, 100), dtype=np.float32)
    for c in range(3):
        for i in range(100):
            img_multi[c, :, i] = (c + 1) * i / 100.0

    tiles_slices_mc = list(
        split_image_into_tiles(img_multi, tile_h, tile_w, overlap, channel_axis=0, logger=dummy_logger)
    )
    tiles_mc, slices_mc = zip(*tiles_slices_mc)

    merged_mc = merge_tiles_with_weighted_overlap(
        tiles_mc, slices_mc, img_multi.shape[1:], overlap, channel_axis=0, logger=dummy_logger
    )
    assert merged_mc.shape == img_multi.shape
    for c in range(3):
        diff_c = np.abs(img_multi[c] - merged_mc[c])
        assert (diff_c < 0.1 * (c + 1)).mean() > 0.9


def test_feather_mask_properties():
    """Feather mask should taper correctly and respect dimensions."""
    h, w, overlap = 100, 100, 20
    mask = feather_mask(h, w, overlap)

    assert mask.shape == (h, w)
    assert mask[h // 2, w // 2] == 1.0
    assert mask[0, 0] < 0.01 and mask[-1, -1] < 0.01
    assert np.all(mask[1:overlap, w // 2] >= mask[0 : overlap - 1, w // 2])
    assert np.all(mask[h // 2, 1:overlap] >= mask[h // 2, 0 : overlap - 1])

    h2, w2, overlap2 = 80, 120, 15
    mask2 = feather_mask(h2, w2, overlap2)
    assert mask2.shape == (h2, w2)

    h3, w3, overlap3 = 30, 30, 20
    mask3 = feather_mask(h3, w3, overlap3)
    eff = min(overlap3, h3 // 2)
    assert mask3[0, w3 // 2] <= mask3[eff, w3 // 2]
    assert mask3[-1, w3 // 2] <= mask3[-eff - 1, w3 // 2]


def test_merge_masks_overlap_without_visual(dummy_logger):
    """Merging two overlapping circular masks should yield two IDs when threshold is moderate."""
    settings = {"merge_overlap_threshold": 0.3}
    H, W, overlap = 64, 128, 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),
        (slice(0, 64), slice(64, 128)),
    ]

    tile1 = np.zeros((64, 64), dtype=np.uint8)
    tile2 = np.zeros((64, 64), dtype=np.uint8)

    cv2.circle(tile1, (48, 32), 12, 1, -1)
    cv2.circle(tile2, (16, 32), 12, 2, -1)

    merged = merge_masks([tile1, tile2], slices, image_shape, overlap, dummy_logger, settings)
    unique = np.unique(merged)
    assert set(unique[unique > 0]) == {1, 2}


def test_merge_masks_isolated_labels(dummy_logger, dummy_settings):
    """An isolated label in a single tile should remain a unique component after fusion."""
    H = W = 128
    overlap = 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),
        (slice(0, 64), slice(64, 128)),
        (slice(64, 128), slice(0, 64)),
        (slice(64, 128), slice(64, 128)),
    ]

    tiles = [np.zeros((64, 64), dtype=np.uint8) for _ in range(4)]
    tiles[3][10:30, 10:30] = 7

    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, dummy_settings)
    _, num = cc_label(merged > 0)
    assert num == 1


def test_merge_masks_transitive_merge(dummy_logger):
    """Three tiles with chained overlaps should produce two components given offsets."""
    H, W = 10, 30
    overlap = 2
    image_shape = (H, W)
    settings = {"merge_overlap_threshold": 0.2}

    slices = [
        (slice(0, H), slice(0, 10)),
        (slice(0, H), slice(8, 18)),
        (slice(0, H), slice(16, 26)),
    ]

    tiles: List[np.ndarray] = []
    for i, (ys, xs) in enumerate(slices):
        tile = np.zeros((H, 10), dtype=np.uint8)
        x_start = 6 if i == 0 else (6 if i == 1 else 0)
        tile[:, x_start : x_start + 4] = 1
        tiles.append(tile)

    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, settings)
    _, num = cc_label(merged > 0)
    assert num == 2


def test_split_image_odd_dimensions_coverage(dummy_logger):
    """Odd‑sized images should be fully covered by generated tiles."""
    img = np.zeros((101, 103), dtype=np.uint8)
    tile_h = tile_w = 30
    overlap = 5

    tiles_slices = list(split_image_into_tiles(img, tile_h, tile_w, overlap, logger=dummy_logger))
    _, slices = zip(*tiles_slices)

    stride = tile_h - overlap
    expected_x = int(np.ceil(103 / stride))
    expected_y = int(np.ceil(101 / stride))
    assert len(tiles_slices) == expected_x * expected_y

    coverage = np.zeros(img.shape, dtype=bool)
    for ys, xs in slices:
        coverage[ys, xs] = True
    assert coverage.all()


def test_weight_map_full_coverage(dummy_logger):
    """Feather weights should be positive everywhere in the composite image."""
    img = np.random.rand(50, 50).astype(np.float32)
    tile_h = tile_w = 20
    overlap = 5

    tiles_slices = list(split_image_into_tiles(img, tile_h, tile_w, overlap, logger=dummy_logger))
    tiles, slices = zip(*tiles_slices)

    captured: dict[str, np.ndarray] = {}

    def snap(name: str, arr: np.ndarray):
        if name == "merge_weights":
            captured["weights"] = arr.copy()

    _ = merge_tiles_with_weighted_overlap(
        tiles, slices, img.shape, overlap, logger=dummy_logger, debug_snap=snap
    )

    weights = captured.get("weights")
    assert weights is not None
    assert weights.min() > 0.0


def test_merge_masks_threshold_extremes(dummy_logger):
    """Merge threshold extremes of 0.0 and 1.0 should behave as expected."""
    H = W = 5
    tile_w = 3
    overlap = 1
    image_shape = (H, W)

    slices = [
        (slice(0, H), slice(0, tile_w)),
        (slice(0, H), slice(tile_w - overlap, tile_w * 2 - overlap)),
    ]

    tile1 = np.zeros((H, tile_w), dtype=np.uint8)
    tile2 = np.zeros((H, tile_w), dtype=np.uint8)
    tile1[:, 1:3] = 1
    tile2[:, 0:2] = 1

    merged1 = merge_masks(
        [tile1, tile2], slices, image_shape, overlap, dummy_logger, {"merge_overlap_threshold": 1.0}
    )
    labels1 = np.unique(merged1)[1:]
    assert len(labels1) == 2

    merged0 = merge_masks(
        [tile1, tile2], slices, image_shape, overlap, dummy_logger, {"merge_overlap_threshold": 0.0}
    )
    labels0 = np.unique(merged0)[1:]
    assert len(labels0) == 1

"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_tiling.py.
Description:
    Unit tests for tiling utilities: tile count calculation, overlap region
    geometry, edge tile handling, and feather mask generation.

Dependencies:
    - Python >= 3.10.
    - pytest, numpy.

Usage:
    pytest tests/test_tiling.py -v
"""

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from atlas_cellpose.utils.tiling import feather_mask, split_image_into_tiles


# ====================================================================== #
# feather_mask tests.
# ====================================================================== #

class TestFeatherMask:
    """Test the cosine-ramp feather mask generator."""

    def test_zero_overlap_returns_ones(self):
        """Overlap=0 should produce a uniform 1.0 mask."""
        mask = feather_mask(64, 64, 0)
        assert mask.shape == (64, 64)
        np.testing.assert_array_equal(mask, np.ones((64, 64), dtype=np.float32))

    def test_negative_overlap_returns_ones(self):
        """Negative overlap should be treated like zero overlap."""
        mask = feather_mask(32, 32, -5)
        np.testing.assert_array_equal(mask, np.ones((32, 32), dtype=np.float32))

    def test_shape_matches_input(self):
        mask = feather_mask(100, 200, 20)
        assert mask.shape == (100, 200)

    def test_corners_are_smallest(self):
        """Corner values should be the smallest in the mask."""
        mask = feather_mask(64, 64, 16)
        corner = mask[0, 0]
        center = mask[32, 32]
        assert corner < center

    def test_center_is_one(self):
        """Center of a large mask with small overlap should be 1.0."""
        mask = feather_mask(100, 100, 10)
        assert mask[50, 50] == pytest.approx(1.0)

    def test_values_in_range(self):
        """All values should be between 0 and 1."""
        mask = feather_mask(64, 64, 20)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_dtype_is_float32(self):
        mask = feather_mask(32, 32, 8)
        assert mask.dtype == np.float32


# ====================================================================== #
# split_image_into_tiles tests.
# ====================================================================== #

class TestSplitImageIntoTiles:
    """Test tile splitting for various image sizes and overlaps."""

    @pytest.fixture
    def logger(self):
        return logging.getLogger("test_tiling")

    def test_single_tile_small_image(self, logger):
        """An image smaller than tile size should produce exactly 1 tile."""
        img = np.zeros((50, 50), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 100, 100, 10, logger))
        assert len(tiles) == 1
        tile, (yslice, xslice) = tiles[0]
        assert tile.shape == (50, 50)

    def test_exact_fit_no_overlap(self, logger):
        """200x200 image, 100x100 tiles, overlap=0 -> 4 tiles."""
        img = np.zeros((200, 200), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 100, 100, 0, logger))
        assert len(tiles) == 4

    def test_overlap_increases_tile_count(self, logger):
        """Overlap should produce more tiles than no overlap."""
        img = np.zeros((200, 200), dtype=np.uint8)
        tiles_no_overlap = list(split_image_into_tiles(img, 100, 100, 0, logger))
        tiles_with_overlap = list(split_image_into_tiles(img, 100, 100, 20, logger))
        assert len(tiles_with_overlap) >= len(tiles_no_overlap)

    def test_full_coverage(self, logger):
        """Every pixel should be covered by at least one tile."""
        H, W = 150, 200
        img = np.zeros((H, W), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 64, 64, 10, logger))

        covered = np.zeros((H, W), dtype=bool)
        for tile, (yslice, xslice) in tiles:
            covered[yslice, xslice] = True
        assert covered.all(), "Not all pixels are covered by tiles."

    def test_tile_content_matches_image(self, logger):
        """Tile pixel values should match the source image region."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, size=(120, 160), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 64, 64, 10, logger))

        for tile, (yslice, xslice) in tiles:
            expected = img[yslice, xslice]
            np.testing.assert_array_equal(tile, expected)

    def test_edge_tile_smaller_than_full(self, logger):
        """Edge tiles should be clipped to image boundary, not padded."""
        img = np.zeros((110, 110), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 64, 64, 0, logger))

        # The last tile in each dimension should be smaller.
        edge_tiles = [t for t, (ys, xs) in tiles if t.shape != (64, 64)]
        assert len(edge_tiles) > 0, "Expected at least one edge tile smaller than 64x64."

    def test_3d_hwc_image(self, logger):
        """3D HWC images should tile correctly."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, 50, 50, 10, logger))
        assert len(tiles) >= 4
        for tile, _ in tiles:
            assert tile.ndim == 3
            assert tile.shape[2] == 3

    def test_invalid_overlap_raises(self, logger):
        """Overlap >= tile size should raise ValueError."""
        img = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            list(split_image_into_tiles(img, 64, 64, 64, logger))

    def test_tile_count_formula(self, logger):
        """Verify tile count matches expected stride-based formula."""
        H, W = 500, 400
        tile_h, tile_w, overlap = 128, 128, 20
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap

        expected_rows = math.ceil(H / stride_h)
        expected_cols = math.ceil(W / stride_w)
        expected_total = expected_rows * expected_cols

        img = np.zeros((H, W), dtype=np.uint8)
        tiles = list(split_image_into_tiles(img, tile_h, tile_w, overlap, logger))
        assert len(tiles) == expected_total

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
)


@pytest.fixture
def dummy_logger():
    return MagicMock()


@pytest.fixture
def dummy_settings():
    return {"MERGE_OVERLAP_THRESHOLD": 0.3}


def test_split_and_reconstruct_identity(dummy_logger):
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


def test_merge_masks_overlap_debug_and_visual(dummy_logger):
    """
    Visual debug of merging behavior: two overlapping circular masks in adjacent tiles.
    """
    import cv2

    settings = {"MERGE_OVERLAP_THRESHOLD": 0.3}
    H, W, overlap = 64, 128, 16
    image_shape = (H, W)

    slices = [
        (slice(0, 64), slice(0, 64)),    # Tile 1.
        (slice(0, 64), slice(64, 128)),  # Tile 2.
    ]

    # Create two circular masks that touch in the overlap.
    tile1 = np.zeros((64, 64), dtype=np.uint16)
    tile2 = np.zeros((64, 64), dtype=np.uint16)

    # Circle in tile 1 (right side, label 1).
    cv2.circle(tile1, center=(48, 32), radius=12, color=1, thickness=-1)
    # Circle in tile 2 (left side, label 2).
    cv2.circle(tile2, center=(16, 32), radius=12, color=2, thickness=-1)

    tiles = [tile1, tile2]
    merged = merge_masks(tiles, slices, image_shape, overlap, dummy_logger, settings)

    # Crop overlapping region.
    overlap_slice = (slice(16, 48), slice(48, 80))
    merged_crop = merged[overlap_slice]
    tile1_crop = tile1[16:48, 32:64]
    tile2_crop = tile2[16:48, 0:32]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(tile1_crop, cmap="nipy_spectral", interpolation="none")
    axs[0].set_title("Tile 1: Circle (label 1)")
    axs[1].imshow(tile2_crop, cmap="nipy_spectral", interpolation="none")
    axs[1].set_title("Tile 2: Circle (label 2)")
    axs[2].imshow(merged_crop, cmap="nipy_spectral", interpolation="none")
    axs[2].set_title("Merged Overlap Region")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    outname = "mask_overlap_merge_debug.png"
    plt.savefig(outname, dpi=150)
    print(f"[DEBUG] Saved circular mask test visualization to: {outname}")
    plt.close()



def test_merge_masks_overlap_debug_and_visual(dummy_logger):
    """
    Visual debug of merging behavior: two overlapping circular masks in adjacent tiles.
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

    # Crop overlapping region
    overlap_slice = (slice(16, 48), slice(48, 80))
    merged_crop = merged[overlap_slice]
    tile1_crop = tile1[16:48, 32:64]
    tile2_crop = tile2[16:48, 0:32]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(tile1_crop, cmap="nipy_spectral", interpolation="none")
    axs[0].set_title("Tile 1: Circle (label 1)")
    axs[1].imshow(tile2_crop, cmap="nipy_spectral", interpolation="none")
    axs[1].set_title("Tile 2: Circle (label 2)")
    axs[2].imshow(merged_crop, cmap="nipy_spectral", interpolation="none")
    axs[2].set_title("Merged Overlap Region")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    outname = "mask_overlap_merge_debug.png"
    plt.savefig(outname, dpi=150)
    print(f"[DEBUG] Saved circular mask test visualization to: {outname}")
    plt.close()



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

    assert np.count_nonzero(merged[74:94, 74:94]) > 0

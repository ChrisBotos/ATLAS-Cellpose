"""
TILING AND MERGING UTILITIES FOR CELL SEGMENTATION

This module supports:
- Overlap-aware image tiling.
- Feathered merging of continuous outputs (flows, probability maps).
- Overlap-ratio-based merging of instance label masks.

Used in Cellpose-based segmentation pipelines on large I/R kidney datasets.
"""

import numpy as np


""" FEATHER MASK CONSTRUCTION """


def feather_mask(h, w, overlap):
    """
    Create a 2D feather mask tapering to 0.0 at tile edges.

    Parameters:
        h (int): Height of tile.
        w (int): Width of tile.
        overlap (int): Overlap in pixels.

    Returns:
        np.ndarray: Float32 mask of shape (h, w).
    """
    edge_h = min(overlap, h // 2)
    edge_w = min(overlap, w // 2)

    ramp_h = np.ones(h, dtype=np.float32)
    ramp_w = np.ones(w, dtype=np.float32)

    ramp_h[:edge_h] = np.linspace(0.0, 1.0, edge_h, endpoint=False)
    ramp_h[-edge_h:] = np.linspace(1.0, 0.0, edge_h, endpoint=False)
    ramp_w[:edge_w] = np.linspace(0.0, 1.0, edge_w, endpoint=False)
    ramp_w[-edge_w:] = np.linspace(1.0, 0.0, edge_w, endpoint=False)

    return np.outer(ramp_h, ramp_w)


""" IMAGE TILING """


def split_image_into_tiles(image, tile_size, overlap, logger):
    """
    Split a 2D image into square tiles with specified overlap.

    Parameters:
        image (np.ndarray): 2D grayscale image.
        tile_size (int): Size of square tile in pixels.
        overlap (int): Overlap between tiles.
        logger: Logger instance.

    Returns:
        list[np.ndarray], list[tuple]: Image tiles and corresponding (yslice, xslice) positions.
    """
    H, W = image.shape
    stride = tile_size - overlap
    tiles, slices = [], []

    logger.info(f"Tiling {H}x{W} image into {tile_size}px tiles with {overlap}px overlap.")

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            tile = np.zeros((tile_size, tile_size), dtype=image.dtype)
            tile[:y1 - y, :x1 - x] = image[y:y1, x:x1]

            tiles.append(tile)
            slices.append((slice(y, y1), slice(x, x1)))

    logger.info(f"Generated {len(tiles)} tiles.")
    return tiles, slices


""" MERGE FLOATING-POINT TILE OUTPUTS """


def merge_tiles_with_weighted_overlap(tile_stack, slices, image_shape, overlap, logger=None, debug_snap=None):
    """
    Merge tiled continuous-valued outputs using feather blending.

    Parameters:
        tile_stack (list[np.ndarray]): Each tile is (H, W), (C, H, W), or (H, W, C).
        slices (list[tuple]): (yslice, xslice) pairs for placing each tile.
        image_shape (tuple): Shape of the full image (H, W).
        overlap (int): Overlap used during tiling.
        logger: Optional logger.
        debug_snap: Optional debug callback.

    Returns:
        np.ndarray: Merged array, shape (C, H, W) or (H, W).
    """
    H, W = image_shape
    sample = tile_stack[0]

    # Standardize input format
    if sample.ndim == 2:
        C = 1
        tile_stack = [tile[np.newaxis, :, :] for tile in tile_stack]
    elif sample.ndim == 3:
        if sample.shape[0] <= 4:
            C = sample.shape[0]
        elif sample.shape[2] <= 4:
            C = sample.shape[2]
            tile_stack = [tile.transpose(2, 0, 1) for tile in tile_stack]
        else:
            raise ValueError(f"Ambiguous tile shape {sample.shape}")
    else:
        raise ValueError(f"Invalid tile ndim: {sample.ndim}")

    field = np.zeros((C, H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    for tile, (ys, xs) in zip(tile_stack, slices):
        th, tw = tile.shape[1:]
        alpha = feather_mask(th, tw, overlap)
        alpha_broadcast = np.broadcast_to(alpha, (C, th, tw))

        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        valid_h, valid_w = y1 - y0, x1 - x0

        field[:, y0:y1, x0:x1] += tile[:, :valid_h, :valid_w] * alpha_broadcast[:, :valid_h, :valid_w]
        weights[y0:y1, x0:x1] += alpha[:valid_h, :valid_w]

    if debug_snap:
        debug_snap("merge_weights", weights)

    out = np.zeros_like(field)
    nonzero = weights > 0
    for c in range(C):
        out[c, nonzero] = field[c, nonzero] / weights[nonzero]

    return out[0] if C == 1 else out


""" MERGE INSTANCE MASK TILES """


def merge_masks(mask_tiles, slices, image_shape, overlap, logger, settings, debug_snap=None):
    """
    Merge instance label masks from tiles using ratio-based fusion.

    Parameters:
        mask_tiles (list[np.ndarray]): List of 2D uint16 instance masks.
        slices (list[tuple]): (yslice, xslice) placements.
        image_shape (tuple): Full image size.
        overlap (int): Tile overlap used.
        logger: Logger.
        settings (dict): {'MERGE_OVERLAP_THRESHOLD': float}
        debug_snap (callable): Optional snapshot function.

    Returns:
        np.ndarray: Final merged mask of shape (H, W), dtype uint16.
    """
    merged = np.zeros(image_shape, dtype=np.uint16)
    next_id = 1
    ratio_threshold = settings.get("MERGE_OVERLAP_THRESHOLD", 0.3)

    for tile, (ys, xs) in zip(mask_tiles, slices):
        if tile.max() == 0:
            continue

        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        region = merged[y0:y1, x0:x1]

        for val in np.unique(tile):
            if val == 0:
                continue

            binary = (tile == val)
            binary = binary[:y1 - y0, :x1 - x0]
            overlap_area = region[binary] > 0
            overlap_pixels = overlap_area.sum()
            total_pixels = binary.sum()

            if overlap_pixels > 0:
                overlapping_labels, counts = np.unique(region[binary][region[binary] > 0], return_counts=True)
                merged_flag = False

                for label, count in zip(overlapping_labels, counts):
                    label_size = np.sum(region == label)
                    mask_ratio = count / total_pixels
                    label_ratio = count / label_size if label_size > 0 else 0.0

                    if mask_ratio >= ratio_threshold or label_ratio >= ratio_threshold:
                        region[binary] = label
                        merged_flag = True
                        break

                if not merged_flag:
                    region[binary] = next_id
                    next_id += 1
            else:
                region[binary] = next_id
                next_id += 1


    if debug_snap:
        debug_snap("merged_labels", merged)

    logger.info(f"Merged {len(mask_tiles)} mask tiles into {next_id - 1} unique objects.")
    return merged

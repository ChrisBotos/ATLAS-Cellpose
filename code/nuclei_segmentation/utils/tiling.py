"""
Image Tiling and Merging Utilities for Large-Scale Kidney Tissue Analysis.

This module provides specialized functions for handling large microscopy images
through tiling and merging approaches. It enables processing of gigapixel-scale
kidney tissue images that would otherwise exceed memory limits.

The module includes:
1. Functions to split large images into overlapping tiles
2. Algorithms to merge segmentation results from individual tiles
3. Specialized weighted merging for flow fields and probability maps
4. Intelligent handling of object boundaries at tile edges

These utilities are critical for analyzing whole kidney sections after I/R injury,
where the entire tissue context is important for understanding spatial patterns
of damage and repair.
"""

# Standard library imports.
import logging
import numpy as np


def split_image_into_tiles(image, tile_size, overlap, logger):
    """
    Split a large image into overlapping tiles for processing.

    Args:
        image: Input image as numpy array
        tile_size: Size of each tile (square)
        overlap: Fraction of overlap between tiles (0-1)
        logger: Logger object

    Returns:
        tiles: List of image tiles
        slices: List of slice tuples for reconstructing the full image
    """
    h, w = image.shape
    logger.info(f"Splitting {h}×{w} image into tiles of size {tile_size}×{tile_size} with {overlap*100:.1f}% overlap")

    # If the image is smaller than the tile size, just use the whole image.
    if h <= tile_size and w <= tile_size:
        logger.info("Image is smaller than tile size, using entire image as a single tile.")
        return [image], [(slice(0, h), slice(0, w))]

    # Calculate effective step size based on overlap.
    step = int(tile_size * (1 - overlap))
    if step <= 0:
        logger.warning(f"Overlap too high ({overlap}), reducing to 0.8.")
        overlap = 0.8
        step = int(tile_size * (1 - overlap))

    # Calculate number of tiles in each dimension.
    n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
    n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)

    # If we'd create too many tiles, adjust the tile size or overlap.
    max_tiles = 100  # Arbitrary limit to prevent excessive memory usage.
    if n_h * n_w > max_tiles:
        logger.warning(f"Too many tiles ({n_h}×{n_w}={n_h*n_w}), adjusting parameters.")

        # Try increasing step size (reducing overlap).
        if overlap > 0.1:
            overlap = 0.1
            step = int(tile_size * (1 - overlap))
            n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
            n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)

            if n_h * n_w <= max_tiles:
                logger.info(f"Reduced overlap to {overlap:.1f}, new tile count: {n_h}×{n_w}={n_h*n_w}")
            else:
                # If still too many, increase tile size.
                orig_tile_size = tile_size
                tile_size = min(h, w, tile_size * 2)
                step = int(tile_size * (1 - overlap))
                n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
                n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)
                logger.info(f"Increased tile size from {orig_tile_size} to {tile_size}, new tile count: {n_h}×{n_w}={n_h*n_w}")

    logger.info(f"Creating {n_h}×{n_w}={n_h*n_w} tiles")

    tiles = []
    slices = []

    for i in range(n_h):
        for j in range(n_w):
            # Calculate tile boundaries.
            y_start = min(i * step, h - tile_size) if h > tile_size else 0
            x_start = min(j * step, w - tile_size) if w > tile_size else 0
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Extract tile.
            tile = image[y_start:y_end, x_start:x_end]

            # Handle tiles smaller than tile_size (at edges).
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                # Create a new tile of the correct size.
                new_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                new_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = new_tile

            tiles.append(tile)
            slices.append((slice(y_start, y_end), slice(x_start, x_end)))

    logger.info(f"Created {len(tiles)} tiles")
    return tiles, slices


def merge_tiles_with_weighted_overlap(
        tile_stack:      list[np.ndarray],
        slices:          list[tuple[slice, slice]],
        image_shape:     tuple[int, int],
        overlap:         float,
        logger:          logging.Logger | None = None,
        dtype:           np.float32 = np.float32
    ) -> np.ndarray:
    """
    Merge a list of overlapping flow- or probability-tiles back into one seamless field.

    This function is critical for reconstructing tiled segmentation results without visible
    seams or discontinuities. It uses a weighted blending approach where each tile's contribution
    tapers linearly to zero at its edges, creating smooth transitions in the overlap regions.
    This is particularly important for flow fields and probability maps where discontinuities
    would create artifacts in the final segmentation.

    Parameters:
        tile_stack : list[np.ndarray]
            Each element is either (2, H, W), (H, W, 2) or (H, W).
        slices : list[tuple[slice, slice]]
            The (row_slice, col_slice) that positions each tile on the canvas.
        image_shape : tuple[int, int]
            (H, W) of the original image.
        overlap : float
            Fractional overlap that was used when tiling (0–1).
        logger : Optional logger for DEBUG / INFO prints.
        dtype : Data type of the returned array (default float32).

    Returns:
        np.ndarray: (2, H, W) for vector fields or (H, W) for single-channel maps.
    """

    # Sanity checks to ensure inputs are valid.
    # Instead of assertion, print diagnostic information for better debugging.
    if len(tile_stack) != len(slices):
        raise ValueError(f"Mismatch: {len(tile_stack)} tiles vs {len(slices)} slices.")

    H, W = image_shape
    flow_accum   = None                              # Deferred allocation.
    weight_accum = np.zeros((H, W), dtype=dtype)

    # Helper function to create a 2D feather mask for smooth blending.
    def _feather_mask(h: int, w: int, ov: float) -> np.ndarray:
        """
        Build a (h × w) mask that is 1.0 in the tile center and decays
        linearly to 0.0 at each border across an edge band of width
        edge = ov * size / 2. This creates a smooth transition between tiles.

        This approach is similar to feathering techniques used in image stitching
        and panorama creation, ensuring that the transition between tiles is
        imperceptible in the final result.
        """
        edge_h = max(1, int(ov * h / 2))
        edge_w = max(1, int(ov * w / 2))

        ramp_h = np.ones(h, dtype=dtype)
        ramp_w = np.ones(w, dtype=dtype)

        ramp_h[:edge_h]  = np.linspace(0.0, 1.0, edge_h,  endpoint=False)
        ramp_h[-edge_h:] = np.linspace(1.0, 0.0, edge_h,  endpoint=False)[::-1]

        ramp_w[:edge_w]  = np.linspace(0.0, 1.0, edge_w,  endpoint=False)
        ramp_w[-edge_w:] = np.linspace(1.0, 0.0, edge_w,  endpoint=False)[::-1]

        return np.outer(ramp_h, ramp_w)

    # Main accumulation loop to blend all tiles with their weights.
    for idx, (tile, slc) in enumerate(zip(tile_stack, slices), start=1):

        if logger:
            logger.debug(f"merge_tiles_with_weighted_overlap • tile {idx}/{len(tile_stack)}.")

        # Standardize tile format to (C, h, w) for consistent processing.
        if tile.ndim == 2:                          # (h, w)  → single-channel
            tile = tile[np.newaxis, ...]
        elif tile.ndim == 3:
            if tile.shape[0] <= 4:                  # (C, h, w) – channels FIRST
                pass
            else:                                   # (h, w, C) – channels LAST
                tile = tile.transpose(2, 0, 1)
        else:
            raise ValueError(f"Tile #{idx} has unsupported shape {tile.shape}.")


        tile = tile.astype(dtype, copy=False)
        C, th, tw = tile.shape

        if flow_accum is None:
            flow_accum = np.zeros((C, H, W), dtype=dtype)

        alpha = _feather_mask(th, tw, overlap)
        alpha_broadcast = np.broadcast_to(alpha, (C, th, tw))

        rs, cs = slc
        flow_accum[:, rs, cs] += tile * alpha_broadcast
        weight_accum[rs, cs]  += alpha

    # Normalize the accumulated values by the weights to get the final blended result.
    nz = weight_accum > 0.0
    output = np.zeros_like(flow_accum, dtype=dtype)
    output[:, nz] = flow_accum[:, nz] / weight_accum[nz]

    if logger:
        logger.info(f"Merged {len(tile_stack)} tiles → {output.shape[0]}-channel field "
                    f"(overlap={overlap:.2f}).")

    # Return 2-D array for single-channel maps.
    return output[0] if output.shape[0] == 1 else output


def merge_masks(tiles, slices, image_shape, overlap, logger, settings):
    """
    Efficiently stitch Cellpose-generated tiled masks into a single label image.

    This function handles the challenging task of merging segmentation masks from multiple
    tiles while preserving object identity. Unlike flow fields which can be blended,
    segmentation masks contain discrete object IDs that must be carefully reconciled
    at tile boundaries to avoid duplicate or fragmented nuclei.

    The algorithm uses an overlap-based approach to determine when objects spanning
    multiple tiles should be merged or kept separate. This is critical for accurate
    quantification of nuclear features in densely packed kidney tissue regions.

    Args:
        tiles: List of segmentation mask tiles.
        slices: List of slice tuples for positioning each tile.
        image_shape: Shape of the output image (height, width).
        overlap: Fractional overlap between tiles.
        logger: Logger for progress information.
        settings: Dictionary containing configuration parameters including MERGE_OVERLAP_THRESHOLD.

    Returns:
        numpy.ndarray: Merged segmentation mask with consistent object IDs.
    """
    # Initialize the output mask.
    merged_mask = np.zeros(image_shape, dtype=np.uint16)
    next_label = 1

    logger.info(f"Merging {len(tiles)} mask tiles with overlap={overlap:.2f}")

    # Simple approach: directly copy tiles to the output mask, handling overlaps.
    # This avoids complex relabeling that might cause issues.
    for i, (tile, slc) in enumerate(zip(tiles, slices)):
        if i % 10 == 0:
            logger.info(f"Processing tile {i+1}/{len(tiles)}")

        # Skip empty tiles.
        if np.max(tile) == 0:
            continue

        # Get the region in the merged mask where this tile will go.
        mask_region = merged_mask[slc]

        # For each object in this tile, process it individually.
        for label in np.unique(tile)[1:]:  # Skip background (0).
            # Create a binary mask for this object to isolate it.
            obj_mask = (tile == label)

            # Check if this object overlaps with existing objects in the merged mask.
            overlap_mask = (mask_region > 0) & obj_mask

            if np.sum(overlap_mask) == 0:
                # No overlap, assign a new label to this object.
                mask_region[obj_mask] = next_label
                next_label += 1
            else:
                # There's overlap - check how much of the object overlaps with existing objects.
                overlap_ratio = np.sum(overlap_mask) / np.sum(obj_mask)

                # Use the user-defined threshold from settings for determining when to merge objects.
                if overlap_ratio < settings["MERGE_OVERLAP_THRESHOLD"]:  # Less than threshold overlap, treat as new object.
                    mask_region[obj_mask & ~overlap_mask] = next_label
                    next_label += 1
                else:
                    # Significant overlap - find the most overlapping existing label.
                    existing_labels = mask_region[overlap_mask]
                    unique_labels, counts = np.unique(existing_labels, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]

                    # Extend the existing object by assigning the same label.
                    mask_region[obj_mask & ~overlap_mask] = most_common_label

    # Count final objects.
    unique_labels = np.unique(merged_mask)
    num_objects = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
    logger.info(f"Merged {len(tiles)} tiles → {num_objects} unique objects")

    return merged_mask

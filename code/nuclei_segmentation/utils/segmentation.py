"""
Nuclei Segmentation Utilities for Kidney I/R Injury Analysis.

This module provides specialized functions for segmenting nuclei in kidney tissue
images after ischemia-reperfusion injury. It includes:

1. Functions for running Cellpose segmentation on whole images or tiles
2. Edge detection refinement for improving segmentation boundaries
3. Utilities for handling large images through tiling and merging

These functions are optimized for the challenges of kidney tissue analysis,
where nuclear morphology and density can vary significantly between healthy
and injured regions.
"""

# Standard library imports.
import numpy as np
import cv2
import torch
import logging

# Import from other utility modules.
from code.nuclei_segmentation.utils.tiling import split_image_into_tiles, merge_masks, merge_tiles_with_weighted_overlap


def run_cellpose_on_tiles(model, image, cellpose_params, settings, logger):
    """
    Run Cellpose segmentation on an image, with optional tiling for large images.

    This function handles the core segmentation process, using either a direct Cellpose
    call for smaller images or a tiled approach for large images that would exceed GPU
    memory. The tiled approach segments each tile independently and then carefully
    merges the results to create a seamless final segmentation.

    For kidney tissue analysis, tiling is essential as whole-slide images often exceed
    several gigabytes in size, far beyond what can be processed in a single GPU pass.

    Args:
        model: Initialized Cellpose model.
        image: Input image as numpy array.
        cellpose_params: Dictionary of Cellpose parameters.
        settings: Dictionary of pipeline settings.
        logger: Logger for progress information.

    Returns:
        tuple: (masks, flows, num_cells) - Segmentation masks, flow fields, and cell count.
    """
    h, w = image.shape
    tile_side_length = settings["tile_side_length"]
    overlap = settings["TILE_OVERLAP"]
    use_tiling = settings["USE_TILING"] and (tile_side_length < h or tile_side_length < w)

    # ──────────────────────────────────────────────────────────
    # 1) Process without tiling - direct Cellpose call for smaller images
    # ──────────────────────────────────────────────────────────
    if not use_tiling:
        logger.info("Tiling disabled – processing full image.")

        # Add debug info about the image for troubleshooting.
        logger.info(f"Image shape: {image.shape}, min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}.")

        try:
            masks, flows, *_ = model.eval(
                image[..., None],                          # add channel axis
                diameter          = cellpose_params["diameter"],
                channels          = cellpose_params["channels"],
                flow_threshold    = cellpose_params["flow_threshold"],
                cellprob_threshold= cellpose_params["cellprob_threshold"],
                resample          = cellpose_params["resample"],
                augment           = False,
                batch_size        = cellpose_params["batch_size"],
                do_3D             = False
            )

            # Count cells and log information.
            num_cells = len(np.unique(masks)) - 1 if 0 in np.unique(masks) else len(np.unique(masks))
            logger.info(f"Detected {num_cells} cells in full image")

            return masks, flows, num_cells                # ← flows already a list

        except Exception as e:
            logger.error(f"Error processing full image: {e}")
            # Return empty masks and flows.
            empty_masks = np.zeros_like(image, dtype=np.uint16)
            empty_flows = [np.zeros((2, *image.shape), dtype=np.float32),
                          None,
                          np.zeros_like(image, dtype=np.float32)]
            return empty_masks, empty_flows, 0

    # ──────────────────────────────────────────────────────────
    # 2) Process with tiling - segment tiles independently then merge results
    # ──────────────────────────────────────────────────────────
    tiles, slices = split_image_into_tiles(image, tile_side_length, overlap, logger)
    logger.info(f"Processing {len(tiles)} tiles.")

    # Storage for tile results.
    mask_tiles        = []
    flow_xy_tiles     = []   # flows[0]  (2-ch)
    cellprob_tiles    = []   # flows[2]  (1-ch)
    total_cells       = 0

    for idx, tile in enumerate(tiles, start=1):
        logger.info(f"  ↳ tile {idx}/{len(tiles)}")

        # Add debug info about the tile for monitoring progress.
        logger.info(f"    Tile shape: {tile.shape}, min: {tile.min()}, max: {tile.max()}, mean: {tile.mean():.2f}.")

        # Run Cellpose on this tile.
        try:
            masks, flows, *_ = model.eval(
                tile[..., None],  # Add channel dimension
                diameter          = cellpose_params["diameter"],
                channels          = cellpose_params["channels"],
                flow_threshold    = cellpose_params["flow_threshold"],
                cellprob_threshold= cellpose_params["cellprob_threshold"],
                resample          = cellpose_params["resample"],
                augment           = False,  # No augmentation for inference
                batch_size        = cellpose_params["batch_size"],
                do_3D             = False
            )

            # Log information about the segmentation results for quality monitoring.
            num_cells = len(np.unique(masks)) - 1 if 0 in np.unique(masks) else len(np.unique(masks))
            logger.info(f"    Detected {num_cells} cells in tile {idx}.")

            mask_tiles.append(masks)
            flow_xy_tiles.append(flows[0])                 # shape (2, h, w)

            # Handle the case where flows[2] might be None
            if flows[2] is not None:
                cellprob_tiles.append(flows[2])            # shape (h, w)
            else:
                logger.warning(f"    No probability map returned for tile {idx}, using zeros.")
                cellprob_tiles.append(np.zeros_like(tile, dtype=np.float32))

            total_cells += num_cells

        except Exception as e:
            logger.error(f"Error processing tile {idx}: {e}")
            # Add an empty mask for this tile to maintain indexing.
            mask_tiles.append(np.zeros_like(tile, dtype=np.uint16))
            # Add placeholder flows.
            flow_xy_tiles.append(np.zeros((2, *tile.shape), dtype=np.float32))
            cellprob_tiles.append(np.zeros_like(tile, dtype=np.float32))

    # Stitch the tiled results into a seamless final segmentation.
    merged_masks      = merge_masks(mask_tiles,  slices, image.shape, overlap, logger, settings)
    merged_flow_xy    = merge_tiles_with_weighted_overlap(flow_xy_tiles,  slices, image.shape, overlap, logger)
    merged_cellprob   = merge_tiles_with_weighted_overlap(cellprob_tiles, slices, image.shape, overlap, logger)

    # Ensure **same API** as vanilla Cellpose: a 3-element list.
    merged_flows = [merged_flow_xy, merged_cellprob, None]

    return merged_masks, merged_flows, total_cells


def refine_segmentation_with_edges(image, masks, settings, logger):
    """
    Refine segmentation masks using Canny edge detection.

    This function improves segmentation accuracy by incorporating edge information
    from the original image. In kidney tissue, nuclei often have distinct boundaries
    that may not be perfectly captured by Cellpose. By detecting these edges and
    using them to refine the segmentation, we can achieve more accurate delineation
    of nuclear boundaries, especially in densely packed regions.

    Args:
        image: Original grayscale image.
        masks: Initial segmentation masks from Cellpose.
        settings: Dictionary containing edge detection parameters.
        logger: Logger for progress information.

    Returns:
        numpy.ndarray: Refined segmentation masks with improved boundaries.
    """
    logger.info("Applying edge detection based refinement to the segmentation mask")

    # Verify that image and masks have the same shape
    if image.shape != masks.shape:
        logger.error(f"Shape mismatch: image {image.shape} vs masks {masks.shape}")
        logger.warning("Cannot apply edge detection with mismatched shapes")

        # Find common region that can be used for both
        common_h = min(image.shape[0], masks.shape[0])
        common_w = min(image.shape[1], masks.shape[1])

        logger.info(f"Using common region of size {common_h}x{common_w}")

        # Crop both to common size
        image = image[:common_h, :common_w]
        masks = masks[:common_h, :common_w]

        logger.info(f"Cropped image to {image.shape} and masks to {masks.shape}")

    # Apply Canny edge detection
    edges = cv2.Canny(image,
                      threshold1=settings.get("CANNY_THRESHOLD1", 50),
                      threshold2=settings.get("CANNY_THRESHOLD2", 150))

    # Dilate edges to ensure they fully separate touching nuclei
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create binary mask from segmentation
    binary_mask = (masks > 0).astype(np.uint8) * 255

    # Subtract edges from binary mask
    refined_mask = cv2.subtract(binary_mask, dilated_edges)

    # Connected components analysis to get new labels
    num_labels, refined_labels = cv2.connectedComponents(refined_mask)

    logger.info(f"Refined segmentation into {num_labels - 1} objects after edge detection")
    return refined_labels

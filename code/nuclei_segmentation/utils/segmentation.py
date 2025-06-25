"""
NUCLEI SEGMENTATION FOR 2D IMAGES.

This module provides:
1. Cellpose segmentation (with optional tiling to handle large fields of view).
2. Canny edge-based postprocessing for refinement of connected or poorly separated nuclei.
"""

import numpy as np
import cv2

from .tiling import split_image_into_tiles, merge_masks, merge_tiles_with_weighted_overlap


def run_cellpose_on_tiles(model, image, cellpose_params, settings, logger):
    """
    Segment a large 2D grayscale image using Cellpose with optional tiling.

    Parameters:
    - model: Initialized Cellpose model.
    - image (np.ndarray): Grayscale input image, shape (H, W), dtype uint8 or float32.
    - cellpose_params (dict): Parameters for Cellpose evaluation.
    - settings (dict): Includes tile_side_length, tile_overlap, use_tiling.
    - logger: Logger instance for status reporting.

    Returns:
    - masks (np.ndarray): Label image, shape (H, W), dtype uint16.
    - flows (list of np.ndarray): [flow_xy (2,H,W), cellprob (H,W), None].
    - total_cells (int): Total number of detected nuclei.
    """

    height, width = image.shape
    tile_size = settings.get("tile_side_length")
    
    # If tile_overlap is a number between 0 and 1, treat it as a fraction.
    # Otherwise, treat it as a pixel count.
    tile_overlap = settings.get("tile_overlap")
    if isinstance(tile_overlap, (int, float)) and 0 <= tile_overlap <= 1:
        overlap = int(tile_size * tile_overlap)
    else:
        overlap = int(tile_overlap)  # Assume it's already in pixels
        
    # Ensure overlap is reasonable (not larger than tile_size/2).
    if overlap > tile_size // 2:
        logger.warning(f"Overlap {overlap} is larger than half the tile size {tile_size // 2}, clamping to {tile_size // 2}.")
    overlap = min(overlap, tile_size // 2)
    
    use_tiling = settings.get("use_tiling") and (height > tile_size or width > tile_size)

    logger.info(f"Segmentation initiated. Image shape: {image.shape}. Tiling: {use_tiling}")

    if not use_tiling:
        return run_single_pass_cellpose(model, image, cellpose_params, logger)

    tiles, slices = split_image_into_tiles(image, tile_size, overlap, logger)
    logger.info(f"Tiling produced {len(tiles)} subregions.")

    mask_tiles, flow_xy_tiles, cellprob_tiles = [], [], []
    total_cells = 0

    for i, tile in enumerate(tiles):
        logger.info(f"→ Segmenting tile {i + 1}/{len(tiles)} — shape: {tile.shape}, mean intensity: {tile.mean():.2f}")

        try:
            masks, flows, *_ = model.eval(
                tile[..., None],
                diameter=cellpose_params["diameter"],
                channels=cellpose_params["channels"],
                flow_threshold=cellpose_params["flow_threshold"],
                cellprob_threshold=cellpose_params["cellprob_threshold"],
                resample=cellpose_params["resample"],
                augment=False,
                batch_size=cellpose_params["batch_size"],
                do_3D=False,
            )

            num_cells = np.max(masks)
            logger.info(f"  ↪ Detected {num_cells} nuclei.")
            total_cells += num_cells

            mask_tiles.append(masks.astype(np.uint32))
            flow_xy_tiles.append(flows[0])  # shape (2, H, W)
            cellprob_tiles.append(flows[1])  # shape (H, W)

        except Exception as e:
            logger.error(f"  ✗ Tile {i + 1} failed: {e}")
            mask_tiles.append(np.zeros_like(tile, dtype=np.uint32))
            flow_xy_tiles.append(np.zeros((2, *tile.shape), dtype=np.float32))
            cellprob_tiles.append(np.zeros(tile.shape, dtype=np.float32))

    merged_masks = merge_masks(mask_tiles, slices, image.shape, overlap, logger, settings).astype(np.uint32)
    merged_flow_xy = merge_tiles_with_weighted_overlap(flow_xy_tiles, slices, image.shape, overlap, logger)
    merged_cellprob = merge_tiles_with_weighted_overlap(cellprob_tiles, slices, image.shape, overlap, logger)

    return merged_masks, [merged_flow_xy, merged_cellprob, None], total_cells


def run_single_pass_cellpose(model, image, cellpose_params, logger):
    """
    Run Cellpose segmentation on a single grayscale image.

    Parameters:
    - model: Cellpose model.
    - image (np.ndarray): Grayscale image, shape (H, W), dtype uint8 or float32.
    - cellpose_params (dict): Parameters for Cellpose evaluation.
    - logger: Logger instance.

    Returns:
    - masks (np.ndarray): Label image, dtype uint16.
    - flows (list): [flow_xy (2,H,W), cellprob (H,W), None].
    - total_cells (int): Number of labeled cells.
    """

    logger.info("Running full-image Cellpose segmentation (no tiling).")

    try:
        masks, flows, *_ = model.eval(
            image[..., None],
            diameter=cellpose_params["diameter"],
            channels=cellpose_params["channels"],
            flow_threshold=cellpose_params["flow_threshold"],
            cellprob_threshold=cellpose_params["cellprob_threshold"],
            resample=cellpose_params["resample"],
            augment=False,
            batch_size=cellpose_params["batch_size"],
            do_3D=False,
        )

        num_cells = np.max(masks)
        logger.info(f"Detected {num_cells} nuclei in full image.")
        return masks.astype(np.uint32), [flows[0], flows[1], None], num_cells

    except Exception as e:
        logger.error(f"✗ Cellpose failed on full image: {e}")
        return (
            np.zeros_like(image, dtype=np.uint32),
            [np.zeros((2, *image.shape), dtype=np.float32), np.zeros(image.shape, dtype=np.float32), None],
            0,
        )


def refine_segmentation_with_edges(image, masks, settings, logger):
    """
    Refine segmentation by subtracting strong edges from mask interior.

    Parameters:
    - image (np.ndarray): Grayscale image, shape (H, W), dtype uint8 or float32.
    - masks (np.ndarray): Binary or label mask, shape (H, W), dtype uint16.
    - settings (dict): Contains 'canny_threshold1', 'canny_threshold2'.
    - logger: Logger instance.

    Returns:
    - labeled (np.ndarray): Connected component-labeled refined mask.
    """

    if image.shape != masks.shape:
        logger.warning(f"Shape mismatch. Cropping masks and image to minimum common area.")
        h, w = min(image.shape[0], masks.shape[0]), min(image.shape[1], masks.shape[1])
        image = image[:h, :w]
        masks = masks[:h, :w]

    t1 = settings.get("canny_threshold1", 50)
    t2 = settings.get("canny_threshold2", 150)

    edges = cv2.Canny(image, threshold1=t1, threshold2=t2)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    binary_mask = (masks > 0).astype(np.uint8)
    cleaned_mask = np.logical_and(binary_mask, dilated_edges == 0).astype(np.uint8)

    n_labels, labeled = cv2.connectedComponents(cleaned_mask)

    logger.info(f"Edge-refined segmentation yielded {n_labels - 1} connected regions.")
    return labeled

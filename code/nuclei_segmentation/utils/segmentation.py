"""
NUCLEI SEGMENTATION FOR 2D IMAGES (REFactored).

Author: Christos Botos.
Affiliation: Institute of Molecular Biology and Biotechnology.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Wrapper functions to run Cellpose on large histological images with square
    tiling, optional merging, and diagnostic logging.  This version syncs with
    the **backwards‑compatible** tiling utilities (split_image_into_tiles,
    merge_tiles_with_weighted_overlap, merge_masks) introduced in 2025‑06‑28.

Key Updates:
    • Calls split_image_into_tiles with explicit ``tile_h, tile_w`` arguments
      to avoid positional ambiguity (tile_h = tile_w = tile_size).
    • Passes ``logger=logger`` as keyword to merge_tiles_with_weighted_overlap
      to match its updated signature.
    • Adds type hints and full‑stop comments for clarity.
"""

from __future__ import annotations

from typing import Tuple, List

import numpy as np

from .tiling import (
    split_image_into_tiles,
    merge_masks,
    merge_tiles_with_weighted_overlap,
)


# -----------------------------------------------------------------------------
# Core function.
# -----------------------------------------------------------------------------

def run_cellpose_on_tiles(
    model,
    image: np.ndarray,
    cellpose_params: dict,
    settings: dict,
    logger,
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """Segment *image* with optional tiling and return merged results.

    Parameters:
        model:   Pre‑loaded Cellpose model.
        image:   2‑D grayscale array (H×W).
        cellpose_params: Dict of Cellpose hyper‑parameters.
        settings: Dict with tiling keys ``tile_side_length``, ``tile_overlap``,
                  and boolean ``use_tiling``.
        logger:  Logger for status output.

    Returns:
        masks (H×W uint32), [flow_xy, cellprob, None], total_cells.
    """

    H, W = image.shape
    tile_size: int = settings.get("tile_side_length")

    # Interpret overlap: percentage or pixel count.
    overlap_cfg = settings.get("tile_overlap")
    if isinstance(overlap_cfg, (int, float)) and 0 <= overlap_cfg <= 1:
        overlap = int(tile_size * overlap_cfg)
    else:
        overlap = int(overlap_cfg)
    overlap = min(overlap, tile_size // 2)  # Clamp.

    use_tiling: bool = settings.get("use_tiling") and (H > tile_size or W > tile_size)
    logger.info("Segmentation initiated. Image shape: %s. Tiling: %s.", image.shape, use_tiling)

    # ──────────────────────────────────────────────────────────────
    # 1.  Fallback: single‑pass mode.
    # ──────────────────────────────────────────────────────────────

    if not use_tiling:
        return _run_single_pass_cellpose(model, image, cellpose_params, logger)

    # ──────────────────────────────────────────────────────────────
    # 2.  Tile the image (explicit tile_h, tile_w interface).
    # ──────────────────────────────────────────────────────────────

    tile_iter = split_image_into_tiles(image, tile_size, tile_size, overlap, logger)

    # Materialise once; keeps the streaming API, but we still want random access later.
    tiles, slices = zip(*tile_iter)  # tiles -> tuple, slices -> tuple
    tiles = list(tiles)
    slices = list(slices)

    logger.info("Tiling produced %d subregions.", len(tiles))

    mask_tiles: List[np.ndarray] = []
    flow_xy_tiles: List[np.ndarray] = []
    cellprob_tiles: List[np.ndarray] = []

    total_cells = 0
    for idx, tile in enumerate(tiles):
        logger.info(
            "→ Segmenting tile %d/%d — shape=%s, mean=%.2f.",
            idx + 1, len(tiles), tile.shape, tile.mean()
        )

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

            num_cells = int(masks.max())
            logger.info("  ↪ Detected %d nuclei.", num_cells)
            total_cells += num_cells

            '''Convert Cellpose output to safe uint32 masks.'''
            if masks is None:  # Cellpose found 0 nuclei.
                tile_mask = np.zeros(tile.shape[-2:],  # Works for HW or CHW tiles.
                                     dtype=np.uint32)
            else:
                tile_mask = masks.astype(np.uint32)

            mask_tiles.append(tile_mask)  # ← use tile_mask from now on.
            flow_xy_tiles.append(flows[0])  # shape (2, H, W)
            cellprob_tiles.append(flows[1])  # shape (H, W)

        except Exception as exc:
            logger.error("  ✗ Tile %d failed: %s", idx + 1, exc)
            mask_tiles.append(np.zeros_like(tile, dtype=np.uint32))
            flow_xy_tiles.append(np.zeros((2, *tile.shape), dtype=np.float32))
            cellprob_tiles.append(np.zeros(tile.shape, dtype=np.float32))

    # ──────────────────────────────────────────────────────────────
    # 3.  Merge results.
    # ──────────────────────────────────────────────────────────────

    merged_masks = merge_masks(
        mask_tiles, slices, image.shape, overlap, logger, settings
    ).astype(np.uint32)

    merged_flow_xy = merge_tiles_with_weighted_overlap(
        flow_xy_tiles, slices, image.shape, overlap, logger=logger
    )
    merged_cellprob = merge_tiles_with_weighted_overlap(
        cellprob_tiles, slices, image.shape, overlap, logger=logger
    )

    return merged_masks, [merged_flow_xy, merged_cellprob, None], total_cells


# -----------------------------------------------------------------------------
# Helper: single‑pass Cellpose call.
# -----------------------------------------------------------------------------

def _run_single_pass_cellpose(
    model,
    image: np.ndarray,
    cellpose_params: dict,
    logger,
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """Run Cellpose on the full image without tiling."""

    logger.info("Running full‑image Cellpose segmentation (no tiling).")

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

        num_cells = int(masks.max())
        logger.info("Detected %d nuclei in full image.", num_cells)
        return masks.astype(np.uint32), [flows[0], flows[1], None], num_cells

    except Exception as exc:
        logger.error("✗ Cellpose failed on full image: %s", exc)
        return (
            np.zeros_like(image, dtype=np.uint32),
            [
                np.zeros((2, *image.shape), dtype=np.float32),
                np.zeros(image.shape, dtype=np.float32),
                None,
            ],
            0,
        )

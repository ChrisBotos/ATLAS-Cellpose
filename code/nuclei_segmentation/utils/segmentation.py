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

from numpy.lib.format import open_memmap
from pathlib import Path
import numpy as np

from .tiling import (
    split_image_into_tiles,
    merge_masks,
    merge_tiles_with_weighted_overlap,
)

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


# -----------------------------------------------------------------------------
# Core function.
# -----------------------------------------------------------------------------


def run_cellpose_on_tiles(
    model,
    image: np.ndarray,
    cellpose_params: dict,
    settings: dict,
    logger,
) -> Tuple[np.memmap, List[None], int]:
    """
    Author: Christos Botos.
    Affiliation: Institute of Molecular Biology and Biotechnology.
    Contact: botoschristos@gmail.com

    Function Name
    -------------
    run_cellpose_on_tiles

    Description
    -----------
    Segment *image* with Cellpose, optionally in a tiled fashion, while
    streaming the resulting mask directly to an on-disk NumPy mem-map.
    This keeps peak RAM usage essentially constant, even for very
    large slides.

    Parameters
    ----------
    model : cellpose.models.Cellpose
        A pre-loaded Cellpose model.
    image : np.ndarray
        Two-dimensional grayscale image of shape (H, W).
    cellpose_params : dict
        Hyper-parameters forwarded to ``model.eval``.
    settings : dict
        Must contain the keys:
            • output_dir (str | Path) – Results root directory.
            • tile_side_length (int) – Square tile edge length in pixels.
            • tile_overlap (int | float) – Overlap (pixels or fraction).
            • use_tiling (bool, optional) – Force/disable tiling.
    logger : logging.Logger
        Logger for status and diagnostics.

    Returns
    -------
    Tuple[np.memmap, List[None], int]
        masks_mm : np.memmap
            On-disk (H × W) uint32 segmentation mask.
        [None, None, None]
            Placeholder for Cellpose flows (disabled to save memory).
        total_cells : int
            Total number of labelled objects.
    """
    # ------------------------------------------------------------------ #
    #                         Disk-backed mask                           #
    # ------------------------------------------------------------------ #
    masks_dir = Path(settings["output_dir"]) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    H, W = map(int, image.shape)
    masks_mm = open_memmap(
        filename=str(masks_dir / "tile_masks.tmp"),   # ← temporary file
        mode="w+",
        dtype=np.uint32,
        shape=(H, W),
    )

    # ------------------------------------------------------------------ #
    #                       Tiling geometry                              #
    # ------------------------------------------------------------------ #
    tile_size = int(settings["tile_side_length"])
    ov_cfg = settings["tile_overlap"]

    if isinstance(ov_cfg, (int, float)) and 0 <= ov_cfg <= 1:
        overlap = int(tile_size * ov_cfg)
    else:
        overlap = int(ov_cfg)
    overlap = min(overlap, tile_size // 2)     # Never exceed half a tile.

    auto_tiling = H > tile_size or W > tile_size
    use_tiling = settings.get("use_tiling", True) and auto_tiling

    logger.info(
        "Segmentation initiated – image shape %s.  Tiling enabled: %s.",
        image.shape, use_tiling,
    )

    # ------------------------------------------------------------------ #
    #               Fast path: single-pass on the full image             #
    # ------------------------------------------------------------------ #
    if not use_tiling:
        masks, _, total_cells = _run_single_pass_cellpose(
            model, image, cellpose_params, logger
        )
        masks_mm[:] = masks.astype(np.uint32)
        masks_mm.flush()
        return masks_mm, [None, None, None], total_cells

    # ------------------------------------------------------------------ #
    #                  Slow path: tiled segmentation                     #
    # ------------------------------------------------------------------ #
    tile_iter = split_image_into_tiles(
        img=image,
        tile_h=tile_size,
        tile_w=tile_size,
        overlap=overlap,
        logger=logger,
    )
    tiles, slices = zip(*tile_iter)
    n_tiles = len(tiles)
    logger.info("Tiling produced %d sub-regions.", n_tiles)

    next_gid = 1
    total_cells = 0

    # Prepare lists for merging after segmentation
    mask_tiles = []
    tile_slices = []

    for idx, (tile, (ys, xs)) in enumerate(zip(tiles, slices), start=1):
        logger.info(
            "→ Segmenting tile %d/%d — shape=%s, mean=%.2f.",
            idx, n_tiles, tile.shape, tile.mean(),
        )
        try:
            masks, *_ = model.eval(
                tile[..., None],                         # Add dummy channel.
                diameter=cellpose_params["diameter"],
                channels=cellpose_params["channels"],
                flow_threshold=cellpose_params["flow_threshold"],
                cellprob_threshold=cellpose_params["cellprob_threshold"],
                resample=cellpose_params["resample"],
                augment=False,
                batch_size=cellpose_params["batch_size"],
                do_3D=False,
            )

            # Cellpose returns None when nothing is found.
            if masks is None:
                tile_mask = np.zeros(tile.shape, dtype=np.uint32)
            else:
                tile_mask = masks.astype(np.uint32)

            # Vectorised relabelling: make every tile label globally unique.
            non_zero = tile_mask != 0
            n_labels_tile = int(tile_mask.max())

            if n_labels_tile:
                tile_mask[non_zero] += next_gid
                masks_mm[ys, xs][non_zero] = tile_mask[non_zero]
                next_gid += n_labels_tile
                total_cells += n_labels_tile

            logger.info("  ↪ Detected %d cells.", n_labels_tile)

            # Append tile mask and slice for later merging
            mask_tiles.append(tile_mask.copy())
            tile_slices.append((ys, xs))

        except Exception as exc:  # noqa: BLE001  (broad except is intentional here)
            logger.error("  ✗ Tile %d failed: %s", idx, exc)

    # ------------------------------------------------------------------ #
    #                  Merge all tile masks into one                     #
    # ------------------------------------------------------------------ #
    from .tiling import merge_masks

    merged = merge_masks(
        mask_tiles,
        tile_slices,
        image_shape=(H, W),
        overlap=overlap,
        logger=logger,
        settings=settings,
    )

    # Overwrite memmap with fused result
    masks_mm[:] = merged
    masks_mm.flush()

    logger.info("Finished writing %d total cells to disk.", total_cells)

    # We deliberately drop the flow arrays to keep memory usage low.
    return masks_mm, [None, None, None], total_cells


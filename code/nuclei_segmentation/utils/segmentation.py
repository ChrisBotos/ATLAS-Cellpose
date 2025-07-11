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
)

from .merge_streaming import merge_masks_streaming

# Type alias for readability.
MaskReturn = Tuple[np.memmap, List[None], int]

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


"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: run_cellpose_on_tiles.py.
Description:
    Segment a 2‑D grayscale image with Cellpose. The function automatically splits
    very large images into overlapping tiles, runs Cellpose on each tile, and
    writes the resulting instance mask directly to an on‑disk NumPy mem‑map so
    that peak RAM usage remains roughly constant.

    The present revision implements the lightweight memory‑footprint fix
    described as *solution2.2* in the discussion: each tile mask is appended to
    the list for later merging **without** creating an additional in‑RAM copy,
    and the local variable holding the array is set to *None* immediately after
    use. This avoids keeping two identical copies of every tile simultaneously
    and typically saves 1–2GB on 40k×40k whole‑slide images.

Dependencies:
    • Python≥3.10.
    • numpy, pathlib, numpy.memmap, cellpose, pytest (for the optional test).
"""


"""------------------------------------------------------------------------
Function
--------
"""

def run_cellpose_on_tiles(
    model,
    image: np.ndarray,
    cellpose_params: dict,
    settings: dict,
    logger,
) -> MaskReturn:
    """Segment *image* with Cellpose, streaming the mask to disk.

    The function switches automatically between a *single‑pass* mode (for images
    that fit comfortably into memory) and a *tiled* mode for larger slides. In
    tiled mode each mask tile is appended to *mask_tiles* **without** calling
    ``copy()``. The local reference is subsequently nulled so that Python’s
    garbage collector can reclaim the array as soon as the merge finishes.

    Parameters
    ----------
    model : cellpose.models.Cellpose
        A pre‑loaded Cellpose model instance.
    image : np.ndarray
        Two‑dimensional grayscale image of shape *(H,W)*.
    cellpose_params : dict
        Hyper‑parameters forwarded verbatim to ``model.eval``.
    settings : dict
        Must contain at least the following keys:
            • ``output_dir`` (str|Path)   – Root directory for results.
            • ``tile_side_length`` (int)    – Square tile edge length in px.
            • ``tile_overlap`` (int|float) – Overlap (px or fraction).
            • ``use_tiling`` (bool, optional) – Force or disable tiling.
    logger : logging.Logger
        Logger for progress updates and diagnostics.

    Returns
    -------
    masks_mm : np.memmap
        On‑disk *(H×W)* array containing the final fused uint32 mask.
    _flows : list[None, None, None]
        Placeholder. Cellpose flows are discarded to save memory.
    total_cells : int
        Total number of labelled objects across the whole slide.
    """

    # ------------------------------------------------------------------
    # Create an on‑disk mem‑map that will store the final mask.
    # ------------------------------------------------------------------
    masks_dir = Path(settings["output_dir"]).expanduser().resolve() / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    H, W = (int(dim) for dim in image.shape)
    masks_mm = open_memmap(
        filename=str(masks_dir / "tile_masks.tmp"),  # Temporary file.
        mode="w+",
        dtype=np.uint32,
        shape=(H, W),
    )

    # ------------------------------------------------------------------
    # Determine tiling geometry.
    # ------------------------------------------------------------------
    tile_size = int(settings["tile_side_length"])
    overlap_cfg = settings["tile_overlap"]

    if isinstance(overlap_cfg, (int, float)) and 0 <= overlap_cfg <= 1:
        overlap = int(tile_size * overlap_cfg)
    else:
        overlap = int(overlap_cfg)
    overlap = min(overlap, tile_size // 2)  # Overlap never exceeds half a tile.

    auto_tiling = H > tile_size or W > tile_size
    use_tiling = settings.get("use_tiling", True) and auto_tiling

    logger.info(
        "Segmentation initiated – image shape %s.  Tiling enabled: %s.",
        image.shape,
        use_tiling,
    )

    # ------------------------------------------------------------------
    # Fast path: run Cellpose on the full image if tiling is unnecessary.
    # ------------------------------------------------------------------
    if not use_tiling:
        masks, _, total_cells = _run_single_pass_cellpose(model, image, cellpose_params, logger)
        masks_mm[:] = masks.astype(np.uint32)
        masks_mm.flush()
        return masks_mm, [None, None, None], total_cells

    # ------------------------------------------------------------------
    # Slow path: split the image into tiles and process each tile individually.
    # ------------------------------------------------------------------
    tile_iter = split_image_into_tiles(
        img=image,
        tile_h=tile_size,
        tile_w=tile_size,
        overlap=overlap,
        logger=logger,
    )

    # *tiles* is a sequence of image patches; *slices* locates each patch in the
    # original coordinate system.
    tiles, slices = zip(*tile_iter)
    n_tiles = len(tiles)
    logger.info("Tiling produced %d sub‑regions.", n_tiles)

    next_gid: int = 1  # Next global instance ID.
    total_cells: int = 0

    mask_tiles: List[np.ndarray] = []
    tile_slices: List[Tuple[slice, slice]] = []

    for idx, (tile, (ys, xs)) in enumerate(zip(tiles, slices), start=1):
        logger.info(
            "→ Segmenting tile %d/%d — shape=%s, mean=%.2f.",
            idx,
            n_tiles,
            tile.shape,
            float(tile.mean()),
        )
        try:
            # Cellpose requires a channel dimension even for grayscale input.
            masks, *_ = model.eval(
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

            # Cellpose returns *None* when it finds no objects.
            tile_mask: np.ndarray = (
                np.zeros(tile.shape, dtype=np.uint32) if masks is None else masks.astype(np.uint32)
            )

            # Vectorised relabelling ensures unique IDs across tiles.
            non_zero = tile_mask != 0
            n_labels_tile: int = int(tile_mask.max())

            if n_labels_tile:
                tile_mask[non_zero] += next_gid
                masks_mm[ys, xs][non_zero] = tile_mask[non_zero]
                next_gid += n_labels_tile
                total_cells += n_labels_tile

            logger.info("  ↪ Detected %d cells.", n_labels_tile)

            # Append the *original* tile mask (no .copy()) for later merging.
            mask_tiles.append(tile_mask)
            tile_mask = None  # Drop the local reference to free RAM promptly.

            tile_slices.append((ys, xs))

        except Exception as exc:  # noqa: BLE001 – broad except is intentional.
            logger.error("  ✗ Tile %d failed: %s", idx, exc)

    # ------------------------------------------------------------------
    # Merge the individual tile masks into one coherent global mask.
    # ------------------------------------------------------------------

    merged = merge_masks_streaming(
        mask_tiles,  # list[np.ndarray]
        tile_slices,  # list[(slice,slice)]
        (H, W),  # full image shape
        overlap=overlap,
        logger=logger,
        settings=settings,
    )


    # ------------------------------------------------------------------ #
    # Persist the fused mask *and* free per-tile allocations immediately #
    # ------------------------------------------------------------------ #

    mask_tiles.clear()  # Drop references so GC can release them now.
    tile_slices.clear()

    masks_mm[:] = merged
    masks_mm.flush()

    logger.info("Finished writing %d total cells to disk.", total_cells)

    # Flows are deliberately discarded to keep the memory footprint minimal.
    return masks_mm, [None, None, None], total_cells

"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos

Script Name: merge_streaming.py.
Description:
    Memory‑efficient streaming merger for segmentation mask tiles. Keeps only
    one tile in RAM at any time by committing each tile directly into a memory‑
    mapped global canvas. Designed for slides with thousands of tiles.

Dependencies:
    • Python >= 3.10.
    • numpy, matplotlib, psutil (optional for debugging).

Usage (drop‑in inside tiling.py):
    from merge_streaming import merge_masks_streaming

    merged = merge_masks_streaming(
        mask_tiles,
        tile_slices,
        full_image_shape=(H, W),
        overlap=overlap_px,
        logger=logger,
        settings=settings,
        qc=settings.get("qc_overlays", False),
    )

Changes v1.1 – 2025‑07‑12
    • Fix: correct tempfile handling – np.memmap now receives a *path*, not a
      file descriptor; prevents TypeError on Python ≥3.11.
    • Added configurable `memmap_path` to override the temporary location.
    • Robust dtype selection (uint16 / uint32) and validation.

"""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from random import random
from typing import Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["merge_masks_streaming"]


def _temp_datfile(prefix: str = "iri_global_") -> Path:
    """Return an on‑disk filename for the global memmap and ensure the handle is
    closed immediately (np.memmap will reopen it)."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".dat")
    Path(path).unlink(missing_ok=True)  # np.memmap will recreate
    try:
        import os
        os.close(fd)
    except OSError:
        pass
    return Path(path)


def merge_masks_streaming(
    mask_tiles: Sequence[np.ndarray],
    slices: Sequence[Tuple[slice, slice]],
    full_image_shape: Tuple[int, int],
    overlap: int,
    logger,
    settings: Dict,
    qc: bool = False,
) -> np.ndarray:
    """Merge *mask_tiles* into a single canvas using ≈constant memory.

    Parameters
    ----------
    mask_tiles : list[np.ndarray]
        Label images (each tile) as returned by Cellpose.
    slices : list[(slice, slice)]
        (y_slice, x_slice) for the tile's location in the global canvas.
    full_image_shape : (H, W)
        Height × width of the full slide.
    overlap : int
        Number of pixels that overlap between neighbouring tiles.
    logger : logging.Logger | None
        For progress messages; may be *None*.
    settings : dict
        Global pipeline settings – looks for:
            • "merge_overlap_threshold"   (float)
            • "memmap_dtype"             ("uint16" or "uint32")
            • "memmap_path"              (optional str path)
            • "qc_downsample_factor"     (int, default 4)
    qc : bool, default False
        Produce down‑sampled colour preview overlays.

    Returns
    -------
    np.ndarray
        The merged label canvas (loaded in memory); caller may `.astype(np.uint16)`
        or flush to disk as needed.
    """
    H, W = full_image_shape
    ov_thr: float = float(settings.get("merge_overlap_threshold", 0.3))
    dtype = np.uint16 if str(settings.get("memmap_dtype", "uint32")).lower() == "uint16" else np.uint32

    # ------------------------------------------------------------------
    # Allocate global memmap.
    # ------------------------------------------------------------------
    mm_path = Path(settings.get("memmap_path")) if settings.get("memmap_path") else _temp_datfile()
    merged = np.memmap(mm_path, mode="w+", dtype=dtype, shape=(H, W))

    if logger:
        logger.info("Streaming merge: %d tiles → %dx%d canvas (%s)", len(mask_tiles), W, H, dtype.__name__)

    # ── optional QC preview --------------------------------------------------
    ds = max(1, int(settings.get("qc_downsample_factor", 4)))
    if qc:
        bh, bw = H // ds, W // ds
        before: np.ndarray = np.zeros((bh, bw, 3), dtype=np.float32)
        weights: np.ndarray = np.zeros((bh, bw),     dtype=np.float32)

    next_gid = 1
    border_touch: Dict[int, bool] = {}

    for idx, (tile, (ys, xs)) in enumerate(zip(mask_tiles, slices), 1):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        roi = merged[y0:y1, x0:x1]

        if qc:
            yy0, yy1 = y0 // ds, y1 // ds
            xx0, xx1 = x0 // ds, x1 // ds
            colour = np.array([random(), random(), random()], dtype=np.float32)
            bin_tile = (tile > 0).astype(np.float32)
            before[yy0:yy1, xx0:xx1]  += colour * bin_tile[::ds, ::ds]
            weights[yy0:yy1, xx0:xx1] += bin_tile[::ds, ::ds]

        # ------------------------------------------------------------------
        # Merge algorithm (same logic as batch version) --------------------
        # ------------------------------------------------------------------
        for lbl in np.unique(tile):
            if lbl == 0:
                continue
            local_mask = tile == lbl
            area_local = int(local_mask.sum())

            # Fast border stub check
            rim = np.zeros_like(tile, dtype=bool)
            if x0 > 0: rim[:, :overlap]  = True
            if x1 < W: rim[:, -overlap:] = True
            if y0 > 0: rim[:overlap, :]  = True
            if y1 < H: rim[-overlap:, :] = True
            if (rim & local_mask).sum() < ov_thr * area_local:
                gid = next_gid
                next_gid += 1
                roi[local_mask] = gid
                border_touch[gid] = False
                continue

            # Is this mask touching slide border? (global coords)
            local_border = (y0 == 0 or y1 == H or x0 == 0 or x1 == W) and (
                local_mask[0, :].any() or local_mask[-1, :].any() or local_mask[:, 0].any() or local_mask[:, -1].any()
            )

            touched: np.ndarray = np.unique(roi[local_mask])
            touched = touched[touched > 0]

            chosen_gid = None
            kill_local = False

            for gid in touched:
                area_gid = int((roi == gid).sum())
                ov_area = int(np.logical_and(roi == gid, local_mask).sum())
                gid_border = border_touch.get(int(gid), False)
                if (ov_area / area_local) >= ov_thr or (ov_area / area_gid) >= ov_thr:
                    if local_border ^ gid_border:
                        if local_border:
                            kill_local = True
                        else:
                            merged[merged == gid] = 0
                            chosen_gid = int(gid)
                            border_touch[chosen_gid] = False
                        break
                    else:
                        chosen_gid = int(gid)
                        border_touch[chosen_gid] = local_border or gid_border
                        break

            if kill_local:
                if logger:
                    logger.info("✂ Removed border stub (%d px) at tile %d, lbl %d.", area_local, idx, lbl)
                continue

            if chosen_gid is None:
                chosen_gid = next_gid
                next_gid += 1
                border_touch[chosen_gid] = local_border

            roi[local_mask] = chosen_gid

        # free tile memory immediately
        mask_tiles[idx - 1] = None
        if idx % 64 == 0:
            gc.collect()

    if qc:
        valid = weights > 0
        before[valid] /= weights[valid, None]
        viz_dir = Path(settings.get("output_dir", ".")) / "tile_merge_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(viz_dir / "before_lowres.png", np.clip(before, 0, 1))

    if logger:
        logger.info("Streaming merge completed – %d unique objects.", next_gid - 1)

    merged_arr = np.asarray(merged)

    # Optional: clean up tmp file if the array is small enough to keep in RAM
    try:
        mm_path.unlink(missing_ok=True)
    except PermissionError:
        pass

    return merged_arr

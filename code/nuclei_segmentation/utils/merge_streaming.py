"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_streaming.py.
Description:
    Memory‑efficient streaming merger for instance‑segmentation mask tiles.
    Only one tile is ever resident in RAM; the full‑slide canvas is stored as
    a NumPy memory‑map on disk, so peak usage stays < size(tile) + ε.

    The merge algorithm obeys the following rule (which I endorse):

        1. Does the local object have ≥ *threshold* of its pixels inside the
           overlap band?  If **No**, give it a fresh global ID and skip merging.
        2. Does the local object share ≥ *threshold* of its pixels with any
           global object **or** does any global object share ≥ *threshold* of
           its own pixels with the local object?  If **No**, break.
        3. Does the local object touch its tile border while the other object
           does **not** touch its own tile border?  If **Yes**, remove the
           border stub and break.
        4. Merge the two objects under the surviving global ID and repeat step
           2 until no further eligible neighbour exists.

    The present implementation follows the rule verbatim and maintains the
    same streaming strategy as earlier revisions; the only behavioural change
    is a fix that prevents an infinite loop when an object overlaps several
    neighbours.

Dependencies:
    • Python ≥ 3.10.
    • numpy, matplotlib.

Usage example:
    from merge_streaming import merge_masks_streaming

    merged = merge_masks_streaming(
        mask_tiles,
        tile_slices,
        full_image_shape=(H, W),
        overlap=overlap_px,
        logger=logger,
        settings={"merge_overlap_threshold": 0.3, "output_dir": "./results"},
        qc=True,
    )

Outputs (when *qc* is True):
    • <output_dir>/tile_merge_viz/before_lowres.png
    • <output_dir>/tile_merge_viz/after_lowres.png

"""
from __future__ import annotations

import gc
import math
import tempfile
from pathlib import Path
from random import random
from typing import Dict, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["merge_masks_streaming"]


# -----------------------------------------------------------------------------
# Utility helpers.
# -----------------------------------------------------------------------------

def _temp_mm_file(prefix: str = "iri_global_") -> Path:
    """Return a temporary filename for a mem‑mapped canvas."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".dat")
    Path(path).unlink(missing_ok=True)  # The mem‑map will recreate the file.
    try:
        import os
        os.close(fd)
    except OSError:
        pass
    return Path(path)


def _down_factor(h: int, w: int, target: int = 2000) -> int:
    """Integer stride that scales the longest side to *≤ target*."""
    return max(1, int(math.ceil(max(h, w) / target)))


def _save_overlay(lbl_img: np.ndarray, path: Path, target_px: int = 2000) -> None:
    """Save a down‑sampled, pseudo‑coloured thumbnail of *lbl_img*."""
    ds = _down_factor(*lbl_img.shape, target_px)
    cmap = np.random.default_rng(0).random((int(lbl_img.max()) + 1, 3))
    thumb = cmap[lbl_img[::ds, ::ds]]
    plt.imsave(path, thumb)


def _touches_tile_border(mask: np.ndarray) -> bool:
    """Return True if *mask* touches any edge of its tile."""
    return bool(
        mask[0, :].any()      # top
        or mask[-1, :].any()  # bottom
        or mask[:, 0].any()   # left
        or mask[:, -1].any()  # right
    )



# -----------------------------------------------------------------------------
# Core function.
# -----------------------------------------------------------------------------

def merge_masks_streaming(
    mask_tiles: Sequence[np.ndarray],
    slices: Sequence[Tuple[slice, slice]],
    full_image_shape: Tuple[int, int],
    overlap: int,
    logger,
    settings: Dict,
    *,
    qc: bool = False,
) -> np.ndarray:
    """Fuse *mask_tiles* into a slide‑sized label map.

    Parameters
    ----------
    mask_tiles : sequence of uint32/uint16 arrays
        The segmented tiles, in the same order as *slices*.
    slices : sequence of (y_slice, x_slice)
        Placement of each tile within the slide.
    full_image_shape : (H, W)
        Height × width of the whole slide.
    overlap : int
        Overlap width in pixels (same for x and y).
    logger : logging.Logger | None
        Progress sink; may be *None*.
    settings : dict
        Requires ``merge_overlap_threshold`` (float in 0‑1).
    qc : bool, default False
        If *True*, write down‑sampled overlays for visual QC.

    Returns
    -------
    numpy.ndarray (uint16/uint32)
        The merged mask fully loaded into RAM.
    """
    H, W = full_image_shape
    thr = float(settings.get("merge_overlap_threshold", 0.3))

    dtype = np.uint32 if max(tile.max() for tile in mask_tiles) > 65535 else np.uint16
    mm_path = Path(settings.get("memmap_path", _temp_mm_file()))
    merged = np.memmap(mm_path, mode="w+", dtype=dtype, shape=(H, W))

    if logger:
        logger.info(
            "Streaming‑merge %d tiles → %dx%d canvas (%s).",
            len(mask_tiles),
            W,
            H,
            dtype.__name__,
        )

    # QC thumbnail initialisation.
    if qc:
        ds = _down_factor(H, W, 2000)
        bh, bw = H // ds, W // ds
        before = np.zeros((bh, bw, 3), dtype=np.float32)
        weights = np.zeros((bh, bw), dtype=np.float32)
        viz_dir = Path(settings.get("output_dir", ".")) / "tile_merge_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

    next_gid: int = 1  # Next unused global ID.
    gid_area: Dict[int, int] = {}
    gid_border: Dict[int, bool] = {}

    for idx, (tile, (ys, xs)) in enumerate(zip(mask_tiles, slices), start=1):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        roi = merged[y0:y1, x0:x1]

        # QC – flat random colour per tile object.
        if qc:
            yy0, yy1 = y0 // ds, y1 // ds
            xx0, xx1 = x0 // ds, x1 // ds
            colour = np.array([random(), random(), random()], dtype=np.float32)
            bin_tile = (tile > 0).astype(np.float32)
            before[yy0:yy1, xx0:xx1] += colour * bin_tile[::ds, ::ds]
            weights[yy0:yy1, xx0:xx1] += bin_tile[::ds, ::ds]

        # Pre‑compute the overlap rim for this tile.
        rim = np.zeros_like(tile, dtype=bool)
        if x0 > 0:
            rim[:, :overlap] = True
        if x1 < W:
            rim[:, -overlap:] = True
        if y0 > 0:
            rim[:overlap, :] = True
        if y1 < H:
            rim[-overlap:, :] = True

        for lbl in np.unique(tile):
            if lbl == 0:
                continue
            local_mask = tile == lbl
            area_local = int(local_mask.sum())
            if area_local == 0:
                continue

            # ── Rule 1 ────────────────────────────────────────────────────
            if (rim & local_mask).sum() < thr * area_local:
                gid = next_gid
                next_gid += 1
                roi[local_mask] = gid
                gid_area[gid] = area_local
                gid_border[gid] = _touches_tile_border(local_mask)
                continue

            # ── Rule 2–4 loop ────────────────────────────────────────────
            local_border_flag = _touches_tile_border(local_mask)
            kill_local = False
            chosen_gid = None
            seen_gids: Set[int] = set()  # Prevent infinite re‑visits.

            while True:
                touched = [
                    g for g in np.unique(roi[local_mask]) if g > 0 and g not in seen_gids
                ]
                if not touched:
                    break
                merged_this_round = False

                for gid in touched:
                    seen_gids.add(gid)
                    area_gid = gid_area.get(gid)
                    if area_gid is None:
                        area_gid = int((merged == gid).sum())
                        gid_area[gid] = area_gid

                    ov_area = int(np.logical_and(roi == gid, local_mask).sum())

                    # Rule 2 – asymmetric overlap.
                    if (ov_area / area_local) < thr and (ov_area / area_gid) < thr:
                        continue

                    # Rule 3 – border‑stub arbitration.
                    gid_border_flag = gid_border.get(gid, False)
                    if local_border_flag and not gid_border_flag:
                        kill_local = True
                        break
                    if gid_border_flag and not local_border_flag:
                        merged[merged == gid] = 0  # Remove stub object.
                        gid_area.pop(gid, None)
                        gid_border.pop(gid, None)
                        continue  # Check other neighbours.

                    # Rule 4 – merge under *gid*.
                    chosen_gid = gid
                    roi[local_mask] = gid  # Stamp now → prevents infinite loop.
                    gid_area[gid] += area_local
                    gid_border[gid] = local_border_flag or gid_border_flag
                    merged_this_round = True
                    break  # Re‑evaluate overlaps.

                if kill_local or not merged_this_round:
                    break

            if kill_local:
                if logger:
                    logger.info(
                        "✂\uFE0F  Removed border stub (%d px) at tile %d label %d.",
                        area_local,
                        idx,
                        lbl,
                    )
                continue

            if chosen_gid is None:  # No neighbour matched.
                chosen_gid = next_gid
                next_gid += 1
                gid_area[chosen_gid] = area_local
                gid_border[chosen_gid] = local_border_flag
                roi[local_mask] = chosen_gid

        # Release tile to keep memory footprint flat.
        mask_tiles[idx - 1] = None
        if idx % 64 == 0:
            gc.collect()

    # ── QC thumbnails ─────────────────────────────────────────────────────
    if qc:
        valid = weights > 0
        before[valid] /= weights[valid, None]
        plt.imsave(viz_dir / "before_lowres.png", np.clip(before, 0.0, 1.0))
        _save_overlay(np.asarray(merged), viz_dir / "after_lowres.png")

    if logger:
        logger.info("Streaming merge produced %d unique objects.", next_gid - 1)

    merged_arr = np.asarray(merged)

    # Clean up mem‑map.
    try:
        mm_path.unlink(missing_ok=True)
    except PermissionError:
        pass

    return merged_arr
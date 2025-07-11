"""
TILING AND MERGING UTILITIES FOR CELL SEGMENTATION.

Author: Christos Botos.
Affiliation: Institute of Molecular Biology and Biotechnology.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: tiling.py.
Description:
    Overlap‑aware image tiling, feather‑blended merging of continuous outputs (flows, probability maps) and ratio‑based fusion of instance label masks. The module also produces visual quality‑control overlays *before* and *after* mask merging so that border artefacts become immediately visible.

Dependencies:
    • Python>=3.10.
    • numpy, scipy, matplotlib.

All public helpers are unit‑tested in`tests/test_tiling.py` so that future refactors cannot silently break the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence, Tuple
from tempfile import mkstemp
import os

import matplotlib.pyplot as plt
import numpy as np

"""Feather mask utilities."""

def feather_mask(h: int, w: int, overlap: int) -> np.ndarray:
    """Generate a 2-D cosine-ramp mask that tapers from 0 at the tile edge
    to 1 in the interior.

    If *overlap* ≤ 0 the mask is uniformly 1.0, so blending reduces to a
    straight copy and no broadcasting errors can occur.
    """
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    edge_h = min(overlap, h // 2)
    edge_w = min(overlap, w // 2)

    ramp_h = np.ones(h, dtype=np.float32)
    ramp_w = np.ones(w, dtype=np.float32)

    if edge_h > 0:
        ramp_h[:edge_h]  = np.linspace(0.0, 1.0, edge_h, endpoint=False)
        ramp_h[-edge_h:] = np.linspace(1.0, 0.0, edge_h, endpoint=False)
    if edge_w > 0:
        ramp_w[:edge_w]  = np.linspace(0.0, 1.0, edge_w, endpoint=False)
        ramp_w[-edge_w:] = np.linspace(1.0, 0.0, edge_w, endpoint=False)

    return np.outer(ramp_h, ramp_w)

"""Image tiler (generator, no zero‑padding)."""

'''Image tiler.'''

def split_image_into_tiles(
    img: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int,
    logger,
    *,
    channel_axis: int | None = None,
):
    """Yield minimally sized tiles plus their placement slices.

    * No zero-padding is introduced. Border tiles take the exact size required
      to reach the image edge. Padding, if needed, belongs in the model
      pre-processor.
    * Supports 2-D, CHW and HWC tensors.  If *channel_axis* is *None* the helper
      guesses the layout (≤ 4 channels → CHW).
    """

    if img.ndim == 2:
        H, W = img.shape
        channel_axis = None
    elif img.ndim == 3:
        if channel_axis is None:
            channel_axis = 0 if img.shape[0] <= 4 else 2
        if channel_axis == 0:          # CHW layout.
            H, W = img.shape[1:]
        elif channel_axis == 2:        # HWC layout.
            H, W = img.shape[:2]
        else:
            raise ValueError("channel_axis must be 0, 2 or None.")
    else:
        raise ValueError("Input must be a 2-D or 3-D array.")

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("Overlap must be smaller than tile dimensions.")

    logger.info(
        "Tiling %dx%d image into %dx%d px tiles (overlap=%d).",
        H, W, tile_h, tile_w, overlap,
    )

    for y0 in range(0, H, stride_h):
        for x0 in range(0, W, stride_w):
            y1 = min(y0 + tile_h, H)
            x1 = min(x0 + tile_w, W)

            if channel_axis is None or channel_axis == 2:  # HW or HWC.
                tile = img[y0:y1, x0:x1]
            else:                                          # CHW.
                tile = img[:, y0:y1, x0:x1]

            yield tile, (slice(y0, y1), slice(x0, x1))


"""Feather-blend a stack of continuous tiles into one full-resolution array."""

def _memmap_array(shape, dtype, logger):
    """
    Create a temporary memmap on the OS scratch drive and return the mmap obj.
    """
    fd, path = mkstemp(prefix="iri_merge_", suffix=".dat")
    os.close(fd)                           # We only need the path.
    logger.debug(f"Allocated memmap: {path}  shape={shape}  dtype={dtype}")
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def merge_tiles_with_weighted_overlap(
    tile_stack: Sequence[np.ndarray],
    slices: Sequence[Tuple[slice, slice]],
    image_shape: Tuple[int, int],
    overlap: int,
    *,
    channel_axis: int | None = None,
    logger=None,
    debug_snap=None,
    mem_threshold_gb: float = 4.0,
) -> np.ndarray:
    """
    Feather-blend a sequence of overlapping tiles into one float32 canvas.

    Handles both greyscale (H×W) and multichannel (C×H×W or H×W×C) tiles.
    Greyscale tiles are promoted to (1,H,W) internally so broadcasting is safe.
    """
    if not tile_stack:
        return np.zeros(image_shape, dtype=np.float32)

    sample = tile_stack[0]
    H, W = image_shape

    # Detect channel axis / count.
    if sample.ndim == 2:
        channel_axis = None
        n_chan = 1
    else:
        if channel_axis is None:
            if 3 in sample.shape:
                channel_axis = sample.shape.index(3)
            elif 4 in sample.shape:
                channel_axis = sample.shape.index(4)
            else:
                channel_axis = 0
        n_chan = sample.shape[channel_axis]

    # Memory-aware allocator (RAM or memmap).
    def _allocate(shape: Tuple[int, ...]) -> np.ndarray:
        bytes_needed = np.prod(shape, dtype=np.int64) * 4   # float32
        if bytes_needed > mem_threshold_gb * (1024**3):
            return _memmap_array(shape, dtype=np.float32, logger=logger)
        return np.zeros(shape, dtype=np.float32)

    # Always work CHW internally (even greyscale → (1,H,W)).
    field   = _allocate((n_chan if n_chan > 1 else 1, H, W))
    weights = _allocate((H, W))

    def _tile_to_chw(tile: np.ndarray) -> np.ndarray:
        if tile.ndim == 2:
            return tile.astype(np.float32)[None, ...]
        if channel_axis == 0:
            return tile.astype(np.float32)
        return np.moveaxis(tile, channel_axis, 0).astype(np.float32)

    # Accumulate.
    for tile_raw, (ys, xs) in zip(tile_stack, slices):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        h, w   = y1 - y0, x1 - x0

        tile_chw = _tile_to_chw(tile_raw)
        mask = feather_mask(h, w, overlap).astype(np.float32)
        mask[mask == 0.0] = 1e-6          # avoid div-by-zero later

        field[:, y0:y1, x0:x1] += tile_chw * mask
        weights[y0:y1, x0:x1]  += mask

    field /= weights.clip(min=1e-6)[None, ...]

    if debug_snap is not None:
        debug_snap("merge_weights", weights)

    # Restore original channel layout.
    if n_chan == 1:
        return field[0]
    if channel_axis == 0:
        return field
    return np.moveaxis(field, 0, channel_axis)


'''Instance-mask fusion with QC overlays.'''

def merge_masks(
    mask_tiles: Sequence[np.ndarray],
    slices: Sequence[Tuple[slice, slice]],
    image_shape: Tuple[int, int],
    overlap: int,
    logger,
    settings: dict,
    debug_snap=None,
):
    """Fuse tiled instance masks into a global label map.

    * Each new tile label is either assigned a fresh global ID or
      merged with an existing ID when the overlap ratio exceeds
      ``settings["merge_overlap_threshold"]``.
    * Saves "before" (per-tile colours) and "after" (final labels)
      diagnostic PNGs in ``<output_dir>/tile_merge_viz/``.
    """

    logger.info("merge_masks settings: overlap_threshold=%.2f, border_rule=ON", settings["merge_overlap_threshold"])
    total_objs = sum(np.unique(t, return_counts=True)[1].size - 1 for t in mask_tiles)
    logger.info("  → total input objects across all tiles = %d", total_objs)

    H, W = image_shape
    merged = np.zeros((H, W), dtype=np.uint32)
    next_gid = 1
    thr = float(settings.get("merge_overlap_threshold", 0.3))

    out_dir = Path(settings.get("output_dir", "."))
    viz_dir = out_dir / "tile_merge_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Pre-merge overlay.
    before  = np.zeros((H, W, 3), dtype=np.float32)
    weights = np.zeros((H, W),     dtype=np.float32)
    rng = np.random.default_rng(0)

    for tile, (ys, xs) in zip(mask_tiles, slices):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        colour = rng.random(3)
        binary = (tile[: y1 - y0, : x1 - x0] > 0).astype(np.float32)

        before[y0:y1, x0:x1]  += binary[..., None] * colour
        weights[y0:y1, x0:x1] += binary

    valid = weights > 0
    before[valid] /= weights[valid][:, None]
    plt.imsave(viz_dir / "before_merge_overlay.png", np.clip(before, 0.0, 1.0))

    '''Merge criterion: overlap-gate → border rule.'''
    border_touch: dict[int, bool] = {}    # Global ID → touches true border?.

    for tile_idx, (tile, (ys, xs)) in enumerate(zip(mask_tiles, slices), 1):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        roi = merged[y0:y1, x0:x1]

        # Overlap band for Stage A (rim = 0-overlap px inside this tile).
        rim = np.zeros_like(tile, dtype=bool)
        if x0 > 0: rim[:, :overlap]  = True
        if x1 < W: rim[:, -overlap:] = True
        if y0 > 0: rim[:overlap, :]  = True
        if y1 < H: rim[-overlap:, :] = True

        # True 1-px border for Stage B.
        border = np.zeros_like(tile, dtype=bool)
        border[0, :], border[-1, :] = True, True
        border[:, 0], border[:, -1] = True, True

        for lbl in np.unique(tile):
            if lbl == 0:
                continue

            local_mask = tile == lbl
            area_local = local_mask.sum()

            '''Stage A – does the object enter the overlap band?'''
            overlap_pixels = (rim & local_mask).sum()
            if overlap_pixels < thr * area_local:
                # Overlap too small → cannot duplicate, write and continue.
                gid = next_gid
                next_gid += 1
                roi[local_mask] = gid
                border_touch[gid] = (border & local_mask).any()
                continue

            local_border = (border & local_mask).any()

            '''Stage B – compare with already placed IDs under border rule.'''
            touched = np.unique(roi[local_mask])
            touched = touched[touched > 0]          # IDs in neighbouring tiles.

            chosen_gid = None
            kill_local = False

            for gid in touched:
                area_gid   = (roi == gid).sum()
                ov_area    = np.logical_and(roi == gid, local_mask).sum()
                gid_border = border_touch.get(gid, False)

                # Asymmetric OR ratio test (same as before).
                if (ov_area / area_local) >= thr or (ov_area / area_gid) >= thr:

                    # Apply border rule 
                    if local_border ^ gid_border:          # exactly one touches
                        if local_border:
                            kill_local = True               # delete local stub
                        else:
                            merged[merged == gid] = 0       # remove border stub
                            chosen_gid = gid                # reuse ID
                            border_touch[gid] = False
                        break
                    else:                                  # both border or none
                        chosen_gid = gid
                        border_touch[gid] = local_border or gid_border
                        break
                    # 

            if kill_local:
                if logger:
                    logger.info("✂ Removed border stub (%d px) at tile %d, lbl %d.",
                                area_local, tile_idx, lbl)
                continue

            if chosen_gid is None:                         # no neighbour matched
                chosen_gid = next_gid
                next_gid += 1
                border_touch[chosen_gid] = local_border

            roi[local_mask] = chosen_gid

    # Post-merge overlay 
    cmap = rng.random((next_gid + 1, 3))
    after = cmap[merged]
    plt.imsave(viz_dir / "after_merge_overlay.png", np.clip(after, 0.0, 1.0))

    if debug_snap is not None:
        debug_snap("merged_mask", merged)

    logger.info("Mask fusion produced %d unique IDs.", next_gid - 1)

    uniq, counts = np.unique(merged, return_counts=True)
    logger.info("  → total objects after merge      = %d", uniq.size - 1)
    removed = total_objs - (uniq.size - 1)
    logger.info("  → objects removed by merge rules = %d", removed)

    return merged

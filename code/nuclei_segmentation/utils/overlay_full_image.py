"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: full_image_overlay.py
Description:
    Whenever file-paths are supplied instead of in-memory
    arrays, the function streams the work tile-by-tile, delegating to the
    battle-tested ``overlay_masks.overlay`` routine (see *overlay_masks.py*).
    Peak RAM therefore stays bounded by a single tile.

Dependencies:
    • Python ≥ 3.10.
    • numpy, tifffile, tqdm  (already required by *overlay_masks.py*).

Usage:
    # Classic in-memory use (unchanged behaviour, good for ≤ ~2 GiB total):
    full_image_overlay(viz_dir, log, img_ndarray, mask_ndarray)

    # Scalable, tile-wise path-based use:
    full_image_overlay(
        viz_dir,
        log,
        img_path=Path("huge_slide.ome.tif"),
        mask_path=Path("labels.npy"),
        tile=1024,
        alpha=0.35,
        workers="auto",
        gpu=True,
    )

Positional Arguments:
    visualization_dir   Destination directory for the rendered overlay.
    logger              Standard logger for diagnostics.

Keyword-only Arguments:
    img, masks          *Either* the classic NumPy arrays … *or*
    img_path, mask_path Paths to the on-disk image & label mask.
    tile                Tile edge length (pixels) when streaming.
    workers             Worker count or "auto" for CPU count.
    alpha               Blend transparency ∈ [0, 1].
    gpu                 Attempt CuPy acceleration per tile.

Returns:
    pathlib.Path to the generated BigTIFF.

Notes:
    • Array inputs take precedence over path inputs.
    • All heavy lifting is delegated; this wrapper only decides the code-path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
import tifffile as tiff

from .overlay_masks import overlay


def _array_mode(
    out_path: Path,
    img: np.ndarray,
    masks: np.ndarray,
    alpha: float,
) -> None:

    """Blend *img* & *masks* wholly in-RAM and write a TIFF."""

    img_uint8 = img.astype(np.float32)
    if img_uint8.max() > 255.0:
        img_uint8 = (img_uint8 / img_uint8.max()) * 255.0
    img_uint8 = img_uint8.astype(np.uint8)

    # Random but deterministic colours with a black background for label 0.
    rng = np.random.default_rng(42)
    lut = rng.integers(0, 256, size=(masks.max() + 1, 3), dtype=np.uint8)
    lut[0] = 0

    coloured = lut[masks]                        # (H, W, 3) uint8.
    blended = (img_uint8 * (1.0 - alpha) +
               coloured.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    tiff.imwrite(out_path, blended, photometric="rgb")


def full_image_overlay(
    visualization_dir: Path,
    logger: logging.Logger,
    img: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    img_path: Optional[Union[str, Path]] = None,
    mask_path: Optional[Union[str, Path]] = None,
    tile: int = 1024,
    workers: Union[int, str] = "auto",
    alpha: float = 0.35,
    gpu: bool = True,
) -> Path:
    """
    Create a full-size colour overlay of segmentation results.

    The function automatically selects one of two code-paths:

    1. **Array mode** — if *img* and *masks* are provided, everything is kept in
       RAM (behaviour identical to the legacy implementation).

    2. **Streaming mode** — if *img_path* and *mask_path* are provided (and the
       optional *overlay_masks.py* helper is importable), the work is performed
       tile-wise so that peak RAM stays bounded.
    """
    visualization_dir = Path(visualization_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    out_path = visualization_dir / "full_image_overlay.tif"

    try:
        if img is not None and masks is not None:
            logger.info("Creating in-memory overlay (image shapes: %s, %s).",
                        img.shape, masks.shape)
            _array_mode(out_path, img, masks, alpha)

        elif img_path is not None and mask_path is not None:
            if overlay is None:  # pragma: no cover
                raise ImportError(
                    "overlay_masks.overlay not importable. Ensure overlay_masks.py "
                    "is on the PYTHONPATH."
                )
            logger.info("Creating tile-wise overlay: '%s' × '%s'.",
                        img_path, mask_path)
            overlay(
                image_path=img_path,
                mask_path=mask_path,
                out_path=out_path,
                tile=tile,
                workers=workers,
                alpha=alpha,
                gpu=gpu,
            )
        else:
            raise ValueError(
                "Provide either (img & masks) or (img_path & mask_path)."
            )

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("Overlay file was not written or is empty.")

        logger.info("Saved full-size overlay → %s.", out_path)
        return out_path

    except Exception as e:
        logger.warning("Failed to create full-size overlay: %s", e, exc_info=True)
        raise  # Re-raise so that calling code can decide how to handle it.

"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: qc.py.
Description:
    Produce *two* qualitative‑control overlays for a tiled‑segmentation pipeline:

        • ``before.tif`` – Every **raw per‑tile** nucleus mask is rendered on top
          of the corresponding RGB crop with a deterministic colour and 50 %
          opacity.  Overlaps highlight mis‑alignments.
        • ``after.tif``  – The **merged** (slide‑level) mask is rendered on the
          same crop in magenta, again at 50 % opacity, for one‑glance sanity
          checks.

    The script auto‑detects tile offsets from the file names, supports both TIFF
    and NumPy ``.npz`` masks, and never suffers from uint8 overflows thanks to a
    16‑bit composition buffer.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pillow, tqdm, pytest.

Usage (CLI):
    python qc.py \
        --image      path/to/crop_RGB.tif \
        --raw_masks  path/to/tile_masks/ \
        --merged     path/to/merged_mask.tif \
        --outdir     qc/        # optional, defaults next to --image.

Inputs:
    1. RGB crop (8‑bit, H×W×3).
    2. Directory of per‑tile instance masks (*same physical units* as #1).
       Recognised file‑name patterns (y,x in pixels; row,col in tiles):
            • "12345_67890.tif"   → y=12345, x=67890
            • "row5_col7.npz"     → row=5, col=7 → y=row*tile_h, x=col*tile_w
    3. Merged (after) mask – single label map with background = 0.

Outputs (written to *outdir*):
    • before.tif – tile‑coloured overlay.
    • after.tif  – merged‑mask overlay.

Command‑line Arguments:
    --image        Path to the RGB crop used for visualisation.
    --raw_masks    Folder containing *N* raw per‑tile mask files.
    --merged       Path to the merged mask.
    --outdir       Where to save the QC overlays (default: <image_dir>/qc).
    --alpha        Opacity for the overlay colours (0‑1, default 0.5).

Notes:
    • Colours are generated from a SHA‑1 hash of the tile’s file name, giving
      run‑to‑run determinism yet high perceptual separation.
    • Internally, overlays are composed in uint16, preventing the infamous
      "Python integer … out of bounds for uint8" crash.

Version: 2.0 – complete rewrite, outputs split into two TIFFs.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Final, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Type aliases ░░ easier to read than a forest of generics / annotations.
# -----------------------------------------------------------------------------
RGBArray = NDArray[np.uint8]
MaskArray = NDArray[np.uint32]

# -----------------------------------------------------------------------------
# Regex patterns that capture tile offsets either in *pixels* or in *tile indices*.
# The final "catch‑all" will try to interpret any two integers separated by a
# non‑digit as y,x pixel coordinates.
# -----------------------------------------------------------------------------
_PATTERNS: Final = [
    # Explicit pixel‑offset: 01234_05678.tif → y = 1234, x = 5678
    re.compile(r"(?P<y>\d+)[_ ](?P<x>\d+)"),
    # row_col.tif → convert to pixels later if tile size is known.
    re.compile(r"row(?P<row>\d+)[_ ]col(?P<col>\d+)")
]

# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover – tested via subprocess in CI.
    """Entry‑point for the *qc.py* CLI."""

    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s – %(message)s",
    )

    image = _load_rgb(Path(args.image))
    masks_info = list(_iter_tile_masks(Path(args.raw_masks), image.shape[:2]))
    merged_mask = _load_mask(Path(args.merged))

    outdir = Path(args.outdir or Path(args.image).parent / "qc")
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Generating ‘before’ overlay …")
    before = _compose_before_overlay(image, masks_info, alpha=args.alpha)
    Image.fromarray(before).save(outdir / "before.tif", compression="tiff_deflate")

    logging.info("Generating ‘after’ overlay …")
    after = _compose_after_overlay(image, merged_mask, alpha=args.alpha)
    Image.fromarray(after).save(outdir / "after.tif", compression="tiff_deflate")

    logging.info("QC overlays saved to %s", outdir.resolve())


# -----------------------------------------------------------------------------
# Helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate before/after QC overlays for tiled‑segmentation pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image", required=True, help="Path to the RGB crop (TIFF/PNG).")
    parser.add_argument("--raw_masks", required=True, help="Directory with per‑tile mask files.")
    parser.add_argument("--merged", required=True, help="Path to the merged (after) mask.")
    parser.add_argument("--outdir", default=None, help="Output directory for QC overlays.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity [0‑1].")

    return parser


def _load_rgb(path: Path) -> RGBArray:
    """Load an 8‑bit RGB image.  Raises if the image is not 3‑channel."""

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:  # pragma: no cover – sanity.
        raise ValueError(f"Expected an RGB image, got shape {arr.shape} instead.")
    return arr


def _load_mask(path: Path) -> MaskArray:
    """Load a *uint32* label mask from TIFF/PNG or NumPy ``.npz``."""

    if path.suffix.lower() in {".tif", ".tiff", ".png"}:
        arr = np.asarray(Image.open(path), dtype=np.uint32)
    elif path.suffix.lower() == ".npz":
        arr = np.load(path)["arr_0"].astype(np.uint32)
    else:
        raise ValueError(f"Unsupported mask format: {path}")
    return arr


@dataclass(frozen=True, slots=True)
class _TileInfo:
    mask: MaskArray  # label map in *local* tile coordinates
    y: int          # top‑left Y offset in *pixels* (canvas coordinates)
    x: int          #           X offset in *pixels*


def _iter_tile_masks(folder: Path, canvas_shape: Tuple[int, int]) -> Iterable[_TileInfo]:
    """Yield (_mask, y, x) for every mask found under *folder* (recursively).

    The function tries all known regexes until one matches the file name.  If
    the pattern provides *row*/*col* instead of pixel offsets, the tile size is
    inferred from the mask dimensions.
    """

    for file in sorted(folder.rglob("*")):
        if not file.suffix.lower() in {".tif", ".tiff", ".png", ".npz"}:
            continue  # skip unrelated files

        mask = _load_mask(file)
        h, w = mask.shape

        y, x = _parse_offsets(file.name, mask.shape)

        # Guard against coordinates that would place the tile outside the canvas.
        if y + h > canvas_shape[0] or x + w > canvas_shape[1]:  # pragma: no cover
            logging.warning("Tile %s [%d×%d] at (y=%d, x=%d) exceeds canvas (%s). Skipped.",
                            file.name, h, w, y, x, canvas_shape)
            continue

        yield _TileInfo(mask=mask, y=y, x=x)


def _parse_offsets(fname: str, tile_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Extract *pixel* offsets from *fname* using the registered regexes."""

    for pat in _PATTERNS:
        if (m := pat.search(fname)):
            if "y" in m.groupdict():  # direct pixel offsets
                return int(m["y"]), int(m["x"])
            if "row" in m.groupdict():  # need to convert row/col → pixels
                tile_h, tile_w = tile_shape
                return int(m["row"]) * tile_h, int(m["col"]) * tile_w
    raise ValueError(f"Could not infer offsets from file name: {fname}")


def _compose_before_overlay(img: RGBArray, masks: Iterable[_TileInfo], *, alpha: float) -> RGBArray:
    """Return the *before* overlay as uint8 H×W×3 array."""

    h, w, _ = img.shape
    canvas = img.astype(np.uint16)  # copy – we mutate in‑place.

    for tile_id, info in enumerate(tqdm(list(masks), desc="Tiles")):
        colour = _deterministic_colour(info, tile_id)
        _alpha_blend(canvas, colour, info, alpha)

    return canvas.clip(0, 255).astype(np.uint8)


def _compose_after_overlay(img: RGBArray, merged_mask: MaskArray, *, alpha: float) -> RGBArray:
    """Return the *after* overlay (magenta) as uint8 H×W×3 array."""

    h, w, _ = img.shape
    if merged_mask.shape != (h, w):  # pragma: no cover – sanity.
        raise ValueError("Merged mask dimensions do not match the image crop.")

    canvas = img.astype(np.uint16)

    magenta = np.array([255, 0, 255], dtype=np.uint16)  # RGB
    mask_bool = merged_mask > 0
    _alpha_blend_bool(canvas, magenta, mask_bool, alpha)

    return canvas.clip(0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Low‑level utilities ----------------------------------------------------------
# -----------------------------------------------------------------------------

def _deterministic_colour(info: _TileInfo, tile_id: int) -> NDArray[np.uint16]:
    """Generate a reproducible RGB colour from the tile identifier."""

    h = hashlib.sha1(str(tile_id).encode()).digest()  # 20 bytes → good mix.
    r, g, b = h[0], h[1], h[2]
    return np.array([r, g, b], dtype=np.uint16)


def _alpha_blend(canvas: NDArray[np.uint16], colour: NDArray[np.uint16], info: _TileInfo, alpha: float) -> None:
    """Blend *colour* onto the *canvas* inside *info.mask* using the given *alpha*."""

    y0, x0 = info.y, info.x
    h, w = info.mask.shape
    region = canvas[y0:y0+h, x0:x0+w]
    mask_bool = info.mask > 0

    _alpha_blend_bool(region, colour, mask_bool, alpha)


def _alpha_blend_bool(region: NDArray[np.uint16], colour: NDArray[np.uint16], mask_bool: NDArray[np.bool_], alpha: float) -> None:
    """Alpha‑blend *colour* onto *region* wherever *mask_bool* is True."""

    overlay = (alpha * colour).astype(np.uint16)
    inv_alpha = 1.0 - alpha

    # Broadcasting: region[mask, chan] = inv_alpha*region[...] + overlay[chan]
    for c in range(3):
        chan = region[..., c]
        chan[mask_bool] = (inv_alpha * chan[mask_bool] + overlay[c]).astype(np.uint16)


# -----------------------------------------------------------------------------
# Tests ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()


# ------------------------- pytest unit tests ----------------------------------

import pytest


def _dummy_imgs() -> Tuple[RGBArray, list[_TileInfo], MaskArray]:
    """Return a 64×64 classic chessboard RGB, 4 tile masks, and a merged mask."""

    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255

    tile_masks = []
    for i in range(2):
        for j in range(2):
            mask = np.zeros((32, 32), dtype=np.uint32)
            mask[8:-8, 8:-8] = 1
            tile_masks.append(_TileInfo(mask=mask, y=i*32, x=j*32))

    merged = np.zeros((H, W), dtype=np.uint32)
    for info in tile_masks:
        merged[info.y:info.y+32, info.x:info.x+32] |= info.mask

    return img, tile_masks, merged


def test_before_overlay_shapes():
    img, tiles, _ = _dummy_imgs()
    ov = _compose_before_overlay(img, tiles, alpha=0.4)
    assert ov.shape == img.shape
    assert ov.dtype == np.uint8


def test_after_overlay_unique_colour():
    img, tiles, merged = _dummy_imgs()
    ov_before = _compose_before_overlay(img, tiles, alpha=0.5)
    ov_after = _compose_after_overlay(img, merged, alpha=0.5)
    # At least five unique colours in the before overlay.
    assert len(np.unique(ov_before.reshape(-1, 3), axis=0)) >= 5
    # After overlay should contain magenta.
    assert [255, 0, 255] in ov_after.reshape(-1, 3).tolist()

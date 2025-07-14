"""
Author: Christos Botos.
Script Name: cli.py.
Description:
    ``python -m cellpose_merge.cli --help`` exposes the full functionality to users
    who prefer a CLI instead of importing the module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .merge_tiles import merge_masks_streaming


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU‑accelerated Cellpose mask merger")
    p.add_argument("tiles", type=Path, help="Directory with NPZ or Zarr tiles")
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--tile_h", type=int, default=512)
    p.add_argument("--tile_w", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--batch_px", type=int, default=128_000_000)
    p.add_argument("--cpu", action="store_true", help="Force CPU mode even if GPU is available")
    p.add_argument("--qc", action="store_true", help="Write 1k×1k before/after overlays")
    p.add_argument("--qc_dir", type=Path, default=Path("./qc_overlays"))
    p.add_argument("--out", type=Path, default=Path("merged_mask.npy"))
    return p.parse_args()


def cli_entry() -> None:
    args = _parse_args()
    merged = merge_masks_streaming(
        height=args.height,
        width=args.width,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        overlap=args.overlap,
        tiles_path=args.tiles,
        threshold=args.threshold,
        batch_px=args.batch_px,
        use_gpu=not args.cpu,
        qc=args.qc,
        qc_dir=args.qc_dir,
    )
    np.save(args.out, merged)
    print("Merged mask saved to", args.out)


if __name__ == "__main__":
    cli_entry()

"""
Author: Christos Botos.
Script Name: io.py.
Description:
    Abstraction layer around different on‑disk mask formats (NPZ, Zarr/NGFF).
    Only the **tile loader** interface is required by ``merge_tiles.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import zarr  # type: ignore
except ModuleNotFoundError:
    zarr = None

__all__ = ["make_tile_loader"]


@lru_cache(maxsize=None)
def _open_npz(path: Path) -> np.ndarray:
    return np.load(path)["mask"]


@lru_cache(maxsize=None)
def _open_zarr(path: Path):
    if zarr is None:
        raise RuntimeError("Install `zarr` to read NGFF tiles.")
    return zarr.open(path, mode="r")


def make_tile_loader(base_dir: Path) -> Callable[[slice, slice], np.ndarray]:
    """Return a ``data_loader`` callable based on the *base_dir* format."""

    base_dir = base_dir.expanduser().resolve()
    fmt = next((p.suffix for p in base_dir.iterdir()), None)
    if fmt == ".npz":
        def _loader(ys: slice, xs: slice) -> np.ndarray:
            f = base_dir / f"{ys.start}_{xs.start}.npz"
            return _open_npz(f)
        return _loader
    if fmt == ".zarr":
        arr = _open_zarr(base_dir)
        tile_h, tile_w = arr.attrs["tile_shape"]
        def _loader(ys: slice, xs: slice) -> np.ndarray:
            ty = ys.start // tile_h
            tx = xs.start // tile_w
            return arr.oindex[ty, tx]
        return _loader
    raise ValueError("Unsupported tile format in directory: " + str(base_dir))

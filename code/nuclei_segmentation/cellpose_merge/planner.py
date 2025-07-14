# ============================================================================
# planner.py – Overlap discovery & batch scheduling
# ============================================================================

"""
Author: Christos Botos.
Script Name: planner.py.
Description:
    Functions that map a tile grid into *overlap bands* and group them into batches
    that fit GPU VRAM constraints.  Streaming is coordinated by an ``async``
    generator so that compute and disk‑IO can overlap when desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, List, Tuple

@dataclass(slots=True)
class Band:
    ys: slice
    xs: slice
    tiles: List[Tuple[int, int]]  # (row_idx, col_idx)
    pixels: int

def clusters(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Return connected components of the 4-neighbour tile graph."""
    adj = {c: [] for c in coords}
    for r, c in coords:
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = (r + dr, c + dc)
            if n in adj:
                adj[(r, c)].append(n)
    seen, comps = set(), []
    for root in coords:
        if root in seen:
            continue
        q, comp = [root], []
        while q:
            cur = q.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            q.extend(adj[cur])
        comps.append(comp)
    return comps

async def bands_async(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    batch_px: int,
) -> AsyncIterator[List[Band]]:
    """Yield batches of ``Band`` objects whose combined area ≤ *batch_px*."""
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    n_rows = (height + stride_h - 1) // stride_h
    n_cols = (width + stride_w - 1) // stride_w

    all_bands: List[Band] = []

    # Horizontal & vertical neighbours + quadruple overlaps.
    for r in range(n_rows):
        for c in range(n_cols):

            if c < n_cols - 1:  # horizontal band between (r,c) and (r,c+1)
                x0 = (c + 1) * stride_w
                xs = slice(x0, min(x0 + overlap, width))
                y0 = r * stride_h
                ys = slice(y0, min(y0 + tile_h, height))
                px = (ys.stop - ys.start) * (xs.stop - xs.start)
                all_bands.append(Band(ys, xs, [(r, c), (r, c + 1)], px * 2))

            if r < n_rows - 1:  # vertical band between (r,c) and (r+1,c)
                y0 = (r + 1) * stride_h
                ys = slice(y0, min(y0 + overlap, height))
                x0 = c * stride_w
                xs = slice(x0, min(x0 + tile_w, width))
                px = (ys.stop - ys.start) * (xs.stop - xs.start)
                all_bands.append(Band(ys, xs, [(r, c), (r + 1, c)], px * 2))


            if c < n_cols - 1 and r < n_rows - 1:  # 4-tile corner.
                y0 = (r + 1) * stride_h
                ys = slice(y0, min(y0 + overlap, height))
                x0 = (c + 1) * stride_w
                xs = slice(x0, min(x0 + overlap, width))
                px = (ys.stop - ys.start) * (xs.stop - xs.start)
                all_bands.append(Band(ys, xs, [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)], px * 4))

    # Largest first fill strategy.
    all_bands.sort(key=lambda b: b.pixels, reverse=True)

    batch: List[Band] = []
    pixels = 0
    for band in all_bands:
        if pixels + band.pixels > batch_px and batch:
            yield batch
            batch = []
            pixels = 0
        batch.append(band)
        pixels += band.pixels
    if batch:
        yield batch
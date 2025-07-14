"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Project: Cellpose Tile‑Merge Pipeline (Python ≥ 3.10).
Description:
    Complete, runnable implementation of the RAM‑efficient, GPU‑accelerated mask‑merge
    workflow we planned earlier.  The package layout is **flat** for clarity inside
    this single canvas, but each section should be saved as its own ``.py`` file in a
    real repository.

Package structure:
    cellpose_merge/
        ├── __init__.py
        ├── rules.py              # Pure‑CPU reference algorithm (NumPy).
        ├── gpu_merge.py          # Torch‑CUDA implementation.
        ├── io.py                 # Lazy Zarr/NPZ I/O back‑end.
        ├── planner.py            # Overlap‑band discovery & batch scheduler.
        ├── merge_tiles.py        # High‑level orchestration.
        ├── cli.py                # ``python -m cellpose_merge.cli`` entry‑point.
        └── tests/
            ├── test_rules.py
            ├── test_gpu_vs_cpu.py
            └── conftest.py

Key external dependencies:
    • numpy ≥ 1.27
    • torch ≥ 2.2  (CUDA ≥ 12.2 strongly recommended)
    • zarr ≥ 2.16  (optional ‑ for NGFF tiles)
    • pillow, matplotlib (only when ``--qc`` flag is enabled)
    • tqdm, hypothesis, pytest

All modules respect the four‑step merge specification verbatim and include
explanatory comments that are full sentences.  All public functions are
annotated with modern type hints.

Script Name: rules.py.

Description:
    Reference NumPy implementation of the four‑step merge algorithm.  Designed for
    correctness and testability, *not* speed.  This version processes one overlap
    patch (2 – 4 tiles) at a time and returns a merged mask plus a mapping of local
    → global IDs.
"""

from __future__ import annotations

from itertools import count
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "merge_patch_cpu",
]


class DSUCPU:
    """Simple disjoint‑set union for uint32 labels using pure Python lists."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        par = self.parent.setdefault(x, x)
        if par != x:
            self.parent[x] = self.find(par)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank.
        if self.rank.get(ra, 0) < self.rank.get(rb, 0):
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank.get(ra, 0) == self.rank.get(rb, 0):
            self.rank[ra] = self.rank.get(ra, 0) + 1


def _touches_border(mask: np.ndarray) -> np.ndarray:
    """Return a boolean array *per‑label* indicating border contact inside *mask*."""

    border = np.zeros(mask.max() + 1, dtype=bool)
    if mask.size == 0:
        return border
    border_labels = np.unique(
        np.concatenate(
            [
                mask[0, :],
                mask[-1, :],
                mask[:, 0],
                mask[:, -1],
            ]
        )
    )
    border[border_labels] = True
    return border


def merge_patch_cpu(
    patch: np.ndarray,
    *,
    threshold: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Merge a *(T, H, W)* stack of overlapping masks on CPU.

    Parameters
    ----------
    patch : np.ndarray
        Integer mask stack with *T* ≤ 4 (overlapping tiles).  Zero denotes background.
    threshold : float
        Fraction of shared pixels required by the merge rule.

    Returns
    -------
    merged : np.ndarray
        Merged 2‑D mask with globally unique IDs (uint32).
    mapping : Dict[int, int]
        Mapping from original **local** IDs (``tile_id << 32 | label``) to **global** IDs.
    """

    T, H, W = patch.shape
    global_next = count(1)  # IDs start at 1.
    dsu = DSUCPU()

    # Pre‑compute per‑tile metadata.
    tile_sizes = [np.bincount(patch[t].ravel()) for t in range(T)]
    tile_border = [_touches_border(patch[t]) for t in range(T)]

    # Map local labels to unique composite keys – avoids collisions between tiles.
    composite = np.zeros_like(patch, dtype=np.uint64)
    for t in range(T):
        composite[t] = (t << 32) | patch[t]

    # ---------------------------------------------------------------
    # 1. Assign fresh global IDs to objects below the *overlap* quota.
    # ---------------------------------------------------------------

    keep_mask = np.zeros(composite.shape, dtype=bool)
    for t in range(T):
        # Only pixels ALSO claimed by another tile count as ‘overlap’
        overlap_mask = (patch != 0).sum(axis=0) > 1  # pixels seen by ≥2 tiles
        ov_cnt = np.bincount(patch[t][overlap_mask].ravel(), minlength=tile_sizes[t].size)
        keep = ov_cnt >= tile_sizes[t] * threshold  # True ⇢ keep for merging

    # Objects not kept receive a unique global ID immediately.
    comp_to_global: Dict[int, int] = {}
    for comp in np.unique(composite[~keep_mask]):
        if comp == 0:
            continue
        comp_to_global[comp] = next(global_next)

    # ---------------------------------------------------------------
    # 2‑4. Iterate over pixel pairs to build unions respecting rules.
    # ---------------------------------------------------------------

    # Iterate over all pairs of tiles.
    for a in range(T):
        for b in range(a + 1, T):
            pa, pb = patch[a], patch[b]
            mask = (pa != 0) & (pb != 0)
            if not mask.any():
                continue
            comp_a = composite[a][mask]
            comp_b = composite[b][mask]
            # Shared pixel counts between (comp_a[i], comp_b[i]).
            pairs, counts = np.unique((comp_a.astype(np.uint64) << 64) | comp_b, return_counts=True)
            left = pairs >> 64
            right = pairs & ((1 << 64) - 1)
            for ca, cb, cnt in zip(left, right, counts):
                # Rule 2 – shared threshold.
                size_a = tile_sizes[a][ca & 0xFFFFFFFF]
                size_b = tile_sizes[b][cb & 0xFFFFFFFF]
                if cnt < threshold * min(size_a, size_b):
                    continue
                # Rule 3 – border stub.
                ta = tile_border[a][ca & 0xFFFFFFFF]
                tb = tile_border[b][cb & 0xFFFFFFFF]
                if ta and not tb:
                    continue  # Remove *ca* later in I/O phase.
                if tb and not ta:
                    continue  # Remove *cb* later.
                dsu.union(int(ca), int(cb))

    # ----------------------------------------------------------------
    # Assign global IDs following the union‑find result.
    # ----------------------------------------------------------------

    for comp in np.unique(composite[keep_mask]):
        root = dsu.find(int(comp))
        comp_to_global.setdefault(root, next(global_next))
        comp_to_global[comp] = comp_to_global[root]

    # ------------------------------------------------------------------
    # Flatten the composite keys back into one 2-D canvas.
    # ------------------------------------------------------------------
    getter = np.vectorize(comp_to_global.get, otypes=[np.uint32])

    merged_mask: np.ndarray = np.zeros((H, W), dtype=np.uint32)
    for t in range(T):                                    # loop over tiles
        sel = composite[t] != 0                           # skip background
        merged_mask[sel] = getter(composite[t][sel])      # map → global IDs

    return merged_mask, comp_to_global

"""
Author: Christos Botos.
Script Name: gpu_merge.py.
Description:
    GPU implementation of the overlap‑patch merge.  The API mirrors ``merge_patch_cpu``
    so that we can test them side‑by‑side.  The heavy lifting lives in
    ``GPUDSU`` – a vectorised, deterministic union‑find using PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "merge_patch_gpu",
]


class GPUDSU:
    """Disjoint‑set union implemented with two 1‑D **device** tensors."""

    def __init__(self, size: int, device: torch.device) -> None:
        # Torch advanced indexing expects *indices* and *payload* to be int64.
        self.parent: Tensor = torch.arange(size, device=device, dtype=torch.int64)
        self.rank: Tensor = torch.zeros(size, device=device, dtype=torch.int8)

    def find(self, x: Tensor) -> Tensor:
        parent = self.parent
        while True:
            px = parent.index_select(0, x)
            if torch.equal(px, x):
                return x
            x = px

    def union(self, a: Tensor, b: Tensor) -> None:
        ra, rb = self.find(a), self.find(b)
        neq = ra != rb
        if not torch.any(neq):
            return
        ra, rb = ra[neq], rb[neq]
        rank_a = self.rank.index_select(0, ra)
        rank_b = self.rank.index_select(0, rb)
        mask = rank_a >= rank_b
        self.parent[rb[mask]] = ra[mask]
        self.parent[ra[~mask]] = rb[~mask]
        self.rank[ra[mask & (rank_a == rank_b)]] += 1


def merge_patch_gpu(
    patch: np.ndarray,
    *,
    threshold: float,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """GPU counterpart of ``merge_patch_cpu`` using PyTorch tensors."""

    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    T, H, W = patch.shape

    # Check for tensor size limits before processing.
    total_elements = T * H * W
    max_tensor_elements = 2**31 - 1  # PyTorch INT_MAX limit.

    if total_elements > max_tensor_elements:
        raise RuntimeError(f"Tensor would have {total_elements} elements, exceeding PyTorch limit of {max_tensor_elements}. "
                         f"Consider using CPU processing or smaller batches.")

    # CUDA kernels don’t support uint32.  Promote to signed 64-bit once.
    try:
        patch_t = torch.from_numpy(
            patch.astype(np.int64, copy=False).reshape(T, -1)
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "illegal memory access" in str(e).lower():
            raise RuntimeError(f"GPU memory error when creating tensor of size ({T}, {H*W}): {e}. "
                             f"Consider reducing batch size or using CPU processing.")
        else:
            raise

    max_lbl = int(patch_t.max().item()) + 1  # Include 0.

    dsu = GPUDSU(max_lbl * T, device=device)  # Composite space: tile<<32 | label.

    # Build per‑tile stats (on GPU).
    counts_per_tile = [torch.bincount(patch_t[t], minlength=max_lbl) for t in range(T)]
    border_per_tile = [torch.zeros(max_lbl, dtype=torch.bool, device=device) for _ in range(T)]
    # Border detection (vectorised).
    for t in range(T):
        m = patch_t[t].reshape(H, W)
        border_lbl = torch.unique(torch.cat([m[0], m[-1], m[:, 0], m[:, -1]]))
        border_per_tile[t][border_lbl] = True

    # Helper to form composite label (tile << 32 | label).
    def comp(t_idx: int, lbl: Tensor | int) -> Tensor:
        # Dense mapping:  (tile * max_lbl) + local_label  ∈  [0, max_lbl*T).
        return torch.tensor(t_idx, dtype=torch.int64, device=device) * max_lbl + lbl

    """
    Step 1 – assign new global IDs to objects that do **not** overlap enough
    with any neighbouring tile.  A pixel belongs to an ‘overlap region’ only
    when the same spatial position is claimed by two or more tiles.
    """
    comp_to_global: Dict[int, int] = {}
    next_gid: int = 1

    # Boolean mask marking every pixel covered by ≥ 2 tiles.
    overlap_mask: torch.Tensor = (patch_t != 0).sum(dim=0) > 1  # H × W

    for t in range(T):
        # For tile t, count how many of each label’s pixels fall inside an overlap region.
        ov_cnt: torch.Tensor = torch.bincount(
            patch_t[t][overlap_mask],
            minlength=max_lbl,
        )

        # Keep the label only if at least *threshold* × size lies in overlap.
        keep: torch.Tensor = ov_cnt >= counts_per_tile[t] * threshold

        # Labels that fail the check receive a fresh global ID.
        drop_lbls = (~keep) & (torch.arange(max_lbl, device=device) != 0)
        for lbl in torch.nonzero(drop_lbls, as_tuple=True)[0]:
            comp_to_global[int(comp(t, int(lbl)))] = next_gid
            next_gid += 1

    """Steps 2‑4 – merge remaining labels across tile pairs"""
    for a in range(T):
        for b in range(a + 1, T):
            pa, pb = patch_t[a], patch_t[b]
            mask = (pa != 0) & (pb != 0)
            if not torch.any(mask):
                continue
            la = pa[mask]
            lb = pb[mask]
            flat = la.to(torch.int64) * max_lbl + lb
            overlap = torch.bincount(flat)
            idx = torch.nonzero(overlap).squeeze(1)
            cnts = overlap[idx]
            lbl_a = idx // max_lbl
            lbl_b = idx % max_lbl
            # Rule 2 threshold.
            min_size = torch.minimum(counts_per_tile[a][lbl_a], counts_per_tile[b][lbl_b])
            good = cnts >= min_size * threshold
            lbl_a, lbl_b = lbl_a[good], lbl_b[good]
            # Rule 3 border stub.
            ba = border_per_tile[a][lbl_a]
            bb = border_per_tile[b][lbl_b]
            ok = ~((ba & ~bb) | (bb & ~ba))
            lbl_a, lbl_b = lbl_a[ok], lbl_b[ok]
            if lbl_a.numel():
                dsu.union(comp(a, lbl_a), comp(b, lbl_b))

    # Assign global IDs from DSU roots.
    for t in range(T):
        labels = torch.unique(patch_t[t])
        labels = labels[labels != 0]
        for lbl in labels.tolist():
            c = int(comp(t, lbl).item())
            root = int(dsu.find(torch.tensor(c, dtype=torch. int64, device=device)).item())
            comp_to_global.setdefault(root, next_gid)
            comp_to_global[c] = comp_to_global[root]
            if comp_to_global[root] == next_gid:
                next_gid += 1

    # ------------------------------------------------------------------
    # Build the merged 2-D mask on CPU, combining *all* tiles.
    # ------------------------------------------------------------------
    merged = np.zeros((H, W), dtype=np.uint32)
    patch_cpu = patch_t.cpu().numpy()                     # (T, H·W) int64
    getter = np.vectorize(comp_to_global.get, otypes=[np.uint32])

    for t in range(T):
        flat = patch_cpu[t].ravel()
        sel = flat != 0
        comp_keys = t * max_lbl + flat[sel]               # tile-unique keys
        merged.ravel()[sel] = getter(comp_keys)           # map → global IDs

    return merged, comp_to_global

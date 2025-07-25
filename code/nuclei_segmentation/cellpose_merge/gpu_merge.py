"""
Author: Christos Botos.
Script Name: gpu_merge.py.
Description:
    GPU implementation of the overlap‑patch merge.  The API mirrors ``merge_tiles_cpu_3step``
    so that we can test them side‑by‑side.  The heavy lifting lives in
    ``GPUDSU`` – a vectorised, deterministic union‑find using PyTorch tensors.
"""

from __future__ import annotations

from .rules import merge_tiles_cpu_3step

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "merge_patch_gpu_3step",
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


def merge_patch_gpu_3step(
    patch: np.ndarray,
    tile1_border_nuclei: set = None,
    tile2_border_nuclei: set = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    GPU implementation of the new 3-step merge algorithm.

    This function implements the simplified 3-step merging rule on GPU:
    1. Priority Selection: Tile with most nuclei gets priority, if they are equal the first one is chosen.
    2. Border Deletion: Delete all priority tile masks that touch the border of the
       priority tile, while preserving all non-priority masks that touch the priority
       tile border.
    3. Cleanup: Remove remaining non-priority nuclei in overlap region.

    Parameters
    ----------
    patch : np.ndarray
        Integer mask stack with T ≤ 4 (overlapping tiles). Zero denotes background.

    Returns
    -------
    merged : np.ndarray
        Merged 2D mask with globally unique IDs (uint32).
    mapping : Dict[int, int]
        Mapping from original local IDs to global IDs.
    """
    import logging

    # TODO: Implement proper GPU 3-step merge algorithm.
    # For now, this function is a placeholder that should not be called directly.
    # The actual merging is handled by the two-phase merge system using CPU implementation.

    logging.warning("merge_patch_gpu_3step called directly - this function is not yet implemented")
    logging.warning("Use the two-phase merge system instead, which handles CPU/GPU fallback properly")
    raise NotImplementedError("GPU 3-step merge is not yet implemented")

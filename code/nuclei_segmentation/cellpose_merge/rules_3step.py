"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: rules_3step.py.
Description:
    Reference NumPy implementation of the new 3-step merge algorithm for kidney
    I/R injury tissue analysis. This simplified approach replaces the previous
    4-step method with a more efficient and scientifically intuitive priority-based
    merging strategy.
    
    The new 3-step merging rule:
    1. Priority Selection: When two overlapping tiles are detected, the tile with
       the most nuclei gets priority.
    2. Border Deletion: Delete all priority tile masks that touch the border of the
       priority tile, while preserving all non-priority masks that touch the priority
       tile border.
    3. Cleanup: Delete all remaining non-priority masks in the overlapping region
       (except the preserved border-touching ones).

Dependencies:
    • Python ≥ 3.10.
    • numpy for core array operations.
    • logging for scientific workflow tracking.

Key Features:
    • Simplified 3-step algorithm for better performance and clarity.
    • Priority-based approach ensures better nucleus preservation.
    • Memory-efficient processing with reduced intermediate data structures.
    • Comprehensive error handling for bioinformatics workflows.
    • Scientific context-aware logging for kidney tissue analysis.

Notes:
    • This implementation is optimized for kidney I/R injury tissue segmentation.
    • The priority-based approach reduces over-segmentation artifacts.
    • Designed for correctness and testability, with GPU acceleration available separately.
"""

from __future__ import annotations

import logging
from itertools import count
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "merge_patch_cpu_3step",
]


def _count_nuclei_in_tile(tile_mask: NDArray[np.uint32]) -> int:
    """
    Count the number of unique nuclei in a tile mask.
    
    Parameters
    ----------
    tile_mask : NDArray[np.uint32]
        Tile mask with nucleus labels (background = 0).
        
    Returns
    -------
    int
        Number of unique nuclei in the tile.
    """
    unique_labels = np.unique(tile_mask)
    # Exclude background (label 0).
    return len(unique_labels[unique_labels > 0])


def _find_border_touching_nuclei(tile_mask: NDArray[np.uint32]) -> set:
    """
    Find all nucleus labels that touch the border of a tile.

    Parameters
    ----------
    tile_mask : NDArray[np.uint32]
        Tile mask with nucleus labels.

    Returns
    -------
    set
        Set of nucleus labels that touch the tile border.
    """
    if tile_mask.size == 0:
        return set()

    border_labels = set()
    h, w = tile_mask.shape

    # Check all four borders.
    if h > 0 and w > 0:
        # Top and bottom borders.
        border_labels.update(np.unique(tile_mask[0, :]))
        border_labels.update(np.unique(tile_mask[-1, :]))
        # Left and right borders.
        border_labels.update(np.unique(tile_mask[:, 0]))
        border_labels.update(np.unique(tile_mask[:, -1]))

    # Remove background label.
    border_labels.discard(0)
    return border_labels


def _find_nuclei_touching_priority_border(non_priority_tile: NDArray[np.uint32], priority_tile: NDArray[np.uint32]) -> set:
    """
    Find all nucleus labels from non-priority tile that touch the priority tile's border.

    This identifies cross-boundary nuclei that span from non-priority tiles into
    the priority tile region.

    Parameters
    ----------
    non_priority_tile : NDArray[np.uint32]
        Non-priority tile mask with nucleus labels.
    priority_tile : NDArray[np.uint32]
        Priority tile mask with nucleus labels.

    Returns
    -------
    set
        Set of nucleus labels from non-priority tile that touch priority tile borders.
    """
    if non_priority_tile.size == 0 or priority_tile.size == 0:
        return set()

    # Get priority tile border positions.
    h, w = priority_tile.shape
    priority_border_mask = np.zeros((h, w), dtype=bool)

    if h > 0 and w > 0:
        # Mark all priority tile border positions.
        priority_border_mask[0, :] = True   # Top border.
        priority_border_mask[-1, :] = True  # Bottom border.
        priority_border_mask[:, 0] = True   # Left border.
        priority_border_mask[:, -1] = True  # Right border.

    # Find non-priority nuclei that overlap with priority tile border positions.
    cross_boundary_labels = set()

    for label in np.unique(non_priority_tile[non_priority_tile > 0]):
        nucleus_mask = non_priority_tile == label

        # Check if this nucleus overlaps with any priority tile border position.
        if np.any(nucleus_mask & priority_border_mask):
            cross_boundary_labels.add(label)

    return cross_boundary_labels


def merge_patch_cpu_3step(
    patch: NDArray[np.uint32],
) -> Tuple[NDArray[np.uint32], Dict[int, int]]:
    """
    Merge a (T, H, W) stack of overlapping masks using the new 3-step algorithm.
    
    This function implements the simplified 3-step merging rule:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching priority borders,
       preserve non-priority nuclei touching priority borders
    3. Cleanup: Remove remaining non-priority nuclei in overlap region
    
    Parameters
    ----------
    patch : NDArray[np.uint32]
        Integer mask stack with T ≤ 4 (overlapping tiles). Zero denotes background.
        
    Returns
    -------
    merged : NDArray[np.uint32]
        Merged 2D mask with globally unique IDs (uint32).
    mapping : Dict[int, int]
        Mapping from original local IDs to global IDs.
    """
    T, H, W = patch.shape
    
    # Safety check for large patches.
    total_elements = T * H * W
    max_cpu_elements = 2**28  # 256M elements = ~1GB for uint32.
    
    if total_elements > max_cpu_elements:
        raise RuntimeError(f"CPU patch would have {total_elements} elements "
                         f"({total_elements * 4 / (1024**3):.2f} GB), exceeding safe CPU limit "
                         f"of {max_cpu_elements} elements ({max_cpu_elements * 4 / (1024**3):.2f} GB). "
                         f"Patch shape: ({T}, {H}, {W}). "
                         f"Consider using incremental processing instead.")
    
    logging.debug(f"3-step CPU merge processing patch: ({T}, {H}, {W}), "
                 f"memory estimate: {total_elements * 4 / (1024**3):.2f} GB")
    
    # Handle single tile case.
    if T == 1:
        merged = patch[0].copy()
        # Create simple mapping for single tile.
        unique_labels = np.unique(merged[merged > 0])
        mapping = {int(label): int(label) for label in unique_labels}
        return merged, mapping
    
    # Step 1: Priority Selection - Find tile with most nuclei.
    nuclei_counts = [_count_nuclei_in_tile(patch[t]) for t in range(T)]
    priority_tile_idx = np.argmax(nuclei_counts)

    logging.debug(f"Nuclei counts per tile: {nuclei_counts}")
    logging.debug(f"Priority tile: {priority_tile_idx} with {nuclei_counts[priority_tile_idx]} nuclei")

    # Initialize merged mask as empty - we'll build it step by step.
    H, W = patch.shape[1], patch.shape[2]
    merged = np.zeros((H, W), dtype=np.uint32)
    global_next = count(1)
    mapping = {}

    priority_tile = patch[priority_tile_idx]

    # Step 2: Priority Border Deletion
    # Delete ALL masks from priority tile that touch ANY border of the priority tile.
    priority_border_nuclei = _find_border_touching_nuclei(priority_tile)

    logging.debug(f"Priority tile border-touching nuclei (will be deleted): {priority_border_nuclei}")

    # Add only non-border-touching nuclei from priority tile.
    for label in np.unique(priority_tile[priority_tile > 0]):
        if label not in priority_border_nuclei:
            # This nucleus doesn't touch the priority tile border, so keep it.
            nucleus_mask = priority_tile == label
            global_label = next(global_next)
            mapping[int(label)] = global_label
            merged[nucleus_mask] = global_label
            logging.debug(f"Kept priority internal nucleus: {label} -> {global_label}")
        else:
            logging.debug(f"Deleted priority border nucleus: {label}")

    # Step 3: Non-Priority Preservation Rule
    # For each non-priority tile: keep masks that touch priority tile border, delete others.
    for t in range(T):
        if t == priority_tile_idx:
            continue

        non_priority_tile = patch[t]

        # Find non-priority nuclei that touch the PRIORITY TILE'S border.
        cross_boundary_nuclei = _find_nuclei_touching_priority_border(non_priority_tile, priority_tile)

        logging.debug(f"Processing non-priority tile {t} with {nuclei_counts[t]} nuclei")
        logging.debug(f"Non-priority nuclei touching priority border (cross-boundary): {cross_boundary_nuclei}")

        # Process each nucleus in the non-priority tile.
        for label in np.unique(non_priority_tile[non_priority_tile > 0]):
            nucleus_mask = non_priority_tile == label

            if label in cross_boundary_nuclei:
                # This nucleus touches the priority tile border - KEEP it (cross-boundary).
                global_label = next(global_next)
                mapping[int(label)] = global_label
                merged[nucleus_mask] = global_label
                logging.debug(f"Kept cross-boundary nucleus: {label} -> {global_label}")
            else:
                # This nucleus doesn't touch the priority tile border - DELETE it.
                logging.debug(f"Deleted non-priority nucleus (not cross-boundary): {label}")
    
    final_nuclei_count = len(np.unique(merged[merged > 0]))
    logging.debug(f"3-step merge completed: {final_nuclei_count} nuclei in final mask")
    
    return merged, mapping

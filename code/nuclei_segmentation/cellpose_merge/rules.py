"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: rules.py.
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
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "merge_tiles_cpu_3step",
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


def _find_border_touching_nuclei(
    tile_mask: NDArray[np.uint32],
    overlap_length: int,
    direction: str
) -> Tuple[set, set]:
    """
    Find nucleus labels that touch an internal overlap boundary and those in the overlap region.

    This function returns two sets of nuclei:
    1. Nuclei that touch the boundary line (with ±1 pixel buffer)
    2. Nuclei that are completely beyond the boundary line in the overlap region

    Parameters
    ----------
    tile_mask : NDArray[np.uint32]
        Tile mask with nucleus labels.
    overlap_length : int, default 0
        Overlap distance in pixels. When 0, boundary line is at tile border.
    direction : str
        Direction of overlap boundary to check ('right', 'left', 'up', 'down').
        Always required.

    Returns
    -------
    Tuple[set, set]
        First set: nucleus labels that touch the boundary line (with ±1 buffer).
        Second set: nucleus labels completely beyond the boundary line in overlap region.

    Notes
    -----
    Boundary line positions:
    - 'right': vertical line at column (width - overlap_length)
    - 'left': vertical line at column overlap_length
    - 'up': horizontal line at row overlap_length
    - 'down': horizontal line at row (height - overlap_length)

    When overlap_length=0:
    - 'right': line at column width (right border)
    - 'left': line at column 0 (left border)
    - 'up': line at row 0 (top border)
    - 'down': line at row height (bottom border)

    Detection includes ±1 pixel buffer around the boundary line for touching nuclei.
    """
    if tile_mask.size == 0:
        return set(), set()

    h, w = tile_mask.shape
    boundary_touching_nuclei = set()
    overlap_region_nuclei = set()

    if h <= 0 or w <= 0:
        return set(), set()

    # Calculate boundary line position based on direction.
    if direction == 'right':
        # Vertical line at column (width - overlap_length).
        line_pos = w - overlap_length

        # Find nuclei touching the boundary line (with ±1 buffer).
        if overlap_length == 0:
            cols_to_check = [c for c in [w-2, w-1] if 0 <= c < w]  # Right border and one pixel left.
        else:
            if line_pos < 0 or line_pos >= w:
                return set(), set()
            cols_to_check = [c for c in [line_pos-1, line_pos, line_pos+1] if 0 <= c < w]

        for col in cols_to_check:
            boundary_touching_nuclei.update(np.unique(tile_mask[:, col]))

        # Find nuclei completely beyond the boundary line (in overlap region).
        # These are nuclei that exist entirely beyond the boundary (not touching it).
        if overlap_length > 0 and line_pos < w:
            beyond_cols = list(range(line_pos + 2, w))  # Columns well beyond the boundary line (skip buffer zone).
            for col in beyond_cols:
                if col < w:
                    overlap_region_nuclei.update(np.unique(tile_mask[:, col]))

    elif direction == 'left':
        # Vertical line at column overlap_length.
        line_pos = overlap_length

        # Find nuclei touching the boundary line (with ±1 buffer).
        if overlap_length == 0:
            cols_to_check = [c for c in [0, 1] if 0 <= c < w]  # Left border and one pixel right.
        else:
            if line_pos < 0 or line_pos >= w:
                return set(), set()
            cols_to_check = [c for c in [line_pos-1, line_pos, line_pos+1] if 0 <= c < w]

        for col in cols_to_check:
            boundary_touching_nuclei.update(np.unique(tile_mask[:, col]))

        # Find nuclei completely beyond the boundary line (in overlap region).
        # These are nuclei that exist entirely beyond the boundary (not touching it).
        if overlap_length > 0 and line_pos > 1:
            beyond_cols = list(range(0, line_pos - 2))  # Columns well before the boundary line (skip buffer zone).
            for col in beyond_cols:
                if col >= 0:
                    overlap_region_nuclei.update(np.unique(tile_mask[:, col]))

    elif direction == 'up':
        # Horizontal line at row overlap_length.
        line_pos = overlap_length

        # Find nuclei touching the boundary line (with ±1 buffer).
        if overlap_length == 0:
            rows_to_check = [r for r in [0, 1] if 0 <= r < h]  # Top border and one pixel down.
        else:
            if line_pos < 0 or line_pos >= h:
                return set(), set()
            rows_to_check = [r for r in [line_pos-1, line_pos, line_pos+1] if 0 <= r < h]

        for row in rows_to_check:
            boundary_touching_nuclei.update(np.unique(tile_mask[row, :]))

        # Find nuclei completely beyond the boundary line (in overlap region).
        # These are nuclei that exist entirely beyond the boundary (not touching it).
        if overlap_length > 0 and line_pos > 1:
            beyond_rows = list(range(0, line_pos - 2))  # Rows well before the boundary line (skip buffer zone).
            for row in beyond_rows:
                if row >= 0:
                    overlap_region_nuclei.update(np.unique(tile_mask[row, :]))

    elif direction == 'down':
        # Horizontal line at row (height - overlap_length).
        line_pos = h - overlap_length

        # Find nuclei touching the boundary line (with ±1 buffer).
        if overlap_length == 0:
            rows_to_check = [r for r in [h-2, h-1] if 0 <= r < h]  # Bottom border and one pixel up.
        else:
            if line_pos < 0 or line_pos >= h:
                return set(), set()
            rows_to_check = [r for r in [line_pos-1, line_pos, line_pos+1] if 0 <= r < h]

        for row in rows_to_check:
            boundary_touching_nuclei.update(np.unique(tile_mask[row, :]))

        # Find nuclei completely beyond the boundary line (in overlap region).
        # These are nuclei that exist entirely beyond the boundary (not touching it).
        if overlap_length > 0 and line_pos < h:
            beyond_rows = list(range(line_pos + 2, h))  # Rows well beyond the boundary line (skip buffer zone).
            for row in beyond_rows:
                if row < h:
                    overlap_region_nuclei.update(np.unique(tile_mask[row, :]))
    else:
        raise ValueError(f"Invalid direction '{direction}'. Must be 'right', 'left', 'up', or 'down'.")

    # Remove background label from both sets.
    boundary_touching_nuclei.discard(0)
    overlap_region_nuclei.discard(0)

    return boundary_touching_nuclei, overlap_region_nuclei


def merge_tiles_cpu_3step(
    tile1_path: Union[str, Path],
    tile2_path: Union[str, Path],
    overlap_length: int,
    tile_relationship: str,
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]:
    """
    Enhanced 3-step merging function that properly utilizes internal boundary detection.

    This function implements the complete 3-step merging algorithm using the enhanced
    _find_border_touching_nuclei function to properly identify nuclei extending into
    overlap regions. Critical for accurate kidney I/R injury spatial analysis.

    Parameters
    ----------
    tile1_path : Union[str, Path]
        Path to the first whole tile mask .npz file.
    tile2_path : Union[str, Path]
        Path to the second whole tile mask .npz file.
    overlap_length : int
        Overlap distance in pixels between the tiles.
    tile_relationship : str
        Spatial relationship between tiles. Must be one of:
        - "tile1_above_tile2": tile1 is positioned above tile2
        - "tile1_left_of_tile2": tile1 is positioned to the left of tile2
        - "tile1_below_tile2": tile1 is positioned below tile2
        - "tile1_right_of_tile2": tile1 is positioned to the right of tile2

    Returns
    -------
    Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]
        Updated tile1_mask, updated tile2_mask, and mapping of preserved nucleus IDs.
        The mapping shows which original IDs were preserved or reassigned.

    Notes
    -----
    The 3-step algorithm:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching tile borders
    3. Cross-boundary Preservation: Preserve non-priority nuclei extending into overlap
    4. Cleanup: Remove remaining non-priority nuclei in overlap region
    5. Directional Expansion: Expand preserved cross-boundary nuclei for better coverage

    The key insight is using overlap_length=None for priority tile border detection
    and overlap_length=actual_overlap with appropriate direction for non-priority
    tile internal boundary detection.
    """
    # Convert paths to Path objects.
    tile1_path = Path(tile1_path)
    tile2_path = Path(tile2_path)

    # Validate inputs.
    if not tile1_path.exists():
        raise FileNotFoundError(f"Tile1 mask file not found: {tile1_path}")
    if not tile2_path.exists():
        raise FileNotFoundError(f"Tile2 mask file not found: {tile2_path}")

    valid_relationships = {
        "tile1_above_tile2", "tile1_left_of_tile2",
        "tile1_below_tile2", "tile1_right_of_tile2"
    }
    if tile_relationship not in valid_relationships:
        raise ValueError(f"Invalid tile_relationship '{tile_relationship}'. "
                        f"Must be one of: {valid_relationships}")

    if overlap_length <= 0:
        raise ValueError(f"overlap_length must be positive, got {overlap_length}")

    logging.info(f"Enhanced 3-step merge: {tile1_path.name} and {tile2_path.name}")
    logging.info(f"Relationship: {tile_relationship}, overlap: {overlap_length} pixels")

    # Step 1: Load complete tile masks.
    logging.debug("Loading complete tile masks...")
    tile1_data = np.load(tile1_path)
    tile2_data = np.load(tile2_path)

    tile1_mask = tile1_data["mask"].astype(np.uint32)
    tile2_mask = tile2_data["mask"].astype(np.uint32)

    logging.debug(f"Tile1 shape: {tile1_mask.shape}")
    logging.debug(f"Tile2 shape: {tile2_mask.shape}")

    # Step 2: Priority Selection based on nucleus count.
    tile1_nuclei_count = _count_nuclei_in_tile(tile1_mask)
    tile2_nuclei_count = _count_nuclei_in_tile(tile2_mask)

    logging.info(f"Tile1 nuclei count: {tile1_nuclei_count}")
    logging.info(f"Tile2 nuclei count: {tile2_nuclei_count}")

    if tile1_nuclei_count >= tile2_nuclei_count:
        priority_tile_mask = tile1_mask
        non_priority_tile_mask = tile2_mask
        priority_is_tile1 = True
        logging.info("Priority: Tile1 (higher nucleus count)")
    else:
        priority_tile_mask = tile2_mask
        non_priority_tile_mask = tile1_mask
        priority_is_tile1 = False
        logging.info("Priority: Tile2 (higher nucleus count)")

    # Step 3: Determine directions for internal boundary detection.
    # Map tile relationships to boundary directions.
    direction_mapping = {
        "tile1_above_tile2": {
            "tile1_direction": "down",   # tile1 nuclei extending down into overlap
            "tile2_direction": "up"      # tile2 nuclei extending up into overlap
        },
        "tile1_left_of_tile2": {
            "tile1_direction": "right",  # tile1 nuclei extending right into overlap
            "tile2_direction": "left"    # tile2 nuclei extending left into overlap
        },
        "tile1_below_tile2": {
            "tile1_direction": "up",     # tile1 nuclei extending up into overlap
            "tile2_direction": "down"    # tile2 nuclei extending down into overlap
        },
        "tile1_right_of_tile2": {
            "tile1_direction": "left",   # tile1 nuclei extending left into overlap
            "tile2_direction": "right"   # tile2 nuclei extending right into overlap
        }
    }

    directions = direction_mapping[tile_relationship]

    # Step 4: Border Detection Calls.
    # Priority tile: Use overlap_length=0 with the appropriate direction to find tile border nuclei.
    if priority_is_tile1:
        priority_direction = directions["tile1_direction"]
        non_priority_direction = directions["tile2_direction"]
    else:
        priority_direction = directions["tile2_direction"]
        non_priority_direction = directions["tile1_direction"]

    # Priority tile: Find nuclei touching the specific border direction.
    priority_border_nuclei, _ = _find_border_touching_nuclei(priority_tile_mask,
                                                             0,
                                                             priority_direction)
    logging.info(f"Priority tile border-touching nuclei: {len(priority_border_nuclei)} nuclei")
    logging.debug(f"Priority border nuclei IDs: {priority_border_nuclei}")

    # Non-priority tile: Find nuclei extending into overlap region.
    non_priority_boundary_nuclei, non_priority_overlap_nuclei = _find_border_touching_nuclei(
        non_priority_tile_mask,
        overlap_length,
        non_priority_direction
    )

    # Combine boundary-touching and overlap region nuclei for cross-boundary detection.
    all_non_priority_overlap_nuclei = non_priority_boundary_nuclei.union(non_priority_overlap_nuclei)

    logging.info(f"Non-priority nuclei touching boundary: {len(non_priority_boundary_nuclei)} nuclei")
    logging.info(f"Non-priority nuclei in overlap region: {len(non_priority_overlap_nuclei)} nuclei")
    logging.info(f"Total non-priority overlap nuclei: {len(all_non_priority_overlap_nuclei)} nuclei")
    logging.debug(f"Non-priority boundary nuclei IDs: {non_priority_boundary_nuclei}")
    logging.debug(f"Non-priority overlap region nuclei IDs: {non_priority_overlap_nuclei}")

    # Step 5: Apply 3-step merging rules.
    # Create working copies of the tile masks.
    updated_tile1_mask = tile1_mask.copy()
    updated_tile2_mask = tile2_mask.copy()
    mapping = {}

    # Step 5a: Border Deletion - Remove priority tile nuclei touching the tile border on that direction.
    priority_deleted_count = 0
    for nucleus_id in priority_border_nuclei:
        if priority_is_tile1:
            nucleus_mask = updated_tile1_mask == nucleus_id
            updated_tile1_mask[nucleus_mask] = 0
        else:
            nucleus_mask = updated_tile2_mask == nucleus_id
            updated_tile2_mask[nucleus_mask] = 0
        priority_deleted_count += 1
        logging.debug(f"STEP 2 DELETE: Priority nucleus {nucleus_id} (touches tile border)")

    logging.info(f"STEP 2 SUMMARY: Deleted {priority_deleted_count} priority border-touching nuclei")

    # Step 5b: Cross-boundary Preservation - Preserve non-priority nuclei extending into overlap.
    # These are nuclei that touch the boundary line (cross-boundary nuclei).
    cross_boundary_preserved_count = 0
    for nucleus_id in non_priority_boundary_nuclei:
        # These nuclei are preserved (no deletion), just log for tracking.
        mapping[nucleus_id] = nucleus_id  # Preserve original ID.
        cross_boundary_preserved_count += 1
        logging.debug(f"STEP 3 PRESERVE: Cross-boundary nucleus {nucleus_id} (extends into overlap)")

    logging.info(f"STEP 3 SUMMARY: Preserved {cross_boundary_preserved_count} cross-boundary nuclei")

    # Step 5c: Cleanup - Delete non-priority nuclei in overlap region that are NOT cross-boundary.
    # These are nuclei completely in the overlap region that don't extend from the non-priority tile.
    cleanup_deleted_count = 0
    for nucleus_id in non_priority_overlap_nuclei:
        # Only delete if this nucleus is NOT a cross-boundary nucleus.
        if nucleus_id not in non_priority_boundary_nuclei:
            if priority_is_tile1:
                # Non-priority is tile2, delete from tile2.
                nucleus_mask = updated_tile2_mask == nucleus_id
                updated_tile2_mask[nucleus_mask] = 0
            else:
                # Non-priority is tile1, delete from tile1.
                nucleus_mask = updated_tile1_mask == nucleus_id
                updated_tile1_mask[nucleus_mask] = 0
            cleanup_deleted_count += 1
            logging.debug(f"STEP 4 DELETE: Non-priority nucleus {nucleus_id} (in overlap region, not cross-boundary)")

    logging.info(f"STEP 4 SUMMARY: Deleted {cleanup_deleted_count} non-priority nuclei in overlap region")

    # Step 7: Create Debug Visualization.
    logging.info("STEP 5: Creating comprehensive debug visualization")
    try:
        _create_debug_visualization(
            original_tile1=tile1_mask,
            original_tile2=tile2_mask,
            final_tile1=updated_tile1_mask,
            final_tile2=updated_tile2_mask,
            cross_boundary_nuclei=non_priority_boundary_nuclei,
            expansion_mask=expansion_mask,
            tile1_path=tile1_path,
            tile2_path=tile2_path,
            tile_relationship=tile_relationship
        )
    except Exception as e:
        logging.warning(f"Debug visualization failed: {e}")

    # Final logging with expansion statistics.
    final_tile1_nuclei = len(np.unique(updated_tile1_mask[updated_tile1_mask > 0]))
    final_tile2_nuclei = len(np.unique(updated_tile2_mask[updated_tile2_mask > 0]))

    logging.info(f"Enhanced 3-step merge with gap-filling expansion completed:")
    logging.info(f"  Tile1: {tile1_nuclei_count} -> {final_tile1_nuclei} nuclei")
    logging.info(f"  Tile2: {tile2_nuclei_count} -> {final_tile2_nuclei} nuclei")
    logging.info(f"  Cross-boundary nuclei preserved: {len(mapping)}")

    return updated_tile1_mask, updated_tile2_mask, mapping
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


# def _expand_masks_directionally(
#     masks1: NDArray[np.uint32],
#     masks2: NDArray[np.uint32],
#     direction: str,
#     expansion_pixels: int = 6,
#     target_nuclei: set = None
# ) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
#     """
#     Expand nucleus masks directionally with gap-filling algorithm for improved spatial coverage.
#
#     This function performs controlled directional expansion of nucleus masks with an advanced
#     gap-filling strategy. When expansion encounters another nucleus, the gap is split evenly
#     between both nuclei, creating more natural boundary transitions in kidney I/R injury
#     tissue analysis.
#
#     The enhanced expansion algorithm:
#     1. Identifies target nucleus labels in masks1 (or all if target_nuclei is None)
#     2. For each target nucleus, expands row by row (or column by column) in the specified direction
#     3. When encountering another nucleus, calculates the gap distance
#     4. Fills the gap by splitting space evenly between both nuclei
#     5. Tracks all newly added pixels for visualization and validation
#
#     Parameters
#     ----------
#     masks1 : NDArray[np.uint32]
#         Primary mask array containing nuclei to be expanded.
#     masks2 : NDArray[np.uint32]
#         Secondary mask array used for conflict detection during expansion.
#     direction : str
#         Direction of expansion. Must be one of: 'up', 'down', 'left', 'right'.
#     expansion_pixels : int, default 6
#         Maximum number of pixels to expand in the specified direction.
#     target_nuclei : set, optional
#         Set of nucleus IDs to expand. If None, expands all nuclei in masks1.
#
#     Returns
#     -------
#     Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]
#         - Updated masks1 with expanded nuclei
#         - Updated masks2 with gap-filled nuclei
#         - Binary mask showing all newly added pixels during expansion
#
#     Raises
#     ------
#     ValueError
#         If direction is not one of the valid options or expansion_pixels is negative.
#
#     Notes
#     -----
#     The gap-filling approach creates more natural boundaries by:
#     - Preventing abrupt stops when nuclei encounter obstacles
#     - Ensuring both nuclei benefit from boundary optimization
#     - Maintaining spatial continuity in tissue morphometry analysis
#     """
#     if masks1.size == 0 or masks2.size == 0:
#         logging.debug("Empty input masks provided to expansion function")
#         return masks1.copy(), masks2.copy(), np.zeros_like(masks1, dtype=np.uint8)
#
#     if expansion_pixels <= 0:
#         raise ValueError(f"expansion_pixels must be positive, got {expansion_pixels}")
#
#     valid_directions = {'up', 'down', 'left', 'right'}
#     if direction not in valid_directions:
#         raise ValueError(f"Invalid direction '{direction}'. Must be one of: {valid_directions}")
#
#     # Create working copies and expansion tracking mask.
#     expanded_masks1 = masks1.copy()
#     expanded_masks2 = masks2.copy()
#     expansion_mask = np.zeros_like(masks1, dtype=np.uint8)
#     h, w = expanded_masks1.shape
#
#     # Get nucleus labels to expand.
#     if target_nuclei is not None:
#         # Only expand specified target nuclei.
#         all_labels = np.unique(masks1[masks1 > 0])
#         unique_labels = [label for label in all_labels if label in target_nuclei]
#         logging.debug(f"Expanding {len(unique_labels)} target nuclei {direction} with gap-filling algorithm")
#     else:
#         # Expand all nuclei (excluding background).
#         unique_labels = np.unique(masks1[masks1 > 0])
#         logging.debug(f"Expanding {len(unique_labels)} nuclei {direction} with gap-filling algorithm")
#
#     if len(unique_labels) == 0:
#         logging.debug("No nuclei found for expansion")
#         return expanded_masks1, expanded_masks2, expansion_mask
#
#     # Track expansion statistics.
#     total_pixels_expanded = 0
#     total_gaps_filled = 0
#     nuclei_expanded_count = 0
#
#     # Process each nucleus individually.
#     for nucleus_id in unique_labels:
#         nucleus_mask = (masks1 == nucleus_id)
#         if not np.any(nucleus_mask):
#             continue
#
#         nucleus_expanded = False
#         pixels_expanded_this_nucleus = 0
#         gaps_filled_this_nucleus = 0
#
#         if direction == 'up':
#             # Expand upward row by row with gap-filling.
#             for col in range(w):
#                 col_mask = nucleus_mask[:, col]
#                 if np.any(col_mask):
#                     top_row = np.where(col_mask)[0][0]
#
#                     # Find the nearest obstacle in this column.
#                     obstacle_row = -1
#                     for check_row in range(top_row - 1, max(-1, top_row - expansion_pixels - 1), -1):
#                         if expanded_masks2[check_row, col] > 0:
#                             obstacle_row = check_row
#                             break
#
#                     if obstacle_row >= 0:
#                         # Calculate gap and fill evenly.
#                         gap_size = top_row - obstacle_row - 1
#                         if gap_size > 0:
#                             half_gap = gap_size // 2
#
#                             # Expand nucleus_id upward by half the gap.
#                             for step in range(1, half_gap + 1):
#                                 new_row = top_row - step
#                                 if new_row >= 0:
#                                     expanded_masks1[new_row, col] = nucleus_id
#                                     expansion_mask[new_row, col] = 1
#                                     pixels_expanded_this_nucleus += 1
#                                     nucleus_expanded = True
#
#                             # Expand obstacle nucleus downward by remaining gap.
#                             obstacle_id = expanded_masks2[obstacle_row, col]
#                             for step in range(gap_size - half_gap):
#                                 new_row = obstacle_row + step + 1
#                                 if new_row < h and new_row < top_row:
#                                     expanded_masks2[new_row, col] = obstacle_id
#                                     expansion_mask[new_row, col] = 1
#
#                             gaps_filled_this_nucleus += 1
#                     else:
#                         # No obstacle found, expand normally up to max distance.
#                         for step in range(1, expansion_pixels + 1):
#                             new_row = top_row - step
#                             if new_row < 0:
#                                 break
#                             if expanded_masks1[new_row, col] == 0 and expanded_masks2[new_row, col] == 0:
#                                 expanded_masks1[new_row, col] = nucleus_id
#                                 expansion_mask[new_row, col] = 1
#                                 pixels_expanded_this_nucleus += 1
#                                 nucleus_expanded = True
#                             else:
#                                 break
#
#         elif direction == 'down':
#             # Expand downward row by row with gap-filling.
#             for col in range(w):
#                 col_mask = nucleus_mask[:, col]
#                 if np.any(col_mask):
#                     bottom_row = np.where(col_mask)[0][-1]
#
#                     # Find the nearest obstacle in this column.
#                     obstacle_row = h
#                     for check_row in range(bottom_row + 1, min(h, bottom_row + expansion_pixels + 1)):
#                         if expanded_masks2[check_row, col] > 0:
#                             obstacle_row = check_row
#                             break
#
#                     if obstacle_row < h:
#                         # Calculate gap and fill evenly.
#                         gap_size = obstacle_row - bottom_row - 1
#                         if gap_size > 0:
#                             half_gap = gap_size // 2
#
#                             # Expand nucleus_id downward by half the gap.
#                             for step in range(1, half_gap + 1):
#                                 new_row = bottom_row + step
#                                 if new_row < h:
#                                     expanded_masks1[new_row, col] = nucleus_id
#                                     expansion_mask[new_row, col] = 1
#                                     pixels_expanded_this_nucleus += 1
#                                     nucleus_expanded = True
#
#                             # Expand obstacle nucleus upward by remaining gap.
#                             obstacle_id = expanded_masks2[obstacle_row, col]
#                             for step in range(gap_size - half_gap):
#                                 new_row = obstacle_row - step - 1
#                                 if new_row >= 0 and new_row > bottom_row:
#                                     expanded_masks2[new_row, col] = obstacle_id
#                                     expansion_mask[new_row, col] = 1
#
#                             gaps_filled_this_nucleus += 1
#                     else:
#                         # No obstacle found, expand normally up to max distance.
#                         for step in range(1, expansion_pixels + 1):
#                             new_row = bottom_row + step
#                             if new_row >= h:
#                                 break
#                             if expanded_masks1[new_row, col] == 0 and expanded_masks2[new_row, col] == 0:
#                                 expanded_masks1[new_row, col] = nucleus_id
#                                 expansion_mask[new_row, col] = 1
#                                 pixels_expanded_this_nucleus += 1
#                                 nucleus_expanded = True
#                             else:
#                                 break
#
#         elif direction == 'left':
#             # Expand leftward column by column with gap-filling.
#             for row in range(h):
#                 row_mask = nucleus_mask[row, :]
#                 if np.any(row_mask):
#                     left_col = np.where(row_mask)[0][0]
#
#                     # Find the nearest obstacle in this row.
#                     obstacle_col = -1
#                     for check_col in range(left_col - 1, max(-1, left_col - expansion_pixels - 1), -1):
#                         if expanded_masks2[row, check_col] > 0:
#                             obstacle_col = check_col
#                             break
#
#                     if obstacle_col >= 0:
#                         # Calculate gap and fill evenly.
#                         gap_size = left_col - obstacle_col - 1
#                         if gap_size > 0:
#                             half_gap = gap_size // 2
#
#                             # Expand nucleus_id leftward by half the gap.
#                             for step in range(1, half_gap + 1):
#                                 new_col = left_col - step
#                                 if new_col >= 0:
#                                     expanded_masks1[row, new_col] = nucleus_id
#                                     expansion_mask[row, new_col] = 1
#                                     pixels_expanded_this_nucleus += 1
#                                     nucleus_expanded = True
#
#                             # Expand obstacle nucleus rightward by remaining gap.
#                             obstacle_id = expanded_masks2[row, obstacle_col]
#                             for step in range(gap_size - half_gap):
#                                 new_col = obstacle_col + step + 1
#                                 if new_col < w and new_col < left_col:
#                                     expanded_masks2[row, new_col] = obstacle_id
#                                     expansion_mask[row, new_col] = 1
#
#                             gaps_filled_this_nucleus += 1
#                     else:
#                         # No obstacle found, expand normally up to max distance.
#                         for step in range(1, expansion_pixels + 1):
#                             new_col = left_col - step
#                             if new_col < 0:
#                                 break
#                             if expanded_masks1[row, new_col] == 0 and expanded_masks2[row, new_col] == 0:
#                                 expanded_masks1[row, new_col] = nucleus_id
#                                 expansion_mask[row, new_col] = 1
#                                 pixels_expanded_this_nucleus += 1
#                                 nucleus_expanded = True
#                             else:
#                                 break
#
#         elif direction == 'right':
#             # Expand rightward column by column with gap-filling.
#             for row in range(h):
#                 row_mask = nucleus_mask[row, :]
#                 if np.any(row_mask):
#                     right_col = np.where(row_mask)[0][-1]
#
#                     # Find the nearest obstacle in this row.
#                     obstacle_col = w
#                     for check_col in range(right_col + 1, min(w, right_col + expansion_pixels + 1)):
#                         if expanded_masks2[row, check_col] > 0:
#                             obstacle_col = check_col
#                             break
#
#                     if obstacle_col < w:
#                         # Calculate gap and fill evenly.
#                         gap_size = obstacle_col - right_col - 1
#                         if gap_size > 0:
#                             half_gap = gap_size // 2
#
#                             # Expand nucleus_id rightward by half the gap.
#                             for step in range(1, half_gap + 1):
#                                 new_col = right_col + step
#                                 if new_col < w:
#                                     expanded_masks1[row, new_col] = nucleus_id
#                                     expansion_mask[row, new_col] = 1
#                                     pixels_expanded_this_nucleus += 1
#                                     nucleus_expanded = True
#
#                             # Expand obstacle nucleus leftward by remaining gap.
#                             obstacle_id = expanded_masks2[row, obstacle_col]
#                             for step in range(gap_size - half_gap):
#                                 new_col = obstacle_col - step - 1
#                                 if new_col >= 0 and new_col > right_col:
#                                     expanded_masks2[row, new_col] = obstacle_id
#                                     expansion_mask[row, new_col] = 1
#
#                             gaps_filled_this_nucleus += 1
#                     else:
#                         # No obstacle found, expand normally up to max distance.
#                         for step in range(1, expansion_pixels + 1):
#                             new_col = right_col + step
#                             if new_col >= w:
#                                 break
#                             if expanded_masks1[row, new_col] == 0 and expanded_masks2[row, new_col] == 0:
#                                 expanded_masks1[row, new_col] = nucleus_id
#                                 expansion_mask[row, new_col] = 1
#                                 pixels_expanded_this_nucleus += 1
#                                 nucleus_expanded = True
#                             else:
#                                 break
#
#         if nucleus_expanded:
#             nuclei_expanded_count += 1
#             total_pixels_expanded += pixels_expanded_this_nucleus
#             total_gaps_filled += gaps_filled_this_nucleus
#             logging.debug(f"Nucleus {nucleus_id}: expanded by {pixels_expanded_this_nucleus} pixels, "
#                          f"filled {gaps_filled_this_nucleus} gaps {direction}")
#
#     logging.info(f"Gap-filling expansion completed: {nuclei_expanded_count}/{len(unique_labels)} nuclei expanded")
#     logging.info(f"Total pixels expanded {direction}: {total_pixels_expanded}")
#     logging.info(f"Total gaps filled: {total_gaps_filled}")
#
#     return expanded_masks1, expanded_masks2, expansion_mask


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

    # Step 6: Directional Expansion with Gap-Filling - Expand preserved cross-boundary nuclei.
    # This step helps recover nucleus boundaries that may have been truncated during tiling.
    logging.info("STEP 6: Applying directional expansion with gap-filling to preserved cross-boundary nuclei")

    expansion_mask = np.zeros_like(updated_tile1_mask, dtype=np.uint8)

    # if len(non_priority_boundary_nuclei) > 0:
    #     # Determine expansion direction based on tile relationship.
    #     # Cross-boundary nuclei should expand toward the priority tile.
    #     expansion_direction_mapping = {
    #         "tile1_above_tile2": "down" if not priority_is_tile1 else "up",
    #         "tile1_left_of_tile2": "right" if not priority_is_tile1 else "left",
    #         "tile1_below_tile2": "up" if not priority_is_tile1 else "down",
    #         "tile1_right_of_tile2": "left" if not priority_is_tile1 else "right"
    #     }
    #
    #     expansion_direction = expansion_direction_mapping[tile_relationship]
    #
    #     # Apply gap-filling expansion directly to the appropriate tile masks.
    #     if priority_is_tile1:
    #         # Non-priority is tile2, expand cross-boundary nuclei in tile2.
    #         logging.debug(f"Expanding tile2 nuclei {expansion_direction} toward tile1 with gap-filling")
    #         updated_tile2_mask, updated_tile1_mask, expansion_mask = _expand_masks_directionally(
    #             updated_tile2_mask,
    #             updated_tile1_mask,
    #             expansion_direction,
    #             expansion_pixels=6,
    #             target_nuclei=non_priority_boundary_nuclei
    #         )
    #     else:
    #         # Non-priority is tile1, expand cross-boundary nuclei in tile1.
    #         logging.debug(f"Expanding tile1 nuclei {expansion_direction} toward tile2 with gap-filling")
    #         updated_tile1_mask, updated_tile2_mask, expansion_mask = _expand_masks_directionally(
    #             updated_tile1_mask,
    #             updated_tile2_mask,
    #             expansion_direction,
    #             expansion_pixels=6,
    #             target_nuclei=non_priority_boundary_nuclei
    #         )
    #
    #     logging.info(f"STEP 6 SUMMARY: Gap-filling expansion applied {expansion_direction} to "
    #                 f"{len(non_priority_boundary_nuclei)} cross-boundary nuclei")
    #     logging.info(f"Total expansion pixels added: {np.sum(expansion_mask)}")
    # else:
    #     logging.info("STEP 6 SUMMARY: No cross-boundary nuclei to expand")
    #     expansion_direction = "none"

    # Step 7: Create Debug Visualization.
    logging.info("STEP 7: Creating comprehensive debug visualization")
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
    # logging.info(f"  Expansion direction applied: {expansion_direction}")
    # logging.info(f"  Total expansion pixels: {np.sum(expansion_mask)}")

    return updated_tile1_mask, updated_tile2_mask, mapping


def _create_debug_visualization(
    original_tile1: NDArray[np.uint32],
    original_tile2: NDArray[np.uint32],
    final_tile1: NDArray[np.uint32],
    final_tile2: NDArray[np.uint32],
    cross_boundary_nuclei: set,
    expansion_mask: NDArray[np.uint8],
    tile1_path: Path,
    tile2_path: Path,
    tile_relationship: str
) -> None:
    """
    Create comprehensive debug visualization of the 3-step merging process.

    This function generates a color-coded debug image that clearly shows the results
    of each step in the merging algorithm, enabling validation and optimization of
    the nucleus boundary handling in kidney I/R injury tissue analysis.

    Color coding:
    - Green pixels: Preserved cross-boundary nuclei (non-priority nuclei kept in Step 3)
    - Red pixels: All other nuclei (priority nuclei and non-cross-boundary nuclei)
    - Purple pixels: Newly expanded pixels added during directional expansion (Step 6)
    - Black pixels: Background regions

    Parameters
    ----------
    original_tile1, original_tile2 : NDArray[np.uint32]
        Original tile masks before merging.
    final_tile1, final_tile2 : NDArray[np.uint32]
        Final tile masks after complete 3-step merging.
    cross_boundary_nuclei : set
        Set of nucleus IDs that were preserved as cross-boundary nuclei.
    expansion_mask : NDArray[np.uint8]
        Binary mask showing pixels added during directional expansion.
    tile1_path, tile2_path : Path
        Paths to the original tile files for naming the debug output.
    tile_relationship : str
        Spatial relationship between tiles for debug filename.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Create combined visualization canvas.
    h1, w1 = final_tile1.shape
    h2, w2 = final_tile2.shape
    max_h = max(h1, h2)
    combined_w = w1 + w2 + 20  # Add gap between tiles

    # Initialize RGB debug image.
    debug_image = np.zeros((max_h, combined_w, 3), dtype=np.uint8)

    # Process tile1 (left side).
    for nucleus_id in np.unique(final_tile1[final_tile1 > 0]):
        nucleus_pixels = (final_tile1 == nucleus_id)

        if nucleus_id in cross_boundary_nuclei:
            # Green for preserved cross-boundary nuclei.
            debug_image[:h1, :w1, 0][nucleus_pixels] = 0    # R
            debug_image[:h1, :w1, 1][nucleus_pixels] = 255  # G
            debug_image[:h1, :w1, 2][nucleus_pixels] = 0    # B
        else:
            # Red for all other nuclei.
            debug_image[:h1, :w1, 0][nucleus_pixels] = 255  # R
            debug_image[:h1, :w1, 1][nucleus_pixels] = 0    # G
            debug_image[:h1, :w1, 2][nucleus_pixels] = 0    # B

    # Process tile2 (right side).
    tile2_offset = w1 + 20
    for nucleus_id in np.unique(final_tile2[final_tile2 > 0]):
        nucleus_pixels = (final_tile2 == nucleus_id)
        shifted_pixels = np.zeros((max_h, combined_w), dtype=bool)
        shifted_pixels[:h2, tile2_offset:tile2_offset+w2] = nucleus_pixels

        if nucleus_id in cross_boundary_nuclei:
            # Green for preserved cross-boundary nuclei.
            debug_image[shifted_pixels, 0] = 0    # R
            debug_image[shifted_pixels, 1] = 255  # G
            debug_image[shifted_pixels, 2] = 0    # B
        else:
            # Red for all other nuclei.
            debug_image[shifted_pixels, 0] = 255  # R
            debug_image[shifted_pixels, 1] = 0    # G
            debug_image[shifted_pixels, 2] = 0    # B

    # Overlay expansion pixels in purple.
    # Show all expansion pixels - the expansion algorithm now only expands cross-boundary nuclei.
    exp_h, exp_w = expansion_mask.shape

    # Handle potential dimension mismatches between expansion_mask and tiles.
    tile1_exp_h = min(h1, exp_h)
    tile1_exp_w = min(w1, exp_w)
    tile2_exp_h = min(h2, exp_h)
    tile2_exp_w = min(w2, exp_w)

    # Since we now only expand cross-boundary nuclei, all expansion pixels are valid.
    expansion_pixels_tile1 = (expansion_mask[:tile1_exp_h, :tile1_exp_w] > 0)
    expansion_pixels_tile2 = (expansion_mask[:tile2_exp_h, :tile2_exp_w] > 0)

    # Purple overlay for tile1 expansion pixels.
    debug_image[:tile1_exp_h, :tile1_exp_w, 0][expansion_pixels_tile1] = 128  # R
    debug_image[:tile1_exp_h, :tile1_exp_w, 1][expansion_pixels_tile1] = 0    # G
    debug_image[:tile1_exp_h, :tile1_exp_w, 2][expansion_pixels_tile1] = 128  # B

    # Purple overlay for tile2 expansion pixels (shifted).
    shifted_expansion = np.zeros((max_h, combined_w), dtype=bool)
    shifted_expansion[:tile2_exp_h, tile2_offset:tile2_offset+tile2_exp_w] = expansion_pixels_tile2
    debug_image[shifted_expansion, 0] = 128  # R
    debug_image[shifted_expansion, 1] = 0    # G
    debug_image[shifted_expansion, 2] = 128  # B

    # Create debug output filename.
    tile1_name = tile1_path.stem
    tile2_name = tile2_path.stem
    debug_filename = f"debug_merge_{tile1_name}_{tile2_name}_{tile_relationship}.png"
    debug_path = tile1_path.parent / "debug_visualizations" / debug_filename

    # Ensure debug directory exists.
    debug_path.parent.mkdir(exist_ok=True)

    # Save debug visualization.
    plt.figure(figsize=(15, 8))
    plt.imshow(debug_image)
    plt.title(f"3-Step Merge Debug: {tile_relationship}\n"
              f"Green=Cross-boundary nuclei, Red=Other nuclei, Purple=Expanded pixels")
    plt.axis('off')

    # Add legend.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Cross-boundary nuclei (preserved)'),
        Patch(facecolor='red', label='Other nuclei'),
        Patch(facecolor='purple', label='Expanded pixels'),
        Patch(facecolor='black', label='Background')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Debug visualization saved: {debug_path}")

    # Also save a detailed statistics file.
    stats_path = debug_path.with_suffix('.txt')
    with open(stats_path, 'w') as f:
        f.write(f"3-Step Merge Debug Statistics\n")
        f.write(f"============================\n\n")
        f.write(f"Tile1: {tile1_name}\n")
        f.write(f"Tile2: {tile2_name}\n")
        f.write(f"Relationship: {tile_relationship}\n\n")
        f.write(f"Cross-boundary nuclei: {len(cross_boundary_nuclei)}\n")
        f.write(f"Cross-boundary IDs: {sorted(cross_boundary_nuclei)}\n\n")
        f.write(f"Expansion pixels added: {np.sum(expansion_mask)}\n")
        f.write(f"Final tile1 nuclei: {len(np.unique(final_tile1[final_tile1 > 0]))}\n")
        f.write(f"Final tile2 nuclei: {len(np.unique(final_tile2[final_tile2 > 0]))}\n")

    logging.info(f"Debug statistics saved: {stats_path}")

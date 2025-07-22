"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: two_phase_merge.py.
Description:
    Two-phase tile merging implementation that systematically applies the new 3-step
    merging rules to overlapping tile pairs. This approach replaces the complex
    4-step algorithm with a simpler, more efficient priority-based method that
    processes vertical overlaps first, then horizontal overlaps.

    The new 3-step merging rule:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching borders, preserve
       non-priority nuclei touching priority borders
    3. Cleanup: Remove remaining non-priority nuclei in overlap region

    The key insight is that nuclei can move outside their original tile boundaries
    during merging, so the second phase must use updated tile masks from the first
    phase to maintain merge consistency across the entire image.

Dependencies:
    • Python ≥ 3.10.
    • numpy, torch, tqdm.
    • cellpose_merge.rules, cellpose_merge.gpu_merge.

Key Features:
    • Systematic two-phase overlap processing for consistent merge results.
    • GPU-accelerated pairwise tile merging with automatic CPU fallback.
    • Simplified 3-step algorithm for better performance and scientific accuracy.
    • Memory-efficient processing of tile pairs instead of large clusters.
    • Comprehensive progress tracking and debug logging for bioinformatics workflows.
"""

from __future__ import annotations

import logging
import traceback
from typing import Dict, List, Tuple, Callable, Optional, Set
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Type aliases for clarity.
TileCoord = Tuple[int, int]
TilePair = Tuple[TileCoord, TileCoord]


def _load_tile_from_storage(coord: TileCoord, storage_dir: Path) -> NDArray[np.uint32]:
    """Load a tile mask from persistent storage."""
    r, c = coord
    tile_filename = f"{r}_{c}.npz"
    tile_path = storage_dir / tile_filename

    if not tile_path.exists():
        raise FileNotFoundError(f"Tile mask not found: {tile_path}")

    tile_data = np.load(tile_path)
    return tile_data["mask"]


def _save_tile_to_storage(coord: TileCoord, tile_mask: NDArray[np.uint32], storage_dir: Path) -> None:
    """Save a tile mask to persistent storage."""
    r, c = coord
    tile_filename = f"{r}_{c}.npz"
    tile_path = storage_dir / tile_filename
    np.savez_compressed(tile_path, mask=tile_mask)
OverlapDict = Dict[TilePair, Tuple[slice, slice, slice, slice]]


def create_overlap_dictionaries(
    coords: List[TileCoord],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> Tuple[OverlapDict, OverlapDict]:
    """
    Generate dictionaries mapping tile coordinate pairs to their overlap regions.
    
    This function identifies all adjacent tile pairs and calculates their exact
    overlap regions in global coordinates. The separation into vertical and
    horizontal overlaps enables systematic two-phase processing that prevents
    merge conflicts and ensures consistent nucleus boundary handling.
    
    Parameters
    ----------
    coords : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the image.
    tile_h, tile_w : int
        Dimensions of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
        
    Returns
    -------
    Tuple[Dict, Dict]
        Two dictionaries mapping tile pairs to overlap regions:
        - vertical_overlapping_regions: horizontally adjacent tiles (same row)
        - horizontal_overlapping_regions: vertically adjacent tiles (same column)
        Each value is (slice1_y, slice1_x, slice2_y, slice2_x) for the overlap region.
    """
    logging.debug(f"Creating overlap dictionaries for {len(coords)} tiles with {overlap}px overlap")
    logging.debug(f"Tile coordinates: {sorted(coords)}")

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    coord_set = set(coords)

    logging.debug(f"Stride: {stride_h}x{stride_w}, Tile size: {tile_h}x{tile_w}")
    
    vertical_overlapping_regions: OverlapDict = {}
    horizontal_overlapping_regions: OverlapDict = {}
    
    for r, c in coords:
        # Check for horizontal neighbor (same row, next column).
        right_neighbor = (r, c + 1)
        if right_neighbor in coord_set:
            # Calculate overlap region between current tile and right neighbor.
            # Current tile: columns [c*stride_w, c*stride_w + tile_w)
            # Right tile: columns [(c+1)*stride_w, (c+1)*stride_w + tile_w)
            # Overlap: columns [(c+1)*stride_w, c*stride_w + tile_w)
            
            overlap_start_col = (c + 1) * stride_w
            overlap_end_col = c * stride_w + tile_w
            
            if overlap_start_col < overlap_end_col:  # Ensure valid overlap.
                # Global coordinates for the overlap region.
                global_row_start = r * stride_h
                global_row_end = r * stride_h + tile_h
                
                # Slices for current tile (left).
                left_slice_y = slice(0, tile_h)
                left_slice_x = slice(overlap_start_col - c * stride_w, tile_w)
                
                # Slices for right tile.
                right_slice_y = slice(0, tile_h)
                right_slice_x = slice(0, overlap_end_col - (c + 1) * stride_w)
                
                vertical_overlapping_regions[((r, c), right_neighbor)] = (
                    left_slice_y, left_slice_x, right_slice_y, right_slice_x
                )
        
        # Check for vertical neighbor (next row, same column).
        bottom_neighbor = (r + 1, c)
        if bottom_neighbor in coord_set:
            # Calculate overlap region between current tile and bottom neighbor.
            overlap_start_row = (r + 1) * stride_h
            overlap_end_row = r * stride_h + tile_h
            
            if overlap_start_row < overlap_end_row:  # Ensure valid overlap.
                # Global coordinates for the overlap region.
                global_col_start = c * stride_w
                global_col_end = c * stride_w + tile_w
                
                # Slices for current tile (top).
                top_slice_y = slice(overlap_start_row - r * stride_h, tile_h)
                top_slice_x = slice(0, tile_w)
                
                # Slices for bottom tile.
                bottom_slice_y = slice(0, overlap_end_row - (r + 1) * stride_h)
                bottom_slice_x = slice(0, tile_w)
                
                horizontal_overlapping_regions[((r, c), bottom_neighbor)] = (
                    top_slice_y, top_slice_x, bottom_slice_y, bottom_slice_x
                )
    
    logging.debug(f"Found {len(vertical_overlapping_regions)} vertical overlaps and "
                 f"{len(horizontal_overlapping_regions)} horizontal overlaps")
    
    return vertical_overlapping_regions, horizontal_overlapping_regions


def merge_two_tiles(
    tile1_mask: NDArray[np.uint32],
    tile2_mask: NDArray[np.uint32],
    overlap_slices: Tuple[slice, slice, slice, slice],
    use_gpu: bool = True,
    gid_offset: int = 0
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]:
    """
    Apply the 3-step merging rules between exactly two overlapping tiles.

    CORE PRINCIPLE: Complete nuclei are processed, not just overlap portions.
    When a nucleus has ANY pixels in the overlap region, the ENTIRE nucleus
    (including parts outside the overlap) is subject to the 3-step merging rules.

    The 3-step algorithm:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching borders, preserve
       non-priority nuclei touching priority borders (cross-boundary nuclei)
    3. Cleanup: Remove remaining non-priority nuclei in overlap region

    Parameters
    ----------
    tile1_mask, tile2_mask : NDArray[np.uint32]
        Input tile masks to be merged.
    overlap_slices : Tuple[slice, slice, slice, slice]
        Slices defining the overlap region: (tile1_y, tile1_x, tile2_y, tile2_x).
    use_gpu : bool, default True
        Whether to use GPU acceleration for merge operations.
    gid_offset : int, default 0
        Global ID offset (unused in current implementation but kept for compatibility).

    Returns
    -------
    Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]
        Updated tile1_mask, updated tile2_mask, and mapping of preserved nucleus IDs.
        The mapping shows which original IDs were preserved (original_id -> original_id).
    """
    try:
        from .rules import merge_patch_cpu_3step
    except ImportError:
        # Fallback for when running as script
        from rules import merge_patch_cpu_3step

    if use_gpu and TORCH_AVAILABLE:
        try:
            try:
                from .gpu_merge import merge_patch_gpu_3step
            except ImportError:
                from gpu_merge import merge_patch_gpu_3step
            merge_fn = merge_patch_gpu_3step
        except ImportError:
            logging.warning("GPU merge not available, falling back to CPU")
            merge_fn = merge_patch_cpu_3step
    else:
        merge_fn = merge_patch_cpu_3step

    tile1_slice_y, tile1_slice_x, tile2_slice_y, tile2_slice_x = overlap_slices

    # Extract overlap regions from both tiles.
    overlap1 = tile1_mask[tile1_slice_y, tile1_slice_x]
    overlap2 = tile2_mask[tile2_slice_y, tile2_slice_x]

    # Ensure both overlap regions have the same shape.
    if overlap1.shape != overlap2.shape:
        logging.warning(f"Overlap shape mismatch: {overlap1.shape} vs {overlap2.shape}")
        min_h = min(overlap1.shape[0], overlap2.shape[0])
        min_w = min(overlap1.shape[1], overlap2.shape[1])
        overlap1 = overlap1[:min_h, :min_w]
        overlap2 = overlap2[:min_h, :min_w]

    # Check if there are any nuclei in the overlap regions.
    nuclei_in_overlap1 = set(np.unique(overlap1[overlap1 > 0]))
    nuclei_in_overlap2 = set(np.unique(overlap2[overlap2 > 0]))

    if len(nuclei_in_overlap1) == 0 and len(nuclei_in_overlap2) == 0:
        logging.debug("No nuclei in overlap region, returning original tiles")
        return tile1_mask, tile2_mask, {}

    # CRITICAL FIX: Identify cross-boundary nuclei for special handling.
    # Cross-boundary nuclei (same ID in both tiles) should be preserved as single entities.
    cross_boundary_nuclei = nuclei_in_overlap1 & nuclei_in_overlap2

    if cross_boundary_nuclei:
        logging.debug(f"Detected cross-boundary nuclei: {cross_boundary_nuclei}")

    # Create 3D patch for merge algorithm: (2, H, W) using overlap regions.
    patch = np.stack([overlap1, overlap2], axis=0)

    # Apply the 3-step merging algorithm to (processed) overlap regions.
    try:
        merged_overlap, mapping = merge_fn(patch)

        if len(mapping) == 0:
            logging.debug("No nuclei changes from merge algorithm, returning original tiles")
            return tile1_mask, tile2_mask, {}

        # CRITICAL FIX: Apply merge decisions to COMPLETE nuclei with proper cross-boundary handling.
        # This ensures no nucleus fragmentation occurs and cross-boundary nuclei are handled correctly.
        updated_tile1 = tile1_mask.copy()
        updated_tile2 = tile2_mask.copy()

        # Find all nuclei that had any pixels in the overlap regions.
        all_overlap_nuclei = nuclei_in_overlap1 | nuclei_in_overlap2

        # STEP 1: Handle cross-boundary nuclei (same ID in both tiles).
        # Cross-boundary nuclei require special handling to prevent fragmentation.

        for nucleus_id in cross_boundary_nuclei:
            # Cross-boundary nuclei should be preserved unless explicitly deleted by 3-step rules.
            # Check if this nucleus was processed by the 3-step algorithm.

            if nucleus_id in mapping:
                # This cross-boundary nucleus was preserved with a new ID.
                new_id = mapping[nucleus_id]

                # Update ALL pixels of this nucleus in BOTH tiles to the new ID.
                updated_tile1[tile1_mask == nucleus_id] = new_id
                updated_tile2[tile2_mask == nucleus_id] = new_id

                logging.debug(f"Preserved cross-boundary nucleus {nucleus_id} -> {new_id} in both tiles")
            else:
                # Check if this nucleus was deleted by the 3-step algorithm.
                # If the nucleus appears in the original overlap but not in the merged result,
                # it was deleted.
                nucleus_in_merged = np.any(merged_overlap == nucleus_id)

                if not nucleus_in_merged:
                    # This cross-boundary nucleus was deleted by the 3-step algorithm.
                    # Remove ALL pixels from BOTH tiles.
                    updated_tile1[tile1_mask == nucleus_id] = 0
                    updated_tile2[tile2_mask == nucleus_id] = 0

                    logging.debug(f"Deleted cross-boundary nucleus {nucleus_id} from both tiles")
                else:
                    # This nucleus still exists in the merged result but wasn't remapped.
                    # This can happen if the nucleus kept its original ID.
                    logging.debug(f"Cross-boundary nucleus {nucleus_id} preserved with original ID")

        # STEP 2: Handle single-tile nuclei (only in one tile's overlap region).
        single_tile_nuclei = all_overlap_nuclei - cross_boundary_nuclei

        for nucleus_id in single_tile_nuclei:
            if nucleus_id in mapping:
                # This nucleus was preserved with a new ID.
                new_id = mapping[nucleus_id]

                # Update ALL pixels of this nucleus in the appropriate tile.
                if nucleus_id in nuclei_in_overlap1:
                    updated_tile1[tile1_mask == nucleus_id] = new_id
                    logging.debug(f"Preserved single-tile nucleus {nucleus_id} -> {new_id} in tile1")

                if nucleus_id in nuclei_in_overlap2:
                    updated_tile2[tile2_mask == nucleus_id] = new_id
                    logging.debug(f"Preserved single-tile nucleus {nucleus_id} -> {new_id} in tile2")
            else:
                # This nucleus was deleted.
                # Remove ALL pixels from the appropriate tile.
                if nucleus_id in nuclei_in_overlap1:
                    updated_tile1[tile1_mask == nucleus_id] = 0
                    logging.debug(f"Deleted single-tile nucleus {nucleus_id} from tile1")

                if nucleus_id in nuclei_in_overlap2:
                    updated_tile2[tile2_mask == nucleus_id] = 0
                    logging.debug(f"Deleted single-tile nucleus {nucleus_id} from tile2")

        # STEP 3: Ensure overlap regions are identical in both tiles.
        # CRITICAL: Only update pixels that don't conflict with complete nucleus decisions.
        # If a nucleus was completely deleted from a tile, don't restore it in the overlap.

        # Create masks for pixels that should be updated in each tile.
        update_mask1 = np.ones_like(merged_overlap, dtype=bool)
        update_mask2 = np.ones_like(merged_overlap, dtype=bool)

        # For each nucleus in the merged overlap, check if it conflicts with tile decisions.
        merged_nuclei = set(np.unique(merged_overlap[merged_overlap > 0]))

        for nucleus_id in merged_nuclei:
            nucleus_pixels_in_merged = merged_overlap == nucleus_id

            # Check if this nucleus was completely deleted from tile1.
            nucleus_exists_in_tile1 = np.any(updated_tile1 == nucleus_id)
            if not nucleus_exists_in_tile1:
                # Don't restore this nucleus in tile1's overlap region.
                update_mask1[nucleus_pixels_in_merged] = False
                logging.debug(f"Preventing restoration of deleted nucleus {nucleus_id} in tile1 overlap")

            # Check if this nucleus was completely deleted from tile2.
            nucleus_exists_in_tile2 = np.any(updated_tile2 == nucleus_id)
            if not nucleus_exists_in_tile2:
                # Don't restore this nucleus in tile2's overlap region.
                update_mask2[nucleus_pixels_in_merged] = False
                logging.debug(f"Preventing restoration of deleted nucleus {nucleus_id} in tile2 overlap")

        # Apply the merged overlap only where it doesn't conflict with tile decisions.
        updated_tile1[tile1_slice_y, tile1_slice_x][update_mask1] = merged_overlap[update_mask1]
        updated_tile2[tile2_slice_y, tile2_slice_x][update_mask2] = merged_overlap[update_mask2]

        # Ensure both overlap regions are identical by taking the union of valid updates.
        final_overlap = np.zeros_like(merged_overlap)
        final_overlap[update_mask1 & update_mask2] = merged_overlap[update_mask1 & update_mask2]

        updated_tile1[tile1_slice_y, tile1_slice_x] = final_overlap
        updated_tile2[tile2_slice_y, tile2_slice_x] = final_overlap

        logging.debug(f"Successfully merged tile pair: {len(mapping)} nuclei processed")

        return updated_tile1, updated_tile2, mapping

    except Exception as e:
        logging.error(f"Error during tile pair merge: {e}")
        logging.error(traceback.format_exc())
        # Return original tiles if merge fails.
        return tile1_mask, tile2_mask, {}


def merge_tiles_two_phase(
    coords: List[TileCoord],
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    use_gpu: bool = True,
    merge_batch_size: int = 4,
    debug_mode: bool = False,
    output_dir: Optional[Path] = None,
) -> NDArray[np.uint32]:
    """
    Two-phase tile merging with systematic overlap processing and persistent storage.

    CORE PRINCIPLE: No new global IDs are created during merging. Only preserve or delete existing IDs.
    Cross-boundary nuclei maintain their original IDs throughout both tiles after merging.

    This function implements a two-phase approach to tile merging using the 3-step algorithm:
    1. Phase 1: Process all vertical overlaps (horizontally adjacent tiles)
       - Save intermediate results to merged_tile_masks_npz/
    2. Phase 2: Process all horizontal overlaps (vertically adjacent tiles)
       - Load from merged_tile_masks_npz/, process, and save back
    3. Final Assembly: Combine all tiles into final merged mask

    The 3-step merging rule:
    1. Priority Selection: Tile with most nuclei gets priority
    2. Border Deletion: Remove priority tile nuclei touching borders, preserve
       non-priority nuclei touching priority borders (cross-boundary nuclei)
    3. Cleanup: Remove remaining non-priority nuclei in overlap region

    This systematic approach ensures consistent merge rule application and
    better handling of cross-boundary nuclei. The persistent storage allows
    for debugging and validation of intermediate results.

    Parameters
    ----------
    coords : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles.
    loader : Callable
        Function to load tile data for given row/column slices.
    height, width : int
        Full image dimensions in pixels.
    tile_h, tile_w : int
        Individual tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    use_gpu : bool, default True
        Whether to use GPU acceleration.
    merge_batch_size : int, default 4
        Number of tile pairs to process in parallel during each phase.
    debug_mode : bool, default False
        Whether to enable debug logging.
    output_dir : Optional[Path], default None
        Output directory for persistent storage. If None, uses current directory.

    Returns
    -------
    NDArray[np.uint32]
        Final merged mask with original nucleus IDs preserved.
    """
    logging.info(f"Starting two-phase merge for {len(coords)} tiles")

    # Set up persistent storage directories.
    if output_dir is None:
        output_dir = Path(".")

    merged_masks_dir = output_dir / "masks" / "merged_tile_masks_npz"
    merged_masks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Using persistent storage: {merged_masks_dir}")

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    # Phase 0: Load and save original tile masks to persistent storage.
    logging.info("Phase 0: Initializing persistent tile storage")

    for r, c in tqdm(coords, desc="Initializing tiles"):
        y_start = r * stride_h
        y_end = min(height, y_start + tile_h)
        x_start = c * stride_w
        x_end = min(width, x_start + tile_w)

        # Load original tile data.
        tile_data = loader(slice(y_start, y_end), slice(x_start, x_end))

        # Save to persistent storage with same naming convention as original tiles.
        tile_filename = f"{r}_{c}.npz"
        tile_path = merged_masks_dir / tile_filename
        np.savez_compressed(tile_path, mask=tile_data)

        if debug_mode:
            nuclei_count = len(np.unique(tile_data[tile_data > 0]))
            logging.debug(f"Initialized tile ({r},{c}): {nuclei_count} nuclei -> {tile_path}")

    # Create overlap dictionaries.
    vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
        coords, tile_h, tile_w, overlap
    )

    # Initialize global ID offset (not used in current 3-step implementation but required for compatibility).
    gid_offset = 0

    # Phase 1: Process all vertical overlaps (horizontally adjacent tiles).
    logging.info(f"Phase 1: Processing {len(vertical_overlaps)} vertical overlaps")

    for tile_pair, overlap_slices in tqdm(vertical_overlaps.items(), desc="Vertical overlaps"):
        coord1, coord2 = tile_pair

        try:
            # Load tiles from persistent storage.
            tile1 = _load_tile_from_storage(coord1, merged_masks_dir)
            tile2 = _load_tile_from_storage(coord2, merged_masks_dir)

            # Debug logging for merge operations.
            if debug_mode:
                nuclei_before_1 = len(np.unique(tile1[tile1 > 0]))
                nuclei_before_2 = len(np.unique(tile2[tile2 > 0]))
                logging.debug(f"Phase 1: Merging tiles {coord1} ({nuclei_before_1} nuclei) and {coord2} ({nuclei_before_2} nuclei)")
                logging.debug(f"Overlap region: {overlap_slices}")

            # Perform merge using 3-step algorithm.
            updated_tile1, updated_tile2, _ = merge_two_tiles(
                tile1,
                tile2,
                overlap_slices,
                use_gpu=use_gpu,
                gid_offset=gid_offset
            )

            # Save updated tiles back to persistent storage.
            _save_tile_to_storage(coord1, updated_tile1, merged_masks_dir)
            _save_tile_to_storage(coord2, updated_tile2, merged_masks_dir)

            # Debug logging for merge results.
            if debug_mode:
                nuclei_after_1 = len(np.unique(updated_tile1[updated_tile1 > 0]))
                nuclei_after_2 = len(np.unique(updated_tile2[updated_tile2 > 0]))
                logging.debug(f"Phase 1: After merge: tile {coord1} has {nuclei_after_1} nuclei, tile {coord2} has {nuclei_after_2} nuclei")

        except Exception as e:
            logging.error(f"Phase 1: Failed to process tile pair {coord1}-{coord2}: {e}")
            continue
    
    # Phase 2: Process all horizontal overlaps (vertically adjacent tiles).
    # CRITICAL: Use updated masks from Phase 1 (loaded from persistent storage).
    logging.info(f"Phase 2: Processing {len(horizontal_overlaps)} horizontal overlaps")

    for tile_pair, overlap_slices in tqdm(horizontal_overlaps.items(), desc="Horizontal overlaps"):
        coord1, coord2 = tile_pair

        try:
            # Load tiles from persistent storage (includes Phase 1 updates).
            tile1 = _load_tile_from_storage(coord1, merged_masks_dir)
            tile2 = _load_tile_from_storage(coord2, merged_masks_dir)

            # Debug logging for merge operations.
            if debug_mode:
                nuclei_before_1 = len(np.unique(tile1[tile1 > 0]))
                nuclei_before_2 = len(np.unique(tile2[tile2 > 0]))
                logging.debug(f"Phase 2: Merging tiles {coord1} ({nuclei_before_1} nuclei) and {coord2} ({nuclei_before_2} nuclei)")
                logging.debug(f"Overlap region: {overlap_slices}")

            # Perform merge using 3-step algorithm.
            updated_tile1, updated_tile2, _ = merge_two_tiles(
                tile1,
                tile2,
                overlap_slices,
                use_gpu=use_gpu,
                gid_offset=gid_offset
            )

            # Save updated tiles back to persistent storage.
            _save_tile_to_storage(coord1, updated_tile1, merged_masks_dir)
            _save_tile_to_storage(coord2, updated_tile2, merged_masks_dir)

            # Debug logging for merge results.
            if debug_mode:
                nuclei_after_1 = len(np.unique(updated_tile1[updated_tile1 > 0]))
                nuclei_after_2 = len(np.unique(updated_tile2[updated_tile2 > 0]))
                logging.debug(f"Phase 2: After merge: tile {coord1} has {nuclei_after_1} nuclei, tile {coord2} has {nuclei_after_2} nuclei")

        except Exception as e:
            logging.error(f"Phase 2: Failed to process tile pair {coord1}-{coord2}: {e}")
            continue
    
    # Phase 3: Assemble final merged image from persistent storage.
    logging.info("Phase 3: Assembling final merged image from persistent storage")
    merged = np.zeros((height, width), dtype=np.uint32)

    for r, c in coords:
        try:
            # Load final merged tile from persistent storage.
            tile_mask = _load_tile_from_storage((r, c), merged_masks_dir)

            y_start = r * stride_h
            y_end = min(height, y_start + tile_h)
            x_start = c * stride_w
            x_end = min(width, x_start + tile_w)

            # Handle edge tiles that may be smaller than tile_h x tile_w.
            actual_h = y_end - y_start
            actual_w = x_end - x_start

            merged[y_start:y_end, x_start:x_end] = tile_mask[:actual_h, :actual_w]

            if debug_mode:
                nuclei_count = len(np.unique(tile_mask[tile_mask > 0]))
                logging.debug(f"Assembled tile ({r},{c}): {nuclei_count} nuclei")

        except Exception as e:
            logging.error(f"Failed to load final tile ({r},{c}): {e}")
            continue

    final_nuclei_count = len(np.unique(merged[merged > 0]))
    logging.info(f"Two-phase merge completed. Final image contains {final_nuclei_count} nuclei")

    # Log merge efficiency for quality assessment.
    if debug_mode:
        # Calculate total input nuclei by loading original tiles.
        total_input_nuclei = 0
        for r, c in coords:
            y_start = r * stride_h
            y_end = min(height, y_start + tile_h)
            x_start = c * stride_w
            x_end = min(width, x_start + tile_w)

            original_tile = loader(slice(y_start, y_end), slice(x_start, x_end))
            total_input_nuclei += len(np.unique(original_tile[original_tile > 0]))

        merge_efficiency = (final_nuclei_count / total_input_nuclei) * 100 if total_input_nuclei > 0 else 0
        logging.debug(f"Merge efficiency: {final_nuclei_count}/{total_input_nuclei} = {merge_efficiency:.1f}%")

    logging.info(f"Persistent storage directory: {merged_masks_dir}")
    logging.info(f"Intermediate tile masks saved for debugging and validation")

    return merged
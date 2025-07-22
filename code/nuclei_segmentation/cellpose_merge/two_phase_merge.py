"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: two_phase_merge.py.
Description:
    Two-phase tile merging implementation that systematically applies the 4-step
    merging rules to overlapping tile pairs. This approach replaces the complex
    cluster-based batching strategy with a simpler, more reliable method that
    processes vertical overlaps first, then horizontal overlaps.
    
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
    • Cross-boundary nucleus tracking to handle merge-induced tile boundary changes.
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
    logging.info(f"Creating overlap dictionaries for {len(coords)} tiles with {overlap}px overlap")
    
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    coord_set = set(coords)
    
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
    
    logging.info(f"Found {len(vertical_overlapping_regions)} vertical overlaps and "
                f"{len(horizontal_overlapping_regions)} horizontal overlaps")
    
    return vertical_overlapping_regions, horizontal_overlapping_regions


def merge_two_tiles(
    tile1_mask: NDArray[np.uint32],
    tile2_mask: NDArray[np.uint32],
    overlap_slices: Tuple[slice, slice, slice, slice],
    threshold: float = 0.3,
    use_gpu: bool = True,
    gid_offset: int = 0
) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]:
    """
    Apply 4-step merging rules between exactly two overlapping tiles.
    
    This function creates a small patch containing only the overlap region,
    applies the standard merge algorithm, and then updates both input tiles
    with the merged results. This ensures that nuclei can move across tile
    boundaries during merging while maintaining consistency.
    
    Parameters
    ----------
    tile1_mask, tile2_mask : NDArray[np.uint32]
        Input tile masks to be merged.
    overlap_slices : Tuple[slice, slice, slice, slice]
        Slices defining the overlap region: (tile1_y, tile1_x, tile2_y, tile2_x).
    threshold : float, default 0.3
        Overlap threshold for merging decisions.
    use_gpu : bool, default True
        Whether to use GPU acceleration for merge operations.
    gid_offset : int, default 0
        Global ID offset to ensure unique nucleus IDs.
        
    Returns
    -------
    Tuple[NDArray[np.uint32], NDArray[np.uint32], Dict[int, int]]
        Updated tile1_mask, updated tile2_mask, and mapping dictionary.
    """
    from .rules import merge_patch_cpu
    
    if use_gpu and TORCH_AVAILABLE:
        try:
            from .gpu_merge import merge_patch_gpu
            merge_fn = merge_patch_gpu
        except ImportError:
            logging.warning("GPU merge not available, falling back to CPU")
            merge_fn = merge_patch_cpu
    else:
        merge_fn = merge_patch_cpu
    
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
    
    # Create 3D patch for merge algorithm: (2, H, W).
    patch = np.stack([overlap1, overlap2], axis=0)
    
    # Apply merge algorithm to the overlap region.
    try:
        merged_overlap, mapping = merge_fn(patch, threshold=threshold)
        
        # Apply global ID offset if specified.
        if gid_offset > 0:
            nucleus_mask = merged_overlap != 0
            merged_overlap = merged_overlap.astype(np.uint32)
            merged_overlap[nucleus_mask] += int(gid_offset)
        
        # Update both tiles with merged results in their overlap regions.
        updated_tile1 = tile1_mask.copy()
        updated_tile2 = tile2_mask.copy()
        
        updated_tile1[tile1_slice_y, tile1_slice_x] = merged_overlap
        updated_tile2[tile2_slice_y, tile2_slice_x] = merged_overlap
        
        logging.debug(f"Successfully merged tile pair with {len(np.unique(merged_overlap)) - 1} nuclei in overlap")
        
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
    threshold: float = 0.3,
    use_gpu: bool = True,
    merge_batch_size: int = 4,
    gid_offset: int = 0
) -> NDArray[np.uint32]:
    """
    Merge tiles using systematic two-phase overlap processing.
    
    This function implements the new two-phase merging strategy that processes
    all vertical overlaps first, then all horizontal overlaps. This ensures
    consistent merge results and proper handling of nuclei that move across
    tile boundaries during merging.
    
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
    threshold : float, default 0.3
        Overlap threshold for merging decisions.
    use_gpu : bool, default True
        Whether to use GPU acceleration.
    merge_batch_size : int, default 4
        Number of tile pairs to process in parallel during each phase.
    gid_offset : int, default 0
        Starting global ID offset.
        
    Returns
    -------
    NDArray[np.uint32]
        Final merged mask with unique nucleus IDs.
    """
    logging.info(f"Starting two-phase merge for {len(coords)} tiles")
    
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    # Load all tiles into memory for processing.
    tile_masks: Dict[TileCoord, NDArray[np.uint32]] = {}
    
    logging.info("Loading all tile masks into memory")
    for r, c in tqdm(coords, desc="Loading tiles"):
        y_start = r * stride_h
        y_end = min(height, y_start + tile_h)
        x_start = c * stride_w
        x_end = min(width, x_start + tile_w)
        
        tile_masks[(r, c)] = loader(slice(y_start, y_end), slice(x_start, x_end))
    
    # Create overlap dictionaries.
    vertical_overlaps, horizontal_overlaps = create_overlap_dictionaries(
        coords, tile_h, tile_w, overlap
    )
    
    # Phase 1: Process all vertical overlaps (horizontally adjacent tiles).
    logging.info(f"Phase 1: Processing {len(vertical_overlaps)} vertical overlaps")
    
    for tile_pair, overlap_slices in tqdm(vertical_overlaps.items(), desc="Vertical overlaps"):
        coord1, coord2 = tile_pair
        
        if coord1 in tile_masks and coord2 in tile_masks:
            updated_tile1, updated_tile2, _ = merge_two_tiles(
                tile_masks[coord1],
                tile_masks[coord2],
                overlap_slices,
                threshold=threshold,
                use_gpu=use_gpu,
                gid_offset=gid_offset
            )
            
            # Update tiles with merged results.
            tile_masks[coord1] = updated_tile1
            tile_masks[coord2] = updated_tile2
    
    # Phase 2: Process all horizontal overlaps (vertically adjacent tiles).
    # CRITICAL: Use updated masks from Phase 1.
    logging.info(f"Phase 2: Processing {len(horizontal_overlaps)} horizontal overlaps")
    
    for tile_pair, overlap_slices in tqdm(horizontal_overlaps.items(), desc="Horizontal overlaps"):
        coord1, coord2 = tile_pair
        
        if coord1 in tile_masks and coord2 in tile_masks:
            updated_tile1, updated_tile2, _ = merge_two_tiles(
                tile_masks[coord1],
                tile_masks[coord2],
                overlap_slices,
                threshold=threshold,
                use_gpu=use_gpu,
                gid_offset=gid_offset
            )
            
            # Update tiles with merged results.
            tile_masks[coord1] = updated_tile1
            tile_masks[coord2] = updated_tile2
    
    # Assemble final merged image.
    logging.info("Assembling final merged image")
    merged = np.zeros((height, width), dtype=np.uint32)
    
    for (r, c), tile_mask in tile_masks.items():
        y_start = r * stride_h
        y_end = min(height, y_start + tile_h)
        x_start = c * stride_w
        x_end = min(width, x_start + tile_w)
        
        # Handle edge tiles that may be smaller than tile_h x tile_w.
        actual_h = y_end - y_start
        actual_w = x_end - x_start
        
        merged[y_start:y_end, x_start:x_end] = tile_mask[:actual_h, :actual_w]
    
    logging.info(f"Two-phase merge completed. Final image contains {len(np.unique(merged)) - 1} nuclei")
    
    return merged
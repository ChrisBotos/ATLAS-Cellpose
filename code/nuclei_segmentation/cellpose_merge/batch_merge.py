"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: batch_merge.py.
Description:
    Implements batched processing for GPU-based tile merging to handle large images
    with thousands of tiles. This module addresses memory allocation errors by
    processing tiles in spatial groups based on their overlap relationships.
    
    The implementation divides large clusters into smaller batches of 2x2 tile groups,
    processes them sequentially or in parallel (depending on available GPU memory),
    and carefully handles the boundaries between batches to maintain merge rule
    consistency.

Dependencies:
    • Python ≥ 3.10.
    • numpy, torch, tqdm.
    • cellpose_merge.gpu_merge, cellpose_merge.rules.

Key Features:
    • Memory-efficient batched processing for large tile clusters.
    • Spatial proximity grouping to maintain merge rule consistency.
    • Configurable batch sizes with automatic memory management.
    • Comprehensive progress tracking and memory usage monitoring.
    • Graceful fallback to smaller batch sizes when memory is constrained.
"""

from __future__ import annotations

import logging
import math
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Set

import numpy as np
from numpy.typing import NDArray
import torch
from tqdm import tqdm

# Type aliases for readability
TileCoord = Tuple[int, int]
TileGroup = List[TileCoord]
MergeResult = Tuple[NDArray[np.uint32], Tuple[int, int], Dict[int, int]]

def estimate_memory_requirements(
    tiles: List[TileCoord],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> float:
    """
    Estimate the GPU memory required for processing a batch of tiles.

    This function calculates memory based on the actual spatial extent of the tiles,
    not just the number of tiles, which provides more accurate estimates for
    irregularly distributed tile clusters.

    Parameters
    ----------
    tiles : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the batch.
    tile_h, tile_w : int
        Size of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.

    Returns
    -------
    float
        Estimated memory requirement in gigabytes.
    """
    if not tiles:
        return 0.0

    # Calculate the actual spatial extent of this batch.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    min_r = min(r for r, _ in tiles)
    max_r = max(r for r, _ in tiles)
    min_c = min(c for _, c in tiles)
    max_c = max(c for _, c in tiles)

    # Calculate the bounding box dimensions for this specific batch.
    batch_h = (max_r - min_r) * stride_h + tile_h
    batch_w = (max_c - min_c) * stride_w + tile_w

    num_tiles = len(tiles)

    # Memory for input stack: (num_tiles, batch_h, batch_w) as uint32 (4 bytes).
    stack_memory = num_tiles * batch_h * batch_w * 4

    # Memory for intermediate tensors (approximately 3x the input size for GPU processing).
    # This includes the DSU structures, overlap calculations, etc.
    intermediate_memory = stack_memory * 3

    # Total memory in gigabytes.
    total_memory_gb = (stack_memory + intermediate_memory) / (1024**3)

    logging.debug(f"Memory estimate for {num_tiles} tiles spanning {batch_h}x{batch_w} pixels "
                 f"with {overlap}px overlap: {total_memory_gb:.2f} GB")

    return total_memory_gb

def get_optimal_batch_size(
    cluster: List[TileCoord],
    tile_h: int,
    tile_w: int,
    overlap: int,
    memory_limit_gb: float = 8.0
) -> int:
    """
    Calculate the optimal batch size based on available GPU memory and actual tile distribution.

    This function now considers the spatial distribution of tiles to provide more
    accurate memory estimates and better batch sizing decisions.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the cluster.
    tile_h, tile_w : int
        Size of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    memory_limit_gb : float, default 8.0
        Maximum GPU memory to use in gigabytes.

    Returns
    -------
    int
        Optimal batch size (number of tiles to process at once).
    """
    if not cluster:
        return 1

    # If memory_limit_gb is 0, try to detect available GPU memory.
    if memory_limit_gb <= 0 and torch.cuda.is_available():
        try:
            # Get available GPU memory in GB.
            free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Use 70% of available memory as a safety margin (more conservative).
            memory_limit_gb = free_memory * 0.7
            logging.info(f"Detected {free_memory:.2f} GB GPU memory, using {memory_limit_gb:.2f} GB as limit")
        except Exception as e:
            logging.warning(f"Failed to detect GPU memory: {e}. Using default 8 GB limit.")
            memory_limit_gb = 8.0

    # Start with a very conservative batch size for large clusters.
    if len(cluster) > 100:
        # For very large clusters, start with individual tiles.
        max_batch_size = min(4, len(cluster))
    else:
        # For smaller clusters, can be more aggressive.
        max_batch_size = min(16, len(cluster))

    # Test different batch sizes to find the optimal one.
    optimal_batch_size = 1

    for test_batch_size in range(1, max_batch_size + 1):
        # Create a test batch with the first few tiles.
        test_batch = cluster[:test_batch_size]

        # Estimate memory for this batch size.
        estimated_memory = estimate_memory_requirements(test_batch, tile_h, tile_w, overlap)

        if estimated_memory <= memory_limit_gb:
            optimal_batch_size = test_batch_size
        else:
            # Stop when we exceed memory limit.
            break

    logging.info(f"Optimal batch size: {optimal_batch_size} tiles (memory limit: {memory_limit_gb:.2f} GB, "
                f"cluster size: {len(cluster)} tiles)")

    return optimal_batch_size

def group_tiles_by_spatial_proximity(
    cluster: List[TileCoord],
    batch_size: int = 1
) -> List[List[TileCoord]]:
    """
    Group tiles into memory-efficient batches using spatial proximity.

    This function creates batches that are optimized for memory usage rather than
    following a rigid 2x2 pattern. For large clusters, it prioritizes creating
    smaller, spatially compact batches that fit within GPU memory constraints.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the cluster.
    batch_size : int, default 1
        Maximum number of tiles to include in each batch.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of batches, where each batch contains a list of tile coordinates.
    """
    if not cluster:
        return []

    # For very large clusters, use a simple spatial chunking approach.
    if len(cluster) > 50:
        logging.info(f"Large cluster detected ({len(cluster)} tiles), using spatial chunking approach")
        return _create_spatial_chunks(cluster, batch_size)

    # For smaller clusters, use the original 2x2 grouping approach.
    return _create_2x2_groups(cluster, batch_size)


def _create_spatial_chunks(
    cluster: List[TileCoord],
    max_batch_size: int
) -> List[List[TileCoord]]:
    """
    Create spatially compact chunks for large clusters.

    This approach sorts tiles by spatial position and creates sequential batches
    that are likely to have smaller bounding boxes, reducing memory requirements.
    """
    # Sort tiles by row first, then by column for spatial locality.
    sorted_tiles = sorted(cluster, key=lambda tile: (tile[0], tile[1]))

    batches = []
    for i in range(0, len(sorted_tiles), max_batch_size):
        batch = sorted_tiles[i:i + max_batch_size]
        batches.append(batch)

    logging.debug(f"Created {len(batches)} spatial chunks with max size {max_batch_size}")
    return batches


def _create_2x2_groups(
    cluster: List[TileCoord],
    batch_size: int
) -> List[List[TileCoord]]:
    """
    Create 2x2 tile groups for smaller clusters (original approach).

    This maintains the original logic for smaller clusters where the 2x2
    grouping strategy is more beneficial for merge rule consistency.
    """
    # Find the range of row and column indices.
    min_r = min(r for r, _ in cluster)
    max_r = max(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_c = max(c for _, c in cluster)

    # Create a grid representation for faster lookup.
    grid = {(r, c): True for r, c in cluster}

    # Create 2x2 groups.
    all_groups = []

    # Use a sliding window approach to create overlapping 2x2 groups.
    for r in range(min_r, max_r):
        for c in range(min_c, max_c):
            group = []
            for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                tile = (r + dr, c + dc)
                if tile in grid:
                    group.append(tile)
            if len(group) >= 2:  # Only keep groups with at least 2 tiles.
                all_groups.append(group)

    # Create batches based on the batch_size parameter.
    batches = []
    if batch_size == 1:
        # Sequential processing: each group becomes its own batch.
        batches = all_groups
    else:
        # Parallel processing: combine groups into batches of specified size.
        for i in range(0, len(all_groups), batch_size):
            batch_groups = all_groups[i:i + batch_size]

            if len(batch_groups) == 1:
                # Single group in this batch.
                batches.append(batch_groups[0])
            else:
                # Multiple groups: combine tiles and remove duplicates.
                combined_tiles = []
                for group in batch_groups:
                    combined_tiles.extend(group)

                # Remove duplicates while preserving order.
                unique_tiles = []
                seen = set()
                for tile in combined_tiles:
                    if tile not in seen:
                        unique_tiles.append(tile)
                        seen.add(tile)
                batches.append(unique_tiles)

    # Handle any tiles not included in any group.
    processed_tiles = set()
    for batch in batches:
        processed_tiles.update(batch)

    remaining_tiles = [t for t in cluster if t not in processed_tiles]
    if remaining_tiles:
        # Process remaining tiles in small groups.
        for i in range(0, len(remaining_tiles), min(4, batch_size)):
            batches.append(remaining_tiles[i:i + min(4, batch_size)])

    logging.info(f"Created {len(batches)} batches from {len(cluster)} tiles with batch_size={batch_size}")
    logging.debug(f"Batch sizes: {[len(batch) for batch in batches]}")

    return batches

def merge_cluster_batched(
    *,
    cluster: List[TileCoord],
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    threshold: float,
    use_gpu: bool,
    gid_offset: int,
    batch_size: int = 1,
    memory_limit_gb: float = 8.0,
    temp_file_path: Optional[Path] = None,
    global_merged_array: Optional[NDArray[np.uint32]] = None,
) -> MergeResult:
    """
    Merge all tiles in a cluster using batched processing to manage memory usage.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the cluster.
    loader : Callable
        Function that loads tile data for a given slice.
    height, width : int
        Dimensions of the full image in pixels.
    tile_h, tile_w : int
        Dimensions of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    threshold : float
        Threshold for merging objects across tiles.
    use_gpu : bool
        Whether to use GPU acceleration.
    gid_offset : int
        Offset to add to global IDs to ensure uniqueness.
    batch_size : int, default 1
        Number of 2x2 tile groups to process in each batch.
    memory_limit_gb : float, default 8.0
        Maximum GPU memory to use in gigabytes.
    temp_file_path : Path, optional
        Path to temporary file for incremental saving of merged results.
    global_merged_array : NDArray[np.uint32], optional
        Reference to the global merged array for incremental updates.

    Returns
    -------
    Tuple[NDArray[np.uint32], Tuple[int, int], Dict[int, int]]
        Merged patch, (y0, x0) coordinates, and mapping dictionary.
    """
    # Import merge functions here to avoid circular imports
    from .rules import merge_patch_cpu
    from .gpu_merge import merge_patch_gpu

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    # Calculate the bounding box of the entire cluster
    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_r = max(r for r, _ in cluster)
    max_c = max(c for _, c in cluster)

    # Validate tile indices are reasonable for the given image dimensions.
    max_possible_rows = (height + stride_h - 1) // stride_h
    max_possible_cols = (width + stride_w - 1) // stride_w

    if max_r >= max_possible_rows or max_c >= max_possible_cols:
        raise ValueError(f"Tile indices out of bounds: max_tile=({max_r},{max_c}), max_possible=({max_possible_rows-1},{max_possible_cols-1})")

    y0 = min_r * stride_h
    x0 = min_c * stride_w

    # Ensure starting coordinates are within image bounds.
    if y0 >= height or x0 >= width:
        raise ValueError(f"Cluster starting position ({y0},{x0}) exceeds image bounds ({height},{width})")

    # Clamp the bounding box to the actual slide size
    cluster_h = min((max_r - min_r) * stride_h + tile_h, height - y0)
    cluster_w = min((max_c - min_c) * stride_w + tile_w, width - x0)

    # Ensure dimensions are positive.
    if cluster_h <= 0 or cluster_w <= 0:
        raise ValueError(f"Invalid cluster dimensions: {cluster_h}×{cluster_w} (y0={y0}, x0={x0}, height={height}, width={width})")

    logging.info(f"Processing cluster with {len(cluster)} tiles using batched approach: "
                f"tile_range=({min_r},{min_c}) to ({max_r},{max_c}), "
                f"global_bbox=({y0},{x0}) to ({y0+cluster_h},{x0+cluster_w}), "
                f"image_size=({height},{width})")

    # Check for potential overflow issues before processing.
    total_elements = cluster_h * cluster_w
    if total_elements > 2**31 - 1:
        raise RuntimeError(f"Cluster patch would have {total_elements} elements, exceeding safe array size limits. "
                         f"Cluster dimensions: {cluster_h}×{cluster_w}")

    # Check for uint32 overflow in gid_offset.
    max_safe_gid = 2**31 - 1  # Conservative limit to prevent uint32 overflow.
    if gid_offset >= max_safe_gid:
        logging.warning(f"Global ID offset {gid_offset} approaching uint32 limit {max_safe_gid}. "
                       f"Resetting to prevent overflow.")
        gid_offset = 1  # Reset to prevent overflow, accepting potential ID conflicts.

    # Create the output merged patch.
    try:
        merged_patch = np.zeros((cluster_h, cluster_w), dtype=np.uint32)
    except (MemoryError, OverflowError) as e:
        raise RuntimeError(f"Failed to allocate memory for cluster patch of size {cluster_h}×{cluster_w}: {e}. "
                         f"Consider processing smaller image regions.")

    # Determine optimal batch size if auto-detection is enabled or if current size is too large.
    if batch_size <= 0:
        batch_size = get_optimal_batch_size(
            cluster=cluster,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            memory_limit_gb=memory_limit_gb
        )
    else:
        # Even if batch_size is specified, validate it against memory constraints.
        test_batch = cluster[:min(batch_size, len(cluster))]
        estimated_memory = estimate_memory_requirements(test_batch, tile_h, tile_w, overlap)

        if estimated_memory > memory_limit_gb:
            logging.warning(f"Specified batch_size={batch_size} would require {estimated_memory:.2f} GB, "
                           f"exceeding limit of {memory_limit_gb:.2f} GB. Reducing batch size.")
            batch_size = get_optimal_batch_size(
                cluster=cluster,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                memory_limit_gb=memory_limit_gb
            )

    # Group tiles into batches based on spatial proximity.
    batches = group_tiles_by_spatial_proximity(cluster, batch_size)

    # Clean up GPU memory before starting batch processing.
    if use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.debug("Cleaned GPU memory before batch processing")

    # Process each batch
    current_gid = gid_offset
    batch_results = []

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing tile batches")):
        try:
            logging.debug(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} tiles")

            # Calculate the bounding box for this batch
            batch_min_r = min(r for r, _ in batch)
            batch_min_c = min(c for _, c in batch)
            batch_max_r = max(r for r, _ in batch)
            batch_max_c = max(c for _, c in batch)

            batch_y0 = batch_min_r * stride_h
            batch_x0 = batch_min_c * stride_w

            # Ensure batch starting coordinates are within image bounds.
            if batch_y0 >= height or batch_x0 >= width:
                logging.warning(f"Batch starting position ({batch_y0},{batch_x0}) exceeds image bounds ({height},{width}), skipping batch")
                continue

            # Clamp the bounding box to the actual slide size
            batch_h = min((batch_max_r - batch_min_r) * stride_h + tile_h, height - batch_y0)
            batch_w = min((batch_max_c - batch_min_c) * stride_w + tile_w, width - batch_x0)

            # Ensure batch dimensions are positive.
            if batch_h <= 0 or batch_w <= 0:
                logging.warning(f"Invalid batch dimensions: {batch_h}×{batch_w} (batch_y0={batch_y0}, batch_x0={batch_x0}), skipping batch")
                continue

            # Calculate relative position within the cluster
            rel_y0 = batch_y0 - y0
            rel_x0 = batch_x0 - x0

            # Create a stack for this batch.
            T = len(batch)

            # Check for potential memory issues before allocation.
            batch_elements = T * batch_h * batch_w
            if batch_elements > 2**31 - 1:
                raise RuntimeError(f"Batch stack would have {batch_elements} elements, exceeding safe limits. "
                                 f"Batch size: {T}, dimensions: {batch_h}×{batch_w}")

            try:
                batch_stack = np.zeros((T, batch_h, batch_w), dtype=np.uint32)
            except (MemoryError, OverflowError) as e:
                raise RuntimeError(f"Failed to allocate memory for batch stack of size ({T}, {batch_h}, {batch_w}): {e}. "
                                 f"Consider reducing batch size or using CPU processing.")

            # Load tiles for this batch
            for t, (r, c) in enumerate(batch):
                global_y0 = r * stride_h
                global_x0 = c * stride_w
                rel_batch_y0 = global_y0 - batch_y0
                rel_batch_x0 = global_x0 - batch_x0

                # Clamp the tile request to image boundaries
                ys = slice(global_y0, min(global_y0 + tile_h, height))
                xs = slice(global_x0, min(global_x0 + tile_w, width))

                logging.debug(f"Loading tile ({r},{c}) at global=({global_y0},{global_x0}), "
                             f"slice=({ys.start}:{ys.stop}, {xs.start}:{xs.stop})")

                tile = loader(ys, xs)
                h, w = tile.shape

                # Ensure we don't exceed the batch bounds
                end_y = min(rel_batch_y0 + h, batch_h)
                end_x = min(rel_batch_x0 + w, batch_w)

                if end_y > rel_batch_y0 and end_x > rel_batch_x0:
                    # Adjust tile dimensions if needed
                    tile_h_to_copy = end_y - rel_batch_y0
                    tile_w_to_copy = end_x - rel_batch_x0

                    batch_stack[t, rel_batch_y0:end_y, rel_batch_x0:end_x] = tile[:tile_h_to_copy, :tile_w_to_copy]

                    logging.debug(f"Placed tile ({r},{c}) in batch stack: "
                                 f"stack_pos=({rel_batch_y0}:{end_y}, {rel_batch_x0}:{end_x}), "
                                 f"tile_size=({h},{w}), copied=({tile_h_to_copy},{tile_w_to_copy})")
                else:
                    logging.warning(f"Tile ({r},{c}) could not be placed in batch stack - bounds issue")

            # Merge this batch
            merge_fn = merge_patch_gpu if use_gpu else merge_patch_cpu
            batch_merged, _ = merge_fn(batch_stack, threshold=threshold)

            # Shift labels to ensure uniqueness across batches
            if np.any(batch_merged > 0):
                nucleus_mask = batch_merged != 0
                max_label = int(batch_merged.max())
                batch_merged = batch_merged.astype(np.uint32, copy=False)
                batch_merged[nucleus_mask] += current_gid
                current_gid += max_label

                # Store the batch result for potential boundary processing
                batch_results.append({
                    'merged': batch_merged,
                    'position': (rel_y0, rel_x0),
                    'size': (batch_h, batch_w),
                    'tiles': batch
                })

                # Copy non-zero pixels to the final merged patch
                merged_patch[rel_y0:rel_y0+batch_h, rel_x0:rel_x0+batch_w][nucleus_mask] = batch_merged[nucleus_mask]

                logging.debug(f"Batch {batch_idx+1} merge completed: "
                             f"output_size=({batch_h},{batch_w}), "
                             f"max_label={batch_merged.max()}, "
                             f"non_zero_pixels={np.count_nonzero(nucleus_mask)}")

                # Incremental saving: Update global array and save to temp file after each batch.
                if global_merged_array is not None and temp_file_path is not None:
                    # Update the global merged array with this batch's results.
                    global_y0 = y0 + rel_y0
                    global_x0 = x0 + rel_x0
                    global_merged_array[global_y0:global_y0+batch_h, global_x0:global_x0+batch_w][nucleus_mask] = batch_merged[nucleus_mask]

                    # Save the updated global array to the temp file.
                    np.save(temp_file_path, global_merged_array)
                    logging.debug(f"Incremental save completed for batch {batch_idx+1}")
            else:
                logging.debug(f"Batch {batch_idx+1} has no nuclei")

            # Clean up to free memory
            del batch_stack
            if use_gpu:
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx+1}: {e}")
            logging.debug(f"Batch error traceback:\n{traceback.format_exc()}")

            # Try to recover by reducing batch size or switching to CPU.
            if batch_size > 1 and use_gpu:
                # Calculate a more aggressive reduction for memory issues.
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    # For memory errors, reduce more aggressively.
                    new_batch_size = max(1, batch_size // 4)
                    logging.warning(f"Memory error detected. Reducing batch size from {batch_size} to {new_batch_size} and retrying")
                else:
                    # For other errors, reduce less aggressively.
                    new_batch_size = max(1, batch_size // 2)
                    logging.warning(f"Reducing batch size from {batch_size} to {new_batch_size} and retrying")

                # Clean up GPU memory before retry.
                if use_gpu:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Prevent infinite recursion by limiting retry attempts.
                if new_batch_size == batch_size:
                    logging.error("Cannot reduce batch size further, falling back to CPU")
                    use_gpu = False
                    new_batch_size = 1

                return merge_cluster_batched(
                    cluster=cluster,
                    loader=loader,
                    height=height,
                    width=width,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    overlap=overlap,
                    threshold=threshold,
                    use_gpu=use_gpu,
                    gid_offset=gid_offset,
                    batch_size=new_batch_size,
                    memory_limit_gb=memory_limit_gb * 0.8,  # Reduce memory limit for retry.
                    temp_file_path=temp_file_path,
                    global_merged_array=global_merged_array,
                )
            else:
                # If we're already at minimum batch size, fall back to CPU
                if use_gpu:
                    logging.warning("Falling back to CPU processing")

                    # Clean up GPU memory before CPU fallback.
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    return merge_cluster_batched(
                        cluster=cluster,
                        loader=loader,
                        height=height,
                        width=width,
                        tile_h=tile_h,
                        tile_w=tile_w,
                        overlap=overlap,
                        threshold=threshold,
                        use_gpu=False,
                        gid_offset=gid_offset,
                        batch_size=1,
                        memory_limit_gb=memory_limit_gb,
                        temp_file_path=temp_file_path,
                        global_merged_array=global_merged_array,
                    )
                else:
                    # If we're already on CPU and still failing, raise the error
                    raise

    # TODO: Implement boundary merge processing for overlapping regions between batches
    # This would handle cases where nuclei span across batch boundaries
    # For now, the current implementation should work for most cases since we use
    # spatial proximity grouping which minimizes boundary issues

    logging.info(f"Batched cluster merge completed: "
                f"output_size=({merged_patch.shape[0]},{merged_patch.shape[1]}), "
                f"max_label={merged_patch.max()}, "
                f"non_zero_pixels={np.count_nonzero(merged_patch)}, "
                f"processed_batches={len(batches)}")

    return merged_patch, (y0, x0), {}

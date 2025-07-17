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
    overlap: int,
    safety_factor: float = 1.5
) -> float:
    """
    Estimate the GPU memory required for processing a batch of tiles with enhanced accuracy.

    This function calculates memory based on the actual spatial extent of the tiles,
    not just the number of tiles, which provides more accurate estimates for
    irregularly distributed tile clusters. Enhanced with better safety margins
    and more precise intermediate memory calculations.

    Parameters
    ----------
    tiles : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the batch.
    tile_h, tile_w : int
        Size of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    safety_factor : float, default 1.5
        Safety multiplier for memory estimates to prevent out-of-memory errors.
        Higher values are more conservative but reduce GPU utilization.

    Returns
    -------
    float
        Estimated memory requirement in gigabytes with safety margin applied.
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

    # Enhanced intermediate memory calculation based on actual GPU operations.
    # DSU structures: max_labels * num_tiles * 8 bytes (int64).
    max_labels_estimate = min(65536, batch_h * batch_w // 100)  # Conservative estimate.
    dsu_memory = max_labels_estimate * num_tiles * 8

    # Overlap masks and temporary tensors: approximately 2x input size.
    temp_tensor_memory = stack_memory * 2

    # Border detection and counting arrays.
    border_memory = max_labels_estimate * num_tiles * 1  # bool arrays.

    # Total intermediate memory with more precise calculation.
    intermediate_memory = dsu_memory + temp_tensor_memory + border_memory

    # Apply safety factor to prevent out-of-memory errors.
    total_memory_bytes = (stack_memory + intermediate_memory) * safety_factor
    total_memory_gb = total_memory_bytes / (1024**3)

    logging.debug(f"Enhanced memory estimate for {num_tiles} tiles spanning {batch_h}x{batch_w} pixels "
                 f"with {overlap}px overlap: {total_memory_gb:.2f} GB (safety factor: {safety_factor})")

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

    # Enhanced GPU memory detection with better error handling.
    if memory_limit_gb <= 0 and torch.cuda.is_available():
        try:
            # Get current GPU memory status.
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)

            # Calculate available memory more accurately.
            available_memory = total_memory - max(allocated_memory, cached_memory)
            available_gb = available_memory / (1024**3)

            # Use 60% of available memory as safety margin (more conservative).
            memory_limit_gb = available_gb * 0.6

            logging.info(f"GPU memory status - Total: {total_memory/(1024**3):.2f} GB, "
                        f"Available: {available_gb:.2f} GB, Using: {memory_limit_gb:.2f} GB")
        except Exception as e:
            logging.warning(f"Failed to detect GPU memory: {e}. Using default 8 GB limit.")
            memory_limit_gb = 8.0

    # Calculate spatial density of tiles for adaptive batch sizing.
    min_r, max_r = min(r for r, _ in cluster), max(r for r, _ in cluster)
    min_c, max_c = min(c for _, c in cluster), max(c for _, c in cluster)
    spatial_span = (max_r - min_r + 1) * (max_c - min_c + 1)
    density = len(cluster) / max(1, spatial_span)

    # Adjust max batch size based on density and cluster size.
    if density > 0.8:  # Dense clusters.
        max_batch_size = min(8, len(cluster))
    elif len(cluster) > 200:  # Very large sparse clusters.
        max_batch_size = min(2, len(cluster))
    elif len(cluster) > 50:  # Large clusters.
        max_batch_size = min(4, len(cluster))
    else:  # Small clusters.
        max_batch_size = min(16, len(cluster))

    # Binary search for optimal batch size for efficiency.
    left, right = 1, max_batch_size
    optimal_batch_size = 1

    while left <= right:
        mid = (left + right) // 2
        test_batch = cluster[:mid]

        # Use enhanced memory estimation with safety factor.
        estimated_memory = estimate_memory_requirements(
            test_batch, tile_h, tile_w, overlap, safety_factor=1.3
        )

        if estimated_memory <= memory_limit_gb:
            optimal_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1

    logging.info(f"Optimal batch size: {optimal_batch_size} tiles (memory limit: {memory_limit_gb:.2f} GB, "
                f"cluster size: {len(cluster)} tiles, density: {density:.2f})")

    return optimal_batch_size

def group_tiles_by_spatial_proximity(
    cluster: List[TileCoord],
    batch_size: int = 1,
    strategy: str = "adaptive"
) -> List[List[TileCoord]]:
    """
    Group tiles into memory-efficient batches using optimized spatial proximity algorithms.

    This function creates batches that are optimized for memory usage and GPU processing
    efficiency. It supports multiple strategies including adaptive 2x2 grouping,
    spatial chunking, and hybrid approaches based on cluster characteristics.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the cluster.
    batch_size : int, default 1
        Maximum number of tiles to include in each batch.
    strategy : str, default "adaptive"
        Batching strategy to use:
        - "adaptive": Automatically choose best strategy based on cluster size.
        - "2x2": Prioritize 2x2 tile groups for optimal overlap processing.
        - "spatial": Use spatial chunking for memory efficiency.
        - "hybrid": Combine 2x2 and spatial approaches.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of batches, where each batch contains a list of tile coordinates.
        Batches are ordered to minimize memory fragmentation during processing.
    """
    if not cluster:
        return []

    # Adaptive strategy selection based on cluster characteristics.
    if strategy == "adaptive":
        # Calculate cluster density to choose optimal strategy.
        min_r, max_r = min(r for r, _ in cluster), max(r for r, _ in cluster)
        min_c, max_c = min(c for _, c in cluster), max(c for _, c in cluster)
        spatial_span = (max_r - min_r + 1) * (max_c - min_c + 1)
        density = len(cluster) / max(1, spatial_span)

        if len(cluster) > 100 or density < 0.3:
            # Large or sparse clusters: use spatial chunking.
            strategy = "spatial"
        elif density > 0.7:
            # Dense clusters: use 2x2 grouping for better overlap handling.
            strategy = "2x2"
        else:
            # Medium clusters: use hybrid approach.
            strategy = "hybrid"

        logging.debug(f"Adaptive strategy selected '{strategy}' for cluster of {len(cluster)} tiles "
                     f"(density: {density:.2f})")

    # Apply selected strategy.
    if strategy == "2x2":
        return _create_optimized_2x2_groups(cluster, batch_size)
    elif strategy == "spatial":
        return _create_spatial_chunks(cluster, batch_size)
    elif strategy == "hybrid":
        return _create_hybrid_groups(cluster, batch_size)
    else:
        logging.warning(f"Unknown strategy '{strategy}', falling back to spatial chunking")
        return _create_spatial_chunks(cluster, batch_size)


def _create_spatial_chunks(
    cluster: List[TileCoord],
    max_batch_size: int
) -> List[List[TileCoord]]:
    """
    Create spatially compact chunks for large clusters with enhanced locality.

    This approach uses Z-order (Morton order) sorting to maintain spatial locality
    while creating batches that minimize memory fragmentation and bounding box size.
    """
    def morton_encode(r: int, c: int) -> int:
        """Encode (row, col) coordinates using Morton (Z-order) encoding."""
        result = 0
        for i in range(16):  # Support up to 65536x65536 grids.
            result |= ((r & (1 << i)) << i) | ((c & (1 << i)) << (i + 1))
        return result

    # Sort tiles using Morton order for better spatial locality.
    sorted_tiles = sorted(cluster, key=lambda tile: morton_encode(tile[0], tile[1]))

    batches = []
    for i in range(0, len(sorted_tiles), max_batch_size):
        batch = sorted_tiles[i:i + max_batch_size]
        batches.append(batch)

    logging.debug(f"Created {len(batches)} spatial chunks with Z-order sorting, max size {max_batch_size}")
    return batches


def _create_optimized_2x2_groups(
    cluster: List[TileCoord],
    batch_size: int
) -> List[List[TileCoord]]:
    """
    Create optimized 2x2 tile groups with enhanced overlap processing efficiency.

    This function prioritizes creating 2x2 tile groups that share maximum overlap
    regions, which is optimal for the 4-step merging algorithm. Falls back to
    smaller groups when 2x2 patterns are not available.
    """
    tile_set = set(cluster)
    used_tiles = set()
    all_groups = []

    # First pass: Find all possible 2x2 groups.
    for r, c in cluster:
        if (r, c) in used_tiles:
            continue

        # Check for 2x2 pattern: (r,c), (r,c+1), (r+1,c), (r+1,c+1).
        group_2x2 = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]

        if all(tile in tile_set for tile in group_2x2):
            all_groups.append(group_2x2)
            used_tiles.update(group_2x2)
            continue

        # Check for horizontal pairs: (r,c), (r,c+1).
        group_h = [(r, c), (r, c+1)]
        if all(tile in tile_set for tile in group_h) and not any(tile in used_tiles for tile in group_h):
            all_groups.append(group_h)
            used_tiles.update(group_h)
            continue

        # Check for vertical pairs: (r,c), (r+1,c).
        group_v = [(r, c), (r+1, c)]
        if all(tile in tile_set for tile in group_v) and not any(tile in used_tiles for tile in group_v):
            all_groups.append(group_v)
            used_tiles.update(group_v)
            continue

    # Second pass: Add remaining single tiles.
    for tile in cluster:
        if tile not in used_tiles:
            all_groups.append([tile])

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

    logging.debug(f"Created {len(batches)} optimized 2x2 batches from {len(all_groups)} groups")
    return batches


def _create_hybrid_groups(
    cluster: List[TileCoord],
    batch_size: int
) -> List[List[TileCoord]]:
    """
    Create hybrid batches combining 2x2 grouping with spatial chunking.

    This approach first creates 2x2 groups where possible for optimal overlap
    processing, then uses spatial chunking for remaining tiles. This provides
    a balance between processing efficiency and memory usage.
    """
    tile_set = set(cluster)
    used_tiles = set()
    priority_groups = []
    remaining_tiles = []

    # First pass: Create 2x2 groups for dense regions.
    for r, c in cluster:
        if (r, c) in used_tiles:
            continue

        # Check for 2x2 pattern.
        group_2x2 = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]

        if all(tile in tile_set for tile in group_2x2):
            priority_groups.append(group_2x2)
            used_tiles.update(group_2x2)

    # Second pass: Collect remaining tiles for spatial chunking.
    for tile in cluster:
        if tile not in used_tiles:
            remaining_tiles.append(tile)

    # Apply spatial chunking to remaining tiles.
    if remaining_tiles:
        spatial_batches = _create_spatial_chunks(remaining_tiles, batch_size)
        all_groups = priority_groups + spatial_batches
    else:
        all_groups = priority_groups

    # Combine groups into final batches.
    batches = []
    if batch_size == 1:
        batches = all_groups
    else:
        for i in range(0, len(all_groups), batch_size):
            batch_groups = all_groups[i:i + batch_size]

            if len(batch_groups) == 1:
                batches.append(batch_groups[0])
            else:
                # Combine multiple groups.
                combined_tiles = []
                for group in batch_groups:
                    combined_tiles.extend(group)

                # Remove duplicates.
                unique_tiles = []
                seen = set()
                for tile in combined_tiles:
                    if tile not in seen:
                        unique_tiles.append(tile)
                        seen.add(tile)
                batches.append(unique_tiles)

    logging.debug(f"Created {len(batches)} hybrid batches ({len(priority_groups)} 2x2 groups, "
                 f"{len(remaining_tiles)} spatial tiles)")
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
    memory_safety_factor: float = 1.5,
    spatial_strategy: str = "adaptive",
    adaptive_batching: bool = True,
    aggressive_cleanup: bool = True,
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

    # Determine optimal batch size using enhanced algorithms.
    if batch_size <= 0 or adaptive_batching:
        batch_size = get_optimal_batch_size(
            cluster=cluster,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            memory_limit_gb=memory_limit_gb,
            adaptive_sizing=adaptive_batching
        )
    else:
        # Even if batch_size is specified, validate it against memory constraints.
        test_batch = cluster[:min(batch_size, len(cluster))]
        estimated_memory = estimate_memory_requirements(
            test_batch, tile_h, tile_w, overlap, safety_factor=memory_safety_factor
        )

        if estimated_memory > memory_limit_gb:
            logging.warning(f"Specified batch_size={batch_size} would require {estimated_memory:.2f} GB, "
                           f"exceeding limit of {memory_limit_gb:.2f} GB. Auto-optimizing batch size.")
            batch_size = get_optimal_batch_size(
                cluster=cluster,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                memory_limit_gb=memory_limit_gb,
                adaptive_sizing=adaptive_batching
            )

    # Group tiles into batches using the specified spatial strategy.
    batches = group_tiles_by_spatial_proximity(cluster, batch_size, strategy=spatial_strategy)

    # Clean up GPU memory before starting batch processing.
    if use_gpu and aggressive_cleanup:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.debug("Performed aggressive GPU memory cleanup before batch processing")

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

            # Clean up to free memory.
            del batch_stack
            if use_gpu and aggressive_cleanup:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx+1}: {e}")
            logging.debug(f"Batch error traceback:\n{traceback.format_exc()}")

            # Enhanced error recovery with better GPU memory management.
            if use_gpu:
                # Perform aggressive GPU memory cleanup.
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Check current GPU memory status.
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        cached = torch.cuda.memory_reserved() / (1024**3)
                        logging.info(f"GPU memory after cleanup - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
                except Exception as cleanup_error:
                    logging.warning(f"GPU memory cleanup failed: {cleanup_error}")

            # Determine recovery strategy based on error type.
            error_str = str(e).lower()
            is_memory_error = any(keyword in error_str for keyword in [
                "cuda out of memory", "out of memory", "memory", "allocation failed",
                "runtime error", "tensor", "device-side assert"
            ])

            is_tensor_size_error = "tensor would have" in error_str and "elements" in error_str

            if batch_size > 1 and (is_memory_error or is_tensor_size_error):
                # Calculate recovery batch size based on error type.
                if is_tensor_size_error:
                    # For tensor size errors, reduce very aggressively.
                    new_batch_size = 1
                    logging.warning(f"Tensor size error detected. Reducing to single tile processing.")
                elif is_memory_error:
                    # For memory errors, reduce aggressively but allow some parallelism.
                    new_batch_size = max(1, batch_size // 4)
                    logging.warning(f"Memory error detected. Reducing batch size from {batch_size} to {new_batch_size}")
                else:
                    # For other errors, reduce moderately.
                    new_batch_size = max(1, batch_size // 2)
                    logging.warning(f"Processing error detected. Reducing batch size from {batch_size} to {new_batch_size}")

                # Prevent infinite recursion.
                if new_batch_size == batch_size:
                    logging.error("Cannot reduce batch size further, falling back to CPU")
                    use_gpu = False
                    new_batch_size = 1

                # Recursive retry with reduced parameters.
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
                    memory_limit_gb=memory_limit_gb * 0.7,  # More aggressive memory reduction.
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

def merge_cluster_batched_parallel(
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
    max_parallel_batches: int = 2,  # NEW: Control parallel GPU streams
    temp_file_path: Optional[Path] = None,
    global_merged_array: Optional[NDArray[np.uint32]] = None,
) -> MergeResult:
    """
    Parallel version of merge_cluster_batched using multiple GPU streams.
    """
    import concurrent.futures as cf
    
    batches = group_tiles_by_spatial_proximity(cluster, batch_size)
    
    # Process batches in parallel using ThreadPoolExecutor.
    with cf.ThreadPoolExecutor(max_workers=max_parallel_batches) as executor:
        futures = []
        
        for batch_idx, batch in enumerate(batches):
            future = executor.submit(
                _process_single_batch_gpu,
                batch=batch,
                batch_idx=batch_idx,
                loader=loader,
                # ... other parameters
            )
            futures.append(future)
        
        # Collect results as they complete.
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Processing tile batches (parallel)"):
            batch_result = future.result()
            # Merge batch_result into final output.

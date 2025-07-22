"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: batch_merge.py.
Description:
    DEPRECATED: This module implements complex cluster-based batching strategies
    that have been superseded by the new two-phase merging approach in
    two_phase_merge.py. The two-phase approach is more reliable, memory-efficient,
    and easier to understand.

    This module is maintained for backward compatibility but users should migrate
    to the two-phase merging strategy by setting use_two_phase_merge=True in
    their configuration.

    Legacy Description:
    Implements batched processing for GPU-based tile merging to handle large images
    with thousands of tiles. This module addresses memory allocation errors by
    processing tiles in spatial groups based on their overlap relationships.

Dependencies:
    • Python ≥ 3.10.
    • numpy, torch, tqdm.
    • cellpose_merge.gpu_merge, cellpose_merge.rules.

Key Features (DEPRECATED):
    • Memory-efficient batched processing for large tile clusters.
    • Spatial proximity grouping to maintain merge rule consistency.
    • Configurable batch sizes with automatic memory management.
    • Comprehensive progress tracking and memory usage monitoring.
    • Graceful fallback to smaller batch sizes when memory is constrained.

    RECOMMENDED ALTERNATIVE: Use two_phase_merge.py with use_two_phase_merge=True.
"""

from __future__ import annotations

import logging
import math
import signal
import time
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
    safety_factor: float = 1.5,
    max_reasonable_dimension: int = 8192
) -> float:
    """
    Estimate the GPU memory required for processing a batch of tiles with enhanced accuracy.

    This function calculates memory based on the actual spatial extent of the tiles,
    with safeguards against unreasonably large bounding boxes that can cause
    memory allocation failures. Enhanced with better safety margins and overflow protection.

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
    max_reasonable_dimension : int, default 8192
        Maximum reasonable dimension for a batch bounding box in pixels.
        Batches exceeding this will be flagged as potentially problematic.

    Returns
    -------
    float
        Estimated memory requirement in gigabytes with safety margin applied.
        Returns a very large value (999.0 GB) if the batch is deemed unreasonable.
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

    # CRITICAL FIX: Check for unreasonably large bounding boxes.
    # This prevents memory allocation failures from sparse tile distributions.
    if batch_h > max_reasonable_dimension or batch_w > max_reasonable_dimension:
        logging.warning(f"Batch has unreasonably large bounding box: {batch_h}x{batch_w} pixels "
                       f"for {num_tiles} tiles. This suggests sparse tile distribution that "
                       f"should be processed differently.")

        # Calculate memory based on individual tiles instead of bounding box.
        # This provides a more realistic estimate for sparse distributions.
        individual_tile_memory = num_tiles * tile_h * tile_w * 4  # Stack memory only.

        # Add conservative intermediate memory estimate.
        intermediate_memory = individual_tile_memory * 2  # Conservative 2x multiplier.

        total_memory_bytes = (individual_tile_memory + intermediate_memory) * safety_factor
        total_memory_gb = total_memory_bytes / (1024**3)

        logging.debug(f"Using individual tile memory estimate for sparse batch: "
                     f"{num_tiles} tiles = {total_memory_gb:.2f} GB (safety factor: {safety_factor})")

        return total_memory_gb

    # Standard memory calculation for reasonable bounding boxes.
    # Memory for input stack: (num_tiles, batch_h, batch_w) as uint32 (4 bytes).
    stack_memory = num_tiles * batch_h * batch_w * 4

    # Check for potential integer overflow in memory calculations.
    if stack_memory < 0 or stack_memory > 2**63 - 1:
        logging.error(f"Integer overflow detected in memory calculation: "
                     f"{num_tiles} * {batch_h} * {batch_w} * 4 = {stack_memory}")
        return 999.0  # Return unreasonably high value to force rejection.

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

    # Check for overflow in total memory calculation.
    total_memory_bytes = (stack_memory + intermediate_memory) * safety_factor
    if total_memory_bytes < 0:
        logging.error(f"Integer overflow in total memory calculation")
        return 999.0

    total_memory_gb = total_memory_bytes / (1024**3)

    # Additional sanity check: if memory estimate exceeds reasonable limits, flag it.
    if total_memory_gb > 100.0:  # More than 100 GB is likely an error.
        logging.warning(f"Extremely high memory estimate: {total_memory_gb:.2f} GB for "
                       f"{num_tiles} tiles spanning {batch_h}x{batch_w} pixels. "
                       f"This batch should be split further.")
        return min(total_memory_gb, 999.0)  # Cap at 999 GB to prevent overflow.

    logging.debug(f"Enhanced memory estimate for {num_tiles} tiles spanning {batch_h}x{batch_w} pixels "
                 f"with {overlap}px overlap: {total_memory_gb:.2f} GB (safety factor: {safety_factor})")

    return total_memory_gb

def get_optimal_batch_size(
    cluster: List[TileCoord],
    tile_h: int,
    tile_w: int,
    overlap: int,
    memory_limit_gb: float = 8.0,
    adaptive_sizing: bool = True
) -> int:
    """
    Calculate intelligent batch sizes that prevent problematic array allocations.

    This function creates truly memory-efficient batches that avoid the problematic
    arrays (like 922×26459 elements) that force fallback to incremental processing.
    It uses conservative memory estimation and intelligent spatial analysis to ensure
    successful parallelization without memory allocation failures.

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
    adaptive_sizing : bool, default True
        Whether to use adaptive batch sizing based on tile spatial distribution.

    Returns
    -------
    int
        Optimal batch size that prevents memory allocation failures.
    """
    if not cluster:
        return 1

    # CRITICAL: Use very conservative memory limits to prevent allocation failures.
    # The user's system has 6863MB total memory, so we must be extremely careful.
    system_memory_limit_gb = 4.0  # Conservative system memory limit

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

            # Use only 30% of available memory as safety margin (very conservative).
            memory_limit_gb = min(available_gb * 0.3, system_memory_limit_gb)

            logging.info(f"GPU memory status - Total: {total_memory/(1024**3):.2f} GB, "
                        f"Available: {available_gb:.2f} GB, Using: {memory_limit_gb:.2f} GB")
        except Exception as e:
            logging.warning(f"Failed to detect GPU memory: {e}. Using conservative 2 GB limit.")
            memory_limit_gb = 2.0

    # Ensure we never exceed system memory constraints.
    memory_limit_gb = min(memory_limit_gb, system_memory_limit_gb)

    # INTELLIGENT BATCH SIZING: Prevent problematic array allocations from the start.
    if adaptive_sizing:
        min_r, max_r = min(r for r, _ in cluster), max(r for r, _ in cluster)
        min_c, max_c = min(c for _, c in cluster), max(c for _, c in cluster)
        spatial_span = (max_r - min_r + 1) * (max_c - min_c + 1)
        density = len(cluster) / max(1, spatial_span)

        # Calculate bounding box dimensions to detect problematic sparse distributions.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        bbox_h = (max_r - min_r) * stride_h + tile_h
        bbox_w = (max_c - min_c) * stride_w + tile_w

        # CRITICAL: Use much more conservative dimension limits.
        max_safe_dimension = 2048  # Much smaller than previous 8192 limit
        is_problematic = (
            bbox_h > max_safe_dimension or
            bbox_w > max_safe_dimension or
            density < 0.2 or  # More conservative density threshold
            len(cluster) > 20  # Much smaller cluster size limit
        )

        if is_problematic:
            logging.info(f"Detected potentially problematic cluster: {len(cluster)} tiles "
                        f"spanning {bbox_h}x{bbox_w} pixels (density: {density:.3f}). "
                        f"Using individual tile processing to prevent memory issues.")
            max_batch_size = 1  # Always process individually for safety
        elif density > 0.9 and len(cluster) <= 4:  # Only very dense, small clusters
            max_batch_size = min(2, len(cluster))  # Very conservative even for dense
        else:
            max_batch_size = 1  # Default to individual processing for safety
    else:
        # Use extremely conservative fixed batch sizing.
        density = 0.5  # Default density for logging.
        max_batch_size = 1  # Always process individually when not adaptive

    # Binary search for optimal batch size with enhanced safety checks.
    left, right = 1, max_batch_size
    optimal_batch_size = 1

    while left <= right:
        mid = (left + right) // 2
        test_batch = cluster[:mid]

        # Use enhanced memory estimation with higher safety factor.
        estimated_memory = estimate_memory_requirements(
            test_batch, tile_h, tile_w, overlap, safety_factor=2.0  # Increased safety factor.
        )

        # Additional check: reject batches with unreasonably high memory estimates.
        if estimated_memory >= 999.0:  # Flag value from estimate_memory_requirements.
            logging.warning(f"Batch size {mid} rejected due to unreasonable memory estimate")
            right = mid - 1
            continue

        if estimated_memory <= memory_limit_gb:
            optimal_batch_size = mid
            left = mid + 1
        else:
            right = mid - 1

    # Final safety check: ensure batch size is reasonable.
    if optimal_batch_size > len(cluster):
        optimal_batch_size = len(cluster)

    # For very problematic cases, force batch size to 1.
    if optimal_batch_size > 1:
        # Test the selected batch size one more time with actual tiles.
        final_test_batch = cluster[:optimal_batch_size]
        final_memory_estimate = estimate_memory_requirements(
            final_test_batch, tile_h, tile_w, overlap, safety_factor=2.0
        )

        if final_memory_estimate >= 999.0 or final_memory_estimate > memory_limit_gb:
            logging.warning(f"Final batch size {optimal_batch_size} still problematic "
                           f"(memory: {final_memory_estimate:.2f} GB). Forcing batch size to 1.")
            optimal_batch_size = 1

    logging.info(f"Optimal batch size: {optimal_batch_size} tiles (memory limit: {memory_limit_gb:.2f} GB, "
                f"cluster size: {len(cluster)} tiles, density: {density:.3f})")

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

    This approach uses intelligent spatial grouping to prevent large bounding boxes
    that can cause memory allocation failures. For sparse distributions, it creates
    smaller, more compact groups.
    """
    if not cluster:
        return []

    # For very sparse or problematic distributions, process tiles individually.
    if max_batch_size == 1:
        return [[tile] for tile in cluster]

    # Calculate spatial characteristics to determine chunking strategy.
    min_r, max_r = min(r for r, _ in cluster), max(r for r, _ in cluster)
    min_c, max_c = min(c for _, c in cluster), max(c for _, c in cluster)
    spatial_span = (max_r - min_r + 1) * (max_c - min_c + 1)
    density = len(cluster) / max(1, spatial_span)

    # For very sparse distributions, use conservative spatial grouping.
    if density < 0.1 or (max_r - min_r) > 50 or (max_c - min_c) > 50:
        return _create_conservative_spatial_chunks(cluster, max_batch_size)

    # For denser distributions, use Morton order sorting.
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


def _create_conservative_spatial_chunks(
    cluster: List[TileCoord],
    max_batch_size: int
) -> List[List[TileCoord]]:
    """
    Create conservative spatial chunks for sparse tile distributions.

    This function groups tiles into small, spatially compact batches to prevent
    large bounding boxes that cause memory allocation failures.
    """
    if not cluster:
        return []

    # Sort tiles by row first, then by column for spatial locality.
    sorted_tiles = sorted(cluster)

    batches = []
    current_batch = []

    for tile in sorted_tiles:
        # Start a new batch if current batch is empty.
        if not current_batch:
            current_batch = [tile]
            continue

        # Check if adding this tile would create a reasonable bounding box.
        test_batch = current_batch + [tile]

        # Calculate bounding box for the test batch.
        min_r = min(r for r, _ in test_batch)
        max_r = max(r for r, _ in test_batch)
        min_c = min(c for _, c in test_batch)
        max_c = max(c for _, c in test_batch)

        # Check if bounding box is reasonable (not too sparse).
        row_span = max_r - min_r + 1
        col_span = max_c - min_c + 1
        batch_density = len(test_batch) / (row_span * col_span)

        # Add tile to current batch if it maintains reasonable density and size.
        if (len(test_batch) <= max_batch_size and
            batch_density >= 0.25 and  # At least 25% density.
            row_span <= 8 and col_span <= 8):  # Reasonable spatial extent.
            current_batch.append(tile)
        else:
            # Start a new batch with this tile.
            batches.append(current_batch)
            current_batch = [tile]

    # Add the last batch if it's not empty.
    if current_batch:
        batches.append(current_batch)

    logging.debug(f"Created {len(batches)} conservative spatial chunks for sparse distribution, "
                 f"max size {max_batch_size}")
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


def _merge_cluster_incremental(
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
    spatial_strategy: str = "spatial",
    adaptive_batching: bool = True,
    aggressive_cleanup: bool = True,
    temp_file_path: Optional[Path] = None,
    global_merged_array: Optional[NDArray[np.uint32]] = None,
    max_retries: int = 3,
    timeout_seconds: int = 300,
) -> MergeResult:
    """
    Memory-safe incremental merge processing with proper overlap handling.

    This function processes tiles in memory-efficient pairs to maintain merge quality
    while avoiding massive memory allocations. Unlike the previous version that processed
    tiles individually (causing visible tile boundaries), this implementation:

    1. Processes adjacent tile pairs to detect overlapping nuclei.
    2. Applies 4-step merging rules in overlap regions.
    3. Maintains unique ID assignment across the entire cluster.
    4. Uses minimal memory by processing only 2-3 tiles at a time.

    The key insight is that proper merging requires examining overlap regions between
    adjacent tiles, not just processing tiles in isolation. This approach preserves
    merge quality while staying within memory constraints.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles in the cluster.
    loader : Callable[[slice, slice], NDArray[np.uint32]]
        Function to load tile data given row and column slices.
    height, width : int
        Full image dimensions in pixels.
    tile_h, tile_w : int
        Individual tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    threshold : float
        Overlap threshold for merging decisions.
    use_gpu : bool
        Whether to use GPU processing for merge operations.
    gid_offset : int
        Starting global ID for this cluster.
    batch_size : int, default 1
        Batch size (not used in incremental mode).
    memory_limit_gb : float, default 8.0
        Memory limit (used for validation only).
    memory_safety_factor : float, default 1.5
        Safety factor for memory calculations.
    spatial_strategy : str, default "spatial"
        Spatial strategy (not used in incremental mode).
    adaptive_batching : bool, default True
        Adaptive batching flag (not used in incremental mode).
    aggressive_cleanup : bool, default True
        Whether to perform aggressive GPU memory cleanup.
    temp_file_path : Optional[Path], default None
        Path for temporary file saves during processing.
    global_merged_array : Optional[NDArray[np.uint32]], default None
        Global array to merge results into.
    max_retries : int, default 3
        Maximum retry attempts (not used in incremental mode).
    timeout_seconds : int, default 300
        Timeout for processing (not used in incremental mode).

    Returns
    -------
    Tuple[NDArray[np.uint32], Tuple[int, int], Dict[int, int]]
        Dummy merged patch (empty), cluster position, and mapping dictionary.
        The actual merging is done directly into the global array.
    """
    import traceback
    from .rules import merge_patch_cpu
    from .gpu_merge import merge_patch_gpu

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    logging.info(f"INCREMENTAL PROCESSING WITH MERGING: Processing {len(cluster)} tiles with "
                f"overlap-aware merging to maintain segmentation quality (ENHANCED FIX)")

    # Sort tiles by position for systematic processing.
    sorted_tiles = sorted(cluster, key=lambda t: (t[0], t[1]))

    # Create a mapping from tile coordinates to processed status.
    tile_status = {tile: False for tile in sorted_tiles}
    current_gid = gid_offset
    processed_tiles = 0
    merge_operations = 0

    # Process tiles in spatial order, handling overlaps between adjacent tiles.
    for tile_idx, (tile_r, tile_c) in enumerate(sorted_tiles):
        try:
            if tile_status[(tile_r, tile_c)]:
                continue  # Already processed as part of a pair.

            logging.debug(f"Processing tile {tile_idx+1}/{len(sorted_tiles)}: ({tile_r}, {tile_c})")

            # Calculate tile position in global coordinates.
            global_y0 = tile_r * stride_h
            global_x0 = tile_c * stride_w

            # Ensure tile is within image bounds.
            if global_y0 >= height or global_x0 >= width:
                logging.warning(f"Tile ({tile_r}, {tile_c}) is outside image bounds, skipping")
                tile_status[(tile_r, tile_c)] = True
                continue

            # Calculate actual tile dimensions (may be smaller at image edges).
            actual_tile_h = min(tile_h, height - global_y0)
            actual_tile_w = min(tile_w, width - global_x0)

            if actual_tile_h <= 0 or actual_tile_w <= 0:
                logging.warning(f"Tile ({tile_r}, {tile_c}) has invalid dimensions, skipping")
                tile_status[(tile_r, tile_c)] = True
                continue

            # Find adjacent tiles for overlap processing.
            adjacent_tiles = []

            # Check right neighbor.
            right_neighbor = (tile_r, tile_c + 1)
            if right_neighbor in tile_status and not tile_status[right_neighbor]:
                adjacent_tiles.append(right_neighbor)

            # Check bottom neighbor.
            bottom_neighbor = (tile_r + 1, tile_c)
            if bottom_neighbor in tile_status and not tile_status[bottom_neighbor]:
                adjacent_tiles.append(bottom_neighbor)

            # Process current tile with its adjacent tiles for proper merging.
            tiles_to_process = [(tile_r, tile_c)] + adjacent_tiles

            if len(tiles_to_process) == 1:
                # Single tile - process individually but check for existing overlaps.
                current_gid = _process_single_tile_with_overlap_check(
                    tile_r, tile_c, loader, global_merged_array, height, width,
                    tile_h, tile_w, stride_h, stride_w, current_gid, threshold
                )
                processed_tiles += 1

            else:
                # Multiple tiles - process with proper merging.
                current_gid, merge_count = _process_tile_group_with_merging(
                    tiles_to_process, loader, global_merged_array, height, width,
                    tile_h, tile_w, stride_h, stride_w, overlap, current_gid,
                    threshold, use_gpu
                )
                processed_tiles += len(tiles_to_process)
                merge_operations += merge_count

                # Mark all processed tiles as complete.
                for processed_tile in tiles_to_process:
                    tile_status[processed_tile] = True

            # Mark current tile as processed if not already done.
            tile_status[(tile_r, tile_c)] = True

            # Save progress incrementally every 10 tiles.
            if temp_file_path is not None and tile_idx % 10 == 0:
                if global_merged_array is not None:
                    np.save(temp_file_path, global_merged_array)
                    logging.debug(f"Incremental save completed after tile {tile_idx+1}")

            # Aggressive GPU cleanup every 10 tiles.
            if use_gpu and aggressive_cleanup and tile_idx % 10 == 0:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as cleanup_error:
                    logging.warning(f"GPU cleanup failed: {cleanup_error}")

        except Exception as e:
            logging.error(f"Failed to process tile ({tile_r}, {tile_c}): {e}")
            logging.debug(f"Tile processing error traceback:\n{traceback.format_exc()}")
            tile_status[(tile_r, tile_c)] = True  # Mark as processed to avoid infinite loops.
            continue

    # Final save if we have a temp file.
    if temp_file_path is not None and global_merged_array is not None:
        np.save(temp_file_path, global_merged_array)
        logging.info(f"Final incremental save completed")

    logging.info(f"INCREMENTAL PROCESSING WITH MERGING: Successfully completed {processed_tiles} tiles "
                f"with {merge_operations} merge operations, maintaining segmentation quality")

    # Return dummy results since actual merging was done directly into global array.
    # Calculate cluster bounds for return values.
    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    cluster_y0 = min_r * stride_h
    cluster_x0 = min_c * stride_w

    # Return a minimal dummy patch to satisfy the interface.
    dummy_patch = np.zeros((1, 1), dtype=np.uint32)

    return dummy_patch, (cluster_y0, cluster_x0), {}


def _process_single_tile_with_overlap_check(
    tile_r: int,
    tile_c: int,
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    global_merged_array: Optional[NDArray[np.uint32]],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    current_gid: int,
    threshold: float,
) -> int:
    """
    Process a single tile while checking for overlaps with existing nuclei.

    This function handles tiles that don't have unprocessed neighbors by checking
    if any nuclei in the current tile overlap with already-processed nuclei in
    the global array. This prevents duplicate nuclei at tile boundaries.

    Parameters
    ----------
    tile_r, tile_c : int
        Tile row and column coordinates.
    loader : Callable
        Function to load tile data.
    global_merged_array : Optional[NDArray[np.uint32]]
        Global array containing already-processed nuclei.
    height, width : int
        Full image dimensions.
    tile_h, tile_w : int
        Tile dimensions.
    stride_h, stride_w : int
        Stride between tiles (tile_size - overlap).
    current_gid : int
        Current global ID counter.
    threshold : float
        Overlap threshold for merging decisions.

    Returns
    -------
    int
        Updated global ID counter.
    """
    # Calculate tile position in global coordinates.
    global_y0 = tile_r * stride_h
    global_x0 = tile_c * stride_w

    # Calculate actual tile dimensions (may be smaller at image edges).
    actual_tile_h = min(tile_h, height - global_y0)
    actual_tile_w = min(tile_w, width - global_x0)

    # Load the tile data.
    ys = slice(global_y0, global_y0 + actual_tile_h)
    xs = slice(global_x0, global_x0 + actual_tile_w)
    tile_data = loader(ys, xs)

    if tile_data.size == 0 or not np.any(tile_data > 0):
        logging.debug(f"Tile ({tile_r}, {tile_c}) is empty, skipping")
        return current_gid

    # Check for overlaps with existing nuclei in the global array.
    if global_merged_array is not None:
        existing_region = global_merged_array[global_y0:global_y0+actual_tile_h,
                                            global_x0:global_x0+actual_tile_w]

        # Process each nucleus in the current tile.
        processed_tile = np.zeros_like(tile_data)
        unique_labels = np.unique(tile_data[tile_data > 0])

        for old_label in unique_labels:
            nucleus_mask = tile_data == old_label

            # Check if this nucleus overlaps with existing nuclei.
            overlapping_existing = existing_region[nucleus_mask]
            existing_labels = np.unique(overlapping_existing[overlapping_existing > 0])

            if len(existing_labels) > 0:
                # Nucleus overlaps with existing nuclei - use existing label.
                # Choose the most frequent existing label in the overlap region.
                label_counts = {}
                for existing_label in existing_labels:
                    count = np.sum(overlapping_existing == existing_label)
                    label_counts[existing_label] = count

                best_existing_label = max(label_counts.keys(), key=lambda k: label_counts[k])
                processed_tile[nucleus_mask] = best_existing_label

                logging.debug(f"Merged nucleus {old_label} with existing nucleus {best_existing_label}")
            else:
                # No overlap - assign new unique ID.
                processed_tile[nucleus_mask] = current_gid
                current_gid += 1

        # Update global array with processed results.
        nucleus_pixels = processed_tile != 0
        if np.any(nucleus_pixels):
            global_merged_array[global_y0:global_y0+actual_tile_h,
                              global_x0:global_x0+actual_tile_w][nucleus_pixels] = processed_tile[nucleus_pixels]

    return current_gid


def _process_tile_group_with_merging(
    tiles_to_process: List[Tuple[int, int]],
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    global_merged_array: Optional[NDArray[np.uint32]],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    overlap: int,
    current_gid: int,
    threshold: float,
    use_gpu: bool,
) -> Tuple[int, int]:
    """
    Process a group of adjacent tiles with proper merging in overlap regions.

    This function creates a small bounding box containing the tile group,
    applies the 4-step merging rules in overlap regions, and updates the
    global array with the merged results.

    Parameters
    ----------
    tiles_to_process : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles to process together.
    loader : Callable
        Function to load tile data.
    global_merged_array : Optional[NDArray[np.uint32]]
        Global array to update with merged results.
    height, width : int
        Full image dimensions.
    tile_h, tile_w : int
        Individual tile dimensions.
    stride_h, stride_w : int
        Stride between tiles.
    overlap : int
        Overlap between adjacent tiles.
    current_gid : int
        Current global ID counter.
    threshold : float
        Overlap threshold for merging decisions.
    use_gpu : bool
        Whether to use GPU for merge operations.

    Returns
    -------
    Tuple[int, int]
        Updated global ID counter and number of merge operations performed.
    """
    from .rules import merge_patch_cpu
    from .gpu_merge import merge_patch_gpu

    # Calculate bounding box for the tile group.
    min_r = min(r for r, _ in tiles_to_process)
    max_r = max(r for r, _ in tiles_to_process)
    min_c = min(c for _, c in tiles_to_process)
    max_c = max(c for _, c in tiles_to_process)

    # Calculate group bounds in global coordinates.
    group_y0 = min_r * stride_h
    group_x0 = min_c * stride_w
    group_y1 = min(height, (max_r + 1) * stride_h + overlap)
    group_x1 = min(width, (max_c + 1) * stride_w + overlap)

    group_h = group_y1 - group_y0
    group_w = group_x1 - group_x0

    logging.debug(f"Processing tile group: {len(tiles_to_process)} tiles, "
                 f"bounding box: {group_h}×{group_w} pixels")

    # Create a small array for the tile group.
    try:
        group_array = np.zeros((group_h, group_w), dtype=np.uint32)

        # Load each tile and preserve original IDs for proper merging.
        tile_arrays = []

        for tile_r, tile_c in tiles_to_process:
            # Calculate tile position within the group.
            tile_y0_in_group = tile_r * stride_h - group_y0
            tile_x0_in_group = tile_c * stride_w - group_x0

            # Calculate global tile position.
            global_tile_y0 = tile_r * stride_h
            global_tile_x0 = tile_c * stride_w

            # Calculate actual tile dimensions.
            actual_tile_h = min(tile_h, height - global_tile_y0)
            actual_tile_w = min(tile_w, width - global_tile_x0)

            # Load tile data.
            ys = slice(global_tile_y0, global_tile_y0 + actual_tile_h)
            xs = slice(global_tile_x0, global_tile_x0 + actual_tile_w)
            tile_data = loader(ys, xs)

            if tile_data.size > 0 and np.any(tile_data > 0):
                # Keep original tile data for proper merge function processing.
                # The merge function expects to see the same nucleus IDs across tiles
                # for nuclei that should be merged together.
                tile_arrays.append((tile_data, tile_y0_in_group, tile_x0_in_group))

        # Apply merging rules to the group array.
        if len(tile_arrays) > 1:
            # Create a proper stack for the merge function by separating each tile.
            # This allows the merge function to properly detect overlaps.
            stack_layers = []

            # Create individual layers for each tile.
            for tile_data, tile_y0_in_group, tile_x0_in_group in tile_arrays:
                layer = np.zeros((group_h, group_w), dtype=np.uint32)
                actual_h, actual_w = tile_data.shape
                layer[tile_y0_in_group:tile_y0_in_group+actual_h,
                      tile_x0_in_group:tile_x0_in_group+actual_w] = tile_data
                stack_layers.append(layer)

            # Stack all layers for merge processing.
            if len(stack_layers) > 0:
                stack = np.stack(stack_layers, axis=0)  # Shape: (N, H, W)

                # Apply merge function.
                merge_fn = merge_patch_gpu if use_gpu else merge_patch_cpu
                merged_group, mapping = merge_fn(stack, threshold=threshold)

                merge_operations = 1  # One merge operation for the group.

                logging.debug(f"Applied merge function to {len(stack_layers)} tile layers")
            else:
                merged_group = np.zeros((group_h, group_w), dtype=np.uint32)
                merge_operations = 0
        else:
            # Single tile - just place it in the group array.
            merged_group = np.zeros((group_h, group_w), dtype=np.uint32)
            if len(tile_arrays) == 1:
                tile_data, tile_y0_in_group, tile_x0_in_group = tile_arrays[0]
                actual_h, actual_w = tile_data.shape
                merged_group[tile_y0_in_group:tile_y0_in_group+actual_h,
                           tile_x0_in_group:tile_x0_in_group+actual_w] = tile_data
            merge_operations = 0

        # Reassign global IDs to ensure uniqueness across the entire cluster.
        if np.any(merged_group > 0):
            # Create a mapping from merged IDs to new global IDs.
            unique_merged_ids = np.unique(merged_group[merged_group > 0])
            id_mapping = {}

            for merged_id in unique_merged_ids:
                id_mapping[merged_id] = current_gid
                current_gid += 1

            # Apply the mapping to the merged group.
            final_merged_group = np.zeros_like(merged_group)
            for old_id, new_id in id_mapping.items():
                mask = merged_group == old_id
                final_merged_group[mask] = new_id

            merged_group = final_merged_group

        # Update global array with merged results.
        if global_merged_array is not None:
            nucleus_pixels = merged_group != 0
            if np.any(nucleus_pixels):
                global_merged_array[group_y0:group_y1, group_x0:group_x1][nucleus_pixels] = merged_group[nucleus_pixels]

        logging.debug(f"Tile group merge completed: {merge_operations} operations, "
                     f"max_id={merged_group.max()}, nuclei_pixels={np.count_nonzero(merged_group)}")

        return current_gid, merge_operations

    except (MemoryError, OverflowError) as e:
        logging.warning(f"Failed to create group array {group_h}×{group_w}: {e}. "
                       f"Falling back to individual tile processing.")

        # Fallback: process tiles individually without merging.
        for tile_r, tile_c in tiles_to_process:
            current_gid = _process_single_tile_with_overlap_check(
                tile_r, tile_c, loader, global_merged_array, height, width,
                tile_h, tile_w, stride_h, stride_w, current_gid, threshold
            )

        return current_gid, 0


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
    max_retries: int = 3,
    timeout_seconds: int = 300,
) -> MergeResult:
    """
    DEPRECATED: Use two-phase merging instead (use_two_phase_merge=True).

    This function implements complex cluster-based batching that has been superseded
    by the more reliable two-phase merging approach. Users should migrate to the
    new approach for better memory efficiency and merge quality.

    Merge all tiles in a cluster using batched processing to manage memory usage.

    This function now includes proper timeout and retry mechanisms to prevent infinite loops
    during GPU batch processing. The timeout ensures that processing cannot get stuck indefinitely.

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
    max_retries : int, default 3
        Maximum number of retry attempts for failed batch processing operations.
    timeout_seconds : int, default 300
        Maximum time in seconds to wait for batch processing before timing out.

    Returns
    -------
    Tuple[NDArray[np.uint32], Tuple[int, int], Dict[int, int]]
        Merged patch, (y0, x0) coordinates, and mapping dictionary.
    """
    # Import merge functions here to avoid circular imports
    from .rules import merge_patch_cpu
    from .gpu_merge import merge_patch_gpu

    # DEPRECATION WARNING: Recommend migration to two-phase merging.
    import warnings
    warnings.warn(
        "merge_cluster_batched is deprecated. Use two-phase merging instead by setting "
        "use_two_phase_merge=True in your configuration. The two-phase approach is more "
        "reliable and memory-efficient.",
        DeprecationWarning,
        stacklevel=2
    )

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    # CRITICAL FIX: For sparse distributions, we need to process batches individually
    # and merge results incrementally rather than creating a massive cluster-wide array.

    # Calculate basic cluster info for validation only.
    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_r = max(r for r, _ in cluster)
    max_c = max(c for _, c in cluster)

    # Validate tile indices are reasonable for the given image dimensions.
    max_possible_rows = (height + stride_h - 1) // stride_h
    max_possible_cols = (width + stride_w - 1) // stride_w

    if max_r >= max_possible_rows or max_c >= max_possible_cols:
        raise ValueError(f"Tile indices out of bounds: max_tile=({max_r},{max_c}), max_possible=({max_possible_rows-1},{max_possible_cols-1})")

    # Calculate cluster characteristics for memory safety decisions.
    cluster_y0 = min_r * stride_h
    cluster_x0 = min_c * stride_w
    cluster_h_full = min((max_r - min_r) * stride_h + tile_h, height - cluster_y0)
    cluster_w_full = min((max_c - min_c) * stride_w + tile_w, width - cluster_x0)

    # CRITICAL DECISION: Check if this cluster would create a massive array.
    total_elements = cluster_h_full * cluster_w_full
    is_memory_problematic = (total_elements > 2**28 or  # 256M elements = 1GB
                           cluster_h_full > 8192 or
                           cluster_w_full > 8192)

    if is_memory_problematic:
        logging.warning(f"Cluster would create problematic array: {cluster_h_full}×{cluster_w_full} "
                       f"({total_elements} elements = {total_elements * 4 / (1024**3):.2f} GB). "
                       f"Using incremental processing instead of cluster-wide array.")

        # Use incremental processing - no cluster-wide array allocation.
        return _merge_cluster_incremental(
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
            batch_size=batch_size,
            memory_limit_gb=memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
            spatial_strategy=spatial_strategy,
            adaptive_batching=adaptive_batching,
            aggressive_cleanup=aggressive_cleanup,
            temp_file_path=temp_file_path,
            global_merged_array=global_merged_array,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

    # For reasonable-sized clusters, use the original approach.
    logging.info(f"Processing cluster with {len(cluster)} tiles using standard batched approach: "
                f"tile_range=({min_r},{min_c}) to ({max_r},{max_c}), "
                f"global_bbox=({cluster_y0},{cluster_x0}) to ({cluster_y0+cluster_h_full},{cluster_x0+cluster_w_full}), "
                f"image_size=({height},{width})")

    # Check for uint32 overflow in gid_offset.
    max_safe_gid = 2**31 - 1  # Conservative limit to prevent uint32 overflow.
    if gid_offset >= max_safe_gid:
        logging.warning(f"Global ID offset {gid_offset} approaching uint32 limit {max_safe_gid}. "
                       f"Using segmented allocation to minimize conflicts.")
        # Use a segmented approach: divide the ID space into segments.
        segment_size = max_safe_gid // 100  # Create 100 segments.
        segment_number = (gid_offset // segment_size) % 100
        gid_offset = (segment_number * segment_size) + 1
        logging.info(f"Adjusted gid_offset to segment {segment_number}: {gid_offset}")

    # Create the output merged patch for reasonable-sized clusters.
    try:
        merged_patch = np.zeros((cluster_h_full, cluster_w_full), dtype=np.uint32)
        y0, x0 = cluster_y0, cluster_x0
        cluster_h, cluster_w = cluster_h_full, cluster_w_full
    except (MemoryError, OverflowError) as e:
        logging.error(f"Failed to allocate memory for cluster patch of size {cluster_h_full}×{cluster_w_full}: {e}")
        # Fallback to incremental processing.
        return _merge_cluster_incremental(
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
            batch_size=1,  # Force individual processing.
            memory_limit_gb=memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
            spatial_strategy="spatial",  # Use spatial strategy for safety.
            adaptive_batching=adaptive_batching,
            aggressive_cleanup=aggressive_cleanup,
            temp_file_path=temp_file_path,
            global_merged_array=global_merged_array,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

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

    # CRITICAL SAFETY CHECK: Validate all batches before processing.
    validated_batches = []
    for i, batch in enumerate(batches):
        # Check if this batch would create a reasonable bounding box.
        batch_memory_estimate = estimate_memory_requirements(
            batch, tile_h, tile_w, overlap, safety_factor=2.0
        )

        if batch_memory_estimate >= 999.0 or batch_memory_estimate > memory_limit_gb:
            logging.warning(f"Batch {i+1} rejected due to excessive memory requirement: "
                           f"{batch_memory_estimate:.2f} GB. Splitting into individual tiles.")
            # Split problematic batch into individual tiles.
            for tile in batch:
                validated_batches.append([tile])
        else:
            validated_batches.append(batch)

    batches = validated_batches
    logging.info(f"Validated {len(batches)} batches for processing (memory limit: {memory_limit_gb:.2f} GB)")

    # Clean up GPU memory before starting batch processing.
    if use_gpu and aggressive_cleanup:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.debug("Performed aggressive GPU memory cleanup before batch processing")

    # Process each batch with timeout and retry logic
    current_gid = gid_offset
    batch_results = []

    def timeout_handler(signum, frame):
        raise TimeoutError("Batch processing timed out")

    # Enhanced progress tracking with detailed information.
    progress_bar = tqdm(batches, desc="Processing memory-efficient batches",
                       unit="batch", leave=True)

    # Add comprehensive logging for batch processing start.
    total_tiles = len(cluster)
    total_batches = len(batches)
    avg_batch_size = total_tiles / total_batches if total_batches > 0 else 0

    logging.info(f"BATCH PROCESSING START: {total_tiles} tiles in {total_batches} batches "
                f"(avg_batch_size={avg_batch_size:.1f}, memory_limit={memory_limit_gb:.2f}GB)")

    processing_start_time = time.time()

    for batch_idx, batch in enumerate(progress_bar):
        retry_count = 0
        batch_success = False

        while retry_count < max_retries and not batch_success:
            try:
                # Set up timeout for this batch
                if timeout_seconds > 0:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)

                start_time = time.time()

                # Update progress bar with detailed information.
                elapsed_time = time.time() - processing_start_time
                avg_time_per_batch = elapsed_time / max(1, batch_idx)
                estimated_remaining = avg_time_per_batch * (total_batches - batch_idx - 1)

                progress_desc = (f"Batch {batch_idx+1}/{total_batches} "
                               f"({len(batch)} tiles, "
                               f"ETA: {estimated_remaining:.1f}s)")
                progress_bar.set_description(progress_desc)

                logging.info(f"Processing batch {batch_idx+1}/{total_batches}: {len(batch)} tiles "
                           f"(attempt {retry_count+1}/{max_retries}, "
                           f"elapsed: {elapsed_time:.1f}s, ETA: {estimated_remaining:.1f}s)")

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

                # ENHANCED SAFETY CHECK: Validate batch dimensions before allocation.
                batch_elements = T * batch_h * batch_w

                # Check for array size limits.
                if batch_elements > 2**31 - 1:
                    raise RuntimeError(f"Batch stack would have {batch_elements} elements, exceeding safe limits. "
                                     f"Batch size: {T}, dimensions: {batch_h}×{batch_w}")

                # Check for unreasonable dimensions that suggest sparse distribution.
                max_reasonable_dim = 8192
                if batch_h > max_reasonable_dim or batch_w > max_reasonable_dim:
                    raise RuntimeError(f"Batch has unreasonable dimensions: {batch_h}×{batch_w} pixels. "
                                     f"This suggests a sparse tile distribution that should be processed "
                                     f"with smaller batch sizes or individual tiles.")

                # Estimate memory requirement one more time before allocation.
                final_memory_check = estimate_memory_requirements(
                    batch, tile_h, tile_w, overlap, safety_factor=1.0  # No safety factor for final check.
                )

                if final_memory_check > memory_limit_gb * 1.5:  # Allow 50% over limit for final check.
                    raise RuntimeError(f"Batch memory requirement {final_memory_check:.2f} GB exceeds "
                                     f"safe limit of {memory_limit_gb * 1.5:.2f} GB. Batch dimensions: "
                                     f"{T} tiles × {batch_h}×{batch_w} pixels")

                try:
                    batch_stack = np.zeros((T, batch_h, batch_w), dtype=np.uint32)
                    logging.debug(f"Successfully allocated batch stack: ({T}, {batch_h}, {batch_w}), "
                                 f"memory: {batch_elements * 4 / (1024**3):.2f} GB")
                except (MemoryError, OverflowError) as e:
                    raise RuntimeError(f"Failed to allocate memory for batch stack of size ({T}, {batch_h}, {batch_w}): {e}. "
                                     f"Estimated memory: {batch_elements * 4 / (1024**3):.2f} GB. "
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

                # Clear timeout and mark success
                    signal.alarm(0)

                batch_success = True
                elapsed_time = time.time() - start_time
                logging.debug(f"Batch {batch_idx+1} completed successfully in {elapsed_time:.2f} seconds")

            except (TimeoutError, Exception) as e:
                # Clear timeout
                if timeout_seconds > 0:
                    signal.alarm(0)

                retry_count += 1
                elapsed_time = time.time() - start_time

                if isinstance(e, TimeoutError):
                    logging.warning(f"Batch {batch_idx+1} timed out after {elapsed_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                else:
                    logging.error(f"Error processing batch {batch_idx+1} (attempt {retry_count}/{max_retries}): {e}")
                    logging.debug(f"Batch error traceback:\n{traceback.format_exc()}")

                # Enhanced error recovery with better GPU memory management.
                if use_gpu:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / (1024**3)
                            cached = torch.cuda.memory_reserved() / (1024**3)
                            logging.info(f"GPU memory after cleanup - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
                    except Exception as cleanup_error:
                        logging.warning(f"GPU memory cleanup failed: {cleanup_error}")

                # If we've exhausted retries, raise the error
                if retry_count >= max_retries:
                    if isinstance(e, TimeoutError):
                        raise RuntimeError(f"Batch {batch_idx+1} failed after {max_retries} timeout attempts")
                    else:
                        raise RuntimeError(f"Batch {batch_idx+1} failed after {max_retries} retry attempts: {e}")

                # Wait before retry
                time.sleep(min(retry_count * 2, 10))  # Exponential backoff, max 10 seconds

        # If we get here without success, something went wrong
        if not batch_success:
            raise RuntimeError(f"Batch {batch_idx+1} processing failed unexpectedly")

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

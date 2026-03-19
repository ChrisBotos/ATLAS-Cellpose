"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_memory.py.
Description:
    Memory management utilities for tile mask merging in tissue analysis.
    This module provides accurate memory estimation, feasibility checking, and adaptive
    cluster splitting strategies to prevent memory allocation failures during large
    image processing.

    Key Features:
    - Accurate memory estimation for sparse cluster processing.
    - Realistic memory calculations based on actual tile data rather than bounding boxes.
    - Sparsity detection to handle clusters with few tiles spread across large areas.
    - Overflow protection for uint32/uint64 data types.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations and memory calculations.

Usage:
    from utils.merge_memory import estimate_cluster_requirements, check_cluster_feasibility
    
    memory_gb, dimensions = estimate_cluster_requirements(cluster, tile_h, tile_w, overlap)
    is_feasible, reason = check_cluster_feasibility(cluster, tile_h, tile_w, overlap, height, width)

Arguments:
    None (utility functions only).

Inputs:
    - Tile cluster coordinates and processing parameters.
    - Image dimensions and memory constraints.

Outputs:
    - Accurate memory requirement estimates in gigabytes.
    - Feasibility assessments with detailed reasoning.
    - Adaptive cluster splitting recommendations.

Key Features:
    - Realistic memory estimation for sparse and dense tile clusters.
    - Comprehensive feasibility checking with multiple constraint validation.
    - Adaptive splitting strategies for memory-constrained processing.
    - Enhanced debugging and diagnostic information.

Notes:
    - Addresses the critical bug where sparse clusters caused massive memory estimates.
    - Implements conservative safety factors to prevent out-of-memory errors.
    - Provides detailed logging for troubleshooting memory issues.
"""

from __future__ import annotations

import logging
import traceback
from typing import List, Tuple

import numpy as np


def estimate_cluster_requirements(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> Tuple[float, Tuple[int, int]]:
    """
    Estimate memory requirements and bounding box dimensions for a tile cluster.
    
    CRITICAL FIX: This function now correctly handles sparse clusters by calculating
    memory based on actual tile processing requirements rather than bounding box
    dimensions. This prevents the 200+ GB allocation attempts that caused failures.
    
    Args:
        cluster (List[Tuple[int, int]]): List of (row, col) coordinates for tiles in the cluster.
        tile_h (int): Tile height in pixels.
        tile_w (int): Tile width in pixels.
        overlap (int): Overlap between adjacent tiles in pixels.

    Returns:
        Tuple[float, Tuple[int, int]]: Memory requirement in GB and (height, width) dimensions.

    Notes:
        The memory calculation now uses a hybrid approach:
        - For dense clusters: Uses bounding box approach (traditional method).
        - For sparse clusters: Uses tile-based calculation to prevent massive overestimates.
    """
    if not cluster:
        return 0.0, (0, 0)
    
    try:
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        min_r = min(r for r, _ in cluster)
        max_r = max(r for r, _ in cluster)
        min_c = min(c for _, c in cluster)
        max_c = max(c for _, c in cluster)
        
        # Calculate bounding box dimensions.
        bbox_h = (max_r - min_r) * stride_h + tile_h
        bbox_w = (max_c - min_c) * stride_w + tile_w
        
        num_tiles = len(cluster)
        
        # CRITICAL FIX: Calculate sparsity ratio to detect problematic clusters.
        total_tile_pixels = num_tiles * tile_h * tile_w
        bbox_pixels = bbox_h * bbox_w
        sparsity_ratio = bbox_pixels / total_tile_pixels if total_tile_pixels > 0 else 1.0
        
        # Memory for actual tile data: num_tiles * tile_h * tile_w * 4 bytes (uint32).
        actual_tile_memory = num_tiles * tile_h * tile_w * 4 / (1024**3)  # GB
        
        if sparsity_ratio > 10.0:  # Sparse cluster detected.
            logging.warning(f"Sparse cluster detected: {num_tiles} tiles in {bbox_h}x{bbox_w} bbox "
                           f"(sparsity ratio: {sparsity_ratio:.1f})")
            
            # For sparse clusters, use tile-based memory calculation.
            # This prevents the massive overestimates that caused 200+ GB allocations.
            workspace_memory = actual_tile_memory * 3.0  # Conservative but realistic multiplier.
            
            # Use a more reasonable bounding box for sparse clusters.
            # Calculate the minimum bounding box needed for actual processing.
            effective_bbox_h = min(bbox_h, num_tiles * tile_h)  # Cap at reasonable size.
            effective_bbox_w = min(bbox_w, num_tiles * tile_w)
            
            logging.info(f"Sparse cluster memory calculation: actual_tiles={actual_tile_memory:.3f}GB, "
                        f"workspace={workspace_memory:.3f}GB, effective_bbox={effective_bbox_h}x{effective_bbox_w}")
            
            # Return effective dimensions, not the massive bounding box.
            return (actual_tile_memory + workspace_memory) * 1.2, (effective_bbox_h, effective_bbox_w)
        else:
            # For dense clusters, use traditional bounding box approach.
            workspace_memory = bbox_h * bbox_w * 4 / (1024**3)  # GB
            total_memory = (actual_tile_memory + workspace_memory) * 1.5  # Reasonable safety factor.
            
            logging.debug(f"Dense cluster memory calculation: tiles={actual_tile_memory:.3f}GB, "
                         f"workspace={workspace_memory:.3f}GB, total={total_memory:.3f}GB")
            
            return total_memory, (bbox_h, bbox_w)
            
    except Exception as e:
        logging.error(f"Error estimating cluster requirements: {e}")
        logging.debug(f"Memory estimation error traceback:\n{traceback.format_exc()}")
        # Return conservative estimates on error.
        return 1.0, (tile_h, tile_w)


def check_cluster_feasibility(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    height: int,
    width: int,
    memory_limit_gb: float = 16.0
) -> Tuple[bool, str]:
    """
    Check if a cluster can be processed with standard algorithms.
    
    This function performs comprehensive feasibility checking including memory limits,
    array size constraints, coordinate validation, and overflow protection.
    
    Args:
        cluster (List[Tuple[int, int]]): List of (row, col) coordinates for tiles in the cluster.
        tile_h (int): Tile height in pixels.
        tile_w (int): Tile width in pixels.
        overlap (int): Overlap between adjacent tiles in pixels.
        height (int): Full image height in pixels.
        width (int): Full image width in pixels.
        memory_limit_gb (float): Maximum memory limit in gigabytes. Defaults to 16.0.

    Returns:
        Tuple[bool, str]: (is_feasible, reason_if_not_feasible).
    """
    try:
        cluster_size = len(cluster)
        
        # Check for uint32 overflow in composite keys (tile index << 32).
        if cluster_size >= (1 << 32):
            return False, f"Cluster has {cluster_size} tiles, exceeding uint32 composite key limit"
        
        # Calculate cluster bounding box with proper coordinate validation.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        min_r = min(r for r, _ in cluster)
        min_c = min(c for _, c in cluster)
        max_r = max(r for r, _ in cluster)
        max_c = max(c for _, c in cluster)
        
        # Validate tile indices are reasonable for the given image dimensions.
        max_possible_rows = (height + stride_h - 1) // stride_h
        max_possible_cols = (width + stride_w - 1) // stride_w
        
        if max_r >= max_possible_rows or max_c >= max_possible_cols:
            return False, f"Tile indices out of bounds: max_tile=({max_r},{max_c}), max_possible=({max_possible_rows-1},{max_possible_cols-1})"
        
        y0 = min_r * stride_h
        x0 = min_c * stride_w
        
        # Ensure starting coordinates are within image bounds.
        if y0 >= height or x0 >= width:
            return False, f"Cluster starting position ({y0},{x0}) exceeds image bounds ({height},{width})"
        
        # Calculate cluster dimensions with proper boundary clamping.
        cluster_h = min((max_r - min_r) * stride_h + tile_h, height - y0)
        cluster_w = min((max_c - min_c) * stride_w + tile_w, width - x0)
        
        # Ensure dimensions are positive and reasonable.
        if cluster_h <= 0 or cluster_w <= 0:
            return False, f"Invalid cluster dimensions: {cluster_h}×{cluster_w} (y0={y0}, x0={x0}, height={height}, width={width})"
        
        # Check for array size limits - the merged patch array size.
        merged_patch_elements = cluster_h * cluster_w
        
        # NumPy array size limit (platform dependent, but typically around 2^63-1 for 64-bit).
        max_array_elements = 2**31 - 1  # Conservative limit for compatibility.
        if merged_patch_elements > max_array_elements:
            return False, f"Merged patch would have {merged_patch_elements} elements, exceeding NumPy limit of {max_array_elements}"
        
        # Check for stack array size limits (cluster_size × tile_h × tile_w).
        stack_elements = cluster_size * tile_h * tile_w
        if stack_elements > max_array_elements:
            return False, f"Tile stack would have {stack_elements} elements, exceeding NumPy limit of {max_array_elements}"
        
        # Memory requirement check using the fixed estimation function.
        estimated_memory, _ = estimate_cluster_requirements(cluster, tile_h, tile_w, overlap)
        if estimated_memory > memory_limit_gb:
            return False, f"Estimated memory requirement {estimated_memory:.1f} GB exceeds limit of {memory_limit_gb} GB"
        
        return True, ""
        
    except Exception as e:
        logging.error(f"Error checking cluster feasibility: {e}")
        logging.debug(f"Feasibility check error traceback:\n{traceback.format_exc()}")
        return False, f"Feasibility check failed due to error: {e}"


def estimate_cluster_memory_simple(
    cluster_size: int,
    cluster_h: int,
    cluster_w: int
) -> float:
    """
    Simple memory estimation for processing a cluster (legacy compatibility).
    
    Args:
        cluster_size (int): Number of tiles in the cluster.
        cluster_h (int): Height of the cluster bounding box in pixels.
        cluster_w (int): Width of the cluster bounding box in pixels.

    Returns:
        float: Estimated memory requirement in gigabytes.
    """
    try:
        # Memory for the 3D stack: (cluster_size, cluster_h, cluster_w) as uint32 (4 bytes).
        stack_memory_bytes = cluster_size * cluster_h * cluster_w * 4
        
        # Additional memory for intermediate processing (conservative estimate: 2x).
        total_memory_bytes = stack_memory_bytes * 2
        
        # Convert to gigabytes.
        memory_gb = total_memory_bytes / (1024**3)
        
        logging.debug(f"Simple cluster memory estimate: {cluster_size} tiles, "
                     f"{cluster_h}×{cluster_w} pixels = {memory_gb:.2f} GB")
        
        return memory_gb
        
    except Exception as e:
        logging.warning(f"Error in simple memory estimation: {e}")
        return 1.0  # Conservative fallback.

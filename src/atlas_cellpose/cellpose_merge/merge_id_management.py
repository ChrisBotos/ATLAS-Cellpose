"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_id_management.py.
Description:
    ID management utilities for tile mask merging in tissue analysis.
    This module provides robust global ID allocation, overflow prevention, and
    data type management for very large images with millions of nuclei.

    Key Features:
    - Uint32/uint64 overflow protection for large image processing.
    - Segmented ID allocation strategy to minimize conflicts during counter resets.
    - Automatic data type selection based on image size and nucleus count.
    - Comprehensive logging and diagnostic information for ID management.

Dependencies:
    • Python ≥ 3.10.
    • numpy for data type operations.

Usage:
    from atlas_cellpose.cellpose_merge.merge_id_management import get_next_safe_gid_range, select_optimal_dtype
    
    new_gid, offset, was_reset = get_next_safe_gid_range(current_gid, patch_max, max_safe_gid, reset_count, segment_size)
    dtype = select_optimal_dtype(estimated_max_nuclei)

Arguments:
    None (utility functions only).

Inputs:
    - Current global ID counters and patch maximum values.
    - Image size estimates and nucleus count predictions.

Outputs:
    - Safe global ID ranges with overflow protection.
    - Optimal data types for mask storage.
    - Reset indicators and conflict prevention strategies.

Key Features:
    - Segmented ID allocation to prevent conflicts during resets.
    - Automatic uint64 upgrade for very large images.
    - Conservative safety margins to prevent overflow errors.
    - Detailed logging for ID management troubleshooting.

Notes:
    - Addresses int32/uint32 overflow issues in large image processing.
    - Implements intelligent data type selection based on image characteristics.
    - Provides comprehensive error handling and recovery strategies.
"""

from __future__ import annotations

import logging
import traceback
from typing import Tuple, Union

import numpy as np


def get_next_safe_gid_range(
    current_gid: int,
    patch_max: int,
    max_safe_gid: int,
    reset_count: int,
    segment_size: int
) -> Tuple[int, int, bool]:
    """
    Calculate the next safe global ID range to prevent uint32 overflow.
    
    This function implements a segmented ID allocation strategy that reduces
    the likelihood of ID conflicts when counter resets are necessary. It's
    designed to handle very large images with millions of nuclei.
    
    Args:
        current_gid (int): Current global ID counter value.
        patch_max (int): Maximum ID value in the current patch.
        max_safe_gid (int): Maximum safe ID value before overflow risk.
        reset_count (int): Number of times the counter has been reset.
        segment_size (int): Size of each ID segment to prevent conflicts.

    Returns:
        Tuple[int, int, bool]: (new_gid_counter, gid_offset, was_reset)
            - new_gid_counter: Updated global ID counter.
            - gid_offset: Offset to apply to patch IDs.
            - was_reset: Whether a reset occurred.
        
    Notes
    -----
    The segmented approach helps prevent ID conflicts by allocating distinct
    ranges for each reset cycle, improving reliability for large image processing.
    """
    try:
        # Check if adding patch_max would exceed the safe limit.
        if current_gid + patch_max > max_safe_gid:
            # Calculate the next segment start to avoid conflicts.
            next_segment_start = (reset_count + 1) * segment_size + 1
            
            # Ensure we don't exceed the absolute uint32 limit.
            if next_segment_start + patch_max > 2**32 - 1:
                logging.error(f"Exhausted all available uint32 ID space. "
                             f"Current segment: {reset_count + 1}, next_start: {next_segment_start}, "
                             f"patch_max: {patch_max}, uint32_limit: {2**32 - 1}")
                logging.error(f"Consider using uint64 data type or reducing image size.")
                
                # Fall back to simple reset as last resort.
                next_segment_start = 1
                logging.warning(f"Falling back to simple reset at ID 1")
            
            logging.warning(f"Global ID counter approaching uint32 limit. "
                           f"Current: {current_gid:,}, patch_max: {patch_max:,}, limit: {max_safe_gid:,}")
            logging.info(f"Resetting to segment {reset_count + 1} starting at ID {next_segment_start:,} "
                        f"to minimize conflicts.")
            
            return next_segment_start + patch_max, next_segment_start, True
        else:
            # Normal case: no reset needed.
            return current_gid + patch_max, current_gid, False
            
    except Exception as e:
        logging.error(f"Error calculating next safe GID range: {e}")
        logging.debug(f"GID range calculation error traceback:\n{traceback.format_exc()}")
        # Return safe fallback values.
        return current_gid + 1, current_gid, False


def select_optimal_dtype(
    estimated_max_nuclei: int,
    image_size_pixels: int = 0,
    safety_factor: float = 2.0
) -> np.dtype:
    """
    Select optimal data type for mask storage based on expected nucleus count.
    
    This function automatically selects between uint16, uint32, and uint64 based
    on the estimated maximum number of nuclei in the image, preventing overflow
    issues while minimizing memory usage.
    
    Args:
        estimated_max_nuclei (int): Estimated maximum number of nuclei in the image.
        image_size_pixels (int): Total image size in pixels (used for additional validation).
            Defaults to 0.
        safety_factor (float): Safety multiplier for the estimate to account for uncertainty.
            Defaults to 2.0.

    Returns:
        np.dtype: Optimal numpy data type for mask storage.
        
    Notes
    -----
    Selection criteria:
    - uint16: Up to ~32K nuclei (suitable for small tissue regions).
    - uint32: Up to ~2B nuclei (suitable for most whole-slide images).
    - uint64: For extremely large images or high-density tissues.
    """
    try:
        # Apply safety factor to account for estimation uncertainty.
        safe_max_nuclei = int(estimated_max_nuclei * safety_factor)
        
        # Data type limits.
        uint16_limit = 2**16 - 1  # Equals 65,535.
        uint32_limit = 2**32 - 1  # Equals 4,294,967,295.
        
        if safe_max_nuclei <= uint16_limit:
            selected_dtype = np.uint16
            logging.info(f"Selected uint16 for {estimated_max_nuclei:,} estimated nuclei "
                        f"(safe_max: {safe_max_nuclei:,}, limit: {uint16_limit:,})")
        elif safe_max_nuclei <= uint32_limit:
            selected_dtype = np.uint32
            logging.info(f"Selected uint32 for {estimated_max_nuclei:,} estimated nuclei "
                        f"(safe_max: {safe_max_nuclei:,}, limit: {uint32_limit:,})")
        else:
            selected_dtype = np.uint64
            logging.warning(f"Selected uint64 for {estimated_max_nuclei:,} estimated nuclei "
                           f"(safe_max: {safe_max_nuclei:,}, exceeds uint32 limit)")
            logging.warning(f"Very large image detected - consider processing in regions")
        
        # Additional validation based on image size.
        if image_size_pixels > 0:
            max_possible_nuclei = image_size_pixels // 100  # Assume min 100 pixels per nucleus.
            if estimated_max_nuclei > max_possible_nuclei:
                logging.warning(f"Estimated nuclei count ({estimated_max_nuclei:,}) seems high "
                               f"for image size ({image_size_pixels:,} pixels). "
                               f"Max reasonable: {max_possible_nuclei:,}")
        
        return selected_dtype
        
    except Exception as e:
        logging.error(f"Error selecting optimal data type: {e}")
        logging.debug(f"Data type selection error traceback:\n{traceback.format_exc()}")
        # Return safe default.
        return np.uint32


def estimate_max_nuclei_from_image(
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    avg_nuclei_per_tile: float = 50.0
) -> int:
    """
    Estimate maximum number of nuclei in an image based on dimensions.
    
    This function provides a rough estimate of the maximum number of nuclei
    expected in an image, useful for data type selection and memory planning.
    
    Args:
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        tile_h (int): Tile height in pixels.
        tile_w (int): Tile width in pixels.
        avg_nuclei_per_tile (float): Average number of nuclei expected per tile.
            Defaults to 50.0.

    Returns:
        int: Estimated maximum number of nuclei in the image.
        
    Notes
    -----
    This is a rough estimate based on tile count and average density.
    Actual nucleus counts can vary significantly based on tissue type,
    staining quality, and segmentation parameters.
    """
    try:
        # Calculate number of tiles needed to cover the image.
        overlap = min(tile_h, tile_w) // 4  # Assume 25% overlap.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        num_tiles_h = (height + stride_h - 1) // stride_h
        num_tiles_w = (width + stride_w - 1) // stride_w
        total_tiles = num_tiles_h * num_tiles_w
        
        # Estimate total nuclei.
        estimated_nuclei = int(total_tiles * avg_nuclei_per_tile)
        
        logging.info(f"Nucleus count estimation: {height}x{width} image, "
                    f"{total_tiles:,} tiles, {avg_nuclei_per_tile} nuclei/tile "
                    f"→ {estimated_nuclei:,} estimated nuclei")
        
        return estimated_nuclei
        
    except Exception as e:
        logging.error(f"Error estimating max nuclei count: {e}")
        logging.debug(f"Nuclei estimation error traceback:\n{traceback.format_exc()}")
        # Return conservative estimate.
        return 1000000  # 1 million nuclei as fallback.


def get_safe_id_limits(dtype: np.dtype) -> Tuple[int, int, int]:
    """
    Get safe ID limits for the specified data type.
    
    Args:
        dtype (np.dtype): Data type for mask storage.

    Returns:
        Tuple[int, int, int]: (max_safe_id, segment_size, max_segments)
            - max_safe_id: Conservative maximum ID before overflow risk.
            - segment_size: Recommended segment size for ID allocation.
            - max_segments: Maximum number of segments possible.
    """
    try:
        if dtype == np.uint16:
            max_value = 2**16 - 1
            max_safe_id = int(max_value * 0.9)  # 90% of maximum.
            segment_size = 5000
        elif dtype == np.uint32:
            max_value = 2**32 - 1
            max_safe_id = int(max_value * 0.9)  # 90% of maximum.
            segment_size = 100000000  # 100M per segment.
        elif dtype == np.uint64:
            max_value = 2**64 - 1
            max_safe_id = int(max_value * 0.9)  # 90% of maximum.
            segment_size = 1000000000000  # 1T per segment.
        else:
            # Fallback for unknown types.
            max_safe_id = 2000000000  # Equals 2B.
            segment_size = 100000000
        
        max_segments = max_safe_id // segment_size
        
        logging.debug(f"Safe ID limits for {dtype}: max_safe={max_safe_id:,}, "
                     f"segment_size={segment_size:,}, max_segments={max_segments}")
        
        return max_safe_id, segment_size, max_segments
        
    except Exception as e:
        logging.error(f"Error getting safe ID limits for {dtype}: {e}")
        # Return conservative defaults.
        return 2000000000, 100000000, 20

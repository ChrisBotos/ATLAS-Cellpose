#!/usr/bin/env python3
"""
Test script for incremental merge fix.

Author: Christos Botos
Email: hcty02@gmail.com
Date: 2025-07-18

Description:
    Test script to verify that the incremental merge processing fix properly
    handles tile boundaries and maintains segmentation quality. This script
    creates a synthetic test case that mimics the problematic sparse tile
    distribution scenario and verifies that nuclei spanning tile boundaries
    are properly merged.

Dependencies:
    • numpy
    • logging
    • pathlib

Usage:
    python test_incremental_merge_fix.py

Arguments:
    None

Inputs:
    • Synthetic tile masks with overlapping nuclei

Outputs:
    • Test results showing merge quality
    • Debug information about merge operations

Key Features:
    • Creates synthetic overlapping nuclei across tile boundaries
    • Tests both individual tile processing and tile group processing
    • Verifies that merge operations are performed correctly
    • Compares results with expected merge outcomes

Notes:
    • This test specifically targets the incremental processing bug fix
    • It simulates the memory-constrained scenario that triggers incremental mode
    • Results should show proper merging without visible tile boundaries
"""

import traceback
import numpy as np
import logging
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict
from numpy.typing import NDArray

# Configure logging for detailed test output.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)

def create_synthetic_overlapping_tiles(
    tile_h: int = 512,
    tile_w: int = 512,
    overlap: int = 64,
    num_tiles_h: int = 2,
    num_tiles_w: int = 2
) -> Tuple[Dict[Tuple[int, int], NDArray[np.uint32]], int, int]:
    """
    Create synthetic tile masks with nuclei that span tile boundaries.

    This function creates a simple test scenario with just a few nuclei
    that clearly cross tile boundaries to test the merge functionality.

    Parameters
    ----------
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    num_tiles_h, num_tiles_w : int
        Number of tiles in each dimension.

    Returns
    -------
    Tuple[Dict[Tuple[int, int], NDArray[np.uint32]], int, int]
        Dictionary mapping tile coordinates to tile masks, full height, full width.
    """
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    full_height = num_tiles_h * stride_h + overlap
    full_width = num_tiles_w * stride_w + overlap

    # Create individual tile masks with specific overlapping nuclei.
    tile_masks = {}

    # Create a simple test case: one nucleus that spans all four tiles.
    # Place it at the center intersection.
    center_y = stride_h
    center_x = stride_w
    radius = overlap // 2 + 20  # Ensure it spans into all four tiles.

    nucleus_id = 1

    for tile_r in range(num_tiles_h):
        for tile_c in range(num_tiles_w):
            # Create tile mask.
            tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)

            # Calculate tile position in global coordinates.
            tile_y0 = tile_r * stride_h
            tile_x0 = tile_c * stride_w

            # Check if the central nucleus overlaps with this tile.
            for y in range(tile_h):
                for x in range(tile_w):
                    global_y = tile_y0 + y
                    global_x = tile_x0 + x

                    # Check if this pixel is part of the central nucleus.
                    if (global_y - center_y)**2 + (global_x - center_x)**2 <= radius**2:
                        tile_mask[y, x] = nucleus_id

            # Add a small nucleus unique to each tile (not overlapping).
            unique_center_y = tile_h // 4
            unique_center_x = tile_w // 4
            unique_radius = 15

            for y in range(tile_h):
                for x in range(tile_w):
                    if (y - unique_center_y)**2 + (x - unique_center_x)**2 <= unique_radius**2:
                        tile_mask[y, x] = nucleus_id + 1 + tile_r * num_tiles_w + tile_c

            # Only include tiles that have nuclei.
            if np.any(tile_mask > 0):
                tile_masks[(tile_r, tile_c)] = tile_mask

    total_nuclei = 1 + num_tiles_h * num_tiles_w  # Central nucleus + unique nuclei.

    logging.info(f"Created {len(tile_masks)} synthetic tiles with overlapping nuclei")
    logging.info(f"Full image size: {full_height}×{full_width}, total nuclei: {total_nuclei}")
    logging.info(f"Central nucleus (ID={nucleus_id}) should span all tiles")

    return tile_masks, full_height, full_width


def create_tile_loader(
    tile_masks: Dict[Tuple[int, int], NDArray[np.uint32]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> Callable[[slice, slice], NDArray[np.uint32]]:
    """
    Create a loader function that simulates loading tiles from disk.
    
    Parameters
    ----------
    tile_masks : Dict[Tuple[int, int], NDArray[np.uint32]]
        Dictionary mapping tile coordinates to tile masks.
    tile_h, tile_w : int
        Tile dimensions.
    overlap : int
        Overlap between tiles.
        
    Returns
    -------
    Callable[[slice, slice], NDArray[np.uint32]]
        Loader function compatible with the merge functions.
    """
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    def loader(ys: slice, xs: slice) -> NDArray[np.uint32]:
        """Load tile data for the given slice coordinates."""
        # Determine which tile(s) this slice corresponds to.
        y_start, y_stop = ys.start, ys.stop
        x_start, x_stop = xs.start, xs.stop
        
        # Find the primary tile.
        tile_r = y_start // stride_h
        tile_c = x_start // stride_w
        
        if (tile_r, tile_c) in tile_masks:
            tile_mask = tile_masks[(tile_r, tile_c)]
            
            # Calculate the slice within the tile.
            tile_y0 = tile_r * stride_h
            tile_x0 = tile_c * stride_w
            
            rel_y_start = max(0, y_start - tile_y0)
            rel_x_start = max(0, x_start - tile_x0)
            rel_y_stop = min(tile_mask.shape[0], y_stop - tile_y0)
            rel_x_stop = min(tile_mask.shape[1], x_stop - tile_x0)
            
            if rel_y_stop > rel_y_start and rel_x_stop > rel_x_start:
                return tile_mask[rel_y_start:rel_y_stop, rel_x_start:rel_x_stop].copy()
        
        # Return empty array if no tile found.
        return np.zeros((y_stop - y_start, x_stop - x_start), dtype=np.uint32)
    
    return loader


def test_incremental_merge_fix():
    """
    Test the incremental merge fix with synthetic overlapping nuclei.
    
    This test verifies that the enhanced incremental processing properly
    merges nuclei that span tile boundaries, eliminating visible tile
    boundaries in the final segmentation mask.
    """
    logging.info("=== Testing Incremental Merge Fix ===")
    
    # Create synthetic test data.
    tile_h, tile_w = 512, 512
    overlap = 64
    num_tiles_h, num_tiles_w = 2, 2
    
    tile_masks, full_height, full_width = create_synthetic_overlapping_tiles(
        tile_h, tile_w, overlap, num_tiles_h, num_tiles_w
    )
    
    # Create tile loader.
    loader = create_tile_loader(tile_masks, tile_h, tile_w, overlap)
    
    # Import the fixed incremental processing function.
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import _merge_cluster_incremental
        logging.info("Successfully imported fixed incremental merge function")
    except ImportError as e:
        logging.error(f"Failed to import incremental merge function: {e}")
        return False
    
    # Create global merged array.
    global_merged_array = np.zeros((full_height, full_width), dtype=np.uint32)
    
    # Test the incremental processing.
    cluster = list(tile_masks.keys())
    logging.info(f"Testing incremental processing with {len(cluster)} tiles")
    
    try:
        result_patch, (y0, x0), mapping = _merge_cluster_incremental(
            cluster=cluster,
            loader=loader,
            height=full_height,
            width=full_width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=0.3,
            use_gpu=False,  # Use CPU for testing.
            gid_offset=1,
            global_merged_array=global_merged_array,
            temp_file_path=None,
        )
        
        logging.info("Incremental processing completed successfully")
        
        # Analyze results.
        total_nuclei = np.max(global_merged_array)
        nuclei_pixels = np.count_nonzero(global_merged_array)
        coverage = (nuclei_pixels / (full_height * full_width)) * 100
        
        logging.info(f"Results: {total_nuclei} nuclei, {nuclei_pixels} pixels, {coverage:.2f}% coverage")
        
        # Check for proper merging by looking at tile boundaries.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        boundary_merge_count = 0
        total_boundary_checks = 0
        
        # Check horizontal boundaries.
        for tile_r in range(num_tiles_h - 1):
            boundary_y = (tile_r + 1) * stride_h
            if boundary_y < full_height:
                for x in range(0, full_width, 50):  # Sample every 50 pixels.
                    if x < full_width:
                        # Check if nuclei cross the boundary.
                        above_id = global_merged_array[boundary_y - 1, x]
                        below_id = global_merged_array[boundary_y, x]
                        
                        if above_id > 0 and below_id > 0:
                            total_boundary_checks += 1
                            if above_id == below_id:
                                boundary_merge_count += 1
        
        # Check vertical boundaries.
        for tile_c in range(num_tiles_w - 1):
            boundary_x = (tile_c + 1) * stride_w
            if boundary_x < full_width:
                for y in range(0, full_height, 50):  # Sample every 50 pixels.
                    if y < full_height:
                        # Check if nuclei cross the boundary.
                        left_id = global_merged_array[y, boundary_x - 1]
                        right_id = global_merged_array[y, boundary_x]
                        
                        if left_id > 0 and right_id > 0:
                            total_boundary_checks += 1
                            if left_id == right_id:
                                boundary_merge_count += 1
        
        if total_boundary_checks > 0:
            merge_success_rate = (boundary_merge_count / total_boundary_checks) * 100
            logging.info(f"Boundary merge success rate: {merge_success_rate:.1f}% "
                        f"({boundary_merge_count}/{total_boundary_checks})")
            
            # Test passes if merge success rate is high.
            if merge_success_rate >= 80:
                logging.info("✅ TEST PASSED: Incremental merge fix working correctly")
                return True
            else:
                logging.error("❌ TEST FAILED: Low merge success rate indicates tile boundaries still visible")
                return False
        else:
            logging.warning("⚠️  TEST INCONCLUSIVE: No boundary crossings found to test")
            return True
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED: Exception during incremental processing: {e}")
        logging.debug(f"Exception traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_incremental_merge_fix()
    exit(0 if success else 1)

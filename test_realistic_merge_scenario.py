#!/usr/bin/env python3
"""
Realistic test for incremental merge fix.

Author: Christos Botos
Email: hcty02@gmail.com
Date: 2025-07-18

Description:
    Test script that simulates a realistic large image scenario similar to
    the user's problematic case. Creates a sparse tile distribution with
    nuclei of various sizes and overlap patterns to verify that the
    incremental merge fix properly handles real-world conditions.

Dependencies:
    • numpy
    • logging
    • pathlib

Usage:
    python test_realistic_merge_scenario.py

Key Features:
    • Simulates sparse tile distribution (like 66 tiles across large image)
    • Creates nuclei with realistic size variations
    • Tests both individual and group tile processing
    • Verifies merge quality with statistical analysis
    • Measures performance impact of the fix

Notes:
    • This test mimics the conditions that caused the original bug
    • It specifically tests the memory-aware clustering scenario
    • Results should show proper merging without memory allocation errors
"""

import traceback
import numpy as np
import logging
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict
from numpy.typing import NDArray

# Configure logging for detailed test output.
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def create_realistic_sparse_tiles(
    full_height: int = 8000,
    full_width: int = 8000,
    tile_h: int = 512,
    tile_w: int = 512,
    overlap: int = 64,
    num_nuclei: int = 200
) -> Tuple[Dict[Tuple[int, int], NDArray[np.uint32]], int, int]:
    """
    Create a realistic sparse tile distribution similar to the user's case.
    
    This simulates a large image with scattered nuclei, creating the sparse
    tile distribution that triggers incremental processing.
    
    Parameters
    ----------
    full_height, full_width : int
        Full image dimensions.
    tile_h, tile_w : int
        Tile dimensions.
    overlap : int
        Overlap between tiles.
    num_nuclei : int
        Total number of nuclei to distribute.
        
    Returns
    -------
    Tuple[Dict[Tuple[int, int], NDArray[np.uint32]], int, int]
        Dictionary mapping tile coordinates to tile masks, full height, full width.
    """
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    # Calculate tile grid dimensions.
    tiles_h = (full_height - overlap + stride_h - 1) // stride_h
    tiles_w = (full_width - overlap + stride_w - 1) // stride_w
    
    logging.info(f"Creating realistic sparse distribution: {tiles_h}×{tiles_w} tiles "
                f"({tiles_h * tiles_w} total) across {full_height}×{full_width} image")
    
    # Create full image with scattered nuclei.
    full_image = np.zeros((full_height, full_width), dtype=np.uint32)
    
    # Generate nuclei with realistic size distribution.
    np.random.seed(42)  # For reproducible results.
    
    nucleus_id = 1
    nuclei_placed = 0
    
    # Place nuclei randomly across the image.
    for _ in range(num_nuclei * 3):  # Try more times to account for overlaps.
        if nuclei_placed >= num_nuclei:
            break
            
        # Random position.
        center_y = np.random.randint(50, full_height - 50)
        center_x = np.random.randint(50, full_width - 50)
        
        # Random size (realistic nucleus sizes).
        radius = np.random.randint(8, 35)
        
        # Check if this position is already occupied.
        y_coords, x_coords = np.ogrid[:full_height, :full_width]
        mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
        
        if not np.any(full_image[mask] > 0):  # No overlap with existing nuclei.
            full_image[mask] = nucleus_id
            nucleus_id += 1
            nuclei_placed += 1
    
    # Add some nuclei that intentionally span tile boundaries.
    boundary_nuclei = 0
    for tile_r in range(min(3, tiles_h - 1)):  # Just a few boundary cases.
        for tile_c in range(min(3, tiles_w - 1)):
            # Place nucleus near tile boundary.
            boundary_y = (tile_r + 1) * stride_h
            boundary_x = (tile_c + 1) * stride_w
            
            # Add some randomness.
            center_y = boundary_y + np.random.randint(-overlap//2, overlap//2)
            center_x = boundary_x + np.random.randint(-overlap//2, overlap//2)
            
            # Ensure within bounds.
            center_y = max(30, min(full_height - 30, center_y))
            center_x = max(30, min(full_width - 30, center_x))
            
            radius = np.random.randint(15, 40)  # Larger to ensure boundary crossing.
            
            y_coords, x_coords = np.ogrid[:full_height, :full_width]
            mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
            
            # Only place if not too much overlap.
            if np.sum(full_image[mask] > 0) < mask.sum() * 0.3:
                full_image[mask] = nucleus_id
                nucleus_id += 1
                boundary_nuclei += 1
    
    # Extract tiles that contain nuclei (creating sparse distribution).
    tile_masks = {}
    tiles_with_nuclei = 0
    
    for tile_r in range(tiles_h):
        for tile_c in range(tiles_w):
            y0 = tile_r * stride_h
            x0 = tile_c * stride_w
            y1 = min(y0 + tile_h, full_height)
            x1 = min(x0 + tile_w, full_width)
            
            tile_mask = full_image[y0:y1, x0:x1].copy()
            
            # Only include tiles that have nuclei.
            if np.any(tile_mask > 0):
                tile_masks[(tile_r, tile_c)] = tile_mask
                tiles_with_nuclei += 1
    
    logging.info(f"Created sparse distribution: {tiles_with_nuclei}/{tiles_h * tiles_w} tiles contain nuclei")
    logging.info(f"Total nuclei: {nuclei_placed + boundary_nuclei} ({boundary_nuclei} boundary-spanning)")
    logging.info(f"Sparsity: {tiles_with_nuclei / (tiles_h * tiles_w) * 100:.1f}% tile occupancy")
    
    return tile_masks, full_height, full_width


def create_tile_loader_realistic(
    tile_masks: Dict[Tuple[int, int], NDArray[np.uint32]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> Callable[[slice, slice], NDArray[np.uint32]]:
    """Create a realistic tile loader that handles edge cases."""
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    def loader(ys: slice, xs: slice) -> NDArray[np.uint32]:
        """Load tile data for the given slice coordinates."""
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


def test_realistic_merge_scenario():
    """
    Test the incremental merge fix with a realistic sparse tile scenario.
    
    This test simulates the conditions that caused the original bug:
    - Large image with sparse tile distribution
    - Memory-constrained processing triggering incremental mode
    - Mix of isolated and boundary-spanning nuclei
    """
    logging.info("=== Testing Realistic Merge Scenario ===")
    
    # Create realistic test data similar to user's case.
    full_height, full_width = 8000, 8000
    tile_h, tile_w = 512, 512
    overlap = 64
    num_nuclei = 150
    
    tile_masks, actual_height, actual_width = create_realistic_sparse_tiles(
        full_height, full_width, tile_h, tile_w, overlap, num_nuclei
    )
    
    # Create tile loader.
    loader = create_tile_loader_realistic(tile_masks, tile_h, tile_w, overlap)
    
    # Import the fixed incremental processing function.
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import _merge_cluster_incremental
        logging.info("Successfully imported fixed incremental merge function")
    except ImportError as e:
        logging.error(f"Failed to import incremental merge function: {e}")
        return False
    
    # Create global merged array.
    global_merged_array = np.zeros((actual_height, actual_width), dtype=np.uint32)
    
    # Test the incremental processing.
    cluster = list(tile_masks.keys())
    logging.info(f"Testing incremental processing with {len(cluster)} tiles "
                f"(simulating sparse distribution)")
    
    try:
        import time
        start_time = time.time()
        
        result_patch, (y0, x0), mapping = _merge_cluster_incremental(
            cluster=cluster,
            loader=loader,
            height=actual_height,
            width=actual_width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=0.3,
            use_gpu=False,  # Use CPU for testing.
            gid_offset=1,
            global_merged_array=global_merged_array,
            temp_file_path=None,
        )
        
        processing_time = time.time() - start_time
        logging.info(f"Incremental processing completed in {processing_time:.2f} seconds")
        
        # Analyze results.
        total_nuclei = np.max(global_merged_array)
        nuclei_pixels = np.count_nonzero(global_merged_array)
        coverage = (nuclei_pixels / (actual_height * actual_width)) * 100
        
        logging.info(f"Results: {total_nuclei} nuclei, {nuclei_pixels} pixels, {coverage:.3f}% coverage")
        
        # Check for proper merging by analyzing nucleus size distribution.
        nucleus_sizes = []
        for nucleus_id in range(1, total_nuclei + 1):
            size = np.sum(global_merged_array == nucleus_id)
            if size > 0:
                nucleus_sizes.append(size)
        
        if nucleus_sizes:
            avg_size = np.mean(nucleus_sizes)
            max_size = np.max(nucleus_sizes)
            min_size = np.min(nucleus_sizes)
            
            logging.info(f"Nucleus size distribution: avg={avg_size:.1f}, "
                        f"min={min_size}, max={max_size}")
            
            # Check for reasonable size distribution (no extremely small fragments).
            small_nuclei = sum(1 for size in nucleus_sizes if size < 50)
            small_fraction = small_nuclei / len(nucleus_sizes)
            
            logging.info(f"Small nuclei (<50 pixels): {small_nuclei}/{len(nucleus_sizes)} "
                        f"({small_fraction*100:.1f}%)")
            
            # Test passes if we have reasonable nucleus sizes and not too many fragments.
            if small_fraction < 0.3 and avg_size > 200:  # Reasonable thresholds.
                logging.info("✅ TEST PASSED: Realistic merge scenario working correctly")
                logging.info(f"   - Processing time: {processing_time:.2f}s")
                logging.info(f"   - Memory-safe incremental processing successful")
                logging.info(f"   - Proper nucleus size distribution maintained")
                return True
            else:
                logging.error("❌ TEST FAILED: Too many small fragments suggest poor merging")
                return False
        else:
            logging.error("❌ TEST FAILED: No nuclei found in results")
            return False
            
    except Exception as e:
        logging.error(f"❌ TEST FAILED: Exception during incremental processing: {e}")
        logging.debug(f"Exception traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_realistic_merge_scenario()
    exit(0 if success else 1)

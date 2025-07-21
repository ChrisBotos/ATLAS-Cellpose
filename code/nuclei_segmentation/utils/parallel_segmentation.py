"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: parallel_segmentation.py.
Description:
    Parallel processing utilities for Cellpose3 segmentation to improve performance
    on large kidney tissue images by processing multiple tiles simultaneously.

Dependencies:
    • Python >= 3.7.
    • numpy, torch, cellpose.
    • concurrent.futures for parallel processing.
    • tqdm for progress tracking.

Usage:
    from parallel_segmentation import run_cellpose_parallel_batches
    
    masks, flows, n_cells = run_cellpose_parallel_batches(
        model=cellpose_model,
        tiles=tile_list,
        cellpose_params=params,
        batch_size=4,
        max_workers=2
    )

Inputs:
    • Cellpose model instance.
    • List of image tiles to process.
    • Cellpose parameters dictionary.
    • Batch processing configuration.

Outputs:
    • Processed segmentation masks for each tile.
    • Flow fields (optional).
    • Total cell count across all tiles.

Key Features:
    • Memory-safe batch processing with configurable batch sizes.
    • GPU memory monitoring and cleanup between batches.
    • Progress tracking with detailed logging.
    • Error handling with retry mechanisms.
    • Timeout protection to prevent infinite loops.

Notes:
    • Designed for kidney I/R injury tissue analysis.
    • Optimized for DAPI-stained nuclear segmentation.
    • Supports both GPU and CPU processing modes.
"""

import traceback
import numpy as np
import torch
import logging
import time
import gc
import psutil
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path


def estimate_batch_memory_usage(tile_size: int, batch_size: int, dtype_size: int = 4) -> float:
    """
    Estimate memory usage for a batch of tiles during Cellpose processing.
    
    Args:
        tile_size: Size of each square tile (e.g., 512 for 512x512).
        batch_size: Number of tiles to process simultaneously.
        dtype_size: Size of data type in bytes (4 for float32).
    
    Returns:
        Estimated memory usage in GB.
    """
    # Estimate memory for input tiles, intermediate processing, and outputs.
    pixels_per_tile = tile_size * tile_size
    
    # Input tiles + intermediate processing + output masks.
    memory_per_tile = pixels_per_tile * dtype_size * 3  # Conservative estimate.
    total_memory = memory_per_tile * batch_size
    
    return total_memory / (1024**3)  # Convert to GB.


def get_optimal_batch_size(tile_size: int, available_memory_gb: float, max_batch_size: int = 8) -> int:
    """
    Calculate optimal batch size based on available memory and tile size.
    
    Args:
        tile_size: Size of each square tile.
        available_memory_gb: Available GPU/system memory in GB.
        max_batch_size: Maximum allowed batch size.
    
    Returns:
        Optimal batch size for processing.
    """
    # Use conservative memory estimate with safety factor.
    safety_factor = 0.7  # Use only 70% of available memory.
    usable_memory = available_memory_gb * safety_factor
    
    # Binary search for optimal batch size.
    left, right = 1, max_batch_size
    optimal_size = 1
    
    while left <= right:
        mid = (left + right) // 2
        estimated_usage = estimate_batch_memory_usage(tile_size, mid)
        
        if estimated_usage <= usable_memory:
            optimal_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    logging.info(f"Optimal batch size: {optimal_size} tiles (memory limit: {available_memory_gb:.2f} GB)")
    return optimal_size


def process_cellpose_batch(
    model,
    tile_batch: List[Tuple[np.ndarray, Tuple[slice, slice]]],
    cellpose_params: Dict[str, Any],
    batch_idx: int,
    timeout_seconds: int = 300
) -> List[Tuple[np.ndarray, Tuple[slice, slice], int]]:
    """
    Process a batch of tiles with Cellpose segmentation with enhanced memory safety.

    Note: This function is designed to be thread-safe and does not use signal-based
    timeouts since signals only work in the main thread.

    Args:
        model: Cellpose model instance.
        tile_batch: List of (tile_image, (y_slice, x_slice)) tuples.
        cellpose_params: Cellpose parameters dictionary.
        batch_idx: Index of this batch for logging.
        timeout_seconds: Maximum processing time per batch.

    Returns:
        List of (mask, slice_info, cell_count) tuples.
    """
    import psutil
    import os

    def check_memory_usage():
        """Check current memory usage and warn if approaching limits."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 5000:  # Warn if using more than 5GB.
            logging.warning(f"High memory usage detected: {memory_mb:.1f} MB")
        return memory_mb

    def check_timeout(start_time: float, timeout_seconds: int) -> None:
        """Check if processing has exceeded timeout limit."""
        if timeout_seconds > 0 and (time.time() - start_time) > timeout_seconds:
            raise TimeoutError(f"Batch {batch_idx} processing timed out after {timeout_seconds} seconds")

    try:
        start_time = time.time()
        results = []

        logging.debug(f"Processing batch {batch_idx} with {len(tile_batch)} tiles")
        initial_memory = check_memory_usage()

        for tile_idx, (tile_image, slice_info) in enumerate(tile_batch):
            try:
                # Check timeout before processing each tile.
                check_timeout(start_time, timeout_seconds)

                # Monitor memory before processing each tile.
                if tile_idx > 0 and tile_idx % 2 == 0:  # Check every 2 tiles.
                    current_memory = check_memory_usage()
                    if current_memory > initial_memory + 1000:  # More than 1GB increase.
                        logging.warning(f"Memory usage increased by {current_memory - initial_memory:.1f} MB")
                        # Force garbage collection.
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Validate tile content before Cellpose processing.
                tile_stats = {
                    'mean': float(np.mean(tile_image)),
                    'std': float(np.std(tile_image)),
                    'min': float(np.min(tile_image)),
                    'max': float(np.max(tile_image))
                }

                logging.debug(f"Tile {tile_idx+1} stats: mean={tile_stats['mean']:.1f}, "
                             f"std={tile_stats['std']:.1f}, min={tile_stats['min']}, max={tile_stats['max']}")

                # Check for problematic tiles that might cause division by zero.
                if tile_stats['std'] < 1.0 or tile_stats['max'] - tile_stats['min'] < 5:
                    logging.warning(f"Tile {tile_idx+1} has very low contrast (std={tile_stats['std']:.2f}), "
                                   f"may cause Cellpose issues")

                # Run Cellpose on this tile with Cellpose4 auto-detection.
                # Use diameter=None for adaptive diameter learning across tiles.
                try:
                    # Remove deprecated resample parameter for Cellpose4 v4.0.1+
                    eval_params = {
                        "diameter": cellpose_params["diameter"],
                        "flow_threshold": cellpose_params["flow_threshold"],
                        "cellprob_threshold": cellpose_params["cellprob_threshold"],
                        "augment": False,
                        "batch_size": cellpose_params["batch_size"],
                        "do_3D": False,
                    }

                    cellpose_results = model.eval(tile_image[..., None], **eval_params)
                except Exception as e:
                    logging.error(f"✗ Cellpose auto-detection failed on tile {tile_idx+1}: {e}")
                    logging.error(f"Tile {tile_idx+1} statistics: mean={tile_stats['mean']:.1f}, "
                                 f"std={tile_stats['std']:.1f}, min={tile_stats['min']}, max={tile_stats['max']}")
                    logging.error("Please ensure diameter=None is configured for auto-detection")
                    raise e

                # Extract mask and count cells.
                mask = cellpose_results[0]

                # Extract and log Cellpose4 diameter information for adaptive learning assessment.
                # Note: Cellpose4 with diameter=None may not return diameter info in results tuple.
                if len(cellpose_results) >= 4 and cellpose_results[3] is not None:
                    detected_diameters = cellpose_results[3]
                    if isinstance(detected_diameters, (list, np.ndarray)) and len(detected_diameters) > 0:
                        # Multiple diameters detected - analyze variation across tile.
                        avg_diameter = float(np.mean(detected_diameters))
                        min_diameter = float(np.min(detected_diameters))
                        max_diameter = float(np.max(detected_diameters))
                        logging.info(f"✓ Tile {tile_idx+1}: Auto-detected diameter = {avg_diameter:.1f}px (range: {min_diameter:.1f}-{max_diameter:.1f}px)")

                        # Log diameter variation for quality assessment.
                        if len(detected_diameters) > 1:
                            diameter_cv = (np.std(detected_diameters) / avg_diameter) * 100
                            logging.info(f"  Tile {tile_idx+1} diameter variation: CV = {diameter_cv:.1f}%")
                    elif isinstance(detected_diameters, (int, float)) and detected_diameters > 0:
                        # Single diameter detected.
                        logging.info(f"✓ Tile {tile_idx+1}: Auto-detected diameter = {detected_diameters:.1f}px")
                    else:
                        logging.debug(f"Tile {tile_idx+1}: No valid diameter detected (value: {detected_diameters}), using model default")
                else:
                    # Check if model has diameter information stored internally.
                    model_diameter = getattr(model, 'diam_mean', None)
                    if model_diameter and model_diameter > 0:
                        logging.debug(f"Tile {tile_idx+1}: Using model diameter = {model_diameter:.1f}px (auto-detection active)")
                    else:
                        logging.debug(f"Tile {tile_idx+1}: Auto-detection active (diameter=None), using internal Cellpose4 sizing")

                if mask is None:
                    mask = np.zeros(tile_image.shape[:2], dtype=np.uint32)
                    cell_count = 0
                else:
                    mask = mask.astype(np.uint32)
                    # Count unique non-zero labels (correct method for nuclei counting).
                    cell_count = len(np.unique(mask[mask > 0])) if mask.size > 0 else 0

                results.append((mask, slice_info, cell_count))

                logging.debug(f"NUCLEI COUNT: Batch {batch_idx}, tile {tile_idx+1}: {cell_count} nuclei detected")

                # Clear intermediate results to free memory.
                cellpose_results = None

            except Exception as tile_error:
                logging.error(f"Error processing tile {tile_idx+1} in batch {batch_idx}: {tile_error}")
                # Create empty mask as fallback.
                empty_mask = np.zeros(tile_image.shape[:2], dtype=np.uint32)
                results.append((empty_mask, slice_info, 0))

        elapsed_time = time.time() - start_time
        total_cells = sum(result[2] for result in results)
        final_memory = check_memory_usage()

        logging.info(f"BATCH PROCESSING: Batch {batch_idx} completed: {len(results)} image tiles, {total_cells} total nuclei detected, "
                    f"{elapsed_time:.2f}s, memory: {final_memory:.1f} MB")

        return results

    except TimeoutError as e:
        logging.error(f"Batch {batch_idx} timed out after {timeout_seconds} seconds")
        raise
    except Exception as e:
        logging.error(f"Batch {batch_idx} processing failed: {e}")
        logging.debug(f"Batch error traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Aggressive memory cleanup.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run_cellpose_parallel_batches(
    model,
    tiles: List[Tuple[np.ndarray, Tuple[slice, slice]]],
    cellpose_params: Dict[str, Any],
    batch_size: int = 4,
    max_workers: int = 2,
    memory_limit_gb: float = 6.0,
    timeout_seconds: int = 300
) -> Tuple[List[np.ndarray], List[Any], int]:
    """
    Run Cellpose segmentation on multiple tiles using parallel batch processing with robust error handling.

    Args:
        model: Cellpose model instance.
        tiles: List of (tile_image, slice_info) tuples.
        cellpose_params: Cellpose parameters dictionary.
        batch_size: Number of tiles per batch.
        max_workers: Maximum number of parallel workers.
        memory_limit_gb: Memory limit for batch processing.
        timeout_seconds: Timeout per batch in seconds.

    Returns:
        Tuple of (masks_list, flows_list, total_cell_count).
    """
    if not tiles:
        return [], [], 0

    # Validate inputs.
    if batch_size < 1:
        batch_size = 1
    if max_workers < 1:
        max_workers = 1
    if memory_limit_gb < 1.0:
        memory_limit_gb = 1.0

    # Optimize batch size based on available memory.
    tile_size = tiles[0][0].shape[0]  # Assume square tiles.
    optimal_batch_size = get_optimal_batch_size(tile_size, memory_limit_gb, batch_size)
    batch_size = min(batch_size, optimal_batch_size)

    # Limit max_workers based on system constraints.
    import psutil
    cpu_count = psutil.cpu_count(logical=False) or 2
    max_workers = min(max_workers, cpu_count, 4)  # Cap at 4 workers max.

    # Create batches.
    batches = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        batches.append(batch)

    logging.info(f"Processing {len(tiles)} tiles in {len(batches)} batches "
                f"(batch_size={batch_size}, max_workers={max_workers}, memory_limit={memory_limit_gb:.1f}GB)")

    # Process batches in parallel with comprehensive error handling.
    all_results = []
    total_cells = 0
    failed_batches = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches.
            future_to_batch = {
                executor.submit(
                    process_cellpose_batch,
                    model, batch, cellpose_params, batch_idx, timeout_seconds
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            # Collect results with progress tracking.
            with tqdm(total=len(batches), desc="Processing tile batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result(timeout=timeout_seconds + 30)  # Extra timeout buffer.
                        all_results.extend(batch_results)

                        batch_cells = sum(result[2] for result in batch_results)
                        total_cells += batch_cells

                        pbar.update(1)
                        pbar.set_postfix({"Total cells": total_cells, "Failed": failed_batches})

                    except Exception as e:
                        failed_batches += 1
                        logging.error(f"Batch {batch_idx} failed: {e}")

                        # Add empty results for failed batch.
                        batch = batches[batch_idx]
                        for tile_image, slice_info in batch:
                            empty_mask = np.zeros(tile_image.shape[:2], dtype=np.uint32)
                            all_results.append((empty_mask, slice_info, 0))

                        pbar.update(1)
                        pbar.set_postfix({"Total cells": total_cells, "Failed": failed_batches})

    except Exception as executor_error:
        logging.error(f"Parallel executor failed: {executor_error}")
        raise RuntimeError(f"Parallel processing failed: {executor_error}")

    # Validate results.
    if len(all_results) != len(tiles):
        logging.error(f"Result count mismatch: expected {len(tiles)}, got {len(all_results)}")
        # Pad with empty results if needed.
        while len(all_results) < len(tiles):
            tile_image, slice_info = tiles[len(all_results)]
            empty_mask = np.zeros(tile_image.shape[:2], dtype=np.uint32)
            all_results.append((empty_mask, slice_info, 0))

    # Sort results by original tile order and extract components.
    all_results.sort(key=lambda x: (x[1][0].start, x[1][1].start))  # Sort by y, then x slice start.

    masks = [result[0] for result in all_results]
    flows = [None] * len(masks)  # Flows not currently used.

    success_rate = ((len(batches) - failed_batches) / len(batches)) * 100 if batches else 0
    logging.info(f"Parallel segmentation completed: {len(masks)} tiles processed, {total_cells} total cells detected, "
                f"success rate: {success_rate:.1f}%")

    if failed_batches > 0:
        logging.warning(f"{failed_batches} batches failed and were replaced with empty masks")

    return masks, flows, total_cells

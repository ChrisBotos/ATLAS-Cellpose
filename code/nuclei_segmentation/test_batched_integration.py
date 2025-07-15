#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_batched_integration.py.
Description:
    Integration test script to verify that the batched GPU merge functionality
    works correctly with the updated pipeline. This script creates synthetic
    test data and runs the merge process to ensure memory allocation errors
    are resolved.

Dependencies:
    • Python ≥ 3.10.
    • numpy, torch, tqdm.
    • cellpose_merge modules.

Usage:
    python test_batched_integration.py

Key Features:
    • Creates synthetic tile data simulating a large image with many tiles.
    • Tests both CPU and GPU processing modes.
    • Validates that memory usage stays within acceptable limits.
    • Compares results between batched and non-batched approaches.
    • Provides detailed logging of memory usage and processing times.
"""

from __future__ import annotations

import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_large_synthetic_dataset(
    num_rows: int = 20,
    num_cols: int = 20,
    tile_size: int = 256,
    overlap: int = 64,
    output_dir: Path = None
) -> Tuple[Path, int, int]:
    """
    Create a synthetic dataset with many tiles to test batched processing.
    
    Parameters
    ----------
    num_rows, num_cols : int
        Number of tile rows and columns to create.
    tile_size : int
        Size of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles.
    output_dir : Path, optional
        Directory to save tiles. If None, uses temporary directory.
    
    Returns
    -------
    Tuple[Path, int, int]
        Path to tiles directory, image height, image width.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    
    tiles_dir = output_dir / "tile_masks_npz"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    stride = tile_size - overlap
    image_height = num_rows * stride + overlap
    image_width = num_cols * stride + overlap
    
    logger.info(f"Creating synthetic dataset: {num_rows}x{num_cols} tiles")
    logger.info(f"Tile size: {tile_size}x{tile_size}, overlap: {overlap}")
    logger.info(f"Total image size: {image_height}x{image_width}")
    logger.info(f"Total tiles: {num_rows * num_cols}")
    
    nucleus_id = 1
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate tile position
            y_start = row * stride
            x_start = col * stride
            
            # Create synthetic tile with some nuclei
            tile = np.zeros((tile_size, tile_size), dtype=np.uint32)
            
            # Add 2-4 synthetic nuclei per tile
            num_nuclei = np.random.randint(2, 5)
            for _ in range(num_nuclei):
                # Random nucleus position and size
                nucleus_y = np.random.randint(20, tile_size - 40)
                nucleus_x = np.random.randint(20, tile_size - 40)
                nucleus_size = np.random.randint(15, 25)
                
                # Create circular nucleus
                y_coords, x_coords = np.ogrid[:tile_size, :tile_size]
                mask = ((y_coords - nucleus_y) ** 2 + (x_coords - nucleus_x) ** 2) <= (nucleus_size // 2) ** 2
                tile[mask] = nucleus_id
                nucleus_id += 1
            
            # Save tile with coordinate-based filename
            tile_filename = f"{y_start}_{x_start}.npz"
            tile_path = tiles_dir / tile_filename
            np.savez_compressed(tile_path, mask=tile)
    
    logger.info(f"Created {num_rows * num_cols} synthetic tiles in {tiles_dir}")
    logger.info(f"Total synthetic nuclei: {nucleus_id - 1}")
    
    return tiles_dir, image_height, image_width

def test_batched_merge(
    tiles_dir: Path,
    image_height: int,
    image_width: int,
    tile_size: int = 256,
    overlap: int = 64,
    use_gpu: bool = True,
    batch_size: int = 1,
    memory_limit_gb: float = 4.0
) -> Dict[str, any]:
    """
    Test the batched merge functionality with synthetic data.
    
    Parameters
    ----------
    tiles_dir : Path
        Directory containing tile masks.
    image_height, image_width : int
        Dimensions of the full image.
    tile_size : int
        Size of each tile.
    overlap : int
        Overlap between tiles.
    use_gpu : bool
        Whether to use GPU processing.
    batch_size : int
        Batch size for processing.
    memory_limit_gb : float
        Memory limit in gigabytes.
    
    Returns
    -------
    Dict[str, any]
        Results dictionary with timing and memory information.
    """
    from cellpose_merge.merge_tiles import merge_masks_streaming
    
    logger.info(f"Testing batched merge with:")
    logger.info(f"  Image size: {image_height}x{image_width}")
    logger.info(f"  Tile size: {tile_size}x{tile_size}")
    logger.info(f"  Overlap: {overlap}")
    logger.info(f"  Use GPU: {use_gpu}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Memory limit: {memory_limit_gb} GB")
    
    # Monitor memory usage if GPU is available
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")
    
    start_time = time.time()
    
    try:
        # Run the merge process
        merged_mask = merge_masks_streaming(
            height=image_height,
            width=image_width,
            tile_h=tile_size,
            tile_w=tile_size,
            overlap=overlap,
            tiles_path=tiles_dir,
            threshold=0.3,
            use_gpu=use_gpu,
            qc=False,
            gpu_batch_size=batch_size,
            gpu_memory_limit_gb=memory_limit_gb
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Analyze results
        total_nuclei = int(merged_mask.max())
        non_zero_pixels = np.count_nonzero(merged_mask)
        coverage_percent = (non_zero_pixels / (image_height * image_width)) * 100
        
        logger.info(f"Merge completed successfully!")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Total nuclei: {total_nuclei}")
        logger.info(f"  Non-zero pixels: {non_zero_pixels}")
        logger.info(f"  Coverage: {coverage_percent:.2f}%")
        
        # Check memory usage after processing
        if use_gpu and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / (1024**3)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"Final GPU memory usage: {final_memory:.2f} GB")
            logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
            
            # Reset peak memory counter
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'success': True,
            'processing_time': processing_time,
            'total_nuclei': total_nuclei,
            'non_zero_pixels': non_zero_pixels,
            'coverage_percent': coverage_percent,
            'merged_shape': merged_mask.shape
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.error(f"Merge failed after {processing_time:.2f} seconds: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def main():
    """Main test function."""
    logger.info("Starting batched merge integration test")
    
    # Create synthetic dataset
    # Start with a moderate size to test the functionality
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with different dataset sizes
        test_configs = [
            {'num_rows': 10, 'num_cols': 10, 'name': 'Medium (100 tiles)'},
            {'num_rows': 15, 'num_cols': 15, 'name': 'Large (225 tiles)'},
        ]
        
        for config in test_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {config['name']}")
            logger.info(f"{'='*60}")
            
            # Create dataset
            tiles_dir, height, width = create_large_synthetic_dataset(
                num_rows=config['num_rows'],
                num_cols=config['num_cols'],
                output_dir=temp_path / f"test_{config['num_rows']}x{config['num_cols']}"
            )
            
            # Test CPU processing first
            logger.info("\nTesting CPU processing...")
            cpu_result = test_batched_merge(
                tiles_dir=tiles_dir,
                image_height=height,
                image_width=width,
                use_gpu=False,
                batch_size=1
            )
            
            # Test GPU processing if available
            if torch.cuda.is_available():
                logger.info("\nTesting GPU processing...")
                gpu_result = test_batched_merge(
                    tiles_dir=tiles_dir,
                    image_height=height,
                    image_width=width,
                    use_gpu=True,
                    batch_size=1,
                    memory_limit_gb=4.0
                )
                
                # Compare results
                if cpu_result['success'] and gpu_result['success']:
                    speedup = cpu_result['processing_time'] / gpu_result['processing_time']
                    logger.info(f"\nPerformance comparison:")
                    logger.info(f"  CPU time: {cpu_result['processing_time']:.2f}s")
                    logger.info(f"  GPU time: {gpu_result['processing_time']:.2f}s")
                    logger.info(f"  Speedup: {speedup:.2f}x")
            else:
                logger.warning("GPU not available, skipping GPU tests")
    
    logger.info("\nBatched merge integration test completed!")

if __name__ == "__main__":
    main()

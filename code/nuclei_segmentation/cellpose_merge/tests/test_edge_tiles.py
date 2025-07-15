"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_edge_tiles.py.
Description:
    Comprehensive tests for edge tile handling in the merge_tiles module.
    These tests specifically verify that tiles extending beyond image boundaries
    are properly processed and merged, ensuring complete image coverage.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pathlib.
    • merge_tiles module from the parent directory.

Usage:
    pytest test_edge_tiles.py -v

Inputs:
    • Synthetic tile masks created for testing edge cases.

Outputs:
    • Test results verifying correct edge tile processing.

Key Features:
    • Tests for edge tiles that extend beyond image boundaries.
    • Verification of complete image coverage after merging.
    • Edge case handling for various image and tile size combinations.
    • Debugging output validation for edge tile processing.

Notes:
    • This test suite was created to address the critical bug where edge tiles
      were being discarded instead of properly merged.
    • All test functions include comprehensive assertions and debugging output.
"""

import traceback
import numpy as np
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple
from numpy.typing import NDArray

# Import the module under test.
import sys
sys.path.append(str(Path(__file__).parent.parent))
from .merge_tiles import merge_masks_streaming


def create_edge_test_tiles(
    height: int, 
    width: int, 
    tile_h: int, 
    tile_w: int, 
    overlap: int,
    temp_dir: Path
) -> Tuple[int, int]:
    """
    Create synthetic tile masks that include edge tiles extending beyond image boundaries.
    
    This function simulates the scenario where Cellpose has processed tiles that
    extend beyond the actual image dimensions, which is the root cause of the
    edge tile merging bug.
    
    Parameters
    ----------
    height, width : int
        Target image dimensions in pixels.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    temp_dir : Path
        Directory to save the synthetic tile masks.
        
    Returns
    -------
    Tuple[int, int]
        Number of tiles created and expected number of edge tiles.
    """
    
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    # Calculate the grid of tiles needed to cover the entire image.
    n_rows = (height + stride_h - 1) // stride_h
    n_cols = (width + stride_w - 1) // stride_w
    
    tiles_created = 0
    edge_tiles = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate tile position in global coordinates.
            global_y0 = r * stride_h
            global_x0 = c * stride_w
            
            # Determine actual tile size (may be smaller for edge tiles).
            actual_tile_h = min(tile_h, height - global_y0)
            actual_tile_w = min(tile_w, width - global_x0)
            
            # Skip tiles that would be completely outside the image.
            if actual_tile_h <= 0 or actual_tile_w <= 0:
                continue
                
            # Create a synthetic mask with unique labels for this tile.
            tile_mask = np.zeros((actual_tile_h, actual_tile_w), dtype=np.uint32)
            
            # Add some synthetic nuclei to the tile.
            # Use a pattern that will help us verify correct merging.
            label_id = r * n_cols + c + 1
            
            # Create a few synthetic nuclei in the tile.
            if actual_tile_h >= 20 and actual_tile_w >= 20:
                # Top-left nucleus.
                tile_mask[5:15, 5:15] = label_id
                
                # Bottom-right nucleus (if there's space).
                if actual_tile_h >= 30 and actual_tile_w >= 30:
                    tile_mask[actual_tile_h-15:actual_tile_h-5, actual_tile_w-15:actual_tile_w-5] = label_id + 1000
            
            # Save the tile using pixel coordinate naming convention.
            tile_filename = f"{global_y0}_{global_x0}.npz"
            tile_path = temp_dir / tile_filename
            np.savez_compressed(tile_path, mask=tile_mask)
            
            tiles_created += 1
            
            # Count edge tiles (tiles that extend to image boundaries).
            if (global_y0 + tile_h > height or global_x0 + tile_w > width or
                global_y0 == 0 or global_x0 == 0):
                edge_tiles += 1
    
    return tiles_created, edge_tiles


def test_edge_tile_coverage():
    """
    Test that edge tiles are properly processed and provide complete image coverage.
    
    This test creates a scenario where tiles extend beyond image boundaries
    and verifies that the merging process handles them correctly.
    """
    
    # Test parameters designed to create edge tiles.
    height, width = 1000, 800  # Non-square image to test asymmetric edge handling.
    tile_h, tile_w = 256, 256
    overlap = 64
    
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create synthetic tiles including edge cases.
        tiles_created, edge_tiles = create_edge_test_tiles(
            height, width, tile_h, tile_w, overlap, temp_path
        )
        
        print(f"Created {tiles_created} tiles, {edge_tiles} are edge tiles")
        assert edge_tiles > 0, "Test setup should create edge tiles"
        
        # Run the merge process.
        merged = merge_masks_streaming(
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            tiles_path=temp_path,
            threshold=0.3,
            use_gpu=False,  # Use CPU for deterministic testing.
            qc=False
        )
        
        # Verify the merged result has the correct dimensions.
        assert merged.shape == (height, width), f"Expected shape ({height}, {width}), got {merged.shape}"
        
        # Verify that we have non-zero pixels (nuclei were merged).
        total_nuclei_pixels = np.count_nonzero(merged)
        assert total_nuclei_pixels > 0, "Merged mask should contain nuclei pixels"
        
        # Critical test: Verify edge coverage.
        # Check that all four edges of the image have some segmentation.
        edge_margin = 50  # Check 50 pixels from each edge.
        
        top_edge = merged[:edge_margin, :].sum()
        bottom_edge = merged[-edge_margin:, :].sum()
        left_edge = merged[:, :edge_margin].sum()
        right_edge = merged[:, -edge_margin:].sum()
        
        print(f"Edge coverage: top={top_edge}, bottom={bottom_edge}, left={left_edge}, right={right_edge}")
        
        # All edges should have some coverage (this was the bug - edges were empty).
        assert top_edge > 0, "Top edge should have segmentation coverage"
        assert bottom_edge > 0, "Bottom edge should have segmentation coverage"
        assert left_edge > 0, "Left edge should have segmentation coverage"
        assert right_edge > 0, "Right edge should have segmentation coverage"
        
        print(f"✓ Edge tile test passed: {total_nuclei_pixels} nuclei pixels with full edge coverage")


def test_single_edge_tile():
    """
    Test processing of a single edge tile that extends beyond image boundaries.
    
    This is a focused test for the specific edge case that was causing the bug.
    """
    
    # Create a small image with a single tile that extends beyond boundaries.
    height, width = 200, 150
    tile_h, tile_w = 256, 256  # Tile larger than image.
    overlap = 32
    
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a single tile at (0,0) that extends beyond the image.
        tile_mask = np.zeros((height, width), dtype=np.uint32)  # Actual size within image.
        tile_mask[50:100, 50:100] = 1  # Add a nucleus.
        
        tile_filename = "0_0.npz"
        tile_path = temp_path / tile_filename
        np.savez_compressed(tile_path, mask=tile_mask)
        
        # Run the merge process.
        merged = merge_masks_streaming(
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            tiles_path=temp_path,
            threshold=0.3,
            use_gpu=False,
            qc=False
        )
        
        # Verify the result.
        assert merged.shape == (height, width), f"Expected shape ({height}, {width}), got {merged.shape}"
        assert np.count_nonzero(merged) > 0, "Should have nuclei pixels from the edge tile"
        assert merged[75, 75] == 1, "Nucleus should be preserved at the expected location"
        
        print("✓ Single edge tile test passed")


if __name__ == "__main__":
    # Run tests directly for debugging.
    test_edge_tile_coverage()
    test_single_edge_tile()
    print("All edge tile tests passed!")

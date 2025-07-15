"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_edge_fix.py.
Description:
    Simple test script to verify that the edge tile merging bug has been fixed.
    This script creates synthetic tiles that extend beyond image boundaries and
    verifies that they are properly merged to provide complete image coverage.

Dependencies:
    • Python >= 3.10.
    • numpy, pathlib.
    • merge_tiles module from the current directory.

Usage:
    python test_edge_fix.py

Inputs:
    • Synthetic tile masks created for testing edge cases.

Outputs:
    • Test results printed to console.
    • Verification of complete image coverage after merging.

Key Features:
    • Creates edge tiles that extend beyond image boundaries.
    • Tests the fixed merging logic for complete coverage.
    • Provides clear pass/fail results with diagnostic information.

Notes:
    • This test was created to verify the fix for the critical edge tile bug.
    • The test creates a scenario that would have failed before the fix.
"""

import traceback
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import logging

# Set up logging to see debug output.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import the fixed merge function.
from .merge_tiles import merge_masks_streaming


def create_test_tiles(temp_dir: Path) -> None:
    """
    Create synthetic tiles that include edge cases.
    
    This creates tiles that extend beyond image boundaries to test
    the edge tile handling fix.
    """
    
    # Create tiles that would cover a 1000x800 image with 256x256 tiles and 64px overlap.
    height, width = 1000, 800
    tile_h, tile_w = 256, 256
    overlap = 64
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    # Calculate required tiles.
    n_rows = (height + stride_h - 1) // stride_h
    n_cols = (width + stride_w - 1) // stride_w
    
    print(f"Creating test tiles for {height}x{width} image")
    print(f"Tile grid: {n_rows} rows x {n_cols} cols")
    
    tiles_created = 0
    edge_tiles = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate tile position.
            global_y0 = r * stride_h
            global_x0 = c * stride_w
            
            # Determine actual tile size (may be smaller for edge tiles).
            actual_tile_h = min(tile_h, height - global_y0)
            actual_tile_w = min(tile_w, width - global_x0)
            
            if actual_tile_h <= 0 or actual_tile_w <= 0:
                continue
                
            # Create synthetic mask.
            tile_mask = np.zeros((actual_tile_h, actual_tile_w), dtype=np.uint32)
            
            # Add synthetic nuclei.
            label_id = r * n_cols + c + 1
            
            if actual_tile_h >= 20 and actual_tile_w >= 20:
                # Add a nucleus in the center.
                center_y = actual_tile_h // 2
                center_x = actual_tile_w // 2
                tile_mask[center_y-5:center_y+5, center_x-5:center_x+5] = label_id
            
            # Save tile using pixel coordinate naming.
            tile_filename = f"{global_y0}_{global_x0}.npz"
            tile_path = temp_dir / tile_filename
            np.savez_compressed(tile_path, mask=tile_mask)
            
            tiles_created += 1
            
            # Count edge tiles.
            if (global_y0 + tile_h > height or global_x0 + tile_w > width or
                global_y0 == 0 or global_x0 == 0):
                edge_tiles += 1
    
    print(f"Created {tiles_created} tiles, {edge_tiles} are edge tiles")


def test_edge_tile_fix():
    """
    Test that the edge tile fix works correctly.
    """
    
    print("=" * 60)
    print("TESTING EDGE TILE MERGING FIX")
    print("=" * 60)
    
    # Test parameters.
    height, width = 1000, 800
    tile_h, tile_w = 256, 256
    overlap = 64
    
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test tiles.
            create_test_tiles(temp_path)
            
            print("\nRunning merge process...")
            
            # Run the merge with our fixed code.
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
            
            # Verify results.
            print(f"\nMerge completed successfully!")
            print(f"Output shape: {merged.shape}")
            print(f"Expected shape: ({height}, {width})")
            
            assert merged.shape == (height, width), f"Shape mismatch: got {merged.shape}, expected ({height}, {width})"
            
            total_nuclei = np.count_nonzero(merged)
            print(f"Total nuclei pixels: {total_nuclei}")
            
            assert total_nuclei > 0, "No nuclei found in merged result"
            
            # CRITICAL TEST: Check edge coverage.
            edge_margin = 50
            
            top_edge = merged[:edge_margin, :].sum()
            bottom_edge = merged[-edge_margin:, :].sum()
            left_edge = merged[:, :edge_margin].sum()
            right_edge = merged[:, -edge_margin:].sum()
            
            print(f"\nEdge coverage test:")
            print(f"  Top edge (first {edge_margin} rows): {top_edge} pixels")
            print(f"  Bottom edge (last {edge_margin} rows): {bottom_edge} pixels")
            print(f"  Left edge (first {edge_margin} cols): {left_edge} pixels")
            print(f"  Right edge (last {edge_margin} cols): {right_edge} pixels")
            
            # Before the fix, edges would be zero. After the fix, they should have coverage.
            edges_with_coverage = sum(1 for edge in [top_edge, bottom_edge, left_edge, right_edge] if edge > 0)
            
            print(f"\nEdges with coverage: {edges_with_coverage}/4")
            
            if edges_with_coverage == 4:
                print("✓ EDGE TILE FIX SUCCESSFUL: All edges have segmentation coverage!")
                return True
            else:
                print("✗ EDGE TILE FIX FAILED: Some edges have no coverage")
                return False
                
    except Exception as e:
        print(f"✗ TEST FAILED WITH ERROR: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_edge_tile_fix()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EDGE TILE FIX VERIFICATION PASSED!")
        print("The tile merging bug has been successfully fixed.")
        print("Edge tiles are now properly processed and merged.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ EDGE TILE FIX VERIFICATION FAILED!")
        print("The bug may not be fully resolved.")
        print("=" * 60)

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_tile_borders.py.
Description:
    Debug script to examine tile border detection and verify that complete tile
    border information is being correctly computed and passed to the merge function.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.

Usage:
    python debug_tile_borders.py

Key Features:
    • Loads actual tile masks from the latest run.
    • Tests border detection on complete tiles vs overlap regions.
    • Identifies why Step 2 border deletion is not working.

Notes:
    • This script helps identify the root cause of persistent straight-line artifacts.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the cellpose_merge directory to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'nuclei_segmentation', 'cellpose_merge'))

from rules import _find_border_touching_nuclei

def find_latest_results_dir():
    """Find the latest results directory."""
    results_dir = Path("C:/Projects/I-R-Injury-Spatial-Multiomics-Analysis/results")
    
    # Find all directories that match the pattern.
    pattern_dirs = list(results_dir.glob("*test_new_cellpose4_diameter0_large_crop"))
    
    if not pattern_dirs:
        raise FileNotFoundError("No results directories found")
    
    # Sort by modification time and get the latest.
    latest_dir = max(pattern_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir

def load_tile_masks(results_dir):
    """Load individual tile masks from the results directory."""
    tile_masks_dir = results_dir / "masks" / "tile_masks_npz"
    
    if not tile_masks_dir.exists():
        raise FileNotFoundError(f"Tile masks directory not found: {tile_masks_dir}")
    
    print(f"Loading tile masks from: {tile_masks_dir}")
    
    # Find all .npz files.
    npz_files = list(tile_masks_dir.glob("*.npz"))
    npz_files.sort()
    
    print(f"Found {len(npz_files)} tile mask files")
    
    tiles = {}
    for npz_file in npz_files:
        data = np.load(npz_file)
        tile_id = npz_file.stem
        mask = data['mask']
        tiles[tile_id] = mask
        print(f"Loaded {tile_id}: shape {mask.shape}, nuclei: {len(np.unique(mask[mask > 0]))}")
    
    return tiles

def test_border_detection_on_tiles(tiles):
    """Test border detection on actual tile masks."""
    print("\n=== Testing Border Detection on Complete Tiles ===")
    
    for tile_id, mask in tiles.items():
        border_nuclei = _find_border_touching_nuclei(mask)
        total_nuclei = len(np.unique(mask[mask > 0]))
        
        print(f"\nTile {tile_id}:")
        print(f"  Shape: {mask.shape}")
        print(f"  Total nuclei: {total_nuclei}")
        print(f"  Border-touching nuclei: {len(border_nuclei)}")
        print(f"  Border nuclei IDs: {sorted(border_nuclei) if len(border_nuclei) <= 10 else f'{len(border_nuclei)} nuclei'}")
        
        # Calculate border percentage.
        border_percentage = len(border_nuclei) / total_nuclei * 100 if total_nuclei > 0 else 0
        print(f"  Border percentage: {border_percentage:.1f}%")

def simulate_overlap_merge(tiles):
    """Simulate the overlap merge process to see what's happening."""
    print("\n=== Simulating Overlap Merge Process ===")
    
    # Get the first two tiles for testing.
    tile_ids = list(tiles.keys())
    if len(tile_ids) < 2:
        print("Need at least 2 tiles for overlap testing")
        return
    
    tile1_id = tile_ids[0]
    tile2_id = tile_ids[1]
    
    tile1_mask = tiles[tile1_id]
    tile2_mask = tiles[tile2_id]
    
    print(f"Testing overlap between {tile1_id} and {tile2_id}")
    
    # Get complete tile border information.
    tile1_border_nuclei = _find_border_touching_nuclei(tile1_mask)
    tile2_border_nuclei = _find_border_touching_nuclei(tile2_mask)
    
    print(f"Complete tile1 border nuclei: {len(tile1_border_nuclei)}")
    print(f"Complete tile2 border nuclei: {len(tile2_border_nuclei)}")
    
    # Simulate overlap regions (assume they overlap in some region).
    # For testing, let's take the right edge of tile1 and left edge of tile2.
    overlap_width = 102  # Standard overlap.
    
    if tile1_mask.shape[1] >= overlap_width and tile2_mask.shape[1] >= overlap_width:
        overlap1 = tile1_mask[:, -overlap_width:]  # Right edge of tile1.
        overlap2 = tile2_mask[:, :overlap_width]   # Left edge of tile2.
        
        print(f"Overlap1 shape: {overlap1.shape}")
        print(f"Overlap2 shape: {overlap2.shape}")
        
        # Get nuclei in overlap regions.
        overlap1_nuclei = set(np.unique(overlap1[overlap1 > 0]))
        overlap2_nuclei = set(np.unique(overlap2[overlap2 > 0]))
        
        print(f"Overlap1 nuclei: {len(overlap1_nuclei)} - {sorted(overlap1_nuclei) if len(overlap1_nuclei) <= 10 else f'{len(overlap1_nuclei)} nuclei'}")
        print(f"Overlap2 nuclei: {len(overlap2_nuclei)} - {sorted(overlap2_nuclei) if len(overlap2_nuclei) <= 10 else f'{len(overlap2_nuclei)} nuclei'}")
        
        # Test border detection on overlap regions vs complete tiles.
        overlap1_border_nuclei = _find_border_touching_nuclei(overlap1)
        overlap2_border_nuclei = _find_border_touching_nuclei(overlap2)
        
        print(f"\nBorder detection comparison:")
        print(f"Overlap1 border nuclei (overlap region): {len(overlap1_border_nuclei)}")
        print(f"Overlap2 border nuclei (overlap region): {len(overlap2_border_nuclei)}")
        
        # Check which nuclei are incorrectly identified as border-touching.
        for nucleus_id in overlap1_nuclei:
            in_overlap_border = nucleus_id in overlap1_border_nuclei
            in_complete_border = nucleus_id in tile1_border_nuclei
            
            if in_overlap_border != in_complete_border:
                print(f"  MISMATCH: Nucleus {nucleus_id} - overlap border: {in_overlap_border}, complete border: {in_complete_border}")
        
        for nucleus_id in overlap2_nuclei:
            in_overlap_border = nucleus_id in overlap2_border_nuclei
            in_complete_border = nucleus_id in tile2_border_nuclei
            
            if in_overlap_border != in_complete_border:
                print(f"  MISMATCH: Nucleus {nucleus_id} - overlap border: {in_overlap_border}, complete border: {in_complete_border}")

def main():
    """Main debugging function."""
    print("Tile Border Detection Debug Script")
    print("=" * 50)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load tile masks.
        tiles = load_tile_masks(results_dir)
        
        # Test border detection on complete tiles.
        test_border_detection_on_tiles(tiles)
        
        # Simulate overlap merge process.
        simulate_overlap_merge(tiles)
        
        print("\n=== Debug Summary ===")
        print("Check the output above for:")
        print("1. Whether complete tile border detection is working correctly")
        print("2. Whether there are mismatches between overlap and complete border detection")
        print("3. Whether the tile border information is being correctly computed")
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

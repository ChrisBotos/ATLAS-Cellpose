"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: investigate_interior_deletion_bug.py.
Description:
    Critical investigation script to identify the bug causing nuclei deletion
    from tile INTERIOR regions during merging. This should NEVER happen in
    correct merge behavior - only overlap regions should be processed.

Dependencies:
    • Python ≥ 3.10.
    • numpy, matplotlib, pathlib.

Usage:
    python investigate_interior_deletion_bug.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/
    • Merged tile masks from masks/merged_tile_masks_npz/

Outputs:
    • Detailed analysis of interior nuclei deletion.
    • Pixel-by-pixel comparison maps.
    • Root cause identification.

Key Features:
    • Compares original vs merged tile masks.
    • Identifies nuclei deleted from non-overlapping regions.
    • Analyzes deletion patterns and affected nucleus IDs.
    • Provides visual debugging output.

Notes:
    • This bug is causing massive nuclei loss in tile interiors.
    • The merging algorithm should ONLY modify overlap regions.
    • Interior regions must remain completely untouched.
"""

import traceback
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Set
from numpy.typing import NDArray

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tile_mask(tile_path: Path) -> NDArray[np.uint32]:
    """Load a tile mask from .npz file."""
    try:
        data = np.load(tile_path)
        return data["mask"].astype(np.uint32)
    except Exception as e:
        logging.error(f"Failed to load tile mask from {tile_path}: {e}")
        return np.zeros((512, 512), dtype=np.uint32)

def analyze_tile_interior_deletion(results_dir: Path):
    """
    Analyze nuclei deletion in tile interior regions.
    """
    print("=" * 80)
    print("CRITICAL BUG INVESTIGATION: INTERIOR NUCLEI DELETION")
    print("=" * 80)
    
    # Define paths.
    original_tiles_dir = results_dir / "masks" / "tile_masks_npz"
    merged_tiles_dir = results_dir / "masks" / "merged_tile_masks_npz"
    
    if not original_tiles_dir.exists():
        print(f"❌ Original tiles directory not found: {original_tiles_dir}")
        return
    
    if not merged_tiles_dir.exists():
        print(f"❌ Merged tiles directory not found: {merged_tiles_dir}")
        return
    
    # Focus on top-left tiles as mentioned.
    target_tiles = [
        (0, 0),    # Top-left corner.
        (0, 410),  # Top row, second tile.
        (410, 0),  # Second row, first tile.
        (410, 410) # Second row, second tile.
    ]
    
    print(f"Analyzing {len(target_tiles)} target tiles for interior deletion bug...")
    
    total_interior_deletions = 0
    
    for tile_coord in target_tiles:
        r, c = tile_coord
        tile_name = f"{r}_{c}.npz"
        
        print(f"\n{'='*60}")
        print(f"ANALYZING TILE: {tile_name}")
        print(f"{'='*60}")
        
        # Load original and merged masks.
        original_path = original_tiles_dir / tile_name
        merged_path = merged_tiles_dir / tile_name
        
        if not original_path.exists():
            print(f"❌ Original tile not found: {original_path}")
            continue
            
        if not merged_path.exists():
            print(f"❌ Merged tile not found: {merged_path}")
            continue
        
        original_mask = load_tile_mask(original_path)
        merged_mask = load_tile_mask(merged_path)
        
        # Count nuclei.
        original_nuclei = np.unique(original_mask[original_mask > 0])
        merged_nuclei = np.unique(merged_mask[merged_mask > 0])
        
        print(f"Original nuclei count: {len(original_nuclei)}")
        print(f"Merged nuclei count: {len(merged_nuclei)}")
        print(f"Nuclei lost: {len(original_nuclei) - len(merged_nuclei)}")
        
        if len(merged_nuclei) >= len(original_nuclei):
            print("✅ No nuclei loss detected in this tile")
            continue
        
        # Identify deleted nuclei.
        deleted_nuclei = set(original_nuclei) - set(merged_nuclei)
        print(f"Deleted nucleus IDs: {sorted(deleted_nuclei)}")
        
        # CRITICAL ANALYSIS: Check if deletions are in interior regions.
        interior_deletions = analyze_interior_deletions(
            original_mask, merged_mask, deleted_nuclei, tile_coord
        )
        
        total_interior_deletions += interior_deletions
        
        # Generate visual comparison.
        create_deletion_visualization(
            original_mask, merged_mask, tile_coord, results_dir
        )
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {total_interior_deletions} nuclei deleted from tile INTERIORS")
    print(f"{'='*80}")
    
    if total_interior_deletions > 0:
        print("❌ CRITICAL BUG CONFIRMED: Nuclei are being deleted from tile interiors!")
        print("   This should NEVER happen - only overlap regions should be modified.")
    else:
        print("✅ No interior deletions detected - bug may be elsewhere.")

def analyze_interior_deletions(
    original_mask: NDArray[np.uint32], 
    merged_mask: NDArray[np.uint32], 
    deleted_nuclei: Set[int],
    tile_coord: Tuple[int, int]
) -> int:
    """
    Analyze which deleted nuclei were in interior (non-overlap) regions.
    """
    r, c = tile_coord
    h, w = original_mask.shape
    
    # Define overlap regions based on tile position.
    # Standard overlap is 102 pixels.
    overlap = 102
    
    # Define interior region (non-overlapping area).
    # For a 512x512 tile with 102 pixel overlap:
    interior_top = overlap if r > 0 else 0
    interior_bottom = h - overlap if r < 1230 else h  # Assuming 4x4 grid.
    interior_left = overlap if c > 0 else 0
    interior_right = w - overlap if c < 1230 else w
    
    print(f"Interior region: [{interior_top}:{interior_bottom}, {interior_left}:{interior_right}]")
    
    interior_deletions = 0
    
    for nucleus_id in deleted_nuclei:
        # Find where this nucleus was located in the original mask.
        nucleus_mask = (original_mask == nucleus_id)
        
        if not np.any(nucleus_mask):
            continue
        
        # Get nucleus bounding box.
        rows, cols = np.where(nucleus_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Check if nucleus is completely in interior region.
        completely_interior = (
            min_row >= interior_top and max_row < interior_bottom and
            min_col >= interior_left and max_col < interior_right
        )
        
        # Check if nucleus touches interior region.
        touches_interior = (
            max_row >= interior_top and min_row < interior_bottom and
            max_col >= interior_left and min_col < interior_right
        )
        
        if completely_interior:
            print(f"  ❌ CRITICAL: Nucleus {nucleus_id} deleted from INTERIOR region!")
            print(f"     Location: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
            interior_deletions += 1
        elif touches_interior:
            print(f"  ⚠️  WARNING: Nucleus {nucleus_id} partially in interior region")
            print(f"     Location: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        else:
            print(f"  ✅ OK: Nucleus {nucleus_id} only in overlap region")
            print(f"     Location: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    
    return interior_deletions

def create_deletion_visualization(
    original_mask: NDArray[np.uint32],
    merged_mask: NDArray[np.uint32], 
    tile_coord: Tuple[int, int],
    results_dir: Path
):
    """
    Create a visualization showing deleted nuclei.
    """
    r, c = tile_coord
    
    # Create difference mask.
    deletion_mask = (original_mask > 0) & (merged_mask == 0)
    
    # Create visualization.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original mask.
    axes[0].imshow(original_mask > 0, cmap='gray')
    axes[0].set_title(f'Original Tile ({r}, {c})')
    axes[0].set_xlabel(f'Nuclei: {len(np.unique(original_mask[original_mask > 0]))}')
    
    # Merged mask.
    axes[1].imshow(merged_mask > 0, cmap='gray')
    axes[1].set_title(f'Merged Tile ({r}, {c})')
    axes[1].set_xlabel(f'Nuclei: {len(np.unique(merged_mask[merged_mask > 0]))}')
    
    # Deletion mask.
    axes[2].imshow(deletion_mask, cmap='Reds')
    axes[2].set_title(f'Deleted Nuclei ({r}, {c})')
    axes[2].set_xlabel(f'Deleted pixels: {np.sum(deletion_mask)}')
    
    # Add overlap region boundaries.
    overlap = 102
    h, w = original_mask.shape
    
    for ax in axes:
        # Draw overlap boundaries.
        if r > 0:  # Top overlap.
            ax.axhline(y=overlap, color='red', linestyle='--', alpha=0.7)
        if r < 1230:  # Bottom overlap.
            ax.axhline(y=h-overlap, color='red', linestyle='--', alpha=0.7)
        if c > 0:  # Left overlap.
            ax.axvline(x=overlap, color='red', linestyle='--', alpha=0.7)
        if c < 1230:  # Right overlap.
            ax.axvline(x=w-overlap, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save visualization.
    output_path = results_dir / f"debug_tile_deletion_{r}_{c}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved deletion visualization: {output_path}")

if __name__ == "__main__":
    try:
        results_dir = Path("results/20250723_043122_cpu_cellpose4_diameter0_large_crop")
        
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            exit(1)
        
        analyze_tile_interior_deletion(results_dir)
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

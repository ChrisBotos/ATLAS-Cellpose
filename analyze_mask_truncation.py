#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: analyze_mask_truncation.py.
Description:
    Analyze mask truncation issues by comparing before/after merging QC visualizations
    and examining the underlying mask data to identify where non-priority tile nuclei
    are being improperly truncated at tile boundaries.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.
    • PIL for image loading.

Usage:
    python analyze_mask_truncation.py

Key Features:
    • Compares before_merging.tif and after_merging.tif QC visualizations.
    • Examines original tile masks vs merged tile masks.
    • Identifies specific nuclei that are being truncated.
    • Visualizes the truncation issue at tile boundaries.

Notes:
    • This script helps identify why non-priority nuclei are being cut at boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

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

def load_qc_images(results_dir):
    """Load the before and after merging QC images."""
    qc_dir = results_dir / "merge_qc_overlays"
    
    before_path = qc_dir / "before_merging.tif"
    after_path = qc_dir / "after_merging.tif"
    
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(f"QC images not found in {qc_dir}")
    
    print(f"Loading QC images from: {qc_dir}")
    
    before_img = np.array(Image.open(before_path))
    after_img = np.array(Image.open(after_path))
    
    print(f"Before image shape: {before_img.shape}")
    print(f"After image shape: {after_img.shape}")
    
    return before_img, after_img

def load_mask_data(results_dir):
    """Load original and merged mask data."""
    masks_dir = results_dir / "masks"
    
    # Load original tile masks.
    original_tiles_dir = masks_dir / "tile_masks_npz"
    merged_tiles_dir = masks_dir / "merged_tile_masks_npz"
    final_mask_path = masks_dir / "segmentation_masks.npy"
    
    print(f"Loading mask data from: {masks_dir}")
    
    # Load original tiles.
    original_tiles = {}
    if original_tiles_dir.exists():
        for npz_file in original_tiles_dir.glob("*.npz"):
            data = np.load(npz_file)
            tile_id = npz_file.stem
            original_tiles[tile_id] = data['mask']
            print(f"Loaded original tile {tile_id}: shape {data['mask'].shape}, nuclei: {len(np.unique(data['mask'][data['mask'] > 0]))}")
    
    # Load merged tiles.
    merged_tiles = {}
    if merged_tiles_dir.exists():
        for npz_file in merged_tiles_dir.glob("*.npz"):
            data = np.load(npz_file)
            tile_id = npz_file.stem
            merged_tiles[tile_id] = data['mask']
            print(f"Loaded merged tile {tile_id}: shape {data['mask'].shape}, nuclei: {len(np.unique(data['mask'][data['mask'] > 0]))}")
    
    # Load final mask.
    final_mask = None
    if final_mask_path.exists():
        final_mask = np.load(final_mask_path)
        print(f"Loaded final mask: shape {final_mask.shape}, nuclei: {len(np.unique(final_mask[final_mask > 0]))}")
    
    return original_tiles, merged_tiles, final_mask

def analyze_boundary_region(original_tiles, merged_tiles, boundary_type="vertical", boundary_pos=410):
    """Analyze a specific boundary region for truncation issues."""
    print(f"\n=== Analyzing {boundary_type} boundary at position {boundary_pos} ===")
    
    if boundary_type == "vertical":
        # For vertical boundary at x=410, compare tiles (0,0) and (0,410).
        if boundary_pos == 410:
            tile1_id, tile2_id = "0_0", "0_410"
        elif boundary_pos == 820:
            tile1_id, tile2_id = "0_410", "0_820"
        else:
            print(f"Unsupported vertical boundary position: {boundary_pos}")
            return
    else:  # horizontal
        # For horizontal boundary at y=410, compare tiles (0,0) and (410,0).
        if boundary_pos == 410:
            tile1_id, tile2_id = "0_0", "410_0"
        elif boundary_pos == 820:
            tile1_id, tile2_id = "410_0", "820_0"
        else:
            print(f"Unsupported horizontal boundary position: {boundary_pos}")
            return
    
    # Get original and merged tiles.
    orig_tile1 = original_tiles.get(tile1_id)
    orig_tile2 = original_tiles.get(tile2_id)

    # Try different naming conventions for merged tiles.
    merged_tile1 = merged_tiles.get(tile1_id)
    merged_tile2 = merged_tiles.get(tile2_id)

    # If not found, try grid-based naming (0_0 -> 0_0, 0_410 -> 0_1, etc.).
    if merged_tile1 is None or merged_tile2 is None:
        print(f"Trying alternative merged tile naming...")
        print(f"Available merged tiles: {list(merged_tiles.keys())}")

        # Map position-based names to grid-based names.
        pos_to_grid = {
            "0_0": "0_0", "0_410": "0_1", "0_820": "0_2",
            "410_0": "1_0", "410_410": "1_1", "410_820": "1_2",
            "820_0": "2_0", "820_410": "2_1", "820_820": "2_2"
        }

        grid_tile1_id = pos_to_grid.get(tile1_id)
        grid_tile2_id = pos_to_grid.get(tile2_id)

        if grid_tile1_id and grid_tile2_id:
            merged_tile1 = merged_tiles.get(grid_tile1_id)
            merged_tile2 = merged_tiles.get(grid_tile2_id)
            print(f"Using grid naming: {tile1_id} -> {grid_tile1_id}, {tile2_id} -> {grid_tile2_id}")
    
    if any(tile is None for tile in [orig_tile1, orig_tile2, merged_tile1, merged_tile2]):
        print(f"Missing tiles for boundary analysis: {tile1_id}, {tile2_id}")
        return
    
    print(f"Comparing tiles {tile1_id} and {tile2_id}")
    print(f"Original {tile1_id}: {orig_tile1.shape}, nuclei: {len(np.unique(orig_tile1[orig_tile1 > 0]))}")
    print(f"Original {tile2_id}: {orig_tile2.shape}, nuclei: {len(np.unique(orig_tile2[orig_tile2 > 0]))}")
    print(f"Merged {tile1_id}: {merged_tile1.shape}, nuclei: {len(np.unique(merged_tile1[merged_tile1 > 0]))}")
    print(f"Merged {tile2_id}: {merged_tile2.shape}, nuclei: {len(np.unique(merged_tile2[merged_tile2 > 0]))}")
    
    # Analyze the overlap region.
    overlap = 102
    
    if boundary_type == "vertical":
        # Vertical boundary - analyze the right edge of tile1 and left edge of tile2.
        tile1_overlap = orig_tile1[:, -overlap:]
        tile2_overlap = orig_tile2[:, :overlap]
        merged_tile1_overlap = merged_tile1[:, -overlap:]
        merged_tile2_overlap = merged_tile2[:, :overlap]
    else:  # horizontal
        # Horizontal boundary - analyze the bottom edge of tile1 and top edge of tile2.
        tile1_overlap = orig_tile1[-overlap:, :]
        tile2_overlap = orig_tile2[:overlap, :]
        merged_tile1_overlap = merged_tile1[-overlap:, :]
        merged_tile2_overlap = merged_tile2[:overlap, :]
    
    # Find nuclei in overlap regions.
    orig_tile1_nuclei = set(np.unique(tile1_overlap[tile1_overlap > 0]))
    orig_tile2_nuclei = set(np.unique(tile2_overlap[tile2_overlap > 0]))
    merged_tile1_nuclei = set(np.unique(merged_tile1_overlap[merged_tile1_overlap > 0]))
    merged_tile2_nuclei = set(np.unique(merged_tile2_overlap[merged_tile2_overlap > 0]))
    
    print(f"Original overlap nuclei - {tile1_id}: {len(orig_tile1_nuclei)}, {tile2_id}: {len(orig_tile2_nuclei)}")
    print(f"Merged overlap nuclei - {tile1_id}: {len(merged_tile1_nuclei)}, {tile2_id}: {len(merged_tile2_nuclei)}")
    
    # Check for truncation by comparing nucleus shapes.
    print(f"\n=== Checking for nucleus truncation ===")
    
    # Sample a few nuclei from tile2 (non-priority) to check if they're truncated.
    sample_nuclei = list(orig_tile2_nuclei)[:5]  # Check first 5 nuclei.
    
    for nucleus_id in sample_nuclei:
        # Get the full nucleus shape from original tile2.
        orig_nucleus_mask = (orig_tile2 == nucleus_id)
        orig_nucleus_pixels = np.sum(orig_nucleus_mask)
        
        # Get the nucleus shape from merged tile2.
        merged_nucleus_mask = (merged_tile2 == nucleus_id)
        merged_nucleus_pixels = np.sum(merged_nucleus_mask)
        
        # Check if the nucleus exists in merged tile1 (cross-boundary).
        merged_tile1_has_nucleus = nucleus_id in merged_tile1_nuclei
        
        print(f"Nucleus {nucleus_id}:")
        print(f"  Original pixels: {orig_nucleus_pixels}")
        print(f"  Merged pixels: {merged_nucleus_pixels}")
        print(f"  In merged tile1: {merged_tile1_has_nucleus}")
        
        if merged_nucleus_pixels < orig_nucleus_pixels:
            print(f"  ❌ TRUNCATED: Lost {orig_nucleus_pixels - merged_nucleus_pixels} pixels")
        elif merged_nucleus_pixels == orig_nucleus_pixels:
            print(f"  ✅ PRESERVED: Same pixel count")
        else:
            print(f"  ⚠️  EXPANDED: Gained {merged_nucleus_pixels - orig_nucleus_pixels} pixels")

def create_boundary_comparison_visualization(before_img, after_img, output_path):
    """Create a visualization comparing before and after images at tile boundaries."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full images.
    axes[0, 0].imshow(before_img)
    axes[0, 0].set_title('Before Merging (Full)')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(after_img)
    axes[1, 0].set_title('After Merging (Full)')
    axes[1, 0].axis('off')
    
    # Add tile boundary lines.
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.axvline(410, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(820, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(410, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(820, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
    
    # Crop around vertical boundary at x=410.
    crop_width = 100
    x_center = 410
    y_center = before_img.shape[0] // 2
    
    x_start = max(0, x_center - crop_width)
    x_end = min(before_img.shape[1], x_center + crop_width)
    y_start = max(0, y_center - crop_width)
    y_end = min(before_img.shape[0], y_center + crop_width)
    
    before_crop = before_img[y_start:y_end, x_start:x_end]
    after_crop = after_img[y_start:y_end, x_start:x_end]
    
    axes[0, 1].imshow(before_crop)
    axes[0, 1].set_title(f'Before: Vertical Boundary x={x_center}')
    axes[0, 1].axvline(x_center - x_start, color='red', linestyle='-', linewidth=2)
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(after_crop)
    axes[1, 1].set_title(f'After: Vertical Boundary x={x_center}')
    axes[1, 1].axvline(x_center - x_start, color='red', linestyle='-', linewidth=2)
    axes[1, 1].axis('off')
    
    # Crop around horizontal boundary at y=410.
    y_center = 410
    x_center = before_img.shape[1] // 2
    
    x_start = max(0, x_center - crop_width)
    x_end = min(before_img.shape[1], x_center + crop_width)
    y_start = max(0, y_center - crop_width)
    y_end = min(before_img.shape[0], y_center + crop_width)
    
    before_crop = before_img[y_start:y_end, x_start:x_end]
    after_crop = after_img[y_start:y_end, x_start:x_end]
    
    axes[0, 2].imshow(before_crop)
    axes[0, 2].set_title(f'Before: Horizontal Boundary y={y_center}')
    axes[0, 2].axhline(y_center - y_start, color='red', linestyle='-', linewidth=2)
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(after_crop)
    axes[1, 2].set_title(f'After: Horizontal Boundary y={y_center}')
    axes[1, 2].axhline(y_center - y_start, color='red', linestyle='-', linewidth=2)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Boundary comparison visualization saved to: {output_path}")

def main():
    """Main analysis function."""
    print("Mask Truncation Analysis")
    print("=" * 50)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load QC images.
        before_img, after_img = load_qc_images(results_dir)
        
        # Load mask data.
        original_tiles, merged_tiles, final_mask = load_mask_data(results_dir)
        
        # Analyze specific boundary regions.
        analyze_boundary_region(original_tiles, merged_tiles, "vertical", 410)
        analyze_boundary_region(original_tiles, merged_tiles, "horizontal", 410)
        
        # Create comparison visualization.
        output_path = "mask_truncation_analysis.png"
        create_boundary_comparison_visualization(before_img, after_img, output_path)
        
        print(f"\n=== ANALYSIS SUMMARY ===")
        print("This analysis compares before/after QC images and examines mask data")
        print("to identify where non-priority tile nuclei are being truncated.")
        print("Look for:")
        print("1. Nuclei that appear complete in 'before' but truncated in 'after'")
        print("2. Sharp boundaries in 'after' that were smooth in 'before'")
        print("3. Pixel count reductions in merged tiles vs original tiles")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: examine_original_tiles.py.
Description:
    Examine the original individual tile masks before any merging to determine
    if straight-line artifacts are already present in the Cellpose segmentation.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.
    • skimage for image processing.

Usage:
    python examine_original_tiles.py

Key Features:
    • Loads original tile masks from tile_masks_npz directory.
    • Analyzes edge patterns in individual tiles.
    • Reconstructs the image from original tiles without merging.
    • Determines if artifacts come from Cellpose or the merge process.

Notes:
    • This script examines the raw Cellpose output before any merge processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature
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

def load_original_tiles(results_dir):
    """Load original tile masks before any merging."""
    tile_masks_dir = results_dir / "masks" / "tile_masks_npz"
    
    if not tile_masks_dir.exists():
        raise FileNotFoundError(f"Original tile masks directory not found: {tile_masks_dir}")
    
    print(f"Loading original tiles from: {tile_masks_dir}")
    
    # Find all .npz files.
    npz_files = list(tile_masks_dir.glob("*.npz"))
    npz_files.sort()
    
    print(f"Found {len(npz_files)} original tile files")
    
    tiles = {}
    for npz_file in npz_files:
        data = np.load(npz_file)
        tile_id = npz_file.stem
        mask = data['mask']
        tiles[tile_id] = mask
        
        nuclei_count = len(np.unique(mask[mask > 0]))
        print(f"Loaded {tile_id}: shape {mask.shape}, nuclei: {nuclei_count}")
    
    return tiles

def reconstruct_from_original_tiles(tiles):
    """Reconstruct the image from original tiles with simple overwriting."""
    print("\n=== Reconstructing from Original Tiles ===")
    
    # Parse tile coordinates and determine grid.
    # The tile names are like "0_0", "0_410", "410_0", etc.
    # These represent (row_start, col_start) positions, not grid indices.
    tile_coords = {}
    positions = []

    for tile_id, mask in tiles.items():
        parts = tile_id.split('_')
        if len(parts) == 2:
            row_pos, col_pos = int(parts[0]), int(parts[1])
            positions.append((row_pos, col_pos))
            tile_coords[(row_pos, col_pos)] = mask

    # Determine unique row and column positions.
    unique_rows = sorted(set(pos[0] for pos in positions))
    unique_cols = sorted(set(pos[1] for pos in positions))

    print(f"Unique row positions: {unique_rows}")
    print(f"Unique col positions: {unique_cols}")

    max_row_pos = max(unique_rows)
    max_col_pos = max(unique_cols)
    
    print(f"Grid positions: rows {unique_rows}, cols {unique_cols}")

    # Determine the final image size based on the maximum positions and tile sizes.
    # Find the largest tile at each position to determine final dimensions.
    max_height = 0
    max_width = 0

    for (row_pos, col_pos), mask in tile_coords.items():
        final_row = row_pos + mask.shape[0]
        final_col = col_pos + mask.shape[1]
        max_height = max(max_height, final_row)
        max_width = max(max_width, final_col)

    full_height = max_height
    full_width = max_width
    
    print(f"Reconstructed image size: {full_height} x {full_width}")
    
    reconstructed = np.zeros((full_height, full_width), dtype=np.uint32)
    
    # Place tiles with simple overwriting (later tiles overwrite earlier ones).
    for (row_pos, col_pos), mask in tile_coords.items():
        y_start = row_pos
        x_start = col_pos
        y_end = y_start + mask.shape[0]
        x_end = x_start + mask.shape[1]

        # Handle boundary cases.
        y_end = min(y_end, full_height)
        x_end = min(x_end, full_width)

        mask_h = y_end - y_start
        mask_w = x_end - x_start

        # Simple overwrite - this will create straight-line artifacts!
        reconstructed[y_start:y_end, x_start:x_end] = mask[:mask_h, :mask_w]

        print(f"Placed tile at ({row_pos},{col_pos}) -> [{y_start}:{y_end}, {x_start}:{x_end}]")
    
    total_nuclei = len(np.unique(reconstructed[reconstructed > 0]))
    print(f"Total nuclei in reconstructed image: {total_nuclei}")
    
    return reconstructed

def analyze_tile_boundaries(reconstructed):
    """Analyze edge density at tile boundaries in the reconstructed image."""
    print("\n=== Analyzing Tile Boundaries in Reconstructed Image ===")
    
    # Detect edges.
    edges = feature.canny(reconstructed.astype(float), sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
    # Expected tile boundaries.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    h, w = reconstructed.shape
    
    # Expected boundaries.
    vertical_boundaries = []
    x = step
    while x < w:
        vertical_boundaries.append(x)
        x += step
    
    horizontal_boundaries = []
    y = step
    while y < h:
        horizontal_boundaries.append(y)
        y += step
    
    print(f"Expected vertical boundaries: {vertical_boundaries}")
    print(f"Expected horizontal boundaries: {horizontal_boundaries}")
    
    # Analyze edge density.
    boundary_width = 5
    
    print("\n=== Edge Density Analysis ===")
    
    for x in vertical_boundaries:
        if x - boundary_width >= 0 and x + boundary_width < w:
            region = edges[:, x-boundary_width:x+boundary_width+1]
            edge_density = np.sum(region) / region.size
            
            # Check unique labels in this region.
            mask_region = reconstructed[:, x-boundary_width:x+boundary_width+1]
            unique_labels = len(np.unique(mask_region[mask_region > 0]))
            
            print(f"Vertical boundary x={x}: edge density={edge_density:.4f}, unique labels={unique_labels}")
    
    for y in horizontal_boundaries:
        if y - boundary_width >= 0 and y + boundary_width < h:
            region = edges[y-boundary_width:y+boundary_width+1, :]
            edge_density = np.sum(region) / region.size
            
            # Check unique labels in this region.
            mask_region = reconstructed[y-boundary_width:y+boundary_width+1, :]
            unique_labels = len(np.unique(mask_region[mask_region > 0]))
            
            print(f"Horizontal boundary y={y}: edge density={edge_density:.4f}, unique labels={unique_labels}")
    
    return edges

def create_comparison_visualization(reconstructed, edges, output_path):
    """Create a visualization comparing original tile reconstruction with edge analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original reconstruction.
    axes[0, 0].imshow(reconstructed, cmap='nipy_spectral')
    axes[0, 0].set_title('Original Tiles Reconstruction (Simple Overwrite)')
    axes[0, 0].axis('off')
    
    # Binary mask.
    binary = reconstructed > 0
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # Edge detection.
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Detected Edges')
    axes[1, 0].axis('off')
    
    # Overlay.
    overlay = np.zeros((*reconstructed.shape, 3), dtype=np.uint8)
    overlay[reconstructed > 0] = [100, 100, 100]  # Gray for nuclei.
    overlay[edges] = [255, 0, 0]  # Red for edges.
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Edges on Original Reconstruction')
    axes[1, 1].axis('off')
    
    # Add tile boundary lines.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    h, w = reconstructed.shape
    
    # Vertical boundaries.
    x = step
    while x < w:
        for ax in axes.flat:
            ax.axvline(x, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        x += step
    
    # Horizontal boundaries.
    y = step
    while y < h:
        for ax in axes.flat:
            ax.axhline(y, color='yellow', linestyle='--', alpha=0.7, linewidth=1)
        y += step
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison visualization saved to: {output_path}")

def main():
    """Main analysis function."""
    print("Original Tiles Analysis")
    print("=" * 50)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load original tile masks.
        tiles = load_original_tiles(results_dir)
        
        # Reconstruct from original tiles with simple overwriting.
        reconstructed = reconstruct_from_original_tiles(tiles)
        
        # Analyze tile boundaries.
        edges = analyze_tile_boundaries(reconstructed)
        
        # Create visualization.
        output_path = "original_tiles_analysis.png"
        create_comparison_visualization(reconstructed, edges, output_path)
        
        # Final assessment.
        print(f"\n=== FINAL ASSESSMENT ===")
        print("This analysis reconstructs the image from ORIGINAL tiles using simple overwriting.")
        print("If artifacts are present here, they come from:")
        print("1. The original Cellpose segmentation creating straight edges")
        print("2. The simple overwriting process (expected)")
        print("3. The tiling approach itself")
        print("")
        print("If artifacts are NOT present here, then our merge algorithm is the issue.")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

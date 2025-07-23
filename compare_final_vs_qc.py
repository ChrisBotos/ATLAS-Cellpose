#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: compare_final_vs_qc.py.
Description:
    Compare the final assembled segmentation result with the QC visualization
    to determine if the straight-line artifacts are in the actual merge result
    or just in the QC visualization process.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.
    • skimage for image processing.

Usage:
    python compare_final_vs_qc.py

Key Features:
    • Loads the final segmentation_masks.npy result.
    • Loads individual merged tiles and reconstructs the image.
    • Compares the two results to identify discrepancies.
    • Analyzes where the straight-line artifacts are coming from.

Notes:
    • This script helps determine if the issue is in the merge algorithm or QC visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from skimage import feature

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

def load_final_result(results_dir):
    """Load the final assembled segmentation result."""
    final_path = results_dir / "masks" / "segmentation_masks.npy"
    
    if not final_path.exists():
        raise FileNotFoundError(f"Final result not found: {final_path}")
    
    print(f"Loading final result from: {final_path}")
    final_mask = np.load(final_path)
    print(f"Final result shape: {final_mask.shape}")
    print(f"Final result nuclei count: {len(np.unique(final_mask[final_mask > 0]))}")
    
    return final_mask

def reconstruct_from_merged_tiles(results_dir):
    """Reconstruct the image from individual merged tiles."""
    merged_tiles_dir = results_dir / "masks" / "merged_tile_masks_npz"
    
    if not merged_tiles_dir.exists():
        raise FileNotFoundError(f"Merged tiles directory not found: {merged_tiles_dir}")
    
    print(f"Loading merged tiles from: {merged_tiles_dir}")
    
    # Find all tile files.
    tile_files = list(merged_tiles_dir.glob("*.npz"))
    tile_files.sort()
    
    print(f"Found {len(tile_files)} merged tile files")
    
    # Load tiles and determine grid dimensions.
    tiles = {}
    max_row, max_col = 0, 0
    
    for tile_file in tile_files:
        # Parse tile coordinates from filename.
        parts = tile_file.stem.split('_')
        if len(parts) == 2:
            row, col = int(parts[0]), int(parts[1])
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            
            # Load tile data.
            data = np.load(tile_file)
            mask = data['mask']
            tiles[(row, col)] = mask
            
            print(f"Loaded tile ({row}, {col}): shape {mask.shape}, nuclei: {len(np.unique(mask[mask > 0]))}")
    
    print(f"Grid dimensions: {max_row + 1} x {max_col + 1}")
    
    # Reconstruct the full image.
    # Assume tiles are 512x512 with 102 pixel overlap.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    full_height = (max_row + 1) * step + overlap
    full_width = (max_col + 1) * step + overlap
    
    print(f"Reconstructed image size: {full_height} x {full_width}")
    
    reconstructed = np.zeros((full_height, full_width), dtype=np.uint32)
    
    # Place tiles in the reconstructed image.
    for (row, col), mask in tiles.items():
        y_start = row * step
        x_start = col * step
        y_end = y_start + mask.shape[0]
        x_end = x_start + mask.shape[1]
        
        # Handle boundary cases.
        y_end = min(y_end, full_height)
        x_end = min(x_end, full_width)
        
        mask_h = y_end - y_start
        mask_w = x_end - x_start
        
        # Place the tile (overlapping regions will overwrite).
        reconstructed[y_start:y_end, x_start:x_end] = mask[:mask_h, :mask_w]
    
    print(f"Reconstructed nuclei count: {len(np.unique(reconstructed[reconstructed > 0]))}")
    
    return reconstructed

def compare_results(final_mask, reconstructed_mask):
    """Compare the final result with the reconstructed result."""
    print("\n=== Comparing Results ===")
    
    # Check shapes.
    print(f"Final mask shape: {final_mask.shape}")
    print(f"Reconstructed mask shape: {reconstructed_mask.shape}")
    
    if final_mask.shape != reconstructed_mask.shape:
        print("❌ SHAPE MISMATCH: Results have different shapes!")
        return False
    
    # Check if they are identical.
    are_identical = np.array_equal(final_mask, reconstructed_mask)
    print(f"Results are identical: {are_identical}")
    
    if are_identical:
        print("✅ IDENTICAL: Final result and reconstructed result are the same")
        print("✅ This means the straight-line artifacts are in the ACTUAL merge result, not the QC visualization")
        return True
    else:
        print("❌ DIFFERENT: Final result and reconstructed result differ")
        
        # Analyze differences.
        diff_mask = (final_mask != reconstructed_mask)
        diff_pixels = np.sum(diff_mask)
        total_pixels = final_mask.size
        diff_percentage = diff_pixels / total_pixels * 100
        
        print(f"Different pixels: {diff_pixels} / {total_pixels} ({diff_percentage:.2f}%)")
        
        # Check nuclei counts.
        final_nuclei = len(np.unique(final_mask[final_mask > 0]))
        reconstructed_nuclei = len(np.unique(reconstructed_mask[reconstructed_mask > 0]))
        
        print(f"Final nuclei count: {final_nuclei}")
        print(f"Reconstructed nuclei count: {reconstructed_nuclei}")
        
        return False

def analyze_edge_artifacts(mask, title="Mask"):
    """Analyze edge artifacts in a mask."""
    print(f"\n=== Analyzing Edge Artifacts in {title} ===")
    
    # Detect edges.
    edges = feature.canny(mask.astype(float), sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
    # Check for straight lines at expected tile boundaries.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    h, w = mask.shape
    
    # Expected vertical boundaries.
    vertical_boundaries = []
    x = step
    while x < w:
        vertical_boundaries.append(x)
        x += step
    
    # Expected horizontal boundaries.
    horizontal_boundaries = []
    y = step
    while y < h:
        horizontal_boundaries.append(y)
        y += step
    
    print(f"Expected vertical boundaries: {vertical_boundaries}")
    print(f"Expected horizontal boundaries: {horizontal_boundaries}")
    
    # Analyze edge density at boundaries.
    boundary_width = 5
    
    for x in vertical_boundaries:
        if x - boundary_width >= 0 and x + boundary_width < w:
            region = edges[:, x-boundary_width:x+boundary_width+1]
            edge_density = np.sum(region) / region.size
            print(f"Vertical boundary at x={x}: edge density = {edge_density:.4f}")
    
    for y in horizontal_boundaries:
        if y - boundary_width >= 0 and y + boundary_width < h:
            region = edges[y-boundary_width:y+boundary_width+1, :]
            edge_density = np.sum(region) / region.size
            print(f"Horizontal boundary at y={y}: edge density = {edge_density:.4f}")

def main():
    """Main comparison function."""
    print("Final Result vs QC Visualization Comparison")
    print("=" * 60)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load final result.
        final_mask = load_final_result(results_dir)
        
        # Reconstruct from merged tiles.
        reconstructed_mask = reconstruct_from_merged_tiles(results_dir)
        
        # Compare results.
        are_identical = compare_results(final_mask, reconstructed_mask)
        
        # Analyze edge artifacts in both.
        analyze_edge_artifacts(final_mask, "Final Result")
        analyze_edge_artifacts(reconstructed_mask, "Reconstructed Result")
        
        # Conclusion.
        print(f"\n=== CONCLUSION ===")
        if are_identical:
            print("🔍 The straight-line artifacts are in the ACTUAL merge result.")
            print("🔍 The issue is in the merge algorithm, not the QC visualization.")
            print("🔍 Need to investigate why the 3-step algorithm is not eliminating the artifacts.")
        else:
            print("🔍 The final result and reconstructed result differ.")
            print("🔍 The artifacts might be introduced during tile reconstruction for QC.")
            print("🔍 Need to investigate the tile saving/loading process.")
        
        return are_identical
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

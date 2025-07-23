#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: examine_raw_segmentation.py.
Description:
    Examine the raw segmentation result directly to determine if straight-line
    artifacts are in the actual segmentation or introduced during visualization.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.
    • skimage for image processing.

Usage:
    python examine_raw_segmentation.py

Key Features:
    • Loads the raw segmentation_masks.npy file.
    • Analyzes edge patterns directly in the segmentation mask.
    • Creates a simple visualization without complex overlay processing.
    • Determines if artifacts are in the segmentation or visualization.

Notes:
    • This script bypasses all QC visualization processing to examine raw data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature, filters
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

def analyze_raw_segmentation(mask):
    """Analyze the raw segmentation mask for straight-line artifacts."""
    print("=== Raw Segmentation Analysis ===")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Unique nuclei: {len(np.unique(mask[mask > 0]))}")
    print(f"Max label: {mask.max()}")
    
    # Convert to binary edge map.
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect edges using different methods.
    print("\n=== Edge Detection Analysis ===")
    
    # Method 1: Simple gradient.
    grad_y, grad_x = np.gradient(mask.astype(float))
    gradient_edges = np.sqrt(grad_x**2 + grad_y**2) > 0.1
    
    # Method 2: Canny edge detection.
    canny_edges = feature.canny(mask.astype(float), sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
    # Method 3: Binary boundary detection.
    from scipy import ndimage
    # Use morphological gradient instead.
    binary_edges = ndimage.binary_erosion(binary_mask) ^ binary_mask
    
    print(f"Gradient edges: {np.sum(gradient_edges)} pixels")
    print(f"Canny edges: {np.sum(canny_edges)} pixels")
    print(f"Binary edges: {np.sum(binary_edges)} pixels")
    
    return gradient_edges, canny_edges, binary_edges

def check_tile_boundary_artifacts(mask, edges):
    """Check for artifacts specifically at tile boundaries."""
    print("\n=== Tile Boundary Artifact Analysis ===")
    
    # Expected tile boundaries.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    h, w = mask.shape
    
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
    
    # Analyze edge density at boundaries.
    boundary_width = 5
    
    print("\n=== Edge Density at Tile Boundaries ===")
    
    for x in vertical_boundaries:
        if x - boundary_width >= 0 and x + boundary_width < w:
            region = edges[:, x-boundary_width:x+boundary_width+1]
            edge_density = np.sum(region) / region.size
            
            # Also check the raw mask values in this region.
            mask_region = mask[:, x-boundary_width:x+boundary_width+1]
            unique_labels = len(np.unique(mask_region[mask_region > 0]))
            
            print(f"Vertical boundary x={x}: edge density={edge_density:.4f}, unique labels={unique_labels}")
    
    for y in horizontal_boundaries:
        if y - boundary_width >= 0 and y + boundary_width < h:
            region = edges[y-boundary_width:y+boundary_width+1, :]
            edge_density = np.sum(region) / region.size
            
            # Also check the raw mask values in this region.
            mask_region = mask[y-boundary_width:y+boundary_width+1, :]
            unique_labels = len(np.unique(mask_region[mask_region > 0]))
            
            print(f"Horizontal boundary y={y}: edge density={edge_density:.4f}, unique labels={unique_labels}")

def create_simple_visualization(mask, edges, output_path):
    """Create a simple visualization of the raw segmentation and edges."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Raw segmentation mask.
    axes[0, 0].imshow(mask, cmap='nipy_spectral')
    axes[0, 0].set_title('Raw Segmentation Mask')
    axes[0, 0].axis('off')
    
    # Binary mask.
    binary = mask > 0
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # Edge detection result.
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Detected Edges')
    axes[1, 0].axis('off')
    
    # Overlay edges on mask.
    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    overlay[mask > 0] = [100, 100, 100]  # Gray for nuclei.
    overlay[edges] = [255, 0, 0]  # Red for edges.
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Edges Overlay on Mask')
    axes[1, 1].axis('off')
    
    # Add tile boundary lines.
    tile_size = 512
    overlap = 102
    step = tile_size - overlap
    
    h, w = mask.shape
    
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
    
    print(f"Visualization saved to: {output_path}")

def main():
    """Main analysis function."""
    print("Raw Segmentation Analysis")
    print("=" * 50)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load raw segmentation mask.
        mask_path = results_dir / "masks" / "segmentation_masks.npy"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")
        
        print(f"Loading raw segmentation from: {mask_path}")
        mask = np.load(mask_path)
        
        # Analyze the raw segmentation.
        gradient_edges, canny_edges, binary_edges = analyze_raw_segmentation(mask)
        
        # Check for tile boundary artifacts using Canny edges (most sensitive).
        check_tile_boundary_artifacts(mask, canny_edges)
        
        # Create visualization.
        output_path = "raw_segmentation_analysis.png"
        create_simple_visualization(mask, canny_edges, output_path)
        
        # Final assessment.
        print(f"\n=== FINAL ASSESSMENT ===")
        print("This analysis examines the RAW segmentation mask directly,")
        print("bypassing all QC visualization processing.")
        print("If artifacts are still present, they are in the actual segmentation,")
        print("not introduced during visualization.")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

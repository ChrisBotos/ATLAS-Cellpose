#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: analyze_qc_artifacts.py.
Description:
    Analyze QC visualizations to detect straight-line artifacts along tile boundaries.
    This script examines the after_merging.tif file to identify nuclei with unnaturally
    straight edges that correspond to tile boundaries, indicating Step 2 border deletion bugs.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • matplotlib for visualization.
    • skimage for image processing.
    • scipy for edge detection.

Usage:
    python analyze_qc_artifacts.py

Key Features:
    • Loads QC visualization files.
    • Detects straight-line artifacts along tile boundaries.
    • Quantifies the severity of artifacts.
    • Provides visual analysis of problem areas.

Notes:
    • This script helps identify remaining Step 2 border deletion issues.
    • Results indicate whether the fix is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.io as skio
from skimage import filters, feature, morphology
from scipy import ndimage
import sys
import os

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
    """Load the QC visualization images."""
    qc_dir = results_dir / "merge_qc_overlays"
    
    before_path = qc_dir / "before_merging.tif"
    after_path = qc_dir / "after_merging.tif"
    
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(f"QC images not found in {qc_dir}")
    
    print(f"Loading QC images from: {qc_dir}")
    
    before_img = skio.imread(before_path)
    after_img = skio.imread(after_path)
    
    print(f"Before image shape: {before_img.shape}")
    print(f"After image shape: {after_img.shape}")
    
    return before_img, after_img

def detect_straight_line_artifacts(image, tile_size=512, overlap=102):
    """
    Detect straight-line artifacts that correspond to tile boundaries.
    
    Parameters
    ----------
    image : np.ndarray
        QC visualization image.
    tile_size : int
        Size of tiles used in segmentation.
    overlap : int
        Overlap between tiles.
        
    Returns
    -------
    artifacts : list
        List of detected artifact regions.
    """
    print("Analyzing image for straight-line artifacts...")
    
    # Convert to grayscale if needed.
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()
    
    # Calculate expected tile boundary positions.
    h, w = gray.shape[:2]
    step = tile_size - overlap
    
    vertical_boundaries = []
    horizontal_boundaries = []
    
    # Vertical boundaries.
    x = step
    while x < w:
        vertical_boundaries.append(x)
        x += step
    
    # Horizontal boundaries.
    y = step
    while y < h:
        horizontal_boundaries.append(y)
        y += step
    
    print(f"Expected vertical boundaries: {vertical_boundaries}")
    print(f"Expected horizontal boundaries: {horizontal_boundaries}")
    
    # Detect edges in the image.
    edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
    # Analyze regions around expected tile boundaries.
    artifacts = []
    boundary_width = 5  # Pixels around boundary to check.
    
    # Check vertical boundaries.
    for x in vertical_boundaries:
        if x - boundary_width >= 0 and x + boundary_width < w:
            region = edges[:, x-boundary_width:x+boundary_width+1]
            
            # Count vertical edge pixels in this region.
            vertical_edges = np.sum(region)
            
            if vertical_edges > h * 0.1:  # More than 10% of height has edges.
                artifacts.append({
                    'type': 'vertical',
                    'position': x,
                    'severity': vertical_edges / h,
                    'region': region
                })
                print(f"Vertical artifact detected at x={x}, severity={vertical_edges/h:.3f}")
    
    # Check horizontal boundaries.
    for y in horizontal_boundaries:
        if y - boundary_width >= 0 and y + boundary_width < h:
            region = edges[y-boundary_width:y+boundary_width+1, :]
            
            # Count horizontal edge pixels in this region.
            horizontal_edges = np.sum(region)
            
            if horizontal_edges > w * 0.1:  # More than 10% of width has edges.
                artifacts.append({
                    'type': 'horizontal',
                    'position': y,
                    'severity': horizontal_edges / w,
                    'region': region
                })
                print(f"Horizontal artifact detected at y={y}, severity={horizontal_edges/w:.3f}")
    
    return artifacts, edges

def visualize_artifacts(image, artifacts, edges, output_path):
    """Visualize detected artifacts."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image.
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original QC Image')
    axes[0, 0].axis('off')
    
    # Edge detection result.
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Detected Edges')
    axes[0, 1].axis('off')
    
    # Artifacts overlay.
    overlay = image.copy()
    if len(overlay.shape) == 3:
        overlay = overlay.astype(float)
    
    for artifact in artifacts:
        if artifact['type'] == 'vertical':
            x = artifact['position']
            if len(overlay.shape) == 3:
                overlay[:, x-2:x+3, 0] = 255  # Red line.
            else:
                overlay[:, x-2:x+3] = 255
        else:  # horizontal
            y = artifact['position']
            if len(overlay.shape) == 3:
                overlay[y-2:y+3, :, 0] = 255  # Red line.
            else:
                overlay[y-2:y+3, :] = 255
    
    axes[1, 0].imshow(overlay.astype(np.uint8))
    axes[1, 0].set_title(f'Artifacts Highlighted ({len(artifacts)} found)')
    axes[1, 0].axis('off')
    
    # Severity plot.
    if artifacts:
        severities = [a['severity'] for a in artifacts]
        positions = [a['position'] for a in artifacts]
        types = [a['type'] for a in artifacts]
        
        colors = ['red' if t == 'vertical' else 'blue' for t in types]
        axes[1, 1].scatter(positions, severities, c=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Position (pixels)')
        axes[1, 1].set_ylabel('Artifact Severity')
        axes[1, 1].set_title('Artifact Severity by Position')
        axes[1, 1].legend(['Vertical', 'Horizontal'])
    else:
        axes[1, 1].text(0.5, 0.5, 'No artifacts detected!', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=16, color='green', weight='bold')
        axes[1, 1].set_title('Artifact Analysis')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("QC Artifact Analysis for Step 2 Border Deletion Fix")
    print("=" * 60)
    
    try:
        # Find latest results directory.
        results_dir = find_latest_results_dir()
        print(f"Analyzing results from: {results_dir}")
        
        # Load QC images.
        before_img, after_img = load_qc_images(results_dir)
        
        # Analyze the after_merging image for artifacts.
        print("\nAnalyzing after_merging.tif for straight-line artifacts...")
        artifacts, edges = detect_straight_line_artifacts(after_img)
        
        # Generate visualization.
        output_path = "qc_artifact_analysis.png"
        visualize_artifacts(after_img, artifacts, edges, output_path)
        
        # Summary.
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total artifacts detected: {len(artifacts)}")
        
        if len(artifacts) == 0:
            print("✅ SUCCESS: No straight-line artifacts detected!")
            print("✅ Step 2 border deletion fix appears to be working correctly.")
        else:
            print("❌ ISSUE: Straight-line artifacts still present.")
            print("❌ Step 2 border deletion fix needs further refinement.")
            
            for i, artifact in enumerate(artifacts):
                print(f"  Artifact {i+1}: {artifact['type']} at position {artifact['position']}, "
                      f"severity {artifact['severity']:.3f}")
        
        print(f"\nVisualization saved to: {output_path}")
        
        return len(artifacts) == 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

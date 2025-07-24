"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: analyze_cross_boundary_nuclei.py.
Description:
    Diagnostic script to analyze cross-boundary nuclei handling and identify
    the source of 1-pixel separation lines in the merged nuclei segmentation.
    
    This script investigates the Phase 3 assembly conflict resolution logic
    to determine if legitimate cross-boundary nuclei are being fragmented.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • pathlib for file handling.

Usage:
    python analyze_cross_boundary_nuclei.py

Key Features:
    • Analyzes individual tile masks vs merged tile masks.
    • Identifies nuclei that appear in multiple tiles.
    • Detects fragmentation caused by conflict resolution.
    • Maps 1-pixel gaps to their potential causes.

Notes:
    • Focuses on the specific results directory provided by the user.
    • Examines both tile_masks_npz and merged_tile_masks_npz directories.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_mask_from_npz(file_path: Path) -> np.ndarray:
    """Load mask from .npz file."""
    try:
        data = np.load(file_path)
        return data["mask"].astype(np.uint32)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return np.array([])

def analyze_tile_overlap_regions(
    tile1_path: Path, 
    tile2_path: Path, 
    relationship: str,
    overlap: int = 64
) -> Dict:
    """
    Analyze overlap regions between two adjacent tiles.
    
    Parameters
    ----------
    tile1_path, tile2_path : Path
        Paths to the tile mask files.
    relationship : str
        Spatial relationship ('horizontal' or 'vertical').
    overlap : int
        Overlap size in pixels.
        
    Returns
    -------
    Dict
        Analysis results including shared nuclei and gaps.
    """
    tile1_mask = load_mask_from_npz(tile1_path)
    tile2_mask = load_mask_from_npz(tile2_path)
    
    if tile1_mask.size == 0 or tile2_mask.size == 0:
        return {"error": "Failed to load tiles"}
    
    h, w = tile1_mask.shape
    
    # Define overlap regions based on relationship.
    if relationship == "horizontal":  # tile1 left of tile2
        # Overlap is the rightmost part of tile1 and leftmost part of tile2.
        tile1_overlap = tile1_mask[:, -overlap:]
        tile2_overlap = tile2_mask[:, :overlap]
    elif relationship == "vertical":  # tile1 above tile2
        # Overlap is the bottommost part of tile1 and topmost part of tile2.
        tile1_overlap = tile1_mask[-overlap:, :]
        tile2_overlap = tile2_mask[:overlap, :]
    else:
        return {"error": f"Unknown relationship: {relationship}"}
    
    # Find nuclei in overlap regions.
    tile1_nuclei = set(np.unique(tile1_overlap[tile1_overlap > 0]))
    tile2_nuclei = set(np.unique(tile2_overlap[tile2_overlap > 0]))
    
    # Identify potential cross-boundary nuclei.
    # These would be nuclei that appear in both overlap regions.
    shared_nuclei = tile1_nuclei.intersection(tile2_nuclei)
    
    # Check for gaps (zero pixels between nuclei).
    combined_overlap = np.maximum(tile1_overlap, tile2_overlap)
    gap_pixels = np.sum(combined_overlap == 0)
    total_overlap_pixels = combined_overlap.size
    
    return {
        "tile1_nuclei": tile1_nuclei,
        "tile2_nuclei": tile2_nuclei,
        "shared_nuclei": shared_nuclei,
        "gap_pixels": gap_pixels,
        "total_overlap_pixels": total_overlap_pixels,
        "gap_fraction": gap_pixels / total_overlap_pixels if total_overlap_pixels > 0 else 0,
        "tile1_overlap_shape": tile1_overlap.shape,
        "tile2_overlap_shape": tile2_overlap.shape
    }

def compare_original_vs_merged_tiles(
    original_dir: Path,
    merged_dir: Path,
    tile_coords: List[Tuple[int, int]]
) -> Dict:
    """
    Compare original tiles vs merged tiles to identify changes.
    
    Parameters
    ----------
    original_dir : Path
        Directory containing original tile masks.
    merged_dir : Path
        Directory containing merged tile masks.
    tile_coords : List[Tuple[int, int]]
        List of tile coordinates to analyze.
        
    Returns
    -------
    Dict
        Comparison results showing nuclei changes.
    """
    results = {}
    
    for coord in tile_coords:
        y_start, x_start = coord
        filename = f"{y_start}_{x_start}.npz"
        
        original_path = original_dir / filename
        merged_path = merged_dir / filename
        
        if not original_path.exists() or not merged_path.exists():
            continue
            
        original_mask = load_mask_from_npz(original_path)
        merged_mask = load_mask_from_npz(merged_path)
        
        if original_mask.size == 0 or merged_mask.size == 0:
            continue
        
        # Count nuclei before and after merging.
        original_nuclei = set(np.unique(original_mask[original_mask > 0]))
        merged_nuclei = set(np.unique(merged_mask[merged_mask > 0]))
        
        # Identify changes.
        removed_nuclei = original_nuclei - merged_nuclei
        added_nuclei = merged_nuclei - original_nuclei
        preserved_nuclei = original_nuclei.intersection(merged_nuclei)
        
        # Check for fragmentation (same nucleus ID but different shapes).
        fragmentation_detected = False
        for nucleus_id in preserved_nuclei:
            original_pixels = np.sum(original_mask == nucleus_id)
            merged_pixels = np.sum(merged_mask == nucleus_id)
            if merged_pixels < original_pixels * 0.8:  # Significant size reduction.
                fragmentation_detected = True
                break
        
        results[coord] = {
            "original_nuclei_count": len(original_nuclei),
            "merged_nuclei_count": len(merged_nuclei),
            "removed_nuclei": removed_nuclei,
            "added_nuclei": added_nuclei,
            "preserved_nuclei": preserved_nuclei,
            "fragmentation_detected": fragmentation_detected
        }
    
    return results

def analyze_final_merged_mask_gaps(merged_mask_path: Path) -> Dict:
    """
    Analyze the final merged mask for 1-pixel gaps.
    
    Parameters
    ----------
    merged_mask_path : Path
        Path to the final merged segmentation mask.
        
    Returns
    -------
    Dict
        Analysis of gaps in the final mask.
    """
    try:
        merged_mask = np.load(merged_mask_path)
        logger.info(f"Loaded final merged mask: {merged_mask.shape}")
    except Exception as e:
        logger.error(f"Failed to load final mask: {e}")
        return {"error": str(e)}
    
    # Detect 1-pixel gaps between nuclei.
    gaps_detected = []
    h, w = merged_mask.shape
    
    # Check for horizontal gaps (1-pixel wide vertical lines).
    for x in range(1, w-1):
        for y in range(h):
            if merged_mask[y, x] == 0:  # Current pixel is empty.
                left_pixel = merged_mask[y, x-1]
                right_pixel = merged_mask[y, x+1]
                
                # Check if this is a gap between two different nuclei.
                if left_pixel > 0 and right_pixel > 0 and left_pixel != right_pixel:
                    gaps_detected.append({
                        "type": "horizontal_gap",
                        "position": (y, x),
                        "left_nucleus": left_pixel,
                        "right_nucleus": right_pixel
                    })
    
    # Check for vertical gaps (1-pixel wide horizontal lines).
    for y in range(1, h-1):
        for x in range(w):
            if merged_mask[y, x] == 0:  # Current pixel is empty.
                top_pixel = merged_mask[y-1, x]
                bottom_pixel = merged_mask[y+1, x]
                
                # Check if this is a gap between two different nuclei.
                if top_pixel > 0 and bottom_pixel > 0 and top_pixel != bottom_pixel:
                    gaps_detected.append({
                        "type": "vertical_gap",
                        "position": (y, x),
                        "top_nucleus": top_pixel,
                        "bottom_nucleus": bottom_pixel
                    })
    
    return {
        "total_gaps": len(gaps_detected),
        "gaps": gaps_detected[:50],  # Limit to first 50 for analysis.
        "mask_shape": merged_mask.shape,
        "total_nuclei": len(np.unique(merged_mask[merged_mask > 0]))
    }

def main():
    """Main analysis function."""
    # Define paths based on the user's results directory.
    results_dir = Path("C:/Projects/I-R-Injury-Spatial-Multiomics-Analysis/results/20250724_014707_cropped_cpu_cellpose4_diameter0")
    masks_dir = results_dir / "masks"
    
    original_tiles_dir = masks_dir / "tile_masks_npz"
    merged_tiles_dir = masks_dir / "merged_tile_masks_npz"
    final_mask_path = masks_dir / "segmentation_masks.npy"
    
    logger.info("Starting cross-boundary nuclei analysis...")
    logger.info(f"Results directory: {results_dir}")
    
    # Check if directories exist.
    if not original_tiles_dir.exists():
        logger.error(f"Original tiles directory not found: {original_tiles_dir}")
        return
    
    if not merged_tiles_dir.exists():
        logger.error(f"Merged tiles directory not found: {merged_tiles_dir}")
        return
    
    # Get list of tile coordinates from filenames.
    tile_coords = []
    for file_path in original_tiles_dir.glob("*.npz"):
        try:
            filename = file_path.stem
            parts = filename.split("_")
            if len(parts) == 2:
                y_start, x_start = int(parts[0]), int(parts[1])
                tile_coords.append((y_start, x_start))
        except ValueError:
            continue
    
    logger.info(f"Found {len(tile_coords)} tiles to analyze")
    
    # 1. Compare original vs merged tiles.
    logger.info("Comparing original vs merged tiles...")
    comparison_results = compare_original_vs_merged_tiles(
        original_tiles_dir, merged_tiles_dir, tile_coords
    )
    
    # Summarize comparison results.
    total_fragmentation = sum(1 for r in comparison_results.values() if r["fragmentation_detected"])
    total_nuclei_removed = sum(len(r["removed_nuclei"]) for r in comparison_results.values())
    
    logger.info(f"Tiles with fragmentation detected: {total_fragmentation}/{len(comparison_results)}")
    logger.info(f"Total nuclei removed during merging: {total_nuclei_removed}")
    
    # 2. Analyze final merged mask for gaps.
    if final_mask_path.exists():
        logger.info("Analyzing final merged mask for 1-pixel gaps...")
        gap_analysis = analyze_final_merged_mask_gaps(final_mask_path)
        
        if "error" not in gap_analysis:
            logger.info(f"Total 1-pixel gaps detected: {gap_analysis['total_gaps']}")
            logger.info(f"Final mask shape: {gap_analysis['mask_shape']}")
            logger.info(f"Total nuclei in final mask: {gap_analysis['total_nuclei']}")
            
            # Show examples of gaps.
            if gap_analysis["gaps"]:
                logger.info("Examples of detected gaps:")
                for i, gap in enumerate(gap_analysis["gaps"][:5]):
                    logger.info(f"  Gap {i+1}: {gap['type']} at {gap['position']}")
        else:
            logger.error(f"Gap analysis failed: {gap_analysis['error']}")
    else:
        logger.warning(f"Final merged mask not found: {final_mask_path}")
    
    # 3. Analyze specific tile pairs for overlap issues.
    logger.info("Analyzing tile overlap regions...")
    sample_coords = sorted(tile_coords)[:4]  # Analyze first 4 tiles.
    
    for i, coord1 in enumerate(sample_coords):
        for coord2 in sample_coords[i+1:]:
            y1, x1 = coord1
            y2, x2 = coord2
            
            # Determine relationship.
            if y1 == y2 and abs(x1 - x2) == 410:  # Horizontal neighbors (stride = 410).
                relationship = "horizontal"
            elif x1 == x2 and abs(y1 - y2) == 410:  # Vertical neighbors.
                relationship = "vertical"
            else:
                continue  # Not adjacent.
            
            tile1_path = merged_tiles_dir / f"{y1}_{x1}.npz"
            tile2_path = merged_tiles_dir / f"{y2}_{x2}.npz"
            
            if tile1_path.exists() and tile2_path.exists():
                overlap_analysis = analyze_tile_overlap_regions(
                    tile1_path, tile2_path, relationship
                )
                
                if "error" not in overlap_analysis:
                    logger.info(f"Overlap analysis {coord1}-{coord2} ({relationship}):")
                    logger.info(f"  Shared nuclei: {len(overlap_analysis['shared_nuclei'])}")
                    logger.info(f"  Gap fraction: {overlap_analysis['gap_fraction']:.3f}")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()

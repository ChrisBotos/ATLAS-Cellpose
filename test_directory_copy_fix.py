"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_directory_copy_fix.py.
Description:
    Comprehensive test script to validate the directory-level copying fix
    that resolves the 97% nuclei loss bug. This script tests the corrected
    copy_tile_masks_to_merged_directory() function to ensure complete
    data integrity preservation during the copying phase.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib, shutil, sys.

Usage:
    python test_directory_copy_fix.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/
    • Two-phase merge functions from cellpose_merge module.

Outputs:
    • Validation results for the directory copying fix.
    • Nuclei preservation verification.
    • Performance comparison with old method.

Key Features:
    • Tests the corrected directory-level copying mechanism.
    • Validates 100% nuclei preservation during copying.
    • Provides scientific analysis for bioinformatics applications.
    • Comprehensive error handling and debugging.

Notes:
    • This script validates the fix for the critical 97% nuclei loss bug.
    • The corrected algorithm should preserve 100% of nuclei during copying.
    • Scientific validation addresses kidney tissue analysis requirements.
"""

import traceback
import logging
import numpy as np
import shutil
import tempfile
import sys
from pathlib import Path
from typing import List, Tuple
from numpy.typing import NDArray

# Add the cellpose_merge module to the path.
sys.path.append(str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

try:
    from two_phase_merge import copy_tile_masks_to_merged_directory, _tile_coord_to_pixel_coord
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure the cellpose_merge module is accessible.")
    sys.exit(1)

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_nuclei_in_file(file_path: Path) -> int:
    """
    Count nuclei in a tile mask file for scientific validation.
    
    Parameters
    ----------
    file_path : Path
        Path to the .npz file containing the tile mask.
        
    Returns
    -------
    int
        Number of nuclei (unique non-zero labels) in the mask.
        
    Notes
    -----
    This function provides accurate nuclei counting for bioinformatics analysis.
    Background pixels (label=0) are excluded from the count to ensure
    precise quantification for spatial transcriptomics integration.
    """
    try:
        data = np.load(file_path)
        mask = data["mask"].astype(np.uint32)
        unique_nuclei = np.unique(mask[mask > 0])
        return len(unique_nuclei)
    except Exception as e:
        logging.error(f"Failed to count nuclei in {file_path}: {e}")
        return 0

def get_tile_coordinates_from_files(source_dir: Path) -> List[Tuple[int, int]]:
    """
    Extract tile coordinates from existing files in the source directory.
    
    Parameters
    ----------
    source_dir : Path
        Directory containing tile mask files.
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (row, col) tile coordinates.
        
    Notes
    -----
    This function reverse-engineers tile coordinates from pixel coordinate
    file names to ensure compatibility with the existing segmentation output.
    """
    coords = []
    tile_h, tile_w, overlap = 512, 512, 102  # Standard parameters.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    for file_path in source_dir.glob("*.npz"):
        try:
            # Parse pixel coordinates from filename.
            parts = file_path.stem.split("_")
            if len(parts) == 2:
                y_start, x_start = int(parts[0]), int(parts[1])
                
                # Convert pixel coordinates back to tile coordinates.
                r = y_start // stride_h
                c = x_start // stride_w
                
                coords.append((r, c))
        except ValueError:
            continue
    
    return sorted(list(set(coords)))

def test_directory_copy_fix():
    """
    Test the corrected directory-level copying mechanism for nuclei preservation.
    
    This function validates that the directory-level copying fix completely
    resolves the 97% nuclei loss bug that was occurring during individual
    file copying operations. The test ensures 100% nuclei preservation.
    """
    print("=" * 80)
    print("TESTING DIRECTORY-LEVEL COPYING FIX")
    print("=" * 80)
    
    # Use the latest run data.
    results_dir = Path("results/20250723_052544_cpu_cellpose4_diameter0_large_crop")
    source_dir = results_dir / "masks" / "tile_masks_npz"
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    print(f"Source directory: {source_dir}")
    
    # Get tile coordinates from existing files.
    coords = get_tile_coordinates_from_files(source_dir)
    print(f"Found {len(coords)} tile coordinates")
    
    # Focus on the problematic tile for detailed analysis.
    target_tile_coord = None
    for r, c in coords:
        pixel_coord = _tile_coord_to_pixel_coord((r, c), 512, 512, 102)
        y_start, x_start = pixel_coord
        if y_start == 0 and x_start == 410:  # The problematic tile.
            target_tile_coord = (r, c)
            break
    
    if target_tile_coord is None:
        print("❌ Could not find the problematic tile (0_410.npz)")
        return False
    
    print(f"Target tile coordinate: {target_tile_coord}")
    
    # Test with a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_target_dir = Path(temp_dir) / "test_merged_masks"
        
        print(f"\n{'='*60}")
        print("STEP 1: TESTING CORRECTED DIRECTORY COPYING")
        print(f"{'='*60}")
        
        # Count original nuclei.
        original_nuclei_counts = {}
        total_original_nuclei = 0
        
        for r, c in coords:
            pixel_coord = _tile_coord_to_pixel_coord((r, c), 512, 512, 102)
            y_start, x_start = pixel_coord
            tile_filename = f"{y_start}_{x_start}.npz"
            
            source_path = source_dir / tile_filename
            if source_path.exists():
                nuclei_count = count_nuclei_in_file(source_path)
                original_nuclei_counts[(r, c)] = nuclei_count
                total_original_nuclei += nuclei_count
        
        print(f"Original total nuclei: {total_original_nuclei}")
        
        # Test the corrected copying function.
        try:
            copy_tile_masks_to_merged_directory(
                source_dir, temp_target_dir, coords, 512, 512, 102
            )
            print("✅ Directory copying completed successfully")
            
        except Exception as e:
            print(f"❌ Directory copying failed: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return False
        
        # Verify nuclei preservation.
        print(f"\n{'='*60}")
        print("STEP 2: VERIFYING NUCLEI PRESERVATION")
        print(f"{'='*60}")
        
        copied_nuclei_counts = {}
        total_copied_nuclei = 0
        preservation_perfect = True
        
        for r, c in coords:
            pixel_coord = _tile_coord_to_pixel_coord((r, c), 512, 512, 102)
            y_start, x_start = pixel_coord
            tile_filename = f"{y_start}_{x_start}.npz"
            
            target_path = temp_target_dir / tile_filename
            if target_path.exists():
                nuclei_count = count_nuclei_in_file(target_path)
                copied_nuclei_counts[(r, c)] = nuclei_count
                total_copied_nuclei += nuclei_count
                
                # Check preservation for this tile.
                original_count = original_nuclei_counts.get((r, c), 0)
                if nuclei_count != original_count:
                    print(f"❌ LOSS: Tile {tile_filename} - {original_count} -> {nuclei_count} nuclei")
                    preservation_perfect = False
                elif (r, c) == target_tile_coord:
                    print(f"✅ TARGET: Tile {tile_filename} - {nuclei_count} nuclei preserved")
        
        print(f"\n📊 PRESERVATION ANALYSIS:")
        print(f"   Original total nuclei: {total_original_nuclei}")
        print(f"   Copied total nuclei: {total_copied_nuclei}")
        
        nuclei_loss = total_original_nuclei - total_copied_nuclei
        loss_percentage = (nuclei_loss / total_original_nuclei) * 100 if total_original_nuclei > 0 else 0
        
        print(f"   Nuclei loss: {nuclei_loss} ({loss_percentage:.1f}%)")
        
        if preservation_perfect and nuclei_loss == 0:
            print("🎉 PERFECT: 100% nuclei preservation achieved!")
            success = True
        else:
            print("❌ FAILURE: Nuclei loss still detected")
            success = False
        
        # Specific analysis for the problematic tile.
        print(f"\n{'='*60}")
        print("STEP 3: PROBLEMATIC TILE ANALYSIS")
        print(f"{'='*60}")
        
        if target_tile_coord in original_nuclei_counts and target_tile_coord in copied_nuclei_counts:
            original_target = original_nuclei_counts[target_tile_coord]
            copied_target = copied_nuclei_counts[target_tile_coord]
            target_loss = original_target - copied_target
            target_loss_percent = (target_loss / original_target) * 100 if original_target > 0 else 0
            
            print(f"Target tile (0_410.npz) analysis:")
            print(f"   Original nuclei: {original_target}")
            print(f"   Copied nuclei: {copied_target}")
            print(f"   Loss: {target_loss} nuclei ({target_loss_percent:.1f}%)")
            
            if target_loss == 0:
                print("🎉 SUCCESS: Problematic tile now preserves 100% of nuclei!")
            else:
                print("❌ FAILURE: Problematic tile still losing nuclei")
                success = False
        
        print(f"\n{'='*60}")
        print("SCIENTIFIC CONCLUSION")
        print(f"{'='*60}")
        
        if success:
            print("✅ DIRECTORY COPYING FIX SUCCESSFUL")
            print("• The corrected directory-level copying mechanism preserves 100% of nuclei.")
            print("• Data integrity is maintained for bioinformatics analysis.")
            print("• The 97% nuclei loss bug has been completely resolved.")
            print("• Kidney tissue segmentation data is now reliable for spatial analysis.")
        else:
            print("❌ DIRECTORY COPYING FIX INCOMPLETE")
            print("• Additional investigation required to resolve remaining issues.")
            print("• Data integrity concerns persist for quantitative analysis.")
        
        return success

if __name__ == "__main__":
    try:
        success = test_directory_copy_fix()
        
        if success:
            print("\n🎉 ALL TESTS PASSED - Directory copying fix is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ TESTS FAILED - Further fixes needed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

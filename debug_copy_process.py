"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_copy_process.py.
Description:
    Debug script to investigate the file copy and ID reassignment process
    that is causing massive nuclei loss. This script will trace the exact
    steps of the copy and reassignment process to identify where nuclei
    are being lost.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib.

Usage:
    python debug_copy_process.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/
    • Merged tile masks from masks/merged_tile_masks_npz/

Outputs:
    • Step-by-step analysis of the copy and reassignment process.
    • Identification of where nuclei are lost.

Key Features:
    • Traces nuclei counts through each step.
    • Identifies file naming and loading issues.
    • Validates the copy and reassignment process.

Notes:
    • This script investigates the 99% nuclei loss in tile 0_410.npz.
    • The issue appears to be in the copy or ID reassignment process.
"""

import traceback
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_copy_and_reassignment():
    """
    Debug the copy and ID reassignment process step by step.
    """
    print("=" * 80)
    print("DEBUGGING COPY AND ID REASSIGNMENT PROCESS")
    print("=" * 80)
    
    results_dir = Path("results/20250723_043122_cpu_cellpose4_diameter0_large_crop")
    source_dir = results_dir / "masks" / "tile_masks_npz"
    target_dir = results_dir / "masks" / "merged_tile_masks_npz"
    
    # Focus on the problematic tile.
    tile_coord = (0, 410)
    tile_filename = f"{tile_coord[0]}_{tile_coord[1]}.npz"
    
    print(f"Debugging tile: {tile_filename}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Step 1: Check original tile.
    print(f"\n{'='*60}")
    print("STEP 1: CHECKING ORIGINAL TILE")
    print(f"{'='*60}")
    
    original_path = source_dir / tile_filename
    if not original_path.exists():
        print(f"❌ Original tile not found: {original_path}")
        return
    
    original_data = np.load(original_path)
    original_mask = original_data["mask"].astype(np.uint32)
    original_nuclei = np.unique(original_mask[original_mask > 0])
    
    print(f"Original tile path: {original_path}")
    print(f"Original tile shape: {original_mask.shape}")
    print(f"Original nuclei count: {len(original_nuclei)}")
    print(f"Original ID range: {original_nuclei.min()} - {original_nuclei.max()}")
    print(f"Original file size: {original_path.stat().st_size:,} bytes")
    
    # Step 2: Check if copy was successful.
    print(f"\n{'='*60}")
    print("STEP 2: CHECKING COPY PROCESS")
    print(f"{'='*60}")
    
    copied_path = target_dir / tile_filename
    if not copied_path.exists():
        print(f"❌ Copied tile not found: {copied_path}")
        print("   This suggests the copy process failed!")
        return
    
    copied_data = np.load(copied_path)
    copied_mask = copied_data["mask"].astype(np.uint32)
    copied_nuclei = np.unique(copied_mask[copied_mask > 0])
    
    print(f"Copied tile path: {copied_path}")
    print(f"Copied tile shape: {copied_mask.shape}")
    print(f"Copied nuclei count: {len(copied_nuclei)}")
    print(f"Copied ID range: {copied_nuclei.min()} - {copied_nuclei.max()}")
    print(f"Copied file size: {copied_path.stat().st_size:,} bytes")
    
    # Compare original vs copied.
    if np.array_equal(original_mask, copied_mask):
        print("✅ Copy was successful - masks are identical")
    else:
        print("❌ Copy failed - masks are different!")
        diff_pixels = np.sum(original_mask != copied_mask)
        print(f"   Different pixels: {diff_pixels:,}")
        return
    
    # Step 3: Simulate ID reassignment.
    print(f"\n{'='*60}")
    print("STEP 3: SIMULATING ID REASSIGNMENT")
    print(f"{'='*60}")
    
    # Simulate the ID reassignment process.
    test_mask = copied_mask.copy()
    unique_ids = np.unique(test_mask[test_mask > 0])
    current_id = 1000  # Start from a high number to avoid conflicts.
    
    print(f"Before reassignment: {len(unique_ids)} nuclei")
    print(f"ID range before: {unique_ids.min()} - {unique_ids.max()}")
    
    # Apply the same logic as in the actual code.
    for old_id in unique_ids:
        test_mask[test_mask == old_id] = current_id
        current_id += 1
    
    reassigned_nuclei = np.unique(test_mask[test_mask > 0])
    print(f"After reassignment: {len(reassigned_nuclei)} nuclei")
    print(f"ID range after: {reassigned_nuclei.min()} - {reassigned_nuclei.max()}")
    
    if len(reassigned_nuclei) == len(unique_ids):
        print("✅ ID reassignment preserved all nuclei")
    else:
        print("❌ ID reassignment lost nuclei!")
        print(f"   Lost: {len(unique_ids) - len(reassigned_nuclei)} nuclei")
    
    # Step 4: Check actual reassigned file.
    print(f"\n{'='*60}")
    print("STEP 4: CHECKING ACTUAL REASSIGNED FILE")
    print(f"{'='*60}")
    
    # The actual reassigned file should be the same as copied_path since
    # the reassignment happens in-place.
    final_data = np.load(copied_path)
    final_mask = final_data["mask"].astype(np.uint32)
    final_nuclei = np.unique(final_mask[final_mask > 0])
    
    print(f"Final nuclei count: {len(final_nuclei)}")
    print(f"Final ID range: {final_nuclei.min()} - {final_nuclei.max()}")
    
    # Compare with our simulation.
    if len(final_nuclei) == len(reassigned_nuclei):
        print("✅ Actual reassignment matches simulation")
    else:
        print("❌ Actual reassignment differs from simulation!")
        print(f"   Expected: {len(reassigned_nuclei)} nuclei")
        print(f"   Actual: {len(final_nuclei)} nuclei")
        print(f"   Difference: {len(reassigned_nuclei) - len(final_nuclei)} nuclei")
    
    # Step 5: Check for file naming issues.
    print(f"\n{'='*60}")
    print("STEP 5: CHECKING FILE NAMING CONSISTENCY")
    print(f"{'='*60}")
    
    # Check if there are any pixel coordinate files.
    pixel_files = list(target_dir.glob("*.npz"))
    tile_coord_files = [f for f in pixel_files if "_" in f.name and not f.name.startswith("0_")]
    
    print(f"Total files in target directory: {len(pixel_files)}")
    print(f"Files with pixel coordinates: {len(tile_coord_files)}")
    
    if len(tile_coord_files) > 0:
        print("Sample pixel coordinate files:")
        for f in tile_coord_files[:5]:
            print(f"  {f.name}")
    
    # Check if our target file exists with pixel coordinates.
    # For tile (0, 410), the pixel coordinates would be (0, 410*410) = (0, 168100)
    # But this depends on the tile size and overlap.
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"Original nuclei: {len(original_nuclei)}")
    print(f"After copy: {len(copied_nuclei)}")
    print(f"After simulation: {len(reassigned_nuclei)}")
    print(f"Final result: {len(final_nuclei)}")
    
    if len(final_nuclei) < len(original_nuclei) * 0.5:
        print("❌ CRITICAL: >50% nuclei loss detected!")
        print("   This indicates a serious bug in the process.")
    else:
        print("✅ Nuclei preservation looks reasonable.")

if __name__ == "__main__":
    try:
        debug_copy_and_reassignment()
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

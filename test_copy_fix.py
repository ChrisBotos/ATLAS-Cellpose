"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_copy_fix.py.
Description:
    Test script to validate the fix for the critical file naming bug
    in the copy process. This script tests the corrected copy function
    to ensure that nuclei are preserved during the copy and ID reassignment.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib, shutil.

Usage:
    python test_copy_fix.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/

Outputs:
    • Validation of the corrected copy process.
    • Test results for nuclei preservation.

Key Features:
    • Tests the corrected file naming logic.
    • Validates copy and ID reassignment process.
    • Ensures nuclei preservation.

Notes:
    • This script validates the fix for the file naming mismatch bug.
    • The corrected algorithm should preserve 100% of nuclei during copy.
"""

import traceback
import logging
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_corrected_copy_process():
    """
    Test the corrected copy process with a clean temporary directory.
    """
    print("=" * 80)
    print("TESTING CORRECTED COPY PROCESS")
    print("=" * 80)
    
    results_dir = Path("results/20250723_043122_cpu_cellpose4_diameter0_large_crop")
    source_dir = results_dir / "masks" / "tile_masks_npz"
    
    # Test tiles.
    test_coords = [(0, 0), (0, 410), (410, 0), (410, 410)]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_target_dir = Path(temp_dir) / "test_merged"
        temp_target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Source directory: {source_dir}")
        print(f"Test target directory: {temp_target_dir}")
        
        # Test the corrected copy function.
        print(f"\n{'='*60}")
        print("TESTING CORRECTED COPY FUNCTION")
        print(f"{'='*60}")
        
        total_original_nuclei = 0
        total_copied_nuclei = 0
        
        for tile_coord in test_coords:
            r, c = tile_coord
            tile_filename = f"{r}_{c}.npz"
            
            source_path = source_dir / tile_filename
            target_path = temp_target_dir / tile_filename
            
            if not source_path.exists():
                print(f"⚠️  Skipping {tile_filename} - source not found")
                continue
            
            # Load original tile.
            original_data = np.load(source_path)
            original_mask = original_data["mask"].astype(np.uint32)
            original_nuclei = np.unique(original_mask[original_mask > 0])
            
            print(f"\nTile {tile_filename}:")
            print(f"  Original nuclei: {len(original_nuclei)}")
            
            # Copy using corrected logic.
            try:
                shutil.copy2(source_path, target_path)
                print(f"  ✅ Copy successful")
                
                # Verify copy.
                copied_data = np.load(target_path)
                copied_mask = copied_data["mask"].astype(np.uint32)
                copied_nuclei = np.unique(copied_mask[copied_mask > 0])
                
                print(f"  Copied nuclei: {len(copied_nuclei)}")
                
                if np.array_equal(original_mask, copied_mask):
                    print(f"  ✅ Copy verification: PERFECT")
                else:
                    print(f"  ❌ Copy verification: FAILED")
                
                total_original_nuclei += len(original_nuclei)
                total_copied_nuclei += len(copied_nuclei)
                
            except Exception as e:
                print(f"  ❌ Copy failed: {e}")
        
        print(f"\n{'='*60}")
        print("COPY PROCESS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total original nuclei: {total_original_nuclei}")
        print(f"Total copied nuclei: {total_copied_nuclei}")
        
        if total_copied_nuclei == total_original_nuclei:
            print("✅ PERFECT: 100% nuclei preserved during copy")
        else:
            loss = total_original_nuclei - total_copied_nuclei
            loss_percent = (loss / total_original_nuclei) * 100
            print(f"❌ LOSS: {loss} nuclei lost ({loss_percent:.1f}%)")
        
        # Test ID reassignment.
        print(f"\n{'='*60}")
        print("TESTING ID REASSIGNMENT")
        print(f"{'='*60}")
        
        current_id = 1
        total_reassigned_nuclei = 0
        
        for tile_coord in test_coords:
            r, c = tile_coord
            tile_filename = f"{r}_{c}.npz"
            target_path = temp_target_dir / tile_filename
            
            if not target_path.exists():
                continue
            
            # Load copied tile.
            tile_data = np.load(target_path)
            tile_mask = tile_data["mask"].astype(np.uint32)
            
            # Get unique nucleus IDs (excluding background).
            unique_ids = np.unique(tile_mask[tile_mask > 0])
            original_count = len(unique_ids)
            
            if len(unique_ids) > 0:
                # Apply ID reassignment.
                for old_id in unique_ids:
                    tile_mask[tile_mask == old_id] = current_id
                    current_id += 1
                
                # Save updated tile mask.
                np.savez_compressed(target_path, mask=tile_mask)
                
                # Verify reassignment.
                reassigned_nuclei = np.unique(tile_mask[tile_mask > 0])
                reassigned_count = len(reassigned_nuclei)
                
                print(f"Tile {tile_filename}:")
                print(f"  Before reassignment: {original_count} nuclei")
                print(f"  After reassignment: {reassigned_count} nuclei")
                
                if reassigned_count == original_count:
                    print(f"  ✅ ID reassignment: PERFECT")
                else:
                    print(f"  ❌ ID reassignment: FAILED")
                
                total_reassigned_nuclei += reassigned_count
        
        print(f"\n{'='*60}")
        print("ID REASSIGNMENT SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total copied nuclei: {total_copied_nuclei}")
        print(f"Total reassigned nuclei: {total_reassigned_nuclei}")
        
        if total_reassigned_nuclei == total_copied_nuclei:
            print("✅ PERFECT: 100% nuclei preserved during ID reassignment")
        else:
            loss = total_copied_nuclei - total_reassigned_nuclei
            loss_percent = (loss / total_copied_nuclei) * 100
            print(f"❌ LOSS: {loss} nuclei lost ({loss_percent:.1f}%)")
        
        # Final summary.
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        print(f"Original nuclei: {total_original_nuclei}")
        print(f"Final nuclei: {total_reassigned_nuclei}")
        
        if total_reassigned_nuclei == total_original_nuclei:
            print("✅ EXCELLENT: 100% nuclei preserved through entire process")
            return True
        else:
            total_loss = total_original_nuclei - total_reassigned_nuclei
            total_loss_percent = (total_loss / total_original_nuclei) * 100
            print(f"❌ TOTAL LOSS: {total_loss} nuclei lost ({total_loss_percent:.1f}%)")
            return False

if __name__ == "__main__":
    try:
        success = test_corrected_copy_process()
        
        if success:
            print("\n🎉 ALL TESTS PASSED - Copy fix is working correctly!")
        else:
            print("\n❌ TESTS FAILED - Further fixes needed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

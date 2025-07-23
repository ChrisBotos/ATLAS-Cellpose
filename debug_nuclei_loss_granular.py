"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_nuclei_loss_granular.py.
Description:
    Comprehensive debugging script to track nuclei loss at every granular step
    during the copying and ID reassignment phases. This script monitors the
    problematic tile 0_410.npz which loses 699 nuclei (97% loss) from 721 to 22
    nuclei during the two-phase merge process.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib, shutil.

Usage:
    python debug_nuclei_loss_granular.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/
    • Copied tile masks from masks/merged_tile_masks_npz/

Outputs:
    • Step-by-step nuclei count tracking.
    • Identification of exact loss location.
    • Comprehensive debugging logs.

Key Features:
    • Tracks nuclei counts at every critical step.
    • Monitors file I/O operations for data corruption.
    • Provides granular ID reassignment debugging.
    • Identifies the exact step where 97% nuclei loss occurs.

Notes:
    • This script investigates the critical bug causing massive nuclei loss.
    • The focus is on tile 0_410.npz which shows 97% nuclei loss.
    • Scientific analysis addresses bioinformatician users directly.
"""

import traceback
import logging
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from numpy.typing import NDArray

# Set up logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def count_nuclei_in_file(file_path: Path) -> int:
    """
    Count nuclei in a tile mask file.
    
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
    Background pixels (label=0) are excluded from the count.
    """
    try:
        data = np.load(file_path)
        mask = data["mask"].astype(np.uint32)
        unique_nuclei = np.unique(mask[mask > 0])
        return len(unique_nuclei)
    except Exception as e:
        logging.error(f"Failed to count nuclei in {file_path}: {e}")
        return 0

def debug_granular_nuclei_loss():
    """
    Debug the granular nuclei loss during copying and ID reassignment phases.
    
    This function tracks the problematic tile 0_410.npz at every step to identify
    the exact location where the massive nuclei loss occurs. The tile loses
    699 nuclei (97% loss) from 721 to 22 nuclei during processing.
    """
    print("=" * 80)
    print("GRANULAR NUCLEI LOSS DEBUGGING FOR TILE 0_410.npz")
    print("=" * 80)
    
    # Define paths for the latest run.
    results_dir = Path("results/20250723_052544_cpu_cellpose4_diameter0_large_crop")
    source_dir = results_dir / "masks" / "tile_masks_npz"
    target_dir = results_dir / "masks" / "merged_tile_masks_npz"
    
    # Focus on the problematic tile.
    target_tile = "0_410.npz"
    
    print(f"Investigating tile: {target_tile}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Step 1: Check original file before any processing.
    print(f"\n{'='*60}")
    print("STEP 1: ORIGINAL FILE ANALYSIS")
    print(f"{'='*60}")
    
    original_path = source_dir / target_tile
    if not original_path.exists():
        print(f"❌ Original file not found: {original_path}")
        return
    
    original_nuclei = count_nuclei_in_file(original_path)
    print(f"✅ Original {target_tile}: {original_nuclei} nuclei")
    print(f"   File size: {original_path.stat().st_size:,} bytes")
    
    # Step 2: Check copied file after copying phase.
    print(f"\n{'='*60}")
    print("STEP 2: AFTER COPYING PHASE")
    print(f"{'='*60}")
    
    copied_path = target_dir / target_tile
    if not copied_path.exists():
        print(f"❌ Copied file not found: {copied_path}")
        print("   This indicates the copying phase failed!")
        return
    
    copied_nuclei = count_nuclei_in_file(copied_path)
    print(f"✅ Copied {target_tile}: {copied_nuclei} nuclei")
    print(f"   File size: {copied_path.stat().st_size:,} bytes")
    
    # Calculate loss during copying.
    copy_loss = original_nuclei - copied_nuclei
    copy_loss_percent = (copy_loss / original_nuclei) * 100 if original_nuclei > 0 else 0
    
    print(f"📊 COPYING PHASE ANALYSIS:")
    print(f"   Nuclei before copying: {original_nuclei}")
    print(f"   Nuclei after copying: {copied_nuclei}")
    print(f"   Loss during copying: {copy_loss} nuclei ({copy_loss_percent:.1f}%)")
    
    if copy_loss > 0:
        print(f"🚨 CRITICAL: Nuclei loss detected during COPYING phase!")
        print(f"   The copying mechanism is corrupting data!")
    else:
        print(f"✅ GOOD: No nuclei loss during copying phase.")
    
    # Step 3: Simulate the ID reassignment process step by step.
    print(f"\n{'='*60}")
    print("STEP 3: GRANULAR ID REASSIGNMENT SIMULATION")
    print(f"{'='*60}")
    
    # Load the copied file for simulation.
    try:
        copied_data = np.load(copied_path)
        test_mask = copied_data["mask"].astype(np.uint32).copy()
        
        print(f"✅ Loaded copied mask for simulation")
        print(f"   Mask shape: {test_mask.shape}")
        print(f"   Mask dtype: {test_mask.dtype}")
        
        # Get unique IDs before reassignment.
        unique_ids_before = np.unique(test_mask[test_mask > 0])
        print(f"   Unique IDs before reassignment: {len(unique_ids_before)}")
        print(f"   ID range before: {unique_ids_before.min()} - {unique_ids_before.max()}")
        
        # Simulate the ID reassignment process.
        current_id = 1000  # Start from a high number to avoid conflicts.
        
        print(f"\n📋 SIMULATING ID REASSIGNMENT:")
        print(f"   Starting ID: {current_id}")
        
        nuclei_before_reassignment = len(unique_ids_before)
        
        # Apply the same logic as in the actual code.
        for i, old_id in enumerate(unique_ids_before):
            # Count nuclei before this specific replacement.
            nuclei_before_replacement = len(np.unique(test_mask[test_mask > 0]))
            
            # Perform the replacement.
            test_mask[test_mask == old_id] = current_id
            
            # Count nuclei after this specific replacement.
            nuclei_after_replacement = len(np.unique(test_mask[test_mask > 0]))
            
            # Check for loss in this specific operation.
            replacement_loss = nuclei_before_replacement - nuclei_after_replacement
            
            if replacement_loss != 0:
                print(f"   ⚠️  ID {old_id} -> {current_id}: {replacement_loss} nuclei lost!")
            elif i < 5 or i % 100 == 0:  # Show first few and every 100th.
                print(f"   ✅ ID {old_id} -> {current_id}: No loss")
            
            current_id += 1
        
        # Final count after all reassignments.
        unique_ids_after = np.unique(test_mask[test_mask > 0])
        nuclei_after_reassignment = len(unique_ids_after)
        
        print(f"\n📊 ID REASSIGNMENT SIMULATION RESULTS:")
        print(f"   Nuclei before reassignment: {nuclei_before_reassignment}")
        print(f"   Nuclei after reassignment: {nuclei_after_reassignment}")
        
        reassignment_loss = nuclei_before_reassignment - nuclei_after_reassignment
        reassignment_loss_percent = (reassignment_loss / nuclei_before_reassignment) * 100 if nuclei_before_reassignment > 0 else 0
        
        print(f"   Loss during reassignment: {reassignment_loss} nuclei ({reassignment_loss_percent:.1f}%)")
        
        if reassignment_loss > 0:
            print(f"🚨 CRITICAL: Nuclei loss detected during ID REASSIGNMENT!")
        else:
            print(f"✅ GOOD: No nuclei loss during ID reassignment simulation.")
        
    except Exception as e:
        print(f"❌ Error during ID reassignment simulation: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
    
    # Step 4: Compare with actual processed file.
    print(f"\n{'='*60}")
    print("STEP 4: COMPARISON WITH ACTUAL PROCESSED FILE")
    print(f"{'='*60}")
    
    # The actual processed file should be the same as copied_path since
    # ID reassignment happens in-place.
    final_nuclei = count_nuclei_in_file(copied_path)
    
    print(f"📊 FINAL COMPARISON:")
    print(f"   Original nuclei: {original_nuclei}")
    print(f"   After copying: {copied_nuclei}")
    print(f"   After simulation: {nuclei_after_reassignment}")
    print(f"   Actual final: {final_nuclei}")
    
    # Calculate total loss.
    total_loss = original_nuclei - final_nuclei
    total_loss_percent = (total_loss / original_nuclei) * 100 if original_nuclei > 0 else 0
    
    print(f"   Total loss: {total_loss} nuclei ({total_loss_percent:.1f}%)")
    
    # Step 5: Root cause identification.
    print(f"\n{'='*60}")
    print("STEP 5: ROOT CAUSE IDENTIFICATION")
    print(f"{'='*60}")
    
    if copy_loss > total_loss * 0.8:  # If >80% of loss is during copying.
        print(f"🎯 ROOT CAUSE: COPYING PHASE")
        print(f"   The file copying mechanism is corrupting data.")
        print(f"   Loss during copying: {copy_loss} nuclei ({copy_loss_percent:.1f}%)")
        print(f"   This suggests issues with file I/O operations.")
        
    elif reassignment_loss > total_loss * 0.8:  # If >80% of loss is during reassignment.
        print(f"🎯 ROOT CAUSE: ID REASSIGNMENT PHASE")
        print(f"   The ID reassignment process is corrupting data.")
        print(f"   Loss during reassignment: {reassignment_loss} nuclei ({reassignment_loss_percent:.1f}%)")
        print(f"   This suggests issues with array indexing operations.")
        
    else:
        print(f"🎯 ROOT CAUSE: MIXED OR UNKNOWN")
        print(f"   Loss is distributed across multiple phases.")
        print(f"   Further investigation needed.")
    
    print(f"\n{'='*60}")
    print("SCIENTIFIC CONCLUSION")
    print(f"{'='*60}")
    
    print(f"For bioinformatics analysis of kidney tissue segmentation:")
    print(f"• Original detection: {original_nuclei} nuclei in tile 0_410.npz")
    print(f"• Final result: {final_nuclei} nuclei after processing")
    print(f"• Data integrity loss: {total_loss_percent:.1f}% of nuclei lost")
    print(f"• This level of data loss is unacceptable for quantitative analysis.")
    print(f"• Immediate fix required for reliable spatial transcriptomics integration.")

if __name__ == "__main__":
    try:
        debug_granular_nuclei_loss()
        
    except Exception as e:
        print(f"❌ Error during granular debugging: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

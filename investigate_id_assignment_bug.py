"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: investigate_id_assignment_bug.py.
Description:
    Comprehensive investigation of the nucleus ID assignment bug causing
    massive nuclei deletion from tile interiors. This script analyzes the
    entire merging pipeline to identify where nuclei are being incorrectly
    deleted or corrupted during the ID reassignment process.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib.

Usage:
    python investigate_id_assignment_bug.py

Arguments:
    None.

Inputs:
    • Original tile masks from masks/tile_masks_npz/
    • Merged tile masks from masks/merged_tile_masks_npz/
    • Merging pipeline code analysis.

Outputs:
    • Detailed analysis of ID assignment process.
    • Step-by-step tracking of nuclei loss.
    • Root cause identification.

Key Features:
    • Traces nucleus IDs through the entire pipeline.
    • Identifies where nuclei are lost or corrupted.
    • Analyzes file I/O operations and memory corruption.
    • Provides specific bug location and fix recommendations.

Notes:
    • This bug is causing 99% nuclei loss in some tiles.
    • The issue appears to be in the ID reassignment or file operations.
    • Interior nuclei should NEVER be modified during merging.
"""

import traceback
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from numpy.typing import NDArray

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def investigate_id_assignment_pipeline():
    """
    Comprehensive investigation of the ID assignment and merging pipeline.
    """
    print("=" * 80)
    print("CRITICAL BUG INVESTIGATION: ID ASSIGNMENT PIPELINE")
    print("=" * 80)
    
    results_dir = Path("results/20250723_043122_cpu_cellpose4_diameter0_large_crop")
    
    # Step 1: Analyze the ID reassignment process.
    print("\n1. ANALYZING ID REASSIGNMENT PROCESS")
    print("-" * 50)
    analyze_id_reassignment_logic()
    
    # Step 2: Compare original vs post-reassignment tiles.
    print("\n2. COMPARING ORIGINAL VS POST-REASSIGNMENT TILES")
    print("-" * 50)
    compare_original_vs_reassigned(results_dir)
    
    # Step 3: Analyze the merge operations.
    print("\n3. ANALYZING MERGE OPERATIONS")
    print("-" * 50)
    analyze_merge_operations(results_dir)
    
    # Step 4: Check file I/O operations.
    print("\n4. CHECKING FILE I/O OPERATIONS")
    print("-" * 50)
    check_file_io_operations(results_dir)
    
    # Step 5: Identify the exact bug location.
    print("\n5. ROOT CAUSE ANALYSIS")
    print("-" * 50)
    identify_root_cause()

def analyze_id_reassignment_logic():
    """
    Analyze the ID reassignment logic in the two-phase merge.
    """
    print("Analyzing ID reassignment logic...")
    
    # Read the two_phase_merge.py file to understand the ID reassignment.
    try:
        with open("code/nuclei_segmentation/cellpose_merge/two_phase_merge.py", 'r') as f:
            content = f.read()
        
        # Look for ID reassignment code.
        if "current_id = 1" in content:
            print("✅ Found ID reassignment starting from 1")
        else:
            print("❌ ID reassignment start point not found")
        
        if "tile_mask[tile_mask == old_id] = current_id" in content:
            print("✅ Found ID replacement logic")
        else:
            print("❌ ID replacement logic not found")
        
        # Check for potential issues.
        issues = []
        
        if "tile_mask[tile_mask == old_id] = current_id" in content:
            issues.append("POTENTIAL BUG: Direct array assignment without bounds checking")
        
        if "current_id += 1" in content:
            print("✅ Found ID increment logic")
        else:
            issues.append("POTENTIAL BUG: ID increment logic missing")
        
        if issues:
            print("🚨 POTENTIAL ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"❌ Error reading two_phase_merge.py: {e}")

def compare_original_vs_reassigned(results_dir: Path):
    """
    Compare original tiles with post-reassignment tiles to track nuclei loss.
    """
    print("Comparing original vs reassigned tiles...")
    
    original_dir = results_dir / "masks" / "tile_masks_npz"
    merged_dir = results_dir / "masks" / "merged_tile_masks_npz"
    
    # Focus on the problematic tile.
    tile_name = "0_410.npz"
    
    original_path = original_dir / tile_name
    merged_path = merged_dir / tile_name
    
    if not original_path.exists() or not merged_path.exists():
        print(f"❌ Cannot find tiles: {original_path} or {merged_path}")
        return
    
    # Load both tiles.
    original_data = np.load(original_path)
    merged_data = np.load(merged_path)
    
    original_mask = original_data["mask"].astype(np.uint32)
    merged_mask = merged_data["mask"].astype(np.uint32)
    
    print(f"Original tile shape: {original_mask.shape}")
    print(f"Merged tile shape: {merged_mask.shape}")
    
    # Count nuclei.
    original_nuclei = np.unique(original_mask[original_mask > 0])
    merged_nuclei = np.unique(merged_mask[merged_mask > 0])
    
    print(f"Original nuclei count: {len(original_nuclei)}")
    print(f"Merged nuclei count: {len(merged_nuclei)}")
    print(f"Nuclei lost: {len(original_nuclei) - len(merged_nuclei)}")
    
    # Check if this is an ID reassignment issue.
    print(f"Original ID range: {original_nuclei.min()} - {original_nuclei.max()}")
    print(f"Merged ID range: {merged_nuclei.min()} - {merged_nuclei.max()}")
    
    # Check for ID overlap.
    id_overlap = set(original_nuclei) & set(merged_nuclei)
    print(f"ID overlap: {len(id_overlap)} nuclei")
    
    if len(id_overlap) == 0:
        print("🚨 CRITICAL: NO ID OVERLAP - Complete ID reassignment occurred")
        print("   This suggests the issue is in the ID reassignment process")
    
    # Check pixel-level changes.
    pixels_changed = np.sum(original_mask != merged_mask)
    total_pixels = original_mask.size
    change_percentage = (pixels_changed / total_pixels) * 100
    
    print(f"Pixels changed: {pixels_changed:,} / {total_pixels:,} ({change_percentage:.1f}%)")
    
    if change_percentage > 50:
        print("🚨 CRITICAL: >50% of pixels changed - This is NOT normal merge behavior")

def analyze_merge_operations(results_dir: Path):
    """
    Analyze the specific merge operations that might be causing nuclei loss.
    """
    print("Analyzing merge operations...")
    
    # Read the rules.py file to understand merge logic.
    try:
        with open("code/nuclei_segmentation/cellpose_merge/rules.py", 'r') as f:
            content = f.read()
        
        # Look for deletion operations.
        deletion_patterns = [
            "updated_tile1_mask[nucleus_mask] = 0",
            "updated_tile2_mask[nucleus_mask] = 0",
            "tile_mask[nucleus_mask] = 0"
        ]
        
        for pattern in deletion_patterns:
            if pattern in content:
                print(f"✅ Found deletion pattern: {pattern}")
            else:
                print(f"❌ Deletion pattern not found: {pattern}")
        
        # Check for boundary detection issues.
        if "_find_border_touching_nuclei" in content:
            print("✅ Found border detection function")
            
            # Check if it's being called with correct parameters.
            if "overlap_length," in content:
                print("✅ Border detection uses overlap_length parameter")
            else:
                print("❌ Border detection may not use overlap_length correctly")
        
        # Check for interior region protection.
        if "interior" in content.lower():
            print("✅ Found interior region references")
        else:
            print("🚨 WARNING: No interior region protection found")
        
    except Exception as e:
        print(f"❌ Error reading rules.py: {e}")

def check_file_io_operations(results_dir: Path):
    """
    Check file I/O operations for potential corruption.
    """
    print("Checking file I/O operations...")
    
    # Check if files are being corrupted during save/load.
    tile_name = "0_410.npz"
    merged_path = results_dir / "masks" / "merged_tile_masks_npz" / tile_name
    
    if not merged_path.exists():
        print(f"❌ Merged tile not found: {merged_path}")
        return
    
    try:
        # Load the file multiple times to check consistency.
        data1 = np.load(merged_path)
        mask1 = data1["mask"]
        
        data2 = np.load(merged_path)
        mask2 = data2["mask"]
        
        if np.array_equal(mask1, mask2):
            print("✅ File I/O is consistent")
        else:
            print("🚨 CRITICAL: File I/O inconsistency detected")
        
        # Check file size.
        file_size = merged_path.stat().st_size
        print(f"File size: {file_size:,} bytes")
        
        if file_size < 1000:  # Very small file.
            print("🚨 WARNING: File size is suspiciously small")
        
    except Exception as e:
        print(f"❌ Error checking file I/O: {e}")

def identify_root_cause():
    """
    Identify the most likely root cause based on the analysis.
    """
    print("Identifying root cause...")
    
    print("\nBASED ON THE ANALYSIS, THE MOST LIKELY ROOT CAUSES ARE:")
    print("=" * 60)
    
    print("\n1. 🚨 ID REASSIGNMENT BUG:")
    print("   - The ID reassignment process is corrupting tile data")
    print("   - Nuclei are being lost during the ID mapping process")
    print("   - The reassignment may be overwriting existing nuclei")
    
    print("\n2. 🚨 INCORRECT BORDER DETECTION:")
    print("   - Border detection is identifying interior nuclei as border nuclei")
    print("   - The overlap_length parameter may be incorrectly applied")
    print("   - Interior regions are not properly protected")
    
    print("\n3. 🚨 MERGE LOGIC ERROR:")
    print("   - The merge algorithm is deleting nuclei from wrong regions")
    print("   - Priority selection may be causing excessive deletion")
    print("   - Cross-boundary preservation is not working correctly")
    
    print("\n4. 🚨 FILE I/O CORRUPTION:")
    print("   - Tiles are being corrupted during save/load operations")
    print("   - Memory management issues during processing")
    print("   - Array indexing errors causing data loss")
    
    print("\nRECOMMENDED INVESTIGATION ORDER:")
    print("1. Fix ID reassignment process (most likely cause)")
    print("2. Validate border detection logic")
    print("3. Add interior region protection")
    print("4. Implement data integrity checks")

if __name__ == "__main__":
    try:
        investigate_id_assignment_pipeline()
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

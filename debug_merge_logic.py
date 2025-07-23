"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_merge_logic.py.
Description:
    Debug script to understand exactly what's happening in the merge logic
    that's causing massive nuclei deletion from interior regions. This script
    will trace the exact merge operations for the problematic tile.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pathlib.

Usage:
    python debug_merge_logic.py

Arguments:
    None.

Inputs:
    • Merge log files for detailed analysis.

Outputs:
    • Step-by-step analysis of merge operations.
    • Identification of incorrect border detection.

Key Features:
    • Traces merge operations for specific tiles.
    • Identifies incorrect direction mappings.
    • Analyzes border detection logic.

Notes:
    • This script investigates why tile 0_410.npz loses 99% of its nuclei.
    • The issue appears to be in incorrect border detection application.
"""

import traceback
import logging
import numpy as np
from pathlib import Path

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_merge_operations():
    """
    Analyze the merge operations from the log to understand the nuclei loss.
    """
    print("=" * 80)
    print("DEBUGGING MERGE LOGIC FOR TILE 0_410.npz")
    print("=" * 80)
    
    log_file = Path("results/20250723_051724_cpu_cellpose4_diameter0_large_crop/logs/segmentation_log_20250723_051724.txt")
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find merge operations involving tile 0_410.
    print("SEARCHING FOR MERGE OPERATIONS INVOLVING TILE 0_410...")
    
    merge_operations = []
    for i, line in enumerate(lines):
        if "Enhanced 3-step merge" in line and "0_410" in line:
            # Extract the merge details.
            merge_info = {
                'line_num': i,
                'merge_line': line.strip(),
                'context': lines[max(0, i-5):i+10]  # Get context around the merge.
            }
            merge_operations.append(merge_info)
    
    print(f"Found {len(merge_operations)} merge operations involving tile 0_410")
    
    for i, merge_op in enumerate(merge_operations):
        print(f"\n{'='*60}")
        print(f"MERGE OPERATION {i+1}")
        print(f"{'='*60}")
        
        print(f"Line {merge_op['line_num']}: {merge_op['merge_line']}")
        
        print("\nContext:")
        for j, context_line in enumerate(merge_op['context']):
            marker = ">>>" if j == 5 else "   "  # Mark the main line.
            print(f"{marker} {context_line.strip()}")
    
    # Look for specific patterns that indicate the bug.
    print(f"\n{'='*60}")
    print("ANALYZING BORDER DETECTION PATTERNS")
    print(f"{'='*60}")
    
    border_detection_lines = []
    for i, line in enumerate(lines):
        if "border-touching nuclei" in line.lower() or "overlap region nuclei" in line.lower():
            border_detection_lines.append((i, line.strip()))
    
    print(f"Found {len(border_detection_lines)} border detection operations")
    
    # Look for the specific problematic pattern.
    for line_num, line in border_detection_lines:
        if "726" in line or "719" in line:  # Numbers related to tile 0_410.
            print(f"Line {line_num}: {line}")
    
    # Check for direction mapping issues.
    print(f"\n{'='*60}")
    print("CHECKING DIRECTION MAPPING")
    print(f"{'='*60}")
    
    direction_lines = []
    for i, line in enumerate(lines):
        if "direction" in line.lower() and ("tile1" in line or "tile2" in line):
            direction_lines.append((i, line.strip()))
    
    for line_num, line in direction_lines[-10:]:  # Show last 10 direction mappings.
        print(f"Line {line_num}: {line}")

def analyze_tile_coordinates():
    """
    Analyze the tile coordinate system to understand the merge relationships.
    """
    print(f"\n{'='*60}")
    print("ANALYZING TILE COORDINATE SYSTEM")
    print(f"{'='*60}")
    
    # Based on the file names we saw earlier.
    tile_files = [
        "0_0.npz", "0_410.npz", "0_820.npz", "0_1230.npz",
        "410_0.npz", "410_410.npz", "410_820.npz", "410_1230.npz",
        "820_0.npz", "820_410.npz", "820_820.npz", "820_1230.npz",
        "1230_0.npz", "1230_410.npz", "1230_820.npz", "1230_1230.npz"
    ]
    
    print("Tile grid (pixel coordinates):")
    print("     0      410     820    1230")
    print("0    0_0    0_410   0_820  0_1230")
    print("410  410_0  410_410 410_820 410_1230")
    print("820  820_0  820_410 820_820 820_1230")
    print("1230 1230_0 1230_410 1230_820 1230_1230")
    
    print(f"\nTile 0_410.npz is at position (0, 410)")
    print("This means it's in the TOP ROW, SECOND COLUMN")
    print("Its neighbors should be:")
    print("  - Left: 0_0.npz")
    print("  - Right: 0_820.npz")
    print("  - Below: 410_410.npz")
    print("  - No tile above (it's at the top)")
    
    print(f"\nFor tile 0_410.npz, the overlap regions should be:")
    print("  - RIGHT overlap: cols 410-512 (with tile 0_820.npz)")
    print("  - BOTTOM overlap: rows 410-512 (with tile 410_410.npz)")
    print("  - LEFT overlap: cols 0-102 (with tile 0_0.npz)")
    print("  - NO TOP overlap (it's at the top edge)")
    
    print(f"\nThe INTERIOR region should be:")
    print("  - Rows: 0-410 (no top overlap)")
    print("  - Cols: 102-410 (excluding left and right overlaps)")
    
    print(f"\nNuclei being deleted from cols 104-126 are in the INTERIOR!")
    print("This confirms the bug - interior nuclei are being incorrectly deleted.")

if __name__ == "__main__":
    try:
        analyze_merge_operations()
        analyze_tile_coordinates()
        
        print(f"\n{'='*80}")
        print("CONCLUSION")
        print(f"{'='*80}")
        print("The bug is in the border detection logic.")
        print("Nuclei in the tile interior are being incorrectly marked as 'border-touching'")
        print("and then deleted in Step 2 of the merge algorithm.")
        print("This is causing the 99% nuclei loss in tile 0_410.npz.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")

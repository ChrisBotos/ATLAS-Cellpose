"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_border_deletion_validation.py.
Description:
    Validation script to verify that the 3-step merging rules are working correctly,
    specifically focusing on Step 2 (Border Deletion) and Step 3 (Cross-boundary Preservation).
    
    This script loads actual mask outputs from the pipeline and analyzes them to ensure
    that priority border-touching nuclei are being properly deleted and cross-boundary
    nuclei are being preserved.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • pathlib for file handling.

Key Features:
    • Loads before/after merging masks from actual pipeline results.
    • Analyzes nuclei reduction patterns to validate 3-step rules.
    • Provides detailed statistics on border deletion effectiveness.
    • Validates that cross-boundary nuclei are properly preserved.
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Configure logging for detailed analysis.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add the cellpose_merge module to path.
sys.path.append('code/nuclei_segmentation/cellpose_merge')

from rules import merge_tiles_cpu_3step, _find_border_touching_nuclei


def analyze_pipeline_results(results_dir: Path):
    """
    Analyze the actual pipeline results to validate 3-step merging effectiveness.
    
    Parameters
    ----------
    results_dir : Path
        Path to the pipeline results directory.
    """
    print(f"=== Analyzing Pipeline Results: {results_dir.name} ===")
    
    # Load the final merged mask.
    merged_mask_path = results_dir / "masks" / "segmentation_masks.npy"
    if not merged_mask_path.exists():
        print(f"ERROR: Merged mask not found at {merged_mask_path}")
        return
    
    merged_mask = np.load(merged_mask_path)
    print(f"Final merged mask: {merged_mask.shape}, {len(np.unique(merged_mask[merged_mask > 0]))} nuclei")
    
    # Load individual tile masks to analyze before merging.
    tile_masks_dir = results_dir / "masks" / "tile_masks_npz"
    if not tile_masks_dir.exists():
        print(f"ERROR: Tile masks directory not found at {tile_masks_dir}")
        return
    
    # Count nuclei before merging.
    total_nuclei_before = 0
    tile_files = list(tile_masks_dir.glob("*.npz"))
    
    print(f"Found {len(tile_files)} tile mask files")
    
    for tile_file in sorted(tile_files):
        tile_data = np.load(tile_file)
        tile_mask = tile_data['mask']
        nuclei_count = len(np.unique(tile_mask[tile_mask > 0]))
        total_nuclei_before += nuclei_count
        print(f"  {tile_file.name}: {nuclei_count} nuclei")
    
    # Calculate reduction statistics.
    nuclei_after = len(np.unique(merged_mask[merged_mask > 0]))
    nuclei_deleted = total_nuclei_before - nuclei_after
    deletion_rate = (nuclei_deleted / total_nuclei_before) * 100 if total_nuclei_before > 0 else 0
    
    print(f"\n--- Merging Statistics ---")
    print(f"Nuclei before merging: {total_nuclei_before}")
    print(f"Nuclei after merging: {nuclei_after}")
    print(f"Nuclei deleted: {nuclei_deleted}")
    print(f"Deletion rate: {deletion_rate:.1f}%")
    
    # Analyze if the deletion rate is reasonable for 3-step rules.
    if deletion_rate > 70:
        print("⚠️  WARNING: Very high deletion rate (>70%) - may indicate over-aggressive border deletion")
    elif deletion_rate > 40:
        print("✓ GOOD: Moderate deletion rate suggests effective border deletion")
    elif deletion_rate > 10:
        print("✓ OK: Low-moderate deletion rate - border deletion working")
    else:
        print("⚠️  WARNING: Very low deletion rate (<10%) - border deletion may not be working")
    
    return total_nuclei_before, nuclei_after, deletion_rate


def test_3step_algorithm_directly():
    """
    Test the 3-step algorithm directly with controlled synthetic data.
    """
    print("\n=== Direct 3-Step Algorithm Test ===")
    
    # Create a test scenario with known expected behavior.
    overlap1 = np.array([
        [1, 1, 0, 0],  # Priority nucleus 1 (touches top+left borders) → should be DELETED
        [1, 1, 0, 0],
        [0, 0, 2, 2],  # Priority nucleus 2 (touches right+bottom borders) → should be DELETED
        [0, 0, 2, 2]
    ], dtype=np.uint32)
    
    overlap2 = np.array([
        [3, 3, 0, 0],  # Non-priority nucleus 3 (touches borders) → should be PRESERVED (cross-boundary)
        [3, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint32)
    
    print("Test overlap1 (priority tile):")
    print(overlap1)
    print("Test overlap2 (non-priority tile):")
    print(overlap2)
    
    # Test border detection.
    border_nuclei_1 = _find_border_touching_nuclei(overlap1)
    border_nuclei_2 = _find_border_touching_nuclei(overlap2)
    
    print(f"Border-touching nuclei in overlap1: {border_nuclei_1}")
    print(f"Border-touching nuclei in overlap2: {border_nuclei_2}")
    
    # Apply 3-step algorithm.
    patch = np.stack([overlap1, overlap2], axis=0)
    merged, mapping = merge_tiles_cpu_3step(patch)
    
    print("Merged result:")
    print(merged)
    print(f"Mapping: {mapping}")
    
    # Analyze results.
    original_nuclei = {1, 2, 3}
    preserved_nuclei = set(mapping.keys())
    deleted_nuclei = original_nuclei - preserved_nuclei
    
    print(f"Original nuclei: {original_nuclei}")
    print(f"Preserved nuclei: {preserved_nuclei}")
    print(f"Deleted nuclei: {deleted_nuclei}")
    
    # Expected behavior:
    # - Priority nuclei 1, 2 should be deleted (border-touching)
    # - Non-priority nucleus 3 should be preserved (cross-boundary)
    expected_preserved = {3}
    expected_deleted = {1, 2}
    
    if preserved_nuclei == expected_preserved and deleted_nuclei == expected_deleted:
        print("✓ PASS: 3-step algorithm working correctly")
        return True
    else:
        print(f"✗ FAIL: Expected preserved {expected_preserved}, deleted {expected_deleted}")
        print(f"        Got preserved {preserved_nuclei}, deleted {deleted_nuclei}")
        return False


def main():
    """Run validation tests on the 3-step merging implementation."""
    print("Validating 3-Step Border Deletion Implementation")
    print("=" * 50)
    
    # Test 1: Direct algorithm test.
    algorithm_working = test_3step_algorithm_directly()
    
    # Test 2: Analyze actual pipeline results.
    results_dirs = list(Path("results").glob("*test_new_cellpose4_diameter0_large_crop*"))
    
    if not results_dirs:
        print("\nNo pipeline results found to analyze")
        return
    
    # Analyze the most recent results.
    latest_results = max(results_dirs, key=lambda p: p.name)
    total_before, total_after, deletion_rate = analyze_pipeline_results(latest_results)
    
    # Overall assessment.
    print(f"\n=== Overall Assessment ===")
    
    if algorithm_working:
        print("✓ 3-step algorithm core logic: WORKING")
    else:
        print("✗ 3-step algorithm core logic: FAILING")
    
    if 10 <= deletion_rate <= 70:
        print("✓ Pipeline integration: WORKING (reasonable deletion rate)")
    else:
        print("⚠️  Pipeline integration: NEEDS INVESTIGATION (unusual deletion rate)")
    
    # Recommendations.
    print(f"\n=== Recommendations ===")
    if not algorithm_working:
        print("• Fix core 3-step algorithm implementation")
    
    if deletion_rate > 70:
        print("• Investigate over-aggressive border deletion")
        print("• Check if border detection is too sensitive")
    elif deletion_rate < 10:
        print("• Investigate under-active border deletion")
        print("• Check if 3-step rules are being applied correctly")
    
    print("• Examine QC overlays to visually validate merge quality")
    print("• Check merge_qc_overlays/before_merging.tif and after_merging.tif")


if __name__ == "__main__":
    main()

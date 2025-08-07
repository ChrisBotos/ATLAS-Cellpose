"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_spatial_coordinate_mapping.py.
Description:
    Test the spatial coordinate mapping in Step 4 (Mask Redistribution) of the 4-step
    CPU merging algorithm. This test specifically verifies that masks are properly
    redistributed without creating visible boundary lines or cropping artifacts.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations and mask generation.
    • tempfile for temporary file management.
    • logging for detailed debug output.
    • matplotlib for visualization (optional).

Usage:
    python tests/test_spatial_coordinate_mapping.py

Key Features:
    • Creates masks that span tile boundaries to test seamless merging.
    • Validates that no visible demarcation lines exist in final output.
    • Checks that nuclei extending from tile bodies into overlap regions are preserved.
    • Comprehensive pixel-level validation of spatial coordinate mapping.

Notes:
    • Designed to identify and fix the critical Step 4 redistribution bug.
    • Uses controlled synthetic data for reproducible debugging.
    • Focuses on spatial coordinate mapping between overlap regions and full tiles.
"""

import logging
import tempfile
import traceback
from pathlib import Path

import numpy as np

# Add the project root to the Python path.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.cpu_merge import merge_tiles_cpu_4step


def create_boundary_spanning_masks():
    """
    Create test masks with nuclei that span tile boundaries.
    
    This creates a scenario where nuclei extend from tile bodies into overlap regions,
    which is the critical test case for spatial coordinate mapping.
    
    Returns
    -------
    tuple
        (tile1_mask, tile2_mask) where nuclei span the boundary between tiles.
    """
    # Create 100x100 masks.
    tile1_mask = np.zeros((100, 100), dtype=np.uint32)
    tile2_mask = np.zeros((100, 100), dtype=np.uint32)
    
    y_coords, x_coords = np.ogrid[:100, :100]
    
    # For "right" relationship (tile1 right of tile2) with overlap_length=20:
    # - tile1's overlap region: left 20 columns (0-19)
    # - tile2's overlap region: right 20 columns (80-99)
    # - Boundary is at tile1 column 20 / tile2 column 80
    
    # Create nucleus in tile1 that spans from tile body into overlap region.
    # Center at (50, 15) with radius 10 - spans columns 5-25.
    tile1_spanning_nucleus = ((y_coords - 50)**2 + (x_coords - 15)**2) <= 100  # radius 10
    tile1_mask[tile1_spanning_nucleus] = 1
    
    # Create nucleus in tile2 that spans from tile body into overlap region.
    # Center at (50, 85) with radius 10 - spans columns 75-95.
    tile2_spanning_nucleus = ((y_coords - 50)**2 + (x_coords - 85)**2) <= 100  # radius 10
    tile2_mask[tile2_spanning_nucleus] = 2
    
    # Add non-overlapping nuclei for comparison.
    # Tile1 nucleus entirely in tile body.
    tile1_body_nucleus = ((y_coords - 25)**2 + (x_coords - 50)**2) <= 64  # radius 8
    tile1_mask[tile1_body_nucleus] = 3
    
    # Tile2 nucleus entirely in tile body.
    tile2_body_nucleus = ((y_coords - 75)**2 + (x_coords - 50)**2) <= 64  # radius 8
    tile2_mask[tile2_body_nucleus] = 4
    
    return tile1_mask, tile2_mask


def analyze_boundary_artifacts(original_tile1, original_tile2, final_tile1, final_tile2, overlap_length=20):
    """
    Analyze the final merged tiles for boundary artifacts.
    
    Parameters
    ----------
    original_tile1, original_tile2 : np.ndarray
        Original tile masks before merging.
    final_tile1, final_tile2 : np.ndarray
        Final tile masks after merging.
    overlap_length : int
        Length of the overlap region.
        
    Returns
    -------
    dict
        Analysis results including artifact detection.
    """
    results = {
        "boundary_artifacts_detected": False,
        "missing_masks_in_bodies": False,
        "cropped_overlap_regions": False,
        "seamless_boundary": True,
        "details": []
    }
    
    # CRITICAL FIX: Check for missing COVERAGE in tile bodies, not missing original IDs.
    # After merging, original mask IDs may change, but the spatial coverage should be preserved.
    # For "right" relationship: tile1 body is columns 20-99, tile2 body is columns 0-79.

    # Tile1 body region (columns 20-99).
    tile1_body_region = slice(None), slice(overlap_length, None)
    original_tile1_body_coverage = (original_tile1[tile1_body_region] > 0)
    final_tile1_body_coverage = (final_tile1[tile1_body_region] > 0)

    # Check if any pixels that were originally covered are now uncovered.
    lost_coverage_tile1 = original_tile1_body_coverage & ~final_tile1_body_coverage
    if np.any(lost_coverage_tile1):
        results["missing_masks_in_bodies"] = True
        results["seamless_boundary"] = False
        lost_pixels = np.sum(lost_coverage_tile1)
        results["details"].append(f"Lost {lost_pixels} pixels of coverage in tile1 body")

    # Tile2 body region (columns 0-79).
    tile2_body_region = slice(None), slice(None, -overlap_length)
    original_tile2_body_coverage = (original_tile2[tile2_body_region] > 0)
    final_tile2_body_coverage = (final_tile2[tile2_body_region] > 0)

    # Check if any pixels that were originally covered are now uncovered.
    lost_coverage_tile2 = original_tile2_body_coverage & ~final_tile2_body_coverage
    if np.any(lost_coverage_tile2):
        results["missing_masks_in_bodies"] = True
        results["seamless_boundary"] = False
        lost_pixels = np.sum(lost_coverage_tile2)
        results["details"].append(f"Lost {lost_pixels} pixels of coverage in tile2 body")
    
    # Check for visible boundary lines by examining the overlap regions.
    # Tile1 overlap region (columns 0-19).
    tile1_overlap_region = final_tile1[:, :overlap_length]
    tile1_overlap_masks = set(np.unique(tile1_overlap_region)) - {0}

    # Tile2 overlap region (columns 80-99).
    tile2_overlap_region = final_tile2[:, -overlap_length:]
    tile2_overlap_masks = set(np.unique(tile2_overlap_region)) - {0}
    
    # Check if overlap regions are empty (indicating cropping).
    if len(tile1_overlap_masks) == 0 and len(tile2_overlap_masks) == 0:
        results["cropped_overlap_regions"] = True
        results["seamless_boundary"] = False
        results["details"].append("Both overlap regions are empty - masks may have been cropped")
    
    # Check for boundary artifacts by looking for abrupt mask terminations.
    # This is a simplified check - in practice, you'd look for edge discontinuities.
    boundary_col_tile1 = overlap_length  # Column 20 in tile1
    boundary_col_tile2 = -overlap_length - 1  # Column 79 in tile2
    
    # Check if masks abruptly terminate at the boundary.
    tile1_boundary_masks = set(np.unique(final_tile1[:, boundary_col_tile1])) - {0}
    tile2_boundary_masks = set(np.unique(final_tile2[:, boundary_col_tile2])) - {0}
    
    if len(tile1_boundary_masks) == 0 or len(tile2_boundary_masks) == 0:
        results["boundary_artifacts_detected"] = True
        results["seamless_boundary"] = False
        results["details"].append("Potential boundary artifacts detected - masks terminate abruptly at boundary")
    
    return results


def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)


def test_spatial_coordinate_mapping():
    """Test spatial coordinate mapping in mask redistribution."""
    logging.info("=== SPATIAL COORDINATE MAPPING TEST ===")
    
    # Create test masks with boundary-spanning nuclei.
    tile1_mask, tile2_mask = create_boundary_spanning_masks()
    
    logging.info("Created boundary-spanning test masks:")
    logging.info(f"  Tile1 unique masks: {sorted(np.unique(tile1_mask)[1:])}")
    logging.info(f"  Tile2 unique masks: {sorted(np.unique(tile2_mask)[1:])}")
    
    # Analyze original mask distributions.
    overlap_length = 20
    
    # Check which masks span into overlap regions.
    tile1_overlap_region = tile1_mask[:, :overlap_length]
    tile2_overlap_region = tile2_mask[:, -overlap_length:]
    
    tile1_overlap_masks = set(np.unique(tile1_overlap_region)) - {0}
    tile2_overlap_masks = set(np.unique(tile2_overlap_region)) - {0}
    
    logging.info(f"Original overlap analysis:")
    logging.info(f"  Tile1 masks in overlap region: {sorted(tile1_overlap_masks)}")
    logging.info(f"  Tile2 masks in overlap region: {sorted(tile2_overlap_masks)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1_boundary_test.npz"
        tile2_path = temp_path / "tile2_boundary_test.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        try:
            # Test the merge with detailed logging.
            result1, result2, stats = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",
                overlap_length=overlap_length,
                overlap_threshold=0.3
            )
            
            logging.info(f"Merge completed. Stats: {stats}")
            
            # Analyze results for boundary artifacts.
            analysis = analyze_boundary_artifacts(tile1_mask, tile2_mask, result1, result2, overlap_length)
            
            logging.info("=== BOUNDARY ARTIFACT ANALYSIS ===")
            logging.info(f"  Seamless boundary: {analysis['seamless_boundary']}")
            logging.info(f"  Boundary artifacts detected: {analysis['boundary_artifacts_detected']}")
            logging.info(f"  Missing masks in bodies: {analysis['missing_masks_in_bodies']}")
            logging.info(f"  Cropped overlap regions: {analysis['cropped_overlap_regions']}")
            
            if analysis["details"]:
                logging.info("  Details:")
                for detail in analysis["details"]:
                    logging.info(f"    - {detail}")
            
            # Detailed mask analysis.
            final_tile1_masks = sorted(np.unique(result1)[1:])
            final_tile2_masks = sorted(np.unique(result2)[1:])
            
            logging.info(f"Final mask distribution:")
            logging.info(f"  Tile1 final masks: {final_tile1_masks}")
            logging.info(f"  Tile2 final masks: {final_tile2_masks}")
            
            # Check for successful redistribution.
            if analysis["seamless_boundary"]:
                logging.info("✅ SPATIAL COORDINATE MAPPING TEST PASSED")
                logging.info("   No boundary artifacts detected - masks properly redistributed")
            else:
                logging.error("❌ SPATIAL COORDINATE MAPPING TEST FAILED")
                logging.error("   Boundary artifacts detected - redistribution logic has bugs")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Spatial coordinate mapping test failed: {e}")
            logging.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    # Configure detailed logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        success = test_spatial_coordinate_mapping()
        if success:
            print("\n✅ Spatial coordinate mapping test passed!")
        else:
            print("\n❌ Spatial coordinate mapping test failed!")
        
    except Exception as e:
        print(f"\n❌ Spatial coordinate mapping test failed: {e}")
        print(traceback.format_exc())

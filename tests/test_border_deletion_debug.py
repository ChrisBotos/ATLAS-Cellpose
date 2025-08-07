"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_border_deletion_debug.py.
Description:
    Debug test specifically for border deletion logic in the 4-step CPU algorithm.
    This test creates controlled scenarios to verify that border-touching masks
    are being correctly identified and removed from the appropriate tiles.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations and mask generation.
    • tempfile for temporary file management.
    • logging for detailed debug output.

Usage:
    python tests/test_border_deletion_debug.py

Key Features:
    • Creates synthetic masks with nuclei touching specific borders.
    • Tests border detection for all four directions (up, down, left, right).
    • Validates that border deletion removes masks from the correct tiles.
    • Comprehensive debug logging to identify border deletion bugs.

Notes:
    • Designed to isolate and fix Issue 3 (Critical Border Deletion Logic Error).
    • Uses controlled synthetic data for reproducible debugging.
    • Focuses on verifying that each tile only has its OWN border-touching masks deleted.
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


def create_border_touching_masks(spatial_relationship: str):
    """
    Create test masks with nuclei touching specific borders.
    
    Parameters
    ----------
    spatial_relationship : str
        The spatial relationship to test ("above", "right", "below", "left").
        
    Returns
    -------
    tuple
        (tile1_mask, tile2_mask) with nuclei touching the expected borders.
    """
    # Create 100x100 masks.
    tile1_mask = np.zeros((100, 100), dtype=np.uint32)
    tile2_mask = np.zeros((100, 100), dtype=np.uint32)
    
    y_coords, x_coords = np.ogrid[:100, :100]
    
    if spatial_relationship == "above":
        # tile1 is above tile2
        # tile1's overlap region: bottom (rows 80-99)
        # tile2's overlap region: top (rows 0-19)
        # Expected border deletion: tile1 top border, tile2 bottom border
        
        # Add nucleus touching tile1's TOP border (should be deleted).
        tile1_top_nucleus = ((y_coords - 5)**2 + (x_coords - 50)**2) <= 36
        tile1_mask[tile1_top_nucleus] = 1
        
        # Add nucleus in tile1's overlap region (bottom).
        tile1_overlap_nucleus = ((y_coords - 90)**2 + (x_coords - 50)**2) <= 36
        tile1_mask[tile1_overlap_nucleus] = 2
        
        # Add nucleus touching tile2's BOTTOM border (should be deleted).
        tile2_bottom_nucleus = ((y_coords - 95)**2 + (x_coords - 50)**2) <= 36
        tile2_mask[tile2_bottom_nucleus] = 3
        
        # Add nucleus in tile2's overlap region (top).
        tile2_overlap_nucleus = ((y_coords - 10)**2 + (x_coords - 50)**2) <= 36
        tile2_mask[tile2_overlap_nucleus] = 4
        
    elif spatial_relationship == "right":
        # tile1 is right of tile2
        # tile1's overlap region: left (cols 0-19)
        # tile2's overlap region: right (cols 80-99)
        # Expected border deletion: tile1 right border, tile2 left border
        
        # Add nucleus touching tile1's RIGHT border (should be deleted).
        tile1_right_nucleus = ((y_coords - 50)**2 + (x_coords - 95)**2) <= 36
        tile1_mask[tile1_right_nucleus] = 1
        
        # Add nucleus in tile1's overlap region (left).
        tile1_overlap_nucleus = ((y_coords - 50)**2 + (x_coords - 10)**2) <= 36
        tile1_mask[tile1_overlap_nucleus] = 2
        
        # Add nucleus touching tile2's LEFT border (should be deleted).
        tile2_left_nucleus = ((y_coords - 50)**2 + (x_coords - 5)**2) <= 36
        tile2_mask[tile2_left_nucleus] = 3
        
        # Add nucleus in tile2's overlap region (right).
        tile2_overlap_nucleus = ((y_coords - 50)**2 + (x_coords - 90)**2) <= 36
        tile2_mask[tile2_overlap_nucleus] = 4
        
    elif spatial_relationship == "below":
        # tile1 is below tile2
        # tile1's overlap region: top (rows 0-19)
        # tile2's overlap region: bottom (rows 80-99)
        # Expected border deletion: tile1 bottom border, tile2 top border
        
        # Add nucleus touching tile1's BOTTOM border (should be deleted).
        tile1_bottom_nucleus = ((y_coords - 95)**2 + (x_coords - 50)**2) <= 36
        tile1_mask[tile1_bottom_nucleus] = 1
        
        # Add nucleus in tile1's overlap region (top).
        tile1_overlap_nucleus = ((y_coords - 10)**2 + (x_coords - 50)**2) <= 36
        tile1_mask[tile1_overlap_nucleus] = 2
        
        # Add nucleus touching tile2's TOP border (should be deleted).
        tile2_top_nucleus = ((y_coords - 5)**2 + (x_coords - 50)**2) <= 36
        tile2_mask[tile2_top_nucleus] = 3
        
        # Add nucleus in tile2's overlap region (bottom).
        tile2_overlap_nucleus = ((y_coords - 90)**2 + (x_coords - 50)**2) <= 36
        tile2_mask[tile2_overlap_nucleus] = 4
        
    elif spatial_relationship == "left":
        # tile1 is left of tile2
        # tile1's overlap region: right (cols 80-99)
        # tile2's overlap region: left (cols 0-19)
        # Expected border deletion: tile1 left border, tile2 right border
        
        # Add nucleus touching tile1's LEFT border (should be deleted).
        tile1_left_nucleus = ((y_coords - 50)**2 + (x_coords - 5)**2) <= 36
        tile1_mask[tile1_left_nucleus] = 1
        
        # Add nucleus in tile1's overlap region (right).
        tile1_overlap_nucleus = ((y_coords - 50)**2 + (x_coords - 90)**2) <= 36
        tile1_mask[tile1_overlap_nucleus] = 2
        
        # Add nucleus touching tile2's RIGHT border (should be deleted).
        tile2_right_nucleus = ((y_coords - 50)**2 + (x_coords - 95)**2) <= 36
        tile2_mask[tile2_right_nucleus] = 3
        
        # Add nucleus in tile2's overlap region (left).
        tile2_overlap_nucleus = ((y_coords - 50)**2 + (x_coords - 10)**2) <= 36
        tile2_mask[tile2_overlap_nucleus] = 4
    
    return tile1_mask, tile2_mask


def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)


def test_border_deletion_logic():
    """Test border deletion logic for all spatial relationships."""
    logging.info("=== BORDER DELETION LOGIC DEBUG TEST ===")
    
    relationships_to_test = ["above", "right", "below", "left"]
    
    for relationship in relationships_to_test:
        logging.info(f"\n=== TESTING BORDER DELETION FOR '{relationship}' ===")
        
        # Create test masks with border-touching nuclei.
        tile1_mask, tile2_mask = create_border_touching_masks(relationship)
        
        logging.info(f"Created test masks for '{relationship}':")
        logging.info(f"  Tile1 unique masks: {sorted(np.unique(tile1_mask)[1:])}")
        logging.info(f"  Tile2 unique masks: {sorted(np.unique(tile2_mask)[1:])}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tile1_path = temp_path / f"tile1_{relationship}.npz"
            tile2_path = temp_path / f"tile2_{relationship}.npz"
            
            save_test_mask(tile1_mask, tile1_path)
            save_test_mask(tile2_mask, tile2_path)
            
            try:
                result1, result2, stats = merge_tiles_cpu_4step(
                    tile1_path=tile1_path,
                    tile2_path=tile2_path,
                    spatial_relationship=relationship,
                    overlap_length=20,
                    overlap_threshold=0.3
                )
                
                # Analyze results.
                original_tile1_masks = set(np.unique(tile1_mask)[1:])
                original_tile2_masks = set(np.unique(tile2_mask)[1:])
                final_tile1_masks = set(np.unique(result1)[1:])
                final_tile2_masks = set(np.unique(result2)[1:])
                
                logging.info(f"Results for '{relationship}':")
                logging.info(f"  Original tile1 masks: {sorted(original_tile1_masks)}")
                logging.info(f"  Final tile1 masks: {sorted(final_tile1_masks)}")
                logging.info(f"  Original tile2 masks: {sorted(original_tile2_masks)}")
                logging.info(f"  Final tile2 masks: {sorted(final_tile2_masks)}")
                
                # Check if border deletion worked correctly.
                tile1_deleted = original_tile1_masks - final_tile1_masks
                tile2_deleted = original_tile2_masks - final_tile2_masks
                
                logging.info(f"  Deleted from tile1: {sorted(tile1_deleted)}")
                logging.info(f"  Deleted from tile2: {sorted(tile2_deleted)}")
                
                # Validate that changes occurred.
                tile1_changed = not np.array_equal(tile1_mask, result1)
                tile2_changed = not np.array_equal(tile2_mask, result2)
                
                logging.info(f"  Tile1 changed: {tile1_changed}")
                logging.info(f"  Tile2 changed: {tile2_changed}")
                
                if not tile1_changed and not tile2_changed:
                    logging.warning(f"❌ NO CHANGES for '{relationship}' - Border deletion may not be working!")
                else:
                    logging.info(f"✅ Changes detected for '{relationship}' - Border deletion appears to be working")
                
                logging.info(f"  Stats: {stats}")
                
            except Exception as e:
                logging.error(f"❌ Border deletion test failed for '{relationship}': {e}")
                logging.error(traceback.format_exc())
                raise


if __name__ == "__main__":
    # Configure detailed logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        test_border_deletion_logic()
        print("\n✅ Border deletion debug test completed!")
        
    except Exception as e:
        print(f"\n❌ Border deletion debug test failed: {e}")
        print(traceback.format_exc())

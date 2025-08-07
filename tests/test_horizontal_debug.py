"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_horizontal_debug.py.
Description:
    Debug test specifically for horizontal merging issues in the 4-step CPU algorithm.
    This test creates controlled scenarios to identify why horizontal tile merging
    is not working correctly compared to vertical merging.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations and mask generation.
    • tempfile for temporary file management.
    • logging for detailed debug output.

Usage:
    python tests/test_horizontal_debug.py

Key Features:
    • Creates synthetic horizontal tile pairs with known overlap patterns.
    • Tests both left-right and right-left spatial relationships.
    • Comprehensive debug logging to identify the root cause of merging failures.
    • Validates overlap detection, border filtering, and mask redistribution.

Notes:
    • Designed to isolate and fix horizontal merging bugs.
    • Uses controlled synthetic data for reproducible debugging.
    • Focuses on spatial relationship parameter conversion and direction mapping.
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


def create_horizontal_test_masks():
    """
    Create test masks specifically for horizontal merging scenarios.
    
    Returns
    -------
    tuple
        (tile1_mask, tile2_mask) where tile1 is to the RIGHT of tile2.
        Both masks have nuclei in their overlap regions.
    """
    # Create 100x100 masks.
    tile1_mask = np.zeros((100, 100), dtype=np.uint32)
    tile2_mask = np.zeros((100, 100), dtype=np.uint32)
    
    # For "right" relationship (tile1 right of tile2):
    # - tile1's overlap region is its LEFT side (columns 0-19)
    # - tile2's overlap region is its RIGHT side (columns 80-99)
    
    # Add nucleus in tile1's overlap region (left side).
    # Create circular nucleus at (50, 10) with radius 8.
    y_coords, x_coords = np.ogrid[:100, :100]
    tile1_nucleus = ((y_coords - 50)**2 + (x_coords - 10)**2) <= 64  # radius 8
    tile1_mask[tile1_nucleus] = 1
    
    # Add nucleus in tile2's overlap region (right side).
    # Create circular nucleus at (50, 90) with radius 8.
    tile2_nucleus = ((y_coords - 50)**2 + (x_coords - 90)**2) <= 64  # radius 8
    tile2_mask[tile2_nucleus] = 2
    
    # Add non-overlapping nuclei for comparison.
    # Tile1 non-overlap nucleus at (25, 50).
    tile1_non_overlap = ((y_coords - 25)**2 + (x_coords - 50)**2) <= 36  # radius 6
    tile1_mask[tile1_non_overlap] = 3
    
    # Tile2 non-overlap nucleus at (75, 50).
    tile2_non_overlap = ((y_coords - 75)**2 + (x_coords - 50)**2) <= 36  # radius 6
    tile2_mask[tile2_non_overlap] = 4
    
    return tile1_mask, tile2_mask


def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)


def test_horizontal_merging_debug():
    """Test horizontal merging with comprehensive debugging."""
    logging.info("=== HORIZONTAL MERGING DEBUG TEST ===")
    
    # Create test masks.
    tile1_mask, tile2_mask = create_horizontal_test_masks()
    
    logging.info(f"Created test masks:")
    logging.info(f"  Tile1 unique masks: {sorted(np.unique(tile1_mask)[1:])}")  # Exclude 0
    logging.info(f"  Tile2 unique masks: {sorted(np.unique(tile2_mask)[1:])}")  # Exclude 0
    
    # Check overlap regions manually.
    overlap_length = 20
    tile1_left_region = tile1_mask[:, :overlap_length]  # Left 20 columns
    tile2_right_region = tile2_mask[:, -overlap_length:]  # Right 20 columns
    
    tile1_overlap_masks = set(int(label) for label in np.unique(tile1_left_region) if label > 0)
    tile2_overlap_masks = set(int(label) for label in np.unique(tile2_right_region) if label > 0)
    
    logging.info(f"Manual overlap detection:")
    logging.info(f"  Tile1 left region masks: {sorted(tile1_overlap_masks)}")
    logging.info(f"  Tile2 right region masks: {sorted(tile2_overlap_masks)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1_right.npz"
        tile2_path = temp_path / "tile2_left.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        logging.info(f"Saved test masks to: {tile1_path} and {tile2_path}")
        
        # Test horizontal merge with "right" relationship.
        logging.info("=== TESTING 'right' SPATIAL RELATIONSHIP ===")
        try:
            result1, result2, stats = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",  # tile1 is to the right of tile2
                overlap_length=overlap_length,
                overlap_threshold=0.3
            )
            
            logging.info(f"Horizontal merge completed successfully!")
            logging.info(f"Stats: {stats}")
            
            # Validate results.
            final_tile1_masks = sorted(np.unique(result1)[1:])
            final_tile2_masks = sorted(np.unique(result2)[1:])
            
            logging.info(f"Final results:")
            logging.info(f"  Tile1 final masks: {final_tile1_masks}")
            logging.info(f"  Tile2 final masks: {final_tile2_masks}")
            
            # Check if any changes occurred.
            tile1_changed = not np.array_equal(tile1_mask, result1)
            tile2_changed = not np.array_equal(tile2_mask, result2)
            
            logging.info(f"  Tile1 changed: {tile1_changed}")
            logging.info(f"  Tile2 changed: {tile2_changed}")
            
            if not tile1_changed and not tile2_changed:
                logging.warning("NO CHANGES DETECTED - This indicates a bug!")
            
        except Exception as e:
            logging.error(f"Horizontal merge failed: {e}")
            logging.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    # Configure detailed logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        test_horizontal_merging_debug()
        print("Horizontal merging debug test completed!")
        
    except Exception as e:
        print(f"Debug test failed: {e}")
        print(traceback.format_exc())

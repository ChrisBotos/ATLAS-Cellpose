"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_fixed_merging.py.
Description:
    Test the fixed merge_tiles_cpu_3step function using real tile data from the
    cellpose3 environment to ensure it works correctly.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • logging for debug output.

Key Features:
    • Uses real tile data from recent cellpose3 run.
    • Tests the fixed 3-step merging algorithm.
    • Validates that merging completes without errors.
    • Provides comprehensive debug output.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fixed_merging_with_real_data():
    """
    Test the fixed merge_tiles_cpu_3step function with real tile data.
    """
    try:
        # Import the fixed function.
        from code.nuclei_segmentation.cellpose_merge.rules import merge_tiles_cpu_3step
        
        logging.info("Successfully imported merge_tiles_cpu_3step")
        
        # Use real tile data from the cellpose3 run.
        base_dir = Path("results/20250725_165526_cropped_ss_bIRI2_cpu-merge_cellpose3_diameter0/masks/tile_masks_npz")
        
        if not base_dir.exists():
            logging.error(f"Test data directory not found: {base_dir}")
            return False
        
        # Test with two adjacent tiles that should have overlap.
        tile1_path = base_dir / "0_0.npz"
        tile2_path = base_dir / "0_410.npz"
        
        if not tile1_path.exists() or not tile2_path.exists():
            logging.error(f"Required test tiles not found: {tile1_path}, {tile2_path}")
            return False
        
        # Load and inspect the tiles first.
        tile1_data = np.load(tile1_path)
        tile2_data = np.load(tile2_path)
        tile1_mask = tile1_data["mask"]
        tile2_mask = tile2_data["mask"]
        
        logging.info(f"Tile1 ({tile1_path.name}): shape={tile1_mask.shape}, nuclei={len(np.unique(tile1_mask[tile1_mask > 0]))}")
        logging.info(f"Tile2 ({tile2_path.name}): shape={tile2_mask.shape}, nuclei={len(np.unique(tile2_mask[tile2_mask > 0]))}")
        
        # Test parameters based on the actual configuration.
        overlap_length = 102  # From the log: overlap: 102 pixels
        tile_relationship = "tile1_left_of_tile2"  # tile1 is to the left of tile2
        
        logging.info(f"Testing merge with parameters:")
        logging.info(f"  Overlap length: {overlap_length}")
        logging.info(f"  Relationship: {tile_relationship}")
        
        # Call the merge function.
        updated_tile1, updated_tile2, mapping = merge_tiles_cpu_3step(
            tile1_path=tile1_path,
            tile2_path=tile2_path,
            overlap_length=overlap_length,
            tile_relationship=tile_relationship
        )
        
        logging.info("Merge function completed successfully!")
        logging.info(f"Results:")
        logging.info(f"  Updated tile1 shape: {updated_tile1.shape}")
        logging.info(f"  Updated tile2 shape: {updated_tile2.shape}")
        logging.info(f"  Mapping entries: {len(mapping)}")
        logging.info(f"  Tile1 nuclei: {len(np.unique(updated_tile1[updated_tile1 > 0]))}")
        logging.info(f"  Tile2 nuclei: {len(np.unique(updated_tile2[updated_tile2 > 0]))}")
        
        # Validate results.
        assert updated_tile1.shape == tile1_mask.shape, f"Tile1 shape changed: {updated_tile1.shape} vs {tile1_mask.shape}"
        assert updated_tile2.shape == tile2_mask.shape, f"Tile2 shape changed: {updated_tile2.shape} vs {tile2_mask.shape}"
        assert updated_tile1.dtype == np.uint32, f"Tile1 wrong dtype: {updated_tile1.dtype}"
        assert updated_tile2.dtype == np.uint32, f"Tile2 wrong dtype: {updated_tile2.dtype}"
        
        return True
        
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_tile_handling():
    """
    Test that the merge function handles empty tiles correctly.
    """
    try:
        from code.nuclei_segmentation.cellpose_merge.rules import merge_tiles_cpu_3step
        
        logging.info("Testing empty tile handling...")
        
        # Create temporary directory.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create one normal tile and one empty tile.
            tile1_mask = np.zeros((512, 512), dtype=np.uint32)
            tile1_mask[100:150, 100:150] = 1  # Add one nucleus.
            tile1_mask[200:250, 200:250] = 2  # Add another nucleus.
            
            tile2_mask = np.zeros((512, 512), dtype=np.uint32)  # Empty tile.
            
            # Save tiles.
            tile1_path = temp_path / "tile1.npz"
            tile2_path = temp_path / "tile2.npz"
            np.savez(tile1_path, mask=tile1_mask)
            np.savez(tile2_path, mask=tile2_mask)
            
            # Test merge.
            updated_tile1, updated_tile2, mapping = merge_tiles_cpu_3step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                overlap_length=64,
                tile_relationship="tile1_left_of_tile2"
            )
            
            logging.info("Empty tile test completed successfully!")
            logging.info(f"  Tile1 nuclei: {len(np.unique(updated_tile1[updated_tile1 > 0]))}")
            logging.info(f"  Tile2 nuclei: {len(np.unique(updated_tile2[updated_tile2 > 0]))}")
            logging.info(f"  Mapping entries: {len(mapping)}")
            
            return True
            
    except Exception as e:
        logging.error(f"Empty tile test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.info("Testing fixed merge_tiles_cpu_3step function...")
    
    # Test 1: Real data test.
    success1 = test_fixed_merging_with_real_data()
    
    # Test 2: Empty tile test.
    success2 = test_empty_tile_handling()
    
    if success1 and success2:
        logging.info("✅ ALL TESTS PASSED - Merge function is working correctly!")
    else:
        logging.error("❌ SOME TESTS FAILED - Check the errors above!")
        if not success1:
            logging.error("  - Real data test failed")
        if not success2:
            logging.error("  - Empty tile test failed")

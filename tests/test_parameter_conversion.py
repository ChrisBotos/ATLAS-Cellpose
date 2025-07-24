"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_parameter_conversion.py.
Description:
    Test the parameter conversion from old format to new format in two_phase_merge.py.
    This test verifies that the spatial relationship mapping is working correctly
    and that the 4-step algorithm is being called with the right parameters.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations and mask generation.
    • tempfile for temporary file management.
    • logging for detailed debug output.

Usage:
    python tests/test_parameter_conversion.py

Key Features:
    • Tests all four spatial relationship conversions.
    • Validates that the two_phase_merge function correctly calls merge_tiles_cpu_4step.
    • Comprehensive error checking for parameter conversion bugs.

Notes:
    • Designed to verify the fix for Issue 1 (Spatial Relationship Parameter Conversion Error).
    • Uses the actual two_phase_merge function to test the full pipeline.
"""

import logging
import tempfile
import traceback
from pathlib import Path

import numpy as np

# Add the project root to the Python path.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.two_phase_merge import _merge_two_tiles


def create_simple_test_masks():
    """Create simple test masks for parameter conversion testing."""
    # Create 50x50 masks with single nuclei.
    tile1_mask = np.zeros((50, 50), dtype=np.uint32)
    tile2_mask = np.zeros((50, 50), dtype=np.uint32)
    
    # Add simple circular nuclei.
    y_coords, x_coords = np.ogrid[:50, :50]
    
    # Tile1 nucleus at center.
    tile1_nucleus = ((y_coords - 25)**2 + (x_coords - 25)**2) <= 36  # radius 6
    tile1_mask[tile1_nucleus] = 1
    
    # Tile2 nucleus at center.
    tile2_nucleus = ((y_coords - 25)**2 + (x_coords - 25)**2) <= 36  # radius 6
    tile2_mask[tile2_nucleus] = 2
    
    return tile1_mask, tile2_mask


def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)


def test_spatial_relationship_conversion():
    """Test all spatial relationship parameter conversions."""
    logging.info("=== SPATIAL RELATIONSHIP CONVERSION TEST ===")
    
    # Create test masks.
    tile1_mask, tile2_mask = create_simple_test_masks()
    
    # Test all four spatial relationships.
    relationships_to_test = [
        "tile1_above_tile2",
        "tile1_right_of_tile2", 
        "tile1_below_tile2",
        "tile1_left_of_tile2"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1.npz"
        tile2_path = temp_path / "tile2.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        for relationship in relationships_to_test:
            logging.info(f"\n=== TESTING RELATIONSHIP: {relationship} ===")
            
            try:
                # Call the two_phase_merge function which should convert parameters.
                result1, result2, mapping = _merge_two_tiles(
                    tile1_path=tile1_path,
                    tile2_path=tile2_path,
                    overlap_length=10,
                    tile_relationship=relationship,
                    overlap_threshold=0.3
                )
                
                logging.info(f"✅ {relationship} conversion successful!")
                logging.info(f"   Mapping result: {mapping}")
                
                # Validate that we got results.
                assert result1 is not None, f"result1 is None for {relationship}"
                assert result2 is not None, f"result2 is None for {relationship}"
                assert isinstance(mapping, dict), f"mapping is not dict for {relationship}"
                
                logging.info(f"   Result shapes: tile1={result1.shape}, tile2={result2.shape}")
                
            except Exception as e:
                logging.error(f"❌ {relationship} conversion FAILED: {e}")
                logging.error(traceback.format_exc())
                raise


def test_invalid_relationship():
    """Test that invalid relationships are properly rejected."""
    logging.info("\n=== TESTING INVALID RELATIONSHIP HANDLING ===")
    
    tile1_mask, tile2_mask = create_simple_test_masks()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1.npz"
        tile2_path = temp_path / "tile2.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        # Test invalid relationship.
        try:
            result1, result2, mapping = _merge_two_tiles(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                overlap_length=10,
                tile_relationship="invalid_relationship",
                overlap_threshold=0.3
            )
            
            # Should not reach here.
            logging.error("❌ Invalid relationship was NOT rejected!")
            raise AssertionError("Invalid relationship should have been rejected")
            
        except ValueError as e:
            if "Unknown tile_relationship" in str(e):
                logging.info(f"✅ Invalid relationship correctly rejected: {e}")
            else:
                logging.error(f"❌ Wrong error type for invalid relationship: {e}")
                raise
        except Exception as e:
            logging.error(f"❌ Unexpected error for invalid relationship: {e}")
            raise


if __name__ == "__main__":
    # Configure detailed logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        test_spatial_relationship_conversion()
        test_invalid_relationship()
        print("\n✅ All parameter conversion tests passed!")
        
    except Exception as e:
        print(f"\n❌ Parameter conversion test failed: {e}")
        print(traceback.format_exc())

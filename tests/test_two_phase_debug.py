"""Debug test for two_phase_merge integration."""

import logging
import tempfile
import traceback
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.two_phase_merge import _merge_two_tiles

def create_test_masks():
    """Create test masks for two_phase_merge testing."""
    # Create 100x100 masks
    tile1_mask = np.zeros((100, 100), dtype=np.uint32)
    tile2_mask = np.zeros((100, 100), dtype=np.uint32)
    
    # Add nuclei that will be in overlap regions
    tile1_mask[40:60, 10:30] = 1  # Left side of tile1
    tile2_mask[40:60, 70:90] = 2  # Right side of tile2
    
    return tile1_mask, tile2_mask

def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)

def test_two_phase_merge():
    """Test the two_phase_merge function."""
    print("Testing two_phase_merge integration...")
    
    tile1_mask, tile2_mask = create_test_masks()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create storage directory structure
        storage_dir = temp_path / "tile_masks_npz"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masks with coordinate-based names
        tile1_path = storage_dir / "0_0.npz"
        tile2_path = storage_dir / "0_100.npz"  # Adjacent tile
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        print(f"Saved test masks to {storage_dir}")
        
        try:
            # Test the merge function
            result1, result2, mapping = _merge_two_tiles(
                coord1=(0, 0),
                coord2=(0, 1),  # Adjacent coordinate
                tile_h=100,
                tile_w=100,
                overlap_length=20,
                tile_relationship="tile1_left_of_tile2",
                storage_dir=storage_dir,
                overlap_threshold=0.3
            )
            
            print(f"SUCCESS: Two-phase merge completed")
            print(f"Result shapes: {result1.shape}, {result2.shape}")
            print(f"Mapping: {mapping}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Two-phase merge failed: {e}")
            print(traceback.format_exc())
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_two_phase_merge()
    if success:
        print("✅ Two-phase merge test passed")
    else:
        print("❌ Two-phase merge test failed")

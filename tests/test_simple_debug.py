"""Simple debug test to see what's broken."""

import logging
import tempfile
import traceback
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.rules import merge_tiles_cpu_4step

def create_simple_masks():
    """Create test masks that will trigger overlap detection."""
    # Create 50x50 masks with nuclei in overlap regions
    tile1_mask = np.zeros((50, 50), dtype=np.uint32)
    tile2_mask = np.zeros((50, 50), dtype=np.uint32)

    # For "right" relationship with overlap_length=10:
    # - tile1 overlap region: leftmost 10 columns (0-9)
    # - tile2 overlap region: rightmost 10 columns (40-49)

    # Add nucleus in tile1's overlap region (left side)
    tile1_mask[20:30, 5:15] = 1  # Spans columns 5-14, overlaps with region 0-9

    # Add nucleus in tile2's overlap region (right side)
    tile2_mask[20:30, 35:45] = 2  # Spans columns 35-44, overlaps with region 40-49

    return tile1_mask, tile2_mask

def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)

def test_simple_merge():
    """Test very simple merge."""
    print("Creating simple test masks...")
    tile1_mask, tile2_mask = create_simple_masks()
    
    print(f"Tile1 unique: {np.unique(tile1_mask)}")
    print(f"Tile2 unique: {np.unique(tile2_mask)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1.npz"
        tile2_path = temp_path / "tile2.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        print("Saved test masks, calling merge function...")
        
        try:
            result = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",
                overlap_length=10,
                overlap_threshold=0.3
            )
            
            if result is None:
                print("ERROR: Function returned None!")
                return False
                
            result1, result2, stats = result
            print(f"SUCCESS: Got results with shapes {result1.shape}, {result2.shape}")
            print(f"Stats: {stats}")
            return True
            
        except Exception as e:
            print(f"ERROR: Exception occurred: {e}")
            print(traceback.format_exc())
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_simple_merge()
    if success:
        print("✅ Simple test passed")
    else:
        print("❌ Simple test failed")

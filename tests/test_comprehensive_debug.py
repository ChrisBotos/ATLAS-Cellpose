"""Comprehensive debug test to identify what's broken."""

import logging
import tempfile
import traceback
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.nuclei_segmentation.cellpose_merge.rules import merge_tiles_cpu_4step
from code.nuclei_segmentation.cellpose_merge.two_phase_merge import _merge_two_tiles

def create_realistic_masks():
    """Create realistic test masks that should trigger all merge logic."""
    # Create 100x100 masks
    tile1_mask = np.zeros((100, 100), dtype=np.uint32)
    tile2_mask = np.zeros((100, 100), dtype=np.uint32)
    
    # Create circular nuclei using proper coordinates
    y_coords, x_coords = np.ogrid[:100, :100]
    
    # For "right" relationship (tile1 right of tile2) with overlap_length=20:
    # - tile1 overlap region: leftmost 20 columns (0-19)
    # - tile2 overlap region: rightmost 20 columns (80-99)
    
    # Nucleus 1: In tile1, extends into overlap region
    nucleus1 = ((y_coords - 50)**2 + (x_coords - 15)**2) <= 64  # Center at (50,15), radius 8
    tile1_mask[nucleus1] = 1
    
    # Nucleus 2: In tile2, extends into overlap region  
    nucleus2 = ((y_coords - 50)**2 + (x_coords - 85)**2) <= 64  # Center at (50,85), radius 8
    tile2_mask[nucleus2] = 2
    
    # Nucleus 3: In tile1, not in overlap region
    nucleus3 = ((y_coords - 25)**2 + (x_coords - 50)**2) <= 36  # Center at (25,50), radius 6
    tile1_mask[nucleus3] = 3
    
    # Nucleus 4: In tile2, not in overlap region
    nucleus4 = ((y_coords - 75)**2 + (x_coords - 50)**2) <= 36  # Center at (75,50), radius 6
    tile2_mask[nucleus4] = 4
    
    return tile1_mask, tile2_mask

def save_test_mask(mask: np.ndarray, filepath: Path) -> None:
    """Save a test mask to .npz format."""
    np.savez_compressed(filepath, mask=mask)

def test_direct_merge():
    """Test the merge_tiles_cpu_4step function directly."""
    print("\n=== TESTING DIRECT MERGE FUNCTION ===")
    
    tile1_mask, tile2_mask = create_realistic_masks()
    
    print(f"Created masks:")
    print(f"  Tile1 unique: {sorted(np.unique(tile1_mask)[1:])}")
    print(f"  Tile2 unique: {sorted(np.unique(tile2_mask)[1:])}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "tile1.npz"
        tile2_path = temp_path / "tile2.npz"
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        try:
            result1, result2, stats = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",
                overlap_length=20,
                overlap_threshold=0.3
            )
            
            print(f"✅ Direct merge successful!")
            print(f"  Result shapes: {result1.shape}, {result2.shape}")
            print(f"  Final tile1 masks: {sorted(np.unique(result1)[1:])}")
            print(f"  Final tile2 masks: {sorted(np.unique(result2)[1:])}")
            print(f"  Stats: {stats}")
            
            # Check if changes occurred
            tile1_changed = not np.array_equal(tile1_mask, result1)
            tile2_changed = not np.array_equal(tile2_mask, result2)
            print(f"  Changes: tile1={tile1_changed}, tile2={tile2_changed}")
            
            return True
            
        except Exception as e:
            print(f"❌ Direct merge failed: {e}")
            print(traceback.format_exc())
            return False

def test_two_phase_integration():
    """Test the two_phase_merge integration."""
    print("\n=== TESTING TWO-PHASE INTEGRATION ===")
    
    tile1_mask, tile2_mask = create_realistic_masks()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        storage_dir = temp_path / "tile_masks_npz"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with correct pixel coordinate naming
        # For coord (0,0): pixel coord is (0,0)
        # For coord (0,1): pixel coord is (0,80) with tile_w=100, overlap_length=20
        tile1_path = storage_dir / "0_0.npz"
        tile2_path = storage_dir / "0_80.npz"  # Correct pixel coordinate
        
        save_test_mask(tile1_mask, tile1_path)
        save_test_mask(tile2_mask, tile2_path)
        
        try:
            # Create overlap slices for horizontal relationship
            # tile1 (0,0) left of tile2 (0,1)
            overlap_slices = (
                slice(None),           # tile1_slice_y: all rows
                slice(-20, None),      # tile1_slice_x: rightmost 20 columns
                slice(None),           # tile2_slice_y: all rows  
                slice(None, 20)        # tile2_slice_x: leftmost 20 columns
            )
            
            result1, result2, mapping = _merge_two_tiles(
                coord1=(0, 0),
                coord2=(0, 1),
                overlap_slices=overlap_slices,
                storage_dir=storage_dir,
                overlap_length=20,
                tile_h=100,
                tile_w=100,
                overlap_threshold=0.3
            )
            
            print(f"✅ Two-phase integration successful!")
            print(f"  Result shapes: {result1.shape}, {result2.shape}")
            print(f"  Final tile1 masks: {sorted(np.unique(result1)[1:])}")
            print(f"  Final tile2 masks: {sorted(np.unique(result2)[1:])}")
            print(f"  Mapping: {mapping}")
            
            return True
            
        except Exception as e:
            print(f"❌ Two-phase integration failed: {e}")
            print(traceback.format_exc())
            return False

def test_edge_cases():
    """Test edge cases that might cause issues."""
    print("\n=== TESTING EDGE CASES ===")
    
    # Test 1: Empty masks
    print("Testing empty masks...")
    empty1 = np.zeros((50, 50), dtype=np.uint32)
    empty2 = np.zeros((50, 50), dtype=np.uint32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "empty1.npz"
        tile2_path = temp_path / "empty2.npz"
        
        save_test_mask(empty1, tile1_path)
        save_test_mask(empty2, tile2_path)
        
        try:
            result1, result2, stats = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",
                overlap_length=10,
                overlap_threshold=0.3
            )
            print("✅ Empty masks test passed")
        except Exception as e:
            print(f"❌ Empty masks test failed: {e}")
            return False
    
    # Test 2: Single pixel masks
    print("Testing single pixel masks...")
    single1 = np.zeros((50, 50), dtype=np.uint32)
    single2 = np.zeros((50, 50), dtype=np.uint32)
    single1[25, 5] = 1  # In overlap region
    single2[25, 45] = 2  # In overlap region
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tile1_path = temp_path / "single1.npz"
        tile2_path = temp_path / "single2.npz"
        
        save_test_mask(single1, tile1_path)
        save_test_mask(single2, tile2_path)
        
        try:
            result1, result2, stats = merge_tiles_cpu_4step(
                tile1_path=tile1_path,
                tile2_path=tile2_path,
                spatial_relationship="right",
                overlap_length=10,
                overlap_threshold=0.3
            )
            print("✅ Single pixel masks test passed")
        except Exception as e:
            print(f"❌ Single pixel masks test failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("Running comprehensive debug tests...")
    
    success1 = test_direct_merge()
    success2 = test_two_phase_integration()
    success3 = test_edge_cases()
    
    if success1 and success2 and success3:
        print("\n✅ ALL TESTS PASSED - The system is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - Issues detected!")
        
    print("\nIf you're still experiencing issues, please provide:")
    print("1. The specific error message you're seeing")
    print("2. The exact code/command that's failing")
    print("3. The input data characteristics")
    print("4. The expected vs actual behavior")

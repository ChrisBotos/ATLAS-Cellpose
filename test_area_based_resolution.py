"""
Test script to verify area-based conflict resolution for overlapping masks.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import the functions we need to test.
from code.nuclei_segmentation.cellpose_merge.two_phase_merge import (
    merge_tiles_two_phase,
    _save_tile_to_storage
)

def create_conflict_test_tiles():
    """Create tiles with nuclei that will create conflicts to test area-based resolution."""
    # Test parameters.
    image_height, image_width = 400, 400
    tile_h, tile_w = 256, 256
    overlap = 64
    
    # Create temporary directory.
    temp_dir = Path(tempfile.mkdtemp(prefix="conflict_test_"))
    
    # Create 2x2 tiles with designed conflicts.
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for r, c in coords:
        # Create a tile with nuclei of different sizes.
        tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)
        
        if r == 0 and c == 0:  # Top-left tile
            # Large nucleus (400 pixels) - should lose conflicts due to larger area
            tile_mask[50:70, 50:70] = 1  # 20x20 = 400 pixels
            # Small nucleus (100 pixels) - should win conflicts due to smaller area
            tile_mask[150:160, 150:160] = 2  # 10x10 = 100 pixels
            # Medium nucleus extending to overlap region
            tile_mask[100:120, 200:256] = 3  # Extends to right edge
            
        elif r == 0 and c == 1:  # Top-right tile
            # Another large nucleus that will conflict with nucleus 3
            tile_mask[100:120, 0:40] = 4  # Large nucleus from left edge
            # Small nucleus
            tile_mask[50:60, 100:110] = 5  # 10x10 = 100 pixels
            
        elif r == 1 and c == 0:  # Bottom-left tile
            # Medium nucleus
            tile_mask[50:70, 100:120] = 6  # 20x20 = 400 pixels
            # Small nucleus extending to overlap region
            tile_mask[200:256, 150:160] = 7  # Extends to bottom edge
            
        elif r == 1 and c == 1:  # Bottom-right tile
            # Large nucleus that will conflict with nucleus 7
            tile_mask[0:40, 150:160] = 8  # Large nucleus from top edge
            # Another nucleus
            tile_mask[100:120, 100:120] = 9  # 20x20 = 400 pixels
        
        # Save the tile.
        _save_tile_to_storage((r, c), tile_mask, temp_dir, tile_h, tile_w, overlap)
    
    return coords, temp_dir, image_height, image_width, tile_h, tile_w, overlap

def test_area_based_conflict_resolution():
    """Test that area-based conflict resolution works correctly."""
    print("Creating test tiles with designed conflicts...")
    coords, temp_dir, height, width, tile_h, tile_w, overlap = create_conflict_test_tiles()
    
    try:
        # Set up directory structure.
        test_output_dir = temp_dir.parent / "conflict_test_output"
        test_output_dir.mkdir(exist_ok=True)
        
        masks_dir = test_output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        tile_masks_dir = masks_dir / "tile_masks_npz"
        shutil.move(str(temp_dir), str(tile_masks_dir))
        
        print("Running merge with area-based conflict resolution...")
        
        # Run the merge.
        merged_mask = merge_tiles_two_phase(
            coords=coords,
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            debug_mode=True,
            output_dir=test_output_dir
        )
        
        # Check for individual nucleus masks.
        individual_masks_dir = test_output_dir / "masks" / "individual_nucleus_masks"
        if individual_masks_dir.exists():
            mask_files = list(individual_masks_dir.glob("nucleus_*.npy"))
            print(f"✅ Found {len(mask_files)} individual nucleus mask files")
            
            # Load and analyze individual masks.
            nucleus_masks = {}
            nucleus_areas = {}
            for mask_file in mask_files:
                nucleus_id = int(mask_file.stem.split('_')[1])
                nucleus_mask = np.load(mask_file).astype(bool)
                nucleus_masks[nucleus_id] = nucleus_mask
                nucleus_areas[nucleus_id] = np.sum(nucleus_mask)
                print(f"  Nucleus {nucleus_id}: {nucleus_areas[nucleus_id]} pixels")
            
            # Check for conflicts and verify area-based resolution.
            conflicts_found = 0
            conflicts_resolved_correctly = 0
            
            nucleus_ids = sorted(nucleus_masks.keys())
            for i, id1 in enumerate(nucleus_ids):
                for id2 in nucleus_ids[i+1:]:
                    # Check if these two nuclei share any pixels.
                    overlap_mask = nucleus_masks[id1] & nucleus_masks[id2]
                    overlap_count = np.sum(overlap_mask)
                    
                    if overlap_count > 0:
                        conflicts_found += 1
                        
                        # Check which nucleus won the conflict in the merged mask.
                        # Get a sample overlapping pixel.
                        overlap_coords = np.where(overlap_mask)
                        sample_y, sample_x = overlap_coords[0][0], overlap_coords[1][0]
                        winner_id = merged_mask[sample_y, sample_x]
                        
                        # Verify that the smaller nucleus won.
                        smaller_id = id1 if nucleus_areas[id1] < nucleus_areas[id2] else id2
                        
                        if winner_id == smaller_id:
                            conflicts_resolved_correctly += 1
                            print(f"  ✅ Conflict {id1} vs {id2}: Smaller nucleus {smaller_id} won ({nucleus_areas[smaller_id]} < {nucleus_areas[id1 if id1 != smaller_id else id2]} pixels)")
                        else:
                            print(f"  ❌ Conflict {id1} vs {id2}: Wrong winner {winner_id} (should be {smaller_id})")
            
            print(f"\nCONFLICT RESOLUTION ANALYSIS:")
            print(f"  Total conflicts found: {conflicts_found}")
            print(f"  Correctly resolved: {conflicts_resolved_correctly}")
            print(f"  Resolution accuracy: {conflicts_resolved_correctly/conflicts_found*100:.1f}%" if conflicts_found > 0 else "  No conflicts to resolve")
            
            if conflicts_found > 0 and conflicts_resolved_correctly == conflicts_found:
                print("🎉 SUCCESS: Area-based conflict resolution working correctly!")
                print("✅ Smaller nuclei win conflicts as expected")
                return True
            elif conflicts_found == 0:
                print("⚠️  No conflicts found - test may need adjustment")
                return False
            else:
                print("❌ Area-based conflict resolution not working correctly")
                return False
        else:
            print("❌ Individual nucleus masks directory not found")
            return False
        
    finally:
        # Cleanup.
        shutil.rmtree(test_output_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Testing AREA-BASED CONFLICT RESOLUTION...")
    success = test_area_based_conflict_resolution()
    
    if success:
        print("\n🎉 AREA-BASED CONFLICT RESOLUTION TEST PASSED!")
        print("Smaller nuclei now win conflicts and can share pixels with larger nuclei!")
    else:
        print("\n💥 Test failed - area-based resolution not working as expected")

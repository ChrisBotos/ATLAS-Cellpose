"""
Quick test to validate the priority map fix for black border artifacts.
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

def create_simple_test_tiles():
    """Create a simple 2x2 tile scenario that reproduces the black border issue."""
    # Test parameters.
    image_height, image_width = 400, 400
    tile_h, tile_w = 256, 256
    overlap = 64
    stride_h = tile_h - overlap  # 192
    stride_w = tile_w - overlap  # 192

    # Create temporary directory.
    temp_dir = Path(tempfile.mkdtemp(prefix="priority_test_"))

    # Create 4 tiles in a 2x2 grid.
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for r, c in coords:
        # Create a tile with nuclei that extend to the boundaries.
        tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)

        # Add nuclei that extend across tile boundaries to create the gap scenario.
        nucleus_id = 1 + r * 2 + c

        # Create a large nucleus that spans the entire tile.
        # This simulates the scenario where nuclei cross tile boundaries.
        y_coords, x_coords = np.ogrid[:tile_h, :tile_w]

        # Create multiple nuclei across the tile.
        for i in range(3):
            for j in range(3):
                center_y = (i + 1) * tile_h // 4
                center_x = (j + 1) * tile_w // 4
                distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                nucleus_mask = distance <= 15

                # Only place where there's no existing nucleus.
                available_mask = nucleus_mask & (tile_mask == 0)
                if np.any(available_mask):
                    tile_mask[available_mask] = nucleus_id + i * 3 + j

        # CRITICAL: Add nuclei that extend to the tile edges to create boundary scenarios.
        # Add nuclei near the right edge (for tiles in column 0).
        if c == 0:
            edge_nucleus_id = nucleus_id + 100
            right_edge_mask = x_coords >= (tile_w - overlap // 2)
            center_mask = (y_coords >= tile_h // 3) & (y_coords <= 2 * tile_h // 3)
            edge_mask = right_edge_mask & center_mask & (tile_mask == 0)
            if np.any(edge_mask):
                tile_mask[edge_mask] = edge_nucleus_id

        # Add nuclei near the bottom edge (for tiles in row 0).
        if r == 0:
            edge_nucleus_id = nucleus_id + 200
            bottom_edge_mask = y_coords >= (tile_h - overlap // 2)
            center_mask = (x_coords >= tile_w // 3) & (x_coords <= 2 * tile_w // 3)
            edge_mask = bottom_edge_mask & center_mask & (tile_mask == 0)
            if np.any(edge_mask):
                tile_mask[edge_mask] = edge_nucleus_id

        # Save the tile.
        _save_tile_to_storage((r, c), tile_mask, temp_dir, tile_h, tile_w, overlap)

    return coords, temp_dir, image_height, image_width, tile_h, tile_w, overlap

def test_priority_fix():
    """Test that the priority map fix eliminates black borders."""
    print("Creating test tiles...")
    coords, temp_dir, height, width, tile_h, tile_w, overlap = create_simple_test_tiles()
    
    try:
        # Set up directory structure.
        test_output_dir = temp_dir.parent / "priority_test_output"
        test_output_dir.mkdir(exist_ok=True)
        
        masks_dir = test_output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        tile_masks_dir = masks_dir / "tile_masks_npz"
        shutil.move(str(temp_dir), str(tile_masks_dir))
        
        print("Running merge with priority fix...")
        
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
        
        # Check for gaps at stride boundaries.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        print(f"Checking for gaps at stride boundaries...")
        print(f"Stride: {stride_h} x {stride_w}")
        
        gaps_found = 0
        
        # Check vertical boundaries.
        for x in [stride_w]:  # Only check the middle boundary.
            if x < width:
                vertical_line = merged_mask[:, x]
                zero_count = np.sum(vertical_line == 0)
                total_count = len(vertical_line)
                zero_fraction = zero_count / total_count
                
                print(f"Vertical boundary at x={x}: {zero_count}/{total_count} zeros ({zero_fraction:.1%})")
                
                if zero_fraction > 0.8:  # More than 80% zeros indicates a gap.
                    gaps_found += 1
        
        # Check horizontal boundaries.
        for y in [stride_h]:  # Only check the middle boundary.
            if y < height:
                horizontal_line = merged_mask[y, :]
                zero_count = np.sum(horizontal_line == 0)
                total_count = len(horizontal_line)
                zero_fraction = zero_count / total_count
                
                print(f"Horizontal boundary at y={y}: {zero_count}/{total_count} zeros ({zero_fraction:.1%})")
                
                if zero_fraction > 0.8:  # More than 80% zeros indicates a gap.
                    gaps_found += 1
        
        # Report results.
        if gaps_found == 0:
            print("✅ SUCCESS: No black border artifacts detected!")
            print("The priority map fix is working correctly.")
        else:
            print(f"❌ FAILURE: {gaps_found} gap artifacts still detected.")
            print("The fix needs further refinement.")
        
        # Additional statistics.
        total_pixels = merged_mask.size
        zero_pixels = np.sum(merged_mask == 0)
        coverage = ((total_pixels - zero_pixels) / total_pixels) * 100
        
        print(f"Overall coverage: {coverage:.1f}% ({zero_pixels} zero pixels)")
        print(f"Final nuclei count: {len(np.unique(merged_mask[merged_mask > 0]))}")
        
        return gaps_found == 0
        
    finally:
        # Cleanup.
        shutil.rmtree(test_output_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Testing priority map fix for black border artifacts...")
    success = test_priority_fix()
    
    if success:
        print("\n🎉 Test passed! The fix is working.")
    else:
        print("\n💥 Test failed! The fix needs more work.")

"""Debug script to understand the merge behavior."""

import numpy as np
import logging

# Set up logging.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_tiles():
    """Create synthetic tiles with overlapping nuclei for testing."""
    tile1 = np.zeros((256, 256), dtype=np.uint32)
    tile2 = np.zeros((256, 256), dtype=np.uint32)
    
    # Add nucleus in tile1 that extends into overlap region.
    tile1[100:150, 200:256] = 1
    
    # Add overlapping nucleus in tile2.
    tile2[100:150, 0:30] = 2
    
    # Add non-overlapping nucleus in tile2.
    tile2[50:80, 50:80] = 3
    
    return tile1, tile2

def debug_merge():
    """Debug the merge behavior."""
    from code.nuclei_segmentation.cellpose_merge.two_phase_merge import merge_two_tiles
    
    tile1, tile2 = create_synthetic_tiles()
    
    print("Tile 1 nuclei:", np.unique(tile1[tile1 > 0]))
    print("Tile 2 nuclei:", np.unique(tile2[tile2 > 0]))
    
    # Define overlap region (rightmost 64 pixels of tile1, leftmost 64 pixels of tile2).
    overlap_slices = (
        slice(0, 256),    # tile1_y: full height
        slice(192, 256),  # tile1_x: rightmost 64 pixels
        slice(0, 256),    # tile2_y: full height
        slice(0, 64),     # tile2_x: leftmost 64 pixels
    )
    
    print("\nOverlap region analysis:")
    tile1_overlap = tile1[overlap_slices[0], overlap_slices[1]]
    tile2_overlap = tile2[overlap_slices[2], overlap_slices[3]]
    
    print("Tile1 overlap region nuclei:", np.unique(tile1_overlap[tile1_overlap > 0]))
    print("Tile2 overlap region nuclei:", np.unique(tile2_overlap[tile2_overlap > 0]))

    # Check border-touching behavior.
    from code.nuclei_segmentation.cellpose_merge.rules_3step import _find_border_touching_nuclei

    print("\nBorder analysis:")
    tile1_border = _find_border_touching_nuclei(tile1_overlap)
    tile2_border = _find_border_touching_nuclei(tile2_overlap)
    print("Tile1 overlap border-touching:", tile1_border)
    print("Tile2 overlap border-touching:", tile2_border)
    
    # Apply merge.
    updated_tile1, updated_tile2, mapping = merge_two_tiles(
        tile1, tile2, overlap_slices, use_gpu=False
    )
    
    print("\nAfter merge:")
    print("Updated tile1 nuclei:", np.unique(updated_tile1[updated_tile1 > 0]))
    print("Updated tile2 nuclei:", np.unique(updated_tile2[updated_tile2 > 0]))
    print("Mapping:", mapping)
    
    print("\nChanges:")
    print("Tile1 changed:", not np.array_equal(updated_tile1, tile1))
    print("Tile2 changed:", not np.array_equal(updated_tile2, tile2))

if __name__ == "__main__":
    debug_merge()

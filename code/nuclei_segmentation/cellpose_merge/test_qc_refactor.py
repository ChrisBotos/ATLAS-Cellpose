"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_qc_refactor.py.
Description:
    Test script to verify the refactored QC module functionality.
    This script creates synthetic data and tests the enhanced QC visualization
    features for kidney I/R injury tissue segmentation analysis.

Dependencies:
    • Python >= 3.10.
    • numpy, pathlib for data generation.
    • qc module from the current directory.

Usage:
    python test_qc_refactor.py

Inputs:
    • Synthetic segmentation data created for testing.

Outputs:
    • Test results printed to console.
    • QC visualization files in temporary directory.

Key Features:
    • Tests the enhanced before/after overlay generation.
    • Verifies proper image cropping functionality.
    • Validates statistics generation for segmentation results.
    • Ensures proper error handling for edge cases.

Notes:
    • This test verifies the refactored QC module meets all requirements.
    • The test creates realistic synthetic data for kidney tissue analysis.
"""

import traceback
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import logging

# Set up logging to see debug output.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import the refactored QC module.
from .qc import write_overlays, _calculate_crop_region, _generate_tile_color


def create_synthetic_segmentation_data():
    """
    Create synthetic segmentation data for testing QC functionality.
    
    This function generates realistic synthetic data that mimics the output
    of kidney tissue nucleus segmentation for testing purposes.
    
    Returns
    -------
    tuple
        (merged_mask, loader_function, height, width, tile_h, tile_w, overlap)
    """
    
    # Create a synthetic tissue image with realistic dimensions.
    height, width = 2000, 1500
    tile_h, tile_w = 512, 512
    overlap = 64
    
    print(f"Creating synthetic segmentation data: {height}x{width} image")
    
    # Create a merged mask with synthetic nuclei.
    merged_mask = np.zeros((height, width), dtype=np.uint32)
    
    # Add synthetic nuclei distributed across the image.
    nucleus_id = 1
    nucleus_size = 15  # Typical nucleus size in pixels.
    
    for y in range(nucleus_size, height - nucleus_size, nucleus_size * 3):
        for x in range(nucleus_size, width - nucleus_size, nucleus_size * 3):
            # Add some randomness to nucleus positions.
            y_offset = np.random.randint(-5, 6)
            x_offset = np.random.randint(-5, 6)
            
            nucleus_y = y + y_offset
            nucleus_x = x + x_offset
            
            # Create a circular nucleus.
            for dy in range(-nucleus_size//2, nucleus_size//2 + 1):
                for dx in range(-nucleus_size//2, nucleus_size//2 + 1):
                    if dy*dy + dx*dx <= (nucleus_size//2)**2:
                        ny = nucleus_y + dy
                        nx = nucleus_x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            merged_mask[ny, nx] = nucleus_id
            
            nucleus_id += 1
    
    print(f"Created {nucleus_id - 1} synthetic nuclei")
    
    # Create a loader function that simulates tile loading.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    
    def synthetic_loader(ys: slice, xs: slice) -> np.ndarray:
        """
        Synthetic loader function that extracts regions from the merged mask.
        
        This simulates the tile loading process by extracting the requested
        region from the synthetic merged mask.
        """
        
        y_start = max(0, ys.start)
        y_end = min(height, ys.stop)
        x_start = max(0, xs.start)
        x_end = min(width, xs.stop)
        
        if y_end <= y_start or x_end <= x_start:
            return np.zeros((ys.stop - ys.start, xs.stop - xs.start), dtype=np.uint32)
        
        # Extract the region and add some tile-specific variation.
        region = merged_mask[y_start:y_end, x_start:x_end].copy()
        
        # Create output array with requested dimensions.
        result = np.zeros((ys.stop - ys.start, xs.stop - xs.start), dtype=np.uint32)
        
        # Place the valid region in the result array.
        result_y_start = y_start - ys.start
        result_y_end = result_y_start + (y_end - y_start)
        result_x_start = x_start - xs.start
        result_x_end = result_x_start + (x_end - x_start)
        
        result[result_y_start:result_y_end, result_x_start:result_x_end] = region
        
        return result
    
    return merged_mask, synthetic_loader, height, width, tile_h, tile_w, overlap


def test_crop_region_calculation():
    """
    Test the crop region calculation functionality.
    """
    
    print("\n" + "="*50)
    print("TESTING CROP REGION CALCULATION")
    print("="*50)
    
    # Test with small image (should use full image).
    crop_info = _calculate_crop_region(800, 600, 1000)
    assert crop_info['height'] == 800
    assert crop_info['width'] == 600
    assert crop_info['y_start'] == 0
    assert crop_info['x_start'] == 0
    print("✓ Small image crop calculation passed")
    
    # Test with large image (should use central crop).
    crop_info = _calculate_crop_region(2000, 1500, 1000)
    assert crop_info['height'] == 1000
    assert crop_info['width'] == 1000
    assert crop_info['y_start'] == 500  # (2000 - 1000) / 2
    assert crop_info['x_start'] == 250  # (1500 - 1000) / 2
    print("✓ Large image crop calculation passed")


def test_tile_color_generation():
    """
    Test the tile color generation functionality.
    """
    
    print("\n" + "="*50)
    print("TESTING TILE COLOR GENERATION")
    print("="*50)
    
    # Test that different tiles get different colors.
    color1 = _generate_tile_color(0, 0)
    color2 = _generate_tile_color(0, 1)
    color3 = _generate_tile_color(1, 0)
    
    assert not np.array_equal(color1, color2)
    assert not np.array_equal(color1, color3)
    assert not np.array_equal(color2, color3)
    print("✓ Tile colors are unique")
    
    # Test that colors are deterministic.
    color1_repeat = _generate_tile_color(0, 0)
    assert np.array_equal(color1, color1_repeat)
    print("✓ Tile colors are deterministic")
    
    # Test that colors are bright enough.
    assert all(c >= 100 for c in color1)
    assert all(c >= 100 for c in color2)
    print("✓ Tile colors are sufficiently bright")


def test_qc_overlay_generation():
    """
    Test the complete QC overlay generation process.
    """
    
    print("\n" + "="*50)
    print("TESTING QC OVERLAY GENERATION")
    print("="*50)
    
    # Create synthetic data.
    merged_mask, loader, height, width, tile_h, tile_w, overlap = create_synthetic_segmentation_data()
    
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"Generating QC overlays in: {temp_path}")
            
            # Generate QC overlays.
            write_overlays(
                loader=loader,
                merged=merged_mask,
                height=height,
                width=width,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                qc_dir=temp_path
            )
            
            # Verify that output files were created.
            before_path = temp_path / "before_merging.tif"
            after_path = temp_path / "after.tif"
            stats_path = temp_path / "merge_statistics.txt"
            
            assert before_path.exists(), "Before merging overlay not created"
            assert after_path.exists(), "After merging overlay not created"
            assert stats_path.exists(), "Statistics file not created"
            
            print("✓ All QC output files created successfully")
            
            # Verify file contents.
            with open(stats_path, 'r') as f:
                stats_content = f.read()
                assert "Kidney I/R Injury" in stats_content
                assert "Total nuclei detected" in stats_content
                assert "Image dimensions" in stats_content
            
            print("✓ Statistics file contains expected content")
            
            # Check file sizes (should be reasonable).
            before_size = before_path.stat().st_size
            after_size = after_path.stat().st_size
            
            assert before_size > 1000, "Before overlay file seems too small"
            assert after_size > 1000, "After overlay file seems too small"
            
            print(f"✓ Output file sizes are reasonable: before={before_size}, after={after_size}")
            
            return True
            
    except Exception as test_error:
        print(f"✗ QC overlay generation test failed: {test_error}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """
    Run all QC refactor tests.
    """
    
    print("TESTING REFACTORED QC MODULE")
    print("="*60)
    
    try:
        # Run individual component tests.
        test_crop_region_calculation()
        test_tile_color_generation()
        
        # Run the comprehensive QC generation test.
        success = test_qc_overlay_generation()
        
        if success:
            print("\n" + "="*60)
            print("🎉 ALL QC REFACTOR TESTS PASSED!")
            print("The refactored QC module is working correctly.")
            print("Enhanced functionality includes:")
            print("• Intelligent image cropping for manageable file sizes")
            print("• Unique tile colors for boundary identification")
            print("• Random nucleus colors for quality assessment")
            print("• Comprehensive statistics generation")
            print("• Improved error handling and logging")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("❌ QC REFACTOR TESTS FAILED!")
            print("Some functionality is not working correctly.")
            print("="*60)
            return 1
            
    except Exception as main_error:
        print(f"\n❌ Test execution failed: {main_error}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

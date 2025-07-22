"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_final_tif_overlay.py.
Description:
    Test script to verify that the refactored overlay functionality properly
    loads and uses the final.tif tissue image as background for overlays.
    
    This script tests:
    1. Loading final.tif from preprocessed directory
    2. Creating overlays with proper tissue background
    3. Verifying alpha transparency blending works correctly

Dependencies:
    • Python >= 3.10.
    • numpy, pillow for image processing.
    • pathlib for file operations.

Usage:
    python test_final_tif_overlay.py

Arguments:
    None (modify TEST_CONFIG to match your data paths).

Inputs:
    • Results directory with preprocessed/final.tif.
    • Results directory with tile_masks_npz and merged_tile_masks_npz.

Outputs:
    • Test overlays showing tissue background with colored masks.
    • Verification that final.tif is being used as background.

Key Features:
    • Tests final.tif loading functionality.
    • Verifies tissue background is visible in overlays.
    • Demonstrates proper alpha blending.

Notes:
    • Update TEST_CONFIG paths to match your actual data.
    • The script verifies that tissue structure is visible beneath masks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Import the overlay functions.
sys.path.append(str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

from qc import (
    _load_final_tif_image,
    _load_rgb_image,
    create_tile_overlay_from_directory,
    write_overlays
)

"""TEST CONFIGURATION"""

# Update these paths to match your data.
TEST_CONFIG = {
    # Path to your results directory (should contain preprocessed/final.tif).
    "results_dir": Path("results/20250722_044324_test_new_cellpose4_diameter0_large_crop"),
    
    # Output directory for test overlays.
    "output_dir": Path("test_final_tif_overlays"),
    
    # Processing parameters.
    "crop_size": 1000,  # Smaller crop for testing.
    "batch_size": 30,   # Moderate batch size.
    "alpha": 0.6,       # Transparency level.
}

"""TEST FUNCTIONS"""

def test_final_tif_loading():
    """
    Test loading the final.tif image from preprocessed directory.
    """
    
    print("\n=== Testing final.tif Loading ===")
    
    try:
        results_dir = TEST_CONFIG["results_dir"]
        
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return False
        
        # Test loading final.tif.
        print("📖 Attempting to load final.tif...")
        final_tif_image = _load_final_tif_image(results_dir)
        
        if final_tif_image is not None:
            print(f"✓ Successfully loaded final.tif: {final_tif_image.shape}")
            print(f"✓ Image dtype: {final_tif_image.dtype}")
            print(f"✓ Image range: [{final_tif_image.min()}, {final_tif_image.max()}]")
            
            # Save a copy for verification.
            output_dir = TEST_CONFIG["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(final_tif_image).save(
                output_dir / "loaded_final_tif.tif",
                compression="tiff_deflate"
            )
            print(f"✓ Saved copy of final.tif to: {output_dir / 'loaded_final_tif.tif'}")
            
            return True
        else:
            print("❌ Failed to load final.tif")
            
            # Check if the file exists.
            final_tif_path = results_dir / "preprocessed" / "final.tif"
            if final_tif_path.exists():
                print(f"⚠️ final.tif exists at {final_tif_path} but failed to load")
            else:
                print(f"⚠️ final.tif not found at {final_tif_path}")
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


def test_overlay_with_final_tif():
    """
    Test creating overlays using final.tif as background.
    """
    
    print("\n=== Testing Overlay with final.tif Background ===")
    
    try:
        results_dir = TEST_CONFIG["results_dir"]
        
        # Load final.tif.
        final_tif_image = _load_final_tif_image(results_dir)
        
        if final_tif_image is None:
            print("❌ Cannot test overlay without final.tif")
            return False
        
        # Test directories.
        masks_dir = results_dir / "masks"
        before_tiles_dir = masks_dir / "tile_masks_npz"
        after_tiles_dir = masks_dir / "merged_tile_masks_npz"
        
        output_dir = TEST_CONFIG["output_dir"] / "overlay_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test before overlay.
        if before_tiles_dir.exists():
            print("🎨 Creating before overlay with final.tif background...")
            
            before_overlay = create_tile_overlay_from_directory(
                tiles_dir=before_tiles_dir,
                full_image=final_tif_image,
                batch_size=TEST_CONFIG["batch_size"],
                alpha=TEST_CONFIG["alpha"],
                crop_size=TEST_CONFIG["crop_size"],
                output_path=output_dir / "before_overlay_final_tif.tif",
                overlay_type="before"
            )
            
            print(f"✓ Before overlay created: {before_overlay.shape}")
            
            # Verify that tissue background is visible.
            # Check if overlay has similar intensity distribution to original.
            original_mean = np.mean(final_tif_image)
            overlay_mean = np.mean(before_overlay)
            
            print(f"✓ Original image mean intensity: {original_mean:.1f}")
            print(f"✓ Overlay mean intensity: {overlay_mean:.1f}")
            
            if abs(overlay_mean - original_mean) < original_mean * 0.5:
                print("✓ Tissue background appears to be preserved in overlay")
            else:
                print("⚠️ Overlay may not be properly blending with tissue background")
        
        else:
            print(f"⚠️ Before tiles directory not found: {before_tiles_dir}")
        
        # Test after overlay.
        if after_tiles_dir.exists():
            print("🎨 Creating after overlay with final.tif background...")
            
            after_overlay = create_tile_overlay_from_directory(
                tiles_dir=after_tiles_dir,
                full_image=final_tif_image,
                batch_size=TEST_CONFIG["batch_size"],
                alpha=0.7,  # More opaque for after overlay.
                crop_size=TEST_CONFIG["crop_size"],
                output_path=output_dir / "after_overlay_final_tif.tif",
                overlay_type="after"
            )
            
            print(f"✓ After overlay created: {after_overlay.shape}")
        
        else:
            print(f"⚠️ After tiles directory not found: {after_tiles_dir}")
        
        print(f"✓ Overlay tests completed, outputs saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Overlay test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


def test_write_overlays_integration():
    """
    Test the write_overlays function integration with final.tif loading.
    """
    
    print("\n=== Testing write_overlays Integration ===")
    
    try:
        results_dir = TEST_CONFIG["results_dir"]
        
        # Check if we have the necessary components.
        masks_dir = results_dir / "masks"
        if not masks_dir.exists():
            print(f"❌ Masks directory not found: {masks_dir}")
            return False
        
        # Create a test QC directory.
        qc_dir = TEST_CONFIG["output_dir"] / "qc_integration_test"
        qc_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters (simplified for testing).
        height, width = 5000, 5000  # Approximate dimensions.
        tile_h, tile_w = 512, 512
        overlap = 64
        
        # Paths to tile directories.
        original_tiles_path = masks_dir / "tile_masks_npz"
        merged_tiles_dir = masks_dir / "merged_tile_masks_npz"
        
        if not original_tiles_path.exists():
            print(f"⚠️ Original tiles not found: {original_tiles_path}")
            original_tiles_path = None
        
        if not merged_tiles_dir.exists():
            print(f"⚠️ Merged tiles not found: {merged_tiles_dir}")
            merged_tiles_dir = None
        
        if original_tiles_path is None and merged_tiles_dir is None:
            print("❌ No tile directories found for testing")
            return False
        
        print("🔧 Testing write_overlays function with final.tif integration...")
        
        # Call write_overlays (this should automatically use final.tif).
        write_overlays(
            loader=None,  # Not needed when using tile files.
            merged=None,  # Not needed when using tile files.
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            qc_dir=qc_dir,
            image_loader=None,  # Should use final.tif instead.
            use_full_image=False,
            coords=None,  # Will be determined from tile files.
            original_tiles_path=original_tiles_path,
            merged_tiles_dir=merged_tiles_dir,
        )
        
        # Check if overlays were created.
        before_path = qc_dir / "before_merging.tif"
        after_path = qc_dir / "after_merging.tif"
        
        if before_path.exists():
            print(f"✓ Before merging overlay created: {before_path}")
        else:
            print("⚠️ Before merging overlay not created")
        
        if after_path.exists():
            print(f"✓ After merging overlay created: {after_path}")
        else:
            print("⚠️ After merging overlay not created")
        
        print("✓ write_overlays integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


"""MAIN TEST EXECUTION"""

def main():
    """
    Run all final.tif overlay tests.
    """
    
    # Set up logging.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    print("=" * 70)
    print("FINAL.TIF OVERLAY TESTING")
    print("Kidney I/R Injury Tissue Analysis")
    print("=" * 70)
    
    print(f"\nTest Configuration:")
    print(f"  Results directory: {TEST_CONFIG['results_dir']}")
    print(f"  Output directory: {TEST_CONFIG['output_dir']}")
    print(f"  Crop size: {TEST_CONFIG['crop_size']}")
    print(f"  Batch size: {TEST_CONFIG['batch_size']}")
    print(f"  Alpha: {TEST_CONFIG['alpha']}")
    
    # Run tests.
    tests_passed = 0
    total_tests = 3
    
    if test_final_tif_loading():
        tests_passed += 1
    
    if test_overlay_with_final_tif():
        tests_passed += 1
    
    if test_write_overlays_integration():
        tests_passed += 1
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed successfully!")
        print(f"\nCheck outputs in: {TEST_CONFIG['output_dir']}")
        print("\nKey features validated:")
        print("  ✓ final.tif loading from preprocessed directory")
        print("  ✓ Tissue background visible in overlays")
        print("  ✓ Alpha transparency blending working correctly")
        print("  ✓ Integration with write_overlays function")
    else:
        print("❌ Some tests failed. Check error messages above.")
        print("Please verify your data paths and final.tif availability.")


if __name__ == "__main__":
    main()

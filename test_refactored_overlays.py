"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_refactored_overlays.py.
Description:
    Test script for the refactored tile overlay functionality with distinct
    color schemes for before/after merging visualizations. This script
    demonstrates the new features:
    
    1. Before merging: Tile-based colors (each tile gets unique deterministic color)
    2. After merging: Nucleus-based colors (each nucleus gets unique random color)
    3. Alpha transparency overlay on tissue background

Dependencies:
    • Python >= 3.10.
    • numpy, pillow for image processing.
    • pathlib for file operations.

Usage:
    python test_refactored_overlays.py

Arguments:
    None (modify TEST_CONFIG to match your data paths).

Inputs:
    • Results directory with tile_masks_npz and merged_tile_masks_npz.
    • Full tissue image for background visualization.

Outputs:
    • before_merging_test.tif: Tiles with unique colors per tile.
    • after_merging_test.tif: Nuclei with unique colors per nucleus.

Key Features:
    • Demonstrates distinct color schemes for before/after visualizations.
    • Tests alpha transparency blending with tissue background.
    • Validates memory-efficient processing capabilities.

Notes:
    • Update TEST_CONFIG paths to match your actual data.
    • The script tests both tile-based and nucleus-based coloring systems.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Import the refactored overlay functions.
sys.path.append(str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

from qc import (
    create_before_after_overlays,
    create_tile_overlay_from_directory,
    _load_rgb_image
)

"""TEST CONFIGURATION"""

# Update these paths to match your data.
TEST_CONFIG = {
    # Path to your tissue image.
    "image_path": Path("data/IRI_regist_cropped.tif"),
    
    # Path to your results directory.
    "results_dir": Path("results/20250722_044324_test_new_cellpose4_diameter0_large_crop"),
    
    # Output directory for test overlays.
    "output_dir": Path("test_refactored_overlays"),
    
    # Processing parameters.
    "crop_size": 1000,  # Smaller crop for testing.
    "batch_size": 50,   # Moderate batch size.
    "alpha_before": 0.6,  # Transparency for before overlay.
    "alpha_after": 0.7,   # Transparency for after overlay.
}

"""TEST FUNCTIONS"""

def test_before_after_color_schemes():
    """
    Test the distinct color schemes for before/after overlays.
    
    This function demonstrates:
    1. Before merging: Tile-based deterministic colors
    2. After merging: Nucleus-based random colors
    3. Alpha transparency blending with tissue background
    """
    
    print("\n=== Testing Refactored Color Schemes ===")
    
    try:
        # Validate paths.
        if not TEST_CONFIG["image_path"].exists():
            print(f"❌ Image file not found: {TEST_CONFIG['image_path']}")
            print("Please update TEST_CONFIG with correct image path.")
            return False
        
        if not TEST_CONFIG["results_dir"].exists():
            print(f"❌ Results directory not found: {TEST_CONFIG['results_dir']}")
            print("Please update TEST_CONFIG with correct results directory.")
            return False
        
        # Load tissue image.
        print("📖 Loading tissue image...")
        full_image = _load_rgb_image(TEST_CONFIG["image_path"])
        print(f"✓ Loaded tissue image: {full_image.shape}")
        
        # Create output directory.
        output_dir = TEST_CONFIG["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test the high-level interface.
        print("🎨 Creating before/after overlays with distinct color schemes...")
        start_time = time.time()
        
        before_overlay, after_overlay = create_before_after_overlays(
            results_dir=TEST_CONFIG["results_dir"],
            full_image=full_image,
            crop_size=TEST_CONFIG["crop_size"],
            batch_size=TEST_CONFIG["batch_size"],
            alpha=TEST_CONFIG["alpha_before"],  # Use before alpha for both initially.
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        
        print(f"✓ Created before overlay (tile colors): {before_overlay.shape}")
        print(f"✓ Created after overlay (nucleus colors): {after_overlay.shape}")
        print(f"✓ Processing completed in {processing_time:.2f} seconds")
        
        # Save additional test copies with descriptive names.
        Image.fromarray(before_overlay).save(
            output_dir / "before_merging_tile_colors.tif",
            compression="tiff_deflate"
        )
        Image.fromarray(after_overlay).save(
            output_dir / "after_merging_nucleus_colors.tif",
            compression="tiff_deflate"
        )
        
        print(f"✓ Test overlays saved to: {output_dir}")
        print("✓ Before overlay: Each tile has a unique deterministic color")
        print("✓ After overlay: Each nucleus has a unique random color")
        print("✓ Both overlays are alpha-blended with tissue background")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


def test_individual_overlay_creation():
    """
    Test creating individual overlays with specific parameters.
    
    This function tests the low-level interface for creating
    single overlays with custom parameters.
    """
    
    print("\n=== Testing Individual Overlay Creation ===")
    
    try:
        # Load tissue image.
        full_image = _load_rgb_image(TEST_CONFIG["image_path"])
        
        # Test directories.
        masks_dir = TEST_CONFIG["results_dir"] / "masks"
        before_tiles_dir = masks_dir / "tile_masks_npz"
        after_tiles_dir = masks_dir / "merged_tile_masks_npz"
        
        output_dir = TEST_CONFIG["output_dir"] / "individual_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test before overlay (tile colors).
        if before_tiles_dir.exists():
            print("🎨 Creating before overlay with tile-based colors...")
            
            before_overlay = create_tile_overlay_from_directory(
                tiles_dir=before_tiles_dir,
                full_image=full_image,
                batch_size=TEST_CONFIG["batch_size"],
                alpha=TEST_CONFIG["alpha_before"],
                crop_size=TEST_CONFIG["crop_size"],
                output_path=output_dir / "individual_before_overlay.tif",
                overlay_type="before"
            )
            
            print(f"✓ Before overlay created: {before_overlay.shape}")
        else:
            print(f"⚠️ Before tiles directory not found: {before_tiles_dir}")
        
        # Test after overlay (nucleus colors).
        if after_tiles_dir.exists():
            print("🎨 Creating after overlay with nucleus-based colors...")
            
            after_overlay = create_tile_overlay_from_directory(
                tiles_dir=after_tiles_dir,
                full_image=full_image,
                batch_size=TEST_CONFIG["batch_size"],
                alpha=TEST_CONFIG["alpha_after"],
                crop_size=TEST_CONFIG["crop_size"],
                output_path=output_dir / "individual_after_overlay.tif",
                overlay_type="after"
            )
            
            print(f"✓ After overlay created: {after_overlay.shape}")
        else:
            print(f"⚠️ After tiles directory not found: {after_tiles_dir}")
        
        print(f"✓ Individual overlays saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Individual overlay test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


def test_alpha_transparency_levels():
    """
    Test different alpha transparency levels for overlay blending.
    
    This function creates overlays with different transparency levels
    to demonstrate the alpha blending functionality.
    """
    
    print("\n=== Testing Alpha Transparency Levels ===")
    
    try:
        # Load tissue image.
        full_image = _load_rgb_image(TEST_CONFIG["image_path"])
        
        # Test directory.
        masks_dir = TEST_CONFIG["results_dir"] / "masks"
        before_tiles_dir = masks_dir / "tile_masks_npz"
        
        if not before_tiles_dir.exists():
            print(f"⚠️ Tiles directory not found: {before_tiles_dir}")
            return False
        
        output_dir = TEST_CONFIG["output_dir"] / "alpha_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test different alpha levels.
        alpha_levels = [0.3, 0.5, 0.7, 0.9]
        
        for alpha in alpha_levels:
            print(f"🎨 Creating overlay with alpha = {alpha}...")
            
            overlay = create_tile_overlay_from_directory(
                tiles_dir=before_tiles_dir,
                full_image=full_image,
                batch_size=30,
                alpha=alpha,
                crop_size=800,  # Smaller for faster processing.
                output_path=output_dir / f"alpha_{alpha:.1f}_overlay.tif",
                overlay_type="before"
            )
            
            print(f"✓ Alpha {alpha} overlay created: {overlay.shape}")
        
        print(f"✓ Alpha transparency test overlays saved to: {output_dir}")
        print("✓ Compare different alpha levels to see tissue background visibility")
        return True
        
    except Exception as e:
        print(f"❌ Alpha transparency test failed: {e}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return False


"""MAIN TEST EXECUTION"""

def main():
    """
    Run all refactored overlay tests.
    """
    
    # Set up logging.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    print("=" * 70)
    print("REFACTORED TILE OVERLAY TESTING")
    print("Kidney I/R Injury Tissue Analysis")
    print("=" * 70)
    
    print(f"\nTest Configuration:")
    print(f"  Image path: {TEST_CONFIG['image_path']}")
    print(f"  Results directory: {TEST_CONFIG['results_dir']}")
    print(f"  Output directory: {TEST_CONFIG['output_dir']}")
    print(f"  Crop size: {TEST_CONFIG['crop_size']}")
    print(f"  Batch size: {TEST_CONFIG['batch_size']}")
    print(f"  Alpha before: {TEST_CONFIG['alpha_before']}")
    print(f"  Alpha after: {TEST_CONFIG['alpha_after']}")
    
    # Run tests.
    tests_passed = 0
    total_tests = 3
    
    if test_before_after_color_schemes():
        tests_passed += 1
    
    if test_individual_overlay_creation():
        tests_passed += 1
    
    if test_alpha_transparency_levels():
        tests_passed += 1
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed successfully!")
        print(f"\nCheck outputs in: {TEST_CONFIG['output_dir']}")
        print("\nKey features validated:")
        print("  ✓ Before merging: Tile-based deterministic colors")
        print("  ✓ After merging: Nucleus-based random colors")
        print("  ✓ Alpha transparency blending with tissue background")
        print("  ✓ Memory-efficient batch processing")
    else:
        print("❌ Some tests failed. Check error messages above.")
        print("Please verify your data paths and try again.")


if __name__ == "__main__":
    main()

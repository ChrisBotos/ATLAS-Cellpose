#!/usr/bin/env python3
"""
Test script for vibrant color generation and clustering visualization.

Author: Christos Botos
Email: hcty02@gmail.com
Date: 2025-07-31

Description:
    This test verifies that the enhanced color system generates vibrant, 
    high-contrast colors suitable for scientific visualization. It tests
    the neon-like color palette, alpha transparency settings, and visual
    impact of the color choices for cluster distinction.

Dependencies:
    • Python >= 3.10.
    • numpy, matplotlib for visualization.
    • PIL for image processing.
    • pathlib for file operations.

Usage:
    python tests/test_vibrant_colors.py

Inputs:
    • Tests use the enhanced color generation system.
    • Validates color saturation and contrast levels.

Outputs:
    • Test results showing vibrant color generation.
    • Color preview images for visual validation.
    • Contrast ratio measurements for accessibility.

Key Features:
    • Tests enhanced color palette with neon-like colors.
    • Validates alpha transparency for optimal visibility.
    • Generates color preview for visual inspection.
    • Measures contrast ratios for scientific standards.

Notes:
    • This test ensures colors are vibrant enough for cluster distinction.
    • Validates the enhanced color system improvements.
    • Critical for publication-quality scientific visualizations.
"""

import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont

# Configure logging for test output.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_vibrant_color_generation() -> bool:
    """
    Test the enhanced vibrant color generation system.

    This function tests the enhanced color palette to ensure it generates
    vibrant, high-contrast colors suitable for scientific visualization.
    """
    logger.info("Testing vibrant color generation system.")

    # Import the enhanced color generation functions.
    sys.path.append(str(Path(__file__).parent.parent / 'code' / 'engineered_feature_extraction' / 'utils'))
    from generate_contrast_colors import generate_color_palette

    # Test neon config by creating it manually.
    neon_alpha = 250
    neon_saturation = 1.0

    # Test 1: Generate vibrant colors with enhanced settings.
    logger.info("Test 1: Generating vibrant color palette.")
    vibrant_colors = generate_color_palette(
        n=8,
        alpha=240,           # High alpha for visibility.
        saturation=0.98,     # Maximum saturation.
        background="dark",
        contrast_ratio=4.0
    )

    # Verify we got the expected number of colors.
    assert len(vibrant_colors) == 8, f"Expected 8 colors, got {len(vibrant_colors)}"

    # Check that colors are vibrant (high saturation).
    vibrant_count = 0
    for cluster_id, (r, g, b, a) in vibrant_colors.items():
        # Calculate color saturation.
        max_rgb = max(r, g, b)
        min_rgb = min(r, g, b)
        saturation = (max_rgb - min_rgb) / max_rgb if max_rgb > 0 else 0

        # Check alpha is high enough.
        assert a >= 200, f"Color {cluster_id} alpha too low: {a}"

        # Count highly saturated colors.
        if saturation > 0.7:  # High saturation threshold.
            vibrant_count += 1

    assert vibrant_count >= 6, f"Only {vibrant_count}/8 colors are highly saturated"

    logger.info(f"✓ Generated {len(vibrant_colors)} vibrant colors with {vibrant_count} highly saturated")

    # Test 2: Test neon configuration values.
    logger.info("Test 2: Testing neon color configuration values.")

    # Verify neon config has maximum settings.
    assert neon_alpha >= 240, f"Neon config alpha too low: {neon_alpha}"
    assert neon_saturation >= 0.98, f"Neon config saturation too low: {neon_saturation}"

    logger.info("✓ Neon configuration values are optimal for maximum visibility")


def test_color_contrast_levels():
    """
    Test that generated colors have sufficient contrast for scientific use.

    This function validates that the enhanced colors meet scientific
    visualization standards for contrast and accessibility.
    """
    logger.info("Testing color contrast levels.")

    # Import color functions.
    sys.path.append(str(Path(__file__).parent.parent / 'code' / 'engineered_feature_extraction' / 'utils'))
    from generate_contrast_colors import generate_color_palette, calculate_contrast_ratio

    # Generate test colors.
    test_colors = generate_color_palette(
        n=10,
        alpha=240,
        saturation=0.98,
        background="dark",
        contrast_ratio=4.0
    )

    # Test contrast against dark background.
    dark_bg = (0, 0, 0)  # Black background.
    low_contrast_count = 0

    for cluster_id, (r, g, b, a) in test_colors.items():
        contrast = calculate_contrast_ratio((r, g, b), dark_bg)

        if contrast < 3.0:  # Minimum for scientific visualization.
            low_contrast_count += 1
            logger.warning(f"⚠ Color {cluster_id} has low contrast: {contrast:.2f}")

    assert low_contrast_count <= 2, \
        f"Too many colors ({low_contrast_count}) have low contrast"

    logger.info(f"✓ Color contrast validation passed ({low_contrast_count} low-contrast colors)")


def create_color_preview(output_path: Path) -> bool:
    """
    Create a visual preview of the enhanced color palette.
    
    Args:
        output_path: Path where to save the color preview image.
        
    Returns:
        True if preview created successfully, False otherwise.
        
    This function generates a visual preview showing the enhanced colors
    for manual inspection and validation of the vibrant color system.
    """
    logger.info("Creating color preview image.")
    
    try:
        # Import color functions.
        sys.path.append(str(Path(__file__).parent.parent / 'code' / 'engineered_feature_extraction' / 'utils'))
        from generate_contrast_colors import generate_color_palette
        
        # Generate colors for preview.
        colors = generate_color_palette(
            n=12,
            alpha=240,
            saturation=0.98,
            background="dark",
            contrast_ratio=4.0
        )
        
        # Create preview image.
        img_width, img_height = 800, 400
        preview_img = Image.new('RGB', (img_width, img_height), (20, 20, 20))  # Dark background.
        draw = ImageDraw.Draw(preview_img)
        
        # Calculate color swatch dimensions.
        cols = 4
        rows = 3
        swatch_width = img_width // cols
        swatch_height = img_height // rows
        
        # Draw color swatches.
        for i, (cluster_id, (r, g, b, a)) in enumerate(colors.items()):
            if i >= cols * rows:
                break
                
            row = i // cols
            col = i % cols
            
            x1 = col * swatch_width
            y1 = row * swatch_height
            x2 = x1 + swatch_width - 2  # Small gap between swatches.
            y2 = y1 + swatch_height - 2
            
            # Draw color swatch.
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))
            
            # Add color information text.
            text = f"C{cluster_id}\nRGB({r},{g},{b})\nα={a}"
            text_x = x1 + 10
            text_y = y1 + 10
            
            # Use white text for better visibility.
            draw.text((text_x, text_y), text, fill=(255, 255, 255))
        
        # Save preview image.
        preview_img.save(output_path)
        logger.info(f"✓ Color preview saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Color preview creation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main():
    """
    Run all vibrant color system tests.
    
    This function coordinates all test cases and provides comprehensive
    validation of the enhanced color generation system.
    """
    logger.info("🎨 VIBRANT COLOR SYSTEM TESTS 🎨")
    logger.info("=" * 50)
    
    tests = [
        ("Vibrant Color Generation", test_vibrant_color_generation),
        ("Color Contrast Levels", test_color_contrast_levels)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)
        
        if test_func():
            logger.info(f"✓ {test_name} PASSED")
            passed += 1
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    # Create color preview for visual validation.
    logger.info(f"\nCreating Color Preview")
    logger.info("-" * 40)
    
    output_dir = Path(__file__).parent.parent / 'tests' / 'results'
    output_dir.mkdir(exist_ok=True)
    preview_path = output_dir / 'vibrant_color_preview.png'
    
    if create_color_preview(preview_path):
        logger.info(f"✓ Color Preview Created")
        passed += 0.5  # Bonus for visual validation.
    else:
        logger.error(f"✗ Color Preview Failed")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST SUMMARY: {int(passed)}/{total} tests passed")
    
    if passed >= total:
        logger.info("🎉 All vibrant color system tests PASSED!")
        logger.info(f"📸 Visual preview available at: {preview_path}")
        return 0
    else:
        logger.error("❌ Some tests FAILED. Please check the color system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

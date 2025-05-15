#!/usr/bin/env python3
"""
CLAHE Parameter Testing Script

This script applies CLAHE with different parameters to a cropped region of an image
to help find the optimal settings for nuclei segmentation.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io as skio
from PIL import Image

# Increase PIL's maximum image size limit
Image.MAX_IMAGE_PIXELS = 10**9  # 1 billion pixels

def crop_image(image, crop_box):
    """
    Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_box: Tuple of (y0, y1, x0, x1), either relative (0–1) or absolute.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    y0, y1, x0, x1 = crop_box

    if all(0 <= val <= 1 for val in crop_box):
        y0, y1 = int(y0 * h), int(y1 * h)
        x0, x1 = int(x0 * w), int(x1 * w)
        print(f"Cropping with relative bbox: ({y0}:{y1}, {x0}:{x1})")
    else:
        y0, y1, x0, x1 = map(int, crop_box)
        print(f"Cropping with absolute bbox: ({y0}:{y1}, {x0}:{x1})")

    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)

    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop dimensions: y=[{y0}:{y1}], x=[{x0}:{x1}]")

    return image[y0:y1, x0:x1]

def convert_16bit_to_8bit(image):
    """
    Convert a 16-bit image to 8-bit using percentile-based dynamic range scaling.

    Args:
        image (np.ndarray): 16-bit grayscale image.

    Returns:
        np.ndarray: 8-bit grayscale image.
    """
    if image.dtype != np.uint16:
        return image

    p_low, p_high = np.percentile(image, (0.5, 99.5))

    if p_high - p_low <= 0:
        p_low, p_high = image.min(), image.max()
        if p_high - p_low <= 0:
            return np.zeros_like(image, dtype=np.uint8)

    normalized = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    return (normalized * 255).astype(np.uint8)

def apply_clahe(image, clip_limit, tile_grid_size):
    """
    Apply CLAHE to enhance local contrast.

    Args:
        image (np.ndarray): Grayscale image.
        clip_limit (float): CLAHE clip limit.
        tile_grid_size (tuple): Grid size for tiles.

    Returns:
        np.ndarray: CLAHE-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

def main():
    # Configuration
    image_path = input("Enter the path to your image file: ")
    output_dir = Path("clahe_test_results")
    output_dir.mkdir(exist_ok=True)

    # Crop settings
    crop_box = (0.5, 0.52, 0.66, 0.68)  # y_start, y_end, x_start, x_end

    # CLAHE parameter ranges to test
    clip_limits = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    grid_sizes = [(32, 32), (64, 64), (128, 128)]

    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    try:
        image = skio.imread(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Image loaded, shape: {image.shape}, dtype: {image.dtype}")

    # Convert to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] >= 3:
        print("Converting RGB to grayscale...")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert to 8-bit if needed
    if image.dtype == np.uint16:
        print("Converting 16-bit to 8-bit...")
        image = convert_16bit_to_8bit(image)

    # Crop image
    print(f"Cropping image with bbox: {crop_box}...")
    cropped = crop_image(image, crop_box)

    # Save original cropped image
    skio.imsave(output_dir / "original_crop.tif", cropped)
    print(f"Saved original cropped image to {output_dir / 'original_crop.tif'}")

    # Create figure for grid of results
    fig, axes = plt.subplots(len(clip_limits), len(grid_sizes),
                             figsize=(4*len(grid_sizes), 4*len(clip_limits)))

    # Apply CLAHE with different parameters
    for i, clip_limit in enumerate(clip_limits):
        for j, grid_size in enumerate(grid_sizes):
            print(f"Applying CLAHE with clip_limit={clip_limit}, grid_size={grid_size}...")

            # Apply CLAHE
            enhanced = apply_clahe(cropped, clip_limit, grid_size)

            # Save individual result
            output_path = output_dir / f"clahe_clip{clip_limit}_grid{grid_size[0]}x{grid_size[1]}.tif"
            skio.imsave(output_path, enhanced)

            # Add to plot
            ax = axes[i, j]
            ax.imshow(enhanced, cmap='gray')
            ax.set_title(f"Clip: {clip_limit}, Grid: {grid_size}")
            ax.axis('off')

    # Save combined figure
    plt.tight_layout()
    plt.savefig(output_dir / "clahe_parameter_comparison.png", dpi=300)
    plt.savefig(output_dir / "clahe_parameter_comparison.pdf")
    print(f"Saved parameter comparison to {output_dir / 'clahe_parameter_comparison.png'}")

    # Create a gamma-corrected version of the best CLAHE result
    print("Creating gamma-corrected versions...")
    best_clahe = apply_clahe(cropped, 10.0, (64, 64))  # Updated best parameters

    gamma_values = [0.8, 1.0, 1.5, 2.0, 2.5]
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(4*len(gamma_values), 4))

    for i, gamma in enumerate(gamma_values):
        # Apply gamma correction
        gamma_corrected = np.power(best_clahe / 255.0, 1.0 / gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)

        # Save individual result
        output_path = output_dir / f"gamma_correction_{gamma}.tif"
        skio.imsave(output_path, gamma_corrected)

        # Add to plot
        ax = axes[i]
        ax.imshow(gamma_corrected, cmap='gray')
        ax.set_title(f"Gamma: {gamma}")
        ax.axis('off')

    # Save combined gamma figure
    plt.tight_layout()
    plt.savefig(output_dir / "gamma_parameter_comparison.png", dpi=300)
    print(f"Saved gamma comparison to {output_dir / 'gamma_parameter_comparison.png'}")

    print("Done! Check the output directory for results.")

if __name__ == "__main__":
    main()
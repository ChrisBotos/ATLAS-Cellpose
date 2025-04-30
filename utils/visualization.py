"""
Visualization utilities for nuclei segmentation.

This module contains functions for creating visualizations of segmentation results,
including overlays, thumbnails, and debug visualizations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import plot
import logging
from pathlib import Path

def small_segmentation_overlay(output_dir, crop_size=1024):
    """
    Create a small overlay image for quick review.
    
    Extracts a central crop from the segmentation results and creates
    an overlay visualization for quick assessment of segmentation quality.
    
    Args:
        output_dir: Directory containing segmentation results.
        crop_size: Size of the cropped region (pixels).
    """
    logger = logging.getLogger()
    
    # Subdirectory for storing the check images
    check_dir = os.path.join(output_dir, "visualizations", "cropped_preview")
    os.makedirs(check_dir, exist_ok=True)
    logger.info(f"Created visualization directory: {check_dir}")

    # Load images
    pre_path = os.path.join(output_dir, "preprocessed", "preprocessed_image.png")
    clahe_path = os.path.join(output_dir, "preprocessed", "contrast_enhanced_image.png")
    gamma_path = os.path.join(output_dir, "preprocessed", "gamma_corrected_image.png")
    masks_path = os.path.join(output_dir, "masks", "masks.npy")
    watershed_path = os.path.join(output_dir, "masks", "segmentation_mask_post_watershed.npy")
    
    # Load preprocessed image (try different possible paths)
    for img_path in [pre_path, clahe_path, gamma_path]:
        if os.path.exists(img_path):
            try:
                img = skio.imread(img_path)
                logger.info(f"Loaded image for overlay: {img_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load {img_path}: {e}")
    else:
        logger.error("No valid preprocessed image found for overlay")
        return
    
    # Load masks (try both regular and watershed masks)
    if os.path.exists(masks_path):
        try:
            masks = np.load(masks_path)
            logger.info(f"Loaded masks for overlay: {masks_path}")
        except Exception as e:
            logger.error(f"Could not load masks: {e}")
            return
    elif os.path.exists(watershed_path):
        try:
            masks = np.load(watershed_path)
            logger.info(f"Loaded watershed masks for overlay: {watershed_path}")
        except Exception as e:
            logger.error(f"Could not load watershed masks: {e}")
            return
    else:
        logger.error("No valid mask file found for overlay")
        return
    
    # Get central crop
    h, w = img.shape[:2]
    y0, x0 = max(0, h//2 - crop_size//2), max(0, w//2 - crop_size//2)
    y1, x1 = min(h, y0 + crop_size), min(w, x0 + crop_size)
    
    img_crop = img[y0:y1, x0:x1]
    masks_crop = masks[y0:y1, x0:x1]
    
    # Create overlay
    overlay = plot.mask_overlay(img_crop, masks_crop, 
                               colors=np.random.rand(np.max(masks_crop) + 1, 3))
    
    # Save overlay
    overlay_path = os.path.join(check_dir, "central_crop_overlay.png")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved central crop overlay to: {overlay_path}")
    
    # Create figure with both original and overlay
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_crop, cmap='gray')
    ax[0].set_title("Original Image (Central Crop)")
    ax[0].axis('off')
    
    ax[1].imshow(overlay)
    ax[1].set_title("Segmentation Overlay")
    ax[1].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(check_dir, "central_crop_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved comparison figure to: {fig_path}")

def generate_full_overlay(image, masks, flows, output_dir, logger):
    """
    Generate and save overlay visualizations of the full image.
    
    Creates mask overlay and segmentation debug visualizations for the
    entire image and saves them to the output directory.
    
    Args:
        image: Input image (grayscale).
        masks: Segmentation masks.
        flows: Flow fields from Cellpose.
        output_dir: Directory where visualizations will be saved.
        logger: Logger instance for logging.
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations", "full_image")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create mask overlay
    overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))
    overlay_path = os.path.join(vis_dir, "mask_overlay.png")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved full overlay image to: {overlay_path}")

    # Create debug visualization with flows
    fig = plt.figure(figsize=(12, 12))
    plot.show_segmentation(fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])
    debug_path = os.path.join(vis_dir, "segmentation_debug.png")
    fig.savefig(debug_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved segmentation debug overlay to: {debug_path}")
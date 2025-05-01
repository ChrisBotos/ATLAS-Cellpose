"""
Visualization Utilities for Kidney I/R Injury Nuclei Segmentation Analysis.

This module provides specialized visualization functions for assessing segmentation
quality and creating publication-ready figures for kidney tissue analysis. Proper
visualization is critical for validating segmentation results and communicating
findings about nuclear morphology changes during ischemia-reperfusion injury.

The utilities handle various visualization needs including:
1. Creating overlay images that show segmentation boundaries on original images
2. Generating cropped previews for quick quality assessment
3. Producing full-size visualizations for detailed inspection
4. Creating comparison views to evaluate preprocessing and refinement steps

These visualizations help researchers identify segmentation issues such as
under-segmentation (merged nuclei) or over-segmentation (fragmented nuclei),
which is particularly important in densely packed regions of injured kidney tissue.
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
    Create a small overlay image for quick review of kidney tissue segmentation.

    This function extracts a central crop from the segmentation results and creates
    an overlay visualization for quick assessment of segmentation quality. This is
    particularly useful for kidney I/R injury analysis where segmentation quality
    can vary across different regions and timepoints due to changes in nuclear
    morphology, density, and tissue architecture.

    The central crop approach focuses on a representative region of the tissue,
    allowing researchers to quickly assess whether nuclei are being properly
    segmented, especially in challenging areas like inflammatory infiltrates
    or damaged tubular regions where nuclei may be densely packed or have
    atypical morphology.

    Args:
        output_dir: Directory containing segmentation results from nuclei_segmentation.py.
        crop_size: Size of the cropped region in pixels (default: 1024).

    Returns:
        None. Visualization images are saved to the output directory.
    """
    logger = logging.getLogger()

    # Subdirectory for storing the check images.
    check_dir = os.path.join(output_dir, "visualizations", "cropped_preview")
    os.makedirs(check_dir, exist_ok=True)
    logger.info(f"Created visualization directory: {check_dir}")

    # Define all possible image paths with both .png and .tif extensions.
    possible_image_paths = [
        os.path.join(output_dir, "preprocessed", "preprocessed_image.png"),
        os.path.join(output_dir, "preprocessed", "preprocessed_image.tif"),
        os.path.join(output_dir, "preprocessed", "cropped_image.png"),
        os.path.join(output_dir, "preprocessed", "cropped_image.tif"),
        os.path.join(output_dir, "preprocessed", "contrast_enhanced_image.png"),
        os.path.join(output_dir, "preprocessed", "contrast_enhanced_image.tif"),
        os.path.join(output_dir, "preprocessed", "gamma_corrected_image.png"),
        os.path.join(output_dir, "preprocessed", "gamma_corrected_image.tif")
    ]

    # Define all possible mask paths.
    possible_mask_paths = [
        os.path.join(output_dir, "masks.npy"),
        os.path.join(output_dir, "segmentation_mask_post_watershed.npy"),
        os.path.join(output_dir, "masks", "masks.npy"),
        os.path.join(output_dir, "masks", "segmentation_mask_post_watershed.npy")
    ]

    # Debug: List all files in the output directory to help diagnose issues.
    logger.info("Searching for image and mask files in output directory...")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif") or file.endswith(".npy"):
                logger.info(f"Found file: {os.path.join(root, file)}")

    # Load preprocessed image (try different possible paths).
    img = None
    for img_path in possible_image_paths:
        if os.path.exists(img_path):
            try:
                img = skio.imread(img_path)
                logger.info(f"Successfully loaded image for overlay: {img_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load {img_path}: {e}")

    if img is None:
        logger.error("No valid preprocessed image found for overlay. Check file paths.")
        return

    # Load masks (try different possible paths).
    masks = None
    for mask_path in possible_mask_paths:
        if os.path.exists(mask_path):
            try:
                masks = np.load(mask_path)
                logger.info(f"Successfully loaded masks for overlay: {mask_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load {mask_path}: {e}")

    if masks is None:
        logger.error("No valid mask file found for overlay. Check file paths.")
        return

    # Get central crop or use the entire image if it's already small enough.
    h, w = img.shape[:2]
    logger.info(f"Image dimensions for overlay: {h}x{w}")

    # If image is already smaller than crop_size, use the entire image.
    if h <= crop_size and w <= crop_size:
        logger.info(f"Image already smaller than crop_size ({crop_size}), using entire image")
        img_crop = img
        masks_crop = masks
    else:
        # Otherwise, take a central crop.
        y0, x0 = max(0, h//2 - crop_size//2), max(0, w//2 - crop_size//2)
        y1, x1 = min(h, y0 + crop_size), min(w, x0 + crop_size)
        logger.info(f"Taking central crop from y={y0}:{y1}, x={x0}:{x1}")

        img_crop = img[y0:y1, x0:x1]

        # Make sure masks has the same shape as img.
        if masks.shape != img.shape:
            logger.warning(f"Mask shape ({masks.shape}) doesn't match image shape ({img.shape})")
            # Try to resize masks if dimensions don't match.
            if len(masks.shape) == 2 and len(img.shape) == 2:
                from skimage.transform import resize
                masks = resize(masks, img.shape, order=0, preserve_range=True).astype(np.uint16)
                logger.info(f"Resized masks to match image shape: {masks.shape}")

        masks_crop = masks[y0:y1, x0:x1]

    # Ensure we have valid data for the overlay.
    if np.max(masks_crop) == 0:
        logger.warning("No segmentation masks found in the cropped region!")
        # Save the cropped image anyway for debugging.
        skio.imsave(os.path.join(check_dir, "cropped_image_no_masks.png"), img_crop)
        return

    # Create overlay.
    logger.info(f"Creating overlay with {np.max(masks_crop)} unique mask labels")
    overlay = plot.mask_overlay(img_crop, masks_crop,
                               colors=np.random.rand(np.max(masks_crop) + 1, 3))

    # Save overlay.
    overlay_path = os.path.join(check_dir, "central_crop_overlay.png")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved central crop overlay to: {overlay_path}")

    # Create figure with both original and overlay.
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_crop, cmap='gray')
    ax[0].set_title("Original Image (Crop)")
    ax[0].axis('off')

    ax[1].imshow(overlay)
    ax[1].set_title("Segmentation Overlay")
    ax[1].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(check_dir, "central_crop_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved comparison figure to: {fig_path}")

    # Also save a full-size overlay if the image is not too large.
    if h * w <= 4000 * 4000:  # Only for reasonably sized images
        try:
            logger.info("Generating full-size overlay...")
            full_overlay = plot.mask_overlay(img, masks,
                                          colors=np.random.rand(np.max(masks) + 1, 3))
            full_overlay_path = os.path.join(check_dir, "full_image_overlay.png")
            skio.imsave(full_overlay_path, (full_overlay * 255).astype(np.uint8))
            logger.info(f"Saved full-size overlay to: {full_overlay_path}")
        except Exception as e:
            logger.warning(f"Could not generate full-size overlay: {e}")

def generate_full_overlay(image, masks, flows, output_dir, logger):
    """
    Generate and save overlay visualizations of the full kidney tissue image.

    This function creates comprehensive visualization outputs for the entire
    tissue section, including mask overlays and flow field visualizations.
    These full-size visualizations are essential for detailed inspection of
    segmentation quality across different kidney tissue compartments (cortex,
    medulla, etc.) and for identifying region-specific segmentation challenges.

    In kidney I/R injury analysis, different regions may show varying degrees
    of damage and inflammatory infiltration, requiring careful assessment of
    segmentation performance throughout the tissue. The flow field visualization
    helps diagnose issues with the Cellpose algorithm's gradient tracking that
    may lead to under- or over-segmentation in specific tissue contexts.

    Args:
        image: Input kidney tissue image (grayscale).
        masks: Segmentation masks from Cellpose or watershed refinement.
        flows: Flow fields from Cellpose (gradient and probability maps).
        output_dir: Directory where visualizations will be saved.
        logger: Logger instance for recording progress.

    Returns:
        None. Visualization images are saved to the specified directory.
    """
    # Create visualizations directory.
    vis_dir = os.path.join(output_dir, "visualizations", "full_image")
    os.makedirs(vis_dir, exist_ok=True)

    # Create mask overlay.
    overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))
    overlay_path = os.path.join(vis_dir, "mask_overlay.png")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved full overlay image to: {overlay_path}")

    # Create debug visualization with flows.
    fig = plt.figure(figsize=(12, 12))
    plot.show_segmentation(fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])
    debug_path = os.path.join(vis_dir, "segmentation_debug.png")
    fig.savefig(debug_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved segmentation debug overlay to: {debug_path}")
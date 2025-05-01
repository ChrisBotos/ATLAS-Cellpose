#!/usr/bin/env python3
"""
Segmentation Overlay Visualization for Kidney I/R Injury Analysis.

This script generates visual quality control outputs for nuclei segmentation results,
allowing you to quickly assess segmentation quality and the effects of
watershed refinement on merged nuclei in kidney tissue sections. These visualizations
are crucial for validating the accuracy of nuclear boundary detection in the context
of ischemia-reperfusion injury, where precise quantification of nuclear morphology
and spatial distribution is essential for downstream analyses.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import plot
import logging

def small_segmentation_overlay(output_dir, crop_size=512):
    """
    Creates and saves cropped overlay comparisons for pre-watershed and post-watershed segmentation.

    This function generates visual quality control outputs that are essential for validating
    segmentation results in kidney tissue analysis. It creates overlays of the segmentation
    masks on the original image, allowing you to assess how well nuclear boundaries
    are captured and whether watershed splitting has correctly separated merged nuclei.

    In kidney I/R injury analysis, proper nuclear segmentation is critical as changes in
    nuclear morphology and density are key indicators of tubular damage, inflammatory
    infiltration, and repair processes. Accurate segmentation is particularly important
    when analyzing proximal tubule injury, which is a hallmark of I/R damage.

    The function performs the following steps:
    1. Creates subdirectory 'check_cropped_part' in 'output_dir'.
    2. Loads:
       - Preprocessed image and optionally CLAHE and gamma-corrected images.
       - Pre-watershed mask from 'masks.npy'.
       - Post-watershed mask from 'segmentation_mask_watershed.png' if it exists.
    3. Crops a centered region of size 'crop_size' from each available image.
    4. Generates:
       (a) 'quick_overlay_summary.png' showing pre, CLAHE, gamma, and the pre-watershed overlay.
       (b) 'comparison_pre_post_watershed.png' side-by-side overlays of pre- vs. post-watershed
           (generated only if the post-watershed file exists).

    Args:
        output_dir: Directory containing segmentation results and where outputs will be saved.
        crop_size: Size in pixels of the central crop to use for visualization (default: 512).

    Returns:
        None. Results are saved as image files in the output directory.
    """

    # -------------------------------
    # 1) Logging setup
    # -------------------------------
    # Configure logger for tracking the segmentation overlay process.
    logger = logging.getLogger("small_segmentation_overlay")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

    # Subdirectory for storing the check images.
    check_dir = os.path.join(output_dir, "check_cropped_part")
    os.makedirs(check_dir, exist_ok=True)
    logger.info(f"Created/Found check directory: {check_dir}.")


    # -------------------------------
    # 2) Load images
    # -------------------------------
    # Define paths to all required and optional image files.
    pre_path       = os.path.join(output_dir, "preprocessed_image.png")
    clahe_path     = os.path.join(output_dir, "contrast_enhanced_image.png")
    gamma_path     = os.path.join(output_dir, "gamma_corrected_image.png")
    masks_path     = os.path.join(output_dir, "masks.npy")  # pre-watershed mask.
    watershed_path = os.path.join(output_dir, "segmentation_mask_watershed.png")  # post-watershed mask (optional).

    # Check essential file existence - we need at least the preprocessed image and pre-watershed mask.
    if not os.path.exists(pre_path):
        logger.error(f"File not found: {pre_path}.")
        return
    if not os.path.exists(masks_path):
        logger.error(f"File not found: {masks_path}.")
        return

    # Load all available images for visualization.
    pre   = skio.imread(pre_path)
    clahe = skio.imread(clahe_path) if os.path.exists(clahe_path) else None
    gamma = skio.imread(gamma_path) if os.path.exists(gamma_path) else None

    # Load pre-watershed mask (NumPy label mask) - this contains the initial cell segmentation.
    masks_pre = np.load(masks_path)

    # Load post-watershed mask if it exists - this contains the refined segmentation after watershed splitting.
    if os.path.exists(watershed_path):
        masks_post = skio.imread(watershed_path)
        watershed_exists = True
    else:
        logger.warning(f"Watershed file not found: {watershed_path}. Skipping post-watershed overlay.")
        watershed_exists = False

    # Ensure shapes match for pre-image and pre-watershed mask - this is critical for proper overlay.
    assert pre.shape == masks_pre.shape, "Mismatch: preprocessed image & pre-watershed mask must have same shape."


    # -------------------------------
    # 3) Crop a centered region
    # -------------------------------
    # Calculate the center of the image and determine crop boundaries.
    h, w = pre.shape
    cy, cx = h // 2, w // 2

    start_y = max(cy - crop_size // 2, 0)
    end_y   = min(start_y + crop_size, h)
    start_x = max(cx - crop_size // 2, 0)
    end_x   = min(start_x + crop_size, w)

    # Crop all images that exist to focus on the central region of interest.
    pre_crop         = pre[start_y:end_y, start_x:end_x]
    masks_pre_crop   = masks_pre[start_y:end_y, start_x:end_x]

    # Crop CLAHE-enhanced image if available and shape matches.
    if clahe is not None and clahe.shape == pre.shape:
        clahe_crop = clahe[start_y:end_y, start_x:end_x]
    else:
        clahe_crop = None

    # Crop gamma-corrected image if available and shape matches.
    if gamma is not None and gamma.shape == pre.shape:
        gamma_crop = gamma[start_y:end_y, start_x:end_x]
    else:
        gamma_crop = None

    # If watershed mask exists, crop it; otherwise, leave it as None.
    if watershed_exists:
        h_w, w_w = masks_post.shape
        # Safe indices: crop only up to the size of the watershed mask.
        end_y_w = min(end_y, h_w)
        end_x_w = min(end_x, w_w)
        masks_post_crop = masks_post[start_y:end_y_w, start_x:end_x_w]
    else:
        masks_post_crop = None

    # Adjust cropping in case the post-watershed mask (if exists) is smaller than the pre-watershed mask.
    # This ensures all images have the same dimensions for proper comparison.
    if watershed_exists:
        final_crop_height = min(pre_crop.shape[0], masks_post_crop.shape[0])
        final_crop_width  = min(pre_crop.shape[1], masks_post_crop.shape[1])
    else:
        final_crop_height = pre_crop.shape[0]
        final_crop_width = pre_crop.shape[1]

    # Apply final crop dimensions to all images for consistency.
    pre_crop       = pre_crop[:final_crop_height, :final_crop_width]
    masks_pre_crop = masks_pre_crop[:final_crop_height, :final_crop_width]
    if clahe_crop is not None:
        clahe_crop = clahe_crop[:final_crop_height, :final_crop_width]
    if gamma_crop is not None:
        gamma_crop = gamma_crop[:final_crop_height, :final_crop_width]
    if watershed_exists:
        masks_post_crop = masks_post_crop[:final_crop_height, :final_crop_width]


    # -------------------------------
    # 4) Generate overlays
    # -------------------------------
    # Pre-watershed overlay - each segmented nucleus gets a random color for visualization.
    overlay_pre = plot.mask_overlay(pre_crop, masks_pre_crop,
                                    colors=np.random.rand(np.max(masks_pre_crop) + 1, 3))

    # Post-watershed overlay (only if watershed exists) - shows how watershed algorithm split merged nuclei.
    if watershed_exists:
        overlay_post = plot.mask_overlay(pre_crop, masks_post_crop,
                                         colors=np.random.rand(np.max(masks_post_crop) + 1, 3))
    else:
        overlay_post = None


    # -------------------------------
    # 5a) Save a figure with pre, CLAHE, gamma, and pre-watershed overlay (2x2)
    # -------------------------------
    # Create a 2x2 figure showing the preprocessing steps and segmentation result.
    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(pre_crop, cmap="gray")
    axes[0, 0].set_title("Preprocessed Image")
    axes[0, 0].axis("off")

    if clahe_crop is not None:
        axes[0, 1].imshow(clahe_crop, cmap="gray")
        axes[0, 1].set_title("CLAHE Enhanced")
        axes[0, 1].axis("off")
    else:
        axes[0, 1].text(0.5, 0.5, "No CLAHE", ha="center", va="center")
        axes[0, 1].axis("off")

    if gamma_crop is not None:
        axes[1, 0].imshow(gamma_crop, cmap="gray")
        axes[1, 0].set_title("Gamma Corrected")
        axes[1, 0].axis("off")
    else:
        axes[1, 0].text(0.5, 0.5, "No Gamma", ha="center", va="center")
        axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_pre)
    axes[1, 1].set_title("Segmentation Masks Overlay")
    axes[1, 1].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(check_dir, "quick_overlay_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"Saved summary overlay figure to: {summary_path}.")


    # -------------------------------
    # 5b) Side-by-side comparison of pre/post-watershed overlays (only if watershed exists)
    # -------------------------------
    # If watershed segmentation exists, create a side-by-side comparison to evaluate its effectiveness.
    if watershed_exists:
        fig2, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(overlay_pre)
        axs[0].set_title("Pre-Watershed Overlay")
        axs[0].axis("off")

        axs[1].imshow(overlay_post)
        axs[1].set_title("Post-Watershed Overlay")
        axs[1].axis("off")

        plt.tight_layout()
        comparison_path = os.path.join(check_dir, "comparison_pre_post_watershed.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        logger.info(f"Saved pre/post comparison overlay figure to: {comparison_path}.")
    else:
        logger.info("Skipped saving pre/post comparison overlay figure due to missing watershed file.")

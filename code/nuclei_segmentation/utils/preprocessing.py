"""
Image Preprocessing Utilities for Kidney I/R Injury Spatial Multiomics Analysis.

This module provides specialized image preprocessing functions optimized for
kidney tissue microscopy images. It handles various preprocessing steps including:
1. Loading images of different formats and bit depths
2. Converting color images to grayscale for DAPI channel analysis
3. Enhancing contrast using CLAHE for better nuclei visibility
4. Applying gamma correction to enhance dim regions
5. Cropping to regions of interest for focused analysis

These preprocessing steps are critical for optimizing nuclear segmentation in
kidney tissue after ischemia-reperfusion injury, where tissue damage can create
challenging imaging conditions with varying signal intensities.
"""

# Standard library imports.
import os
import sys
import traceback
import numpy as np
import cv2
from skimage import io as skio
from pathlib import Path


def convert_16bit_to_8bit(image):
    """
    Convert a 16-bit image to 8-bit using robust percentile-based scaling.

    This function performs an intelligent conversion from 16-bit to 8-bit depth
    using percentile-based contrast stretching. It preserves the maximum amount of
    detail in both dark and bright regions by using the 0.5th and 99.5th percentiles
    rather than simple min/max scaling.

    Args:
        image: Input image as numpy array (expected to be 16-bit).

    Returns:
        numpy.ndarray: 8-bit image with preserved dynamic range.
    """
    if image.dtype != np.uint16:
        return image

    # Use robust percentiles to avoid outlier influence.
    p0_5, p99_5 = np.percentile(image, (0.5, 99.5))

    # Handle edge case of zero dynamic range.
    if p99_5 - p0_5 <= 0:
        p0_5, p99_5 = image.min(), image.max()
        if p99_5 - p0_5 <= 0:  # Still zero range.
            return np.zeros_like(image, dtype=np.uint8)

    # Apply contrast stretching with the robust percentiles.
    normalized = np.clip((image - p0_5) / (p99_5 - p0_5), 0, 1)

    # Convert to 8-bit for further processing.
    return (normalized * 255).astype(np.uint8)


def adaptive_gamma_correction(image, min_gamma=1.5, max_gamma=2.5, logger=None):
    """
    Apply content-adaptive gamma correction to enhance dim nuclei.

    This function automatically determines the optimal gamma correction value
    based on the image's brightness distribution. Darker images receive stronger
    correction (higher gamma) while brighter images receive gentler correction.
    This is particularly effective for DAPI-stained nuclei images with varying
    brightness levels.

    Args:
        image: Input image (numpy array, 8-bit).
        min_gamma: Minimum gamma value for bright images.
        max_gamma: Maximum gamma value for dark images.
        logger: Optional logger for recording the applied gamma value.

    Returns:
        numpy.ndarray: Gamma-corrected image with enhanced dim regions.
    """
    # Calculate median brightness to determine appropriate gamma value.
    median = np.median(image) / 255.0

    # Interpolate gamma value based on image brightness.
    gamma = np.clip(max_gamma - (max_gamma - min_gamma) * median, min_gamma, max_gamma)

    if logger:
        logger.info(f"Applying adaptive gamma correction with γ = {gamma:.2f}")

    # Create lookup table for efficient gamma correction.
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")

    # Apply correction using lookup table for efficiency.
    return cv2.LUT(image, table)


def ensure_matching_shapes(image, masks, logger, crop_only=True):
    """
    Ensure that image and masks have the same shape by cropping to common dimensions.

    This function is used throughout the pipeline to prevent shape mismatches that
    could cause errors or incorrect results. Instead of resizing (which can distort
    segmentation masks), it crops both arrays to their common dimensions.

    Args:
        image: The image array (2D or 3D).
        masks: The mask array (2D).
        logger: Logger for recording information.
        crop_only: If True, only crop; if False, allow padding of smaller array.

    Returns:
        tuple: (image, masks) with matching shapes.
    """
    if image.shape[:2] == masks.shape[:2]:
        logger.debug("Image and masks already have matching shapes.")
        return image, masks

    logger.warning(f"Shape mismatch: image {image.shape} vs masks {masks.shape}")

    # Find common dimensions
    common_h = min(image.shape[0], masks.shape[0])
    common_w = min(image.shape[1], masks.shape[1])

    logger.info(f"Using common region of size {common_h}x{common_w}")

    # Crop both to common size
    image_cropped = image[:common_h, :common_w]
    masks_cropped = masks[:common_h, :common_w]

    logger.info(f"Cropped image to {image_cropped.shape} and masks to {masks_cropped.shape}")

    return image_cropped, masks_cropped


def preprocess_image(image_path, settings, logger):
    """
    Load and preprocess microscopy images for optimal segmentation.

    This function handles the initial processing steps for microscopy images including:
    1. Loading images of various formats and bit depths
    2. Converting color images to grayscale (for DAPI channel)
    3. Handling bit depth conversion with dynamic range preservation
    4. Optional cropping to region of interest
    5. Saving intermediate results for quality control

    Args:
        image_path: Path to the input microscopy image.
        settings: Dictionary containing preprocessing parameters.
        logger: Logger for recording processing steps.

    Returns:
        numpy.ndarray: Preprocessed grayscale image ready for segmentation.

    Raises:
        SystemExit: If image cannot be loaded.
    """
    # Load the image with error handling.
    try:
        image = skio.imread(image_path)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        sys.exit(1)

    logger.info(f"Original image: {image_path}")
    logger.info(f"Image properties: dtype={image.dtype}, shape={image.shape}")

    # Remove alpha channel if present (common in some microscopy formats).
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
        logger.info("Removed alpha channel from image.")

    # Convert 16-bit to 8-bit with dynamic range preservation.
    if image.dtype == np.uint16:
        image = convert_16bit_to_8bit(image)
        logger.info("Converted 16-bit to 8-bit with percentile-based scaling.")

    # Convert color images to grayscale (for DAPI channel).
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info("Converted color image to grayscale for nuclei segmentation.")

    # Create directory for saving preprocessed images.
    preprocessed_dir = os.path.join(settings["OUTPUT_DIR"], "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Save preprocessed image in multiple formats for different uses.
    skio.imsave(os.path.join(preprocessed_dir, "preprocessed_image.tif"), image)  # Lossless TIF.
    skio.imsave(os.path.join(preprocessed_dir, "preprocessed_image.tif"), image)  # PNG for visualization.
    logger.info(f"Saved preprocessed images to {preprocessed_dir} directory.")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) if enabled
    if settings.get("ENHANCE_CONTRAST", False):
        logger.info("Applying CLAHE enhancement...")
        try:
            # Get CLAHE parameters from settings
            clip_limit = settings.get("CLAHE_CLIPLIMIT", 2.0)
            tile_grid_size = settings.get("CLAHE_TILE_GRID_SIZE", (8, 8))

            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            # Apply CLAHE
            clahe_enhanced = clahe.apply(image)

            # Save CLAHE enhanced image
            skio.imsave(os.path.join(preprocessed_dir, "contrast_enhanced_image.tif"), clahe_enhanced)
            skio.imsave(os.path.join(preprocessed_dir, "contrast_enhanced_image.tif"), clahe_enhanced)
            logger.info(f"Saved CLAHE enhanced image with clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")

            # Also save at the root level for backward compatibility
            skio.imsave(os.path.join(settings["OUTPUT_DIR"], "contrast_enhanced_image.tif"), clahe_enhanced)
        except Exception as e:
            logger.error(f"Error applying CLAHE enhancement: {e}")
            logger.error(traceback.format_exc())

    # Apply gamma correction if enabled
    if settings.get("ENHANCE_DIM", False):
        logger.info("Applying gamma correction for dim regions...")
        try:
            # Apply adaptive gamma correction
            gamma_corrected = adaptive_gamma_correction(image, min_gamma=1.5, max_gamma=2.5, logger=logger)

            # Save gamma corrected image
            skio.imsave(os.path.join(preprocessed_dir, "gamma_corrected_image.tif"), gamma_corrected)
            skio.imsave(os.path.join(preprocessed_dir, "gamma_corrected_image.tif"), gamma_corrected)
            logger.info("Saved gamma corrected image")

            # Also save at the root level for backward compatibility
            skio.imsave(os.path.join(settings["OUTPUT_DIR"], "gamma_corrected_image.tif"), gamma_corrected)
        except Exception as e:
            logger.error(f"Error applying gamma correction: {e}")
            logger.error(traceback.format_exc())

    # Apply region-of-interest cropping if enabled.
    if settings.get("CROP_IMAGE", False):
        logger.info("Applying region-of-interest cropping.")

        # Get crop coordinates from settings (format: y_start, y_end, x_start, x_end).
        crop_bbox = settings.get("CROP_BBOX", (0, 1, 0, 1))

        # Handle string format from config file.
        if isinstance(crop_bbox, str):
            crop_values = [float(x.strip()) for x in crop_bbox.split(',')]
            if len(crop_values) == 4:
                y0, y1, x0, x1 = crop_values
            else:
                logger.warning(f"Invalid crop coordinates format: {crop_bbox}. Using full image.")
                y0, y1, x0, x1 = 0, 1, 0, 1
        else:
            y0, y1, x0, x1 = crop_bbox

        h, w = image.shape
        logger.info(f"Original dimensions: {w}×{h} pixels")

        # Determine if coordinates are relative (0-1) or absolute pixels.
        if 0 <= y0 < 1 and 0 <= y1 <= 1 and 0 <= x0 < 1 and 0 <= x1 <= 1:
            # Convert relative coordinates to absolute pixel coordinates.
            y0_px, y1_px = int(y0 * h), int(y1 * h)
            x0_px, x1_px = int(x0 * w), int(x1 * w)
            logger.info(f"Using relative coordinates: ({y0:.2f}, {y1:.2f}, {x0:.2f}, {x1:.2f})")
        else:
            # Use absolute pixel coordinates directly.
            y0_px, y1_px, x0_px, x1_px = int(y0), int(y1), int(x0), int(x1)
            logger.info(f"Using absolute pixel coordinates for cropping.")

        # Validate coordinates are within image bounds.
        y0_px = max(0, min(y0_px, h-1))
        y1_px = max(y0_px+1, min(y1_px, h))
        x0_px = max(0, min(x0_px, w-1))
        x1_px = max(x0_px+1, min(x1_px, w))

        # Apply the crop operation.
        image = image[y0_px:y1_px, x0_px:x1_px]
        logger.info(f"Cropped to region: y=[{y0_px}:{y1_px}], x=[{x0_px}:{x1_px}]")
        logger.info(f"New dimensions: {image.shape[1]}×{image.shape[0]} pixels")

        # Save cropped image in multiple formats.
        skio.imsave(os.path.join(preprocessed_dir, "cropped_image.tif"), image)  # Lossless TIF.
        skio.imsave(os.path.join(preprocessed_dir, "cropped_image.tif"), image)  # PNG for visualization.
        logger.info(f"Saved cropped images to {preprocessed_dir} directory.")

    if settings.get("UPSCALE_FACTOR", 1) > 1:
        image = cv2.resize(image, None,
                           fx=settings["UPSCALE_FACTOR"],
                           fy=settings["UPSCALE_FACTOR"],
                           interpolation=cv2.INTER_LINEAR)
        logger.info(f"Upscaled image to: {image.shape}")

    return image

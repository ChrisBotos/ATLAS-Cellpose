"""
IMAGE PREPROCESSING MODULE

Preprocessing utilities for nuclear segmentation of kidney I/R tissue.

This module provides:
- 16-bit to 8-bit conversion using percentile-based dynamic range.
- Adaptive gamma correction for dim nuclear regions.
- CLAHE contrast enhancement with customizable grid size.
- Shape correction to match image and mask sizes by cropping.
- Optional ROI cropping using relative or absolute bounding boxes.
- Robust image saving to structured folders with logging support.

This is for the preprocessing pipeline of spatial omics kidney data across I/R injury timepoints.
"""

import os
import numpy as np
import cv2
from skimage import io as skio


'''BIT DEPTH CONVERSION'''

def convert_16bit_to_8bit(image: np.ndarray) -> np.ndarray:
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


'''GAMMA CORRECTION'''

def adaptive_gamma_correction(image: np.ndarray, min_gamma: float = 1.5, max_gamma: float = 2.5, logger=None) -> np.ndarray:
    """
    Enhance dim images using adaptive gamma correction.

    Args:
        image (np.ndarray): 8-bit grayscale image.
        min_gamma (float): Gamma for bright images.
        max_gamma (float): Gamma for dim images.
        logger: Optional logger object.

    Returns:
        np.ndarray: Gamma-corrected image.
    """
    brightness = np.median(image) / 255.0
    gamma = np.clip(max_gamma - (max_gamma - min_gamma) * brightness, min_gamma, max_gamma)

    if logger:
        logger.info(f"Adaptive gamma correction with gamma = {gamma:.2f}")

    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


'''CLAHE ENHANCEMENT'''

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), logger=None) -> np.ndarray:
    """
    Apply CLAHE to enhance local contrast.

    Args:
        image (np.ndarray): Grayscale image.
        clip_limit (float): CLAHE clip limit.
        tile_grid_size (tuple): Grid size for tiles.
        logger: Optional logger object.

    Returns:
        np.ndarray: CLAHE-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)

    if logger:
        logger.info(f"CLAHE applied with clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")

    return enhanced


'''CROPPING UTILS'''

def crop_image(image: np.ndarray, crop_bbox, logger=None) -> np.ndarray:
    """
    Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_bbox: Tuple of (y0, y1, x0, x1), either relative (0–1) or absolute.
        logger: Logger object.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape
    y0, y1, x0, x1 = crop_bbox

    if all(0 <= val <= 1 for val in crop_bbox):
        y0, y1 = int(y0 * h), int(y1 * h)
        x0, x1 = int(x0 * w), int(x1 * w)
        if logger:
            logger.info(f"Cropping with relative bbox: ({y0}:{y1}, {x0}:{x1})")
    else:
        y0, y1, x0, x1 = map(int, crop_bbox)
        if logger:
            logger.info(f"Cropping with absolute bbox: ({y0}:{y1}, {x0}:{x1})")

    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)

    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop dimensions: y=[{y0}:{y1}], x=[{x0}:{x1}]")

    return image[y0:y1, x0:x1]


'''FILE SAVING'''

def save_image(image: np.ndarray, path: str, logger=None):
    """
    Save image with logging.

    Args:
        image (np.ndarray): Image to save.
        path (str): File path to save.
        logger: Optional logger.
    """
    try:
        skio.imsave(path, image)
        if logger:
            logger.info(f"Saved image to {path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save {path}: {e}")


'''MAIN PIPELINE'''

def preprocess_image(image_path, settings, logger):
    """
    Complete preprocessing pipeline.

    Args:
        image_path (str): Path to input image.
        settings (dict): Dictionary of preprocessing settings.
        logger: Logger for messages.

    Returns:
        np.ndarray: Final processed image.
    """
    try:
        image = skio.imread(image_path)
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        raise RuntimeError("Image loading failed.") from e

    if logger:
        logger.info(f"Loaded {image_path} with shape {image.shape} and dtype {image.dtype}")

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
        logger.info("Removed alpha channel.")

    if image.dtype == np.uint16:
        image = convert_16bit_to_8bit(image)
        logger.info("Converted from 16-bit to 8-bit.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info("Converted RGB to grayscale.")

    out_dir = os.path.join(settings["OUTPUT_DIR"], "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    save_image(image, os.path.join(out_dir, "initial.tif"), logger)

    if settings.get("ENHANCE_CONTRAST", False):
        image = apply_clahe(image,
                            clip_limit=settings.get("CLAHE_CLIPLIMIT", 2.0),
                            tile_grid_size=settings.get("CLAHE_TILE_GRID_SIZE", (8, 8)),
                            logger=logger)
        save_image(image, os.path.join(out_dir, "clahe.tif"), logger)

    if settings.get("ENHANCE_DIM", False):
        image = adaptive_gamma_correction(image, logger=logger)
        save_image(image, os.path.join(out_dir, "gamma.tif"), logger)

    if settings.get("CROP_IMAGE", False):
        crop_box = settings.get("CROP_BBOX", (0, 1, 0, 1))
        if isinstance(crop_box, str):
            crop_box = [float(x.strip()) for x in crop_box.split(',')]
        image = crop_image(image, crop_box, logger)
        save_image(image, os.path.join(out_dir, "cropped.tif"), logger)

    upscale_factor = settings.get("UPSCALE_FACTOR", 1)
    if upscale_factor > 1:
        image = cv2.resize(image, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)
        logger.info(f"Upscaled image to shape {image.shape}")
        save_image(image, os.path.join(out_dir, "upscaled.tif"), logger)

    return image

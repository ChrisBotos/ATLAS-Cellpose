#!/usr/bin/env python3
"""
Advanced Nuclei Segmentation Pipeline for Spatial Multiomics Analysis of Kidney I/R Injury.

This script processes large microscopy images of kidney tissue sections, optimized for DAPI-stained nuclei.
The pipeline is specifically designed for analyzing ischemia-reperfusion (I/R) kidney injury, a condition
where temporary blood flow restriction followed by restoration causes complex cellular damage patterns.

The pipeline includes:
1. Preprocessing with contrast enhancement and optional cropping to optimize nuclear signal detection.
2. Intelligent tiling for handling large histological images efficiently (>1GB).
3. Cellpose-based deep learning segmentation with parameters optimized for kidney nuclei.
4. Optional refinement using edge detection and watershed splitting for merged nuclei.
5. Comprehensive visualization and quality control outputs for scientific validation.

Designed for analyzing kidney I/R injury across multiple timepoints (10h, 2d, 14d) to capture the
dynamic progression of injury, inflammation, and repair processes. The segmentation results enable
downstream spatial analysis of transcriptomic and metabolomic data in the context of nuclear morphology.

Outputs include segmentation masks, feature measurements, and visualization overlays that can be
used for further quantitative analysis of nuclear morphology changes during I/R injury progression.
"""


# Standard library imports.
import os
import sys
import logging
import configparser
import shutil
import traceback
from datetime import datetime
from pathlib import Path

# Scientific and image processing imports.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import imageio
from PIL import Image
from scipy import ndimage as ndi
from skimage import io as skio
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from cellpose import models, plot

# Own code imports.
from utils.visualization import small_segmentation_overlay

# Increase the maximum allowed image pixels to handle large microscopy images.
Image.MAX_IMAGE_PIXELS = 10 ** 9

# =============================================================================
# PROJECT STRUCTURE SETUP
# =============================================================================
def setup_project_structure():
    """
    Create the project directory structure if it doesn't exist.

    Creates all necessary directories for the pipeline including configs, data,
    results, logs, and debug directories. This ensures the pipeline can run without
    file system errors even on a fresh installation.

    Returns:
        dict: Dictionary containing paths to all project directories.
    """
    # Define base directories.
    base_dir = Path(__file__).parent.absolute()
    dirs = {
        "base": base_dir,
        "configs": base_dir / "configs",
        "data": base_dir / "data",
        "results": base_dir / "results",
        "logs": base_dir / "logs",
        "tests": base_dir / "tests",
        "utils": base_dir / "utils",
        "debug": base_dir / "debug"
    }

    # Create directories if they don't exist.
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    return dirs

# Setup project structure.
PROJECT_DIRS = setup_project_structure()


# =============================================================================
# CONFIG LOADING
# =============================================================================
def load_config(config_path=None):
    """
    Load configuration from the specified INI file.

    Parses the configuration file to extract all settings for the segmentation pipeline.
    Provides sensible defaults for optional parameters and performs validation.

    Args:
        config_path: Path to the configuration INI file. If None, uses the default config.

    Returns:
        tuple: (SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS) dictionaries containing all
               configuration parameters needed for the pipeline.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If there's an error parsing the configuration file.
    """
    # Get current timestamp for output directory naming.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up project directories.
    script_dir = Path(__file__).resolve().parent
    PROJECT_DIRS = {
        "root": script_dir,
        "data": script_dir / "data",
        "results": script_dir / "results",
        "configs": script_dir / "configs"
    }

    # Use default config path if none provided.
    if config_path is None:
        config_path = PROJECT_DIRS["configs"] / "nuclei_segmentation_config.ini"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Parse the configuration file.
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
    except Exception as e:
        raise ValueError(f"Error parsing configuration file: {e}")

    # Parse settings from config with default values for optional parameters.
    SETTINGS = {
        "IMAGE_PATH": config.get("General", "image_path"),
        "OUTPUT_DIR": PROJECT_DIRS["results"] / f"{config.get('General', 'output_dir')}_{timestamp}",
        "UPSCALE_FACTOR": config.getint("General", "upscale_factor", fallback=1),
        "CROP_IMAGE": config.getboolean("General", "crop_image", fallback=False),
        "ENHANCE_CONTRAST": config.getboolean("General", "enhance_contrast", fallback=False),
        "ENHANCE_DIM": config.getboolean("General", "enhance_dim", fallback=False),
        "GENERATE_OVERLAY": config.getboolean("General", "generate_overlay", fallback=True),
        "USE_EDGE_DETECTION": config.getboolean("EdgeDetection", "use_edge_detection", fallback=False),
        "APPLY_WATERSHED": config.getboolean("Watershed", "apply_watershed", fallback=False),
        "USE_TILING": config.getboolean("Tiling", "use_tiling", fallback=True),
        "MERGE_OVERLAP_THRESHOLD": config.getfloat("Tiling", "merge_overlap_threshold", fallback=0.3),
        "DEBUG_MODE": config.getboolean("Debug", "debug_mode", fallback=False) if "Debug" in config.sections() else False,
    }

    # Parse optional settings with try/except to provide defaults.
    try:
        SETTINGS["CROP_BBOX"] = tuple(map(float, config.get("General", "crop_bbox").split(',')))
    except (configparser.NoOptionError, ValueError):
        SETTINGS["CROP_BBOX"] = (0, 1, 0, 1)  # Default: full image.

    try:
        SETTINGS["CLAHE_CLIPLIMIT"] = config.getfloat("CLAHE", "cliplimit", fallback=2.0)
        SETTINGS["CLAHE_TILE_GRID_SIZE"] = tuple(map(int, config.get("CLAHE", "tile_grid_size", fallback="8,8").split(',')))
    except (configparser.NoSectionError, ValueError):
        SETTINGS["CLAHE_CLIPLIMIT"] = 2.0
        SETTINGS["CLAHE_TILE_GRID_SIZE"] = (8, 8)

    try:
        SETTINGS["CANNY_THRESHOLD1"] = config.getint("EdgeDetection", "canny_threshold1", fallback=50)
        SETTINGS["CANNY_THRESHOLD2"] = config.getint("EdgeDetection", "canny_threshold2", fallback=150)
    except configparser.NoSectionError:
        SETTINGS["CANNY_THRESHOLD1"] = 50
        SETTINGS["CANNY_THRESHOLD2"] = 150

    try:
        SETTINGS["AREA_THRESHOLD_FOR_WATERSHED"] = config.getint("Watershed", "area_threshold", fallback=1000)
        SETTINGS["LOCAL_MAXIMA_FOOTPRINT"] = tuple(map(int, config.get("Watershed", "local_maxima_footprint", fallback="3,3").split(',')))
    except configparser.NoSectionError:
        SETTINGS["AREA_THRESHOLD_FOR_WATERSHED"] = 1000
        SETTINGS["LOCAL_MAXIMA_FOOTPRINT"] = (3, 3)

    try:
        SETTINGS["tile_side_length"] = config.getint("Tiling", "tile_side_length", fallback=1024)
        SETTINGS["TILE_OVERLAP"] = config.getfloat("Tiling", "tile_overlap", fallback=0.1)
    except configparser.NoSectionError:
        SETTINGS["tile_side_length"] = 1024
        SETTINGS["TILE_OVERLAP"] = 0.1

    try:
        SETTINGS["SMALL_OVERLAY_SIZE"] = config.getint("Overlay", "small_overlay_size", fallback=1024)
    except configparser.NoSectionError:
        SETTINGS["SMALL_OVERLAY_SIZE"] = 1024

    # Ensure image path is absolute.
    if not os.path.isabs(SETTINGS["IMAGE_PATH"]):
        SETTINGS["IMAGE_PATH"] = PROJECT_DIRS["data"] / SETTINGS["IMAGE_PATH"]

    # Parse Cellpose parameters with defaults.
    CELLPOSE_PARAMS = {
        "model_type": config.get("Cellpose", "model_type", fallback="nuclei"),
        "gpu": config.getboolean("Cellpose", "gpu", fallback=True) and torch.cuda.is_available(),
        "diameter": config.getint("Cellpose", "diameter", fallback=0),
        "flow_threshold": config.getfloat("Cellpose", "flow_threshold", fallback=0.4),
        "cellprob_threshold": config.getfloat("Cellpose", "cellprob_threshold", fallback=0.0),
        "resample": config.getboolean("Cellpose", "resample", fallback=True),
        "stitch_threshold": config.getfloat("Cellpose", "stitch_threshold", fallback=0.4),
        "batch_size": 1  # Placeholder, will be updated dynamically based on GPU memory.
    }

    # Parse channels with error handling.
    try:
        CELLPOSE_PARAMS["channels"] = tuple(map(int, config.get("Cellpose", "channels", fallback="0,0").split(',')))
    except ValueError:
        CELLPOSE_PARAMS["channels"] = (0, 0)  # Default: grayscale.

    return SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS

# Load config values from file.
SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem_per_batch=150_000_000):
    """
    Calculate optimal batch size for GPU processing based on available memory.

    Dynamically determines how many image tiles can be processed in parallel
    based on the GPU memory available and the size of each tile.

    Args:
        tile_pixels: Number of pixels per tile (tile_side_length²).
        bytes_per_pixel: Memory usage per pixel (1 for uint8, ~4 for float32).
        target_mem_per_batch: Target GPU memory usage per batch in bytes.

    Returns:
        int: Optimal batch size for GPU processing.
    """
    if not torch.cuda.is_available():
        return 1

    # Get GPU properties.
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory  # in bytes.

    # Use half of available memory to be safe.
    usable = total_mem // 2

    # Calculate memory needed per tile.
    bytes_per_patch = tile_pixels * bytes_per_pixel

    # Calculate how many tiles fit into target memory.
    max_batch = max(1, usable // (bytes_per_patch * (usable // target_mem_per_batch)))
    return int(max_batch)


# Calculate optimal batch size based on tile dimensions.
tile_pixels = SETTINGS["tile_side_length"] ** 2
CELLPOSE_PARAMS["batch_size"] = choose_batch_size(tile_pixels)


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir, debug_mode=False):
    """
    Configure comprehensive logging system for the segmentation pipeline.

    Creates a dual-output logging system that writes detailed logs to both a file
    and the console. The file logger captures all details including timestamps,
    while the console logger provides a more concise output for interactive use.

    Args:
        output_dir: Directory where log files will be saved.
        debug_mode: If True, sets log level to DEBUG for more detailed logging.

    Returns:
        logger: Configured logging instance ready for use throughout the pipeline.
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create logs directory within output directory.
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up log file path with timestamp for uniqueness.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"segmentation_log_{timestamp}.txt")

    # Configure root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Remove any existing handlers to avoid duplicate logging.
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Create formatters for different outputs.
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # File handler for complete logging.
    fh = logging.FileHandler(log_file)
    fh.setFormatter(file_formatter)
    fh.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.addHandler(fh)

    # Console handler for interactive feedback.
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_formatter)
    ch.setLevel(logging.INFO)  # Console always shows INFO and above.
    logger.addHandler(ch)

    logger.info("===== Nuclei Segmentation Pipeline Started =====")
    logger.info(f"Log file: {log_file}")

    # Copy the config file to the output directory for reproducibility.
    config_backup_path = os.path.join(output_dir, "config_used.ini")
    try:
        shutil.copy2(PROJECT_DIRS["configs"] / "nuclei_segmentation_config.ini", config_backup_path)
        logger.info(f"Configuration backed up to: {config_backup_path}")
    except Exception as e:
        logger.warning(f"Could not backup configuration file: {e}")

    return logger


# =============================================================================
# DEBUG UTILITIES
# =============================================================================
def setup_debug(settings):
    """
    Configure debugging environment for advanced troubleshooting.

    When debug mode is enabled, this function creates a specialized directory
    for storing intermediate processing results as images. This is invaluable for
    diagnosing segmentation issues and understanding the pipeline's behavior.

    Args:
        settings: Dictionary containing configuration settings including DEBUG_MODE.

    Returns:
        function: A function for saving debug images with automatic normalization.
                 Returns the input array to allow inline use in processing chains.
    """
    if not settings.get("DEBUG_MODE", False):
        # Return a no-op function if debug mode is disabled for efficiency.
        return lambda tag, arr: arr

    # Create debug directory within output directory.
    debug_dir = os.path.join(settings["OUTPUT_DIR"], "debug")
    os.makedirs(debug_dir, exist_ok=True)

    def snap(tag, arr):
        """
        Save an intermediate processing result as a debug image.

        Automatically handles normalization for different data types and adds
        timestamps to prevent overwriting previous debug images.

        Args:
            tag: String identifier for the image (used in filename).
            arr: Numpy array containing the image data to save.

        Returns:
            arr: The original input array (allows inline use in processing chains).
        """
        # Normalize array to 8-bit for visualization.
        if arr.dtype != np.uint8:
            v = arr.astype(np.float32)
            v = 255 * (v - v.min()) / (v.np.ptp() + 1e-6)
            v = v.astype(np.uint8)
        else:
            v = arr

        # Save with timestamp to avoid overwriting previous debug images.
        timestamp = datetime.now().strftime("%H%M%S")
        imageio.imwrite(os.path.join(debug_dir, f"{tag}_{timestamp}.tif"), v)
        return arr  # Return the original array for inline use.

    return snap


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
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


def split_image_into_tiles(image, tile_size, overlap, logger):
    """
    Split a large image into overlapping tiles for processing.

    Args:
        image: Input image as numpy array
        tile_size: Size of each tile (square)
        overlap: Fraction of overlap between tiles (0-1)
        logger: Logger object

    Returns:
        tiles: List of image tiles
        slices: List of slice tuples for reconstructing the full image
    """
    h, w = image.shape
    logger.info(f"Splitting {h}×{w} image into tiles of size {tile_size}×{tile_size} with {overlap*100:.1f}% overlap")

    # If the image is smaller than the tile size, just use the whole image.
    if h <= tile_size and w <= tile_size:
        logger.info("Image is smaller than tile size, using entire image as a single tile.")
        return [image], [(slice(0, h), slice(0, w))]

    # Calculate effective step size based on overlap.
    step = int(tile_size * (1 - overlap))
    if step <= 0:
        logger.warning(f"Overlap too high ({overlap}), reducing to 0.8.")
        overlap = 0.8
        step = int(tile_size * (1 - overlap))

    # Calculate number of tiles in each dimension.
    n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
    n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)

    # If we'd create too many tiles, adjust the tile size or overlap.
    max_tiles = 100  # Arbitrary limit to prevent excessive memory usage.
    if n_h * n_w > max_tiles:
        logger.warning(f"Too many tiles ({n_h}×{n_w}={n_h*n_w}), adjusting parameters.")

        # Try increasing step size (reducing overlap).
        if overlap > 0.1:
            overlap = 0.1
            step = int(tile_size * (1 - overlap))
            n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
            n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)

            if n_h * n_w <= max_tiles:
                logger.info(f"Reduced overlap to {overlap:.1f}, new tile count: {n_h}×{n_w}={n_h*n_w}")
            else:
                # If still too many, increase tile size
                orig_tile_size = tile_size
                tile_size = min(h, w, tile_size * 2)
                step = int(tile_size * (1 - overlap))
                n_h = max(1, int(np.ceil((h - tile_size) / step)) + 1)
                n_w = max(1, int(np.ceil((w - tile_size) / step)) + 1)
                logger.info(f"Increased tile size from {orig_tile_size} to {tile_size}, new tile count: {n_h}×{n_w}={n_h*n_w}")

    logger.info(f"Creating {n_h}×{n_w}={n_h*n_w} tiles")

    tiles = []
    slices = []

    for i in range(n_h):
        for j in range(n_w):
            # Calculate tile boundaries
            y_start = min(i * step, h - tile_size) if h > tile_size else 0
            x_start = min(j * step, w - tile_size) if w > tile_size else 0
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Extract tile
            tile = image[y_start:y_end, x_start:x_end]

            # Handle tiles smaller than tile_size (at edges)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                # Create a new tile of the correct size
                new_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                new_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = new_tile

            tiles.append(tile)
            slices.append((slice(y_start, y_end), slice(x_start, x_end)))

    logger.info(f"Created {len(tiles)} tiles")
    return tiles, slices


"""MERGE_TILES_WITH_WEIGHTED_OVERLAP"""
########################################################################
#  Author : <your-initials>                                             #
#  Date   : 2025-04-30                                                  #
#                                                                      #
#  PURPOSE.                                                            #
#  ───────────────────────────────────────────────────────────────────  #
#  Re-assemble a set of overlapping flow or probability tiles into a   #
#  seamless panorama using a feathered α-mask. Each tile’s weight      #
#  tapers linearly to zero inside the overlap band, eliminating seams. #
########################################################################

def merge_tiles_with_weighted_overlap(
        tile_stack:      list[np.ndarray],
        slices:          list[tuple[slice, slice]],
        image_shape:     tuple[int, int],
        overlap:         float,
        logger:          logging.Logger | None = None,
        dtype:           np.float32 = np.float32
    ) -> np.ndarray:
    """
    Merge a list of overlapping flow- or probability-tiles back into one seamless field.

    This function is critical for reconstructing tiled segmentation results without visible
    seams or discontinuities. It uses a weighted blending approach where each tile's contribution
    tapers linearly to zero at its edges, creating smooth transitions in the overlap regions.
    This is particularly important for flow fields and probability maps where discontinuities
    would create artifacts in the final segmentation.

    Parameters:
        tile_stack : list[np.ndarray]
            Each element is either (2, H, W), (H, W, 2) or (H, W).
        slices : list[tuple[slice, slice]]
            The (row_slice, col_slice) that positions each tile on the canvas.
        image_shape : tuple[int, int]
            (H, W) of the original image.
        overlap : float
            Fractional overlap that was used when tiling (0–1).
        logger : Optional logger for DEBUG / INFO prints.
        dtype : Data type of the returned array (default float32).

    Returns:
        np.ndarray: (2, H, W) for vector fields or (H, W) for single-channel maps.
    """

    # Sanity checks to ensure inputs are valid.
    # Instead of assertion, print diagnostic information for better debugging.
    if len(tile_stack) != len(slices):
        raise ValueError(f"Mismatch: {len(tile_stack)} tiles vs {len(slices)} slices.")

    H, W = image_shape
    flow_accum   = None                              # Deferred allocation.
    weight_accum = np.zeros((H, W), dtype=dtype)

    # Helper function to create a 2D feather mask for smooth blending.
    def feather_mask(h: int, w: int, ov: float) -> np.ndarray:
        """
        Build a (h × w) mask that is 1.0 in the tile center and decays
        linearly to 0.0 at each border across an edge band of width
        edge = ov * size / 2. This creates a smooth transition between tiles.

        This approach is similar to feathering techniques used in image stitching
        and panorama creation, ensuring that the transition between tiles is
        imperceptible in the final result.
        """
        edge_h = max(1, int(ov * h / 2))
        edge_w = max(1, int(ov * w / 2))

        ramp_h = np.ones(h, dtype=dtype)
        ramp_w = np.ones(w, dtype=dtype)

        ramp_h[:edge_h]  = np.linspace(0.0, 1.0, edge_h,  endpoint=False)
        ramp_h[-edge_h:] = np.linspace(1.0, 0.0, edge_h,  endpoint=False)[::-1]

        ramp_w[:edge_w]  = np.linspace(0.0, 1.0, edge_w,  endpoint=False)
        ramp_w[-edge_w:] = np.linspace(1.0, 0.0, edge_w,  endpoint=False)[::-1]

        return np.outer(ramp_h, ramp_w)

    # Main accumulation loop to blend all tiles with their weights.
    for idx, (tile, slc) in enumerate(zip(tile_stack, slices), start=1):

        if logger:
            logger.debug(f"merge_tiles_with_weighted_overlap • tile {idx}/{len(tile_stack)}.")

        # Standardize tile format to (C, h, w) for consistent processing.
        if tile.ndim == 2:                          # (h, w)  → single-channel
            tile = tile[np.newaxis, ...]
        elif tile.ndim == 3:
            if tile.shape[0] <= 4:                  # (C, h, w) – channels FIRST
                pass
            else:                                   # (h, w, C) – channels LAST
                tile = tile.transpose(2, 0, 1)
        else:
            raise ValueError(f"Tile #{idx} has unsupported shape {tile.shape}.")


        tile = tile.astype(dtype, copy=False)
        C, th, tw = tile.shape

        if flow_accum is None:
            flow_accum = np.zeros((C, H, W), dtype=dtype)

        alpha = feather_mask(th, tw, overlap)
        alpha_broadcast = np.broadcast_to(alpha, (C, th, tw))

        rs, cs = slc
        flow_accum[:, rs, cs] += tile * alpha_broadcast
        weight_accum[rs, cs]  += alpha

    # Normalize the accumulated values by the weights to get the final blended result.
    nz = weight_accum > 0.0
    output = np.zeros_like(flow_accum, dtype=dtype)
    output[:, nz] = flow_accum[:, nz] / weight_accum[nz]

    if logger:
        logger.info(f"Merged {len(tile_stack)} tiles → {output.shape[0]}-channel field "
                    f"(overlap={overlap:.2f}).")

    # Return 2-D array for single-channel maps.
    return output[0] if output.shape[0] == 1 else output


def merge_masks(tiles, slices, image_shape, overlap, logger, settings):
    """
    Efficiently stitch Cellpose-generated tiled masks into a single label image.

    This function handles the challenging task of merging segmentation masks from multiple
    tiles while preserving object identity. Unlike flow fields which can be blended,
    segmentation masks contain discrete object IDs that must be carefully reconciled
    at tile boundaries to avoid duplicate or fragmented nuclei.

    The algorithm uses an overlap-based approach to determine when objects spanning
    multiple tiles should be merged or kept separate. This is critical for accurate
    quantification of nuclear features in densely packed kidney tissue regions.

    Args:
        tiles: List of segmentation mask tiles.
        slices: List of slice tuples for positioning each tile.
        image_shape: Shape of the output image (height, width).
        overlap: Fractional overlap between tiles.
        logger: Logger for progress information.
        settings: Dictionary containing configuration parameters including MERGE_OVERLAP_THRESHOLD.

    Returns:
        numpy.ndarray: Merged segmentation mask with consistent object IDs.
    """
    # Initialize the output mask
    merged_mask = np.zeros(image_shape, dtype=np.uint16)
    next_label = 1

    logger.info(f"Merging {len(tiles)} mask tiles with overlap={overlap:.2f}")

    # Simple approach: directly copy tiles to the output mask, handling overlaps.
    # This avoids complex relabeling that might cause issues.
    for i, (tile, slc) in enumerate(zip(tiles, slices)):
        if i % 10 == 0:
            logger.info(f"Processing tile {i+1}/{len(tiles)}")

        # Skip empty tiles.
        if np.max(tile) == 0:
            continue

        # Get the region in the merged mask where this tile will go
        mask_region = merged_mask[slc]

        # For each object in this tile, process it individually.
        for label in np.unique(tile)[1:]:  # Skip background (0).
            # Create a binary mask for this object to isolate it.
            obj_mask = (tile == label)

            # Check if this object overlaps with existing objects in the merged mask.
            overlap_mask = (mask_region > 0) & obj_mask

            if np.sum(overlap_mask) == 0:
                # No overlap, assign a new label to this object.
                mask_region[obj_mask] = next_label
                next_label += 1
            else:
                # There's overlap - check how much of the object overlaps with existing objects.
                overlap_ratio = np.sum(overlap_mask) / np.sum(obj_mask)

                # Use the user-defined threshold from settings for determining when to merge objects.
                if overlap_ratio < settings["MERGE_OVERLAP_THRESHOLD"]:  # Less than threshold overlap, treat as new object.
                    mask_region[obj_mask & ~overlap_mask] = next_label
                    next_label += 1
                else:
                    # Significant overlap - find the most overlapping existing label.
                    existing_labels = mask_region[overlap_mask]
                    unique_labels, counts = np.unique(existing_labels, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]

                    # Extend the existing object by assigning the same label.
                    mask_region[obj_mask & ~overlap_mask] = most_common_label

    # Count final objects
    unique_labels = np.unique(merged_mask)
    num_objects = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
    logger.info(f"Merged {len(tiles)} tiles → {num_objects} unique objects")

    return merged_mask


# =============================================================================
# CELLPOSE SEGMENTATION
# =============================================================================
def run_cellpose_on_tiles(model, image, cellpose_params, settings, logger):
    """
    Run Cellpose segmentation on an image, with optional tiling for large images.

    This function handles the core segmentation process, using either a direct Cellpose
    call for smaller images or a tiled approach for large images that would exceed GPU
    memory. The tiled approach segments each tile independently and then carefully
    merges the results to create a seamless final segmentation.

    For kidney tissue analysis, tiling is essential as whole-slide images often exceed
    several gigabytes in size, far beyond what can be processed in a single GPU pass.

    Args:
        model: Initialized Cellpose model.
        image: Input image as numpy array.
        cellpose_params: Dictionary of Cellpose parameters.
        settings: Dictionary of pipeline settings.
        logger: Logger for progress information.

    Returns:
        tuple: (masks, flows, num_cells) - Segmentation masks, flow fields, and cell count.
    """
    h, w = image.shape
    tile_side_length = settings["tile_side_length"]
    overlap = settings["TILE_OVERLAP"]
    use_tiling = settings["USE_TILING"] and (tile_side_length < h or tile_side_length < w)

    # ──────────────────────────────────────────────────────────
    # 1) Process without tiling - direct Cellpose call for smaller images
    # ──────────────────────────────────────────────────────────
    if not use_tiling:
        logger.info("Tiling disabled – processing full image.")

        # Add debug info about the image for troubleshooting.
        logger.info(f"Image shape: {image.shape}, min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}.")

        try:
            masks, flows, *_ = model.eval(
                image[..., None],                          # add channel axis
                diameter          = cellpose_params["diameter"],
                channels          = cellpose_params["channels"],
                flow_threshold    = cellpose_params["flow_threshold"],
                cellprob_threshold= cellpose_params["cellprob_threshold"],
                resample          = cellpose_params["resample"],
                augment           = False,
                batch_size        = cellpose_params["batch_size"],
                do_3D             = False
            )

            # Count cells and log information
            num_cells = len(np.unique(masks)) - 1 if 0 in np.unique(masks) else len(np.unique(masks))
            logger.info(f"Detected {num_cells} cells in full image")

            return masks, flows, num_cells                # ← flows already a list

        except Exception as e:
            logger.error(f"Error processing full image: {e}")
            # Return empty masks and flows
            empty_masks = np.zeros_like(image, dtype=np.uint16)
            empty_flows = [np.zeros((2, *image.shape), dtype=np.float32),
                          None,
                          np.zeros_like(image, dtype=np.float32)]
            return empty_masks, empty_flows, 0

    # ──────────────────────────────────────────────────────────
    # 2) Process with tiling - segment tiles independently then merge results
    # ──────────────────────────────────────────────────────────
    tiles, slices = split_image_into_tiles(image, tile_side_length, overlap, logger)
    logger.info(f"Processing {len(tiles)} tiles.")

    # storage
    mask_tiles        = []
    flow_xy_tiles     = []   # flows[0]  (2-ch)
    cellprob_tiles    = []   # flows[2]  (1-ch)
    total_cells       = 0

    for idx, tile in enumerate(tiles, start=1):
        logger.info(f"  ↳ tile {idx}/{len(tiles)}")

        # Add debug info about the tile for monitoring progress.
        logger.info(f"    Tile shape: {tile.shape}, min: {tile.min()}, max: {tile.max()}, mean: {tile.mean():.2f}.")

        # Run Cellpose on this tile
        try:
            masks, flows, *_ = model.eval(
                tile[..., None],  # Add channel dimension
                diameter          = cellpose_params["diameter"],
                channels          = cellpose_params["channels"],
                flow_threshold    = cellpose_params["flow_threshold"],
                cellprob_threshold= cellpose_params["cellprob_threshold"],
                resample          = cellpose_params["resample"],
                augment           = False,  # No augmentation for inference
                batch_size        = cellpose_params["batch_size"],
                do_3D             = False
            )

            # Log information about the segmentation results for quality monitoring.
            num_cells = len(np.unique(masks)) - 1 if 0 in np.unique(masks) else len(np.unique(masks))
            logger.info(f"    Detected {num_cells} cells in tile {idx}.")

            mask_tiles.append(masks)
            flow_xy_tiles.append(flows[0])                 # shape (2, h, w)
            cellprob_tiles.append(flows[2])                # shape (h, w)
            total_cells += num_cells

        except Exception as e:
            logger.error(f"Error processing tile {idx}: {e}")
            # Add an empty mask for this tile to maintain indexing
            mask_tiles.append(np.zeros_like(tile, dtype=np.uint16))
            # Add placeholder flows
            flow_xy_tiles.append(np.zeros((2, *tile.shape), dtype=np.float32))
            cellprob_tiles.append(np.zeros_like(tile, dtype=np.float32))

    # Stitch the tiled results into a seamless final segmentation
    merged_masks      = merge_masks(mask_tiles,  slices, image.shape, overlap, logger, settings)
    merged_flow_xy    = merge_tiles_with_weighted_overlap(flow_xy_tiles,  slices, image.shape, overlap, logger)
    merged_cellprob   = merge_tiles_with_weighted_overlap(cellprob_tiles, slices, image.shape, overlap, logger)

    # Ensure **same API** as vanilla Cellpose: a 3-element list
    merged_flows = [merged_flow_xy, merged_cellprob, None]

    return merged_masks, merged_flows, total_cells


# =============================================================================
# EDGE DETECTION REFINEMENT (OPTIONAL)
# =============================================================================
def refine_segmentation_with_edges(image, masks, settings, logger):
    """
    Refine segmentation masks using Canny edge detection.

    This function improves segmentation accuracy by incorporating edge information
    from the original image. In kidney tissue, nuclei often have distinct boundaries
    that may not be perfectly captured by Cellpose. By detecting these edges and
    using them to refine the segmentation, we can achieve more accurate delineation
    of nuclear boundaries, especially in densely packed regions.

    Args:
        image: Original grayscale image.
        masks: Initial segmentation masks from Cellpose.
        settings: Dictionary containing edge detection parameters.
        logger: Logger for progress information.

    Returns:
        numpy.ndarray: Refined segmentation masks with improved boundaries.
    """
    logger.info("Applying edge detection based refinement to the segmentation mask")
    edges = cv2.Canny(image,
                      threshold1=settings.get("CANNY_THRESHOLD1", 50),
                      threshold2=settings.get("CANNY_THRESHOLD2", 150))
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    binary_mask = (masks > 0).astype(np.uint8) * 255
    refined_mask = cv2.subtract(binary_mask, dilated_edges)
    num_labels, refined_labels = cv2.connectedComponents(refined_mask)
    logger.info(f"Refined segmentation into {num_labels - 1} objects after edge detection")
    return refined_labels


# =============================================================================
# LOCAL WATERSHED SPLITTING OF LARGE FUSED NUCLEI (OPTIONAL)
# =============================================================================
def identify_and_split_fused_labels(masks, min_area=1000, footprint=(3, 3), logger=None):
    """
    Identify large fused objects, apply local watershed, and reassign labels.

    This function addresses a common issue in nuclear segmentation: the merging of
    adjacent nuclei into single objects. This is particularly problematic in kidney
    tissue after I/R injury, where inflammatory infiltrates create densely packed
    nuclear clusters that Cellpose may fail to separate properly.

    The watershed algorithm uses distance transforms to identify likely centers of
    individual nuclei within merged objects, then separates them based on these centers.
    This is crucial for accurate quantification of nuclear counts and morphology in
    regions of high cellular density.

    Args:
        masks: Initial segmentation masks potentially containing fused objects.
        min_area: Minimum area threshold to identify potentially fused objects.
        footprint: Size of the local maxima filter for finding nuclear centers.
        logger: Optional logger for progress information.

    Returns:
        numpy.ndarray: Refined segmentation with split objects and reassigned labels.
    """
    # Initialize an empty mask to store the refined segmentation results.
    final_mask = np.zeros_like(masks, dtype=np.uint16)

    # Extract properties of all labeled regions in the input mask.
    props = regionprops(masks)

    # Initialize label counter for the new mask.
    current_label = 0

    # Process each region in the original segmentation mask.
    for prop in props:
        # Get region properties.
        area = prop.area
        minr, minc, maxr, maxc = prop.bbox  # Bounding box coordinates.

        # CASE 1: Small objects - keep as is (likely single nuclei).
        if area <= min_area:
            # Assign a new label to this region.
            current_label += 1

            # Copy the region to the final mask with the new label.
            final_mask[prop.coords[:, 0], prop.coords[:, 1]] = current_label

        # CASE 2: Large objects - apply watershed to split potential merged nuclei.
        else:
            if logger:
                logger.info(f"Applying watershed to region with area={area}, label={prop.label}.")

            # Extract the binary mask for this region.
            submask = prop.image.astype(bool)

            # Compute distance transform - each pixel value is the distance to the nearest background pixel.
            # This creates a topographic surface where nuclei centers are peaks.
            distance = ndi.distance_transform_edt(submask)

            # Find local maxima in the distance map - these are likely nuclei centers.
            # The footprint parameter controls the minimum separation between peaks.
            peaks = peak_local_max(distance, footprint=np.ones(footprint), labels=submask)

            # Create a marker image for watershed.
            marker = np.zeros(distance.shape, dtype=bool)

            # Place markers at the detected peaks (nuclei centers).
            if peaks.size > 0:
                marker[tuple(peaks.T)] = True

            # Label the markers for watershed.
            markers, _ = ndi.label(marker)

            # Apply watershed to split the merged nuclei.
            # The negative distance is used so that watershed finds boundaries at the lowest points
            # between peaks (likely the boundaries between touching nuclei).
            local_labels = watershed(-distance, markers, mask=submask)

            # Assign new labels to each watershed-separated region.
            for ul in np.unique(local_labels):
                # Skip background (label 0).
                if ul == 0:
                    continue

                # Assign a new unique label.
                current_label += 1

                # Create a mask for this specific sub-region.
                region_mask = local_labels == ul

                # Place the sub-region in the final mask at the correct position.
                # The bounding box coordinates are used to position the region correctly.
                final_mask[minr:maxr, minc:maxc][region_mask] = current_label

    return final_mask


# =============================================================================
# OVERLAY VISUALIZATION (OPTIONAL)
# =============================================================================
def generate_overlay(image, masks, flows, output_dir, logger):
    """
    Generate and save overlay visualizations for quality assessment.

    This function creates two types of visualization outputs that are essential for
    evaluating segmentation quality in kidney tissue analysis:

    1. A mask overlay showing segmented nuclei boundaries on the original image.
       This helps assess whether nuclear boundaries are accurately captured, which
       is critical for morphological analysis in I/R injury studies.

    2. A debug visualization showing both segmentation masks and flow fields.
       This helps diagnose issues with the Cellpose algorithm's gradient tracking
       that may lead to under- or over-segmentation in specific tissue contexts.

    Args:
        image: Original grayscale image of kidney tissue.
        masks: Segmentation masks from Cellpose or watershed refinement.
        flows: Flow fields from Cellpose (gradient and probability maps).
        output_dir: Directory where visualizations will be saved.
        logger: Logger for recording progress information.

    Returns:
        None. Visualization images are saved to the specified directory.
    """
    # Generate a color overlay of segmentation masks on the original image.
    # Each nucleus gets a random color for clear visual distinction.
    overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))

    # Save the overlay as a PNG image for easy viewing.
    overlay_path = os.path.join(output_dir, "mask_overlay.tif")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved overlay image to: {overlay_path}.")

    # Create a more detailed visualization showing both masks and flow fields.
    # This is useful for diagnosing segmentation issues related to the flow field.
    fig = plt.figure()
    plot.show_segmentation(fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])

    # Save the debug visualization at high resolution.
    debug_path = os.path.join(output_dir, "segmentation_debug.tif")
    fig.savefig(debug_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved segmentation debug overlay to: {debug_path}.")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    try:
        # Load configuration settings for the segmentation pipeline.
        SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()

        # Create output directory for storing results and visualizations.
        output_dir = SETTINGS["OUTPUT_DIR"]
        os.makedirs(output_dir, exist_ok=True)

        # Set up logging system for tracking progress and debugging.
        logger = setup_logging(output_dir, debug_mode=SETTINGS.get("DEBUG_MODE", False))
        logger.info("===== Cellpose Segmentation Pipeline Started =====")

        # Log configuration details for reproducibility.
        logger.info(f"Image path: {SETTINGS['IMAGE_PATH']}.")
        logger.info(f"Output directory: {output_dir}.")
        logger.info(f"Using tiling: {SETTINGS.get('USE_TILING', False)}.")
        if SETTINGS.get('USE_TILING', False):
            logger.info(f"Tile size: {SETTINGS.get('tile_side_length', 'Not specified')}.")
            logger.info(f"Tile overlap: {SETTINGS.get('TILE_OVERLAP', 'Not specified')}.")

        # Verify image path exists before proceeding.
        if not os.path.exists(SETTINGS["IMAGE_PATH"]):
            logger.error(f"Image file not found: {SETTINGS['IMAGE_PATH']}.")
            return 1

        # Back up the configuration file for reproducibility.
        config_backup_path = os.path.join(output_dir, "config_used.ini")
        shutil.copy2(PROJECT_DIRS["configs"] / "nuclei_segmentation_config.ini", config_backup_path)
        logger.info(f"Configuration backed up to: {config_backup_path}.")

        # 1. Preprocess the image to optimize for nuclei detection.
        logger.info("Preprocessing image...")
        image = preprocess_image(SETTINGS["IMAGE_PATH"], SETTINGS, logger)

        # 2. Initialize Cellpose model for deep learning-based segmentation.
        logger.info("Initializing Cellpose model...")

        # Ensure we're using the correct model type for kidney nuclei.
        model_type = CELLPOSE_PARAMS["model_type"]
        logger.info(f"Using Cellpose model: {model_type}.")

        # Initialize the model with pretrained weights optimized for nuclei.
        model = models.Cellpose(model_type=model_type, gpu=CELLPOSE_PARAMS["gpu"])

        # Log device information for performance tracking.
        logger.info(f"Using device: {'cuda' if CELLPOSE_PARAMS['gpu'] and torch.cuda.is_available() else 'cpu'}.")
        if CELLPOSE_PARAMS["gpu"] and torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}.")
        else:
            logger.info("GPU not available or not enabled, using CPU.")

        # 3. Run segmentation on the preprocessed image.
        logger.info("Running segmentation...")
        masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, SETTINGS, logger)

        # 4. Save results.
        logger.info("Saving segmentation results...")

        # Create masks directory.
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Save masks in both root and masks directory for compatibility.
        np.save(os.path.join(output_dir, "masks.npy"), masks)
        np.save(os.path.join(masks_dir, "masks.npy"), masks)

        # Save flows.
        np.savez(os.path.join(output_dir, "flows.npz"),
                 flow0=flows[0],
                 flow1=flows[1],
                 cellprob=flows[2])

        # Save visualization-friendly versions.
        skio.imsave(os.path.join(output_dir, "segmentation_mask.tif"), masks.astype(np.uint16))
        skio.imsave(os.path.join(masks_dir, "segmentation_mask.tif"), masks.astype(np.uint16))

        logger.info(f"Saved segmentation mask and flows. Total cells detected: {total_cells}")

        # 5. Optional: Edge detection refinement.
        if SETTINGS.get("USE_EDGE_DETECTION", False):
            logger.info("Applying edge detection refinement...")
            masks = refine_segmentation_with_edges(image, masks, SETTINGS, logger)
            skio.imsave(os.path.join(output_dir, "refined_segmentation_mask.tif"), masks.astype(np.uint16))
            logger.info("Saved refined segmentation mask after edge detection.")

        # 6. Optional: Watershed splitting.
        if SETTINGS.get("APPLY_WATERSHED", False):
            logger.info("Applying watershed splitting to large objects...")
            lumps_split_mask = identify_and_split_fused_labels(
                masks,
                min_area=SETTINGS.get("AREA_THRESHOLD_FOR_WATERSHED", 1000),
                footprint=SETTINGS.get("LOCAL_MAXIMA_FOOTPRINT", (3, 3)),
                logger=logger
            )
            skio.imsave(os.path.join(output_dir, "segmentation_mask_post_watershed.tif"),
                        lumps_split_mask.astype(np.uint16))
            np.save(os.path.join(output_dir, "segmentation_mask_post_watershed.npy"), lumps_split_mask)
            masks = lumps_split_mask
            logger.info("Saved watershed-processed segmentation mask.")

        # 7. Optional: Generate overlay visualization.
        if SETTINGS.get("GENERATE_OVERLAY", False):
            logger.info("Generating overlay visualization...")
            generate_overlay(image, masks, flows, output_dir, logger)

        # 8. Create a small overlay snippet (cropped) for quick review.
        logger.info("Generating small overlay snippet...")
        small_segmentation_overlay(output_dir, crop_size=SETTINGS.get("SMALL_OVERLAY_SIZE", 512) * SETTINGS.get("UPSCALE_FACTOR", 1))
        logger.info("Small overlay snippet generated successfully.")

        logger.info("===== Cellpose Segmentation Pipeline Completed Successfully =====")
        return 0
    except KeyError as e:
        logger.error(f"Configuration error: Missing required key {e}. Please check your configuration file.")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()
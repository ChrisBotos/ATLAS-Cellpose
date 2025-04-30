#!/usr/bin/env python3
"""
Improved Cellpose Segmentation Pipeline with Omnipose
and Optional Edge Detection, Watershed Splitting, and Tiling

This script preprocesses a large microscopy image, splits it into smaller tiles for segmentation,
performs cell segmentation using Cellpose's nuclei model (ideal for DAPI images), and optionally:
1) Refines segmentation boundaries using Canny edge detection.
2) Identifies large "fused" nuclei by area and splits them via watershed.

Results (segmentation masks, features, summary statistics, overlay images) are saved
to a defined output directory.
"""
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import models, plot
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.util import view_as_windows
from scipy import ndimage as ndi
from PIL import Image
import logging
import configparser
import torch
import shutil
from datetime import datetime
from pathlib import Path

# Import the small overlay snippet function
from utils.visualization import small_segmentation_overlay

# Increase the maximum allowed image pixels
Image.MAX_IMAGE_PIXELS = 10 ** 9

# =============================================================================
# PROJECT STRUCTURE SETUP
# =============================================================================
def setup_project_structure():
    """
    Create the project directory structure if it doesn't exist.
    
    Creates directories for configs, data, results, logs, and tests.
    Returns the paths to these directories as a dictionary.
    """
    # Define base directories
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
    
    # Create directories if they don't exist
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
        
    return dirs

# Setup project structure
PROJECT_DIRS = setup_project_structure()

# =============================================================================
# CONFIG LOADING
# =============================================================================
def load_config(config_path=None):
    """
    Load configuration from the specified INI file.
    
    If no config path is provided, uses the default config in the configs directory.
    Parses the configuration and returns SETTINGS and CELLPOSE_PARAMS dictionaries.
    """
    if config_path is None:
        config_path = PROJECT_DIRS["configs"] / "nuclei_segmentation_config.ini"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    config = configparser.ConfigParser()
    config.read(config_path)

    def get_bool(section, key):
        return config.get(section, key).lower() == "true"

    def get_tuple(section, key, cast_type=float):
        return tuple(cast_type(i.strip()) for i in config.get(section, key).split(','))

    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse settings from config
    SETTINGS = {
        "IMAGE_PATH": Path(config.get("General", "image_path")),
        "OUTPUT_DIR": PROJECT_DIRS["results"] / f"{config.get('General', 'output_dir')}_{timestamp}",
        "UPSCALE_FACTOR": config.getint("General", "upscale_factor"),
        "CROP_IMAGE": get_bool("General", "crop_image"),
        "CROP_BBOX": get_tuple("General", "crop_bbox", float),
        "ENHANCE_CONTRAST": get_bool("General", "enhance_contrast"),
        "ENHANCE_DIM": get_bool("General", "enhance_dim"),
        "GENERATE_OVERLAY": get_bool("General", "generate_overlay"),
        "CLAHE_CLIPLIMIT": config.getfloat("CLAHE", "cliplimit"),
        "CLAHE_TILE_GRID_SIZE": get_tuple("CLAHE", "tile_grid_size", int),
        "USE_EDGE_DETECTION": get_bool("EdgeDetection", "use_edge_detection"),
        "CANNY_THRESHOLD1": config.getint("EdgeDetection", "canny_threshold1"),
        "CANNY_THRESHOLD2": config.getint("EdgeDetection", "canny_threshold2"),
        "APPLY_WATERSHED": get_bool("Watershed", "apply_watershed"),
        "AREA_THRESHOLD_FOR_WATERSHED": config.getint("Watershed", "area_threshold"),
        "LOCAL_MAXIMA_FOOTPRINT": get_tuple("Watershed", "local_maxima_footprint", int),
        "USE_TILING": get_bool("Tiling", "use_tiling"),
        "tile_side_length": config.getint("Tiling", "tile_side_length"),
        "TILE_OVERLAP": config.getfloat("Tiling", "tile_overlap"),
        "SMALL_OVERLAY_SIZE": config.getint("Overlay", "small_overlay_size"),
        "DEBUG_MODE": get_bool("Debug", "debug_mode") if "Debug" in config.sections() else False,
    }

    # Ensure image path is absolute
    if not os.path.isabs(SETTINGS["IMAGE_PATH"]):
        SETTINGS["IMAGE_PATH"] = PROJECT_DIRS["data"] / SETTINGS["IMAGE_PATH"]

    CELLPOSE_PARAMS = {
        "model_type": config.get("Cellpose", "model_type"),
        "gpu": get_bool("Cellpose", "gpu") and torch.cuda.is_available(),
        "diameter": config.getint("Cellpose", "diameter"),
        "channels": get_tuple("Cellpose", "channels", int),
        "flow_threshold": config.getfloat("Cellpose", "flow_threshold"),
        "cellprob_threshold": config.getfloat("Cellpose", "cellprob_threshold"),
        "resample": get_bool("Cellpose", "resample"),
        "stitch_threshold": config.getfloat("Cellpose", "stitch_threshold"),
        "batch_size": 1  # Placeholder, will be updated dynamically
    }

    return SETTINGS, CELLPOSE_PARAMS

# Load config values from file
SETTINGS, CELLPOSE_PARAMS = load_config()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""DEBUG TAPS — remove once issue fixed."""  #TODO remove
import imageio, os
DBG_DIR = "dbg_snapshots"
os.makedirs(DBG_DIR, exist_ok=True)

def _snap(tag:str, arr:np.ndarray):
    """Write `arr` as uint8 PNG with `tag` in the filename."""
    if arr.dtype != np.uint8:
        v = arr.astype(np.float32)
        v = 255*(v - v.min())/(v.ptp()+1e-6)
        v = v.astype(np.uint8)
    else:
        v = arr
    imageio.imwrite(os.path.join(DBG_DIR, f"{tag}.png"), v)
    return arr        # so we can inline it



def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem_per_batch=150_000_000):
    """
    tile_pixels: number of pixels per patch (i.e. tile_side_length**2)
    bytes_per_pixel: 1 for uint8/float32≈4 (you may adjust)
    target_mem_per_batch: how much GPU memory (bytes) to devote per batch item
    """
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory  # in bytes
    # Reserve half the card for other stuff / headroom
    usable = total_mem // 2
    # Approximate bytes per patch: pixels × bytes_per_pixel.
    bytes_per_patch = tile_pixels * bytes_per_pixel
    # How many patches fit into target_mem_per_batch.
    max_batch = max(1, usable // (bytes_per_patch * (usable // target_mem_per_batch)))
    return int(max_batch)


# Example usage (tile_side_length=2048 → ~4.2M pixels):
tile_pixels = SETTINGS["tile_side_length"] ** 2
CELLPOSE_PARAMS["batch_size"] = choose_batch_size(tile_pixels)


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir, debug_mode=False):
    """
    Configure logging to console and file.
    
    Sets up a logger that writes to both a file in the output directory
    and to the console. If debug_mode is True, sets the log level to DEBUG.
    
    Args:
        output_dir: Directory where log file will be saved.
        debug_mode: If True, sets log level to DEBUG instead of INFO.
        
    Returns:
        logger: Configured logging instance.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory within output directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"segmentation_log_{timestamp}.txt")
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Remove any existing handlers to avoid double logging
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(file_formatter)
    fh.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_formatter)
    ch.setLevel(logging.INFO)  # Console always shows INFO and above
    logger.addHandler(ch)

    logger.info("===== Cellpose Segmentation Pipeline Started =====")
    logger.info(f"Log file: {log_file}")
    
    # Copy the config file to the output directory for reproducibility
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
    Set up debug environment if debug mode is enabled.
    
    Creates debug directory and returns a function for saving debug images.
    
    Args:
        settings: Dictionary containing configuration settings.
        
    Returns:
        snap_function: Function for saving debug images.
    """
    if not settings.get("DEBUG_MODE", False):
        # Return a no-op function if debug mode is disabled
        return lambda tag, arr: arr
    
    # Create debug directory within output directory
    debug_dir = os.path.join(settings["OUTPUT_DIR"], "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    def snap(tag, arr):
        """
        Save a debug image with the given tag.
        
        Args:
            tag: String identifier for the image.
            arr: Numpy array containing the image data.
            
        Returns:
            arr: The input array (for inline use).
        """
        if arr.dtype != np.uint8:
            v = arr.astype(np.float32)
            v = 255 * (v - v.min()) / (v.ptp() + 1e-6)
            v = v.astype(np.uint8)
        else:
            v = arr
        
        # Save with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%H%M%S")
        imageio.imwrite(os.path.join(debug_dir, f"{tag}_{timestamp}.png"), v)
        return arr  # Return the original array for inline use
    
    return snap

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def convert_16bit_to_8bit(image):
    """
    Convert a 16-bit image to 8-bit using percentile scaling.
    
    Scales the image using 1st and 99th percentiles to preserve dynamic range
    while converting from 16-bit to 8-bit depth.
    
    Args:
        image: Input image (numpy array).
        
    Returns:
        8-bit image as numpy array.
    """
    if image.dtype != np.uint16:
        return image
    p1, p99 = np.percentile(image, (1, 99))
    if p99 - p1 == 0:
        p1, p99 = image.min(), image.max()
    return np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


def adaptive_gamma_correction(image, min_gamma=1.5, max_gamma=2.5, logger=None):
    """
    Apply adaptive gamma correction based on the image median value.
    
    Adjusts gamma correction strength based on image brightness to enhance
    dim regions while preserving bright areas.
    
    Args:
        image: Input image (numpy array).
        min_gamma: Minimum gamma value.
        max_gamma: Maximum gamma
    """
    median = np.median(image) / 255.0
    gamma = np.clip(max_gamma - (max_gamma - min_gamma) * median, min_gamma, max_gamma)
    if logger:
        logger.info(f"Applying Gamma Correction with γ = {gamma:.2f}")
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_image(image_path, settings, logger):
    """Load and preprocess the image."""
    try:
        image = skio.imread(image_path)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        sys.exit(1)

    logger.info(f"Original dtype: {image.dtype}, shape: {image.shape}")

    # Remove alpha channel if present
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
        logger.info("Removed alpha channel")

    # Convert 16-bit to 8-bit if necessary
    if image.dtype == np.uint16:
        image = convert_16bit_to_8bit(image)
        cv2.imwrite(os.path.join(settings["OUTPUT_DIR"], "converted_8bit.tif"), image)
        logger.info("Converted 16-bit to 8-bit")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info("Converted to grayscale")

    if settings["CROP_IMAGE"]:
        # ── expect four numbers in INI: either fractions (0-1) or absolute pixels
        y0, y1, x0, x1 = settings.get("CROP_BBOX", (0, 1, 0, 1))
        h, w = image.shape
        if 0 <= y0 < 1:  # interpret as relative coordinates.
            y0, y1 = int(y0 * h), int(y1 * h)
            x0, x1 = int(x0 * w), int(x1 * w)
        image = image[y0:y1, x0:x1]
        logger.info(f"Cropped image to shape: {image.shape}")

    if settings["UPSCALE_FACTOR"] > 1:
        image = cv2.resize(image, None,
                           fx=settings["UPSCALE_FACTOR"],
                           fy=settings["UPSCALE_FACTOR"],
                           interpolation=cv2.INTER_LINEAR)
        logger.info(f"Upscaled image to: {image.shape}")

    skio.imsave(os.path.join(settings["OUTPUT_DIR"], "preprocessed_image.png"), image)
    logger.info("Saved preprocessed grayscale image")

    if settings["ENHANCE_CONTRAST"]:
        clahe = cv2.createCLAHE(clipLimit=settings["CLAHE_CLIPLIMIT"],
                                tileGridSize=settings["CLAHE_TILE_GRID_SIZE"])
        image = clahe.apply(image)
        skio.imsave(os.path.join(settings["OUTPUT_DIR"], "contrast_enhanced_image.png"), image)
        logger.info("Applied CLAHE contrast enhancement")

    if settings["ENHANCE_DIM"]:
        image = adaptive_gamma_correction(image, min_gamma=1.2, max_gamma=1.5, logger=logger)
        skio.imsave(os.path.join(settings["OUTPUT_DIR"], "gamma_corrected_image.png"), image)
        logger.info("Applied gamma correction")

    _snap("00_after_preproc", image)  # writes dbg_snapshots/00_after_preproc.png

    return image


def split_image_into_tiles(image, tile_side_length, overlap, logger):
    """
    Split the image into overlapping tiles, ensuring complete coverage.
    
    This function divides an image into tiles of specified size with overlap,
    handling edge cases where image dimensions aren't divisible by tile size.
    Tiles at the right and bottom edges may be smaller than the specified size.
    
    Args:
        image: Input 2D image (grayscale).
        tile_side_length: Size of each tile (pixels).
        overlap: Fractional overlap between tiles (e.g., 0.1 for 10% overlap).
        logger: Logger instance for logging.
        
    Returns:
        tiles: List of tiles as numpy arrays.
        slices: List of slice objects for reconstructing the full image.
    """
    h, w = image.shape
    if tile_side_length > h or tile_side_length > w:
        logger.warning(f"Tile size {tile_side_length} is larger than image dimensions ({h}, {w}). Adjusting tile size.")
        tile_side_length = min(h, w)

    step = int(tile_side_length * (1 - overlap))
    logger.info(f"Splitting image into tiles with size {tile_side_length} and step {step}")
    
    # Calculate positions for tile starting points
    y_positions = list(range(0, h - tile_side_length + 1, step))
    x_positions = list(range(0, w - tile_side_length + 1, step))
    
    # Ensure we cover the entire image by adding the final position if needed
    if h > tile_side_length and y_positions[-1] + tile_side_length < h:
        y_positions.append(h - tile_side_length)
    if w > tile_side_length and x_positions[-1] + tile_side_length < w:
        x_positions.append(w - tile_side_length)
    
    # Extract tiles and record their positions
    tiles = []
    slices = []
    
    for y in y_positions:
        for x in x_positions:
            tile = image[y:y+tile_side_length, x:x+tile_side_length]
            tiles.append(tile)
            slices.append((slice(y, y+tile_side_length), slice(x, x+tile_side_length)))
    
    logger.info(f"Created {len(tiles)} tiles covering the {h}×{w} image.")
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

import numpy as np
import logging

def merge_tiles_with_weighted_overlap(
        tile_stack:      list[np.ndarray],
        slices:          list[tuple[slice, slice]],
        image_shape:     tuple[int, int],
        overlap:         float,
        logger:          logging.Logger | None = None,
        dtype:           np.float32 = np.float32
    ) -> np.ndarray:
    """
    Merge a list of overlapping flow- or probability-tiles back into one field.

    Parameters.
    ───────────
    tile_stack : list[np.ndarray]
        Each element is either (2, H, W), (H, W, 2) or (H, W).
    slices     : list[tuple[slice, slice]]
        The (row_slice, col_slice) that positions each tile on the canvas.
    image_shape: (H, W) of the original image.
    overlap    : Fractional overlap that was used when tiling (0–1).
    logger     : Optional logger for DEBUG / INFO prints.
    dtype      : Data type of the returned array (default float32).

    Returns.
    ────────
    np.ndarray
        (2, H, W) for vector fields or (H, W) for single-channel maps.
    """

    '''Sanity checks.'''
    assert len(tile_stack) == len(slices), \
        "tile_stack and slices must have equal length."

    H, W = image_shape
    flow_accum   = None                              # Deferred allocation.
    weight_accum = np.zeros((H, W), dtype=dtype)

    '''Helper: 2-D feather mask.'''
    def _feather_mask(h: int, w: int, ov: float) -> np.ndarray:
        """
        Build a (h × w) mask that is 1.0 in the tile centre and decays
        linearly to 0.0 at each border across an edge band of width
        edge = ov * size / 2.
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

    '''Main accumulation loop.'''
    for idx, (tile, slc) in enumerate(zip(tile_stack, slices), start=1):

        if logger:
            logger.debug(f"merge_tiles_with_weighted_overlap • tile {idx}/{len(tile_stack)}.")

        # --- standardise to (C, h, w) ---------------------------------------
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

        alpha = _feather_mask(th, tw, overlap)
        alpha_broadcast = np.broadcast_to(alpha, (C, th, tw))

        rs, cs = slc
        flow_accum[:, rs, cs] += tile * alpha_broadcast
        weight_accum[rs, cs]  += alpha

    '''Normalisation.'''
    nz = weight_accum > 0.0
    output = np.zeros_like(flow_accum, dtype=dtype)
    output[:, nz] = flow_accum[:, nz] / weight_accum[nz]

    if logger:
        logger.info(f"Merged {len(tile_stack)} tiles → {output.shape[0]}-channel field "
                    f"(overlap={overlap:.2f}).")

    # Return 2-D array for single-channel maps.
    return output[0] if output.shape[0] == 1 else output


def merge_masks(
        tiles:        list[np.ndarray],
        slices:       list[tuple[slice, slice]],
        image_shape:  tuple[int, int],
        overlap:      float,
        logger,
        merge_overlap_thresh: float = 0.50
    ) -> np.ndarray:
    """
    Stitch Cellpose-generated tiled masks into a single label image.

    Parameters
    ----------
    tiles : list[np.ndarray]
        2-D integer masks coming back from Cellpose (one per tile).
    slices : list[tuple[slice, slice]]
        The (row_slice, col_slice) used to place each tile in the canvas.
    image_shape : tuple
        Height × width of the original image.
    overlap : float
        Fractional tile overlap (not used here but kept for API compatibility).
    logger : logging.Logger
        For nice, centralised reporting.
    merge_overlap_thresh : float, optional
        Two labels are considered the *same* object when

        .. math::
            \\frac{|A \\cap B|}{\\min(|A|, |B|)} \\ge \\text{merge_overlap_thresh}

        where :math:`A` and :math:`B` are the pixel sets of the two labels
        inside the *overlap region only*.

    Returns
    -------
    merged : np.ndarray
        Global mask with unique, compact, 1-based labels.
    """
    merged_mask = np.zeros(image_shape, dtype=np.uint16)

    # Next free global label (0 is background).
    next_label: int = 1

    for tile, slc in zip(tiles, slices):
        tile = tile.astype(np.uint16)
        canvas_view = merged_mask[slc]              # view into the big canvas

        # ---------------------------------------------------------------------
        # 1. Decide which *tile* labels should be mapped onto *existing* labels
        #    and which should receive a new global ID.
        # ---------------------------------------------------------------------
        relabel: dict[int, int] = {}                # tile_id ➜ global_id

        # Pixels where *both* the canvas and the tile already have labels:
        overlap_mask = (canvas_view > 0) & (tile > 0)

        if np.any(overlap_mask):
            # Existing labels *in the overlap only*
            existing = np.unique(canvas_view[overlap_mask])
            existing = existing[existing > 0]

            for t_lbl in np.unique(tile[overlap_mask]):
                if t_lbl == 0:
                    continue
                t_mask = tile == t_lbl               # full tile-sized mask

                # Pixels of this tile label that actually lie inside the overlap
                overlap_pixels = overlap_mask & t_mask
                if not np.any(overlap_pixels):
                    continue

                # Find *one* global label (if any) that matches above threshold.
                best_match, best_score = None, 0.0
                for e_lbl in existing:
                    e_mask = canvas_view == e_lbl

                    intersect = np.logical_and(t_mask, e_mask).sum()
                    if intersect == 0:
                        continue
                    score = intersect / min(t_mask.sum(), e_mask.sum())
                    if score > best_score:
                        best_score, best_match = score, e_lbl

                if best_score >= merge_overlap_thresh:
                    relabel[t_lbl] = best_match

        # ---------------------------------------------------------------------
        # 2. Apply the relabel map or assign a fresh ID, then copy into canvas.
        # ---------------------------------------------------------------------
        for t_lbl in np.unique(tile):
            if t_lbl == 0:
                continue
            if t_lbl not in relabel:
                relabel[t_lbl] = next_label
                next_label += 1
            tile[tile == t_lbl] = relabel[t_lbl]

        # Finally write the (re-id’ed) tile back to the canvas
        canvas_view[tile > 0] = tile[tile > 0]

    logger.info(
        "Merged %d tiles ➜ %d unique objects "
        "(overlap-threshold = %.2f).",
        len(tiles), next_label - 1, merge_overlap_thresh,
    )
    return merged_mask


# =============================================================================
# CELLPOSE SEGMENTATION
# =============================================================================
def run_cellpose_on_tiles(model, image, cellpose_params, settings, logger):
    h, w          = image.shape
    tile_side_length     = settings["tile_side_length"]
    overlap       = settings["TILE_OVERLAP"]
    use_tiling    = settings["USE_TILING"] and (tile_side_length < h or tile_side_length < w)

    # ──────────────────────────────────────────────────────────
    # 1)  ***NO TILING***  →  straight Cellpose call, nothing to merge
    # ──────────────────────────────────────────────────────────
    if not use_tiling:
        logger.info("Tiling disabled – processing full image.")
        masks, flows, *_ = model.eval(
            image[..., None],                          # add channel axis
            diameter          = cellpose_params["diameter"],
            channels          = cellpose_params["channels"],
            flow_threshold    = cellpose_params["flow_threshold"],
            cellprob_threshold= cellpose_params["cellprob_threshold"],
            resample          = cellpose_params["resample"],
            augment=False,
            batch_size        = cellpose_params["batch_size"],
            do_3D=False
        )
        total_cells = np.count_nonzero(masks)
        return masks, flows, total_cells                # ← flows already a list

    # ──────────────────────────────────────────────────────────
    # 2)  ***WITH TILING***  →  run tiles, then stitch
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
        masks, flows, *_ = model.eval(
            tile[..., None],
            diameter          = cellpose_params["diameter"],
            channels          = cellpose_params["channels"],
            flow_threshold    = cellpose_params["flow_threshold"],
            cellprob_threshold= cellpose_params["cellprob_threshold"],
            resample          = cellpose_params["resample"],
            augment=False,
            batch_size        = cellpose_params["batch_size"],
            do_3D=False
        )
        mask_tiles.append(masks)
        flow_xy_tiles.append(flows[0])                 # shape (2, h, w)
        cellprob_tiles.append(flows[2])                # shape (h, w)
        total_cells += np.count_nonzero(masks)

    # ── stitch the results ───────────────────────────────────
    merged_masks      = merge_masks(mask_tiles,  slices, image.shape, overlap, logger)
    merged_flow_xy    = merge_tiles_with_weighted_overlap(flow_xy_tiles,  slices, image.shape, overlap, logger)
    merged_cellprob   = merge_tiles_with_weighted_overlap(cellprob_tiles, slices, image.shape, overlap, logger)

    # Ensure **same API** as vanilla Cellpose: a 3-element list
    merged_flows = [merged_flow_xy, merged_cellprob, None]

    return merged_masks, merged_flows, total_cells


# =============================================================================
# EDGE DETECTION REFINEMENT (OPTIONAL)
# =============================================================================
def refine_segmentation_with_edges(image, masks, settings, logger):
    """Refine segmentation masks using Canny edge detection."""
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
    """Identify large fused objects, apply local watershed, and reassign labels."""
    final_mask = np.zeros_like(masks, dtype=np.uint16)
    props = regionprops(masks)
    current_label = 0
    for prop in props:
        area = prop.area
        minr, minc, maxr, maxc = prop.bbox
        if area <= min_area:
            current_label += 1
            final_mask[prop.coords[:, 0], prop.coords[:, 1]] = current_label
        else:
            if logger:
                logger.info(f"Applying watershed to region with area={area}, label={prop.label}")
            submask = prop.image.astype(bool)
            distance = ndi.distance_transform_edt(submask)
            from skimage.feature import peak_local_max
            peaks = peak_local_max(distance, footprint=np.ones(footprint), labels=submask)
            marker = np.zeros(distance.shape, dtype=bool)
            if peaks.size > 0:
                marker[tuple(peaks.T)] = True
            markers, _ = ndi.label(marker)
            local_labels = watershed(-distance, markers, mask=submask)
            for ul in np.unique(local_labels):
                if ul == 0:
                    continue
                current_label += 1
                region_mask = local_labels == ul
                final_mask[minr:maxr, minc:maxc][region_mask] = current_label
    return final_mask


# =============================================================================
# OVERLAY VISUALIZATION (OPTIONAL)
# =============================================================================
def generate_overlay(image, masks, flows, output_dir, logger):
    """Generate and save overlay visualizations."""
    overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3))
    overlay_path = os.path.join(output_dir, "mask_overlay.png")
    skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
    logger.info(f"Saved overlay image to: {overlay_path}")

    fig = plt.figure()
    plot.show_segmentation(fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])
    debug_path = os.path.join(output_dir, "segmentation_debug.png")
    fig.savefig(debug_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved segmentation debug overlay to: {debug_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    output_dir = SETTINGS["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, debug_mode=SETTINGS.get("DEBUG_MODE", False))
    print("2DEBUG: dir made...")

    # Set up debug environment if enabled
    snap = setup_debug(SETTINGS)

    # 1. Preprocess the image.
    image = preprocess_image(SETTINGS["IMAGE_PATH"], SETTINGS, logger)
    print("3DEBUG: preprocess done...")

    # 2. Segment image by tiling or as a single tile.
    model = models.Cellpose(model_type=CELLPOSE_PARAMS["model_type"],
                            gpu=CELLPOSE_PARAMS["gpu"])
    print("4DEBUG: model made...")

    logger.info(f"Using device: {'cuda' if CELLPOSE_PARAMS['gpu'] else 'cpu'}")
    if CELLPOSE_PARAMS["gpu"]:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, SETTINGS, logger)
    print("5DEBUG: cellpose3 done...")

    # Save the merged mask and flows.
    np.save(os.path.join(output_dir, "masks.npy"), masks)
    np.savez(os.path.join(output_dir, "flows.npz"),
             flow0=flows[0],
             flow1=flows[1],
             cellprob=flows[2])
    skio.imsave(os.path.join(output_dir, "segmentation_mask.png"), masks.astype(np.uint16))
    logger.info(f"Saved segmentation mask and flows. Total cells detected: {total_cells}")
    print("6DEBUG: saving masks done...")

    # 3. Optionally refine segmentation using edge detection.
    if SETTINGS["USE_EDGE_DETECTION"]:
        masks = refine_segmentation_with_edges(image, masks, SETTINGS, logger)
        skio.imsave(os.path.join(output_dir, "refined_segmentation_mask.png"), masks.astype(np.uint16))
        logger.info("Saved refined segmentation mask after edge detection")

    # 4. Optionally apply watershed splitting to large fused nuclei.
    if SETTINGS["APPLY_WATERSHED"]:
        lumps_split_mask = identify_and_split_fused_labels(
            masks,
            min_area=SETTINGS["AREA_THRESHOLD_FOR_WATERSHED"],
            footprint=SETTINGS["LOCAL_MAXIMA_FOOTPRINT"],
            logger=logger
        )
        skio.imsave(os.path.join(output_dir, "segmentation_mask_post_watershed.png"),
                    lumps_split_mask.astype(np.uint16))
        np.save(os.path.join(output_dir, "segmentation_mask_post_watershed.npy"), lumps_split_mask)
        masks = lumps_split_mask

    # 5. Optionally generate overlay visualization.
    if SETTINGS["GENERATE_OVERLAY"]:
        generate_overlay(image, masks, flows, output_dir, logger)

    # 6. Create a small overlay snippet (cropped) for quick review.
    small_segmentation_overlay(output_dir, crop_size=SETTINGS["SMALL_OVERLAY_SIZE"] * SETTINGS["UPSCALE_FACTOR"])


if __name__ == "__main__":
    print("1DEBUG: test starting...")
    import numpy as np

    # Fake 2×2 tiling of a 6×6 image (3×3 tiles, 1-pixel overlap)
    img_shape = (6, 6)
    tiles = [
        np.array([[1, 1, 0],
                  [1, 1, 0],
                  [0, 0, 0]], dtype=np.uint16),  # top-left
        np.array([[0, 2, 2],
                  [0, 2, 2],
                  [0, 0, 0]], dtype=np.uint16),  # top-right
        np.array([[0, 0, 0],
                  [3, 3, 0],
                  [3, 3, 0]], dtype=np.uint16),  # bottom-left
        np.array([[0, 0, 0],
                  [0, 4, 4],
                  [0, 4, 4]], dtype=np.uint16)  # bottom-right
    ]
    slices = [
        (slice(0, 3), slice(0, 3)),
        (slice(0, 3), slice(2, 5)),
        (slice(2, 5), slice(0, 3)),
        (slice(2, 5), slice(2, 5)),
    ]



    merged = merge_masks(tiles, slices, img_shape, overlap=1 / 3, logger=logging.getLogger(__name__))
    assert merged.max() == 4, "Labels should remain distinct with zero mutual overlap"

    main()

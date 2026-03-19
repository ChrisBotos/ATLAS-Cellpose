"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: preprocessing.py.
Description:
    Image preprocessing utilities for nuclear segmentation of tissue samples.
    Provides 16-bit to 8-bit conversion, adaptive gamma correction, CLAHE
    contrast enhancement, shape correction, ROI cropping, and robust image
    saving to structured folders with logging support.

Dependencies:
    - Python >= 3.10.
    - numpy, opencv-python, scikit-image, tifffile.

Usage:
    from utils.preprocessing import preprocess_image
    image = preprocess_image(image_path, settings, logger)
"""

import numpy as np
import cv2
from skimage import io as skio
import tifffile as tiff
import tempfile
from pathlib import Path

'''BIT DEPTH CONVERSION'''

def convert_to_8bit(
    image: np.ndarray,
    *,
    p_low: float | None = None,
    p_high: float | None = None,
    sample_fraction: float = 0.01,
    logger=None,
) -> np.ndarray:
    """
    Convert 16-/32-bit microscopy frames to uint8.

    • If *p_low*/*p_high* are provided, they are used verbatim (deterministic).
    • Otherwise the 0.5 / 99.5 percentiles are estimated from a random subset.
    """
    if image.dtype == np.uint8:
        return image

    if image.dtype not in (np.uint16, np.uint32):
        raise TypeError(f"Unsupported dtype {image.dtype}. Expected uint8/16/32.")

    # Percentile estimation (only when caller did not supply them).
    if (p_low is None) or (p_high is None):
        total_px = image.size
        rng = np.random.default_rng(0)

        if sample_fraction < 1.0:
            n_samples = max(10_000, int(total_px * sample_fraction))
            idx = rng.choice(total_px, n_samples, replace=False)
            sample = image.reshape(-1)[idx]
        else:
            sample = image

        p_low, p_high = np.percentile(sample, (0.5, 99.5))
        if logger:
            logger.debug(
                f"[16->8] sampled p0.5={p_low:.1f}, p99.5={p_high:.1f} "
                f"(n={sample.size:,})"
            )
    else:
        if logger:
            logger.debug(
                f"[16->8] using global p0.5={p_low:.1f}, p99.5={p_high:.1f}"
            )

    if p_high <= p_low:
        p_low, p_high = int(image.min()), int(image.max())
        if p_high <= p_low:
            return np.zeros_like(image, dtype=np.uint8)

    img_f32 = (image.astype(np.float32) - p_low) / (p_high - p_low)
    img_u8 = np.clip(img_f32, 0, 1) * 255
    return img_u8.astype(np.uint8)


'''GAMMA CORRECTION'''

def adaptive_gamma_correction(image: np.ndarray, min_gamma: float = 1.9, max_gamma: float = 2.2, logger=None) -> np.ndarray:
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

def apply_clahe_no_corner(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    logger=None,
) -> np.ndarray:
    """
    Contrast-enhance *img* with CLAHE while avoiding the corner-tile artefact.

    The function no longer pads the image to make it divisible by the grid;
    OpenCV’s CLAHE interpolates per-tile LUTs internally, so padding is
    unnecessary and was the real cause of the residual “box” in the bottom
    right corner.

    If the image is smaller than the requested number of tiles in either
    dimension, the grid is reduced so that every tile keeps at least
    ``MIN_PX_PER_TILE`` pixels in width and height.

    Args:
        img (np.ndarray): Single-channel 8-bit image.
        clip_limit (float): CLAHE clip limit.
        tile_grid_size (tuple[int,int]): Desired number of tiles (rows, cols).
        logger (logging.Logger | None): Optional logger.

    Returns:
        np.ndarray: CLAHE-enhanced image, same dtype and shape as *img*.
    """
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("apply_clahe_no_corner expects a 2-D uint8 array.")

    h, w = img.shape
    n_rows, n_cols = tile_grid_size
    MIN_PX_PER_TILE = 32          # Prevent 1-px tiles on very small patches.

    # Adapt grid if the patch is smaller than requested.
    n_rows = max(1, min(n_rows, h // MIN_PX_PER_TILE))
    n_cols = max(1, min(n_cols, w // MIN_PX_PER_TILE))
    if logger and (n_rows, n_cols) != tile_grid_size:
        logger.debug(
            f"CLAHE grid reduced from {tile_grid_size} to {(n_rows, n_cols)} "
            f"for a {h}×{w} patch."
        )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(n_cols, n_rows))
    return clahe.apply(img)


'''CROPPING UTILS'''

def crop_image(image: np.ndarray, crop_box, logger=None) -> np.ndarray:
    """
    Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_box: Tuple of (x0, x1, y0, y1), either relative (0–1) or absolute.
        logger: Logger object.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape
    x0, x1, y0, y1 = crop_box

    if all(0 <= val <= 1 for val in crop_box):
        x0, x1 = int(x0 * w), int(x1 * w)
        y0, y1 = int(y0 * h), int(y1 * h)
        if logger:
            logger.info(f"Cropping with relative bbox: (x={x0}:{x1}, y={y0}:{y1})")
    else:
        x0, x1, y0, y1 = map(int, crop_box)
        if logger:
            logger.info(f"Cropping with absolute bbox: (x={x0}:{x1}, y={y0}:{y1})")

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

def _lazy_tiff_memmap(path: str, logger):
    """
    Return a numpy.memmap view of the first page of a TIFF – no full read.
    """
    try:
        with tiff.TiffFile(path) as tif:
            page = tif.series[0]          # Assumes single-page WSI / slide.
            arr  = page.asarray(out='memmap')
            logger.info(
                f"Lazy-mapped TIFF: {arr.shape} {arr.dtype} @ {arr.nbytes/1e9:.1f} GB"
            )
            return arr
    except Exception as e:
        logger.error(f"TIFF mem-map failed: {e}")
        raise

def _tile_iter(h, w, tile=8192, overlap=0):
    """
    Generate (y0,y1,x0,x1) windows that cover the image without padding.
    """
    if tile >= h and tile >= w:
        yield 0, h, 0, w
        return

    step = tile - overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            yield y0, min(y0 + tile, h), x0, min(x0 + tile, w)

def preprocess_image(image_path, settings, logger):
    """
    STREAMING version that keeps peak RSS <~2-4 GB even for 80k×80k slides.
    """

    image_path = str(image_path)
    if image_path.lower().endswith((".tif", ".tiff")):
        img = _lazy_tiff_memmap(image_path, logger)
    else:
        img = skio.imread(image_path)     # Non-TIFF formats.
        logger.warning("Non-TIFF input read fully into RAM.")

    logger.info(f"Raw TIFF -> ndim={img.ndim}, shape={img.shape}, dtype={img.dtype}")
    if img.ndim == 3:
        # Assume channels last; if channels first, swapaxes first.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        logger.warning("Converted multi-channel TIFF to single-channel grayscale.")

    out_dir = Path(settings["output_dir"]) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional cropping happens on the mem-map directly.
    if settings.get("crop_image", False):
        crop_box = settings.get("crop_box", (0, 1, 0, 1))
        if isinstance(crop_box, str):
            crop_box = [float(x.strip()) for x in crop_box.split(',')]
        img = crop_image(img, crop_box, logger)  # Still mem-mapped slice.

    # Save a quick preview of the cropped ROI for human QC.
    skio.imsave(out_dir / "first.tif", img, plugin="tifffile")

    H, W = img.shape[:2]
    logger.info(f"Working shape post-crop: {H}×{W}, dtype={img.dtype}")

    # One-off global 0.5 / 99.5 % for deterministic scaling.─
    flat = img.reshape(-1)
    stride = max(1, flat.size // 1_000_000)  # ~1 M evenly spaced samples
    sample = flat[::stride]
    p_low, p_high = np.percentile(sample, (0.5, 99.5))
    logger.info(f"[global 16->8] p0.5={p_low:.1f}, p99.5={p_high:.1f}")

    # Tile-wise 8 bit conversion, CLAHE & gamma (if requested).
    tile_px  = settings.get("tile_side_length", 8192)
    overlap  = int(tile_px * 0.05)

    # Create temporary file with better Windows compatibility.
    try:
        # Use the output directory for temporary files instead of system temp.
        temp_dir = Path(settings["output_dir"]) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file_path = temp_dir / "eight_bit_processing.dat"

        out_mm = np.memmap(
            str(temp_file_path),  # Convert to string for Windows compatibility.
            dtype=np.uint8,
            mode="w+",
            shape=(img.shape[0], img.shape[1]),
        )
        scratch = None  # We'll clean up manually.
        temp_file_to_cleanup = temp_file_path

    except Exception as e:
        logger.error(f"Failed to create temporary memory-mapped file: {e}")
        logger.info("Falling back to in-memory processing (may use more RAM)")
        # Fallback to in-memory processing.
        out_mm = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        scratch = None
        temp_file_to_cleanup = None

    for y0, y1, x0, x1 in _tile_iter(H, W, tile_px, overlap):
        tile = img[y0:y1, x0:x1]

        # 1. Bit-depth.
        tile_u8 = convert_to_8bit(tile, p_low=p_low, p_high=p_high, logger=logger)

        # 2. Optional CLAHE.
        if settings.get("enhance_contrast", False):
            tile_u8 = apply_clahe_no_corner(
                tile_u8,
                clip_limit=settings.get("clahe_cliplimit", 2.0),
                tile_grid_size=settings.get("clahe_tile_grid_size", (8, 8)),
                logger=logger,
            )

        # 3. Optional gamma.
        if settings.get("enhance_dim", False):
            tile_u8 = adaptive_gamma_correction(
                tile_u8,
                min_gamma=settings.get("min_gamma", 1.9),
                max_gamma=settings.get("max_gamma", 2.2),
                logger=logger,
            )

        out_mm[y0:y1, x0:x1] = tile_u8  # Write-back.

    out_mm.flush()                       # Ensure data hits disk.
    img_u8 = np.asarray(out_mm)          # Cheap view; no copy.
    skio.imsave(out_dir / "contrast_and_gc.tif", img_u8)
    logger.info(f"Wrote possibly contrast-enhanced and possibly gamma-corrected image to {out_dir/'contrast_and_gc.tif'}")

    factor = float(settings.get("upscale_factor", 1))
    if factor > 1.0:
        new_h = int(round(img_u8.shape[0] * factor))
        new_w = int(round(img_u8.shape[1] * factor))
        img_up = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        logger.info(f"Upscaled pre-processed image -> {img_up.shape} (factor={factor:g})")

        # Save both the native-res and the up-scaled version for traceability.
        out_dir = Path(settings["output_dir"]) / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        skio.imsave(out_dir / "upscaled.tif", img_up)

        # Clean up temporary file if it exists.
        if 'temp_file_to_cleanup' in locals() and temp_file_to_cleanup and temp_file_to_cleanup.exists():
            try:
                temp_file_to_cleanup.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file_to_cleanup}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
        return img_up


    # Save once for downstream modules.
    out_dir = Path(settings["output_dir"]) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(out_dir / "final.tif", img_u8)
    logger.info(f"Wrote pre-processed slide to {out_dir/'final.tif'}")

    # Clean up temporary file if it exists.
    if 'temp_file_to_cleanup' in locals() and temp_file_to_cleanup and temp_file_to_cleanup.exists():
        try:
            temp_file_to_cleanup.unlink()
            logger.debug(f"Cleaned up temporary file: {temp_file_to_cleanup}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {e}")

    return img_u8

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
                f"[16→8] sampled p0.5={p_low:.1f}, p99.5={p_high:.1f} "
                f"(n={sample.size:,})"
            )
    else:
        if logger:
            logger.debug(
                f"[16→8] using global p0.5={p_low:.1f}, p99.5={p_high:.1f}"
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

def crop_image(image: np.ndarray, crop_box, logger=None) -> np.ndarray:
    """
    Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_box: Tuple of (y0, y1, x0, x1), either relative (0–1) or absolute.
        logger: Logger object.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape
    y0, y1, x0, x1 = crop_box

    if all(0 <= val <= 1 for val in crop_box):
        y0, y1 = int(y0 * h), int(y1 * h)
        x0, x1 = int(x0 * w), int(x1 * w)
        if logger:
            logger.info(f"Cropping with relative bbox: ({y0}:{y1}, {x0}:{x1})")
    else:
        y0, y1, x0, x1 = map(int, crop_box)
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

    logger.info(f"Raw TIFF → ndim={img.ndim}, shape={img.shape}, dtype={img.dtype}")
    if img.ndim == 3:
        # Assume channels last; if channels first, swapaxes first.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        logger.warning("Converted multi-channel TIFF to single-channel grayscale.")

    out_dir = Path(settings["output_dir"]) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional cropping happens on the mem-map directly ───────────────────
    if settings.get("crop_image", False):
        crop_box = settings.get("crop_box", (0, 1, 0, 1))
        if isinstance(crop_box, str):
            crop_box = [float(x.strip()) for x in crop_box.split(',')]
        img = crop_image(img, crop_box, logger)  # Still mem-mapped slice.

    # Save a quick preview of the cropped ROI for human QC.
    skio.imsave(out_dir / "first.tig", img, plugin="tifffile")

    H, W = img.shape[:2]
    logger.info(f"Working shape post-crop: {H}×{W}, dtype={img.dtype}")

    # ── one-off global 0.5 / 99.5 % for deterministic scaling ────────────────
    flat = img.reshape(-1)
    stride = max(1, flat.size // 1_000_000)  # ~1 M evenly spaced samples
    sample = flat[::stride]
    p_low, p_high = np.percentile(sample, (0.5, 99.5))
    logger.info(f"[global 16→8] p0.5={p_low:.1f}, p99.5={p_high:.1f}")

    # ── Tile-wise 8 bit conversion, CLAHE & gamma (if requested) ─────────
    tile_px  = settings.get("tile_side_length", 8192)
    overlap  = int(tile_px * 0.05)
    scratch  = tempfile.TemporaryDirectory(prefix="iri_pre_")
    out_mm   = np.memmap(
        Path(scratch.name) / "eight_bit.dat",
        dtype=np.uint8,
        mode="w+",
        shape=(img.shape[0], img.shape[1]),
    )

    for y0, y1, x0, x1 in _tile_iter(H, W, tile_px, overlap):
        tile = img[y0:y1, x0:x1]

        # 1. Bit-depth.
        tile_u8 = convert_to_8bit(tile, p_low=p_low, p_high=p_high, logger=logger)

        # 2. Optional CLAHE.
        if settings.get("enhance_contrast", False):
            tile_u8 = apply_clahe(
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

    # Full-slide CLAHE (once, not per tile)
    if settings.get("enhance_contrast", False):
        clahe_full = apply_clahe(
            img_u8,
            clip_limit=settings.get("clahe_cliplimit", 2.0),
            tile_grid_size=settings.get("clahe_tile_grid_size", (8, 8)),
            logger=logger,
        )
        skio.imsave(out_dir / "clahe.tif", clahe_full)
        logger.info(f"Wrote CLAHE-only image to {out_dir / 'clahe.tif'}")

    # Full-slide Gamma (once, on the CLAHE image if available)
    if settings.get("enhance_dim", False):
        base = clahe_full if "clahe_full" in locals() else img_u8
        gamma_full = adaptive_gamma_correction(
            base,
            min_gamma=settings.get("min_gamma", 1.9),
            max_gamma=settings.get("max_gamma", 2.2),
            logger=logger,
        )
        skio.imsave(out_dir / "gamma.tif", gamma_full)
        logger.info(f"Wrote gamma-corrected image to {out_dir / 'gamma.tif'}")

    factor = float(settings.get("upscale_factor", 1))
    if factor > 1.0:
        new_h = int(round(img_u8.shape[0] * factor))
        new_w = int(round(img_u8.shape[1] * factor))
        img_up = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        logger.info(f"Upscaled pre-processed image → {img_up.shape} (factor={factor:g})")

        # Save both the native-res and the up-scaled version for traceability.
        out_dir = Path(settings["output_dir"]) / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        skio.imsave(out_dir / "final.tif", img_u8)
        skio.imsave(out_dir / "upscaled.tif", img_up)

        scratch.cleanup()
        return img_up


    # Save once for downstream modules.
    out_dir = Path(settings["output_dir"]) / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(out_dir / "final.tif", img_u8)
    logger.info(f"Wrote pre-processed slide to {out_dir/'final.tif'}")

    scratch.cleanup()
    return img_u8

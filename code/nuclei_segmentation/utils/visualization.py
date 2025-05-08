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
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import plot
import logging
import traceback
from pathlib import Path


def setup_logger(name: str, debug: bool = False, log_file: Path = None) -> logging.Logger:
    """
    Creates and configures a logger for visualization functions.

    Parameters:
    name (str): Name of the logger.
    debug (bool): If True, sets level to DEBUG, otherwise INFO.
    log_file (Path or None): Optional path to log file.

    Returns:
    logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Always set level explicitly
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Define common formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Remove any duplicate handlers of the same type
    existing_types = set()
    for handler in list(logger.handlers):
        if type(handler) in existing_types:
            logger.removeHandler(handler)
        else:
            existing_types.add(type(handler))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        try:
            log_file = Path(log_file).expanduser().resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to create log file handler at {log_file}: {e}")

    # Prevent logs from propagating to root
    logger.propagate = False

    return logger


def create_output_dirs(output_dir: Path, debug: bool, logger) -> tuple[Path, Path]:
    """
    Ensures creation of debug and visualization output directories with fallbacks.

    Parameters:
        output_dir (Path): Base output directory.
        debug (bool): Whether to create debug folder.
        logger (logging.Logger): Logger for reporting.

    Returns:
        tuple: (check_dir, debug_dir) as Path objects.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    debug_dir = None

    if debug:
        debug_dir = output_dir / "debug_snapshots"

        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created debug snapshots directory: {debug_dir}")
        except Exception as e:
            logger.error(f"Error creating debug directory: {e}")
            debug_dir = output_dir

    check_dir = output_dir / "visualizations" / "cropped_preview"
    try:
        check_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created visualization directory: {check_dir}")
    except Exception as e:
        logger.error(f"Failed to create main visualization directory: {e}")

        fallback_dirs = [
            output_dir / "cropped_preview",
            output_dir / "visualizations",
            output_dir
        ]

        for fallback in fallback_dirs:

            try:
                fallback.mkdir(parents=True, exist_ok=True)
                check_dir = fallback
                logger.warning(f"Using fallback directory: {check_dir}")
                break
            except Exception:
                continue

    if not check_dir.exists():
        logger.critical(f"Could not access or create any visualization directory under {output_dir}")
        raise PermissionError("No writable visualization directory available.")

    return check_dir, debug_dir

def load_image(image_paths: list[Path], debug_dir: Path, logger) -> np.ndarray | None:
    """
    Attempts to load an image from the provided list of paths.
    Logs diagnostic information and saves histogram/normalization preview if debug is enabled.

    Parameters:
        image_paths (list[Path]): List of possible image file paths.
        debug_dir (Path): Directory to save debug outputs (can be None).
        logger (Logger): Logger instance for diagnostics.

    Returns:
        np.ndarray | None: Loaded image or None if all failed.
    """

    for path in image_paths:

        try:
            img = skio.imread(str(path))
            logger.info(f"Successfully loaded image: {path}")
            logger.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
            logger.info(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

            if debug_dir:
                # Save histogram of pixel values.
                plt.figure(figsize=(10, 3))
                plt.hist(img.ravel(), bins=256)
                plt.title(f"Histogram of {path.name}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                hist_path = debug_dir / f"histogram_{path.name}.tif"
                plt.savefig(hist_path)
                plt.close()
                logger.debug(f"Saved image histogram to: {hist_path}")

                # Save normalized preview.
                norm = img.copy()
                if norm.max() > 1.0:
                    p2, p98 = np.percentile(norm, (2, 98))
                    norm = np.clip((norm - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
                norm = norm.astype(np.uint8)
                norm_path = debug_dir / f"normalized_{path.name}"
                skio.imsave(str(norm_path), norm)
                logger.debug(f"Saved normalized preview to: {norm_path}")

            return img

        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            continue

    logger.error("All image paths failed to load.")
    return None

def load_masks(mask_paths: list[Path], debug_dir: Path, logger) -> np.ndarray | None:
    """
    Attempts to load segmentation masks from a list of paths.
    Logs basic stats and optionally saves a mask histogram and visual overlay.

    Parameters:
        mask_paths (list[Path]): List of mask .npy files.
        debug_dir (Path): Debug output directory (or None).
        logger (Logger): Logger for diagnostics.

    Returns:
        np.ndarray | None: Loaded mask array or None on failure.
    """

    for path in mask_paths:

        try:
            masks = np.load(str(path))
            logger.info(f"Loaded mask: {path}")
            unique = np.unique(masks)
            num = len(unique) - (1 if 0 in unique else 0)
            logger.info(f"Mask shape: {masks.shape}, dtype: {masks.dtype}, objects: {num}")

            if debug_dir:
                # Histogram of labels.
                plt.figure(figsize=(10, 3))
                plt.hist(masks.ravel(), bins=min(100, num + 1))
                plt.title(f"Label Histogram: {path.name}")
                plt.xlabel("Mask Label")
                plt.ylabel("Pixel Count")
                hist_path = debug_dir / f"mask_hist_{path.name}.tif"
                plt.savefig(hist_path)
                plt.close()
                logger.debug(f"Saved mask histogram to: {hist_path}")

                # Visualization of segmentation.
                if num > 0:
                    colors = np.random.rand(masks.max() + 1, 3)
                    mask_rgb = np.zeros((*masks.shape, 3), dtype=np.uint8)
                    for i in range(1, masks.max() + 1):
                        mask_rgb[masks == i] = (colors[i] * 255).astype(np.uint8)
                    vis_path = debug_dir / f"mask_vis_{path.stem}.tif"
                    skio.imsave(str(vis_path), mask_rgb)
                    logger.debug(f"Saved mask visualization to: {vis_path}")

            return masks

        except Exception as e:
            logger.warning(f"Failed to load mask from {path}: {e}")
            continue

    logger.error("All mask paths failed to load.")
    return None

def generate_tiled_overlay(img, masks, logger, tile_size=2048, overlap=128):
    """
    Generate a full-size overlay by tiling the image and stitching overlays.

    Parameters:
        img (np.ndarray): Grayscale image.
        masks (np.ndarray): Segmentation mask.
        logger (Logger): Logger instance.
        tile_size (int): Size of each tile (default: 2048).
        overlap (int): Overlap between tiles (default: 128).

    Returns:
        np.ndarray: Full overlay image (float32, RGB).
    """

    assert img.shape == masks.shape, "Image and mask must have same shape."

    H, W = img.shape
    overlay = np.zeros((*img.shape[:2], 3), dtype=np.uint8)

    # Precompute color map.
    num_labels = masks.max()
    if num_labels == 0:
        logger.warning("No mask labels found; skipping tiled overlay.")
        return np.stack([img / 255.0] * 3, axis=-1)

    np.random.seed(42)
    colors = np.random.rand(num_labels + 1, 3)
    np.random.seed(None)

    logger.info(f"Generating tiled overlay: shape={img.shape}, overlay_tile_size={tile_size}, overlay_overlap={overlap}")
    step = tile_size - overlap
    weight = np.zeros((H, W, 1), dtype=np.float32)

    for y in range(0, H, step):
        for x in range(0, W, step):
            y0 = y
            x0 = x
            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            img_tile = img[y0:y1, x0:x1]
            mask_tile = masks[y0:y1, x0:x1]

            try:
                tile_overlay = plot.mask_overlay(img_tile, mask_tile, colors=colors)
                overlay[y0:y1, x0:x1] += tile_overlay
                weight[y0:y1, x0:x1] += 1
            except Exception as e:
                logger.warning(f"Failed tile overlay at y={y0}:{y1}, x={x0}:{x1} — {e}")

    # Normalize to prevent over-blending at overlaps.
    weight = np.clip(weight, 1e-3, None)
    overlay = (overlay.astype(np.float32) / weight).astype(np.uint8)

    logger.info("Successfully generated tiled overlay.")
    return overlay


# def normalize_if_dark(img: np.ndarray, logger) -> np.ndarray:
#     """
#     Applies contrast normalization to dark images (mean < 50 for uint8).
#
#     Parameters:
#         img (np.ndarray): Input image.
#         logger (Logger): Logger for reporting.
#
#     Returns:
#         np.ndarray: Contrast-normalized image (if needed).
#     """
#     if img.dtype != np.uint8:
#         logger.debug("Skipping normalization: image is not uint8.")
#         return img
#
#     mean_val = img.mean()
#     if mean_val >= 50:
#         logger.debug(f"Image brightness is acceptable (mean={mean_val:.2f}).")
#         return img
#
#     logger.warning(f"Image appears dark (mean={mean_val:.2f}). Applying normalization.")
#     try:
#         flat_img = img.ravel()
#         sample_size = min(len(flat_img), 1_000_000)
#         sample = np.random.choice(flat_img, sample_size, replace=False)
#         p2, p98 = np.percentile(sample, (2, 98))
#
#         img_float = img.astype(np.float32)
#         img_norm = np.clip((img_float - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
#         img_out = img_norm.astype(np.uint8)
#
#         logger.info(f"Image normalized: min={img_out.min()}, max={img_out.max()}, mean={img_out.mean():.2f}")
#         return img_out
#     except Exception as e:
#         logger.error(f"Normalization failed: {e}")
#         return img


def match_shapes(img: np.ndarray, masks: np.ndarray, logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops image and masks to a common shape if they mismatch.

    Parameters:
        img (np.ndarray): Input image.
        masks (np.ndarray): Segmentation masks.
        logger (Logger): Logger instance.

    Returns:
        tuple: (img_cropped, masks_cropped)
    """
    if img.shape == masks.shape:
        return img, masks

    logger.warning(f"Shape mismatch: image {img.shape}, masks {masks.shape}")
    min_h = min(img.shape[0], masks.shape[0])
    min_w = min(img.shape[1], masks.shape[1])

    logger.info(f"Cropping both arrays to common shape: {min_h}x{min_w}")

    img_cropped = img[:min_h, :min_w]
    masks_cropped = masks[:min_h, :min_w]

    return img_cropped, masks_cropped


def choose_crop_region(img: np.ndarray, crop_size: int, logger) -> tuple[int, int, int, int]:
    """
    Chooses a crop region based on intensity content.

    Parameters:
        img (np.ndarray): Image to crop.
        crop_size (int): Crop size in pixels.
        logger (Logger): Logger.

    Returns:
        tuple: (y0, y1, x0, x1)
    """
    h, w = img.shape[:2]

    if h <= crop_size and w <= crop_size:
        logger.info("Image is small, using full region.")
        return 0, h, 0, w

    # Smart sampling (user ROI + 9-point check).
    candidates = [
        (int(h * 0.5), int(w * 0.66)),  # User-specified bias.
        (h // 2, w // 2),               # Center.
        (h // 4, w // 4),
        (h // 4, 3 * w // 4),
        (3 * h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
        (h // 2, w // 4),
        (h // 2, 3 * w // 4),
        (h // 4, w // 2),
        (3 * h // 4, w // 2),
    ]

    best_score = -1
    best_coords = (h // 2 - crop_size // 2, w // 2 - crop_size // 2)

    for cy, cx in candidates:
        y0 = max(0, cy - crop_size // 2)
        x0 = max(0, cx - crop_size // 2)
        y1 = min(h, y0 + crop_size)
        x1 = min(w, x0 + crop_size)
        crop = img[y0:y1, x0:x1]
        score = crop.mean()

        logger.debug(f"Crop at ({y0}:{y1}, {x0}:{x1}) has mean {score:.2f}")

        if score > best_score:
            best_score = score
            best_coords = (y0, x0)

    y0, x0 = best_coords
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)

    logger.info(f"Chosen crop: y={y0}:{y1}, x={x0}:{x1}, mean={best_score:.2f}")
    return y0, y1, x0, x1


def crop_array(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, name: str, logger) -> np.ndarray:
    """
    Safely crops an array and logs the outcome.

    Parameters:
        arr (np.ndarray): Input array.
        y0, y1, x0, x1 (int): Crop coordinates.
        name (str): For logging.
        logger (Logger): Logger.

    Returns:
        np.ndarray: Cropped array.
    """
    try:
        cropped = arr[y0:y1, x0:x1]
        logger.info(f"Cropped {name} to shape {cropped.shape}")
        return cropped
    except Exception as e:
        logger.error(f"Failed to crop {name}: {e}")
        return np.zeros((y1 - y0, x1 - x0), dtype=arr.dtype)


def create_overlay(img_crop: np.ndarray, masks_crop: np.ndarray, logger) -> np.ndarray:
    """
    Creates an RGB overlay image from grayscale input and segmentation masks.

    Parameters:
        img_crop (np.ndarray): Cropped grayscale image.
        masks_crop (np.ndarray): Cropped masks.
        logger (Logger): Logger.

    Returns:
        np.ndarray: RGB overlay image in float32 [0, 1].
    """
    num_labels = masks_crop.max()

    if num_labels == 0:
        logger.warning("No masks found in cropped region.")
        overlay = np.stack([img_crop / 255.0] * 3, axis=-1).astype(np.float32)
        return overlay

    logger.info(f"Creating overlay for {num_labels} mask labels.")
    np.random.seed(42)
    colors = np.random.rand(num_labels + 1, 3)
    np.random.seed(None)

    total_px = img_crop.shape[0] * img_crop.shape[1]
    try:
        if total_px > 5_000_000:
            logger.info("Large image detected. Creating overlay manually.")
            overlay = np.stack([img_crop / 255.0] * 3, axis=-1).astype(np.float32)
            for i in range(1, num_labels + 1):
                mask = masks_crop == i
                for c in range(3):
                    overlay[..., c][mask] = (
                        0.8 * colors[i, c] + 0.2 * overlay[..., c][mask]
                    )
        else:
            overlay = plot.mask_overlay(img_crop, masks_crop, colors=colors)

        if overlay.mean() < 0.2:
            logger.warning("Overlay too dark. Adjusting brightness.")
            bright = np.clip(img_crop.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
            overlay = plot.mask_overlay(bright, masks_crop, colors=colors)

        return overlay

    except Exception as e:
        logger.error(f"Overlay creation failed: {e}")
        return np.stack([img_crop / 255.0] * 3, axis=-1).astype(np.float32)


def save_overlay_summary(img_crop, overlay, clahe_crop, gamma_crop, output_dir, logger, debug=False):
    """
    Saves a 2x2 summary figure showing preprocessing steps and overlay,
    plus separate image panels and an optional high-contrast overlay.

    Parameters:
    img_crop (np.ndarray): Cropped grayscale input image.
    overlay (np.ndarray): RGB overlay image (float32 [0,1] or uint8).
    clahe_crop (np.ndarray or None): CLAHE-enhanced crop if available.
    gamma_crop (np.ndarray or None): Gamma-corrected crop if available.
    output_dir (Path): Path object where images will be saved.
    logger (Logger): Logger for reporting.
    debug (bool): If True, saves a high-contrast overlay and extra plots.
    """

    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(img_crop, cmap="gray")
        axes[0, 0].set_title("Preprocessed Image")
        axes[0, 0].axis("off")

        if clahe_crop is not None:
            try:
                axes[0, 1].imshow(clahe_crop, cmap="gray")
                axes[0, 1].set_title("CLAHE Enhanced")
            except Exception as e:
                logger.warning(f"CLAHE display error: {e}")
                axes[0, 1].text(0.5, 0.5, "CLAHE Error", ha="center", va="center")
        else:
            axes[0, 1].text(0.5, 0.5, "No CLAHE", ha="center", va="center")
        axes[0, 1].axis("off")

        if gamma_crop is not None:
            try:
                axes[1, 0].imshow(gamma_crop, cmap="gray")
                axes[1, 0].set_title("Gamma Corrected")
            except Exception as e:
                logger.warning(f"Gamma display error: {e}")
                axes[1, 0].text(0.5, 0.5, "Gamma Error", ha="center", va="center")
        else:
            axes[1, 0].text(0.5, 0.5, "No Gamma", ha="center", va="center")
        axes[1, 0].axis("off")

        try:
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title("Segmentation Overlay")
        except Exception as e:
            logger.warning(f"Overlay display error: {e}")
            axes[1, 1].text(0.5, 0.5, "Overlay Error", ha="center", va="center")
        axes[1, 1].axis("off")

        plt.tight_layout()
        fig.savefig(output_dir / "quick_overlay_summary.tif", dpi=300)
        plt.close(fig)
        logger.info("Saved 2x2 overlay summary.")

    except Exception as e:
        logger.error(f"Summary figure failed: {e}")
        logger.error(traceback.format_exc())


    """Save panels individually"""
    try:
        skio.imsave(output_dir / "panel1_preprocessed.tif", img_crop)

        if clahe_crop is not None:
            skio.imsave(output_dir / "panel2_clahe.tif", clahe_crop)

        if gamma_crop is not None:
            skio.imsave(output_dir / "panel3_gamma.tif", gamma_crop)

        if overlay is not None:
            overlay_uint8 = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 else overlay.astype(np.uint8)
            skio.imsave(output_dir / "panel4_overlay.tif", overlay_uint8)

        logger.info("Saved individual image panels.")

        if debug:
            # High-contrast enhancement.
            try:
                contrast = overlay.copy()
                mask = (contrast.sum(axis=-1) > 0)
                contrast[mask] = np.clip(contrast[mask] * 1.5, 0, 1)
                contrast_img = (contrast * 255).astype(np.uint8)
                skio.imsave(output_dir / "debug_high_contrast_overlay.tif", contrast_img)
                logger.debug("Saved high-contrast overlay (debug).")
            except Exception as e:
                logger.warning(f"Failed to create high-contrast overlay: {e}")

    except Exception as e:
        logger.warning(f"Failed to save image panels: {e}")


    """Side-by-side 1×2 layout"""
    try:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_crop, cmap="gray")
        axes[0].set_title("Original Image (Crop)")
        axes[0].axis("off")

        try:
            axes[1].imshow(overlay)
            axes[1].set_title("Segmentation Overlay")
        except Exception:
            axes[1].text(0.5, 0.5, "Overlay Error", ha="center", va="center")
        axes[1].axis("off")

        plt.tight_layout()
        fig2.savefig(output_dir / "central_crop_comparison.tif", dpi=300)
        plt.close(fig2)
        logger.info("Saved 1x2 comparison image.")

    except Exception as e:
        logger.warning(f"Side-by-side comparison failed: {e}")


def small_segmentation_overlay(output_dir, crop_size=1024, debug=False):
    """
    Creates a quick visualization of a central or high-content crop from segmentation output.

    This function loads the preprocessed image and final segmentation mask, extracts a central crop,
    generates overlays (including CLAHE and gamma-corrected versions if available), and saves them.

    Parameters:
        output_dir (str or Path): Directory containing preprocessed image and mask outputs.
        crop_size (int): Size of the crop in pixels (default: 1024).
        debug (bool): If True, enables more verbose logging and saves extra debug images.
    """


    """SETUP"""
    logger = setup_logger("small_segmentation_overlay", debug=debug)
    output_dir = Path(output_dir).expanduser().resolve()
    logger.info(f"Running overlay visualization on: {output_dir}")

    try:
        check_dir, debug_dir = create_output_dirs(output_dir, debug, logger)
    except Exception as e:
        logger.error(f"Cannot create output folders: {e}")
        return


    """FIXED PATHS"""
    img_path = output_dir / "preprocessed" / "cropped.tif"
    clahe_path = output_dir / "preprocessed" / "clahe.tif"
    gamma_path = output_dir / "preprocessed" / "gamma.tif"
    mask_path = output_dir / "masks" / "segmentation_masks.npy"


    """LOAD MAIN IMAGE"""
    if not img_path.exists():
        logger.error(f"Preprocessed image not found: {img_path}")
        return

    img = load_image([img_path], debug_dir, logger)
    if img is None:
        return


    """LOAD MASK"""
    if not mask_path.exists():
        logger.error(f"Segmentation mask not found: {mask_path}")
        return

    masks = load_masks([mask_path], debug_dir, logger)
    if masks is None:
        return


    """SHAPE SYNC"""
    img, masks = match_shapes(img, masks, logger)


    """CROP CENTER REGION"""
    y0, y1, x0, x1 = choose_crop_region(img, crop_size, logger)
    img_crop = crop_array(img, y0, y1, x0, x1, name="image", logger=logger)
    masks_crop = crop_array(masks, y0, y1, x0, x1, name="mask", logger=logger)

    try:
        skio.imsave(check_dir / "cropped_image.tif", img_crop)
        logger.info(f"Saved cropped image to: {check_dir / 'cropped_image.tif'}")
    except Exception as e:
        logger.error(f"Could not save cropped image: {e}")
        return

    if np.max(masks_crop) == 0:
        logger.warning("No mask labels found in the cropped region.")
        return


    """GENERATE OVERLAY"""
    overlay = create_overlay(img_crop, masks_crop, logger)
    try:
        overlay_path = check_dir / "central_crop_overlay.tif"
        skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
        logger.info(f"Saved overlay to: {overlay_path}")
    except Exception as e:
        logger.warning(f"Could not save overlay: {e}")


    """OPTIONAL ENHANCEMENT CROPS"""
    clahe_img = None
    gamma_img = None

    if clahe_path.exists():
        try:
            clahe_img = skio.imread(str(clahe_path))
            clahe_img = crop_array(clahe_img, y0, y1, x0, x1, "CLAHE image", logger)
        except Exception as e:
            logger.warning(f"Failed to load/crop CLAHE image: {e}")
    else:
        logger.info("No CLAHE image found.")

    if gamma_path.exists():
        try:
            gamma_img = skio.imread(str(gamma_path))
            gamma_img = crop_array(gamma_img, y0, y1, x0, x1, "Gamma image", logger)
        except Exception as e:
            logger.warning(f"Failed to load/crop gamma image: {e}")
    else:
        logger.info("No Gamma image found.")


    """SUMMARY PANEL"""
    save_overlay_summary(img_crop, overlay, clahe_img, gamma_img, check_dir, logger, debug=debug)


    """IMAGE OVERLAY"""
    try:
        logger.info("Creating full-size overlay.")
        full_overlay = generate_tiled_overlay(img, masks, logger)
        skio.imsave(check_dir / "full_image_overlay.tif", (full_overlay * 255).astype(np.uint8))
        logger.info("Saved full image overlay.")
    except Exception as e:
        logger.warning(f"Failed to create full-size overlay: {e}")

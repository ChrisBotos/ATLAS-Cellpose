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
import traceback
from pathlib import Path

def _setup_logger(name: str, debug: bool = False, log_file: Path = None) -> logging.Logger:
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


def small_segmentation_overlay(output_dir, crop_size=1024, debug=False):
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
        debug: Enable detailed debug information and additional diagnostic outputs (default: False).

    Returns:
        None. Visualization images are saved to the output directory.
    """
    # Set up logging with appropriate level based on debug parameter
    logger = logging.getLogger("small_segmentation_overlay")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

    # Print diagnostic information about the output directory.
    logger.info(f"Working with output directory: {os.path.abspath(output_dir)}")

    # Create debug snapshots directory if in debug mode.
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug_snapshots")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            logger.debug(f"Created debug snapshots directory: {debug_dir}")
        except Exception as e:
            logger.error(f"Error creating debug directory: {e}")
            debug_dir = output_dir  # Fallback to main output directory

    # Subdirectory for storing the check images - this is critical
    check_dir = os.path.join(output_dir, "visualizations", "cropped_preview")
    try:
        os.makedirs(check_dir, exist_ok=True)
        logger.info(f"Created visualization directory: {check_dir}")
    except Exception as e:
        logger.error(f"Error creating visualization directory: {e}")
        # Try alternative paths as fallbacks
        fallback_dirs = [
            os.path.join(output_dir, "cropped_preview"),
            os.path.join(output_dir, "visualizations"),
            output_dir
        ]

        for fallback_dir in fallback_dirs:

            try:
                os.makedirs(fallback_dir, exist_ok=True)
                check_dir = fallback_dir
                logger.warning(f"Using fallback directory: {check_dir}")
                break
            except Exception:
                continue

    # Verify the directory exists and is writable
    if not os.path.exists(check_dir):
        logger.error(f"Critical error: Could not create or access directory: {check_dir}")
        return

    # Try to write a test file to verify permissions
    test_file = os.path.join(check_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test write permission")
        os.remove(test_file)  # Clean up
        logger.info(f"Verified write permission to: {check_dir}")
    except Exception as e:
        logger.error(f"Cannot write to directory {check_dir}: {e}")
        logger.error("Visualization will likely fail due to permission issues")

    # Define all possible image paths with both .png and .tif extensions.
    # For the main preprocessed image
    preprocessed_paths = [
        os.path.join(output_dir, "preprocessed", "preprocessed_image.png"),
        os.path.join(output_dir, "preprocessed", "preprocessed_image.tif"),
        os.path.join(output_dir, "preprocessed_image.png"),
        os.path.join(output_dir, "preprocessed_image.tif")
    ]

    # For the CLAHE enhanced image
    clahe_paths = [
        os.path.join(output_dir, "preprocessed", "contrast_enhanced_image.png"),
        os.path.join(output_dir, "preprocessed", "contrast_enhanced_image.tif"),
        os.path.join(output_dir, "contrast_enhanced_image.png"),
        os.path.join(output_dir, "contrast_enhanced_image.tif")
    ]

    # For the gamma corrected image
    gamma_paths = [
        os.path.join(output_dir, "preprocessed", "gamma_corrected_image.png"),
        os.path.join(output_dir, "preprocessed", "gamma_corrected_image.tif"),
        os.path.join(output_dir, "gamma_corrected_image.png"),
        os.path.join(output_dir, "gamma_corrected_image.tif")
    ]

    # Combined list for general image loading
    possible_image_paths = preprocessed_paths + [
        os.path.join(output_dir, "preprocessed", "cropped_image.png"),
        os.path.join(output_dir, "preprocessed", "cropped_image.tif"),
        os.path.join(output_dir, "cropped_image.png"),
        os.path.join(output_dir, "cropped_image.tif")
    ]

    # Define all possible mask paths.
    possible_mask_paths = [
        os.path.join(output_dir, "masks.npy"),
        os.path.join(output_dir, "segmentation_mask_post_watershed.npy"),
        os.path.join(output_dir, "masks", "masks.npy"),
        os.path.join(output_dir, "masks", "segmentation_mask_post_watershed.npy")
    ]

    # Debug: List all files in the output directory to help diagnose issues
    logger.info("Searching for image and mask files in output directory...")
    found_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".tif") or file.endswith(".npy"):
                file_path = os.path.join(root, file)
                found_files.append(file_path)
                logger.debug(f"Found file: {file_path}")

    logger.info(f"Found {len(found_files)} relevant files in output directory")

    # Group files by type for better organization in logs
    png_files = [f for f in found_files if f.endswith(".png")]
    tif_files = [f for f in found_files if f.endswith(".tif")]
    npy_files = [f for f in found_files if f.endswith(".npy")]

    logger.info(f"File breakdown: {len(png_files)} PNG, {len(tif_files)} TIF, {len(npy_files)} NPY files")

    # Load preprocessed image (try different possible paths)
    img = None
    for img_path in possible_image_paths:
        if os.path.exists(img_path):
            try:
                img = skio.imread(img_path)
                logger.info(f"Successfully loaded image for overlay: {img_path}")

                # Save diagnostic information about the loaded image
                logger.info(f"Image properties: shape={img.shape}, dtype={img.dtype}, ")
                logger.info(f"Image statistics: min={img.min()}, max={img.max()}, mean={img.mean():.2f}, median={np.median(img):.2f}")

                # Save histogram of image values if in debug mode
                if debug and debug_dir:
                    plt.figure(figsize=(10, 4))
                    plt.hist(img.ravel(), bins=256)
                    plt.title(f"Histogram of {os.path.basename(img_path)}")
                    plt.xlabel("Pixel Value")
                    plt.ylabel("Frequency")
                    hist_path = os.path.join(debug_dir, f"histogram_{os.path.basename(img_path)}.png")
                    plt.savefig(hist_path)
                    plt.close()
                    logger.debug(f"Saved image histogram to: {hist_path}")

                    # Save a normalized version of the image for comparison
                    norm_img = img.copy()
                    if norm_img.max() > 0:  # Avoid division by zero
                        if norm_img.max() <= 1.0:  # Already normalized to [0,1]
                            norm_img = (norm_img * 255).astype(np.uint8)
                        elif norm_img.max() <= 255:  # Already in [0,255] range
                            norm_img = norm_img.astype(np.uint8)
                        else:  # Higher bit depth, normalize to [0,255]
                            p2, p98 = np.percentile(norm_img, (2, 98))
                            norm_img = np.clip((norm_img - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
                            norm_img = norm_img.astype(np.uint8)

                    norm_path = os.path.join(debug_dir, f"normalized_{os.path.basename(img_path)}")
                    skio.imsave(norm_path, norm_img)
                    logger.debug(f"Saved normalized image to: {norm_path}")

                break
            except Exception as e:
                logger.warning(f"Could not load {img_path}: {e}")

    if img is None:
        logger.error("No valid preprocessed image found for overlay. Check file paths.")
        return

    # Check if image is extremely large and downsample if necessary
    if img.shape[0] * img.shape[1] > 50_000_000:  # More than 50 million pixels
        logger.warning(f"Image is extremely large ({img.shape[0]}x{img.shape[1]}). Downsampling for visualization.")
        try:
            from skimage.transform import resize
            # Calculate scale factor to get to about 25 million pixels
            scale_factor = np.sqrt(25_000_000 / (img.shape[0] * img.shape[1]))
            new_shape = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
            logger.info(f"Downsampling image from {img.shape} to {new_shape}")
            img = resize(img, new_shape, order=0, preserve_range=True).astype(np.uint8)
            logger.info(f"Downsampled image shape: {img.shape}")
        except Exception as e:
            logger.error(f"Error downsampling image: {e}")
            logger.error("Proceeding with original image, but processing may be slow or fail.")

    # Normalize image if it appears too dark (mean value < 50 for 8-bit images)
    try:
        if img.dtype == np.uint8 and img.mean() < 50:
            logger.warning(f"Image appears dark (mean={img.mean():.2f}). Applying contrast normalization.")
            # Use a more memory-efficient approach for large images
            if img.size > 10_000_000:  # More than 10 million pixels
                logger.info("Using memory-efficient normalization for large image")
                # Sample a subset of pixels for percentile calculation
                sample_size = min(1_000_000, img.size // 10)
                flat_indices = np.random.choice(img.size, sample_size, replace=False)
                flat_img = img.ravel()[flat_indices]
                p2, p98 = np.percentile(flat_img, (2, 98))
            else:
                p2, p98 = np.percentile(img, (2, 98))

            logger.info(f"Normalizing with percentiles: p2={p2}, p98={p98}")
            img_normalized = np.clip((img.astype(np.float32) - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
            img = img_normalized.astype(np.uint8)
            logger.info(f"After normalization: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
    except Exception as e:
        logger.error(f"Error during image normalization: {e}")
        logger.error("Continuing with original image")

    # Load masks (try different possible paths)
    masks = None
    for mask_path in possible_mask_paths:
        if os.path.exists(mask_path):
            try:
                masks = np.load(mask_path)
                logger.info(f"Successfully loaded masks for overlay: {mask_path}")

                # Save diagnostic information about the loaded masks
                unique_labels = np.unique(masks)
                num_objects = len(unique_labels) - (1 if 0 in unique_labels else 0)
                logger.info(f"Mask properties: shape={masks.shape}, dtype={masks.dtype}")
                logger.info(f"Mask contains {num_objects} unique objects (labels)")

                if debug and debug_dir:
                    # Save a visualization of the mask labels
                    plt.figure(figsize=(10, 4))
                    plt.hist(masks.ravel(), bins=min(100, num_objects+1))
                    plt.title(f"Histogram of mask labels in {os.path.basename(mask_path)}")
                    plt.xlabel("Label Value")
                    plt.ylabel("Frequency")
                    hist_path = os.path.join(debug_dir, f"histogram_masks_{os.path.basename(mask_path)}.png")
                    plt.savefig(hist_path)
                    plt.close()
                    logger.debug(f"Saved mask label histogram to: {hist_path}")

                    # Save a simple visualization of the masks
                    if masks.max() > 0:
                        mask_vis = np.zeros((*masks.shape, 3), dtype=np.uint8)
                        colors = np.random.rand(masks.max() + 1, 3)
                        for i in range(1, masks.max() + 1):
                            mask_vis[masks == i] = (colors[i] * 255).astype(np.uint8)
                        mask_path = os.path.join(debug_dir, f"mask_visualization_{os.path.basename(mask_path)}.png")
                        skio.imsave(mask_path, mask_vis)
                        logger.debug(f"Saved mask visualization to: {mask_path}")

                break
            except Exception as e:
                logger.warning(f"Could not load {mask_path}: {e}")

    if masks is None:
        logger.error("No valid mask file found for overlay. Check file paths.")
        return

    # Get central crop or use the entire image if it's already small enough
    try:
        h, w = img.shape[:2]
        logger.info(f"Image dimensions for overlay: {h}x{w}")

        # Make sure masks and image can be properly cropped together
        if masks.shape != img.shape:
            logger.warning(f"Mask shape ({masks.shape}) doesn't match image shape ({img.shape})")

            # Instead of resizing, we'll find the common region that can be cropped from both
            common_h = min(masks.shape[0], img.shape[0])
            common_w = min(masks.shape[1], img.shape[1])

            logger.info(f"Using common region of size {common_h}x{common_w} for both image and masks")

            # Crop both image and masks to the common size
            if common_h < img.shape[0] or common_w < img.shape[1]:
                img = img[:common_h, :common_w]
                logger.info(f"Cropped image to {img.shape}")

            if common_h < masks.shape[0] or common_w < masks.shape[1]:
                masks = masks[:common_h, :common_w]
                logger.info(f"Cropped masks to {masks.shape}")

            # Verify they now match
            if masks.shape != img.shape:
                logger.error(f"Failed to make shapes match: img={img.shape}, masks={masks.shape}")
                logger.error("Cannot proceed with visualization")
                return

        # If image is already smaller than crop_size, use the entire image
        if h <= crop_size and w <= crop_size:
            logger.info(f"Image already smaller than crop_size ({crop_size}), using entire image")
            img_crop = img
            masks_crop = masks
        else:
            # For very large images, take a crop from a more interesting region (not just center)
            # This helps avoid empty regions in large microscopy images
            if h > 10000 or w > 10000:
                logger.info("Image is very large, searching for region with content...")
                # Try to find a region with more content by sampling a few areas
                best_mean = -1
                best_y0, best_x0 = h//2 - crop_size//2, w//2 - crop_size//2  # Default to center

                # Sample a few regions (center, and 4 quadrants)
                sample_points = [
                    # User-specified region of interest (middle part)
                    (int(h * 0.5), int(w * 0.66)),   # Middle-right area (matches crop_bbox in config)

                    # Standard sampling points
                    (h//2, w//2),                    # Center
                    (h//4, w//4),                    # Top-left quadrant
                    (h//4, 3*w//4),                  # Top-right quadrant
                    (3*h//4, w//4),                  # Bottom-left quadrant
                    (3*h//4, 3*w//4),                # Bottom-right quadrant
                    (h//2, w//4),                    # Middle-left
                    (h//2, 3*w//4),                  # Middle-right
                    (h//4, w//2),                    # Top-middle
                    (3*h//4, w//2)                   # Bottom-middle
                ]

                # Give higher weight to the first point (user's region of interest)
                # by checking it first and setting a higher threshold for other regions
                cy, cx = sample_points[0]
                y0 = max(0, cy - crop_size//2)
                x0 = max(0, cx - crop_size//2)
                y1 = min(h, y0 + crop_size)
                x1 = min(w, x0 + crop_size)

                # Sample the user's region of interest
                sample = img[y0:y1, x0:x1]
                user_roi_mean = sample.mean()
                logger.info(f"User's region of interest at ({y0}:{y1}, {x0}:{x1}) has mean intensity: {user_roi_mean:.2f}")

                # If the user's ROI has reasonable content, use it directly
                if user_roi_mean > 10:  # Threshold for "reasonable content"
                    best_y0, best_x0 = y0, x0
                    best_mean = user_roi_mean
                    logger.info(f"Using user's region of interest with mean={best_mean:.2f}")
                else:
                    # Otherwise, check all sample points
                    sample_points = sample_points[1:]  # Skip the first point as we already checked it

                for cy, cx in sample_points:
                    y0 = max(0, cy - crop_size//2)
                    x0 = max(0, cx - crop_size//2)
                    y1 = min(h, y0 + crop_size)
                    x1 = min(w, x0 + crop_size)

                    # Sample a small region to check content density
                    sample = img[y0:y1, x0:x1]
                    sample_mean = sample.mean()
                    logger.debug(f"Region at ({y0}:{y1}, {x0}:{x1}) has mean intensity: {sample_mean:.2f}")

                    if sample_mean > best_mean:
                        best_mean = sample_mean
                        best_y0, best_x0 = y0, x0

                y0, x0 = best_y0, best_x0
                y1, x1 = min(h, y0 + crop_size), min(w, x0 + crop_size)
                logger.info(f"Selected region with highest content density: y={y0}:{y1}, x={x0}:{x1}, mean={best_mean:.2f}")
            else:
                # Otherwise, take a central crop
                y0, x0 = max(0, h//2 - crop_size//2), max(0, w//2 - crop_size//2)
                y1, x1 = min(h, y0 + crop_size), min(w, x0 + crop_size)
                logger.info(f"Taking central crop from y={y0}:{y1}, x={x0}:{x1}")

            # Extract the crops
            try:
                img_crop = img[y0:y1, x0:x1]
                logger.info(f"Successfully extracted image crop with shape {img_crop.shape}")

                # Save the crop for debugging
                if debug and debug_dir:
                    crop_debug_path = os.path.join(debug_dir, "image_crop_debug.png")
                    skio.imsave(crop_debug_path, img_crop)
                    logger.debug(f"Saved image crop to: {crop_debug_path}")

                # Extract mask crop with error handling
                try:
                    masks_crop = masks[y0:y1, x0:x1]
                    logger.info(f"Successfully extracted mask crop with shape {masks_crop.shape}")
                except Exception as e:
                    logger.error(f"Error extracting mask crop: {e}")
                    logger.error("Attempting to create an empty mask crop")
                    masks_crop = np.zeros_like(img_crop, dtype=np.uint16)
            except Exception as e:
                logger.error(f"Error extracting crops: {e}")
                return
    except Exception as e:
        logger.error(f"Error during cropping: {e}")
        logger.error(traceback.format_exc())
        return

    # Ensure we have valid data for the overlay.
    if np.max(masks_crop) == 0:
        logger.warning("No segmentation masks found in the cropped region!")
        # Save the cropped image anyway for debugging.
        skio.imsave(os.path.join(check_dir, "cropped_image_no_masks.png"), img_crop)
        return

    # Always save the cropped image regardless of what happens next
    cropped_img_path = os.path.join(check_dir, "cropped_image.png")
    try:
        skio.imsave(cropped_img_path, img_crop)
        logger.info(f"Saved cropped image to: {cropped_img_path}")
    except Exception as e:
        logger.error(f"Error saving cropped image: {e}")

    # Verify that all cropped images have the same dimensions before creating overlay
    try:
        # Check dimensions of all cropped images
        img_crop_shape = img_crop.shape
        masks_crop_shape = masks_crop.shape

        if img_crop_shape != masks_crop_shape:
            logger.error(f"Critical error: Cropped image shape {img_crop_shape} doesn't match cropped masks shape {masks_crop_shape}")
            logger.warning("Attempting to fix by cropping both to minimum dimensions")

            # Find common dimensions
            common_h = min(img_crop.shape[0], masks_crop.shape[0])
            common_w = min(img_crop.shape[1], masks_crop.shape[1])

            # Crop both to common dimensions
            img_crop = img_crop[:common_h, :common_w]
            masks_crop = masks_crop[:common_h, :common_w]

            logger.info(f"Cropped both to {common_h}x{common_w} for compatibility")

            # Save the fixed crops for debugging
            debug_img_path = os.path.join(check_dir, "debug_img_crop_fixed.png")
            debug_mask_path = os.path.join(check_dir, "debug_mask_crop_fixed.png")
            skio.imsave(debug_img_path, img_crop)

            # Create a visualization of the mask for debugging
            mask_vis = np.zeros((*masks_crop.shape, 3), dtype=np.uint8)
            if masks_crop.max() > 0:
                colors_debug = np.random.rand(masks_crop.max() + 1, 3)
                for i in range(1, masks_crop.max() + 1):
                    mask_vis[masks_crop == i] = (colors_debug[i] * 255).astype(np.uint8)
            skio.imsave(debug_mask_path, mask_vis)

            logger.info(f"Saved fixed crops to: {debug_img_path} and {debug_mask_path}")

        # Log the final dimensions being used for the overlay
        logger.info(f"Creating overlay with images of shape {img_crop_shape}")

        # Create overlay
        num_labels = np.max(masks_crop)
        logger.info(f"Creating overlay with {num_labels} unique mask labels")

        # Check if we have any masks to overlay
        if num_labels == 0:
            logger.warning("No mask labels found in crop region. Creating a simple image without overlay.")
            # Just save the cropped image
            overlay_path = os.path.join(check_dir, "central_crop_no_masks.png")
            skio.imsave(overlay_path, img_crop)
            logger.info(f"Saved crop without overlay to: {overlay_path}")

            # Create a dummy overlay for the rest of the function
            overlay = np.zeros((*img_crop.shape, 3), dtype=np.float32)
            for i in range(3):
                overlay[..., i] = img_crop / 255.0
        else:
            # Generate consistent colors for better visualization
            np.random.seed(42)  # Use fixed seed for reproducible colors
            colors = np.random.rand(num_labels + 1, 3)
            np.random.seed(None)  # Reset seed

            # Create the overlay with memory usage monitoring
            logger.info("Creating mask overlay...")
            try:
                # Check if the image is too large for direct overlay
                if img_crop.shape[0] * img_crop.shape[1] > 5_000_000:  # More than 5 million pixels
                    logger.warning("Large crop detected. Using memory-efficient overlay approach.")
                    # Create overlay manually to avoid memory issues
                    overlay = np.zeros((*img_crop.shape, 3), dtype=np.float32)

                    # Add grayscale background
                    for i in range(3):
                        overlay[..., i] = img_crop / 255.0

                    # Add colored masks
                    for i in range(1, num_labels + 1):
                        if i % 100 == 0:  # Progress update for many labels
                            logger.debug(f"Processing label {i}/{num_labels}")
                        mask = masks_crop == i
                        if np.any(mask):  # Only process non-empty masks
                            for c in range(3):
                                overlay[mask, c] = colors[i, c] * 0.8 + overlay[mask, c] * 0.2  # Blend with background
                else:
                    # Use cellpose's overlay function for smaller images
                    overlay = plot.mask_overlay(img_crop, masks_crop, colors=colors)

                logger.info("Overlay created successfully")

                # Check if overlay is too dark and adjust if needed
                if overlay.mean() < 0.2:  # Very dark overlay
                    logger.warning("Overlay appears dark. Adjusting brightness...")
                    # Brighten the image component of the overlay
                    brightened_img = np.clip(img_crop.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

                    # Try to create a new overlay with the brightened image
                    try:
                        if img_crop.shape[0] * img_crop.shape[1] > 5_000_000:
                            # Manual brightening for large images
                            for i in range(3):
                                # Find background pixels (where all channels are similar to the original image)
                                bg_mask = np.abs(overlay[..., i] - img_crop / 255.0) < 0.1
                                overlay[..., i][bg_mask] = brightened_img[bg_mask] / 255.0
                        else:
                            overlay = plot.mask_overlay(brightened_img, masks_crop, colors=colors)
                        logger.info(f"Adjusted overlay brightness: mean={overlay.mean():.2f}")
                    except Exception as e:
                        logger.error(f"Error adjusting brightness: {e}")
                        # Continue with original overlay

                # Save overlay
                overlay_path = os.path.join(check_dir, "central_crop_overlay.png")
                skio.imsave(overlay_path, (overlay * 255).astype(np.uint8))
                logger.info(f"Saved central crop overlay to: {overlay_path}")

                # Save debug version with enhanced contrast if in debug mode
                if debug and debug_dir:
                    # Create a high-contrast version
                    high_contrast = np.copy(overlay)
                    # Enhance the non-zero (mask) parts
                    mask = (high_contrast.sum(axis=2) > 0)
                    high_contrast[mask] = np.clip(high_contrast[mask] * 1.5, 0, 1)
                    high_contrast_path = os.path.join(debug_dir, "high_contrast_overlay.png")
                    skio.imsave(high_contrast_path, (high_contrast * 255).astype(np.uint8))
                    logger.debug(f"Saved high-contrast overlay to: {high_contrast_path}")
            except Exception as e:
                logger.error(f"Error creating overlay: {e}")
                logger.error(traceback.format_exc())
                # Try a simpler approach as fallback
                try:
                    logger.warning("Attempting fallback overlay method...")
                    simple_overlay = np.zeros((*img_crop.shape, 3), dtype=np.uint8)
                    # Copy grayscale image to all channels
                    for i in range(3):
                        simple_overlay[..., i] = img_crop
                    # Add colored boundaries
                    for i in range(1, min(num_labels + 1, 1000)):  # Limit to 1000 labels for speed
                        boundary = masks_crop == i
                        if np.any(boundary):
                            color = (colors[i] * 255).astype(np.uint8)
                            for c in range(3):
                                simple_overlay[boundary, c] = color[c]
                    overlay_path = os.path.join(check_dir, "central_crop_overlay_fallback.png")
                    skio.imsave(overlay_path, simple_overlay)
                    logger.info(f"Saved fallback overlay to: {overlay_path}")
                    overlay = simple_overlay / 255.0  # Normalize for consistency with rest of code
                except Exception as e2:
                    logger.error(f"Fallback overlay also failed: {e2}")
                    logger.error(traceback.format_exc())
                    # Save just the image as a last resort
                    try:
                        skio.imsave(os.path.join(check_dir, "image_only.png"), img_crop)
                        logger.info("Saved image without overlay as a fallback")
                        # Create a dummy overlay for the rest of the function
                        overlay = np.zeros((*img_crop.shape, 3), dtype=np.float32)
                        for i in range(3):
                            overlay[..., i] = img_crop / 255.0
                    except:
                        logger.error("All overlay methods failed. Cannot continue.")
                        return
    except Exception as e:
        logger.error(f"Unexpected error in overlay creation: {e}")
        logger.error(traceback.format_exc())
        return

    # Load CLAHE and gamma corrected images if available
    clahe_img = None
    gamma_img = None

    # Try to load CLAHE image
    for clahe_path in clahe_paths:
        if os.path.exists(clahe_path):
            try:
                clahe_img = skio.imread(clahe_path)
                logger.info(f"Successfully loaded CLAHE image: {clahe_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load CLAHE image {clahe_path}: {e}")

    # Try to load gamma corrected image
    for gamma_path in gamma_paths:
        if os.path.exists(gamma_path):
            try:
                gamma_img = skio.imread(gamma_path)
                logger.info(f"Successfully loaded gamma corrected image: {gamma_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load gamma image {gamma_path}: {e}")

    # Crop CLAHE and gamma images if they exist - using the same approach as for masks
    if clahe_img is not None:
        try:
            # Make sure CLAHE image can be properly cropped with the main image
            if clahe_img.shape != img.shape:
                logger.warning(f"CLAHE image shape ({clahe_img.shape}) doesn't match main image shape ({img.shape})")

                # Find the common region that can be cropped
                common_h = min(clahe_img.shape[0], img.shape[0])
                common_w = min(clahe_img.shape[1], img.shape[1])

                logger.info(f"Using common region of size {common_h}x{common_w} for CLAHE image")

                # Crop both to the common size
                if common_h < img.shape[0] or common_w < img.shape[1]:
                    img = img[:common_h, :common_w]
                    logger.info(f"Re-cropped main image to {img.shape}")

                if common_h < clahe_img.shape[0] or common_w < clahe_img.shape[1]:
                    clahe_img = clahe_img[:common_h, :common_w]
                    logger.info(f"Cropped CLAHE image to {clahe_img.shape}")

                # Also update masks to match
                if masks.shape != img.shape:
                    masks = masks[:common_h, :common_w]
                    logger.info(f"Re-cropped masks to {masks.shape}")

            # Apply the same crop as the main image
            if h <= crop_size and w <= crop_size:
                clahe_crop = clahe_img
            else:
                clahe_crop = clahe_img[y0:y1, x0:x1]
                logger.info(f"Cropped CLAHE image to shape {clahe_crop.shape}")
        except Exception as e:
            logger.error(f"Error cropping CLAHE image: {e}")
            clahe_crop = None
    else:
        clahe_crop = None

    # Crop gamma image if it exists - using the same approach
    if gamma_img is not None:
        try:
            # Make sure gamma image can be properly cropped with the main image
            if gamma_img.shape != img.shape:
                logger.warning(f"Gamma image shape ({gamma_img.shape}) doesn't match main image shape ({img.shape})")

                # Find the common region that can be cropped
                common_h = min(gamma_img.shape[0], img.shape[0])
                common_w = min(gamma_img.shape[1], img.shape[1])

                logger.info(f"Using common region of size {common_h}x{common_w} for gamma image")

                # Crop both to the common size
                if common_h < img.shape[0] or common_w < img.shape[1]:
                    img = img[:common_h, :common_w]
                    logger.info(f"Re-cropped main image to {img.shape}")

                    # Also update other images and masks to match
                    if masks.shape != img.shape:
                        masks = masks[:common_h, :common_w]
                        logger.info(f"Re-cropped masks to {masks.shape}")

                    if clahe_crop is not None and clahe_crop.shape != img.shape:
                        clahe_crop = clahe_crop[:common_h, :common_w]
                        logger.info(f"Re-cropped CLAHE image to {clahe_crop.shape}")

                if common_h < gamma_img.shape[0] or common_w < gamma_img.shape[1]:
                    gamma_img = gamma_img[:common_h, :common_w]
                    logger.info(f"Cropped gamma image to {gamma_img.shape}")

            # Apply the same crop as the main image
            if h <= crop_size and w <= crop_size:
                gamma_crop = gamma_img
            else:
                gamma_crop = gamma_img[y0:y1, x0:x1]
                logger.info(f"Cropped gamma image to shape {gamma_crop.shape}")
        except Exception as e:
            logger.error(f"Error cropping gamma image: {e}")
            gamma_crop = None
    else:
        gamma_crop = None

    # Update h, w after all the cropping to ensure consistent dimensions
    h, w = img.shape[:2]
    logger.info(f"Final image dimensions for overlay after all cropping: {h}x{w}")

    # Create a 2x2 figure showing the preprocessing steps and segmentation result
    try:
        logger.info("Creating 2x2 summary figure with preprocessing steps and overlay")
        fig1, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Preprocessed image - always available
        axes[0, 0].imshow(img_crop, cmap="gray")
        axes[0, 0].set_title("Preprocessed Image")
        axes[0, 0].axis("off")

        # CLAHE enhanced image
        if clahe_crop is not None:
            try:
                axes[0, 1].imshow(clahe_crop, cmap="gray")
                axes[0, 1].set_title("CLAHE Enhanced")
            except Exception as e:
                logger.error(f"Error displaying CLAHE image: {e}")
                axes[0, 1].text(0.5, 0.5, "CLAHE Error", ha="center", va="center")
        else:
            axes[0, 1].text(0.5, 0.5, "No CLAHE", ha="center", va="center")
        axes[0, 1].axis("off")

        # Gamma corrected image
        if gamma_crop is not None:
            try:
                axes[1, 0].imshow(gamma_crop, cmap="gray")
                axes[1, 0].set_title("Gamma Corrected")
            except Exception as e:
                logger.error(f"Error displaying gamma image: {e}")
                axes[1, 0].text(0.5, 0.5, "Gamma Error", ha="center", va="center")
        else:
            axes[1, 0].text(0.5, 0.5, "No Gamma", ha="center", va="center")
        axes[1, 0].axis("off")

        # Segmentation overlay
        try:
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title("Segmentation Masks Overlay")
        except Exception as e:
            logger.error(f"Error displaying overlay: {e}")
            # Try to show just the masks as a fallback
            try:
                if masks_crop.max() > 0:
                    # Create a simple colored mask visualization
                    mask_vis = np.zeros((*masks_crop.shape, 3), dtype=np.uint8)
                    colors_vis = np.random.rand(masks_crop.max() + 1, 3)
                    for i in range(1, masks_crop.max() + 1):
                        mask_vis[masks_crop == i] = (colors_vis[i] * 255).astype(np.uint8)
                    axes[1, 1].imshow(mask_vis)
                    axes[1, 1].set_title("Segmentation Masks")
                else:
                    axes[1, 1].text(0.5, 0.5, "No Masks", ha="center", va="center")
            except:
                axes[1, 1].text(0.5, 0.5, "Overlay Error", ha="center", va="center")
        axes[1, 1].axis("off")

        # Save the figure
        plt.tight_layout()
        summary_path = os.path.join(check_dir, "quick_overlay_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        logger.info(f"Saved 4-panel summary figure to: {summary_path}")

        # Also save individual panels as separate files for easier inspection
        try:
            # Save preprocessed image
            skio.imsave(os.path.join(check_dir, "panel1_preprocessed.png"), img_crop)

            # Save CLAHE if available
            if clahe_crop is not None:
                skio.imsave(os.path.join(check_dir, "panel2_clahe.png"), clahe_crop)

            # Save gamma if available
            if gamma_crop is not None:
                skio.imsave(os.path.join(check_dir, "panel3_gamma.png"), gamma_crop)

            # Save overlay
            if isinstance(overlay, np.ndarray):
                overlay_img = (overlay * 255).astype(np.uint8) if overlay.max() <= 1.0 else overlay.astype(np.uint8)
                skio.imsave(os.path.join(check_dir, "panel4_overlay.png"), overlay_img)

            logger.info("Saved individual panel images for easier inspection")
        except Exception as e:
            logger.error(f"Error saving individual panels: {e}")

        # Also create the side-by-side comparison for backward compatibility
        try:
            fig2, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Original image - always available
            ax[0].imshow(img_crop, cmap='gray')
            ax[0].set_title("Original Image (Crop)")
            ax[0].axis('off')

            # Overlay - with fallback
            try:
                ax[1].imshow(overlay)
                ax[1].set_title("Segmentation Overlay")
            except Exception as e:
                logger.error(f"Error displaying overlay in side-by-side comparison: {e}")
                # Try to show just the masks as a fallback
                try:
                    if masks_crop.max() > 0:
                        # Create a simple colored mask visualization
                        mask_vis = np.zeros((*masks_crop.shape, 3), dtype=np.uint8)
                        colors_vis = np.random.rand(masks_crop.max() + 1, 3)
                        for i in range(1, masks_crop.max() + 1):
                            mask_vis[masks_crop == i] = (colors_vis[i] * 255).astype(np.uint8)
                        ax[1].imshow(mask_vis)
                        ax[1].set_title("Segmentation Masks")
                    else:
                        ax[1].text(0.5, 0.5, "No Masks", ha="center", va="center")
                except:
                    ax[1].text(0.5, 0.5, "Overlay Error", ha="center", va="center")
            ax[1].axis('off')

            plt.tight_layout()
            fig_path = os.path.join(check_dir, "central_crop_comparison.png")
            plt.savefig(fig_path, dpi=300)
            plt.close(fig2)
            logger.info(f"Saved side-by-side comparison figure to: {fig_path}")
        except Exception as e:
            logger.error(f"Error creating side-by-side comparison: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error creating summary figures: {e}")
        logger.error(traceback.format_exc())

        # Last resort: save at least the cropped image if everything else fails
        try:
            last_resort_path = os.path.join(check_dir, "last_resort_crop.png")
            skio.imsave(last_resort_path, img_crop)
            logger.warning(f"Saved last resort crop to: {last_resort_path}")
        except Exception as e2:
            logger.error(f"Even last resort save failed: {e2}")

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

def generate_full_overlay(image, masks, flows, output_dir, logger=None):
    """
    Generate and save full-size overlay visualizations for entire tissue.

    Args:
        image (np.ndarray): Grayscale input image.
        masks (np.ndarray): Segmentation masks.
        flows (tuple): Cellpose flow fields.
        output_dir (str or Path): Directory to save results.
        logger (logging.Logger): Optional logger.

    Returns:
        None.
    """

    if logger is None:
        logger = _setup_logger('generate_full_overlay', False)

    # Validate input types.
    if image is None or not isinstance(image, np.ndarray):
        logger.error("Input 'image' must be a valid numpy array.")
        return

    if masks is None or not isinstance(masks, np.ndarray):
        logger.error("Input 'masks' must be a valid numpy array.")
        return

    if flows is None or not isinstance(flows, tuple) or len(flows) == 0:
        logger.error("Input 'flows' must be a non-empty tuple of numpy arrays.")
        return

    if not isinstance(output_dir, (str, Path)):
        logger.error("Output directory must be a string or Path object.")
        return

    # Check array shape agreement.
    if image.shape[:2] != masks.shape:
        logger.warning(f"Image and mask shape mismatch: image={image.shape}, masks={masks.shape}")
        min_h = min(image.shape[0], masks.shape[0])
        min_w = min(image.shape[1], masks.shape[1])
        image = image[:min_h, :min_w]
        masks = masks[:min_h, :min_w]

    vis_dir = Path(output_dir) / 'visualizations/full_image'

    try:
        vis_dir = _ensure_dir(vis_dir)
    except PermissionError as e:
        logger.error(e)
        return

    """Create and save mask overlay."""
    try:
        if masks.max() == 0:
            logger.warning("No masks found; skipping mask overlay.")
        else:
            colors = np.random.rand(masks.max() + 1, 3)
            full = plot.mask_overlay(image, masks, colors=colors)
            skio.imsave(vis_dir / 'mask_overlay.png', (full * 255).astype(np.uint8))
            logger.info(f"Saved full overlay: {vis_dir / 'mask_overlay.png'}")
    except Exception as e:
        logger.error(f"Full overlay failed: {e}")
        traceback.print_exc()

    """Create and save Cellpose flow visualization."""
    try:
        fig = plt.figure(figsize=(10, 10))
        plot.show_segmentation(fig, img=image, maski=masks, flowi=flows[0], channels=[0, 0])
        debug_path = vis_dir / 'segmentation_debug.png'
        fig.savefig(debug_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved flow debug: {debug_path}")
    except Exception as e:
        logger.error(f"Flow debug failed: {e}")
        traceback.print_exc()

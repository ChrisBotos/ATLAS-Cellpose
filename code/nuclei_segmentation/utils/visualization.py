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

    # Create debug snapshots directory if in debug mode
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug_snapshots")
        os.makedirs(debug_dir, exist_ok=True)
        logger.debug(f"Created debug snapshots directory: {debug_dir}")

    # Subdirectory for storing the check images
    check_dir = os.path.join(output_dir, "visualizations", "cropped_preview")
    os.makedirs(check_dir, exist_ok=True)
    logger.info(f"Created visualization directory: {check_dir}")

    # Print diagnostic information about the output directory
    logger.info(f"Working with output directory: {os.path.abspath(output_dir)}")

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

        # Make sure masks has the same shape as img
        if masks.shape != img.shape:
            logger.warning(f"Mask shape ({masks.shape}) doesn't match image shape ({img.shape})")
            try:
                # Try to resize masks if dimensions don't match
                if len(masks.shape) == 2 and len(img.shape) == 2:
                    from skimage.transform import resize
                    logger.info(f"Resizing masks from {masks.shape} to {img.shape}")
                    masks = resize(masks, img.shape, order=0, preserve_range=True).astype(np.uint16)
                    logger.info(f"Resized masks to match image shape: {masks.shape}")
            except Exception as e:
                logger.error(f"Error resizing masks: {e}")
                logger.error("Will attempt to crop masks as-is, but this may cause issues")

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
                    (h//2, w//2),                    # Center
                    (h//4, w//4),                    # Top-left quadrant
                    (h//4, 3*w//4),                  # Top-right quadrant
                    (3*h//4, w//4),                  # Bottom-left quadrant
                    (3*h//4, 3*w//4)                 # Bottom-right quadrant
                ]

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

    # Create overlay
    try:
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
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
import torch
import cv2
import matplotlib.pyplot as plt
from skimage import io as skio
from cellpose import models, plot
from skimage.measure import regionprops
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from PIL import Image
import logging

# Import the small overlay snippet function (assumed to be defined in check_segmentation_overlay.py)
from archived_code.check_segmentation_overlay import small_segmentation_overlay

# Increase the maximum allowed image pixels
Image.MAX_IMAGE_PIXELS = 10**9

# =============================================================================
# PARAMETERS
# =============================================================================
SETTINGS = {
    "UPSCALE_FACTOR": 1,           # e.g., 1 for no upscaling, 4 for 4x upscaling
    "CROP_IMAGE": True,
    "ENHANCE_CONTRAST": True,
    "ENHANCE_DIM": False,
    "GENERATE_OVERLAY": False,
    "IMAGE_PATH": "IRI_regist.tif",  # /exports/archive/hg-funcgenom-research/IRI_multimodal_project/Stereo-seq_IRI/
    "CLAHE_CLIPLIMIT": 5.0,
    "CLAHE_TILE_GRID_SIZE": (32, 32),
    "OUTPUT_DIR": "iri_results_small_0d09-12_5cl_32x32",
    # Switches for optional steps:
    "USE_EDGE_DETECTION": False,  # Toggle Canny-based refinement on/off
    "APPLY_WATERSHED": False,      # Apply local watershed to large lumps
    "USE_TILING": False,            # Toggle tiling on/off
    # Watershed splitting settings:
    "AREA_THRESHOLD_FOR_WATERSHED": 1000,  # min area to consider the region "fused"
    "LOCAL_MAXIMA_FOOTPRINT": (3, 3),      # footprint for local maxima detection in watershed splitting
    # Edge detection thresholds:
    "CANNY_THRESHOLD1": 50,
    "CANNY_THRESHOLD2": 150,
    # Tiling settings:
    "tile_side_length": 2**11,  # Pixels.
    "TILE_OVERLAP": 0.1,             # Must be a fraction (e.g., 0.1 for 10%).
    # Small overlay settings:
    "SMALL_OVERLAY_SIZE": 1024
}

CELLPOSE_PARAMS = {
    "model_type": "nuclei",  # nuclei model is tuned for DAPI-stained images
    "gpu": torch.cuda.is_available(),
    "diameter": 0,          # Adjust based on your expected nuclei size; set 0 for auto-estimation
    "channels": [0, 0],
    "flow_threshold": 0.9,
    "cellprob_threshold": -12,
    "resample": True,
    "stitch_threshold": 0.4,
    "batch_size": "placeholder"   # Updated later.
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
import torch

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
    # approximate bytes per patch: pixels × bytes_per_pixel
    bytes_per_patch = tile_pixels * bytes_per_pixel
    # how many patches fit into target_mem_per_batch
    max_batch = max(1, usable // (bytes_per_patch * (usable // target_mem_per_batch)))
    return int(max_batch)

# Example usage (tile_side_length=2048 → ~4.2M pixels):
tile_pixels = SETTINGS["tile_side_length"]**2
CELLPOSE_PARAMS["batch_size"] = choose_batch_size(tile_pixels)

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Configure logging to console and file."""
    log_file = os.path.join(output_dir, "segmentation_log.txt")
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid double logging
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info("===== Cellpose Segmentation Log =====")
    for k, v in SETTINGS.items():
        logger.info(f"{k}: {v}")
    
    return logger

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def convert_16bit_to_8bit(image):
    """Convert a 16-bit image to 8-bit using percentile scaling."""
    if image.dtype != np.uint16:
        return image
    p1, p99 = np.percentile(image, (1, 99))
    if p99 - p1 == 0:
        p1, p99 = image.min(), image.max()
    return np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

def adaptive_gamma_correction(image, min_gamma=1.5, max_gamma=2.5, logger=None):
    """Apply adaptive gamma correction based on the image median value."""
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
        h, w = image.shape
        image = image[int(8*h//16): int(8.2*h//16), int(12*w//16): int(12.2*w//16)]
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
    
    return image

from skimage.util import view_as_windows

def split_image_into_tiles(image, tile_side_length, overlap, logger):
    """
    Split the image into overlapping tiles.
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
    
    tiles = view_as_windows(image, (tile_side_length, tile_side_length), step)
    slices = []
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            slices.append((slice(i * step, i * step + tile_side_length), slice(j * step, j * step + tile_side_length)))
    
    return tiles.reshape(-1, tile_side_length, tile_side_length), slices

def merge_tiles_with_weighted_overlap(tiles, slices, image_shape, overlap, logger):
    """
    Merge tiles back into a single image with weighted averaging in overlapping regions.
    Args:
        tiles: List of processed tiles (e.g., masks or flows).
        slices: List of slice objects corresponding to the tiles.
        image_shape: Shape of the original image.
        overlap: Fractional overlap between tiles (e.g., 0.1 for 10% overlap).
        logger: Logger instance for logging.
    Returns:
        merged_image: Reconstructed image from tiles.
    """
    # Check if the tiles are multi-channel.
    is_multi_channel = len(tiles[0].shape) == 3
    num_channels = tiles[0].shape[2] if is_multi_channel else 1

    # Initialize merged image and weight map.
    if is_multi_channel:
        merged_image = np.zeros((*image_shape, num_channels), dtype=np.float32)
        weight_map = np.zeros((*image_shape, num_channels), dtype=np.float32)
    else:
        merged_image = np.zeros(image_shape, dtype=np.float32)
        weight_map = np.zeros(image_shape, dtype=np.float32)

    tile_side_length = tiles[0].shape[0]
    step = int(tile_side_length * (1 - overlap))

    for tile, slc in zip(tiles, slices):
        weight = np.ones(tile.shape[:2], dtype=np.float32)  # Weight is 2D, even for multi-channel tiles.

        # Apply linear weights to the edges.
        for i in range(tile.shape[0]):
            weight[i, :] *= min(i / (overlap * tile_side_length), 1, (tile_side_length - i - 1) / (overlap * tile_side_length))
        for j in range(tile.shape[1]):
            weight[:, j] *= min(j / (overlap * tile_side_length), 1, (tile_side_length - j - 1) / (overlap * tile_side_length))

        if is_multi_channel:
            for c in range(num_channels):
                merged_image[slc[0], slc[1], c] += tile[:, :, c] * weight
                weight_map[slc[0], slc[1], c] += weight
        else:
            merged_image[slc] += tile * weight
            weight_map[slc] += weight

    # Normalize by the weight map to avoid intensity artifacts.
    merged_image /= np.maximum(weight_map, 1e-5)
    logger.info("Merged tiles with weighted overlap.")
    return merged_image.astype(np.float32 if is_multi_channel else np.uint16)

# =============================================================================
# CELLPOSE SEGMENTATION
# =============================================================================
def run_cellpose_on_tiles(model, image, cellpose_params, settings, logger):
    """
    Run Cellpose on the entire image or split it into tiles based on the settings.
    Args:
        model: Cellpose model instance.
        image: Input 2D image (grayscale).
        cellpose_params: Parameters for Cellpose.
        settings: General settings dictionary.
        logger: Logger instance for logging.
    Returns:
        merged_masks: Combined segmentation mask for the full image.
        merged_flows: Combined flows for the full image.
        total_cells: Total number of cells detected.
    """
    h, w = image.shape
    tile_side_length = settings["tile_side_length"]
    overlap = settings["TILE_OVERLAP"]

    # Check if tiling is necessary.
    if not settings["USE_TILING"] or (tile_side_length >= h and tile_side_length >= w):
        # Process the entire image as a single tile.
        logger.info("Tiling is disabled or unnecessary. Processing the entire image as a single tile.")
        image = image[..., None]  # Add channel axis.
        masks, flows, styles, diams = model.eval(
            image,
            diameter=cellpose_params["diameter"],
            channels=cellpose_params["channels"],
            flow_threshold=cellpose_params["flow_threshold"],
            cellprob_threshold=cellpose_params["cellprob_threshold"],
            resample=cellpose_params["resample"],
            augment=False,
            batch_size=cellpose_params["batch_size"],
            do_3D=False
        )
        total_cells = (masks > 0).sum()  # Count the number of cells.
        logger.info(f"Total cells detected: {total_cells}")
        return masks, flows, total_cells

    # If tiling is enabled and necessary, split the image into tiles.
    tiles, slices = split_image_into_tiles(image, tile_side_length, overlap, logger)
    logger.info(f"Processing {len(tiles)} tiles.")
    
    # Initialize lists to store results.
    tile_masks = []
    tile_flows = []
    total_cells = 0
    
    # Run Cellpose on each tile.
    for idx, tile in enumerate(tiles):
        logger.info(f"Running Cellpose on tile {idx + 1}/{len(tiles)}")
        tile = tile[..., None]  # Add channel axis.
        masks, flows, styles, diams = model.eval(
            tile,
            diameter=cellpose_params["diameter"],
            channels=cellpose_params["channels"],
            flow_threshold=cellpose_params["flow_threshold"],
            cellprob_threshold=cellpose_params["cellprob_threshold"],
            resample=cellpose_params["resample"],
            augment=False,
            batch_size=cellpose_params["batch_size"],
            do_3D=False
        )
        tile_masks.append(masks)
        tile_flows.append(flows[0])  # Assuming flows[0] is the relevant flow field.
        total_cells += (masks > 0).sum()  # Count the number of cells in the tile.
    
    # Merge the tile masks and flows back into a single image.
    merged_masks = merge_tiles_with_weighted_overlap(tile_masks, slices, image.shape, overlap, logger)
    merged_flows = merge_tiles_with_weighted_overlap(tile_flows, slices, image.shape, overlap, logger)
    
    logger.info(f"Total cells detected: {total_cells}")
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
def identify_and_split_fused_labels(masks, min_area=1000, footprint=(3,3), logger=None):
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
    overlay = plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks)+1, 3))
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
    logger = setup_logging(output_dir)
    
    # 1. Preprocess the image.
    image = preprocess_image(SETTINGS["IMAGE_PATH"], SETTINGS, logger)
    
    # 2. Segment image by tiling or as a single tile.
    model = models.Cellpose(model_type=CELLPOSE_PARAMS["model_type"],
                            gpu=CELLPOSE_PARAMS["gpu"])
    
    logger.info(f"Using device: {'cuda' if CELLPOSE_PARAMS['gpu'] else 'cpu'}")
    if CELLPOSE_PARAMS["gpu"]:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, SETTINGS, logger)
    
    # Save the merged mask and flows.
    np.save(os.path.join(output_dir, "masks.npy"), masks)
    np.save(os.path.join(output_dir, "flows.npy"), flows)
    skio.imsave(os.path.join(output_dir, "segmentation_mask.png"), masks.astype(np.uint16))
    logger.info(f"Saved segmentation mask and flows. Total cells detected: {total_cells}")
    
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
        skio.imsave(os.path.join(output_dir, "segmentation_mask_post_watershed.png"), lumps_split_mask.astype(np.uint16))
        np.save(os.path.join(output_dir, "segmentation_mask_post_watershed.npy"), lumps_split_mask)
        masks = lumps_split_mask

    # 5. Optionally generate overlay visualization.
    if SETTINGS["GENERATE_OVERLAY"]:
        generate_overlay(image, masks, flows, output_dir, logger)
    
    # 6. Create a small overlay snippet (cropped) for quick review.
    small_segmentation_overlay(output_dir, crop_size=SETTINGS["SMALL_OVERLAY_SIZE"] * SETTINGS["UPSCALE_FACTOR"])
    
if __name__ == "__main__":
    main()

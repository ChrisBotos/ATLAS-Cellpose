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
from scipy import ndimage as ndi
from PIL import Image
import logging
import configparser
import torch

# Import the small overlay snippet function (assumed to be defined in check_segmentation_overlay.py)
from check_segmentation_overlay import small_segmentation_overlay

# Increase the maximum allowed image pixels
Image.MAX_IMAGE_PIXELS = 10 ** 9

# =============================================================================
# PARAMETERS
# =============================================================================
"""CONFIG LOADING"""

def load_config(path="nuclei_segmentation_config.ini"):
    config = configparser.ConfigParser()
    config.read(path)

    def get_bool(section, key):
        return config.get(section, key).lower() == "true"

    def get_tuple(section, key, cast_type=float):
        return tuple(cast_type(i.strip()) for i in config.get(section, key).split(','))

    SETTINGS = {
        "IMAGE_PATH": config.get("General", "image_path"),
        "OUTPUT_DIR": config.get("General", "output_dir"),
        "UPSCALE_FACTOR": config.getint("General", "upscale_factor"),
        "CROP_IMAGE": get_bool("General", "crop_image"),
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
        "TILE_SIZE": config.getint("Tiling", "tile_size"),
        "TILE_OVERLAP": config.getfloat("Tiling", "tile_overlap"),
        "SMALL_OVERLAY_SIZE": config.getint("Overlay", "small_overlay_size"),
    }

    CELLPOSE_PARAMS = {
        "model_type": config.get("Cellpose", "model_type"),
        "gpu": get_bool("Cellpose", "gpu") and torch.cuda.is_available(),
        "diameter": config.getint("Cellpose", "diameter"),
        "channels": get_tuple("Cellpose", "channels", int),
        "flow_threshold": config.getfloat("Cellpose", "flow_threshold"),
        "cellprob_threshold": config.getfloat("Cellpose", "cellprob_threshold"),
        "resample": get_bool("Cellpose", "resample"),
        "stitch_threshold": config.getfloat("Cellpose", "stitch_threshold"),
        "batch_size": "placeholder"  # To be updated dynamically below
    }

    return SETTINGS, CELLPOSE_PARAMS

# Load config values from file
SETTINGS, CELLPOSE_PARAMS = load_config()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem_per_batch=150_000_000):
    """
    tile_pixels: number of pixels per patch (i.e. TILE_SIZE**2)
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


# Example usage (TILE_SIZE=2048 → ~4.2M pixels):
tile_pixels = SETTINGS["TILE_SIZE"] ** 2
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
        image = image[int(8 * h // 16): int(8.1 * h // 16), int(12 * w // 16): int(12.1 * w // 16)]
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


def split_image_into_tiles(image, tile_size, overlap, logger):
    """
    Split the image into overlapping tiles.
    Args:
        image: Input 2D image (grayscale).
        tile_size: Size of each tile (pixels).
        overlap: Fractional overlap between tiles (e.g., 0.1 for 10% overlap).
        logger: Logger instance for logging.
    Returns:
        tiles: List of tiles as numpy arrays.
        slices: List of slice objects for reconstructing the full image.
    """
    h, w = image.shape
    if tile_size > h or tile_size > w:
        logger.warning(f"Tile size {tile_size} is larger than image dimensions ({h}, {w}). Adjusting tile size.")
        tile_size = min(h, w)

    step = int(tile_size * (1 - overlap))
    logger.info(f"Splitting image into tiles with size {tile_size} and step {step}")

    tiles = view_as_windows(image, (tile_size, tile_size), step)
    slices = []
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            slices.append((slice(i * step, i * step + tile_size), slice(j * step, j * step + tile_size)))

    return tiles.reshape(-1, tile_size, tile_size), slices


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

        # Standardise to (C, h, w).
        if tile.ndim == 2:                           # (h, w)
            tile = tile[np.newaxis, ...]
        elif tile.ndim == 3 and tile.shape[0] == 2:  # (2, h, w)
            pass
        elif tile.ndim == 3 and tile.shape[-1] == 2: # (h, w, 2)
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
    tile_size     = settings["TILE_SIZE"]
    overlap       = settings["TILE_OVERLAP"]
    use_tiling    = settings["USE_TILING"] and (tile_size < h or tile_size < w)

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
    tiles, slices = split_image_into_tiles(image, tile_size, overlap, logger)
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
    logger = setup_logging(output_dir)
    print("2DEBUG: dir made...")

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

"""
PIPELINE: Modular segmentation flow using Cellpose3 with DAPI-stained nuclear features.
"""

import traceback
import numpy as np
from pathlib import Path
from skimage import io as skio
import torch
from cellpose import models

from utils.preprocessing import preprocess_image
from utils.segmentation import run_cellpose_on_tiles
from utils.watershed import refine_segmentation_with_edges, apply_watershed_to_mask
from utils.visualization import small_segmentation_overlay


def log_config(logger, settings, CELLPOSE_PARAMS):
    if settings.get("debug_mode", False):
        logger.debug("=== settings ===")
        for k, v in settings.items():
            logger.debug(f"{k}: {v}")
        logger.debug("=== CELLPOSE ===")
        for k, v in CELLPOSE_PARAMS.items():
            logger.debug(f"{k}: {v}")


def setup_model(CELLPOSE_PARAMS, logger):
    model_type = CELLPOSE_PARAMS["model_type"]
    use_gpu = CELLPOSE_PARAMS.get("gpu", False)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model = models.Cellpose(model_type=model_type, gpu=(device == 'cuda'))

    logger.info(f"Using Cellpose model: {model_type}")
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            logger.warning("Failed to get CUDA device name.")

    return model


def save_outputs(masks, flows, output_dir, logger):
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    flows_dir = output_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)

    np.save(masks_dir / "segmentation_masks.npy", masks)
    skio.imsave(masks_dir / "segmentation_masks.tif", masks.astype(np.uint16))
    np.savez(flows_dir / "flows.npz", flow0=flows[0], flow1=flows[1], cellprob=flows[2])

    logger.info("Segmentation results saved.")


def apply_postprocessing(image, masks, settings, output_dir, logger):
    if settings.get("use_edge_detection", False):
        logger.info("Running edge refinement...")
        masks = refine_segmentation_with_edges(image, masks, settings, logger)
        skio.imsave(Path(output_dir) / "refined_segmentation_masks.tif", masks.astype(np.uint16))

    if settings.get("apply_watershed", False):
        logger.info("Applying watershed...")
        try:
            masks = apply_watershed_to_mask(
                masks,
                min_area=settings.get("area_threshold_for_watershed", 1000),
                footprint=settings.get("local_maxima_footprint", (3, 3)),
                logger=logger
            )
            skio.imsave(Path(output_dir) / "segmentation_masks_post_watershed.tif", masks.astype(np.uint16))
            np.save(Path(output_dir) / "segmentation_masks_post_watershed.npy", masks)
        except Exception as e:
            logger.error(f"Watershed error: {e}")
            logger.debug(traceback.format_exc())

    return masks


def generate_overlays(image, masks, flows, output_dir, settings, logger):
    try:
        small_segmentation_overlay(
            output_dir,
            crop_size=settings.get("small_overlay_size", 512) * settings.get("upscale_factor", 1),
            debug=settings.get("debug_mode", False)
        )
    except Exception as e:
        logger.warning(f"Overlay generation failed: {e}")
        logger.debug(traceback.format_exc())


def run_segmentation_pipeline(settings, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap=None):
    """
    Orchestrates the full segmentation pipeline for nuclear masks from DAPI-stained images.

    This includes preprocessing, tiling-aware Cellpose segmentation, optional refinement,
    and visualization generation. All outputs are stored in settings["output_dir"].

    Returns:
        int: 0 if successful, 1 otherwise.
    """
    try:
        # Log current configuration.
        log_config(logger, settings, CELLPOSE_PARAMS)

        # Normalize paths early.
        image_path = Path(settings["image_path"]).expanduser().resolve()
        output_dir = Path(settings["output_dir"]).expanduser().resolve()
        settings["output_dir"] = str(output_dir)  # Ensure all downstream functions use the same path.

        # Check image exists.
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return 1

        logger.info(f"Image path: {image_path}")
        logger.info(f"Output directory: {output_dir}")

        # Step 1: Image preprocessing (CLAHE, cropping etc.).
        image = preprocess_image(image_path, settings, logger)

        # Step 2: Load Cellpose model (with GPU if available).
        model = setup_model(CELLPOSE_PARAMS, logger)

        # Step 3: Segmentation.
        logger.info("Running Cellpose segmentation...")
        masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, settings, logger)

        if masks is None or masks.size == 0:
            logger.error("No segmentation masks returned. Aborting.")
            return 1

        logger.info(f"Segmentation completed. Total cells: {total_cells}")

        # Step 4: Save raw outputs.
        save_outputs(masks, flows, output_dir, logger)

        # Step 5: Postprocess with edge refinement and/or watershed.
        masks = apply_postprocessing(image, masks, settings, output_dir, logger)

        # Step 6: Overlay generation (cropped + full).
        generate_overlays(image, masks, flows, output_dir, settings, logger)

        # Step 7: Optional debug snapshot.
        if snap:
            snap.capture("end_of_pipeline", {"masks": masks})

        logger.info("===== Pipeline Completed Successfully =====")
        return 0

    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.debug(traceback.format_exc())
        return 1

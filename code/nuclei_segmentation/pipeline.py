import os
import numpy as np
import traceback
import torch
from skimage import io as skio

from cellpose import models

from utils.preprocessing import preprocess_image, ensure_matching_shapes
from utils.segmentation import run_cellpose_on_tiles
from utils.watershed import apply_watershed_to_mask, refine_segmentation_with_edges
from utils.visualization import generate_full_overlay, small_segmentation_overlay


def run_segmentation_pipeline(SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap):
    try:
        image_path = SETTINGS["IMAGE_PATH"]
        output_subdir = SETTINGS.get("OUTPUT_SUBDIR", "unnamed_run")
        output_dir = os.path.join(PROJECT_DIRS["results"], output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("===== Cellpose Segmentation Pipeline Started =====")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Output directory: {output_dir}")

        if SETTINGS.get("DEBUG_MODE", False):
            logger.info("========== CONFIGURATION PARAMETERS ==========")
            for k, v in SETTINGS.items():
                logger.info(f"[SETTINGS] {k} = {v}")
            for k, v in CELLPOSE_PARAMS.items():
                logger.info(f"[CELLPOSE] {k} = {v}")
            logger.info("==============================================")

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            logger.info(f"Available in {PROJECT_DIRS['data']}:")
            for f in os.listdir(PROJECT_DIRS["data"]):
                logger.info(f"  - {f}")
            return 1

        logger.info("Preprocessing image...")
        image = preprocess_image(image_path, SETTINGS, logger)

        model_type = CELLPOSE_PARAMS["model_type"]
        logger.info(f"Using Cellpose model: {model_type}")
        model = models.Cellpose(model_type=model_type, gpu=CELLPOSE_PARAMS["gpu"])
        device = 'cuda' if CELLPOSE_PARAMS["gpu"] and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        logger.info("Running segmentation...")
        masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, SETTINGS, logger)

        if masks is None or masks.size == 0:
            logger.error("No segmentation masks were returned. Aborting.")
            return 1

        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "masks.npy"), masks)
        np.save(os.path.join(masks_dir, "masks.npy"), masks)
        np.savez(os.path.join(output_dir, "flows.npz"),
                 flow0=flows[0], flow1=flows[1], cellprob=flows[2])
        skio.imsave(os.path.join(output_dir, "segmentation_mask.tif"), masks.astype(np.uint16))
        skio.imsave(os.path.join(masks_dir, "segmentation_mask.tif"), masks.astype(np.uint16))
        logger.info(f"Segmentation completed. Total cells: {total_cells}")

        if SETTINGS.get("USE_EDGE_DETECTION", False):
            logger.info("Applying edge refinement...")
            image, masks = ensure_matching_shapes(image, masks, logger)
            masks = refine_segmentation_with_edges(image, masks, SETTINGS, logger)
            skio.imsave(os.path.join(output_dir, "refined_segmentation_mask.tif"), masks.astype(np.uint16))
            logger.info("Edge refinement completed.")

        if SETTINGS.get("APPLY_WATERSHED", False):
            logger.info("Applying watershed splitting...")
            try:
                masks = apply_watershed_to_mask(
                    masks,
                    min_area=SETTINGS.get("AREA_THRESHOLD_FOR_WATERSHED", 1000),
                    footprint=SETTINGS.get("LOCAL_MAXIMA_FOOTPRINT", (3, 3)),
                    logger=logger
                )
                skio.imsave(os.path.join(output_dir, "segmentation_mask_post_watershed.tif"), masks.astype(np.uint16))
                np.save(os.path.join(output_dir, "segmentation_mask_post_watershed.npy"), masks)
                logger.info("Watershed segmentation saved.")
            except Exception as e:
                logger.error("Watershed error: " + str(e))
                logger.error(traceback.format_exc())

        if SETTINGS.get("GENERATE_OVERLAY", False):
            logger.info("Generating overlay...")
            image, masks = ensure_matching_shapes(image, masks, logger)
            generate_full_overlay(image, masks, flows, output_dir, logger)

        try:
            logger.info("Generating small overlay snippet...")
            small_segmentation_overlay(
                output_dir,
                crop_size=SETTINGS.get("SMALL_OVERLAY_SIZE", 512) * SETTINGS.get("UPSCALE_FACTOR", 1),
                debug=SETTINGS.get("DEBUG_MODE", False)
            )
        except Exception as e:
            logger.error(f"Overlay snippet generation error: {e}")
            logger.error(traceback.format_exc())

        # Optional: trigger debug snapshot
        if snap:
            snap.capture("end_of_pipeline", {"masks": masks})

        logger.info("===== Pipeline Completed Successfully =====")
        return 0

    except Exception as e:
        logger.error("Fatal pipeline error: " + str(e))
        logger.error(traceback.format_exc())
        return 1

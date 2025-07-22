"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: pipeline.py.
Description:
    Modular segmentation flow using Cellpose3 with DAPI-stained nuclear features.

Dependencies:
    • Python >= 3.7.
    • numpy, skimage, torch, cellpose.
    • Custom utility modules for preprocessing, segmentation, watershed, and visualization.

Usage:
    Not meant to be run directly. Imported by run_this.py.

Inputs:
    • DAPI-stained images of kidney tissue sections.
    • Configuration settings for segmentation parameters.

Outputs:
    • Segmentation masks as .npy and .tif files.
    • Flow fields from Cellpose as .npz files.
    • Optional refined masks with edge detection and watershed.
    • Visualization overlays for quality control.

Key Features:
    • Modular pipeline architecture for flexible processing.
    • GPU acceleration when available.
    • Tiling support for large images.
    • Post-processing with edge refinement and watershed.
    • Comprehensive logging and error handling.

Notes:
    • This module is part of the nuclei segmentation package for kidney I/R injury analysis.
    • It implements the core segmentation workflow but is not meant to be run directly.
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
from utils.overlay_full_image import full_image_overlay

from cellpose_merge.merge_tiles import merge_masks_streaming


def log_config(logger, settings, CELLPOSE_PARAMS):
    if settings.get("debug_mode", False):
        logger.debug("=== settings ===")
        for k, v in settings.items():
            logger.debug(f"{k}: {v}")
        logger.debug("=== CELLPOSE ===")
        for k, v in CELLPOSE_PARAMS.items():
            logger.debug(f"{k}: {v}")


def setup_model(CELLPOSE_PARAMS, logger):
    """
    Initialize Cellpose model for nuclear segmentation.

    Uses CellposeModel (Cellpose 4.0+) instead of deprecated Cellpose class.
    Automatically detects GPU availability and configures device accordingly.

    Parameters
    ----------
    CELLPOSE_PARAMS : dict
        Configuration parameters for Cellpose model initialization.
    logger : logging.Logger
        Logger instance for status reporting.

    Returns
    -------
    cellpose.models.CellposeModel
        Initialized Cellpose model ready for segmentation.
    """
    model_type = CELLPOSE_PARAMS["model_type"]
    use_gpu = CELLPOSE_PARAMS.get("gpu", False)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    # Use CellposeModel instead of deprecated Cellpose class (Cellpose 4.0+).
    model = models.CellposeModel(model_type=model_type, gpu=(device == 'cuda'))

    logger.info(f"Using Cellpose model: {model_type}")
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            logger.warning("Failed to get CUDA device name.")

    # Log key Cellpose parameters for adaptive diameter detection.
    diameter = CELLPOSE_PARAMS.get("diameter", 0)
    resample = CELLPOSE_PARAMS.get("resample", True)
    cellprob_threshold = CELLPOSE_PARAMS.get("cellprob_threshold", -9.0)
    flow_threshold = CELLPOSE_PARAMS.get("flow_threshold", 0.4)

    logger.info(f"Cellpose parameters: diameter={diameter} (0=auto-detect), resample={resample}")
    logger.info(f"Detection thresholds: cellprob={cellprob_threshold}, flow={flow_threshold}")

    if diameter == 0 and resample:
        logger.info("ADAPTIVE DIAMETER: Adaptive diameter detection enabled - will optimize per tile")
    elif diameter == 0 and not resample:
        logger.warning("WARNING: diameter=0 with resample=False may not work optimally")
    else:
        logger.info(f"FIXED DIAMETER: Fixed diameter mode: {diameter}px")

    return model


def save_outputs(masks, flows, output_dir, logger):
    output_dir = Path(output_dir)
    masks_dir  = output_dir / "masks"
    flows_dir  = output_dir / "flows"
    masks_dir.mkdir(parents=True, exist_ok=True)
    flows_dir.mkdir(parents=True, exist_ok=True)

    """Masks"""
    dest_npy = masks_dir / "segmentation_masks.npy"
    if isinstance(masks, np.memmap):
        if Path(masks.filename).resolve() != dest_npy.resolve():
            # Use np.asarray without copy parameter for NumPy compatibility.
            np.save(dest_npy, np.asarray(masks))
    else:
        np.save(dest_npy, masks)

    try:
        tif_path = masks_dir / "segmentation_masks.tif"
        skio.imsave(tif_path, masks.astype(np.uint32), plugin="tifffile")
    except Exception as e:
        logger.warning(f"Could not write large TIFF: {e}")

    """Flows (optional)"""
    if flows and all(f is not None for f in flows):
        np.savez(flows_dir / "flows.npz",
                 flow0=flows[0], flow1=flows[1], cellprob=flows[2])
    else:
        logger.info("Flows were disabled; skipping flow output.")



def apply_postprocessing(image, masks, settings, output_dir, logger):
    if settings.get("use_edge_detection", False):
        logger.info("Running edge refinement...")
        masks = refine_segmentation_with_edges(image, masks, settings, logger)
        skio.imsave(Path(output_dir) / "refined_segmentation_masks.tif", masks.astype(np.uint32))

    if settings.get("apply_watershed", False):
        logger.info("Applying watershed...")
        try:
            masks = apply_watershed_to_mask(
                masks,
                min_area=settings.get("area_threshold_for_watershed", 1000),
                footprint=settings.get("local_maxima_footprint", (3, 3)),
                logger=logger
            )
            skio.imsave(Path(output_dir) / "segmentation_masks_post_watershed.tif", masks.astype(np.uint32))
            np.save(Path(output_dir) / "segmentation_masks_post_watershed.npy", masks)
        except Exception as e:
            logger.error(f"Watershed error: {e}")
            logger.debug(traceback.format_exc())

    return masks


def generate_overlays(output_dir, settings, logger, image_path=None, mask_path=None, previous_results_dir=None):
    """
    Generate visualization overlays for cell segmentation results.

    This function creates both small cropped previews and full-image overlays to help
    users assess segmentation quality across different I/R injury time points.
    It supports flexible input paths to work correctly when using previous results from
    different directories.

    Parameters
    ----------
    output_dir : str or Path
        Directory where visualization outputs will be saved.
    settings : dict
        Configuration settings including overlay parameters.
    logger : logging.Logger
        Logger for progress tracking and error reporting.
    image_path : str or Path, optional
        Path to the preprocessed image file. If None, defaults to output_dir/preprocessed/first.tif.
    mask_path : str or Path, optional
        Path to the segmentation masks file. If None, defaults to output_dir/masks/segmentation_masks.npy.
    previous_results_dir : str or Path, optional
        Path to previous results directory for accessing preprocessed images when using previous results.
    """
    try:
        # Determine paths for overlay generation based on provided parameters or defaults.
        if image_path is None:
            image_path = Path(output_dir) / "preprocessed" / "first.tif"
        else:
            image_path = Path(image_path)

        if mask_path is None:
            mask_path = Path(output_dir) / "masks" / "segmentation_masks.npy"
        else:
            mask_path = Path(mask_path)

        logger.info(f"Generating overlays using image: {image_path}")
        logger.info(f"Generating overlays using masks: {mask_path}")

        # Verify that the required files exist before attempting overlay generation.
        if not image_path.exists():
            logger.warning(f"Preprocessed image not found: {image_path}")
            logger.info("Skipping overlay generation due to missing preprocessed image")
            return

        if not mask_path.exists():
            logger.warning(f"Segmentation masks not found: {mask_path}")
            logger.info("Skipping overlay generation due to missing segmentation masks")
            return

        # Generate small cropped overlay for quick quality assessment.
        # Determine preprocessed directory for CLAHE and gamma images.
        if (previous_results_dir is not None and
            settings.get("use_previous_results", False) and
            settings.get("skip_and_copy_preprocessing", False)):
            preprocessed_source_dir = previous_results_dir / "preprocessed"
        else:
            preprocessed_source_dir = Path(output_dir) / "preprocessed"

        small_segmentation_overlay(
            output_dir=output_dir,
            crop_size=settings.get("small_overlay_size", 512) * settings.get("upscale_factor", 1),
            debug=settings.get("debug_mode", False),
            image_path=image_path,
            mask_path=mask_path,
            preprocessed_dir=preprocessed_source_dir
        )

        # Generate full-image overlay for comprehensive visualization.
        full_image_overlay(
            Path(output_dir) / "visualizations",
            logger,
            img_path=image_path,
            mask_path=mask_path,
        )

        logger.info("Overlay generation completed successfully")

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

        if settings.get("use_previous_results"):
            # Use the already-resolved path from settings to avoid path concatenation issues.
            previous_results_dir = settings["previous_results_dir"]
            logger.info("Using previous results from: {}".format(previous_results_dir))

            # Ensure the previous results directory exists.
            if not previous_results_dir.exists():
                logger.error(f"Previous results directory not found: {previous_results_dir}")
                return 1

        if not settings.get("skip_and_copy_preprocessing", False) or not settings.get("use_previous_results", False):
            # Step 1: Image preprocessing (CLAHE, cropping etc.).
            image = preprocess_image(image_path, settings, logger)
        else:
            image = skio.imread(previous_results_dir / "preprocessed" / "final.tif")
            logger.info("Skipped preprocessing and loaded image from: {}".format(previous_results_dir / "preprocessed" / "final.tif"))

        if not settings.get("skip_and_copy_segmentation", False) or not settings.get("use_previous_results", False):
            # Step 2: Load Cellpose model (with GPU if available).
            model = setup_model(CELLPOSE_PARAMS, logger)

            # Step 3: Segmentation.
            logger.info("Running Cellpose segmentation...")
            masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, settings, logger)

            # Step 4: Save raw outputs.
            save_outputs(masks, flows, output_dir, logger)
            logger.info("Segmentation outputs saved to: {}".format(output_dir / "masks" / "segmentation_masks.npy"))
        else:
            # If non_merged_segmentation_masks.npy exists, load it instead of segmentation_masks.npy.
            if (previous_results_dir / "masks" / "non_merged_segmentation_masks.npy").exists():
                masks = np.load(previous_results_dir / "masks" / "non_merged_segmentation_masks.npy")
            else:
                masks = np.load(previous_results_dir / "masks" / "segmentation_masks.npy")
            flows = [None, None, None]
            total_cells = int(masks.max())
            logger.info("Skipped segmentation and loaded masks from: {}".format(previous_results_dir / "masks" / "segmentation_masks.npy"))

        if not settings.get("skip_and_copy_merging", False) or not settings.get("use_previous_results", False):
            # Step 5: Merge masks in tile overlaps.
            if settings.get("use_tiling", False):
                logger.info("Merging masks across tile overlaps...")

                # Convert fractional overlap to pixels for the merge function.
                # This ensures consistency with the segmentation tiling parameters.
                tile_size = settings["tile_side_length"]
                overlap_cfg = settings["tile_overlap"]

                if 0 <= overlap_cfg <= 1:
                    overlap_pixels = int(tile_size * overlap_cfg)
                else:
                    overlap_pixels = int(overlap_cfg)

                # Ensure overlap doesn't exceed half the tile size.
                overlap_pixels = min(overlap_pixels, tile_size // 2)

                logger.info(f"Tile merge parameters: tile_size={tile_size}, overlap={overlap_pixels} pixels")

                # Determine the correct path for tile masks based on whether using previous results.
                if settings.get("use_previous_results", False) and settings.get("skip_and_copy_segmentation", False):
                    # Use tile masks from previous results directory.
                    tiles_source_path = previous_results_dir / "masks" / "tile_masks_npz"
                    logger.info(f"Using tile masks from previous results: {tiles_source_path}")
                else:
                    # Use tile masks from current output directory.
                    tiles_source_path = output_dir / "masks" / "tile_masks_npz"
                    logger.info(f"Using tile masks from current run: {tiles_source_path}")

                masks = merge_masks_streaming(
                    height=image.shape[0],
                    width=image.shape[1],
                    tile_h=settings["tile_side_length"],
                    tile_w=settings["tile_side_length"],
                    overlap=overlap_pixels,
                    tiles_path=tiles_source_path,
                    threshold=settings.get("merge_overlap_threshold", 0.3),
                    qc=settings.get("qc_overlays", True),
                    qc_dir=settings.get("qc_dir", output_dir / "merge_qc_overlays"),
                    qc_merge_use_full_image=settings.get("qc_merge_use_full_image", False),
                    merge_batch_size=settings.get("merge_batch_size", 4),
                    use_two_phase_merge=settings.get("use_two_phase_merge", True),
                )
                # Note: The merge_masks_streaming function now saves directly to masks/ directory.
                # No need for additional save_outputs call to avoid duplicate files.
                logger.info("Merged masks saved to: {}".format(output_dir / "masks" / "segmentation_masks.npy"))

                total_cells = int(masks.max())
        else:
            masks = np.load(previous_results_dir / "masks" / "segmentation_masks.npy")
            logger.info("Skipped merging.")

        if masks is None or masks.size == 0:
            logger.error("No segmentation masks returned. Aborting.")
            return 1

        logger.info(f"Segmentation completed. Total cells: {total_cells}")

        if not settings.get("skip_and_copy_postprocessing", False) or not settings.get("use_previous_results", False):
            # Step 6: Postprocess with edge refinement and/or watershed.
            masks = apply_postprocessing(image, masks, settings, output_dir, logger)
        else:
            logger.info("Skipped postprocessing.")

        if not settings.get("skip_and_copy_visualization", False) or not settings.get("use_previous_results", False):
            # Step 7: Overlay generation (cropped + full).
            # Determine correct paths for overlay generation based on pipeline configuration.

            # Image path: use previous results if preprocessing was skipped.
            if settings.get("use_previous_results", False) and settings.get("skip_and_copy_preprocessing", False):
                overlay_image_path = previous_results_dir / "preprocessed" / "first.tif"
                logger.info(f"Using preprocessed image from previous results for overlays: {overlay_image_path}")
            else:
                overlay_image_path = Path(output_dir) / "preprocessed" / "first.tif"
                logger.info(f"Using preprocessed image from current run for overlays: {overlay_image_path}")

            # Mask path: always use the current output directory masks (either from segmentation or merging).
            overlay_mask_path = Path(output_dir) / "masks" / "segmentation_masks.npy"
            logger.info(f"Using segmentation masks for overlays: {overlay_mask_path}")

            # Determine previous results directory if using previous results.
            prev_results_dir = None
            if settings.get("use_previous_results", False):
                prev_results_dir = previous_results_dir

            generate_overlays(
                output_dir=output_dir,
                settings=settings,
                logger=logger,
                image_path=overlay_image_path,
                mask_path=overlay_mask_path,
                previous_results_dir=prev_results_dir
            )
        else:
            logger.info("Skipped visualizations.")

        # Step 8: Optional debug snapshot.
        if snap:
            snap.capture("end_of_pipeline", {"masks": masks})

        logger.info("===== Pipeline Completed Successfully =====")
        return 0

    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.debug(traceback.format_exc())
        return 1

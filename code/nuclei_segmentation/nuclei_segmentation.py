#!/usr/bin/env python3
"""
Advanced Nuclei Segmentation Pipeline for Spatial Multiomics Analysis of Kidney I/R Injury.

This script processes large microscopy images of kidney tissue sections, optimized for DAPI-stained nuclei.
The pipeline is specifically designed for analyzing ischemia-reperfusion (I/R) kidney injury, a condition
where temporary blood flow restriction followed by restoration causes complex cellular damage patterns.

The pipeline includes:
1. Preprocessing with contrast enhancement and optional cropping to optimize nuclear signal detection.
2. Intelligent tiling for handling large histological images efficiently (>1GB).
3. Cellpose-based deep learning segmentation with parameters optimized for kidney nuclei.
4. Optional refinement using edge detection and watershed splitting for merged nuclei.
5. Comprehensive visualization and quality control outputs for scientific validation.

Designed for analyzing kidney I/R injury across multiple timepoints (10h, 2d, 14d) to capture the
dynamic progression of injury, inflammation, and repair processes. The segmentation results enable
downstream spatial analysis of transcriptomic and metabolomic data in the context of nuclear morphology.

Outputs include segmentation masks, feature measurements, and visualization overlays that can be
used for further quantitative analysis of nuclear morphology changes during I/R injury progression.
"""


# Standard library imports.
import os
import sys
import traceback
import numpy as np
import torch
from pathlib import Path

# Scientific and image processing imports.
from skimage import io as skio
from cellpose import models

# Import utility modules.
from code.nuclei_segmentation.utils.project_setup import setup_project_structure, load_config, choose_batch_size
from code.nuclei_segmentation.utils.preprocessing import preprocess_image, ensure_matching_shapes
from code.nuclei_segmentation.utils.logging_utils import setup_logging, setup_debug
from code.nuclei_segmentation.utils.segmentation import run_cellpose_on_tiles
from code.nuclei_segmentation.utils.watershed import apply_watershed_to_mask, refine_segmentation_with_edges
from code.nuclei_segmentation.utils.visualization import small_segmentation_overlay, generate_full_overlay


def main():
    """
    Main function for the nuclei segmentation pipeline.
    
    This function orchestrates the entire segmentation workflow, including:
    1. Loading configuration settings
    2. Setting up logging and debugging
    3. Preprocessing the input image
    4. Running Cellpose segmentation
    5. Applying optional refinements (edge detection, watershed)
    6. Generating visualizations
    7. Saving results
    
    The function is designed to be robust to errors, with comprehensive logging
    and error handling at each step.
    
    Returns:
        int: 0 for successful execution, 1 for errors
    """
    try:
        # 1. Load configuration settings for the segmentation pipeline.
        SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()

        # Create output directory for storing results and visualizations.
        output_dir = SETTINGS["OUTPUT_DIR"]
        os.makedirs(output_dir, exist_ok=True)

        # 2. Set up logging system for tracking progress and debugging.
        logger = setup_logging(output_dir, debug_mode=SETTINGS.get("DEBUG_MODE", False))
        logger.info("===== Cellpose Segmentation Pipeline Started =====")

        # Log configuration details for reproducibility.
        logger.info(f"Image path: {SETTINGS['IMAGE_PATH']}.")
        logger.info(f"Output directory: {output_dir}.")

        # Set up debug snapshot utility if debug mode is enabled.
        snap = setup_debug(SETTINGS)

        # Enhanced debug output to show all parameters.
        if SETTINGS.get("DEBUG_MODE", False):
            logger.info("========== CONFIGURATION PARAMETERS ==========")
            logger.info("[General Settings]")
            for key in sorted([k for k in SETTINGS.keys() if k not in CELLPOSE_PARAMS]):
                logger.info(f"  {key} = {SETTINGS[key]}")

            logger.info("\n[Cellpose Parameters]")
            for key in sorted(CELLPOSE_PARAMS.keys()):
                logger.info(f"  {key} = {CELLPOSE_PARAMS[key]}")

            # Check for missing parameters that might be used in the code.
            critical_params = [
                "MERGE_OVERLAP_THRESHOLD",
                "USE_TILING",
                "APPLY_WATERSHED",
                "USE_EDGE_DETECTION",
                "ENHANCE_CONTRAST",
                "ENHANCE_DIM"
            ]

            missing_params = [param for param in critical_params if param not in SETTINGS]
            if missing_params:
                logger.warning("\n[WARNING] The following critical parameters are missing from settings:")
                for param in missing_params:
                    logger.warning(f"  {param} - This may cause errors in the pipeline!")
                    # Add default values for missing parameters to prevent crashes.
                    if param == "MERGE_OVERLAP_THRESHOLD":
                        SETTINGS[param] = 0.3
                        logger.warning(f"    Added default value: {param} = {SETTINGS[param]}")

            logger.info("==============================================\n")

        # Standard logging.
        logger.info(f"Using tiling: {SETTINGS.get('USE_TILING', False)}.")
        if SETTINGS.get('USE_TILING', False):
            logger.info(f"Tile size: {SETTINGS.get('tile_side_length', 'Not specified')}.")
            logger.info(f"Tile overlap: {SETTINGS.get('TILE_OVERLAP', 'Not specified')}.")
            logger.info(f"Merge overlap threshold: {SETTINGS.get('MERGE_OVERLAP_THRESHOLD', 'Not specified (using default 0.3)')}.")

        # Verify image path exists before proceeding.
        image_path = SETTINGS["IMAGE_PATH"]
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            logger.error("Please check the image_path in your configuration file.")

            # Try to provide helpful suggestions.
            data_dir = PROJECT_DIRS["data"]
            if os.path.exists(data_dir):
                # List available image files in the data directory.
                image_files = [f for f in os.listdir(data_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                if image_files:
                    logger.info(f"Available image files in {data_dir}:")
                    for img_file in image_files:
                        logger.info(f"  - {img_file}")
                    logger.info("Update your config file with one of these image names.")
                else:
                    logger.info(f"No image files found in {data_dir}. Please add images to this directory.")

            return 1

        # 3. Preprocess the image to optimize for nuclei detection.
        logger.info("Preprocessing image...")
        image = preprocess_image(SETTINGS["IMAGE_PATH"], SETTINGS, logger)
        
        # Save a debug snapshot of the preprocessed image if in debug mode.
        snap("preprocessed_image", image)

        # 4. Initialize Cellpose model for deep learning-based segmentation.
        logger.info("Initializing Cellpose model...")

        # Ensure we're using the correct model type for kidney nuclei.
        model_type = CELLPOSE_PARAMS["model_type"]
        logger.info(f"Using Cellpose model: {model_type}.")

        # Initialize the model with pretrained weights optimized for nuclei.
        model = models.Cellpose(model_type=model_type, gpu=CELLPOSE_PARAMS["gpu"])

        # Log device information for performance tracking.
        logger.info(f"Using device: {'cuda' if CELLPOSE_PARAMS['gpu'] and torch.cuda.is_available() else 'cpu'}.")
        if CELLPOSE_PARAMS["gpu"] and torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}.")
        else:
            logger.info("GPU not available or not enabled, using CPU.")

        # Calculate optimal batch size based on tile dimensions.
        tile_pixels = SETTINGS["tile_side_length"] ** 2
        CELLPOSE_PARAMS["batch_size"] = choose_batch_size(tile_pixels)
        logger.info(f"Using batch size: {CELLPOSE_PARAMS['batch_size']}.")

        # 5. Run segmentation on the preprocessed image.
        logger.info("Running segmentation...")
        masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, SETTINGS, logger)
        
        # Save a debug snapshot of the initial segmentation if in debug mode.
        snap("initial_segmentation", masks)

        # 6. Save results.
        logger.info("Saving segmentation results...")

        # Create masks directory.
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Save masks in both root and masks directory for compatibility.
        np.save(os.path.join(output_dir, "masks.npy"), masks)
        np.save(os.path.join(masks_dir, "masks.npy"), masks)

        # Save flows.
        np.savez(os.path.join(output_dir, "flows.npz"),
                 flow0=flows[0],
                 flow1=flows[1],
                 cellprob=flows[2])

        # Save visualization-friendly versions.
        skio.imsave(os.path.join(output_dir, "segmentation_mask.png"), masks.astype(np.uint16))
        skio.imsave(os.path.join(masks_dir, "segmentation_mask.png"), masks.astype(np.uint16))

        logger.info(f"Saved segmentation mask and flows. Total cells detected: {total_cells}")

        # 7. Optional: Edge detection refinement.
        if SETTINGS.get("USE_EDGE_DETECTION", False):
            logger.info("Applying edge detection refinement...")
            # Ensure image and masks have matching shapes before edge detection.
            image_matched, masks_matched = ensure_matching_shapes(image, masks, logger)
            masks = refine_segmentation_with_edges(image_matched, masks_matched, SETTINGS, logger)
            skio.imsave(os.path.join(output_dir, "refined_segmentation_mask.png"), masks.astype(np.uint16))
            logger.info("Saved refined segmentation mask after edge detection.")
            
            # Save a debug snapshot of the edge-refined segmentation if in debug mode.
            snap("edge_refined_segmentation", masks)

        # 8. Optional: Watershed splitting.
        if SETTINGS.get("APPLY_WATERSHED", False):
            logger.info("Applying watershed splitting to large objects...")
            try:
                # Use the watershed module.
                lumps_split_mask = apply_watershed_to_mask(
                    masks,
                    min_area=SETTINGS.get("AREA_THRESHOLD_FOR_WATERSHED", 1000),
                    footprint=SETTINGS.get("LOCAL_MAXIMA_FOOTPRINT", (3, 3)),
                    logger=logger
                )

                # Save the watershed results.
                skio.imsave(os.path.join(output_dir, "segmentation_mask_post_watershed.png"),
                            lumps_split_mask.astype(np.uint16))
                np.save(os.path.join(output_dir, "segmentation_mask_post_watershed.npy"), lumps_split_mask)

                # Update the masks variable with the watershed results.
                masks = lumps_split_mask
                logger.info("Saved watershed-processed segmentation mask.")
                
                # Save a debug snapshot of the watershed-refined segmentation if in debug mode.
                snap("watershed_refined_segmentation", masks)

                # If in debug mode, also save a comparison visualization.
                if SETTINGS.get("DEBUG_MODE", False):
                    import matplotlib.pyplot as plt
                    
                    # Create a side-by-side comparison of before and after watershed.
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(models.plot.mask_overlay(image, masks, colors=np.random.rand(np.max(masks) + 1, 3)))
                    axes[0].set_title("After Watershed Splitting")
                    axes[0].axis('off')

                    # Load the original masks for comparison.
                    orig_masks = np.load(os.path.join(output_dir, "masks.npy"))
                    axes[1].imshow(models.plot.mask_overlay(image, orig_masks, colors=np.random.rand(np.max(orig_masks) + 1, 3)))
                    axes[1].set_title("Before Watershed Splitting")
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "watershed_comparison.png"), dpi=300)
                    plt.close()
                    logger.info("Saved watershed comparison visualization.")
            except Exception as e:
                logger.error(f"Error in watershed splitting: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing with original masks due to watershed error.")

        # 9. Optional: Generate overlay visualization.
        if SETTINGS.get("GENERATE_OVERLAY", False):
            logger.info("Generating overlay visualization...")
            # Ensure image and masks have matching shapes before generating overlay.
            image_matched, masks_matched = ensure_matching_shapes(image, masks, logger)
            # Generate the full overlay for comprehensive visualization.
            generate_full_overlay(image_matched, masks_matched, flows, output_dir, logger)

        # 10. Create a small overlay snippet (cropped) for quick review.
        logger.info("Generating small overlay snippet...")
        try:
            # Pass the debug parameter from settings to enable detailed diagnostics when needed.
            small_segmentation_overlay(
                output_dir,
                crop_size=SETTINGS.get("SMALL_OVERLAY_SIZE", 512) * SETTINGS.get("UPSCALE_FACTOR", 1),
                debug=SETTINGS.get("DEBUG_MODE", False)
            )
            logger.info("Small overlay snippet generated successfully.")
        except Exception as e:
            logger.error(f"Error generating small overlay snippet: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing despite overlay generation error.")

        logger.info("===== Cellpose Segmentation Pipeline Completed Successfully =====")
        return 0
    except KeyError as e:
        logger.error(f"Configuration error: Missing required key {e}. Please check your configuration file.")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()

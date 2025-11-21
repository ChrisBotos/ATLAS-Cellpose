"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: pipeline.py.
Description:
    Modular segmentation flow using Cellpose3 (recommended) with DAPI-stained nuclear features.
    This pipeline implements the ATLAS-Cellpose workflow with adaptive tiled processing,
    systematic 4-step merging, and comprehensive quality control.

Dependencies:
    • Python >= 3.10.
    • numpy, skimage, torch, cellpose.
    • Custom utility modules for preprocessing, segmentation, watershed, and visualization.

Usage:
    Not meant to be run directly. Imported by run_this.py.
    Recommended execution: ./run_segmentation_instance.sh crop_box "0.38,0.42,0.32,0.36"

Inputs:
    • DAPI-stained images of tissue sections.
    • Configuration settings for segmentation parameters.

Outputs:
    • Segmentation masks as .npy and .tif files.
    • Flow fields from Cellpose as .npz files.
    • Optional refined masks with edge detection and watershed.
    • Visualization overlays for quality control.

Key Features:
    • Modular pipeline architecture for flexible processing.
    • GPU acceleration when available.
    • Adaptive tiling support for large images.
    • 4-step systematic merging algorithm for tile overlap resolution.
    • Post-processing with edge refinement and watershed.
    • Comprehensive logging and error handling.

Notes:
    • This module is part of the ATLAS-Cellpose nuclei segmentation package.
    • It implements the core segmentation workflow but is not meant to be run directly.
    • See README.md for complete pipeline documentation and flowcharts.
"""

import traceback
import numpy as np
from pathlib import Path
from skimage import io as skio
import torch
from cellpose import models

from rich.console import Console

from utils.preprocessing import preprocess_image
from utils.segmentation import run_cellpose_on_tiles
from utils.watershed import refine_segmentation_with_edges, apply_watershed_to_mask
from utils.visualization import small_segmentation_overlay
from utils.overlay_masks import overlay
from utils.filter_masks import filter_masks_programmatic

from cellpose_merge.merge_tiles import merge_masks_streaming

# Initialize Rich console for formatted output.
console = Console()


def log_config(logger, settings, CELLPOSE_PARAMS):
    if settings.get("debug_mode", False):
        logger.debug("=== settings ===")
        for k, v in settings.items():
            logger.debug(f"{k}: {v}")
        logger.debug("=== CELLPOSE ===")
        for k, v in CELLPOSE_PARAMS.items():
            logger.debug(f"{k}: {v}")


def setup_model(CELLPOSE_PARAMS, logger, debug_mode=False):
    """
    Initialize Cellpose model for nuclear segmentation with configurable version support.

    Supports both Cellpose3 and Cellpose4 based on configuration parameter.

    Parameters
    ----------
    CELLPOSE_PARAMS : dict
        Configuration parameters for Cellpose model initialization.
        Must include 'use_cellpose4' boolean parameter.
    logger : logging.Logger
        Logger instance for status reporting.
    debug_mode : bool
        If True, shows detailed parameter information.

    Returns
    -------
    cellpose.models.CellposeModel or cellpose.models.Cellpose
        Initialized Cellpose model ready for segmentation.
    """
    model_type = CELLPOSE_PARAMS["model_type"]
    use_gpu = CELLPOSE_PARAMS.get("gpu", False)
    use_cellpose4 = CELLPOSE_PARAMS.get("use_cellpose4", True)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    # Log which Cellpose version is being used.
    cellpose_version = "Cellpose4" if use_cellpose4 else "Cellpose3"
    console.print(f"\n[cyan]Initializing {cellpose_version} model...[/cyan]")

    if use_cellpose4:
        # Use CellposeModel (Cellpose4) with model_type parameter.
        model = models.CellposeModel(model_type=model_type, gpu=(device == 'cuda'))
        console.print(f"[green]✓[/green] Cellpose4 model initialized: [bold]{model_type}[/bold]")
    else:
        # Try to use Cellpose (Cellpose3) with model_type parameter.
        # Note: This requires Cellpose3 to be installed instead of Cellpose4.
        try:
            model = models.Cellpose(model_type=model_type, gpu=(device == 'cuda'))
            console.print(f"[green]✓[/green] Cellpose3 model initialized: [bold]{model_type}[/bold]")
        except AttributeError:
            console.print("[yellow]⚠[/yellow] Cellpose3 API not available, falling back to Cellpose4")
            if debug_mode:
                logger.warning("To use true Cellpose3, please install cellpose<4.0")
            model = models.CellposeModel(model_type=model_type, gpu=(device == 'cuda'))
            console.print(f"[green]✓[/green] Using Cellpose4 model as fallback: [bold]{model_type}[/bold]")
            # Update the parameter to reflect actual usage.
            CELLPOSE_PARAMS["use_cellpose4"] = True

    # Display device information.
    if device == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"[green]✓[/green] Using GPU: [bold]{gpu_name}[/bold]")
        except Exception:
            console.print(f"[green]✓[/green] Using device: [bold]{device}[/bold]")
    else:
        console.print(f"[blue]ℹ[/blue] Using device: [bold]{device}[/bold]")

    # Log key Cellpose parameters for adaptive diameter detection.
    diameter = CELLPOSE_PARAMS.get("diameter", 0)
    resample = CELLPOSE_PARAMS.get("resample", True)
    cellprob_threshold = CELLPOSE_PARAMS.get("cellprob_threshold", -9.0)
    flow_threshold = CELLPOSE_PARAMS.get("flow_threshold", 0.4)

    if debug_mode:
        console.print(f"[dim]  Parameters: diameter={diameter}, resample={resample}[/dim]")
        console.print(f"[dim]  Thresholds: cellprob={cellprob_threshold}, flow={flow_threshold}[/dim]")

    if (diameter == 0 or diameter is None) and resample:
        console.print("[cyan]  Mode: Adaptive diameter detection enabled[/cyan]")
    elif (diameter == 0 or diameter is None) and not resample:
        console.print("[yellow]⚠[/yellow] Warning: diameter=0/None with resample=False may not work optimally")
    else:
        console.print(f"[cyan]  Mode: Fixed diameter = [bold]{diameter}px[/bold][/cyan]")

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


def generate_overlays(output_dir, settings, logger, image_path=None, mask_path=None, previous_results_dir=None, overlay_suffix=""):
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
    overlay_suffix : str, optional
        Suffix to append to overlay filenames (e.g., "_unfiltered" or "_filtered").
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
        overlay_filename = f"full_image_overlay{overlay_suffix}.tif"
        overlay_output_path = Path(output_dir) / "visualizations" / overlay_filename
        Path(output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating high-quality overlay: {image_path} × {mask_path}")

        # Import OverlayConfig to pass debug_mode.
        from utils.overlay_masks import OverlayConfig

        overlay_config = OverlayConfig(debug_mode=settings.get("debug_mode", False))
        overlay(
            image_path=image_path,
            mask_path=mask_path,
            output_path=overlay_output_path,
            config=overlay_config
        )

        logger.info(f"Overlay generation completed successfully: {overlay_filename}")

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
        # Get debug mode setting.
        debug_mode = settings.get("debug_mode", False)

        # Log current configuration.
        log_config(logger, settings, CELLPOSE_PARAMS)

        # Normalize paths early.
        image_path = Path(settings["image_path"]).expanduser().resolve()
        output_dir = Path(settings["output_dir"]).expanduser().resolve()
        settings["output_dir"] = str(output_dir)  # Ensure all downstream functions use the same path.

        # Check image exists.
        if not image_path.exists():
            console.print(f"[red]✗[/red] Image file not found: {image_path}")
            logger.error(f"Image file not found: {image_path}")
            return 1

        console.print(f"\n[cyan]Input image:[/cyan] [dim]{image_path}[/dim]")
        console.print(f"[cyan]Output directory:[/cyan] [dim]{output_dir}[/dim]")

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
            model = setup_model(CELLPOSE_PARAMS, logger, debug_mode)

            # Step 3: Segmentation.
            console.print("\n[cyan]Running Cellpose segmentation...[/cyan]")
            masks, flows, total_cells = run_cellpose_on_tiles(model, image, CELLPOSE_PARAMS, settings, logger)

            # Step 4: Save raw outputs.
            save_outputs(masks, flows, output_dir, logger)
            if debug_mode:
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
                console.print("\n[cyan]Merging masks across tile overlaps...[/cyan]")

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

                if debug_mode:
                    console.print(f"[dim]  Tile size: {tile_size}px, overlap: {overlap_pixels}px[/dim]")

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
                    use_gpu=settings.get("use_gpu", False),
                    qc=settings.get("qc_overlays", True),
                    qc_dir=settings.get("qc_dir", output_dir / "merge_qc_overlays"),
                    qc_merge_use_full_image=settings.get("qc_merge_use_full_image", False),
                    merge_batch_size=settings.get("merge_batch_size", 4),
                    debug_mode=settings.get("debug_mode", False),
                    output_dir=output_dir,
                )
                # Note: The merge_masks_streaming function now saves directly to masks/ directory.
                # No need for additional save_outputs call to avoid duplicate files.
                logger.info("Merged masks saved to: {}".format(output_dir / "masks" / "segmentation_masks.npy"))

                total_cells = int(masks.max())
        else:
            masks = np.load(previous_results_dir / "masks" / "segmentation_masks.npy")
            logger.info("Skipped merging.")

        if masks is None or masks.size == 0:
            console.print("[red]✗[/red] No segmentation masks returned. Aborting.")
            logger.error("No segmentation masks returned. Aborting.")
            return 1

        console.print(f"\n[green]✓[/green] Segmentation completed: [bold cyan]{total_cells:,}[/bold cyan] nuclei detected")

        if not settings.get("skip_and_copy_postprocessing", False) or not settings.get("use_previous_results", False):
            # Step 6: Postprocess with edge refinement and/or watershed.
            masks = apply_postprocessing(image, masks, settings, output_dir, logger)
        else:
            if debug_mode:
                console.print("[dim]  Skipped postprocessing[/dim]")

        # Store whether filtering will be applied for overlay generation.
        filtering_enabled = False

        if not settings.get("skip_and_copy_filtering", False) or not settings.get("use_previous_results", False):
            # Step 7: Apply morphological filtering to remove artifacts.
            if settings.get("use_filtering", False):
                filtering_enabled = True
                console.print("\n[cyan]Applying morphological filtering...[/cyan]")
                original_count = int(masks.max())

                # Save unfiltered masks before filtering.
                np.save(Path(output_dir) / "masks" / "segmentation_masks_unfiltered.npy", masks)

                masks = filter_masks_programmatic(
                    masks=masks,
                    settings=settings,
                    output_dir=output_dir,
                    logger=logger,
                    intensity_image=image if settings.get("use_intensity_filtering", False) else None
                )
                filtered_count = int(masks.max())
                removed = original_count - filtered_count
                console.print(f"[green]✓[/green] Filtering completed: [bold cyan]{original_count:,}[/bold cyan] → [bold cyan]{filtered_count:,}[/bold cyan] nuclei ([red]{removed:,}[/red] removed)")

                # Save filtered masks to main masks directory.
                np.save(Path(output_dir) / "masks" / "segmentation_masks_filtered.npy", masks)
                try:
                    skio.imsave(Path(output_dir) / "masks" / "segmentation_masks_filtered.tif",
                               masks.astype(np.uint32), plugin="tifffile")
                except Exception as e:
                    if debug_mode:
                        logger.warning(f"Could not write filtered TIFF: {e}")
            else:
                if debug_mode:
                    console.print("[dim]  Morphological filtering disabled[/dim]")
        else:
            if debug_mode:
                console.print("[dim]  Skipped filtering[/dim]")

        if not settings.get("skip_and_copy_visualization", False) or not settings.get("use_previous_results", False):
            # Step 8: Overlay generation (cropped + full).
            console.print("\n[cyan]Generating visualization overlays...[/cyan]")

            # Determine correct paths for overlay generation based on pipeline configuration.
            # Image path: use previous results if preprocessing was skipped.
            if settings.get("use_previous_results", False) and settings.get("skip_and_copy_preprocessing", False):
                overlay_image_path = previous_results_dir / "preprocessed" / "first.tif"
                if debug_mode:
                    logger.info(f"Using preprocessed image from previous results for overlays: {overlay_image_path}")
            else:
                overlay_image_path = Path(output_dir) / "preprocessed" / "first.tif"
                if debug_mode:
                    logger.info(f"Using preprocessed image from current run for overlays: {overlay_image_path}")

            # Determine previous results directory if using previous results.
            prev_results_dir = None
            if settings.get("use_previous_results", False):
                prev_results_dir = previous_results_dir

            # Generate overlays based on filtering status.
            if filtering_enabled:
                # Generate unfiltered overlay.
                console.print("[cyan]  Creating unfiltered overlay...[/cyan]")
                overlay_mask_path_unfiltered = Path(output_dir) / "masks" / "segmentation_masks_unfiltered.npy"
                if debug_mode:
                    logger.info(f"Using unfiltered segmentation masks for overlay: {overlay_mask_path_unfiltered}")

                generate_overlays(
                    output_dir=output_dir,
                    settings=settings,
                    logger=logger,
                    image_path=overlay_image_path,
                    mask_path=overlay_mask_path_unfiltered,
                    previous_results_dir=prev_results_dir,
                    overlay_suffix="_unfiltered"
                )

                # Generate filtered overlay.
                console.print("[cyan]  Creating filtered overlay...[/cyan]")
                overlay_mask_path_filtered = Path(output_dir) / "masks" / "segmentation_masks_filtered.npy"
                if debug_mode:
                    logger.info(f"Using filtered segmentation masks for overlay: {overlay_mask_path_filtered}")

                generate_overlays(
                    output_dir=output_dir,
                    settings=settings,
                    logger=logger,
                    image_path=overlay_image_path,
                    mask_path=overlay_mask_path_filtered,
                    previous_results_dir=prev_results_dir,
                    overlay_suffix="_filtered"
                )
            else:
                # No filtering applied, generate single overlay.
                overlay_mask_path = Path(output_dir) / "masks" / "segmentation_masks.npy"
                if debug_mode:
                    logger.info(f"Using segmentation masks for overlays: {overlay_mask_path}")

                generate_overlays(
                    output_dir=output_dir,
                    settings=settings,
                    logger=logger,
                    image_path=overlay_image_path,
                    mask_path=overlay_mask_path,
                    previous_results_dir=prev_results_dir
                )
        else:
            if debug_mode:
                console.print("[dim]  Skipped visualizations[/dim]")

        # Step 9: Optional debug snapshot.
        if snap:
            snap.capture("end_of_pipeline", {"masks": masks})

        console.print("\n[green bold]═════ Pipeline Completed Successfully ═════[/green bold]\n")
        return 0

    except Exception as e:
        console.print(f"\n[red]✗[/red] Fatal pipeline error: {e}")
        logger.error(f"Fatal pipeline error: {e}")
        logger.debug(traceback.format_exc())
        return 1

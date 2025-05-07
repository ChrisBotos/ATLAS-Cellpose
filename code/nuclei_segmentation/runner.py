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
from utils.project_setup import setup_project_structure, load_config, choose_batch_size
from utils.preprocessing import preprocess_image, ensure_matching_shapes
from utils.logging_utils import setup_logging, setup_debug
from utils.segmentation import run_cellpose_on_tiles
from utils.watershed import apply_watershed_to_mask, refine_segmentation_with_edges
from utils.visualization import small_segmentation_overlay, generate_full_overlay
from pipeline import run_segmentation_pipeline


def main():
    """
    Entry point. Loads config and delegates the full segmentation pipeline.
    Returns:
        int: 0 if successful, 1 if errors occurred.
    """
    try:
        SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()
        os.makedirs(SETTINGS["OUTPUT_DIR"], exist_ok=True)

        logger = setup_logging(SETTINGS["OUTPUT_DIR"], debug_mode=SETTINGS.get("DEBUG_MODE", False))
        snap = setup_debug(SETTINGS)

        logger.info("==== Kidney I/R Nuclei Segmentation Pipeline ====")

        # Run the full segmentation and analysis workflow.
        run_segmentation_pipeline(SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap)

        logger.info("==== Pipeline completed successfully ====")
        return 0

    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()

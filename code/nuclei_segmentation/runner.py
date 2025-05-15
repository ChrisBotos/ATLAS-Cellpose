#!/usr/bin/env python3
"""
ENTRYPOINT: Advanced nuclei segmentation pipeline for kidney I/R injury slices.
"""

import traceback

from utils.project_setup import load_config
from utils.logging_utils import setup_logging
from utils.debug_utils import setup_debug
from pipeline import run_segmentation_pipeline


def main():
    """
    Initializes logging, config, debugging, and launches the pipeline.

    Returns:
        int: 0 on success, 1 on failure.
    """
    try:
        settings, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()

        logger = setup_logging(settings["output_dir"], debug_mode=settings.get("debug_mode", False))
        snap = setup_debug(settings)

        logger.info("==== Kidney I/R Nuclei Segmentation Pipeline Started ====")
        return run_segmentation_pipeline(settings, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ENTRYPOINT: Advanced nuclei segmentation pipeline for kidney I/R injury slices.
"""

import traceback
from pathlib import Path

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
        SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()
        Path(SETTINGS["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

        logger = setup_logging(SETTINGS["OUTPUT_DIR"], debug_mode=SETTINGS.get("DEBUG_MODE", False))
        snap = setup_debug(SETTINGS)

        logger.info("==== Kidney I/R Nuclei Segmentation Pipeline Started ====")
        return run_segmentation_pipeline(SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: run_this.py.
Description:
    Entrypoint script for nuclei segmentation pipeline for kidney I/R injury tissue slices.

Dependencies:
    • Python >= 3.7.
    • Custom utility modules for project setup, logging, and debugging.
    • pipeline module for the segmentation workflow.

Usage:
    python run_this.py

Inputs:
    • Configuration files (loaded automatically from config directory).

Outputs:
    • Segmentation results in the configured output directory.
    • Log files with detailed processing information.
    • Settings snapshot for reproducibility.

Key Features:
    • Robust error handling with full traceback reporting.
    • Configuration management with automatic snapshot creation.
    • Integrated logging and debugging utilities.
    • Clean separation of initialization and processing logic.

Notes:
    • This is the main entry point for the nuclei segmentation pipeline.
    • Configuration is loaded from the project's config files rather than command-line arguments.
"""

import traceback
import json
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
        # 1) Load config and directories.
        settings, CELLPOSE_PARAMS, PROJECT_DIRS = load_config()

        # 2) Snapshot parsed settings to disk.
        snapshot_path = Path(settings["output_dir"]) / "settings_snapshot.json"

        # Build a serializable dict without inline ifs.
        serializable = {}
        for key, value in settings.items():

            if isinstance(value, Path):
                serializable[key] = str(value)
            else:
                serializable[key] = value

        # Write JSON snapshot with explicit encoding.
        with open(snapshot_path, mode="w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2)

        # 3) Determine debug mode explicitly.
        debug_mode = settings.get("debug_mode")
        if debug_mode is None:
            debug_mode = False

        # 4) Set up logging and debug.
        logger = setup_logging(settings["output_dir"], debug_mode)
        snap = setup_debug(settings)

        logger.info("==== Kidney I/R Nuclei Segmentation Pipeline Started ====")

        # 5) Run the pipeline.
        exit_code = run_segmentation_pipeline(
            settings,
            CELLPOSE_PARAMS,
            PROJECT_DIRS,
            logger,
            snap
        )

        return exit_code

    except Exception as e:
        # Always print full traceback—no swallowing errors.
        print(f"[FATAL ERROR] {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    main()

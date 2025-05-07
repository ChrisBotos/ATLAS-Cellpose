"""
Logging and Debug Utilities for Kidney I/R Injury Analysis.

This module provides specialized logging and debugging functions for the
nuclei segmentation pipeline. It includes:

1. Setup for comprehensive logging with file and console output
2. Debug snapshot utilities for saving intermediate processing results
3. Error handling and reporting functions

These utilities are essential for tracking the progress of long-running
segmentation jobs and diagnosing issues in the complex processing pipeline.
"""

# Standard library imports.
import os
import sys
import logging
import traceback
from datetime import datetime
import numpy as np
import imageio


def setup_logging(output_dir, debug_mode=False):
    """
    Configure comprehensive logging system for the segmentation pipeline.

    Creates a dual-output logging system that writes detailed logs to both a file
    and the console. The file logger captures all details including timestamps,
    while the console logger provides a more concise output for interactive use.

    Args:
        output_dir: Directory where log files will be saved.
        debug_mode: If True, sets log level to DEBUG for more detailed logging.

    Returns:
        logger: Configured logging instance ready for use throughout the pipeline.
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create logs directory within output directory.
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up log file path with timestamp for uniqueness.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"segmentation_log_{timestamp}.txt")

    # Configure root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Remove any existing handlers to avoid duplicate logging.
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Create formatters for different outputs.
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # File handler for complete logging.
    fh = logging.FileHandler(log_file)
    fh.setFormatter(file_formatter)
    fh.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.addHandler(fh)

    # Console handler for interactive feedback.
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_formatter)
    ch.setLevel(logging.INFO)  # Console always shows INFO and above.
    logger.addHandler(ch)

    logger.info("===== Nuclei Segmentation Pipeline Started =====")
    logger.info(f"Log file: {log_file}")

    # Note: Config file backup is now handled in the load_config function.
    # We log the existence of the backup file if it exists.
    config_backup_path = os.path.join(output_dir, "config_used.ini")
    if os.path.exists(config_backup_path):
        logger.info(f"Configuration file is backed up at: {config_backup_path}")
    else:
        logger.warning("Configuration backup file not found. This should have been created during config loading.")

    return logger


def setup_debug(settings):
    """
    Configure debugging environment for advanced troubleshooting.

    When debug mode is enabled, this function creates a specialized directory
    for storing intermediate processing results as images. This is invaluable for
    diagnosing segmentation issues and understanding the pipeline's behavior.

    Args:
        settings: Dictionary containing configuration settings including DEBUG_MODE.

    Returns:
        function: A function for saving debug images with automatic normalization.
                 Returns the input array to allow inline use in processing chains.
    """
    if not settings.get("DEBUG_MODE", False):
        # Return a no-op function if debug mode is disabled for efficiency.
        return lambda tag, arr: arr

    # Create debug directory within output directory.
    debug_dir = os.path.join(settings["OUTPUT_DIR"], "debug")
    os.makedirs(debug_dir, exist_ok=True)

    def snap(tag, arr):
        """
        Save an intermediate processing result as a debug image.

        Automatically handles normalization for different data types and adds
        timestamps to prevent overwriting previous debug images.

        Args:
            tag: String identifier for the image (used in filename).
            arr: Numpy array containing the image data to save.

        Returns:
            arr: The original input array (allows inline use in processing chains).
        """
        # Normalize array to 8-bit for visualization.
        if arr.dtype != np.uint8:
            v = arr.astype(np.float32)
            v = 255 * (v - v.min()) / (v.ptp() + 1e-6)
            v = v.astype(np.uint8)
        else:
            v = arr

        # Save with timestamp to avoid overwriting previous debug images.
        timestamp = datetime.now().strftime("%H%M%S")
        imageio.imwrite(os.path.join(debug_dir, f"{tag}_{timestamp}.png"), v)
        return arr  # Return the original array for inline use.

    return snap

"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: logging_utils.py.
Description:
    Logging and debug utilities for the nuclei segmentation pipeline.
    Provides dual-output logging (file and console) with Rich formatting,
    debug snapshot utilities, and error reporting functions.

Dependencies:
    - Python >= 3.10.
    - numpy, imageio, rich.

Usage:
    from utils.logging_utils import setup_logging
    logger = setup_logging(output_dir, debug_mode=False)
"""

# Standard library imports.
import os
import sys
import logging
import traceback
from datetime import datetime
import numpy as np
import imageio

from rich.console import Console
from rich.logging import RichHandler

# Initialize Rich console for formatted output.
console = Console()


def setup_logging(output_dir, debug_mode=False):
    """
    Configure comprehensive logging system for the segmentation pipeline.

    Creates a dual-output logging system that writes detailed logs to both a file
    and the console. The file logger captures all details including timestamps,
    while the console logger provides a more concise output for interactive use.
    Uses Rich library for beautiful formatted console output.

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

    # File handler for complete logging.
    fh = logging.FileHandler(log_file)
    fh.setFormatter(file_formatter)
    fh.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.addHandler(fh)

    # Rich console handler for beautiful interactive feedback.
    # When debug_mode=False, only show WARNING and above on console.
    # When debug_mode=True, show INFO and above on console.
    # DEBUG messages always only go to the log file.
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        show_level=False,  # Don't show log level prefix.
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=debug_mode
    )
    # Set console handler level based on debug_mode.
    rich_handler.setLevel(logging.INFO if debug_mode else logging.WARNING)
    logger.addHandler(rich_handler)

    # Print startup message with Rich formatting.
    console.print("\n[cyan bold]═════ Nuclei Segmentation Pipeline Started ═════[/cyan bold]\n")
    console.print(f"[blue]ℹ[/blue] Log file: [dim]{log_file}[/dim]")

    # Note: Config file backup is now handled in the load_config function.
    # We log the existence of the backup file if it exists.
    config_backup_path = os.path.join(output_dir, "nuclei_segmentation_config_used.ini")
    if os.path.exists(config_backup_path):
        console.print(f"[blue]ℹ[/blue] Configuration backed up at: [dim]{config_backup_path}[/dim]")
    else:
        if debug_mode:
            console.print("[yellow]⚠[/yellow] Configuration backup file not found")

    return logger


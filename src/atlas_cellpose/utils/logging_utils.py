"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: logging_utils.py.
Description:
    Logging factory for dual rich-console + file output.
    Provides a reusable setup_logging function that configures a
    timestamped file handler and a Rich console handler.

Dependencies:
    - Python >= 3.10.
    - rich.

Usage:
    from atlas_cellpose.utils.logging_utils import setup_logging
    logger, console = setup_logging(output_dir, debug_mode=False)
"""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(output_dir, debug_mode=False, console=None):
    """Configure logging with a rich console handler and a file handler.

    Creates a dual-output logging system that writes detailed logs to a
    timestamped file and provides formatted console output via Rich.

    Args:
        output_dir (str | Path): Directory where log files will be saved.
            A logs/ subdirectory is created inside it.
        debug_mode (bool): If True, the file handler captures DEBUG-level
            messages and the console shows INFO and above. If False, both
            handlers use INFO level.
        console (Console | None): Rich console instance to use for the
            console handler. A new one is created when None.

    Returns:
        tuple[logging.Logger, Console]: (logger, rich console).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file for uniqueness across runs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"segmentation_log_{timestamp}.txt"

    if console is None:
        console = Console()

    # Clear existing handlers to allow re-initialisation.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # File handler with timestamps for complete logging.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    root.addHandler(file_handler)

    # Rich console handler for formatted interactive output.
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=debug_mode,
    )
    rich_handler.setLevel(logging.INFO if debug_mode else logging.WARNING)
    root.addHandler(rich_handler)

    logger = logging.getLogger("atlas_cellpose")
    return logger, console


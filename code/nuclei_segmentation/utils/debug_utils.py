"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_utils.py.
Description:
    Debug snapshot management for pipeline introspection. Saves intermediate
    results as compressed .npz files for later inspection.

Dependencies:
    - Python >= 3.10.
    - numpy.

Usage:
    from utils.debug_utils import setup_debug
    snap = setup_debug(settings)
"""

from pathlib import Path
import numpy as np


class Snapshotter:
    """
    Captures intermediate results from the segmentation pipeline for debugging.

    Attributes:
        debug_dir (Path): Directory to store debug files.
    """

    def __init__(self, output_dir):
        self.debug_dir = Path(output_dir) / "debug_snapshots"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, name, data_dict):
        """
        Saves a dictionary of NumPy arrays as a `.npz` file for later inspection.

        Args:
            name (str): Label for the saved file.
            data_dict (dict): Dictionary where each value is a NumPy array.
        """
        filepath = self.debug_dir / f"{name}.npz"
        np.savez_compressed(filepath, **data_dict)


def setup_debug(settings):
    """
    Creates a Snapshotter instance if debugging is enabled.

    Args:
        settings (dict): settings dictionary that may include 'debug_mode' and 'output_dir'.

    Returns:
        Snapshotter or None: Debug snapshotter if enabled, else None.
    """
    if settings.get("debug_mode", False):
        return Snapshotter(settings["output_dir"])
    return None

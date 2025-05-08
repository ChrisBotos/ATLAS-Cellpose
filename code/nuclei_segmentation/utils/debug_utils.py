"""DEBUG UTILITIES — Snapshot management for pipeline introspection."""

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
        settings (dict): Settings dictionary that may include 'DEBUG_MODE' and 'OUTPUT_DIR'.

    Returns:
        Snapshotter or None: Debug snapshotter if enabled, else None.
    """
    if settings.get("DEBUG_MODE", False):
        return Snapshotter(settings["OUTPUT_DIR"])
    return None

"""TEST: Entry point runner script functionality."""

import sys
import os
import pytest
import tempfile
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code/nuclei_segmentation')))
from runner import main


def test_main_runs_without_crashing(tmp_path):
    """
    Smoke test: Ensure `main()` runs to completion with patched internals.

    This test does not validate actual segmentation results, only
    that the entrypoint wiring is correct and does not raise exceptions.
    """
    dummy_output_dir = tmp_path / "output"
    dummy_output_dir.mkdir()

    # Patch config and pipeline logic to simulate successful run.
    with patch("runner.load_config") as mock_config, \
         patch("runner.setup_logging") as mock_logger, \
         patch("runner.setup_debug") as mock_debug, \
         patch("runner.run_segmentation_pipeline") as mock_pipeline:

        mock_config.return_value = (
            {"OUTPUT_DIR": str(dummy_output_dir)},  # SETTINGS
            {"model_type": "nuclei", "gpu": False},  # CELLPOSE_PARAMS
            {"results": str(dummy_output_dir)}       # PROJECT_DIRS
        )
        mock_pipeline.return_value = 0  # Simulate success.

        exit_code = main()

    assert exit_code == 0, "Pipeline should return success status 0."

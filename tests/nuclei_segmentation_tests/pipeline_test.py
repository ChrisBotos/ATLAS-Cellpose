#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: pipeline_test.py.
Description:
    Test suite for the nuclei segmentation pipeline with mocked Cellpose and real I/O.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest, scikit-image.
    • Nuclei segmentation pipeline module.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/pipeline_test.py -v

Inputs:
    • None (tests generate temporary test data).

Outputs:
    • Test results indicating pass/fail status.

Key Features:
    • Integration tests for the full segmentation pipeline.
    • Tests with mocked Cellpose model to avoid GPU dependencies.
    • Verification of pipeline outputs and error handling.

Notes:
    • These tests verify the integration of various components in the nuclei segmentation pipeline.
    • Uses temporary directories for test data to avoid affecting the real file system.
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path
from skimage.io import imsave
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code/nuclei_segmentation')))
from pipeline import run_segmentation_pipeline


@pytest.fixture
def minimal_settings(tmp_path):
    """Returns minimal settings dict with real file paths and output directory."""
    image_path = tmp_path / "dummy_image.tif"
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    imsave(image_path, image)

    return {
        "image_path": str(image_path),
        "output_dir": str(tmp_path / "out"),
        "debug_mode": True,
        "generate_overlay": False,
        "use_edge_detection": False,
        "apply_watershed": False
    }


@pytest.fixture
def dummy_params():
    return {"model_type": "nuclei", "gpu": False}


@pytest.fixture
def dummy_dirs(tmp_path):
    return {"results": str(tmp_path / "results")}


def test_pipeline_generates_outputs(minimal_settings, dummy_params, dummy_dirs):
    """
    Tests that the pipeline completes, saves expected files,
    and correctly calls snapshot.
    """
    dummy_mask = np.ones((128, 128), dtype=np.uint8)
    dummy_flows = [np.zeros((2, 128, 128)), np.zeros((2, 128, 128)), np.zeros((128, 128))]
    logger = MagicMock()
    snap = MagicMock()

    with patch("pipeline.run_cellpose_on_tiles", return_value=(dummy_mask, dummy_flows, 494)), \
         patch("pipeline.setup_model", autospec=True):

        result = run_segmentation_pipeline(minimal_settings, dummy_params, dummy_dirs, logger, snap)

    assert result == 0
    output = Path(minimal_settings["output_dir"])
    assert (output / "masks" / "segmentation_masks.npy").exists()
    assert (output / "masks" / "segmentation_masks.tif").exists()
    assert (output / "flows" / "flows.npz").exists()
    snap.capture.assert_called_once_with("end_of_pipeline", {"masks": dummy_mask})

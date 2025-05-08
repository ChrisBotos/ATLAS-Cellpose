"""TEST: Full pipeline integration (mocked Cellpose, real I/O)."""

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
    """Returns minimal SETTINGS dict with real file paths and output directory."""
    image_path = tmp_path / "dummy_image.tif"
    image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    imsave(image_path, image)

    return {
        "IMAGE_PATH": str(image_path),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "DEBUG_MODE": True,
        "GENERATE_OVERLAY": False,
        "USE_EDGE_DETECTION": False,
        "APPLY_WATERSHED": False
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
    output = Path(minimal_settings["OUTPUT_DIR"])
    assert (output / "masks" / "segmentation_masks.npy").exists()
    assert (output / "masks" / "segmentation_masks.tif").exists()
    assert (output / "flows" / "flows.npz").exists()
    snap.capture.assert_called_once_with("end_of_pipeline", {"masks": dummy_mask})

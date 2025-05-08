"""TEST: Pipeline integration logic (segmentation, overlays, logging)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pipeline import run_segmentation_pipeline


@pytest.fixture
def dummy_settings(tmp_path):
    return {
        "IMAGE_PATH": str(tmp_path / "dummy_image.tif"),
        "OUTPUT_SUBDIR": "test_run",
        "DEBUG_MODE": True,
        "GENERATE_OVERLAY": False,
        "USE_EDGE_DETECTION": False,
        "APPLY_WATERSHED": False
    }


@pytest.fixture
def dummy_params():
    return {
        "model_type": "nuclei",
        "gpu": False
    }


@pytest.fixture
def dummy_dirs(tmp_path):
    return {"results": str(tmp_path / "results")}


def test_pipeline_runs_and_saves_masks(tmp_path, dummy_settings, dummy_params, dummy_dirs):
    """
    Validates that the pipeline completes successfully and saves expected files.
    """
    from skimage.io import imsave

    dummy_image = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    dummy_mask = np.ones((128, 128), dtype=np.uint8)
    dummy_flows = [np.zeros((2, 128, 128)), np.zeros((2, 128, 128)), np.zeros((128, 128))]
    dummy_settings["IMAGE_PATH"] = str(tmp_path / "dummy_image.tif")
    imsave(dummy_settings["IMAGE_PATH"], dummy_image)

    logger = MagicMock()
    snap = MagicMock()

    with patch("pipeline.preprocess_image", return_value=dummy_image), \
         patch("pipeline.run_cellpose_on_tiles", return_value=(dummy_mask, dummy_flows, 123)), \
         patch("pipeline.setup_model", autospec=True), \
         patch("pipeline.save_outputs", autospec=True), \
         patch("pipeline.apply_postprocessing", return_value=dummy_mask), \
         patch("pipeline.generate_overlays"):

        result = run_segmentation_pipeline(dummy_settings, dummy_params, dummy_dirs, logger, snap)

    assert result == 0, "Pipeline should complete without error."
    snap.capture.assert_called_once()

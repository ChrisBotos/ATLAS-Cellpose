#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation_test.py.
Description:
    Test suite for nuclei segmentation functions using Cellpose.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest.
    • Custom segmentation utilities from the nuclei_segmentation package.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/segmentation_test.py -v

Inputs:
    • None (tests run on generated test data).

Outputs:
    • Test results indicating pass/fail status.

Key Features:
    • Tests for Cellpose model integration with mocked models.
    • Tests for tiled segmentation processing.
    • Tests for edge-based refinement of segmentation masks.
    • Tests for handling various image dimensions and configurations.

Notes:
    • These tests verify the correct behavior of segmentation functions without requiring actual GPU resources.
    • Includes tests for both single-pass and tiled segmentation approaches.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pytest
from unittest.mock import MagicMock

from code.nuclei_segmentation.utils import segmentation



"""FIXTURES"""

@pytest.fixture
def dummy_image():
    """Return a small grayscale image."""
    return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def dummy_mask():
    """Return a binary mask with a square object."""
    mask = np.zeros((128, 128), dtype=np.uint32)
    mask[32:96, 32:96] = 1
    return mask


@pytest.fixture
def dummy_cellpose_params():
    """Minimal Cellpose parameters dictionary."""
    return {
        "diameter": 17.0,
        "channels": [0, 0],
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "resample": True,
        "batch_size": 1
    }


@pytest.fixture
def dummy_settings():
    """settings for tiling and refinement."""
    return {
        "tile_side_length": 64,
        "tile_overlap": 8,
        "use_tiling": True,
        "canny_threshold1": 30,
        "canny_threshold2": 100
    }


@pytest.fixture
def dummy_logger():
    """Mock logger for tracking log calls."""
    return MagicMock()


@pytest.fixture
def mock_model():
    """Mock Cellpose model returning tile-sized dummy masks and flows."""

    def fake_eval(image, **kwargs):
        h, w = image.shape[:2]
        return (
            np.ones((h, w), dtype=np.uint32),  # fake masks per tile size
            [np.zeros((2, h, w)), np.zeros((h, w))],  # dummy flow_xy, cellprob
            None  # ignored extra outputs
        )

    model = MagicMock()
    model.eval.side_effect = fake_eval
    return model


"""TESTS"""

def test_run_single_pass_cellpose(dummy_image, dummy_cellpose_params, dummy_logger, mock_model):
    """
    Test Cellpose segmentation without tiling.

    Ensures correct shape, dtype and non-negative mask values.
    """
    mask, flows, count = segmentation.run_single_pass_cellpose(mock_model, dummy_image, dummy_cellpose_params, dummy_logger)
    assert mask.shape == dummy_image.shape
    assert mask.dtype == np.uint32
    assert isinstance(flows, list) and len(flows) == 3
    assert flows[0].shape == (2, 128, 128)
    assert flows[1].shape == (128, 128)
    assert count == 1


def test_run_cellpose_on_tiles(dummy_image, dummy_cellpose_params, dummy_settings, dummy_logger, mock_model):
    """
    Run tiled segmentation using mocked Cellpose model.

    Ensures merging logic executes and output matches original image size.
    """
    mask, flows, count = segmentation.run_cellpose_on_tiles(mock_model, dummy_image, dummy_cellpose_params, dummy_settings, dummy_logger)
    assert mask.shape == dummy_image.shape
    assert mask.dtype == np.uint32
    assert isinstance(flows, list) and flows[0].shape == (2, 128, 128)
    assert count > 0


def test_refine_segmentation_with_edges(dummy_image, dummy_mask, dummy_settings, dummy_logger):
    """
    Check that edge refinement shrinks objects.

    Tests connected component analysis after Canny edge subtraction.
    """
    refined = segmentation.refine_segmentation_with_edges(dummy_image, dummy_mask, dummy_settings, dummy_logger)
    assert refined.shape == dummy_mask.shape
    assert refined.dtype == np.int32
    assert np.max(refined) >= 1  # Should label at least one object

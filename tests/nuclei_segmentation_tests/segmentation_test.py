#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation_test.py.
Description:
    Unit tests for the Cellpose-wrapping helpers in nuclei_segmentation.utils.segmentation.

Dependencies:
    • pytest, numpy.
    • The mocked Cellpose model is created with unittest.mock and requires no GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from unittest.mock import MagicMock

# ----------------------------------------------------------------------
# Make the package importable when tests are called via `pytest -q`.
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]          # Project root.
sys.path.insert(0, str(ROOT))

from code.nuclei_segmentation.utils import segmentation


# ----------------------------------------------------------------------
# Fixtures.
# ----------------------------------------------------------------------

@pytest.fixture
def dummy_image() -> np.ndarray:
    """Return a small grayscale image."""
    return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def dummy_cellpose_params() -> dict:
    """Minimal Cellpose parameters."""
    return {
        "diameter": 17.0,
        "channels": [0, 0],
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "resample": True,
        "batch_size": 1,
    }


@pytest.fixture
def dummy_settings() -> dict:
    """Settings for tiling."""
    return {
        "tile_side_length": 64,     # forces tiling on the 128×128 dummy image.
        "tile_overlap": 8,          # Pixels.
        "use_tiling": True,
    }


@pytest.fixture
def dummy_logger():
    """Return a MagicMock that mimics a logger."""
    return MagicMock()


@pytest.fixture
def mock_model():
    """Return a fake Cellpose model whose eval() always succeeds."""

    def fake_eval(image, **kwargs) -> Tuple[np.ndarray, list, None]:
        h, w = image.shape[:2]
        mask  = np.ones((h, w), dtype=np.uint32)          # single object.
        flows = [np.zeros((2, h, w)), np.zeros((h, w))]   # dummy XY + prob.
        return mask, flows, None

    m = MagicMock()
    m.eval.side_effect = fake_eval
    return m


# ----------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------

def test_single_pass(dummy_image, dummy_cellpose_params, dummy_logger, mock_model):
    """Full-image Cellpose call should return correctly-typed outputs."""
    mask, flows, count = segmentation._run_single_pass_cellpose(
        mock_model, dummy_image, dummy_cellpose_params, dummy_logger
    )

    assert mask.shape == dummy_image.shape
    assert mask.dtype == np.uint32
    assert count == 1

    assert isinstance(flows, list) and len(flows) == 3
    assert flows[0].shape == (2, 128, 128)          # flow-XY.
    assert flows[1].shape == (128, 128)             # cell-prob.


def test_tiled_segmentation(dummy_image, dummy_cellpose_params,
                            dummy_settings, dummy_logger, mock_model):
    """run_cellpose_on_tiles should tile, merge, and return global results."""
    mask, flows, count = segmentation.run_cellpose_on_tiles(
        mock_model,
        dummy_image,
        dummy_cellpose_params,
        dummy_settings,
        dummy_logger
    )

    assert mask.shape == dummy_image.shape
    assert mask.dtype == np.uint32

    # ── Expect exactly one mock nucleus per generated tile. ──
    tile_side   = dummy_settings["tile_side_length"]
    overlap     = dummy_settings["tile_overlap"]
    stride      = tile_side - overlap
    tiles_x     = math.ceil(dummy_image.shape[1] / stride)
    tiles_y     = math.ceil(dummy_image.shape[0] / stride)
    expected    = tiles_x * tiles_y
    assert count == expected


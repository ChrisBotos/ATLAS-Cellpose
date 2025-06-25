#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: preprocessing_test.py.
Description:
    Test suite for image preprocessing utilities used in nuclei segmentation.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest, scikit-image.
    • Custom preprocessing utilities from the nuclei_segmentation package.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/preprocessing_test.py -v

Inputs:
    • None (tests run on generated test data).

Outputs:
    • Test results indicating pass/fail status.

Key Features:
    • Tests for bit-depth conversion and normalization.
    • Tests for gamma correction and contrast enhancement (CLAHE).
    • Tests for image cropping and border handling.
    • Tests for the full preprocessing pipeline used in kidney I/R spatial omics data.

Notes:
    • These tests verify the correct behavior of image preprocessing functions essential for accurate nuclei segmentation.
    • Covers both individual functions and their integration in the preprocessing pipeline.
"""

'''Import statements and test framework.'''
import sys
import os

import numpy as np
import pytest
from skimage.io import imread
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code/nuclei_segmentation/utils')))
import preprocessing


"""FIXTURES"""

@pytest.fixture
def dummy_16bit_image():
    """
    Generates a synthetic 16-bit grayscale image.

    Returns:
        np.ndarray: Simulated 16-bit image with gradient intensities.
    """
    image = np.linspace(1000, 60000, num=128*128, dtype=np.uint32).reshape((128, 128))
    return image


@pytest.fixture
def dummy_8bit_image():
    """
    Generates a synthetic 8-bit grayscale image.

    Returns:
        np.ndarray: Randomized uint8 image.
    """
    return np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def dummy_logger():
    """
    Creates a mock logger to verify log calls.

    Returns:
        MagicMock: Mocked logger.
    """
    return MagicMock()


"""TESTS"""

def test_convert_16bit_to_8bit(dummy_16bit_image):
    """
    Validates correct percentile-based rescaling of 16-bit to 8-bit.

    Args:
        dummy_16bit_image (np.ndarray): High-dynamic-range simulated image.
    """
    out = preprocessing.convert_16bit_to_8bit(dummy_16bit_image)
    assert out.dtype == np.uint8
    assert out.shape == dummy_16bit_image.shape
    assert out.max() <= 255 and out.min() >= 0


def test_adaptive_gamma_correction_brightness(dummy_8bit_image, dummy_logger):
    """
    Tests that gamma correction lifts low-intensity images.

    Args:
        dummy_8bit_image (np.ndarray): Simulated dim image.
        dummy_logger (MagicMock): Logger to track info calls.
    """
    dark = (dummy_8bit_image // 4).astype(np.uint8)
    out = preprocessing.adaptive_gamma_correction(dark, logger=dummy_logger)
    assert out.shape == dark.shape
    assert out.dtype == np.uint8
    assert out.mean() > dark.mean()
    dummy_logger.info.assert_called()


def test_apply_clahe(dummy_8bit_image, dummy_logger):
    """
    Applies CLAHE to a low-contrast image and checks for enhanced variance.

    Args:
        dummy_8bit_image (np.ndarray): Input image.
        dummy_logger (MagicMock): Logger.
    """
    # Create a nearly flat image with mild variation.
    low_contrast = np.random.normal(loc=127, scale=2, size=(128, 128)).clip(0, 255).astype(np.uint8)
    result = preprocessing.apply_clahe(low_contrast, logger=dummy_logger)
    assert result.shape == low_contrast.shape
    assert result.dtype == np.uint8
    assert result.std() > low_contrast.std()  # CLAHE should increase local contrast.
    dummy_logger.info.assert_called()



def test_crop_image_relative_and_absolute(dummy_8bit_image, dummy_logger):
    """
    Verifies both absolute and relative cropping modes.

    Args:
        dummy_8bit_image (np.ndarray): Input image.
        dummy_logger (MagicMock): Logger.
    """
    abs_cropped = preprocessing.crop_image(dummy_8bit_image, (10, 100, 10, 100), logger=dummy_logger)
    rel_cropped = preprocessing.crop_image(dummy_8bit_image, (0.1, 0.8, 0.1, 0.8), logger=dummy_logger)
    assert abs_cropped.shape == (90, 90)
    assert rel_cropped.shape[0] < dummy_8bit_image.shape[0]
    assert rel_cropped.shape[1] < dummy_8bit_image.shape[1]


def test_invalid_crop_raises(dummy_8bit_image):
    """
    Ensures that invalid cropping ranges raise an error.

    Args:
        dummy_8bit_image (np.ndarray): Image to crop.
    """
    with pytest.raises(ValueError):
        preprocessing.crop_image(dummy_8bit_image, (90, 10, 90, 10))


def test_save_image_creates_file(dummy_8bit_image, tmp_path):
    """
    Tests that an image is saved to disk and matches the original content.

    Args:
        dummy_8bit_image (np.ndarray): Image to save.
        tmp_path (Path): Temporary directory.
    """
    out_path = tmp_path / "test_image.tif"
    preprocessing.save_image(dummy_8bit_image, str(out_path))
    reloaded = imread(out_path)
    assert np.array_equal(reloaded, dummy_8bit_image)


def test_preprocess_image_pipeline(tmp_path, dummy_16bit_image, dummy_logger):
    """
    Full pipeline test with all preprocessing steps enabled.

    Args:
        tmp_path (Path): Temporary directory for file IO.
        dummy_16bit_image (np.ndarray): Input 16-bit image.
        dummy_logger (MagicMock): Logger for step tracking.
    """
    from skimage.io import imsave
    image_path = tmp_path / "input_image.tif"
    imsave(image_path, dummy_16bit_image)

    settings = {
        "output_dir": str(tmp_path),
        "enhance_contrast": True,
        "enhance_dim": True,
        "crop_image": True,
        "crop_box": (0.1, 0.9, 0.1, 0.9),
        "upscale_factor": 2
    }

    result = preprocessing.preprocess_image(str(image_path), settings, dummy_logger)
    assert result.ndim == 2
    assert result.shape[0] > dummy_16bit_image.shape[0]
    assert result.dtype == np.uint8
    assert (tmp_path / "preprocessed" / "upscaled.tif").exists()

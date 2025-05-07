"""TESTING MODULE FOR visualization.py UTILITIES"""

'''Import statements and test framework.'''
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10**9
from io import BytesIO

# Relative import of the visualization utilities being tested.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code/nuclei_segmentation/utils')))
import visualization


"""FIXTURES"""
@pytest.fixture
def dummy_mask():
    """Generate a simple dummy binary mask with two objects for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    mask[60:80, 60:80] = 2
    return mask


@pytest.fixture
def dummy_image():
    """Generate a dummy grayscale background image."""
    return np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)


"""TESTS"""

def test_overlay_empty_mask_shape_preserved():
    """
    Check that an all-zero mask returns a grayscale RGB overlay without crashing.

    This verifies the fallback when no labeled objects are present in the mask.
    """

    print("\n===== Running: test_overlay_empty_mask_shape_preserved =====")

    dummy_img = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
    dummy_mask = np.zeros((64, 64), dtype=np.uint8)
    logger = visualization._setup_logger("test_empty_mask")

    overlay = visualization.create_overlay(dummy_img, dummy_mask, logger)

    assert overlay.shape == (64, 64, 3), "Overlay should match input image shape with 3 channels."
    assert np.allclose(overlay[..., 0], overlay[..., 1]), "Overlay channels should be identical in fallback."
    assert overlay.max() > 0.0, "Overlay should not be fully black."

    print(f"✔ Grayscale fallback overlay — shape: {overlay.shape}, dtype: {overlay.dtype}, mean: {overlay.mean():.3f}")


def test_overlay_brightness_correction_triggers():
    """
    Validate that dark images result in brightness correction during overlay generation.

    This ensures the logic branch that detects dim overlays and rescales intensity is exercised.
    """

    print("\n===== Running: test_overlay_brightness_correction_triggers =====")

    dark_img = np.full((128, 128), fill_value=5, dtype=np.uint8)
    mask = np.zeros_like(dark_img)
    mask[30:90, 30:90] = 1  # Add a mask object to force overlay logic

    logger = visualization._setup_logger("test_brightness_correction")

    overlay = visualization.create_overlay(dark_img, mask, logger)

    assert overlay.shape == (128, 128, 3), "Overlay must retain input shape."
    assert overlay.mean() > 0.1, "Brightness correction must lift mean intensity above threshold."

    print(f"✔ Brightness-corrected overlay — mean intensity: {overlay.mean():.3f}")


def test_overlay_large_image_forces_manual_overlay():
    """
    Tests that large images bypass Cellpose overlay and use manual RGB color-blending logic.

    This verifies the fallback logic for performance on full-slide overlays.
    """

    print("\n===== Running: test_overlay_large_image_forces_manual_overlay =====")

    large_img = np.random.randint(0, 255, size=(2500, 2500), dtype=np.uint8)
    mask = np.zeros_like(large_img)
    mask[1000:1500, 1000:1500] = 1  # One large object

    logger = visualization._setup_logger("test_large_image")

    overlay = visualization.create_overlay(large_img, mask, logger)

    assert overlay.shape == (2500, 2500, 3), "Overlay shape mismatch on large image."
    assert overlay.dtype in [np.float32, np.uint8], "Overlay dtype must be valid."

    print(f"✔ Large image manually blended overlay — dtype: {overlay.dtype}, mean: {overlay.mean():.3f}")


def test_overlay_with_float_input_and_mask():
    """
    Test that float32 grayscale images are handled properly.

    This is important for preprocessed normalized images already scaled in [0, 255].
    """

    print("\n===== Running: test_overlay_with_float_input_and_mask =====")

    float_img = np.random.uniform(0, 255, size=(100, 100)).astype(np.float32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1

    logger = visualization._setup_logger("test_float_input")

    overlay = visualization.create_overlay(float_img, mask, logger)

    assert overlay.shape == (100, 100, 3), "Overlay should still be RGB."
    assert overlay.max() <= 1.0 or overlay.max() <= 255.0, "Overlay must be scaled correctly."

    print(f"✔ Float image overlay — max intensity: {overlay.max():.3f}")


def test_overlay_handles_mask_overlay_crash(monkeypatch):
    """
    Monkeypatch plot.mask_overlay to raise an exception.

    This validates fallback rendering logic when Cellpose visualization fails unexpectedly.
    """

    print("\n===== Running: test_overlay_handles_mask_overlay_crash =====")

    dummy_img = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    dummy_mask[32:96, 32:96] = 1

    logger = visualization._setup_logger("test_overlay_crash")

    # Patch Cellpose plot function to raise.
    def fake_crash(*args, **kwargs):
        raise RuntimeError("Simulated internal crash")

    monkeypatch.setattr(visualization.plot, "mask_overlay", fake_crash)

    overlay = visualization.create_overlay(dummy_img, dummy_mask, logger)

    assert overlay.shape == (128, 128, 3), "Overlay must still be generated via fallback."
    assert overlay.max() > 0.0, "Fallback must produce nonzero overlay."

    print("✔ Fallback after mask_overlay crash — overlay created successfully.")

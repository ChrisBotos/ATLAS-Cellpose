"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_filter_masks.py.
Description:
    Unit tests for the filter_masks module: Thresholds validation,
    apply_thresholds logic, and filtering of known artifacts.

Dependencies:
    - Python >= 3.10.
    - pytest, numpy, pandas.

Usage:
    pytest tests/test_filter_masks.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the source tree is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "nuclei_segmentation"))

from utils.filter_masks import Thresholds, apply_thresholds, compute_metrics


# ====================================================================== #
# Thresholds dataclass.
# ====================================================================== #

class TestThresholds:
    """Test Thresholds dataclass validation logic."""

    def test_default_thresholds_are_valid(self):
        th = Thresholds()
        th.validate()  # Should not raise.

    def test_invalid_pixel_range_raises(self):
        th = Thresholds(min_pixels=500, max_pixels=100)
        with pytest.raises(AssertionError, match="pixel"):
            th.validate()

    def test_invalid_circularity_raises(self):
        th = Thresholds(min_circularity=0.8, max_circularity=0.3)
        with pytest.raises(AssertionError, match="circularity"):
            th.validate()

    def test_invalid_solidity_raises(self):
        th = Thresholds(min_solidity=1.1)
        with pytest.raises(AssertionError, match="solidity"):
            th.validate()

    def test_invalid_eccentricity_raises(self):
        th = Thresholds(min_eccentricity=-0.1)
        with pytest.raises(AssertionError, match="eccentricity"):
            th.validate()

    def test_invalid_hole_fraction_raises(self):
        th = Thresholds(max_hole_fraction=1.5)
        with pytest.raises(AssertionError, match="hole"):
            th.validate()

    def test_no_straight_fraction_field(self):
        """Thresholds should not have a straight_fraction field (dead code removed)."""
        th = Thresholds()
        assert not hasattr(th, "min_straight_fraction")
        assert not hasattr(th, "max_straight_fraction")


# ====================================================================== #
# apply_thresholds with synthetic data.
# ====================================================================== #

class TestApplyThresholds:
    """Test apply_thresholds with a hand-crafted DataFrame."""

    @pytest.fixture
    def sample_df(self):
        """Create a small DataFrame mimicking compute_metrics output."""
        return pd.DataFrame({
            "label":        [1,    2,    3,    4   ],
            "area":         [50,   10,   200,  5000],
            "circularity":  [0.9,  0.2,  0.7,  0.5 ],
            "solidity":     [0.95, 0.5,  0.85, 0.7 ],
            "eccentricity": [0.3,  0.95, 0.5,  0.1 ],
            "aspect_ratio": [1.1,  6.0,  1.5,  1.2 ],
            "hole_fraction":[0.01, 0.0,  0.03, 0.2 ],
        })

    def test_permissive_thresholds_pass_all(self, sample_df):
        """Default (wide-open) thresholds pass everything."""
        th = Thresholds()
        passed = apply_thresholds(sample_df, th)
        assert passed.all()

    def test_size_filter_removes_small(self, sample_df):
        """min_pixels=20 should reject label 2 (area=10)."""
        th = Thresholds(min_pixels=20)
        passed = apply_thresholds(sample_df, th)
        assert not passed[1]  # Label 2.
        assert passed[0]      # Label 1.

    def test_size_filter_removes_large(self, sample_df):
        """max_pixels=3000 should reject label 4 (area=5000)."""
        th = Thresholds(max_pixels=3000)
        passed = apply_thresholds(sample_df, th)
        assert not passed[3]  # Label 4.

    def test_circularity_filter(self, sample_df):
        """min_circularity=0.4 should reject label 2 (circularity=0.2)."""
        th = Thresholds(min_circularity=0.4)
        passed = apply_thresholds(sample_df, th)
        assert not passed[1]

    def test_aspect_ratio_filter(self, sample_df):
        """max_aspect_ratio=4.0 should reject label 2 (aspect_ratio=6.0)."""
        th = Thresholds(max_aspect_ratio=4.0)
        passed = apply_thresholds(sample_df, th)
        assert not passed[1]

    def test_hole_fraction_filter(self, sample_df):
        """max_hole_fraction=0.05 should reject label 4 (hole_fraction=0.2)."""
        th = Thresholds(max_hole_fraction=0.05)
        passed = apply_thresholds(sample_df, th)
        assert not passed[3]

    def test_combined_filters(self, sample_df):
        """Only label 1 and 3 should survive strict combined filtering."""
        th = Thresholds(
            min_pixels=20,
            max_pixels=3000,
            min_circularity=0.4,
            max_aspect_ratio=4.0,
            max_hole_fraction=0.05,
            min_solidity=0.6,
        )
        passed = apply_thresholds(sample_df, th)
        surviving = sample_df.label[passed].tolist()
        assert 1 in surviving
        assert 3 in surviving
        assert 2 not in surviving
        assert 4 not in surviving

    def test_border_exclusion(self):
        """exclude_border=True should reject nuclei touching the image border."""
        df = pd.DataFrame({
            "label":        [1,    2   ],
            "area":         [100,  100 ],
            "circularity":  [0.9,  0.9 ],
            "solidity":     [0.9,  0.9 ],
            "eccentricity": [0.3,  0.3 ],
            "aspect_ratio": [1.0,  1.0 ],
            "hole_fraction":[0.0,  0.0 ],
            "border_touch": [False, True],
        })
        th = Thresholds(exclude_border=True)
        passed = apply_thresholds(df, th)
        assert passed[0]
        assert not passed[1]


# ====================================================================== #
# compute_metrics on a tiny synthetic label map.
# ====================================================================== #

class TestComputeMetrics:
    """Test compute_metrics on a known small label map."""

    def test_single_square_nucleus(self):
        """A 10x10 square should have predictable metrics."""
        label_map = np.zeros((30, 30), dtype=np.int32)
        label_map[10:20, 10:20] = 1  # 10x10 square nucleus.
        df = compute_metrics(label_map, None)
        assert len(df) == 1
        assert df.iloc[0]["area"] == 100
        # Discrete perimeter from regionprops makes circularity higher
        # than the continuous pi/4 ≈ 0.785 for a square.
        assert 0.6 < df.iloc[0]["circularity"] <= 1.0
        assert df.iloc[0]["solidity"] == pytest.approx(1.0)

    def test_no_nuclei_returns_empty(self):
        """An all-zero label map should return an empty DataFrame."""
        label_map = np.zeros((20, 20), dtype=np.int32)
        df = compute_metrics(label_map, None)
        assert len(df) == 0

    def test_two_nuclei_counted(self):
        """Two separate objects produce two rows."""
        label_map = np.zeros((30, 30), dtype=np.int32)
        label_map[2:8, 2:8] = 1
        label_map[20:26, 20:26] = 2
        df = compute_metrics(label_map, None)
        assert len(df) == 2

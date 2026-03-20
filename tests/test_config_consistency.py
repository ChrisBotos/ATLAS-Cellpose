"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_config_consistency.py.
Description:
    Tests that Python fallback values in project_setup.py and filter_masks.py
    match the INI source of truth in nuclei_segmentation_config.ini.

Dependencies:
    - Python >= 3.10.
    - pytest.

Usage:
    pytest tests/test_config_consistency.py -v
"""

import configparser
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate the INI config.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INI_PATH = PROJECT_ROOT / "configs" / "nuclei_segmentation_config.ini"


@pytest.fixture(scope="module")
def ini_config():
    """Load the INI config once for the whole module."""
    assert INI_PATH.exists(), f"INI not found at {INI_PATH}"
    cfg = configparser.ConfigParser()
    cfg.read(INI_PATH)
    return cfg


# ====================================================================== #
# A1. project_setup.py fallback values vs. INI.
# ====================================================================== #

class TestProjectSetupFallbacks:
    """Verify that every fallback in project_setup.py matches the INI default."""

    def test_tile_overlap(self, ini_config):
        assert ini_config.getfloat("tiling", "tile_overlap") == 0.2

    def test_clahe_tile_grid_size(self, ini_config):
        raw = ini_config.get("clahe", "tile_grid_size")
        parsed = tuple(int(v.strip()) for v in raw.split(","))
        assert parsed == (32, 32)

    def test_use_cellpose4(self, ini_config):
        assert ini_config.getboolean("cellpose", "use_cellpose4") is False

    def test_max_pixels(self, ini_config):
        assert ini_config.getint("filtering", "max_pixels") == 3000

    def test_min_circularity(self, ini_config):
        assert ini_config.getfloat("filtering", "min_circularity") == 0.4

    def test_min_solidity(self, ini_config):
        assert ini_config.getfloat("filtering", "min_solidity") == 0.6

    def test_max_eccentricity(self, ini_config):
        assert ini_config.getfloat("filtering", "max_eccentricity") == 0.99

    def test_max_aspect_ratio(self, ini_config):
        assert ini_config.getfloat("filtering", "max_aspect_ratio") == 4.00

    def test_max_hole_fraction(self, ini_config):
        assert ini_config.getfloat("filtering", "max_hole_fraction") == 0.05


# ====================================================================== #
# A2. filter_masks.py programmatic-interface fallbacks vs. INI.
# ====================================================================== #

class TestFilterMasksFallbacks:
    """Verify filter_masks_programmatic fallback values match INI."""

    def test_max_pixels_fallback(self, ini_config):
        # filter_masks_programmatic uses settings.get("max_pixels", 3000).
        assert ini_config.getint("filtering", "max_pixels") == 3000

    def test_min_circularity_fallback(self, ini_config):
        assert ini_config.getfloat("filtering", "min_circularity") == 0.4

    def test_min_aspect_ratio_fallback(self, ini_config):
        assert ini_config.getfloat("filtering", "min_aspect_ratio") == 0.50

    def test_max_aspect_ratio_fallback(self, ini_config):
        assert ini_config.getfloat("filtering", "max_aspect_ratio") == 4.00

    def test_max_hole_fraction_fallback(self, ini_config):
        assert ini_config.getfloat("filtering", "max_hole_fraction") == 0.05


# ====================================================================== #
# A3. Config filename consistency.
# ====================================================================== #

class TestConfigFilename:
    """Verify the config backup filename matches across modules."""

    def test_logging_utils_filename_matches_project_setup(self):
        """The filename in logging_utils must match what project_setup saves."""
        import ast

        logging_utils_path = (
            PROJECT_ROOT / "src" / "atlas_cellpose" / "utils" / "logging_utils.py"
        )
        project_setup_path = (
            PROJECT_ROOT / "src" / "atlas_cellpose" / "utils" / "project_setup.py"
        )

        logging_src = logging_utils_path.read_text()
        setup_src = project_setup_path.read_text()

        # project_setup.py saves as "nuclei_segmentation_config_used.ini".
        assert "nuclei_segmentation_config_used.ini" in setup_src
        # logging_utils.py should reference the same name.
        assert "nuclei_segmentation_config_used.ini" in logging_src


# ====================================================================== #
# INI completeness: all expected sections and keys exist.
# ====================================================================== #

class TestINICompleteness:
    """Verify all expected sections and keys exist in the INI."""

    EXPECTED_SECTIONS = [
        "general", "clahe", "gamma_correction", "cellpose",
        "edge_detection", "tiling", "overlay", "debug",
        "filtering", "using_previous_results",
    ]

    def test_all_sections_present(self, ini_config):
        for section in self.EXPECTED_SECTIONS:
            assert ini_config.has_section(section), f"Missing INI section: [{section}]"

    @pytest.mark.parametrize("section,key", [
        ("filtering", "min_pixels"),
        ("filtering", "max_pixels"),
        ("filtering", "min_circularity"),
        ("filtering", "max_circularity"),
        ("filtering", "min_solidity"),
        ("filtering", "max_solidity"),
        ("filtering", "min_eccentricity"),
        ("filtering", "max_eccentricity"),
        ("filtering", "min_aspect_ratio"),
        ("filtering", "max_aspect_ratio"),
        ("filtering", "min_hole_fraction"),
        ("filtering", "max_hole_fraction"),
        ("filtering", "exclude_border"),
        ("tiling", "tile_overlap"),
        ("tiling", "tile_side_length"),
        ("cellpose", "use_cellpose4"),
        ("clahe", "tile_grid_size"),
    ])
    def test_key_exists(self, ini_config, section, key):
        assert ini_config.has_option(section, key), f"Missing key [{section}] {key}"

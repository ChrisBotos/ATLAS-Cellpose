#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: conftest.py.
Description:
    Configuration file for pytest with fixtures used across multiple test modules.

Dependencies:
    • Python >= 3.7.
    • pytest, matplotlib.

Usage:
    Not meant to be run directly. Used by pytest when running tests in this directory.

Key Features:
    • Provides common fixtures for all test modules.
    • Ensures proper cleanup of matplotlib resources after tests.

Notes:
    • This file is automatically loaded by pytest when running tests in this directory.
    • Contains fixtures that are shared across multiple test modules.
"""

import pytest
from matplotlib.testing.conftest import mpl_test_settings

# This fixture ensures that Matplotlib figures are properly cleaned up after each test.
# It is automatically applied to all tests in this directory and its subdirectories.
@pytest.fixture(autouse=True)
def matplotlib_cleanup(mpl_test_settings):
    pass

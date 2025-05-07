import pytest
from matplotlib.testing.conftest import mpl_test_settings

# This fixture ensures that Matplotlib figures are properly cleaned up after each test.
# It is automatically applied to all tests in this directory and its subdirectories.
@pytest.fixture(autouse=True)
def matplotlib_cleanup(mpl_test_settings):
    pass

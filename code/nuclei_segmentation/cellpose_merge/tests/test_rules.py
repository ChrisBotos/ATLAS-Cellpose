"""
Author: Christos Botos.
Description:
    Property tests for ``merge_patch_cpu`` using *hypothesis* with random small masks.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers

from cellpose_merge.rules import merge_patch_cpu


@given(
    arrays(np.uint32, (2, 32, 32), elements=integers(min_value=0, max_value=4)),
    integers(min_value=0, max_value=1).map(lambda x: x + 0.3),
)
@settings(max_examples=30)
def test_merge_idempotent(patch: np.ndarray, thr: float) -> None:
    merged, _ = merge_patch_cpu(patch, threshold=thr)
    merged2, _ = merge_patch_cpu(np.stack([merged, merged]), threshold=thr)
    assert np.array_equal(merged, merged2)

"""
Author: Christos Botos.
Description:
    Verify that ``merge_patch_gpu`` == ``merge_patch_cpu`` for randomly generated
    small patches. Skips if GPU is unavailable.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers

from cellpose_merge.gpu_merge import merge_patch_gpu
from cellpose_merge.rules import merge_patch_cpu


@given(
    arrays(np.uint32, (3, 64, 64), elements=integers(min_value=0, max_value=6)),
    integers(min_value=0, max_value=1).map(lambda x: x + 0.3),
)
@settings(max_examples=20)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")  # type: ignore
def test_cpu_gpu_parity(patch: np.ndarray, thr: float) -> None:
    cp, _ = merge_patch_cpu(patch, threshold=thr)
    gp, _ = merge_patch_gpu(patch, threshold=thr)
    assert np.array_equal(cp, gp)

"""Cellpose-Merge – tile mask merging for the ATLAS-Cellpose pipeline.

Re-exports the main merging functions for tile processing and overlap resolution.
GPU merge is planned but not yet implemented; all merging uses the CPU path.
"""
from .cpu_merge import merge_tiles_cpu_4step
from .two_phase_merge import _merge_two_tiles

# GPU merge is not yet implemented. Import only if explicitly needed.
# See gpu_merge.py for details.

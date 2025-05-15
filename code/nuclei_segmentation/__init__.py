"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Package Name: nuclei_segmentation
Description:
    Nuclei Segmentation Package for Kidney I/R Injury Analysis.

This package contains tools for segmenting and analyzing nuclei in kidney tissue sections
from ischemia-reperfusion injury experiments. It provides a complete workflow from image
preprocessing to feature extraction for downstream analysis of nuclear morphology changes
during kidney injury and repair processes.

Key Components:
    • runner.py: Main entry point for the segmentation pipeline.
    • pipeline.py: Core segmentation workflow implementation.
    • extract_engineered_features.py: Feature extraction from segmented nuclei.

Dependencies:
    • Python >= 3.7.
    • numpy, pandas, PIL, scipy, scikit-image.
    • cellpose for deep learning-based segmentation.
    • torch for GPU acceleration (optional).

Notes:
    • This package is designed for analyzing nuclear morphology changes during kidney I/R injury.
    • It supports both single-image processing and batch processing workflows.
"""

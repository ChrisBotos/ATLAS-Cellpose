"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Package Name: engineered_feature_extraction.
Description:
    Comprehensive nuclear feature extraction package for kidney ischemia-reperfusion injury analysis.
    Provides organized feature extraction and visualization tools with scientific context for
    quantitative analysis of nuclear morphology changes during tissue injury and repair processes.

This package contains refactored tools for extracting and visualizing nuclear morphological features
from segmented DAPI-stained tissue sections. Features are organized into four distinct categories:
shape, size, neighborhood, and texture features, each providing unique insights into cellular
responses during kidney I/R injury progression.

Key Components:
    • extract_engineered_features_refactored.py: Comprehensive feature extraction with category organization.
    • visualize_engineered_features_refactored.py: Publication-quality visualization with timepoint coding.
    • Original legacy modules: extract_engineered_features.py, visualize_engineered_features.py.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scipy, scikit-image, scikit-learn, matplotlib, seaborn, typer.
    • PIL for image handling and traceback for error reporting.
    • Custom utilities from nuclei_segmentation package.

Key Features:
    • Configurable feature categories: shape, size, neighborhood, and texture features.
    • Timepoint-specific color coding for injury progression analysis (10h, 2d, 14d timepoints).
    • Publication-quality violin plots with proper statistical representations.
    • Comprehensive statistical testing with multiple comparison corrections.
    • Scientific formatting optimized for kidney I/R injury research publications.
    • Parallel processing with memory-efficient batch processing for large tissue sections.

Notes:
    • Features are specifically selected for analyzing apoptosis, necrosis, and tissue repair mechanisms.
    • All visualizations include proper scientific formatting and colorblind-accessible palettes.
    • Configuration parameters allow selective feature extraction based on analysis requirements.
    • Comprehensive test suites ensure scientific accuracy and reproducibility of measurements.
"""

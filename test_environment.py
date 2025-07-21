#!/usr/bin/env python3
"""
Test script to verify the iri310 environment is working correctly.
"""

import sys

def test_imports():
    """Test importing all required packages."""
    try:
        import numpy
        print(f"✔ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import scipy
        print(f"✔ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False

    try:
        import pandas
        print(f"✔ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False

    try:
        import torch
        print(f"✔ PyTorch {torch.__version__} (CUDA: {torch.version.cuda})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import torchvision
        print(f"✔ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False

    try:
        import skimage
        print(f"✔ scikit-image {skimage.__version__}")
    except ImportError as e:
        print(f"✗ scikit-image import failed: {e}")
        return False

    try:
        import cellpose
        try:
            version = cellpose.__version__
        except AttributeError:
            version = "imported successfully"
        print(f"✔ Cellpose {version}")
    except ImportError as e:
        print(f"✗ Cellpose import failed: {e}")
        return False

    try:
        import scanpy
        print(f"✔ Scanpy {scanpy.__version__}")
    except ImportError as e:
        print(f"✗ Scanpy import failed: {e}")
        return False

    try:
        import anndata
        print(f"✔ AnnData {anndata.__version__}")
    except ImportError as e:
        print(f"✗ AnnData import failed: {e}")
        return False

    try:
        import transformers
        print(f"✔ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import pyarrow
        print(f"✔ PyArrow {pyarrow.__version__}")
    except ImportError as e:
        print(f"✗ PyArrow import failed: {e}")
        return False

    try:
        import fastparquet
        print(f"✔ FastParquet {fastparquet.__version__}")
    except ImportError as e:
        print(f"✗ FastParquet import failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}")
    print("Testing package imports...")
    print("-" * 40)
    
    success = test_imports()
    
    print("-" * 40)
    if success:
        print("🎉 All packages imported successfully!")
        print("Environment iri310 is ready to use!")
    else:
        print("❌ Some packages failed to import.")
        sys.exit(1)

#!/usr/bin/env python3
"""
Author: Christos Botos
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos

Script Name: test_fixed_environment.py
Description:
    Comprehensive test script to verify that all required packages for the nuclei
    segmentation pipeline are properly installed and functional.

Dependencies:
    • All packages listed in environment_fixed.yml
    • Properly configured conda environment

Usage:
    python test_fixed_environment.py

Key Features:
    • Tests all critical package imports
    • Verifies PyTorch CUDA functionality
    • Tests Cellpose model loading
    • Validates pipeline module imports
    • Provides detailed error reporting with solutions
"""

import sys
import traceback
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def test_import(module_name, description='', test_func=None):
    """Test importing a module and optionally run a test function."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {module_name} {version} - {description}")
        
        if test_func:
            test_func(module)
            
        return True
    except ImportError as e:
        print(f"❌ {module_name} - MISSING: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - ERROR: {e}")
        return False

def test_pytorch(torch_module):
    """Test PyTorch functionality."""
    print(f"   • CUDA available: {torch_module.cuda.is_available()}")
    if torch_module.cuda.is_available():
        print(f"   • CUDA devices: {torch_module.cuda.device_count()}")
        print(f"   • Current device: {torch_module.cuda.current_device()}")

def test_cellpose(cellpose_module):
    """Test Cellpose functionality."""
    try:
        from cellpose import models
        print("   • Cellpose models module imported successfully")
        
        # Test model creation (this will download models if needed)
        print("   • Testing model creation...")
        model = models.Cellpose(gpu=False, model_type='nuclei')
        print("   • Nuclei model created successfully")
        
    except Exception as e:
        print(f"   • Cellpose test failed: {e}")

def main():
    """Run comprehensive environment testing."""
    print_header("NUCLEI SEGMENTATION ENVIRONMENT TEST")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    
    # Test core scientific packages
    print_section("Core Scientific Packages")
    core_success = True
    core_packages = [
        ("numpy", "Scientific computing"),
        ("scipy", "Scientific algorithms"), 
        ("pandas", "Data manipulation"),
        ("matplotlib", "Plotting"),
        ("PIL", "Image processing"),
        ("skimage", "Image analysis"),
        ("sklearn", "Machine learning"),
        ("joblib", "Parallel processing"),
    ]
    
    for pkg, desc in core_packages:
        core_success &= test_import(pkg, desc)
    
    # Test image processing packages
    print_section("Image Processing Packages")
    image_success = True
    image_packages = [
        ("cv2", "Computer vision"),
        ("imageio", "Image I/O"),
        ("imagecodecs", "Image codecs"),
    ]
    
    for pkg, desc in image_packages:
        image_success &= test_import(pkg, desc)
    
    # Test deep learning packages
    print_section("Deep Learning Packages")
    dl_success = True
    dl_success &= test_import("torch", "PyTorch", test_pytorch)
    dl_success &= test_import("torchvision", "PyTorch vision")
    dl_success &= test_import("torchaudio", "PyTorch audio")
    
    # Test specialized packages
    print_section("Specialized Packages")
    spec_success = True
    spec_success &= test_import("cellpose", "Cell segmentation", test_cellpose)
    spec_success &= test_import("anndata", "Annotated data")
    spec_success &= test_import("scanpy", "Single-cell analysis")
    
    # Test utility packages
    print_section("Utility Packages")
    util_success = True
    util_packages = [
        ("rich", "Rich console output"),
        ("typer", "CLI framework"),
        ("tqdm", "Progress bars"),
        ("psutil", "System utilities"),
        ("pyarrow", "Arrow data format"),
        ("fastparquet", "Parquet files"),
    ]
    
    for pkg, desc in util_packages:
        util_success &= test_import(pkg, desc)
    
    # Test pipeline imports
    print_section("Pipeline Module Imports")
    pipeline_success = True
    try:
        sys.path.append('code/nuclei_segmentation')
        from utils.logging_utils import setup_logging
        from utils.project_setup import load_config
        from utils.debug_utils import setup_debug
        from pipeline import run_segmentation_pipeline
        print("✅ All pipeline modules imported successfully")
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        traceback.print_exc()
        pipeline_success = False
    
    # Summary
    print_section("Test Summary")
    all_success = core_success and image_success and dl_success and spec_success and util_success and pipeline_success
    
    if all_success:
        print("🎉 ALL TESTS PASSED! Environment is ready for nuclei segmentation.")
        print("\nTo run the pipeline:")
        print("  cd code/nuclei_segmentation")
        print("  python run_this.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  • Reinstall failed packages: pip install <package_name>")
        print("  • Check conda environment activation")
        print("  • Verify CUDA installation for GPU support")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

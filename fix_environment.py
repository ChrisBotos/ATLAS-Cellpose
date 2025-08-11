#!/usr/bin/env python3
"""
Environment Fix Script for Nuclei Segmentation with Cellpose.

Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Installs missing packages in the current conda environment.
    Handles package installation with proper error handling.

Usage:
    python fix_environment.py

Key Features:
    • Installs missing packages via pip.
    • Tests imports after installation.
    • Provides detailed feedback on success/failure.
"""

import subprocess
import sys
import importlib
import traceback

def print_status(message):
    print(f"✓ {message}")

def print_info(message):
    print(f"ℹ {message}")

def print_error(message):
    print(f"✗ {message}")

def print_header(message):
    print(f"\n=== {message} ===")

def install_package(package_name, version=None):
    """Install a package using pip."""
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
        
        print_info(f"Installing {package_spec}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_spec], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status(f"Successfully installed {package_spec}")
            return True
        else:
            print_error(f"Failed to install {package_spec}: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Error installing {package_spec}: {e}")
        return False

def test_import(module_name, description=''):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print_status(f"{module_name} - {description}")
        return True
    except ImportError as e:
        print_error(f"{module_name} - FAILED: {e}")
        return False

def main():
    """Main function to fix the environment."""
    print_header("Fixing Cellpose Environment")
    
    # List of required packages with versions
    required_packages = [
        ("imageio", "2.31.6"),
        ("tifffile", "2023.9.26"),
        ("opencv-python-headless", "4.8.1.78"),
        ("rich", "13.6.0"),
        ("typer", "0.9.0"),
        ("tqdm", "4.66.1"),
        ("psutil", "5.9.6"),
        ("cellpose", "3.0.10"),
        ("fastremap", "1.14.0"),
        ("roifile", "2023.8.12"),
        ("imagecodecs", "2023.9.18"),
        ("anndata", "0.10.3"),
        ("scanpy", "1.9.6"),
        ("pyarrow", "13.0.0"),
        ("fastparquet", "2023.10.1"),
    ]
    
    # Install missing packages
    print_info("Installing required packages...")
    success_count = 0
    
    for package, version in required_packages:
        if install_package(package, version):
            success_count += 1
    
    print_status(f"Installed {success_count}/{len(required_packages)} packages")
    
    # Test critical imports
    print_header("Testing Package Imports")
    
    test_packages = [
        ("numpy", "Scientific computing"),
        ("scipy", "Scientific algorithms"),
        ("pandas", "Data manipulation"),
        ("matplotlib", "Plotting"),
        ("PIL", "Image processing"),
        ("skimage", "Image analysis"),
        ("sklearn", "Machine learning"),
        ("cv2", "Computer vision"),
        ("imageio", "Image I/O"),
        ("rich", "Rich console output"),
        ("typer", "CLI framework"),
        ("tqdm", "Progress bars"),
        ("torch", "PyTorch"),
        ("torchvision", "PyTorch vision"),
        ("cellpose", "Cell segmentation"),
        ("anndata", "Annotated data"),
        ("scanpy", "Single-cell analysis"),
    ]
    
    print_info("Testing core packages...")
    all_success = True
    
    for module, description in test_packages:
        if not test_import(module, description):
            all_success = False
    
    # Test PyTorch CUDA
    print_header("Testing PyTorch CUDA Support")
    try:
        import torch
        print_info(f"PyTorch version: {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print_info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print_info("CUDA not available - will use CPU")
    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        all_success = False
    
    # Test pipeline imports
    print_header("Testing Pipeline Module Imports")
    try:
        sys.path.append('code/nuclei_segmentation')
        from utils.logging_utils import setup_logging
        from utils.project_setup import load_config
        from utils.debug_utils import setup_debug
        from pipeline import run_segmentation_pipeline
        print_status("All pipeline modules imported successfully")
    except Exception as e:
        print_error(f"Pipeline import failed: {e}")
        traceback.print_exc()
        all_success = False
    
    # Final result
    print_header("Environment Fix Complete")
    if all_success:
        print_status("🎉 All packages installed and tested successfully!")
        print_info("You can now run: python code/nuclei_segmentation/run_this.py")
    else:
        print_error("❌ Some packages failed to install or import")
        print_info("Please check the error messages above and install missing packages manually")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

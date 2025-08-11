#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_environment_setup.py.
Description:
    Comprehensive environment testing script for the I/R injury nuclei segmentation
    pipeline. Validates all dependencies, package versions, and functionality to
    ensure the environment is properly configured for server deployment.

Dependencies:
    • Python >= 3.10 (from venv310_cellpose3 environment).
    • All packages from cellpose3_environment.yml.

Usage:
    # After activating the conda environment:
    conda activate venv310_cellpose3
    python test_environment_setup.py

Arguments:
    None (runs comprehensive environment validation).

Outputs:
    • Detailed environment validation report.
    • Pass/fail status for each component.
    • Recommendations for fixing issues.
    • Environment summary for troubleshooting.

Key Features:
    • Python version and path validation.
    • Package import and version checking.
    • PyTorch CUDA compatibility testing.
    • Cellpose3 functionality verification.
    • Memory and system resource assessment.
    • Server deployment readiness check.

Notes:
    • Run this script after creating the conda environment.
    • Use the output to troubleshoot environment issues.
    • Safe to run multiple times for validation.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import traceback


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def test_python_environment():
    """Test Python version and environment setup."""
    print_section("Python Environment")
    
    # Python version.
    python_version = sys.version
    print(f"✓ Python version: {python_version}")
    
    # Python executable path.
    python_path = sys.executable
    print(f"✓ Python executable: {python_path}")
    
    # Check if we're in the right conda environment.
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"✓ Conda environment: {conda_env}")
    
    if "venv310_cellpose3" in python_path or conda_env == "venv310_cellpose3":
        print("✅ PASS: Running in correct conda environment")
        return True
    else:
        print("❌ FAIL: Not running in venv310_cellpose3 environment")
        print("   SOLUTION: conda activate venv310_cellpose3")
        return False


def test_system_info():
    """Test system information and resources."""
    print_section("System Information")
    
    # Platform info.
    print(f"✓ Platform: {platform.platform()}")
    print(f"✓ Architecture: {platform.architecture()}")
    print(f"✓ Processor: {platform.processor()}")
    
    # Memory info (if psutil available).
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✓ Total memory: {memory.total / (1024**3):.1f} GB")
        print(f"✓ Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"✓ CPU count: {psutil.cpu_count()}")
        return True
    except ImportError:
        print("⚠️  psutil not available (memory info unavailable)")
        return True


def test_core_packages():
    """Test core scientific computing packages."""
    print_section("Core Packages")
    
    packages = {
        "numpy": "Scientific computing",
        "scipy": "Scientific algorithms",
        "pandas": "Data manipulation",
        "matplotlib": "Plotting",
        "PIL": "Image processing",
        "skimage": "Image analysis",
        "tqdm": "Progress bars",
        "joblib": "Parallel processing"
    }
    
    results = {}
    for package, description in packages.items():
        try:
            if package == "PIL":
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package} {version} - {description}")
            results[package] = True
        except ImportError as e:
            print(f"❌ {package} - MISSING ({description})")
            results[package] = False
    
    success_rate = sum(results.values()) / len(results)
    print(f"\nCore packages: {success_rate:.1%} success rate")
    return success_rate > 0.8


def test_pytorch():
    """Test PyTorch installation and CUDA support."""
    print_section("PyTorch")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # CUDA availability.
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  Running in CPU-only mode (slower but functional)")
        
        # Test basic tensor operations.
        test_tensor = torch.randn(10, 10)
        result = torch.matmul(test_tensor, test_tensor.T)
        print("✅ Basic tensor operations working")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available")
        print("   SOLUTION: conda install pytorch torchvision -c pytorch")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


def test_cellpose():
    """Test Cellpose installation and functionality."""
    print_section("Cellpose")
    
    try:
        import cellpose

        # Get version (different methods for different Cellpose versions).
        try:
            version = cellpose.__version__
        except AttributeError:
            try:
                from cellpose import version_str
                version = version_str
            except ImportError:
                version = "unknown"

        print(f"✅ Cellpose version: {version}")

        # Test model creation.
        from cellpose import models
        
        # Test Cellpose3 API.
        try:
            model = models.Cellpose(model_type='nuclei', gpu=False)
            print("✅ Cellpose3 model creation successful")
            
            # Test basic segmentation on dummy data.
            import numpy as np
            dummy_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            masks, flows, styles, diams = model.eval(dummy_image, diameter=None, channels=[0,0])
            print("✅ Basic segmentation test successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Cellpose3 functionality test failed: {e}")
            return False
            
    except ImportError:
        print("❌ Cellpose not available")
        print("   SOLUTION: pip install cellpose==3.0.10")
        return False


def test_project_structure():
    """Test project directory structure."""
    print_section("Project Structure")
    
    required_paths = [
        "code/nuclei_segmentation/run_this.py",
        "configs/nuclei_segmentation_config.ini",
        "cellpose3_environment.yml",
        "README.md"
    ]
    
    all_present = True
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
            all_present = False
    
    if all_present:
        print("✅ All required project files present")
    else:
        print("❌ Some project files missing")
        print("   SOLUTION: Ensure you're in the project root directory")
    
    return all_present


def test_pipeline_imports():
    """Test pipeline-specific imports."""
    print_section("Pipeline Imports")
    
    try:
        # Add project root to path.
        project_root = Path.cwd()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test key pipeline imports.
        from code.nuclei_segmentation.utils.project_setup import load_config
        print("✅ Project configuration loading")
        
        # Test configuration loading.
        settings, cellpose_params, dirs = load_config()
        print("✅ Configuration file parsing")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        print("   SOLUTION: Ensure you're in the project root directory")
        return False


def main():
    """Run comprehensive environment testing."""
    print_header("I/R INJURY NUCLEI SEGMENTATION - ENVIRONMENT TEST")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests.
    tests = [
        ("Python Environment", test_python_environment),
        ("System Information", test_system_info),
        ("Core Packages", test_core_packages),
        ("PyTorch", test_pytorch),
        ("Cellpose", test_cellpose),
        ("Project Structure", test_project_structure),
        ("Pipeline Imports", test_pipeline_imports)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary.
    print_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ENVIRONMENT READY!")
        print("You can now run the nuclei segmentation pipeline:")
        print("  ./run_with_proper_env.sh")
        return 0
    else:
        print("\n⚠️  ENVIRONMENT ISSUES DETECTED")
        print("Please fix the failing tests before running the pipeline.")
        print("See the solutions provided above for each failing test.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

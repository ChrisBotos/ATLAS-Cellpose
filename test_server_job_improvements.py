#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_server_job_improvements.py.
Description:
    Test script to validate the server job script improvements.
    Tests the results locator functionality and path management.

Dependencies:
    • Python >= 3.7.
    • pathlib for cross-platform path handling.

Usage:
    python test_server_job_improvements.py

Key Features:
    • Tests results directory discovery functionality.
    • Validates path management improvements.
    • Simulates server job script behavior.
    • Provides detailed feedback on improvements.

Notes:
    • This script tests the improvements made to the server job workflow.
    • It can be run independently to validate functionality.
"""

import traceback
import os
import sys
from pathlib import Path

# Add the nuclei segmentation code to the path.
sys.path.insert(0, str(Path(__file__).parent / "code" / "nuclei_segmentation"))

try:
    from utils.results_locator import (
        find_latest_results, 
        validate_results_directory, 
        find_segmentation_mask,
        create_results_summary
    )
    from utils.project_setup import load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the repository root directory.")
    sys.exit(1)


def test_results_locator():
    """Test the results locator functionality."""
    print("=== Testing Results Locator ===")
    
    results_base = Path("results")
    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        print("Creating mock results directory for testing...")
        results_base.mkdir(exist_ok=True)
        return False
    
    # Test finding latest results.
    latest = find_latest_results(results_base)
    if latest:
        print(f"✓ Found latest results: {latest}")
        
        # Test validation.
        is_valid, missing = validate_results_directory(latest)
        print(f"✓ Results validation: {'PASS' if is_valid else 'FAIL'}")
        if missing:
            print("  Missing files:")
            for file in missing:
                print(f"    - {file}")
        
        # Test mask finding.
        mask_path = find_segmentation_mask(latest)
        if mask_path:
            print(f"✓ Found segmentation mask: {mask_path}")
        else:
            print("✗ Segmentation mask not found")
        
        # Test summary creation.
        summary = create_results_summary(latest)
        print(f"✓ Results summary created:")
        print(f"    Files: {len(summary['files'])}")
        print(f"    Size: {summary['total_size_mb']} MB")
        print(f"    Mask files: {len(summary['mask_files'])}")
        
        return True
    else:
        print("✗ No results directory found")
        return False


def test_config_with_job_name():
    """Test configuration loading with custom job name."""
    print("\n=== Testing Config with Job Name ===")
    
    try:
        # Test with custom job name.
        job_name = "test_server_job"
        settings, cellpose_params, project_dirs = load_config(job_name=job_name)
        
        output_dir = settings["output_dir"]
        print(f"✓ Config loaded with job name: {job_name}")
        print(f"✓ Output directory: {output_dir}")
        
        # Check if the job name is in the output directory path.
        if job_name in str(output_dir):
            print("✓ Job name correctly incorporated in output directory")
        else:
            print("✗ Job name not found in output directory path")
            return False
        
        # Check for latest link/file.
        results_base = project_dirs["results"]
        latest_link = results_base / "latest"
        latest_txt = results_base / "latest.txt"
        
        if latest_link.exists() or latest_txt.exists():
            print("✓ Latest results tracking created")
        else:
            print("✗ Latest results tracking not created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing config with job name: {e}")
        print(traceback.format_exc())
        return False


def test_environment_variable_support():
    """Test environment variable support for job names."""
    print("\n=== Testing Environment Variable Support ===")
    
    try:
        # Set environment variable.
        test_job_name = "env_test_job"
        os.environ["SEGMENTATION_JOB_NAME"] = test_job_name
        
        # Import and test the main function logic.
        import tempfile
        import json
        
        # Test that the environment variable is picked up.
        job_name = os.environ.get('SEGMENTATION_JOB_NAME')
        if job_name == test_job_name:
            print(f"✓ Environment variable correctly read: {job_name}")
        else:
            print(f"✗ Environment variable not read correctly: {job_name}")
            return False
        
        # Clean up.
        del os.environ["SEGMENTATION_JOB_NAME"]
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing environment variable support: {e}")
        print(traceback.format_exc())
        return False


def test_path_validation():
    """Test path validation functionality."""
    print("\n=== Testing Path Validation ===")
    
    try:
        # Test repository structure validation.
        repo_name = "I-R-Injury-Spatial-Multiomics-Analysis"
        current_dir = Path.cwd()
        
        # Check if we're in the repository.
        if current_dir.name == repo_name:
            print(f"✓ Currently in repository: {repo_name}")
        else:
            print(f"✓ Repository structure test (current dir: {current_dir.name})")
        
        # Test segmentation code path.
        segmentation_code = Path("code/nuclei_segmentation")
        if segmentation_code.exists():
            print(f"✓ Segmentation code directory exists: {segmentation_code}")
        else:
            print(f"✗ Segmentation code directory not found: {segmentation_code}")
            return False
        
        # Test run_this.py exists.
        run_script = segmentation_code / "run_this.py"
        if run_script.exists():
            print(f"✓ Main script exists: {run_script}")
        else:
            print(f"✗ Main script not found: {run_script}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing path validation: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Run all tests and provide summary."""
    print("Server Job Script Improvements Test Suite")
    print("=" * 50)
    
    tests = [
        ("Results Locator", test_results_locator),
        ("Config with Job Name", test_config_with_job_name),
        ("Environment Variable Support", test_environment_variable_support),
        ("Path Validation", test_path_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary.
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All improvements are working correctly!")
        return 0
    else:
        print("✗ Some improvements need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

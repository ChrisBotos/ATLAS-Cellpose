#!/usr/bin/env python3
"""
Test script to verify the load_config function works correctly.
"""

try:
    from code.nuclei_segmentation.runner import load_config
    
    print("Attempting to load configuration...")
    settings, cellpose_params, project_dirs = load_config()
    
    print("\nConfiguration loaded successfully!")
    print(f"Output directory: {settings['OUTPUT_DIR']}")
    print(f"Number of settings: {len(settings)}")
    print(f"Number of Cellpose parameters: {len(cellpose_params)}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

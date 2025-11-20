# Environment Fix Instructions

## Quick Fix (Recommended)

Run these commands in your WSL terminal:

```bash
# Navigate to project directory
cd /mnt/c/Projects/ATLAS-Cellpose

# Remove broken environment
conda env remove -n venv310_cellpose3 -y

# Create new environment from YAML file
conda env create -f environment_fixed.yml

# Activate the new environment
conda activate venv310_cellpose3_fixed

# Test the environment
python test_fixed_environment.py

# Test the actual pipeline
cd code/nuclei_segmentation
python run_this.py
```

## Alternative: Using the Bash Script

If you prefer the automated script:

```bash
# Navigate to project directory
cd /mnt/c/Projects/ATLAS-Cellpose

# Make script executable and run
chmod +x fix_environment.sh
./fix_environment.sh
```

## Manual Fix (If Above Fails)

If the automated methods fail, try this manual approach:

```bash
# Remove any existing environments
conda env remove -n venv310_cellpose3 -y
conda env remove -n venv310_cellpose3_fixed -y

# Create fresh environment
conda create -n venv310_cellpose3_fixed python=3.10 -y
conda activate venv310_cellpose3_fixed

# Install core packages via conda
conda install -c conda-forge numpy=1.26.4 scipy=1.11.4 pandas=2.1.4 matplotlib=3.8.2 pillow=10.1.0 scikit-image=0.22.0 scikit-learn=1.3.2 imageio=2.31.6 rich=13.6.0 tqdm=4.66.1 psutil=5.9.6 -y

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install specialized packages via pip
pip install cellpose==3.0.10 anndata==0.10.3 scanpy==1.9.6 typer==0.9.0 opencv-python-headless==4.8.1.78 imagecodecs==2023.9.18 pyarrow==13.0.0 fastparquet==2023.10.1

# Test the installation
python test_fixed_environment.py
```

## Troubleshooting

### If Cellpose fails to install:
```bash
pip install --no-deps cellpose==3.0.10
pip install fastremap roifile
```

### If PyTorch CUDA doesn't work:
```bash
# For CPU-only version
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### If imports still fail:
```bash
# Check what's missing
python -c "import sys; print(sys.path)"
python -c "import numpy, scipy, pandas, matplotlib, PIL, skimage, cv2, imageio, torch, cellpose; print('All imports successful')"
```

## Expected Output

After successful installation, you should see:
- ✅ All package imports successful
- ✅ PyTorch with CUDA support (if GPU available)
- ✅ Cellpose models loading correctly
- ✅ Pipeline modules importing without errors

## Environment Details

The fixed environment includes:
- **Python 3.10** (optimal compatibility)
- **PyTorch 2.x** with CUDA 11.8 support
- **Cellpose 3.0.10** for nuclei segmentation
- **NumPy 1.26.4** (stable version)
- **scikit-image 0.22.0** for image processing
- **Rich console output** for better logging
- **All bioinformatics packages** (anndata, scanpy)

## Next Steps

Once the environment is working:
1. `cd code/nuclei_segmentation`
2. `python run_this.py`
3. Check that the pipeline runs without import errors

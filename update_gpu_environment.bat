@echo off
REM Author: Christos Botos
REM Script to update conda environment with GPU support for I/R injury analysis

echo ======================================================================
echo GPU Environment Update for I/R Injury Spatial Multiomics Analysis
echo ======================================================================

REM Activate the conda environment
echo Activating conda environment iri310...
call conda activate iri310
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate conda environment iri310
    echo Please make sure the environment exists and conda is properly installed
    pause
    exit /b 1
)

echo Environment activated successfully!
echo.

REM Remove existing CPU-only PyTorch
echo Removing existing CPU-only PyTorch packages...
call conda remove pytorch torchvision torchaudio pytorch-cuda -y
echo.

REM Install GPU-enabled PyTorch
echo Installing GPU-enabled PyTorch with CUDA support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install GPU-enabled PyTorch
    pause
    exit /b 1
)
echo.

REM Install CuPy
echo Installing CuPy for GPU array operations...
call conda install cupy -c conda-forge -y
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install CuPy
    pause
    exit /b 1
)
echo.

REM Verify installation
echo ======================================================================
echo Verifying GPU package installation...
echo ======================================================================

echo Testing PyTorch CUDA support...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch verification failed
    pause
    exit /b 1
)
echo.

echo Testing CuPy GPU support...
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA devices: {cp.cuda.runtime.getDeviceCount()}'); device = cp.cuda.Device() if cp.cuda.runtime.getDeviceCount() > 0 else None; print(f'GPU memory: {device.mem_info[0]/(1024**3):.1f}/{device.mem_info[1]/(1024**3):.1f} GB' if device else 'No GPU')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: CuPy verification failed
    pause
    exit /b 1
)
echo.

echo Testing Cellpose GPU support...
python -c "from cellpose import models; import torch; model = models.Cellpose(gpu=True, model_type='nuclei') if torch.cuda.is_available() else None; print(f'Cellpose GPU enabled: {model.gpu if model else False}'); print(f'Cellpose device: {model.device if model else \"CPU\"}')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Cellpose verification failed
    pause
    exit /b 1
)
echo.

REM Final verification with our check script
echo Running comprehensive GPU support check...
python check_gpu_support.py
echo.

echo ======================================================================
echo GPU ENVIRONMENT UPDATE COMPLETED SUCCESSFULLY!
echo ======================================================================
echo Your environment now supports:
echo   * GPU-accelerated Cellpose segmentation
echo   * GPU-accelerated tile merging with CuPy
echo   * Optimized memory management for large images
echo.
echo You can now run the pipeline with full GPU acceleration!
echo ======================================================================

pause

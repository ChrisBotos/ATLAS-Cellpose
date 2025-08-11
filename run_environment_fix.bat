@echo off
REM Windows batch script to run the environment fix in WSL
REM This script will execute the bash script in your WSL environment

echo Starting environment fix in WSL...
echo.

REM Change to the project directory in WSL and run the fix script
wsl -d Ubuntu -e bash -c "cd /mnt/c/Projects/Nuclei-Segmentation-with-Cellpose && chmod +x fix_environment.sh && ./fix_environment.sh"

echo.
echo Environment fix completed. Check the output above for any errors.
echo.
echo To test the environment, run:
echo wsl -d Ubuntu -e bash -c "cd /mnt/c/Projects/Nuclei-Segmentation-with-Cellpose/code/nuclei_segmentation && conda activate venv310_cellpose3_fixed && python run_this.py"
echo.
pause

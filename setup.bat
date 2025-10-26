@echo off
set VENV_DIR=venv

:: 1. Create the virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: 2. Activate the environment
call "%VENV_DIR%\Scripts\activate.bat"

:: 3. Install standard libraries from requirements.txt
echo [INFO] Installing standard libraries...
pip install -r requirements.txt

:: 4. Install PyTorch with CUDA support
echo [INFO] Installing PyTorch for NVIDIA CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [SUCCESS] Full setup with PyTorch for CUDA is complete.
pause
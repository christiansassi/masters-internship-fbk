@echo off
setlocal

:: --- Configuration ---
set ENV_NAME=tf-gpu
set PYTHON_VERSION=3.10
set CUDATOOLKIT_VERSION=11.2
set CUDNN_VERSION=8.1.0
set TENSORFLOW_VERSION_CONSTRAINT="tensorflow<2.11"
set NUMPY_VERSION_CONSTRAINT="numpy<2"
set OTHER_PACKAGES="pandas python-dotenv rich scikit_learn wandb"
set REQUIREMENTS="requirements.txt"

echo.
echo --- Conda TensorFlow Environment Setup ---
echo.
echo This script will set up a Conda environment named '%ENV_NAME%' with Python %PYTHON_VERSION%.
echo It will then install CUDA, cuDNN, TensorFlow, NumPy, and other specified packages.
echo.

:: --- Check if Conda is installed and available in PATH ---
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda not found in your PATH. Please ensure Anaconda or Miniconda is installed
    echo        and its 'Scripts' directory is added to your system's PATH environment variable.
    echo Exiting.
    goto :eof
)

:: --- Check if environment already exists ---
conda env list | findstr /B /C:"%ENV_NAME% " >nul 2>nul
if %errorlevel% equ 0 (
    echo Warning: Conda environment '%ENV_NAME%' already exists.
    set /p CHOICE="Do you want to remove it and recreate? (y/n): "
    if /i "%CHOICE%"=="y" (
        echo Removing existing environment...
        conda env remove --name %ENV_NAME%
        if %errorlevel% neq 0 (
            echo ERROR: Failed to remove environment '%ENV_NAME%'.
            echo Please resolve the issue or remove it manually, then retry.
            goto :eof
        )
        echo Environment removed.
    ) else (
        echo Setup aborted by user.
        goto :eof
    )
)

:: --- Create Conda Environment ---
echo.
echo Creating Conda environment '%ENV_NAME%' with Python %PYTHON_VERSION%...
conda create --name %ENV_NAME% python=%PYTHON_VERSION% -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to create Conda environment '%ENV_NAME%'.
    echo Please check your Conda installation and try again.
    goto :eof
)
echo Environment '%ENV_NAME%' created successfully.

:: --- Activate Conda Environment ---
echo.
echo Activating environment '%ENV_NAME%'...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate environment '%ENV_NAME%'.
    echo Please ensure Conda is properly configured.
    goto :eof
)
echo Environment activated. Current Python: %PYTHON_VERSION%.

:: --- Install CUDA Toolkit and cuDNN with Conda ---
echo.
echo Installing cudatoolkit=%CUDATOOLKIT_VERSION% and cudnn=%CUDNN_VERSION% from conda-forge...
conda install -c conda-forge cudatoolkit=%CUDATOOLKIT_VERSION% cudnn=%CUDNN_VERSION% -y
if %errorlevel% neq 0 (
    echo WARNING: Failed to install CUDA/cuDNN via Conda.
    echo This might indicate a compatibility issue or network problem.
    echo TensorFlow GPU might not work without these. Proceeding with other installations...
    echo You may need to manually install CUDA/cuDNN or troubleshoot this step.
) else (
    echo CUDA Toolkit and cuDNN installed via Conda.
)

:: --- Install TensorFlow via pip ---
echo.
echo Installing TensorFlow (%TENSORFLOW_VERSION_CONSTRAINT%) using pip...
python -m pip install %TENSORFLOW_VERSION_CONSTRAINT%
if %errorlevel% neq 0 (
    echo ERROR: Failed to install TensorFlow. This is a critical step.
    echo Please check the error messages above for details (e.g., compatibility, network issues).
    goto :eof
)
echo TensorFlow installed.

:: --- Install NumPy via pip (version constraint) ---
echo.
echo Installing NumPy (%NUMPY_VERSION_CONSTRAINT%) using pip...
python -m pip install %NUMPY_VERSION_CONSTRAINT%
if %errorlevel% neq 0 (
    echo WARNING: Failed to install NumPy. TensorFlow might not function correctly.
    echo Please check the error messages.
) else (
    echo NumPy installed.
)

:: --- Install other specified Python packages ---
echo.
echo Installing other Python packages: %OTHER_PACKAGES%...
python -m pip install %OTHER_PACKAGES%
if %errorlevel% neq 0 (
    echo WARNING: Failed to install some or all of the additional Python packages.
    echo Please check the error messages for details.
) else (
    echo Other packages installed.
)

:: --- Install requirements ---
echo.
echo Installing requirements...
python -m pip install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo WARNING: Failed to install some or all of the required Python packages.
    echo Please check the error messages for details.
) else (
    echo Required packages installed.
)

:: --- Verify TensorFlow GPU Installation ---
echo.
echo Verifying TensorFlow GPU installation...
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
if %errorlevel% neq 0 (
    echo WARNING: TensorFlow GPU verification failed (non-zero exit code).
    echo This might mean no GPU was detected or there's a problem with the TensorFlow installation.
) else (
    echo.
    echo TensorFlow GPU verification command executed. Check the output above for GPU devices.
)

echo.
echo --- Setup Complete ---
echo To activate this environment manually in the future, run:
echo    conda activate %ENV_NAME%
echo.

endlocal

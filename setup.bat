@echo off
ECHO Creating HAL 9000 environment for Windows...
ECHO This will set up a dedicated Python environment for the CSM voice assistant

:: Navigate to CSM directory
cd /d %~dp0

:: Create a new environment with Python 3.10
ECHO Creating Python 3.10 environment...
call conda create -n hal python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    ECHO Failed to create conda environment. Please install Anaconda or Miniconda.
    ECHO https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Activate the environment
call conda activate hal
if %ERRORLEVEL% NEQ 0 (
    ECHO Trying alternative activation method...
    call conda.bat activate hal
)

:: Install PyTorch 2.6 with CUDA support (key for CSM model)
ECHO Installing PyTorch 2.6 with CUDA support...
call pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

:: Install Windows-specific dependencies
ECHO Installing Windows-specific dependencies...
call pip install triton-windows
call pip install bitsandbytes --upgrade

:: Install basic dependencies
ECHO Installing basic dependencies...
call pip install numpy==1.26.0
call pip install tokenizers==0.21.0
call pip install transformers==4.49.0
call pip install huggingface_hub==0.28.1
call pip install pyaudio
call pip install PyQt5

:: Install faster-whisper with CUDA support
ECHO Installing faster-whisper...
call pip install faster-whisper==0.9.0

:: Install llama-cpp-python with CUDA support
ECHO Installing llama-cpp-python with CUDA support...
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
call pip install llama-cpp-python==0.2.38

:: Install remaining CSM dependencies
ECHO Installing CSM dependencies...
call pip install moshi==0.2.2
call pip install torchtune==0.4.0
call pip install torchao==0.9.0
call pip install git+https://github.com/SesameAILabs/silentcipher@master

ECHO Environment setup complete!
ECHO To activate this environment, run: conda activate hal
ECHO Then run: python hal9000_assistant.py

pause
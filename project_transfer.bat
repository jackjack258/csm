@echo off
:: This script transfers your voice assistant project to WSL and sets up a launcher

echo Setting up Voice Assistant in WSL...

:: Find Ubuntu WSL distribution path
for /f "tokens=*" %%i in ('wsl -l -v ^| findstr Ubuntu') do set DISTRO=%%i
set DISTRO=%DISTRO:~0,16%
echo Using WSL distribution: %DISTRO%

:: Create directories in WSL
wsl -d %DISTRO% mkdir -p ~/voice-assistant

:: Copy project files to WSL
echo Copying project files to WSL...
set SOURCE_DIR=%~dp0
set WSL_HOME=\\wsl$\%DISTRO%\home\%USERNAME%\voice-assistant

:: Use robocopy to transfer files
robocopy "%SOURCE_DIR%" "%WSL_HOME%" /E /NFL /NDL /NJH /NJS /nc /ns /np

:: Create Python virtual environment in WSL
echo Setting up Python environment in WSL...
wsl -d %DISTRO% -e bash -c "cd ~/voice-assistant && python3 -m venv env && source env/bin/activate && pip install -r requirements.txt"

:: Create launcher script
echo Creating launcher script...
(
echo @echo off
echo echo Starting X Server...
echo start "" "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -nowgl -ac
echo.
echo echo Starting PulseAudio...
echo start "" "C:\PulseAudio\start-pulseaudio.bat"
echo.
echo echo Waiting for services to start...
echo timeout /t 5
echo.
echo echo Starting Voice Assistant...
echo wsl -d %DISTRO% bash -c "cd ~/voice-assistant && source env/bin/activate && python voiceui.py"
) > start-voice-assistant.bat

echo.
echo Setup complete! Before running the application:
echo 1. Install VcXsrv from: https://sourceforge.net/projects/vcxsrv/
echo 2. Install PulseAudio for Windows and configure as described in the guide
echo 3. Run start-voice-assistant.bat to launch the application
echo.
echo See the complete guide for detailed instructions.

pause
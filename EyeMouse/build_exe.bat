@echo off
REM Build EyeMouse.exe with PyInstaller.
REM Run from inside an activated Python 3.11 venv with requirements.txt installed.

setlocal
cd /d "%~dp0"

where pyinstaller >nul 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM EyeMouse imports modules from the sibling GazeOverlay folder, so we add
REM that path explicitly. MediaPipe model files also need to be bundled.
for /f "delims=" %%i in ('python -c "import mediapipe, os; print(os.path.dirname(mediapipe.__file__))"') do set MP_DIR=%%i
echo MediaPipe dir: %MP_DIR%

set GAZE_DIR=%~dp0..\GazeOverlay

pyinstaller ^
    --name EyeMouse ^
    --onefile ^
    --windowed ^
    --noconsole ^
    --paths "%GAZE_DIR%" ^
    --add-data "%GAZE_DIR%\gaze_engine.py;." ^
    --add-data "%GAZE_DIR%\overlay.py;." ^
    --add-data "%GAZE_DIR%\main.py;gaze_overlay_main_src" ^
    --add-data "%MP_DIR%\modules;mediapipe\modules" ^
    --hidden-import gaze_engine ^
    --hidden-import overlay ^
    --hidden-import mediapipe ^
    --hidden-import scipy.spatial.transform._rotation_groups ^
    --collect-all mediapipe ^
    main.py

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo ===============================================
echo   Build complete:  dist\EyeMouse.exe
echo ===============================================
endlocal

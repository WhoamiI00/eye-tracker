@echo off
REM Build GazeOverlay.exe with PyInstaller.
REM Run from inside an activated Python 3.11 venv with requirements.txt installed.

setlocal
cd /d "%~dp0"

where pyinstaller >nul 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM MediaPipe ships model .tflite files inside the package; they need to be bundled.
for /f "delims=" %%i in ('python -c "import mediapipe, os; print(os.path.dirname(mediapipe.__file__))"') do set MP_DIR=%%i
echo MediaPipe dir: %MP_DIR%

pyinstaller ^
    --name GazeOverlay ^
    --onefile ^
    --windowed ^
    --noconsole ^
    --add-data "%MP_DIR%\modules;mediapipe\modules" ^
    --hidden-import mediapipe ^
    --hidden-import scipy.spatial.transform._rotation_groups ^
    --collect-all mediapipe ^
    main.py

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

REM Copy HOTKEYS.txt next to the .exe so users have a cheat sheet
if exist HOTKEYS.txt copy /Y HOTKEYS.txt dist\HOTKEYS.txt >nul

echo.
echo ===============================================
echo   Build complete:  dist\GazeOverlay.exe
echo                    dist\HOTKEYS.txt
echo ===============================================
endlocal

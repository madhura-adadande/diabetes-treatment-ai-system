@echo off
echo ========================================
echo Starting Diabetes AI Backend Server
echo ========================================
echo.

REM Activate virtual environment
call diabetes_env\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure diabetes_env exists and is properly set up
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Navigate to api directory
cd api

REM Start the FastAPI server
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python main.py

pause


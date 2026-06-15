@echo off
:: PQ Analyzer Launcher
:: Double-click this file to open the PQ Analyzer.
:: Requires Python 3.9+ installed from python.org

cd /d "%~dp0"

:: Try pythonw first (no console window), fall back to python
where pythonw >nul 2>&1
if %errorlevel% == 0 (
    start "" pythonw "%~dp0run.py"
) else (
    where python >nul 2>&1
    if %errorlevel% == 0 (
        start "" python "%~dp0run.py"
    ) else (
        echo Python was not found.  Please install Python 3.9+ from python.org
        pause
    )
)

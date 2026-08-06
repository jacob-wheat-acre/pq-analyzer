@echo off
:: PQ Analyzer Launcher
:: Double-click this file to open the PQ Analyzer.
:: Requires Python 3.9+.  On an Xcel-managed PC, request it through
:: Xcel Software Center (link on the intranet homepage) -- see GIT_GUIDE.md.

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
        echo Python was not found on this PC.
        echo.
        echo Try running:  py --version
        echo If that works, Python is installed but not on the system PATH --
        echo ask IT to add it.
        echo.
        echo If Python is not installed, request it through Xcel Software Center
        echo ^(link on the intranet homepage^).  See GIT_GUIDE.md section 1.
        pause
    )
)

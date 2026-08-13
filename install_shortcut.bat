@echo off
:: Creates a "PQ Analyzer" shortcut on the current user's Desktop.
:: Run this once after copying the pq-analyzer folder to the machine.
::
:: This file used to build the shortcut itself, in a line of inline PowerShell
:: that had drifted apart from install_shortcut.py: the .py set the shortcut's
:: icon but looked for the Desktop at ~/Desktop, which OneDrive redirection
:: moves; this .bat found the real Desktop but set no icon at all, so the
:: shortcut it made showed the generic .bat gears. Whichever one you ran, you
:: got half of it. There is one implementation now, and this is a wrapper.

setlocal
cd /d "%~dp0"

where pythonw >nul 2>&1
if %errorlevel% == 0 (
    python "%~dp0install_shortcut.py"
    goto done
)
where python >nul 2>&1
if %errorlevel% == 0 (
    python "%~dp0install_shortcut.py"
    goto done
)
where py >nul 2>&1
if %errorlevel% == 0 (
    py "%~dp0install_shortcut.py"
    goto done
)

echo Python was not found on this PC.
echo.
echo Try running:  py --version
echo If that works, Python is installed but not on the system PATH --
echo ask IT to add it.
echo.
echo If Python is not installed, request it through Xcel Software Center
echo ^(link on the intranet homepage^).  See GIT_GUIDE.md section 1.

:done
pause

@echo off
:: Creates a "PQ Analyzer" shortcut on the current user's Desktop.
:: Run this once after copying the pq-analyzer folder to the machine.

setlocal

set "TOOL_DIR=%~dp0"
set "BAT_FILE=%TOOL_DIR%PQ Analyzer.bat"
set "SHORTCUT=%USERPROFILE%\Desktop\PQ Analyzer.lnk"

:: Use PowerShell to create a proper .lnk shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $sc = $ws.CreateShortcut('%SHORTCUT%'); ^
   $sc.TargetPath = '%BAT_FILE%'; ^
   $sc.WorkingDirectory = '%TOOL_DIR%'; ^
   $sc.Description = 'PQ Analyzer — Power Quality Analysis Tool'; ^
   $sc.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo  Shortcut created on your Desktop: PQ Analyzer
    echo  Double-click it any time to launch the tool.
    echo.
) else (
    echo.
    echo  Could not create shortcut.  You can still launch the tool by
    echo  double-clicking "PQ Analyzer.bat" in the pq-analyzer folder.
    echo.
)

pause

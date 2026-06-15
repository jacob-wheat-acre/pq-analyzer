@echo off
:: Creates a "PQ Analyzer" shortcut on the current user's Desktop.
:: Uses the Windows API to find the real Desktop path, which handles
:: OneDrive folder redirection common on corporate machines.
:: Run this once after copying the pq-analyzer folder to the machine.

setlocal

set "TOOL_DIR=%~dp0"
set "BAT_FILE=%TOOL_DIR%PQ Analyzer.bat"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$d = [Environment]::GetFolderPath('Desktop'); $lnk = Join-Path $d 'PQ Analyzer.lnk'; $ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut($lnk); $sc.TargetPath = '%BAT_FILE%'; $sc.WorkingDirectory = '%TOOL_DIR%'; $sc.Description = 'PQ Analyzer - Power Quality Analysis Tool'; $sc.Save(); if (Test-Path $lnk) { Write-Host ''; Write-Host ' Shortcut created on your Desktop: PQ Analyzer'; Write-Host ' Double-click it any time to launch the tool.'; Write-Host '' } else { Write-Host ''; Write-Host ' Could not create shortcut. You can still launch the tool by'; Write-Host ' double-clicking PQ Analyzer.bat in the pq-analyzer folder.'; Write-Host '' }"

pause

#!/usr/bin/env python3
"""
install_shortcut.py — create a desktop launcher for PQ Analyzer.
  Mac:     builds a .app bundle on the Desktop
  Windows: creates a .lnk shortcut on the Desktop (replaces install_shortcut.bat)

Run once after setting up the tool folder:
  python3 install_shortcut.py
"""

import os
import platform
import shutil
import stat
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).parent.resolve()
DESKTOP  = Path.home() / "Desktop"


# ── Mac ───────────────────────────────────────────────────────────────────────

def install_mac():
    app = DESKTOP / "PQ Analyzer.app"

    # Remove old version if present
    if app.exists():
        shutil.rmtree(app)

    # Clear stale pyc cache so the .app always loads the current source
    pycache = TOOL_DIR / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    # Directory structure
    contents  = app / "Contents"
    macos_dir = contents / "MacOS"
    res_dir   = contents / "Resources"
    for d in (macos_dir, res_dir):
        d.mkdir(parents=True)

    # Use the same python3 that is running this installer — it's already the
    # correct architecture and has all required packages installed.
    # run.py registers the app with NSApplication via ctypes so tkinter
    # widgets render correctly without needing the Python.app wrapper.
    import subprocess
    r = subprocess.run(["which", "python3"], capture_output=True, text=True)
    py3 = r.stdout.strip() or sys.executable

    # Launcher shell script
    launcher = macos_dir / "PQ Analyzer"
    launcher.write_text(
        f'#!/bin/bash\n'
        f'cd "{TOOL_DIR}"\n'
        f'rm -rf "{TOOL_DIR}/__pycache__"\n'
        f'exec arch -arm64 "{py3}" "{TOOL_DIR / "run.py"}"\n'
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Copy icon
    icns_src = TOOL_DIR / "icon.icns"
    if icns_src.exists():
        shutil.copy(icns_src, res_dir / "icon.icns")

    # Info.plist
    (contents / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
        ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        '    <key>CFBundleExecutable</key>\n'
        '    <string>PQ Analyzer</string>\n'
        '    <key>CFBundleIconFile</key>\n'
        '    <string>icon</string>\n'
        '    <key>CFBundleName</key>\n'
        '    <string>PQ Analyzer</string>\n'
        '    <key>CFBundleDisplayName</key>\n'
        '    <string>PQ Analyzer</string>\n'
        '    <key>CFBundleIdentifier</key>\n'
        '    <string>com.xcelenergy.pq-analyzer</string>\n'
        '    <key>CFBundlePackageType</key>\n'
        '    <string>APPL</string>\n'
        '    <key>CFBundleVersion</key>\n'
        '    <string>1.0</string>\n'
        '    <key>CFBundleShortVersionString</key>\n'
        '    <string>1.0</string>\n'
        '    <key>LSUIElement</key>\n'
        '    <false/>\n'
        '</dict>\n'
        '</plist>\n'
    )

    # Tell macOS to refresh the icon cache for this app
    os.system(f'touch "{app}"')

    print(f"\n  PQ Analyzer.app created on your Desktop.")
    print(f"  Double-click it any time to launch the tool.\n")
    print(f"  Note: on first launch macOS may show a security warning.")
    print(f"  If so: System Settings → Privacy & Security → Open Anyway\n")


# ── Windows ───────────────────────────────────────────────────────────────────

def install_windows():
    shortcut = DESKTOP / "PQ Analyzer.lnk"
    bat      = TOOL_DIR / "PQ Analyzer.bat"
    ico      = TOOL_DIR / "icon.ico"

    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut('{shortcut}'); "
        f"$sc.TargetPath = '{bat}'; "
        f"$sc.WorkingDirectory = '{TOOL_DIR}'; "
        f"$sc.IconLocation = '{ico}'; "
        "$sc.Description = 'PQ Analyzer — Power Quality Analysis Tool'; "
        "$sc.Save()"
    )
    import subprocess
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True, text=True,
    )
    if shortcut.exists():
        print(f"\n  Shortcut created: {shortcut}\n")
    else:
        print(f"\n  Could not create shortcut: {result.stderr.strip()}")
        print(f"  You can still launch by double-clicking 'PQ Analyzer.bat'\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if platform.system() == "Darwin":
        install_mac()
    elif platform.system() == "Windows":
        install_windows()
    else:
        print("Unsupported platform — manually create a shortcut to run.py")
        sys.exit(1)

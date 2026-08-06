#!/usr/bin/env python3
"""Verify a PQ Analyzer install and say plainly what to do about anything broken.

Run this when the tool won't start, when the transformer Size picker stays
greyed out, or right after a fresh setup to confirm it took:

    python check_install.py

Written to run on a machine where the libraries are missing or broken, so it
imports nothing outside the standard library at module level.
"""

import os
import platform
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# name → (import name, why the tool needs it)
_REQUIRED = [
    ("numpy",        "numpy",      "all numeric processing"),
    ("pandas",       "pandas",     "interval data and statistics"),
    ("matplotlib",   "matplotlib", "report plots"),
    ("python-docx",  "docx",       "Word report generation"),
    ("Pillow",       "PIL",        "image handling and the app icon"),
]

_MIN_PYTHON = (3, 9)


def _line(char="-"):
    print(char * 72)


def _ok(msg):
    print("  [ OK ]  " + msg)


def _bad(msg):
    print("  [FAIL]  " + msg)


def _warn(msg):
    print("  [WARN]  " + msg)


def check_python():
    """Report the interpreter actually running, and whether it's new enough.

    The interpreter path is the single most useful line in this report: a very
    common failure is pip installing into one Python while the launcher runs
    another, which looks identical to 'nothing is installed'.
    """
    print("Python interpreter")
    v = sys.version_info
    print("     version:  %d.%d.%d" % (v.major, v.minor, v.micro))
    print("  executable:  %s" % sys.executable)
    print("    platform:  %s %s" % (platform.system(), platform.machine()))

    if v[:2] < _MIN_PYTHON:
        _bad("Python %d.%d or later is required." % _MIN_PYTHON)
        return False
    _ok("Python version is new enough.")
    return True


def check_imports():
    """Import each dependency and report its version or its actual error."""
    print("\nRequired libraries")
    failures = []

    for dist, module, purpose in _REQUIRED:
        try:
            mod = __import__(module)
        except Exception as exc:
            _bad("%-12s could not be imported  (%s)" % (dist, purpose))
            print("          %s: %s" % (type(exc).__name__, exc))
            failures.append((dist, exc))
            continue

        version = getattr(mod, "__version__", "unknown version")
        _ok("%-12s %s" % (dist, version))

    return failures


def check_abi():
    """Catch the numpy/pandas binary-compatibility mismatch specifically.

    pandas wheels are compiled against a particular numpy ABI.  When an older
    numpy shadows the expected one, both import individually but using them
    together raises 'numpy.dtype size changed'.
    """
    print("\nnumpy / pandas compatibility")
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        _warn("Skipped — one of numpy/pandas did not import above.")
        return False

    try:
        frame = pd.DataFrame({"a": np.arange(5, dtype="float64")})
        assert float(frame["a"].sum()) == 10.0
    except Exception as exc:
        _bad("numpy and pandas are installed but not binary-compatible.")
        print("          %s: %s" % (type(exc).__name__, exc))
        return False

    _ok("numpy and pandas work together.")
    return True


def check_app_import():
    """Import the analysis engine the same way run.py does.

    This is the check that maps to the reported symptom: when this import
    fails, run.py falls back to empty Blue Book tables and the transformer
    Size picker never enables.
    """
    print("\nPQ Analyzer modules")
    sys.path.insert(0, str(_HERE))
    try:
        from pq_analyzer import _BLUE_BOOK_ISC  # noqa: F401
    except Exception as exc:
        _bad("pq_analyzer failed to import — this is what disables the")
        print("          transformer/ISC lookup and greys out the Size picker.")
        print("          %s: %s" % (type(exc).__name__, exc))
        return False

    _ok("pq_analyzer imported (%d Blue Book entries loaded)." % len(_BLUE_BOOK_ISC))
    return True


def check_location():
    """Warn about cloud-synced folders, which can dehydrate files on read."""
    print("\nInstall location")
    print("  %s" % _HERE)

    lowered = str(_HERE).lower()
    if "onedrive" in lowered or "dropbox" in lowered or "box sync" in lowered:
        _warn("This folder is inside a cloud-synced directory.")
        print("          Sync can replace files with cloud-only placeholders and")
        print("          can conflict with git.  Consider moving the folder to")
        print("          %s." % (Path(os.path.expanduser("~")) / "Documents"))
        return False

    _ok("Not inside a cloud-synced folder.")
    return True


def check_git():
    """Report whether this copy can receive updates via git pull."""
    print("\nGit")
    if (_HERE / ".git").is_dir():
        _ok("This copy came from git clone — `git pull` will work.")
        return True

    _warn("This folder is not a git repository.")
    print("          `git pull` will fail with 'fatal: not a git repository'.")
    print("          See GIT_GUIDE.md section 4 to convert it.")
    return False


def main():
    _line("=")
    print("PQ Analyzer — install check")
    _line("=")

    python_ok = check_python()
    failures = check_imports()
    abi_ok = check_abi() if not failures else False
    app_ok = check_app_import() if not failures else False
    location_ok = check_location()
    git_ok = check_git()

    print()
    _line("=")

    if python_ok and not failures and abi_ok and app_ok:
        print("RESULT: install looks good.")
        if not location_ok or not git_ok:
            print("Warnings above are worth fixing but won't stop the tool running.")
        _line("=")
        return 0

    print("RESULT: this install is broken.")
    print()

    if not python_ok:
        print("Fix Python first — everything else depends on it.")
        print("On an Xcel-managed PC, request Python through Xcel Software")
        print("Center (link on the intranet homepage).  See GIT_GUIDE.md sec. 1.")
        _line("=")
        return 1

    print("Reinstall the libraries.  From inside this folder, run:")
    print()
    # One line, no continuation character — this gets pasted into Windows cmd,
    # where a trailing backslash is not a line continuation.
    print("    python -m pip install --force-reinstall --no-cache-dir "
          "-r pq_analyzer_requirements.txt")
    print()
    print("Use `python -m pip`, not plain `pip` — it guarantees the packages")
    print("go to the interpreter listed above rather than a different Python.")
    print()
    print("If that fails with an SSL, certificate, or proxy error, Xcel's")
    print("network is blocking the package server.  Open a ticket asking for")
    print("access to pypi.org and files.pythonhosted.org.  Do not apply")
    print("workarounds that disable certificate checking.")
    print()
    print("If it still fails, send this entire output to the maintainer.")
    _line("=")
    return 1


if __name__ == "__main__":
    sys.exit(main())

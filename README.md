# PQ Analyzer

Power quality analysis tool for Xcel Energy / PSCo field investigations.
Reads Pronto meter `.pqd` files and produces a Word report with compliance
findings, harmonic diagnostics, and root cause analysis.

---

## Requirements

- Python 3.9 or later
- Dependencies listed in `pq_analyzer_requirements.txt`

---

## Windows — First-Time Setup

### 1. Install Python

Download from [python.org](https://www.python.org/downloads/windows/) and run
the installer.  On the first screen, check **"Add Python to PATH"** before
clicking Install Now.

Verify in a Command Prompt:

```
python --version
```

### 2. Copy the pq-analyzer folder to the PC

Copy the entire `pq-analyzer` folder to a location you can find, such as your
Documents folder or the Desktop.

### 3. Install dependencies

Open a Command Prompt **in the pq-analyzer folder** (Shift+right-click the
folder → "Open PowerShell window here", or use `cd` to navigate), then run:

```
pip install -r pq_analyzer_requirements.txt
```

This installs pandas, numpy, matplotlib, python-docx, and Pillow.  You only
need to do this once.

### 4. Create a Desktop shortcut (optional)

Double-click `install_shortcut.bat` inside the pq-analyzer folder.  This
creates a **PQ Analyzer** shortcut on your Desktop.

### 5. Launch the tool

Double-click **PQ Analyzer** on the Desktop, or double-click `PQ Analyzer.bat`
inside the folder.

The GUI opens with no console window.  If Python is not found, a message
appears in the Command Prompt window pointing you back to step 1.

---

## Mac — First-Time Setup

### 1. Install Python

Download from [python.org](https://www.python.org/downloads/mac-osx/) or
install via Homebrew:

```
brew install python
```

Verify in Terminal:

```
python3 --version
```

### 2. Install dependencies

Open Terminal, navigate to the pq-analyzer folder, and run:

```
pip3 install -r pq_analyzer_requirements.txt
```

### 3. Launch the tool

Double-click `run.py`, or from Terminal:

```
python3 run.py
```

To find the file path to a `.pqd` file: drag the file from Finder into the
Terminal window — the full path appears.  Alternatively, use Option+right-click
on the file in Finder and choose **Copy as Pathname**.

---

## Typical Workflow

1. Open the tool and select a `.pqd` file using the **Browse** button.
2. Fill in site name, address, nominal voltage, and engineer information.
3. Enter ISC (short-circuit current in amps) if known — required for the
   correct IEEE 519 TDD limit and statistical compliance check.
4. Enter transformer nameplate kVA if you want transformer loading reported.
5. Click **Run Analysis**.
6. When complete, click **Open Report** to view the Word document, or
   **Open Folder** to see all output files (report + plots).

Output files are written to a `pq_output/<site-stem>/` subfolder inside the
pq-analyzer directory.

---

## What the Tool Analyzes

| Check | Standard | Notes |
|---|---|---|
| Voltage compliance | ANSI C84.1-2020 | Range A / Range B; peak/min if maxmin record present |
| THD / TDD | IEEE 519-2022 | Basic average check |
| Statistical harmonic compliance | IEEE 519-2022 Clause 5 | P95/P99 weekly; daily VST P99 |
| Per-order harmonic spectrum | IEEE 519-2022 Table 1 & 2 | Requires per-order meter channels |
| Neutral harmonic accumulation | IEEE 519 / IEC | Triplen zero-sequence buildup |
| Harmonic source attribution | Z_h impedance method | Resonance detection + customer/utility split |
| Harmonic load signature | Cosine similarity | 14 reference load types |
| Voltage imbalance | NEMA MG1 | Flag > 3% |
| Current imbalance | PSCo Blue Book | Flag > 10%; tariff limit 15% |
| Power factor | PSCo Tariff Sheet R73 | Minimum 0.90 lagging |
| Flicker (Pst / Plt) | IEEE 1453-2022 | Pst ≤ 1.0; Plt ≤ 0.65 |
| Demand / transformer loading | — | Requires nameplate kVA |
| Event detection | IEEE 1159 | Sags, swells, current steps |
| Root cause analysis | — | Synthesized findings with recommendations |

---

## Troubleshooting

**"Python was not found" on Windows**
Reinstall Python from python.org and make sure "Add to PATH" is checked.
Then close and reopen any Command Prompt windows.

**"Module not found" error when launching**
Run `pip install -r pq_analyzer_requirements.txt` again from inside the
pq-analyzer folder.

**No report generated / Word section blank**
python-docx is missing.  Run `pip install python-docx`.

**File won't load / "No interval data found"**
The tool supports Pronto-format `.pqd` files (obs[23] interval average record).
Other PQDIF variants are not guaranteed to parse correctly.

**Report opens but harmonic signature section is empty**
Per-order harmonic channels (h3_current_a, h5_current_a, etc.) are not present
in the meter export.  Only thd_current_* is available, which is insufficient
for signature matching.

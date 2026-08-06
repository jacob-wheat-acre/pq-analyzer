# PQ Analyzer

Power quality analysis tool for Xcel Energy / PSCo field investigations.
Reads Pronto meter `.pqd` files and produces a Word report with compliance
findings, harmonic diagnostics, and root cause analysis.

---

## Requirements

- Python 3.9 or later
- Git
- Dependencies listed in `pq_analyzer_requirements.txt`

---

## First-Time Setup

**New to this, or setting up a new PC? Follow
[GIT_GUIDE.md](GIT_GUIDE.md).**  It's the full walkthrough — requesting
Python through Xcel's software request system, installing Git, getting the
code, and keeping it updated — written for people who have never used the
command line.

The short version, once Python and Git are installed:

### Windows

```
cd %USERPROFILE%\Documents
git clone https://github.com/jacob-wheat-acre/pq-analyzer.git
cd pq-analyzer
pip install -r pq_analyzer_requirements.txt
```

Then double-click `install_shortcut.bat` once to get a **PQ Analyzer**
Desktop shortcut.  Launch from that shortcut, or from `PQ Analyzer.bat` in
the folder.  The GUI opens with no console window.

> Don't have Python? On an Xcel-managed PC it must be requested through
> **Xcel Software Center** (link on the intranet homepage) — the python.org
> installer is blocked.  See
> [GIT_GUIDE.md § 1](GIT_GUIDE.md#1-install-python).

> Already have a copy of the folder that didn't come from `git clone`?
> Convert it once — see [GIT_GUIDE.md
> § 4](GIT_GUIDE.md#4-get-the-code-git-clone).

### Mac

```
cd ~/Documents
git clone https://github.com/jacob-wheat-acre/pq-analyzer.git
cd pq-analyzer
pip3 install -r pq_analyzer_requirements.txt
python3 run.py
```

To find the file path to a `.pqd` file: drag the file from Finder into the
Terminal window — the full path appears.  Alternatively, Option+right-click
the file in Finder and choose **Copy as Pathname**.

---

## Keeping the Tool Updated

Fixes ship regularly, especially for new meter file variants.  Update
periodically — and any time you're asked to retry after reporting an error.

From inside the `pq-analyzer` folder:

```
git pull
pip install -r pq_analyzer_requirements.txt      # pip3 on Mac
```

`git pull` downloads only what changed.  Your generated reports are safe:
everything in `pq_output/` is excluded from git and is never touched by an
update.

If `git pull` reports an error, see [GIT_GUIDE.md
§ 7](GIT_GUIDE.md#7-when-git-pull-complains).

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
Python is installed but Windows can't find it.  Try `py --version` — the `py`
launcher usually works when `python` doesn't.  If that works, you're fine.  If
neither does, ask IT to add Python to the system PATH.  Close and reopen any
Command Prompt windows afterward.

**"fatal: not a git repository" when running `git pull`**
Either you're not in the `pq-analyzer` folder, or your copy didn't come from
`git clone`.  See [GIT_GUIDE.md § 4](GIT_GUIDE.md#4-get-the-code-git-clone).

**`pip install` fails with an SSL, certificate, or proxy error**
Xcel's network is intercepting the connection to the package server.  Don't
apply workarounds that disable certificate checking — open a ticket asking for
access to pypi.org and files.pythonhosted.org.

**Anything looks wrong and you don't know why — start here**
Run `python check_install.py` from inside the pq-analyzer folder.  It verifies
the Python version, every required library, and the analysis engine itself,
then tells you exactly what to run to fix what's broken.  Include its output
when reporting a problem.

**The transformer Size dropdown is greyed out**
If you haven't chosen a transformer **Type** yet, that's expected — Size
enables once a Type is selected.  If it stays greyed out after selecting a
Type, the install is broken: run `python check_install.py`.

**"Module not found" error when launching**
Run `pip install -r pq_analyzer_requirements.txt` again from inside the
pq-analyzer folder.  If it still fails, run `python check_install.py` — the
usual cause is pip installing into a different Python than the launcher runs.

**No report generated / Word section blank**
python-docx is missing.  Run `pip install python-docx`.

**File won't load / "No interval data found"**
The reader implements IEEE Std 1159.3-2019 (PQDIF), so any standards-compliant
`.pqd` file should load, from Pronto or another vendor.  The error means no
observation record with a common time base across its channels was found.  Run
with `--verbose` to see the observation records the file actually contains.

**Report opens but harmonic signature section is empty**
Per-order harmonic channels (h3_current_a, h5_current_a, etc.) are not present
in the meter export.  Only thd_current_* is available, which is insufficient
for signature matching.

---

## Reporting Problems and Suggesting Changes

Found a bad number, a file that won't load, or a section that reads wrong?
Send it to the maintainer with the `.pqd` file and what you expected — that's
the fastest path to a fix, and fixes reach everyone through `git pull`.

Changes are published by the maintainer only.  The commit and push workflow
is documented in [GIT_GUIDE.md
§ 8](GIT_GUIDE.md#8-for-the-maintainer-pushing-changes).

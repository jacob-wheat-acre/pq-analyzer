# PQ Analyzer — Setup and Update Guide

A from-scratch walkthrough for getting PQ Analyzer onto a new Xcel Energy PC
and keeping it current.  No prior experience with git or the command line is
assumed.

Written for Windows, since that's what the team runs.  Mac notes are at the
end.

**Time required:** about 30 minutes of your attention, plus however long the
Python software request takes to be approved.

---

## Contents

- [0. Why git, and what it actually is](#0-why-git-and-what-it-actually-is)
- [1. Install Python](#1-install-python)
- [2. Install Git](#2-install-git)
- [3. Set up your GitHub sign-in](#3-set-up-your-github-sign-in)
- [4. Get the code (`git clone`)](#4-get-the-code-git-clone)
- [5. Install the Python dependencies](#5-install-the-python-dependencies)
- [6. Getting updates (`git pull`)](#6-getting-updates-git-pull)
- [7. When `git pull` complains](#7-when-git-pull-complains)
- [8. For the maintainer: pushing changes](#8-for-the-maintainer-pushing-changes)
- [9. Glossary](#9-glossary)
- [10. Cheat sheet](#10-cheat-sheet)
- [11. Mac](#11-mac)

---

## 0. Why git, and what it actually is

PQ Analyzer is a set of Python files that changes regularly — new meter file
variants, corrected limits, report fixes.  If your copy came from a network
share or a ZIP someone sent you, there is no way to get just the fixes.  You
get a whole new folder every time, and no way to tell what changed.

Git solves that.  It's a program that tracks a folder's history.  The
authoritative copy of the folder lives on GitHub (a website that hosts git
folders); your PC keeps its own full copy.  One command, `git pull`,
compares the two and downloads only what actually changed — usually a few
kilobytes, in about a second.

Three terms that show up constantly, in plain language:

| Term | Means |
|---|---|
| **repository** (or **repo**) | The tracked folder, plus its whole history. `pq-analyzer` is a repo. |
| **origin** | The nickname your PC uses for the GitHub copy. When you type `git pull`, it pulls from origin. |
| **commit** | One saved snapshot of the folder, with a message describing what changed. History is a list of commits. |

You do **not** need to understand branches, merges, or staging to use the
tool.  Sections 1–7 are all a normal user ever needs.

---

## 1. Install Python

Python is the language PQ Analyzer is written in.  Without it, nothing runs.

On an Xcel-managed PC you cannot install Python by downloading it from
python.org — the installer will be blocked, and you don't have admin rights.
It has to come through the internal software request system.

**How to request it:**

1. Open **Xcel Software Center** — the link is on the intranet homepage.
2. Search the catalog for **Python**.
3. Request **Python 3.9 or later** (3.11 or 3.12 is a good choice if the
   catalog offers a version choice).
4. In the business-justification field, something like:

   > Required to run the PQ Analyzer tool used for power quality field
   > investigations and IEEE 519 / ANSI C84.1 compliance reporting.

5. If the catalog entry offers options, ask for it to be **added to PATH**.
   If there's no such option, don't worry — see the troubleshooting note
   below.

**When it's installed, verify it.**  Press `Win`, type `cmd`, press Enter to
open a Command Prompt, then type:

```
python --version
```

You want to see something like `Python 3.12.4`.

> **If you instead get "Python was not found"** — Python is installed but
> Windows doesn't know where to find it ("not on PATH").  Try `py --version`
> instead; the `py` launcher usually works even when `python` doesn't.  If
> `py` works, you're fine — just use `py` in place of `python` everywhere in
> this guide.  If neither works, go back to the software request and ask IT
> to add Python to the system PATH.

---

## 2. Install Git

Git, unlike Python, is generally not blocked — you can install it yourself.

1. Go to **https://git-scm.com/download/win**.  The download starts
   automatically.
2. Run the installer.
3. **Accept every default.**  There are a lot of screens and none of the
   choices matter for our purposes.  Click Next until you reach Install.
   - The one screen worth a glance is "Choosing the default editor."  The
     default (Vim) is awkward if you ever land in it by accident; if
     Notepad is in the dropdown, pick it.  This is optional.
4. If the installer asks for admin credentials and you don't have them,
   request **Git for Windows** through Xcel Software Center the same way you
   did for Python.

**Verify it.**  Open a *new* Command Prompt (existing windows won't see the
new install) and type:

```
git --version
```

You want `git version 2.x.x`.

---

## 3. Set up your GitHub sign-in

The PQ Analyzer repo is public, so **downloading and updating it requires no
account at all**.  If all you want is to run the tool and get fixes, skip to
section 4 — you're done here.

You only need this section if you'll be sending changes back (see section 8).

**Create the account:** go to https://github.com and sign up.  Use whatever
email you prefer; a GitHub account is yours, not Xcel's.

**Tell git who you are.**  This stamps your name on any commit you make.  In
a Command Prompt, run these two lines, substituting your own details:

```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**Signing in.**  You don't set a password anywhere.  The first time you do
something that requires authentication, Git Credential Manager (installed
alongside Git in section 2) pops open a browser window, you sign in to
GitHub there, and it remembers you from then on.  Nothing else to configure.

---

## 4. Get the code (`git clone`)

`git clone` downloads the repo and sets up the connection to origin in one
step.  This replaces "someone sends me the files."

**Pick where it lives.**  Your Documents folder is a good choice.  Avoid
OneDrive-synced folders if you can — OneDrive and git both want to manage
the same files and occasionally fight over them.

Open a Command Prompt and run:

```
cd %USERPROFILE%\Documents
git clone https://github.com/jacob-wheat-acre/pq-analyzer.git
```

That creates `Documents\pq-analyzer` containing everything.  You only ever
do this once per PC.

**To get into the folder afterward**, either:

```
cd %USERPROFILE%\Documents\pq-analyzer
```

or, in File Explorer, right-click inside the folder and choose **Open in
Terminal** (Windows 11) or Shift+right-click → **Open PowerShell window
here** (Windows 10).  Every git command in this guide has to be run from
inside the folder.

### If you already have a copy that didn't come from `git clone`

If someone handed you the files, that folder isn't tracked by git and `git
pull` won't work in it.  Convert it once:

1. Move any reports you care about out of it.  Everything the tool generates
   lands in `pq_output\` — copy that folder somewhere safe if you need it.
2. Rename the old folder to `pq-analyzer-old` (don't delete it yet).
3. Run the `git clone` command above to get a fresh, tracked copy.
4. Re-run `install_shortcut.bat` in the new folder so the Desktop shortcut
   points at the right place.
5. Once the new copy runs correctly, delete `pq-analyzer-old`.

---

## 5. Install the Python dependencies

PQ Analyzer relies on five outside libraries (pandas, numpy, matplotlib,
python-docx, Pillow).  From inside the `pq-analyzer` folder:

```
pip install -r pq_analyzer_requirements.txt
```

One time only — unless an update changes the list, which is why section 6
tells you to re-run it after pulling.

> **If pip fails with an SSL, certificate, or proxy error**, that's Xcel's
> network intercepting the connection to the package server, not a problem
> with your machine.  Don't paste in workarounds you find online that
> disable certificate checking.  Open a ticket asking for access to
> **pypi.org** and **files.pythonhosted.org** for Python package installs,
> or ask whether there's an internal package mirror to point pip at.

**Launch it** by double-clicking `PQ Analyzer.bat` in the folder, or run
`install_shortcut.bat` once to get a Desktop shortcut.

---

## 6. Getting updates (`git pull`)

This is the payoff, and it's one command.  Open a Command Prompt in the
`pq-analyzer` folder:

```
git pull
```

Typical output:

```
Updating 16cc2fb..a3f19c2
Fast-forward
 pq_report.py | 24 +++++++++++++-----------
 pq_adapter.py |  6 +++---
 2 files changed, 17 insertions(+), 13 deletions(-)
```

"Fast-forward" means it went cleanly.  If it says `Already up to date.`,
you had the latest version already — that's a success, not an error.

Then, because the dependency list occasionally changes:

```
pip install -r pq_analyzer_requirements.txt
```

**When to pull:**

- Any time you're told a bug you reported is fixed.
- Before starting a batch of reports you care about.
- Otherwise, every couple of weeks is plenty.

**Your reports are safe.**  Everything in `pq_output\` is deliberately
excluded from git — pulling never touches, overwrites, or uploads your
generated reports or the site data in them.

**To see what changed** in the update you just pulled:

```
git log --oneline -10
```

That prints the last ten commit messages, newest first.  Press `q` to get
out if it opens a scrolling view.

---

## 7. When `git pull` complains

Two messages come up in practice.

### "Your local changes to the following files would be overwritten by merge"

You edited a tracked file — possibly by accident, possibly by opening a
`.py` file and saving it.  Git won't discard your edit without being told.

**See what you changed:**

```
git status
```

**If you didn't mean to change anything** (the usual case), throw the local
edits away and pull:

```
git checkout -- .
git pull
```

`git checkout -- .` discards *all* uncommitted edits to tracked files.  It
does not touch `pq_output\` or anything else git ignores.

**If you did mean to change something** and want to keep it, don't run that.
Copy the edited file somewhere outside the folder first, then run the two
commands above, then reconcile by hand — or just send the file to the
maintainer and let them fold it in properly.

### "fatal: not a git repository"

You're either in the wrong folder, or that folder never came from `git
clone`.  Check where you are with `cd` (typed alone, it prints the current
folder).  If you're in the right place and still get this, see "If you
already have a copy that didn't come from `git clone`" in section 4.

### Anything else

Copy the full error text and send it to the maintainer.  Git errors are
wordy but specific, and the exact wording is what makes them diagnosable —
a screenshot of the whole window is fine.

---

## 8. For the maintainer: pushing changes

Right now Jacob is the only person who pushes; everyone else pulls.  This
section is the maintainer's loop.

**See what you've changed:**

```
git status              # which files are modified
git diff                # the actual line-by-line changes
```

**Save and publish a change:**

```
git add -A
git commit -m "Short description of what changed and why"
git push
```

- `git add -A` stages every modified file.  To stage selectively, name them:
  `git add pq_report.py pq_constants.py`.
- The commit message is what teammates see in `git log`. Describe the effect
  ("Fix the ISC lookup so the TDD limit isn't defaulted"), not the mechanics
  ("update file").
- `git push` sends it to GitHub.  From that moment, anyone's `git pull`
  picks it up — so push fixes, not experiments.

**Before pushing anything the team will pull:**

1. Run the test suite: `python -m pytest test_pq.py`
2. Bump `__version__` in `pq_constants.py` if the change is meaningful.
3. Update `pq_analyzer_requirements.txt` if you added a library — otherwise
   the team pulls code that immediately fails on their machines.

**If `git push` is rejected** with "updates were rejected because the remote
contains work that you do not have locally," someone else pushed since your
last pull.  Run `git pull`, resolve anything it flags, then push again.

### When collaborators join

The moment a second person is pushing, switch to branches and pull requests
so nothing lands on `main` unreviewed — everyone pulls `main` directly, so a
broken commit there breaks the whole team at once.  Revisit this section
then; it's a different set of instructions, and there's no point teaching it
before it's needed.

---

## 9. Glossary

| Term | Means |
|---|---|
| **repository / repo** | A folder whose full history git tracks. |
| **clone** | Download a repo for the first time, history and all. Once per PC. |
| **origin** | Your PC's nickname for the GitHub copy of the repo. |
| **pull** | Download and apply changes from origin. The update command. |
| **push** | Upload your commits to origin. Maintainer only. |
| **commit** | One saved snapshot with a description. |
| **staging / `git add`** | Marking which changes go into the next commit. |
| **`main`** | The primary line of history. The only branch this project uses. |
| **fast-forward** | A clean update with nothing to reconcile. What you want to see. |
| **merge conflict** | Two people changed the same lines; git needs a human to choose. Rare here. |
| **`.gitignore`** | The list of things git deliberately ignores — for us, `pq_output\`, logs, and caches. |
| **PATH** | The list of folders Windows searches for programs. Python must be on it. |

---

## 10. Cheat sheet

Print this part.  Every command runs from inside the `pq-analyzer` folder.

```
git pull                              Get the latest version
pip install -r pq_analyzer_requirements.txt
                                      Re-install dependencies after a pull
git log --oneline -10                 What changed recently
git status                            Have I modified anything?
git checkout -- .                     Throw away my accidental edits
cd %USERPROFILE%\Documents\pq-analyzer
                                      Get to the folder
```

Maintainer only:

```
git diff                              Review my changes
git add -A                            Stage everything
git commit -m "message"               Save a snapshot
git push                              Publish to the team
```

One-time setup, in order: request Python → install Git → `git clone` →
`pip install -r pq_analyzer_requirements.txt` → `install_shortcut.bat`.

---

## 11. Mac

The steps are identical; the commands differ slightly.

**Python** — usually already present.  Check with `python3 --version` in
Terminal.  If it's missing or older than 3.9, install from python.org or via
Homebrew (`brew install python`).

**Git** — ships with the Xcode Command Line Tools.  Running any git command
prompts you to install them, or trigger it directly:

```
xcode-select --install
```

**Clone:**

```
cd ~/Documents
git clone https://github.com/jacob-wheat-acre/pq-analyzer.git
cd pq-analyzer
pip3 install -r pq_analyzer_requirements.txt
```

**Run:** `python3 run.py`

**Update:** `git pull`, then `pip3 install -r pq_analyzer_requirements.txt`

Use `pip3` and `python3` throughout — plain `pip` and `python` may point at
an old system Python.  To open Terminal in a folder, right-click the folder
in Finder → Services → **New Terminal at Folder**.  To get a `.pqd` file's
path, drag the file into the Terminal window.

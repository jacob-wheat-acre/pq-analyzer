#!/usr/bin/env python3
"""
run.py — PQ Analyzer GUI Launcher
==================================
Double-click this file (or run: python3 run.py) to open the PQ Analyzer.
No command-line flags required.
"""

import platform
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

_SCRIPT = Path(__file__).parent / "pq_analyzer.py"

# ── Platform-appropriate fonts ────────────────────────────────────────────────
_IS_WIN  = platform.system() == "Windows"
_FONT_UI   = ("Segoe UI",    11) if _IS_WIN else ("Helvetica",   11)
_FONT_UI_B = ("Segoe UI",    13, "bold") if _IS_WIN else ("Helvetica", 13, "bold")
_FONT_UI_S = ("Segoe UI",     9) if _IS_WIN else ("Helvetica",    9)
_FONT_MONO = ("Consolas",    10) if _IS_WIN else ("Menlo",        10)
_FONT_MONO_B = ("Consolas",  10, "bold") if _IS_WIN else ("Menlo", 10, "bold")

# ── Colors ───────────────────────────────────────────────────────────────────
_BG        = "#f5f5f5"
_BTN_RUN   = "#1a6fbf"
_BTN_TXT   = "#ffffff"
_LOG_BG    = "#1e1e1e"
_LOG_FG    = "#d4d4d4"
_LOG_ERR   = "#f48771"
_LOG_INFO  = "#4ec9b0"
_LABEL_FG  = "#333333"
# Text fields state their own colours so the desktop appearance cannot invert
# them; see PQApp._force_light_entries.
_ENTRY_BG           = "#ffffff"
_ENTRY_FG           = "#1a1a1a"
_ENTRY_DISABLED_BG  = "#ebebeb"
_ENTRY_DISABLED_FG  = "#9a9a9a"
_FIELD_TRIM         = "#c9c9c9"   # field borders, separators, tab edges
_ISC_FG    = "#1a6fbf"   # blue for auto-populated ISC
_ISC_NONE  = "#888888"   # grey when no ISC resolved

# ── PSCo tariff schedule → CLI key mapping ───────────────────────────────────
_SCHEDULE_KEY = {
    "Schedule R — Residential":               "r",
    "Schedule C — Small Commercial  (< 50 kW)": "c",
    "Schedule SG — C&I Secondary  (≥ 50 kW)":  "sg",
    "Schedule PG — C&I Primary":              "pg",
}

# ── Which way power flows at the meter → CLI key ─────────────────────────────
# Not a tariff schedule: NM and PV ride on top of the schedules above, and the
# solar schedules where the array is off-site (OS-NM, RC, SRCS) measure as
# plain load. What matters here is what is physically behind the meter.
_ROLE_LABELS = [
    "Load only",
    "Load + generation  (NM, PV, RE, AVPP)",
    "Generation only  (producer's array)",
]
_ROLE_KEY = {
    _ROLE_LABELS[0]: "load",
    _ROLE_LABELS[1]: "mixed",
    _ROLE_LABELS[2]: "generation",
}


def _float_or_none(text: str):
    """A blank or unparseable entry box is 'not given', not zero."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None

# ── Which version is running ─────────────────────────────────────────────────
# Read on its own, ahead of the engine import below and out of its try/except:
# when that import fails, the version is the first thing anyone asks for, and a
# copy that cannot say what it is cannot be told apart from a current one.
# pq_constants is pure data with no third-party imports, so it loads even on an
# install where python-docx or matplotlib is missing.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from pq_constants import __version__ as _ENGINE_VERSION
    from pq_constants import ansi_bands, ll_factor
except Exception:
    _ENGINE_VERSION = "unknown"

    def ansi_bands(nominal_v):                       # type: ignore[misc]
        # C84.1 Table 1, over-600 V group, since this fallback is only ever
        # reached by the primary-voltage hint.
        return {"a_min": nominal_v * 0.975, "a_max": nominal_v * 1.05}

    def ll_factor(service_type=None, topology="auto"):   # type: ignore[misc]
        return 2.0 if topology == "split-phase" else 3.0 ** 0.5

# ── Transformer / Blue Book data ──────────────────────────────────────────────
# Import the same lookup tables used by the analysis engine so the GUI always
# stays in sync with what pq_analyzer.py will actually compute.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from pq_analyzer import (
        _BLUE_BOOK_ISC,
        _SERVICE_TYPE_LABEL,
        isc_lookup_type,
        _infer_secondary_v,
        _lookup_isc,
        conductor_options,
        Thresholds,
        ProntoAdapter,
        PQDIFAdapter,
        ChannelMapper,
        extract_dataset,
        _PQDIF_AVAILABLE,
        check_voltage_compliance,
        check_thd,
        check_power_factor,
        check_voltage_imbalance,
        check_current_imbalance,
        check_demand,
        check_individual_harmonics,
        check_individual_voltage_harmonics,
        check_neutral_harmonics,
        check_harmonic_direction,
        check_source_impedance,
        check_harmonic_sources,
        check_spectral_shape,
        check_harmonic_statistics,
        detect_events,
        analyze_root_causes,
        generate_report,
        export_results,
        generate_word_report,
        plot_overview,
        plot_voltage,
        plot_thd,
        plot_summary,
        plot_harmonic_spectrum,
        plot_itic,
        plot_neutral_health,
        plot_demand_profile,
        plot_harmonic_trend,
        plot_imbalance,
        plot_flicker,
    plot_pf_load,
        plot_waveform_capture,
        check_neutral_health,
        check_itic,
        check_line_to_line_voltage,
        check_frequency,
        check_flicker,
        kfactor_by_phase,
        generate_customer_letter,
    )
    from pq_report import _LETTER_CLASSES
    _BOOK_AVAILABLE = True
    _IMPORT_TRACEBACK = ""
except Exception as _import_exc:
    import traceback
    # Held in memory as well as on disk. The window that shows this is the
    # only place the user can see why the tool will not run, and a file is
    # the part of that chain most likely to be missing -- deleted, never
    # written because the install directory is read-only, or written next to
    # a different copy of run.py.
    _IMPORT_TRACEBACK = traceback.format_exc()
    try:
        (Path(__file__).parent / "import_error.log").write_text(_IMPORT_TRACEBACK)
    except Exception:
        pass
    _BLUE_BOOK_ISC = {}
    _SERVICE_TYPE_LABEL = {}
    _LETTER_CLASSES = {}
    _BOOK_AVAILABLE = False

# Display labels for the type picker (ordered for the dropdown)
_TYPE_ORDER = [
    "1ph-overhead",
    "1ph-padmount",
    # Two legs of a three-phase 120/208 transformer — condos and apartments.
    # Sits with the single-phase entries because that is what the customer has;
    # the transformer is the same one a three-phase 120/208 service uses.
    "1ph-208",
    "3ph-padmount",
    "3ph-overhead-wye",
    "3ph-open-delta",
    "3ph-closed-delta",
]
_TYPE_DISPLAY = {k: _SERVICE_TYPE_LABEL.get(k, k) for k in _TYPE_ORDER}
# Sentinel for primary-metered services (no Blue Book kVA/ISC lookup)
_PRIMARY_KEY    = "__primary__"
_PRIMARY_LABEL  = "Primary metered"
# Sentinel for the service-conductor picker
_CONDUCTOR_NONE = "— not specified —"

#: Primary distribution nominals offered to a primary-metered service, from the
#: over-600 V half of ANSI C84.1 Table 1. The combobox is editable rather than
#: readonly: this is a starting list, not the set of voltages that exist, and an
#: engineer metering a service on something not listed must still be able to say
#: so rather than pick the nearest wrong one.
_PRIMARY_LL_CHOICES = [2400, 4160, 4800, 12470, 13200, 13800, 24940, 34500]

#: Variables the form derives rather than accepts: the ISC hint label is
#: rewritten by the transformer cascade, and the details panel's open state is
#: how the window looks, not an entry the run reads. Clearing either would
#: either be undone immediately or would fold a panel the user just opened.
_DERIVED_VARS = {"_isc_auto_var", "_details_open"}

#: Entries that describe the engineer rather than the service. They are the
#: same on every run this person makes, so Clear All leaves them: clearing
#: between sites would mean retyping one's own name and email each time, and
#: a field left blank by accident goes out on a customer document.
_STICKY_VARS = {"_eng_name_var", "_eng_title_var",
                "_eng_email_var"}


def _resolve_secondary_v(svc_type: str, nominal_v: float) -> int:
    """Convert nominal L-N voltage to the secondary (L-L) voltage used as a Blue Book key.

    Thin wrapper over the shared resolver so the label and the analysis run read
    the same Blue Book row.
    """
    try:
        return _infer_secondary_v(isc_lookup_type(svc_type), nominal_v)
    except Exception:
        return 240


def _kva_options(svc_type: str, nominal_v: float) -> list:
    """Return sorted list of kVA sizes available in the Blue Book for this type/voltage."""
    if svc_type == _PRIMARY_KEY or not svc_type:
        return []
    # A single-phase 120/208 service reads the three-phase rows -- same
    # transformer, fewer wires -- so filtering on its own key would leave the
    # Size picker permanently empty.
    lookup = isc_lookup_type(svc_type)
    sec_v = _resolve_secondary_v(svc_type, nominal_v)
    return sorted({k[1] for k in _BLUE_BOOK_ISC if k[0] == lookup and k[2] == sec_v})


def _isc_for(svc_type: str, kva: int, nominal_v: float):
    """Return (isc_amps, note) or (None, '') if not found."""
    if svc_type == _PRIMARY_KEY or not svc_type or not kva:
        return None, ""
    sec_v = _resolve_secondary_v(svc_type, nominal_v)
    isc = _BLUE_BOOK_ISC.get((svc_type, int(kva), sec_v))
    if isc is None:
        return None, ""
    label = _SERVICE_TYPE_LABEL.get(svc_type, svc_type)
    note = f"Blue Book — {label}, {sec_v} V secondary"
    return isc, note


import logging as _logging


class _GUILogHandler(_logging.Handler):
    """Logging handler that routes records into the GUI log widget."""

    def __init__(self, log_widget, after_fn):
        super().__init__()
        self._widget  = log_widget
        self._after   = after_fn

    def emit(self, record):
        msg = self.format(record) + "\n"
        tag = "error" if record.levelno >= _logging.WARNING else "info"
        self._after(0, lambda m=msg, t=tag: self._write(m, t))

    def _write(self, msg, tag):
        self._widget.config(state="normal")
        self._widget.insert("end", msg, tag)
        self._widget.see("end")
        self._widget.config(state="disabled")


class PQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"PQ Analyzer v{_ENGINE_VERSION}")
        self.resizable(True, True)
        self.configure(bg=_BG)
        self.minsize(680, 520)
        self._force_light_entries()
        self._set_icon()
        self._build_ui()
        self._running = False

    def _force_light_entries(self):
        """Keep text fields light whatever the desktop appearance is set to.

        Every frame and label here names its own colour, but a bare tk.Entry
        does not, so on a Mac in dark mode the fields came back with a black
        background inside an otherwise light window -- unreadable, and looking
        like a rendering fault rather than a theme.  There is no dark mode to
        support here: one appearance, stated explicitly.

        Set as an option database default rather than on each widget so a field
        added later cannot quietly reintroduce it.  The log pane is deliberately
        dark and names its own colours, so it is unaffected.
        """
        for option, value in (
            ("background",           _ENTRY_BG),
            ("foreground",           _ENTRY_FG),
            ("insertBackground",     _ENTRY_FG),   # the caret
            ("readonlyBackground",   _ENTRY_BG),
            ("disabledBackground",   _ENTRY_DISABLED_BG),
            ("disabledForeground",   _ENTRY_DISABLED_FG),
            ("highlightBackground",  _BG),         # focus ring surround
        ):
            self.option_add(f"*Entry.{option}", value)

        # The combobox dropdown is a classic Listbox behind a ttk widget, so it
        # takes options rather than styles. Left alone it opens dark on a dark
        # desktop while the closed widget above it is light.
        for option, value in (
            ("background",       _ENTRY_BG),
            ("foreground",       _ENTRY_FG),
            ("selectBackground", _BTN_RUN),
            ("selectForeground", _BTN_TXT),
        ):
            self.option_add(f"*TCombobox*Listbox.{option}", value)

        # The ttk widgets need the theme changed, not their options set. Both
        # the macOS "aqua" theme and the Windows native ones draw their own
        # controls and ignore colour options entirely, taking the desktop
        # appearance instead -- which is why the comboboxes stayed dark while
        # the plain entries above went light. "clam" is the one bundled theme
        # that honours colours on every platform, so it is the only way to pin
        # a single look. It costs the native control shape; the alternative is
        # a form whose fields are unreadable half the time.
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:                       # pragma: no cover - platform
            log.debug("clam theme unavailable; leaving the platform default")

        style.configure(".", background=_BG, foreground=_LABEL_FG,
                        fieldbackground=_ENTRY_BG, font=_FONT_UI)
        style.configure("TCombobox",
                        fieldbackground=_ENTRY_BG, background=_FIELD_TRIM,
                        foreground=_ENTRY_FG, arrowcolor=_LABEL_FG,
                        bordercolor=_FIELD_TRIM, lightcolor=_FIELD_TRIM,
                        darkcolor=_FIELD_TRIM, insertcolor=_ENTRY_FG,
                        padding=3)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", _ENTRY_BG),
                             ("disabled", _ENTRY_DISABLED_BG)],
            foreground=[("readonly", _ENTRY_FG),
                        ("disabled", _ENTRY_DISABLED_FG)],
            arrowcolor=[("disabled", _ENTRY_DISABLED_FG)],
            # Without these the readonly field paints itself with the selection
            # colour the moment it takes focus, which reads as a stuck highlight.
            selectbackground=[("readonly", _ENTRY_BG)],
            selectforeground=[("readonly", _ENTRY_FG)],
        )
        style.configure("TSeparator", background=_FIELD_TRIM)
        style.configure("TScrollbar", background=_BG, troughcolor="#ececec",
                        bordercolor=_BG, arrowcolor=_LABEL_FG,
                        lightcolor=_BG, darkcolor=_BG)
        style.configure("TNotebook", background=_BG, bordercolor=_FIELD_TRIM)
        style.configure("TNotebook.Tab", background="#e4e4e4",
                        foreground=_LABEL_FG, padding=(12, 6),
                        lightcolor="#e4e4e4", bordercolor=_FIELD_TRIM)
        style.map("TNotebook.Tab",
                  background=[("selected", _BG)],
                  foreground=[("selected", _BTN_RUN)],
                  expand=[("selected", (0, 0, 0, 0))])

        # Buttons carry colour, so they have to leave the native theme too --
        # a tk.Button under aqua ignores -background the same way.
        style.configure("Plain.TButton", background=_BG, foreground=_LABEL_FG,
                        bordercolor=_FIELD_TRIM, lightcolor=_BG, darkcolor=_BG,
                        focuscolor=_BG, padding=(12, 7), relief="flat")
        style.map("Plain.TButton",
                  background=[("pressed", "#e0e0e0"), ("active", "#ececec"),
                              ("disabled", _BG)],
                  foreground=[("disabled", _ENTRY_DISABLED_FG)])
        style.configure("Quiet.TButton", background=_BG, foreground="#555555",
                        bordercolor=_BG, lightcolor=_BG, darkcolor=_BG,
                        focuscolor=_BG, padding=(12, 7), relief="flat")
        style.map("Quiet.TButton",
                  background=[("pressed", "#e0e0e0"), ("active", "#ececec")])
        # The collapsible section header: a full-width strip of text with a
        # disclosure arrow, so it is left-aligned and carries no border.
        style.configure("Toggle.TButton", background=_BG, foreground="#555555",
                        bordercolor=_BG, lightcolor=_BG, darkcolor=_BG,
                        focuscolor=_BG, relief="flat", anchor="w",
                        padding=(4, 4), font=_FONT_UI_S)
        style.map("Toggle.TButton",
                  background=[("pressed", "#e6e6e6"), ("active", "#ededed")],
                  foreground=[("active", _LABEL_FG)])
        style.configure("Run.TButton", background=_BTN_RUN, foreground=_BTN_TXT,
                        bordercolor=_BTN_RUN, lightcolor=_BTN_RUN,
                        darkcolor=_BTN_RUN, focuscolor=_BTN_RUN,
                        padding=(20, 8), relief="flat", font=_FONT_UI_B)
        style.map("Run.TButton",
                  background=[("pressed", "#12508e"), ("active", "#155a9e"),
                              ("disabled", "#a8c4e2")],
                  foreground=[("disabled", "#f0f0f0")])

    def _set_icon(self):
        """Title bar, taskbar and Alt-Tab icon.

        Windows wants two separate things here and reports neither when it does
        not get them, which is why this went unnoticed while macOS looked fine:

          * a multi-size DIB .ico — a lone 16 px entry leaves the 32 px taskbar
            and 48 px desktop icon with nothing to draw, and Windows falls back
            to the host interpreter's icon rather than erroring;
          * an explicit AppUserModelID — the taskbar groups a window by that ID,
            and a script run under pythonw.exe inherits Python's own, so the
            taskbar draws the Python logo however the window is decorated.

        Failures are logged rather than swallowed. The icon is cosmetic and must
        never block startup, but "cosmetic" is not "invisible": the last time
        this broke, nothing anywhere said so, and this tool is used on machines
        whose files cannot be sent back for diagnosis.
        """
        log = _logging.getLogger(__name__)
        icon_dir = Path(__file__).parent

        if sys.platform == "win32":
            ico = icon_dir / "icon.ico"
            if ico.exists():
                try:
                    # default= covers this window and every dialog opened from
                    # it; without it the message boxes revert to the Tk feather.
                    self.iconbitmap(default=str(ico))
                    return
                except Exception as exc:
                    log.warning("Could not load %s (%s); falling back to the "
                                "PNG icon.", ico.name, exc)
            else:
                log.warning("%s is missing — run make_icon.py to rebuild it.",
                            ico)

        # PNG path: macOS and Linux always, Windows only if the .ico would not
        # load. Tk 8.6 reads PNG natively, so this does not require Pillow.
        png = icon_dir / "icon.png"
        if not png.exists():
            log.warning("%s is missing — the window will use the default icon.", png)
            return
        try:
            self._tk_icon = tk.PhotoImage(file=str(png))
            self.iconphoto(True, self._tk_icon)
        except Exception as exc:
            log.warning("Could not set the window icon from %s: %s", png, exc)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # ── Broken-install banner ─────────────────────────────────────────────
        # If the pq_analyzer import failed, the Blue Book tables are empty and
        # every transformer type will report "no entries".  Say so here rather
        # than letting the ISC lookup look merely unlucky.
        if not _BOOK_AVAILABLE:
            self._build_import_error_banner()

        # ── File row ──────────────────────────────────────────────────────────
        file_frame = self._file_frame = tk.Frame(self, bg=_BG)
        file_frame.pack(fill="x", **pad)

        tk.Label(file_frame, text="PQD File", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._file_var = tk.StringVar()
        # A path is as often pasted in as browsed to, and the session picker
        # has to appear either way. Debounced, so typing does not scan once
        # per keystroke.
        self._file_var.trace_add("write", self._on_file_changed)
        self._scan_after_id = None
        tk.Entry(file_frame, textvariable=self._file_var, font=_FONT_UI,
                 width=40).pack(side="left", padx=(0, 6), fill="x", expand=True)
        ttk.Button(file_frame, text="Browse…", command=self._browse,
                   style="Plain.TButton").pack(side="left")

        # ── Session row ───────────────────────────────────────────────────────
        # A "download all data" export holds every session the meter still had.
        # Only one is analysed per run, so which one has to be the engineer's
        # choice rather than ours; the row stays hidden for the ordinary
        # single-session file so it is not one more thing to read past.
        self._session_frame = tk.Frame(self, bg=_BG)
        tk.Label(self._session_frame, text="Session", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._session_var = tk.StringVar()
        self._session_combo = ttk.Combobox(
            self._session_frame, textvariable=self._session_var,
            values=[], width=48, font=_FONT_UI, state="readonly",
        )
        self._session_combo.pack(side="left")
        self._session_note = tk.Label(
            self._session_frame, text="", bg=_BG, fg="#c08a3e", font=_FONT_UI_S)
        self._session_note.pack(side="left", padx=(6, 0))
        self._sessions: list = []

        # ── Customer name row ─────────────────────────────────────────────────
        site_frame = tk.Frame(self, bg=_BG)
        site_frame.pack(fill="x", **pad)

        tk.Label(site_frame, text="Customer", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._site_var = tk.StringVar()
        tk.Entry(site_frame, textvariable=self._site_var, font=_FONT_UI,
                 width=40).pack(side="left", fill="x", expand=True)
        tk.Label(site_frame, text="(e.g. Walmart Store 20)", bg=_BG, fg="#888888",
                 font=_FONT_UI_S).pack(side="left", padx=(6, 0))

        # ── Address row (auto-loads from filename) ────────────────────────────
        addr_frame = tk.Frame(self, bg=_BG)
        addr_frame.pack(fill="x", **pad)

        tk.Label(addr_frame, text="Address", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._address_var = tk.StringVar()
        tk.Entry(addr_frame, textvariable=self._address_var, font=_FONT_UI,
                 width=40).pack(side="left", fill="x", expand=True)
        tk.Label(addr_frame, text="(auto-filled from filename)", bg=_BG, fg="#888888",
                 font=_FONT_UI_S).pack(side="left", padx=(6, 0))

        # ── Customer class row ────────────────────────────────────────────────
        cclass_frame = tk.Frame(self, bg=_BG)
        cclass_frame.pack(fill="x", **pad)

        tk.Label(cclass_frame, text="Customer Class", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._cclass_var = tk.StringVar(value="Schedule SG — C&I Secondary  (≥ 50 kW)")
        cclass_combo = ttk.Combobox(
            cclass_frame, textvariable=self._cclass_var,
            values=[
                "Schedule R — Residential",
                "Schedule C — Small Commercial  (< 50 kW)",
                "Schedule SG — C&I Secondary  (≥ 50 kW)",
                "Schedule PG — C&I Primary",
            ],
            width=34, font=_FONT_UI, state="readonly",
        )
        cclass_combo.pack(side="left")
        tk.Label(cclass_frame,
                 text="(PF: Sheet R73 ≥ 0.90 lagging, all classes  |  Sheet R121 near unity, C&I only)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(8, 0))

        # Generation rides on top of the class above rather than replacing it
        # -- Schedule NM applies "as a service element under all rate
        # schedules" -- so it is its own row, not a fifth entry in the list.
        # Three values rather than a checkbox: a plant that only generates is
        # not a further degree of a service that also generates, and the CT
        # polarity check treats them oppositely.
        nm_frame = tk.Frame(self, bg=_BG)
        nm_frame.pack(fill="x", **pad)
        tk.Label(nm_frame, text="Power flow", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._role_var = tk.StringVar(value=_ROLE_LABELS[0])
        role_combo = ttk.Combobox(
            nm_frame, textvariable=self._role_var, values=_ROLE_LABELS,
            width=34, font=_FONT_UI, state="readonly",
        )
        role_combo.pack(side="left")

        tk.Label(nm_frame, text="Rated gen kW", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(12, 4))
        self._rated_kw_var = tk.StringVar(value="")
        rated_entry = tk.Entry(nm_frame, textvariable=self._rated_kw_var, width=8,
                               font=_FONT_UI)
        rated_entry.pack(side="left")

        tk.Label(nm_frame, text="Avg load kW", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(10, 4))
        self._avg_load_var = tk.StringVar(value="")
        avg_entry = tk.Entry(nm_frame, textvariable=self._avg_load_var, width=8,
                             font=_FONT_UI)
        avg_entry.pack(side="left")

        rated_hint = tk.Label(
            nm_frame,
            text="(the two decide 519 vs 1547 — see ? Help)",
            bg=_BG, fg="#888888", font=_FONT_UI_S)
        rated_hint.pack(side="left", padx=(8, 0))

        def _sync_rated(*_a):
            # Both only mean anything where there is generation, so they are
            # greyed out rather than silently ignored.
            on = _ROLE_KEY.get(self._role_var.get(), "load") != "load"
            state = "normal" if on else "disabled"
            rated_entry.configure(state=state)
            avg_entry.configure(state=state)
            rated_hint.configure(fg="#888888" if on else "#555555")

        role_combo.bind("<<ComboboxSelected>>", _sync_rated)
        _sync_rated()

        # ── Billing IL row ────────────────────────────────────────────────────
        il_frame = tk.Frame(self, bg=_BG)
        il_frame.pack(fill="x", **pad)
        tk.Label(il_frame, text="IL from billing", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._il_billing_var = tk.StringVar(value="")
        tk.Entry(il_frame, textvariable=self._il_billing_var, width=10,
                 font=_FONT_UI).pack(side="left")
        tk.Label(il_frame, text="A",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left", padx=(4, 0))
        tk.Label(il_frame,
                 text="(12 months of 15/30-min maximum demands, averaged — "
                      "IEEE 519's own definition; blank uses the recording peak)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(8, 0))

        # ── Service type + nominal row ─────────────────────────────────────────
        svc_frame = tk.Frame(self, bg=_BG)
        svc_frame.pack(fill="x", **pad)

        tk.Label(svc_frame, text="Service Type", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._topo_var = tk.StringVar(value="auto")
        topo_combo = ttk.Combobox(
            svc_frame, textvariable=self._topo_var, state="readonly",
            values=["auto", "3ph-wye", "split-phase"],
            width=16, font=_FONT_UI,
        )
        topo_combo.pack(side="left")
        self._topo_hint = tk.Label(svc_frame, text="(auto-detected)", bg=_BG,
                                    fg="#888888", font=_FONT_UI_S)
        self._topo_hint.pack(side="left", padx=(8, 0))
        topo_combo.bind("<<ComboboxSelected>>", self._on_topo_change)

        # ── Nominal voltage row ────────────────────────────────────────────────
        nom_frame = tk.Frame(self, bg=_BG)
        nom_frame.pack(fill="x", **pad)

        tk.Label(nom_frame, text="Nominal Voltage", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._nominal_var = tk.StringVar(value="120")
        nom_combo = ttk.Combobox(nom_frame, textvariable=self._nominal_var,
                                  values=["120", "208", "240", "277", "480"],
                                  width=7, font=_FONT_UI)
        nom_combo.pack(side="left")
        tk.Label(nom_frame, text="V", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(2, 0))
        tk.Label(nom_frame, text="(120/240 V split-phase  or  120/208 V three-phase wye → pick 120)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(8, 0))
        nom_combo.bind("<<ComboboxSelected>>", self._on_nominal_change)
        nom_combo.bind("<FocusOut>",            self._on_nominal_change)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=12, pady=(4, 0))

        # ── Transformer section label ─────────────────────────────────────────
        xfmr_hdr = tk.Frame(self, bg=_BG)
        xfmr_hdr.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(xfmr_hdr, text="Transformer (optional — enables Blue Book ISC lookup)",
                 bg=_BG, fg="#555555", font=_FONT_UI_S).pack(side="left")

        # ── Transformer type row ───────────────────────────────────────────────
        xtype_frame = tk.Frame(self, bg=_BG)
        xtype_frame.pack(fill="x", padx=12, pady=(0, 4))

        tk.Label(xtype_frame, text="Type", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        type_values = (
            ["— not specified —"]
            + [_TYPE_DISPLAY[k] for k in _TYPE_ORDER]
            + [_PRIMARY_LABEL]
        )
        self._xfmr_type_key = None   # internal key (e.g. "3ph-padmount")
        self._xfmr_type_var = tk.StringVar(value="— not specified —")
        self._type_combo = ttk.Combobox(
            xtype_frame, textvariable=self._xfmr_type_var, state="readonly",
            values=type_values, width=32, font=_FONT_UI,
        )
        self._type_combo.pack(side="left")
        self._type_combo.bind("<<ComboboxSelected>>", self._on_type_change)

        # ── kVA + ISC row ──────────────────────────────────────────────────────
        kva_frame = tk.Frame(self, bg=_BG)
        kva_frame.pack(fill="x", padx=12, pady=(0, 2))

        tk.Label(kva_frame, text="Size", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        self._kva_var = tk.StringVar(value="")
        self._kva_combo = ttk.Combobox(
            kva_frame, textvariable=self._kva_var, state="disabled",
            values=[], width=10, font=_FONT_UI,
        )
        self._kva_combo.pack(side="left")
        tk.Label(kva_frame, text="kVA", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(3, 16))
        self._kva_combo.bind("<<ComboboxSelected>>", self._on_kva_change)

        # ISC auto-label — seeded with the same hint _refresh_kva_options would
        # show, since that callback hasn't run yet on a fresh window.
        self._isc_auto_var = tk.StringVar(
            value="Pick a transformer Type above to enable Size")
        self._isc_auto_lbl = tk.Label(
            kva_frame, textvariable=self._isc_auto_var,
            bg=_BG, fg=_ISC_NONE, font=_FONT_UI_S, anchor="w",
        )
        self._isc_auto_lbl.pack(side="left", fill="x", expand=True)

        # ── ISC override row ───────────────────────────────────────────────────
        isc_frame = tk.Frame(self, bg=_BG)
        isc_frame.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(isc_frame, text="", width=16, bg=_BG).pack(side="left")
        self._isc_override_var = tk.BooleanVar(value=False)
        self._isc_chk = tk.Checkbutton(
            isc_frame, text="Override ISC:", variable=self._isc_override_var,
            bg=_BG, fg=_LABEL_FG, font=_FONT_UI_S,
            command=self._on_isc_override_toggle,
        )
        self._isc_chk.pack(side="left")

        self._isc_manual_var = tk.StringVar(value="")
        self._isc_entry = tk.Entry(
            isc_frame, textvariable=self._isc_manual_var,
            font=_FONT_UI, width=9, state="disabled",
        )
        self._isc_entry.pack(side="left", padx=(4, 2))
        tk.Label(isc_frame, text="A  (from fault study or manual calculation)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left")

        # ── The path from the source to the meter ──────────────────────────────
        # Which controls belong here depends on where the meter sits. Metered on
        # the secondary, the path is the transformer, whatever shared secondary
        # main the service taps, and the service run. Metered on the primary,
        # the transformer and everything below it are the customer's and sit
        # downstream of the meter, so the path is the primary line instead and
        # asking for a conductor size would be asking about the wrong wire.
        self._path_wrap = tk.Frame(self, bg=_BG)
        self._path_wrap.pack(fill="x")

        self._conductor_labels = {label: key for key, label in conductor_options()}
        cond_values = [_CONDUCTOR_NONE] + list(self._conductor_labels.keys())

        # ── Service conductor row ──────────────────────────────────────────────
        # The run between the transformer (or the shared secondary tap) and the
        # meter. With it the measured service impedance gets an expected value
        # to be compared against; without it the measurement still stands,
        # uncompared.
        self._cond_frame = cond_frame = tk.Frame(self._path_wrap, bg=_BG)
        cond_frame.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(cond_frame, text="Service conductor", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        self._conductor_var = tk.StringVar(value=_CONDUCTOR_NONE)
        self._conductor_combo = ttk.Combobox(
            cond_frame, textvariable=self._conductor_var, state="readonly",
            values=cond_values, width=32, font=_FONT_UI,
        )
        self._conductor_combo.pack(side="left")

        tk.Label(cond_frame, text="Run", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(10, 3))
        self._run_length_var = tk.StringVar(value="")
        tk.Entry(cond_frame, textvariable=self._run_length_var,
                 font=_FONT_UI, width=7).pack(side="left")
        tk.Label(cond_frame, text="ft  (transformer to meter — enables the "
                                  "expected-impedance check)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(3, 0))

        # ── Shared secondary row ───────────────────────────────────────────────
        # Where the transformer does not land at this meter but on a secondary
        # main shared with the neighbours, that main carries this customer's
        # current too and its drop is already in the measurement. Leaving it
        # blank says the service is a dedicated run from the transformer.
        self._shared_frame = shared_frame = tk.Frame(self._path_wrap, bg=_BG)
        shared_frame.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(shared_frame, text="Shared secondary", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        self._shared_var = tk.StringVar(value=_CONDUCTOR_NONE)
        self._shared_combo = ttk.Combobox(
            shared_frame, textvariable=self._shared_var, state="readonly",
            values=cond_values, width=32, font=_FONT_UI,
        )
        self._shared_combo.pack(side="left")

        tk.Label(shared_frame, text="Run", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(10, 3))
        self._shared_length_var = tk.StringVar(value="")
        tk.Entry(shared_frame, textvariable=self._shared_length_var,
                 font=_FONT_UI, width=7).pack(side="left")
        tk.Label(shared_frame, text="ft  (transformer to this service's tap — "
                                    "leave blank for a dedicated run)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(3, 0))

        # ── Primary nominal voltage row ───────────────────────────────────────
        # Entered, not inferred. The L-L/L-N ratio in the file recovers the
        # topology -- whether the legs sit 120 or 180 degrees apart -- but says
        # nothing about the nominal, and PSCo runs several primary voltages. The
        # secondary path can afford to guess and snap to the nearest standard
        # value because there are only a handful in play; on the primary that
        # guess would become the ANSI C84.1 band the customer is judged against.
        self._primary_v_frame = primv_frame = tk.Frame(self._path_wrap, bg=_BG)

        tk.Label(primv_frame, text="Primary voltage", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._primary_ll_var = tk.StringVar(value="")
        primv_combo = ttk.Combobox(
            primv_frame, textvariable=self._primary_ll_var,
            values=[str(v) for v in _PRIMARY_LL_CHOICES],
            width=9, font=_FONT_UI,
        )
        primv_combo.pack(side="left")
        tk.Label(primv_frame, text="V L-L", bg=_BG, fg=_LABEL_FG,
                 font=_FONT_UI).pack(side="left", padx=(3, 0))
        # The L-N figure the ANSI check actually runs on is derived, so it is
        # shown as it is derived: an engineer who reads 7621 V here and expected
        # something else has caught a wrong entry before the report is written.
        self._primary_ln_hint = tk.Label(
            primv_frame, text="(nominal at the metering point — sets the "
                              "ANSI C84.1 band)",
            bg=_BG, fg="#888888", font=_FONT_UI_S)
        self._primary_ln_hint.pack(side="left", padx=(8, 0))
        primv_combo.bind("<<ComboboxSelected>>", self._on_primary_v_change)
        primv_combo.bind("<KeyRelease>",         self._on_primary_v_change)
        primv_combo.bind("<FocusOut>",           self._on_primary_v_change)

        # ── Primary line impedance row ─────────────────────────────────────────
        # Entered, not looked up: a primary line's impedance comes off a
        # planning model or a fault study. Z1 is what balanced load current
        # flows in and is what the comparison uses; Z0 is optional and read
        # only where it is the right number -- triplen harmonics, which are
        # zero-sequence, and unbalanced current returning through earth. Z2 is
        # not asked for because a passive line has Z2 = Z1.
        self._primary_frame = prim_frame = tk.Frame(self._path_wrap, bg=_BG)

        tk.Label(prim_frame, text="Primary line Z", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        # Each entry is also an attribute, not only a dict value: Clear All
        # snapshots the form by walking vars(self) for tk variables, and a
        # field it cannot see is a field that carries the last site's
        # impedance into the next run.
        self._primary_r1_var = tk.StringVar(value="")
        self._primary_x1_var = tk.StringVar(value="")
        self._primary_r0_var = tk.StringVar(value="")
        self._primary_x0_var = tk.StringVar(value="")
        self._primary_vars = {
            "r1": self._primary_r1_var, "x1": self._primary_x1_var,
            "r0": self._primary_r0_var, "x0": self._primary_x0_var,
        }
        for field, label in (("r1", "R1"), ("x1", "X1"),
                             ("r0", "R0"), ("x0", "X0")):
            tk.Label(prim_frame, text=label, bg=_BG, fg=_LABEL_FG,
                     font=_FONT_UI).pack(side="left", padx=(0 if field == "r1" else 8, 3))
            tk.Entry(prim_frame, textvariable=self._primary_vars[field],
                     font=_FONT_UI, width=8).pack(side="left")
        tk.Label(prim_frame, text="Ω  (to the metering point; R1/X1 required, "
                                  "R0/X0 optional)",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(side="left", padx=(6, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=12, pady=(4, 0))

        # ── Report details section (collapsible) ──────────────────────────────
        det_hdr = tk.Frame(self, bg=_BG)
        det_hdr.pack(fill="x", padx=12, pady=(4, 0))

        self._details_open = tk.BooleanVar(value=False)
        self._det_toggle_btn = ttk.Button(
            det_hdr, text="▶  Report Details (engineer sign-off)",
            command=self._toggle_details,
            style="Toggle.TButton", cursor="hand2",
        )
        self._det_toggle_btn.pack(side="left", fill="x", expand=True)

        self._details_frame = tk.Frame(self, bg=_BG)
        # Not packed initially — toggled by button

        _W = 18   # label width inside this section

        def _detail_row(label, var, placeholder=""):
            f = tk.Frame(self._details_frame, bg=_BG)
            f.pack(fill="x", padx=12, pady=2)
            tk.Label(f, text=label, width=_W, anchor="w",
                     bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
            e = tk.Entry(f, textvariable=var, font=_FONT_UI, width=38)
            e.pack(side="left", fill="x", expand=True)
            if placeholder:
                tk.Label(f, text=placeholder, bg=_BG, fg="#888888",
                         font=_FONT_UI_S).pack(side="left", padx=(6, 0))
            return e

        # ── Engineer / sign-off ────────────────────────────────────────────────
        # Meter/account number, feeder and substation used to sit above this.
        # They identify the service to us and mean nothing to the customer, so
        # they were taking room on the form to reach a header nobody read.
        self._eng_name_var  = tk.StringVar()
        self._eng_title_var = tk.StringVar()
        self._eng_email_var = tk.StringVar()

        _detail_row("Name",  self._eng_name_var,  "(e.g. Jacob Whitaker)")
        _detail_row("Title", self._eng_title_var, "(default: Electric Area Engineer)")
        _detail_row("Email", self._eng_email_var, "(e.g. jwhitaker@xcelenergy.com)")

        tk.Frame(self._details_frame, bg=_BG, height=6).pack()  # bottom padding

        # ── Divider + Run button ───────────────────────────────────────────────
        self._sep_before_run = ttk.Separator(self, orient="horizontal")
        self._sep_before_run.pack(fill="x", padx=12, pady=4)

        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill="x", padx=12, pady=4)

        self._run_btn = ttk.Button(
            btn_frame, text="Run Analysis", command=self._run,
            style="Run.TButton", cursor="hand2",
        )
        self._run_btn.pack(side="left")

        self._open_btn = ttk.Button(
            btn_frame, text="Open Output Folder", command=self._open_folder,
            style="Plain.TButton", cursor="hand2",
        )
        self._open_btn.pack(side="left", padx=(12, 0))
        self._open_btn.config(state="disabled")

        ttk.Button(
            btn_frame, text="Clear All", command=self._clear_all,
            style="Plain.TButton", cursor="hand2",
        ).pack(side="left", padx=(8, 0))

        ttk.Button(
            btn_frame, text="? Help", command=self._show_help,
            style="Quiet.TButton", cursor="hand2",
        ).pack(side="right")

        ttk.Button(
            btn_frame, text="✉ Feedback", command=self._show_feedback,
            style="Quiet.TButton", cursor="hand2",
        ).pack(side="right")

        # ── Log window ────────────────────────────────────────────────────────
        log_frame = tk.Frame(self, bg=_BG)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self._log = tk.Text(
            log_frame, bg=_LOG_BG, fg=_LOG_FG,
            font=_FONT_MONO, relief="flat",
            state="disabled", wrap="word",
        )
        self._log.tag_config("info",  foreground=_LOG_INFO)
        self._log.tag_config("error", foreground=_LOG_ERR)
        self._log.tag_config("done",  foreground="#b5cea8", font=_FONT_MONO_B)

        scroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log["yscrollcommand"] = scroll.set

        scroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)

        # Snapshot every entry's starting value now that the form is built.
        # Taken from the widgets rather than written out a second time: a
        # hand-maintained list of defaults drifts the moment a field is added,
        # and the field that silently keeps its old value across a Clear All
        # is the one that carries the previous site's data into this run.
        self._input_defaults = {
            name: var.get() for name, var in vars(self).items()
            if isinstance(var, tk.Variable)
            and name not in _DERIVED_VARS and name not in _STICKY_VARS
        }

        self._log_write("Ready.  Select a .pqd file and click Run Analysis.\n")

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear_all(self):
        """Reset the service entries to the values they had when the window opened.

        The engineer's own name, title and email are left alone: they
        describe who is running the tool rather than which service was
        measured, and they go out on a customer document, where a field
        blanked by an unrelated button is worse than one left stale.

        Confirmed only when there is something to lose: a form already at its
        defaults clears silently, so the button never nags, but one carrying a
        typed address and a picked transformer asks first.
        """
        dirty = [name for name, default in self._input_defaults.items()
                 if getattr(self, name).get() != default]
        if dirty and not messagebox.askyesno(
                "Clear all entries?",
                f"This resets {len(dirty)} entr"
                f"{'y' if len(dirty) == 1 else 'ies'} — including the file, "
                "the transformer and conductor pickers "
                "— back to their defaults.\n\nYour engineer name, title and "
                "email are kept.\n\nClear the rest?"):
            return

        for name, default in self._input_defaults.items():
            getattr(self, name).set(default)

        # The pickers cascade: kVA options, the ISC label and the ISC entry's
        # enabled state are derived from the type and the override checkbox,
        # so they are refreshed rather than set, or the form would read as
        # cleared while still offering the previous transformer's sizes.
        self._xfmr_type_key = None
        self._on_type_change()
        self._on_isc_override_toggle()
        self._on_topo_change()

        self._log_clear()
        self._log_write("Cleared.  Service entries are back to their defaults; "
                        "engineer details kept.\n")
        self._log_write("Ready.  Select a .pqd file and click Run Analysis.\n")

    # ── Details section toggle ────────────────────────────────────────────────

    def _toggle_details(self):
        if self._details_open.get():
            self._details_frame.pack_forget()
            self._details_open.set(False)
            self._det_toggle_btn.config(
                text="▶  Report Details (engineer sign-off)")
        else:
            self._details_frame.pack(fill="x", before=self._sep_before_run)
            self._details_open.set(True)
            self._det_toggle_btn.config(
                text="▼  Report Details (engineer sign-off)")

    # ── Transformer cascade callbacks ─────────────────────────────────────────

    def _on_topo_change(self, _event=None):
        hints = {
            "auto":         "(auto-detected from channels)",
            "3ph-wye":      "(three-phase wye — 208Y/120 or 480Y/277)",
            "split-phase":  "(single-phase 120/240 V residential/small commercial)",
        }
        self._topo_hint.config(text=hints.get(self._topo_var.get(), ""))

    def _on_nominal_change(self, _event=None):
        """Re-derive kVA options when nominal voltage changes."""
        self._refresh_kva_options()

    def _on_primary_v_change(self, _event=None):
        """Show the L-N nominal derived from the entered primary L-L voltage.

        The ANSI check runs per phase against L-N, so that is the number the
        entry really sets. Showing it as it is typed lets a wrong entry be
        caught here rather than in the band printed on the finished report.
        """
        text = self._primary_ll_var.get().strip()
        if not text:
            self._primary_ln_hint.config(
                text="(nominal at the metering point — sets the ANSI C84.1 band)",
                fg="#888888")
            return
        try:
            ll = float(text)
        except ValueError:
            self._primary_ln_hint.config(text="(not a number)", fg="#cc6666")
            return
        if ll <= 0:
            self._primary_ln_hint.config(text="(must be above zero)", fg="#cc6666")
            return
        ln = ll / ll_factor(None, self._topo_var.get())
        band = ansi_bands(ln)
        if band.get("a_min") is None:
            self._primary_ln_hint.config(
                text=f"= {ln:,.0f} V L-N  ·  above 34.5 kV, outside ANSI C84.1",
                fg="#cc6666")
            return
        self._primary_ln_hint.config(
            text=f"= {ln:,.0f} V L-N  ·  ANSI C84.1 Range A "
                 f"{band['a_min']:,.0f}–{band['a_max']:,.0f} V",
            fg="#888888")

    def _on_type_change(self, _event=None):
        """Map display label back to internal key, then refresh kVA list."""
        display = self._xfmr_type_var.get()
        if display == "— not specified —":
            self._xfmr_type_key = None
        elif display == _PRIMARY_LABEL:
            self._xfmr_type_key = _PRIMARY_KEY
        else:
            # reverse-lookup
            self._xfmr_type_key = next(
                (k for k, v in _TYPE_DISPLAY.items() if v == display), None
            )
        self._refresh_kva_options()

    def _on_kva_change(self, _event=None):
        self._refresh_isc_label()

    def _on_isc_override_toggle(self):
        if self._isc_override_var.get():
            self._isc_entry.config(state="normal")
            self._isc_entry.focus_set()
        else:
            self._isc_entry.config(state="disabled")

    def _refresh_path_rows(self):
        """Show the controls that describe the path this meter actually sees.

        Metered on the secondary that is the shared main and the service run;
        metered on the primary it is the primary line, and the two conductor
        pickers are asking about wire on the customer's side of the meter.
        """
        primary = self._xfmr_type_key == _PRIMARY_KEY
        for frame in (self._cond_frame, self._shared_frame,
                      self._primary_v_frame, self._primary_frame):
            frame.pack_forget()
        if primary:
            self._primary_v_frame.pack(fill="x", padx=12, pady=(0, 6))
            self._primary_frame.pack(fill="x", padx=12, pady=(0, 6))
        else:
            self._cond_frame.pack(fill="x", padx=12, pady=(0, 6))
            self._shared_frame.pack(fill="x", padx=12, pady=(0, 6))

    def _refresh_kva_options(self):
        """Rebuild kVA combo list for the current type + nominal voltage."""
        key = self._xfmr_type_key
        self._refresh_path_rows()

        if key == _PRIMARY_KEY:
            # Primary metered: no kVA lookup, user must supply ISC manually
            self._kva_combo.config(state="disabled", values=[])
            self._kva_var.set("")
            self._isc_auto_var.set("Enter ISC from primary fault study (use Override below)")
            self._isc_auto_lbl.config(fg=_ISC_NONE)
            self._isc_override_var.set(True)
            self._isc_entry.config(state="normal")
            return

        if not key:
            # Size stays disabled until a Type is chosen.  Say why — a greyed-out
            # control with no explanation reads as a broken tool.
            self._kva_combo.config(state="disabled", values=[])
            self._kva_var.set("")
            self._isc_auto_var.set("Pick a transformer Type above to enable Size")
            self._isc_auto_lbl.config(fg=_ISC_NONE)
            return

        try:
            nominal = float(self._nominal_var.get())
        except ValueError:
            nominal = 120.0

        sizes = _kva_options(key, nominal)
        if not sizes:
            self._kva_combo.config(state="disabled", values=[])
            self._kva_var.set("")
            if not _BOOK_AVAILABLE:
                # The tables are empty because the import failed, not because
                # this type/voltage is missing from the Blue Book.
                self._isc_auto_var.set("Blue Book unavailable — this install is "
                                       "broken (see banner at top)")
            else:
                self._isc_auto_var.set("No Blue Book entries for this type/voltage "
                                       "combination — use Override below")
            self._isc_auto_lbl.config(fg=_ISC_NONE)
            return

        str_sizes = [str(s) for s in sizes]
        self._kva_combo.config(state="readonly", values=str_sizes)

        # Keep existing selection if still valid; otherwise pick first
        cur = self._kva_var.get()
        if cur not in str_sizes:
            self._kva_var.set(str_sizes[0])

        self._refresh_isc_label()

    def _refresh_isc_label(self):
        """Update the ISC auto-label from the current type + kVA selection."""
        key = self._xfmr_type_key
        if not key or key == _PRIMARY_KEY:
            return

        kva_str = self._kva_var.get()
        if not kva_str:
            self._isc_auto_var.set("")
            return

        try:
            nominal = float(self._nominal_var.get())
            kva = int(kva_str)
        except ValueError:
            self._isc_auto_var.set("")
            return

        isc, note = _isc_for(key, kva, nominal)
        if isc is not None:
            self._isc_auto_var.set(f"{isc:,} A  ·  {note}")
            self._isc_auto_lbl.config(fg=_ISC_FG)
        else:
            self._isc_auto_var.set("ISC not found for this combination")
            self._isc_auto_lbl.config(fg=_ISC_NONE)

    # ── File browser ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Pronto PQDIF file",
            filetypes=[("PQD files", "*.pqd"), ("All files", "*.*")],
        )
        if path:
            self._file_var.set(path)
            self._address_var.set(Path(path).stem)

    def _on_file_changed(self, *_args):
        """Re-scan for sessions a moment after the path stops changing."""
        if self._scan_after_id is not None:
            try:
                self.after_cancel(self._scan_after_id)
            except Exception:
                pass
        self._scan_after_id = self.after(400, self._scan_current_file)

    def _scan_current_file(self):
        self._scan_after_id = None
        path = self._file_var.get().strip().strip('"')
        if path.lower().endswith(".pqd") and Path(path).exists():
            self._scan_sessions(path)
        else:
            self._show_sessions([])

    def _scan_sessions(self, path):
        """Fill the session picker for the chosen file, off the UI thread.

        Reading only the time series is fast, but a file on a OneDrive folder
        may still have to come down from the network first, so this never runs
        inline: a picker that freezes the window is worse than no picker.
        """
        def work():
            try:
                sessions = ProntoAdapter.scan_sessions(Path(path))
            except Exception as exc:
                # The run itself will report a file it cannot read, with the
                # full traceback. Here it only means no picker.
                sessions = []
                _logging.getLogger(__name__).debug(
                    "session scan failed for %s: %s", path, exc)
            self.after(0, lambda: self._show_sessions(sessions))

        threading.Thread(target=work, daemon=True).start()

    def _show_sessions(self, sessions):
        """Show the picker only when there is a choice to make."""
        self._sessions = sessions
        if len(sessions) < 2:
            self._session_frame.pack_forget()
            self._session_var.set("")
            return

        labels = []
        longest = max(sessions, key=lambda s: s["intervals"])
        for s in sessions:
            start = (s["start_time"] or "")[:16].replace("T", " ")
            end = (s["end_time"] or "")[11:16]
            labels.append(f"{s['index'] + 1}:  {start} → {end}   "
                          f"({s['hours']:.1f} h, {s['intervals']} intervals)"
                          + ("   — longest" if s is longest else ""))
        self._session_combo.config(values=labels)
        self._session_var.set(labels[longest["index"]])
        self._session_note.config(
            text=f"{len(sessions)} sessions in this file — one is analysed")
        self._session_frame.pack(fill="x", padx=12, pady=6,
                                 after=self._file_frame)

    def _selected_session(self):
        """Zero-based index of the chosen session, or None for a plain file."""
        if len(self._sessions) < 2:
            return None
        try:
            return int(self._session_var.get().split(":", 1)[0]) - 1
        except (ValueError, AttributeError):
            return None

    def _build_import_error_banner(self):
        """Red banner shown when pq_analyzer failed to import.

        Without this the tool opens and looks healthy, but the Blue Book tables
        are empty, so the kVA picker never enables and ISC never autopopulates.
        """
        banner = tk.Frame(self, bg="#7a1c1c")
        banner.pack(fill="x", side="top")

        inner = tk.Frame(banner, bg="#7a1c1c")
        inner.pack(fill="x", padx=12, pady=8)

        tk.Label(
            inner,
            text="This install is broken — a required library failed to load.",
            bg="#7a1c1c", fg="#ffffff", font=_FONT_UI_B, anchor="w",
        ).pack(fill="x")

        tk.Label(
            inner,
            text=("Transformer/ISC lookup is disabled and analysis will not "
                  "run.\n"
                  f"This copy is pq-analyzer {_ENGINE_VERSION}.\n"
                  "From a Command Prompt in the pq-analyzer folder, run:\n"
                  "    python check_install.py\n"
                  "It will identify the problem and give you the fix."),
            bg="#7a1c1c", fg="#f0c0c0", font=_FONT_MONO, anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 6))

        btns = tk.Frame(inner, bg="#7a1c1c")
        btns.pack(fill="x")
        ttk.Button(btns, text="Show the error details",
                   command=self._show_import_error,
                   style="Plain.TButton").pack(side="left")
        tk.Label(btns, text="  (send these to the maintainer)",
                 bg="#7a1c1c", fg="#f0c0c0", font=_FONT_UI_S).pack(side="left")

    def _show_import_error(self):
        """Display the traceback that stopped the tool loading.

        Preferred from memory: it is the traceback this running copy actually
        hit. The file is a fallback for the case where the failure happened in
        an earlier run, and the instructions are the last resort.
        """
        detail = _IMPORT_TRACEBACK
        if not detail:
            log_path = Path(__file__).parent / "import_error.log"
            try:
                detail = (f"From a previous run ({log_path}):\n\n"
                          + log_path.read_text())
            except Exception:
                detail = (
                    "No traceback was recorded, which usually means this "
                    "window is left over from a run that has since been "
                    "fixed.\n\n"
                    "Close the tool and start it again. If the banner comes "
                    "back, run it from a Command Prompt to see the error:\n"
                    "    python run.py"
                )

        win = tk.Toplevel(self)
        win.title("Import error details")
        win.configure(bg=_BG)
        win.geometry("900x500")

        txt = tk.Text(win, bg=_LOG_BG, fg=_LOG_FG, font=_FONT_MONO, wrap="none")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", detail)
        txt.config(state="disabled")

        def _copy():
            self.clipboard_clear()
            self.clipboard_append(detail)

        ttk.Button(win, text="Copy to clipboard", command=_copy,
                   style="Plain.TButton").pack(pady=(0, 8))

    def _open_folder(self):
        folder = _SCRIPT.parent / "pq_output"
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return

        filepath = self._file_var.get().strip()
        if not filepath:
            messagebox.showerror("Missing file", "Please select a .pqd file first.")
            return
        if not Path(filepath).exists():
            messagebox.showerror("File not found", f"Cannot find:\n{filepath}")
            return

        try:
            nominal = float(self._nominal_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Nominal voltage must be a number (e.g. 120).")
            return

        # ISC override
        isc_amps = None
        if self._isc_override_var.get():
            isc_str = self._isc_manual_var.get().strip()
            if isc_str:
                try:
                    isc_amps = float(isc_str)
                except ValueError:
                    messagebox.showerror("Invalid input", "Override ISC must be a number (e.g. 5000).")
                    return

        # Service conductor run length — only meaningful with a conductor
        # picked, and a typo here would silently skew the expected impedance.
        run_length_ft = None
        run_str = self._run_length_var.get().strip()
        if run_str:
            try:
                run_length_ft = float(run_str)
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    "Run length must be a number of feet (e.g. 150).")
                return
        if self._conductor_labels.get(self._conductor_var.get()) and not run_length_ft:
            messagebox.showerror(
                "Invalid input",
                "A service conductor was picked without a run length. Enter "
                "the length in feet from the transformer to the meter, or set "
                "the conductor back to \u2014 not specified \u2014.")
            return

        # Shared secondary — same rule as the service run: a picked conductor
        # with no length silently drops out of the expected impedance instead
        # of announcing that it was ignored.
        shared_length_ft = None
        shared_str = self._shared_length_var.get().strip()
        if shared_str:
            try:
                shared_length_ft = float(shared_str)
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    "Shared secondary length must be a number of feet (e.g. 300).")
                return
        shared_key = self._conductor_labels.get(self._shared_var.get())
        if shared_key and not shared_length_ft:
            messagebox.showerror(
                "Invalid input",
                "A shared secondary was picked without a run length. Enter the "
                "length in feet from the transformer to this service's tap, or "
                "set it back to — not specified —.")
            return

        # Transformer kVA
        xfmr_key = self._xfmr_type_key

        # Primary line impedance, for a service metered on the high side. R1
        # and X1 carry the comparison, so they go together; R0 and X0 are
        # optional and used only where zero sequence is the right number.
        primary_metered = xfmr_key == _PRIMARY_KEY
        primary_z = {}
        for field, var in self._primary_vars.items():
            text = var.get().strip()
            if not text:
                continue
            try:
                primary_z[field] = float(text)
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    f"Primary line {field.upper()} must be a number of ohms "
                    "(e.g. 0.42).")
                return
        # The primary nominal is required, not optional. Falling back to
        # inference here would put the guess this field exists to remove back
        # into the ANSI band, and it would do it silently.
        primary_ll = None
        if primary_metered:
            text = self._primary_ll_var.get().strip()
            try:
                primary_ll = float(text)
            except ValueError:
                primary_ll = None
            if primary_ll is None or primary_ll <= 0:
                messagebox.showerror(
                    "Primary voltage required",
                    "Enter the primary line-to-line voltage for this service.\n\n"
                    "It sets the ANSI C84.1 band the readings are judged "
                    "against. Nothing in the meter file names the primary "
                    "voltage, so it cannot be inferred — a guess here would "
                    "become the limit printed in the report.")
                return
        if primary_metered and ("r1" in primary_z) != ("x1" in primary_z):
            messagebox.showerror(
                "Invalid input",
                "Primary line impedance needs both R1 and X1. Enter the other "
                "one, or clear both and the impedance will be measured without "
                "an expected value to compare against.")
            return
        kva = None
        kva_str = self._kva_var.get().strip()
        if xfmr_key and xfmr_key != _PRIMARY_KEY and kva_str:
            try:
                kva = float(kva_str)
            except ValueError:
                pass

        params = {
            "filepath":       filepath,
            "session":        self._selected_session(),
            "nominal":        nominal,
            "cclass_key":     _SCHEDULE_KEY.get(self._cclass_var.get(), "sg"),
            "service_role":   _ROLE_KEY.get(self._role_var.get(), "load"),
            "rated_ac_kw":    _float_or_none(self._rated_kw_var.get()),
            "annual_avg_load_kw": _float_or_none(self._avg_load_var.get()),
            "il_amps_billing":    _float_or_none(self._il_billing_var.get()),
            "site":           self._site_var.get().strip(),
            "address":        self._address_var.get().strip(),
            "engineer":       self._eng_name_var.get().strip(),
            "engineer_title": self._eng_title_var.get().strip(),
            "engineer_email": self._eng_email_var.get().strip(),
            "xfmr_key":       xfmr_key,
            "kva":            kva,
            "isc_amps":       isc_amps,
            # The service-type and topology pickers decide how many phases the
            # report and its charts describe; without these the plots fall back
            # to guessing from which channels happen to be present.
            "topology":       self._topo_var.get(),
            # A primary-metered service is measured above its own transformer,
            # so the secondary conductors are downstream of the meter and are
            # dropped here rather than quietly added to the expected impedance.
            "conductor_key":  None if primary_metered else
                              self._conductor_labels.get(self._conductor_var.get()),
            "run_length_ft":  None if primary_metered else run_length_ft,
            "shared_secondary_key": None if primary_metered else shared_key,
            "shared_secondary_ft":  None if primary_metered else shared_length_ft,
            "primary_metered":      primary_metered,
            "primary_ll_voltage":   primary_ll,
            "primary_r1_ohm":       primary_z.get("r1"),
            "primary_x1_ohm":       primary_z.get("x1"),
            "primary_r0_ohm":       primary_z.get("r0"),
            "primary_x0_ohm":       primary_z.get("x0"),
        }

        self._log_clear()
        self._run_btn.config(state="disabled", text="Running…")
        self._open_btn.config(state="disabled")
        self._running = True

        threading.Thread(target=self._run_direct, args=(params,), daemon=True).start()

    def _run_direct(self, params):
        handler = _GUILogHandler(self._log, self.after)
        handler.setFormatter(_logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_log = _logging.getLogger()
        root_log.addHandler(handler)
        try:
            self._do_analysis(params)
        except Exception as exc:
            import traceback
            self._log_write(f"\nError: {exc}\n{traceback.format_exc()}\n", tag="error")
        finally:
            root_log.removeHandler(handler)
            self.after(0, self._reset_run_btn)

    def _do_analysis(self, params):
        # Stop here, with the reason, rather than part-way down this function.
        # Everything imported from pq_analyzer comes in through one try/except,
        # so a single failed import leaves every one of those names undefined
        # and the run dies later at whichever name it reaches first --
        # "NameError: name 'Thresholds' is not defined", which describes the
        # symptom and hides the cause. The traceback that matters was recorded
        # at startup, and the user cannot send us the .pqd file to reproduce
        # with, so the message has to carry its own evidence.
        if not _BOOK_AVAILABLE:
            raise RuntimeError(
                "The analysis engine did not load when this tool started, so "
                "no file can be analysed. This is an install problem, not a "
                "problem with " + Path(params["filepath"]).name + ".\n\n"
                f"Version: pq-analyzer {_ENGINE_VERSION}\n\n"
                "The failure at startup was:\n\n"
                + (_IMPORT_TRACEBACK
                   or f"(not recorded — see {Path(__file__).parent / 'import_error.log'})")
            )

        filepath = params["filepath"]
        nominal  = params["nominal"]
        outdir   = _SCRIPT.parent / "pq_output"
        stem     = Path(filepath).stem

        # On a primary-metered service the entered L-L voltage is the nominal,
        # and the per-phase ANSI check runs against L-N -- so it is derived here
        # rather than read from the secondary-oriented Nominal Voltage picker,
        # which is not describing this service. Logged because a derived limit
        # that appears without explanation is the kind a reader cannot check.
        primary_ll = params.get("primary_ll_voltage")
        if params.get("primary_metered") and primary_ll:
            nominal = primary_ll / ll_factor(None, params.get("topology", "auto"))
            _band = ansi_bands(nominal)
            _logging.getLogger(__name__).info(
                "Primary-metered service: %.0f V L-L entered -> %.1f V L-N "
                "nominal; ANSI C84.1 Range A %s V L-N.",
                primary_ll, nominal,
                "not defined above 34.5 kV" if _band.get("a_min") is None
                else f"{_band['a_min']:.1f}-{_band['a_max']:.1f}",
            )

        # ── ISC resolution ────────────────────────────────────────────────────
        isc_amps   = params["isc_amps"]
        isc_source = None
        xfmr_key   = params["xfmr_key"]
        kva        = params["kva"]
        if isc_amps is not None:
            isc_source = f"Manual override ({isc_amps:.0f} A)"
        elif xfmr_key and xfmr_key != _PRIMARY_KEY and kva:
            result = _lookup_isc(xfmr_key, kva, nominal)
            if result:
                isc_amps, isc_note = result
                isc_source = isc_note

        thresh = Thresholds(
            nominal_voltage=nominal,
            customer_class=params["cclass_key"],
            service_role=params["service_role"],
            rated_ac_kw=params["rated_ac_kw"],
            annual_avg_load_kw=params["annual_avg_load_kw"],
            il_amps_billing=params["il_amps_billing"],
            isc_amps=isc_amps,
            isc_source=isc_source,
            transformer_kva=kva,
            service_type=params.get("xfmr_key"),
            topology=params.get("topology", "auto"),
            conductor_key=params.get("conductor_key"),
            run_length_ft=params.get("run_length_ft"),
            shared_secondary_key=params.get("shared_secondary_key"),
            shared_secondary_ft=params.get("shared_secondary_ft"),
            primary_metered=bool(params.get("primary_metered")),
            primary_ll_voltage=primary_ll,
            primary_r1_ohm=params.get("primary_r1_ohm"),
            primary_x1_ohm=params.get("primary_x1_ohm"),
            primary_r0_ohm=params.get("primary_r0_ohm"),
            primary_x0_ohm=params.get("primary_x0_ohm"),
        )

        # ── Adapter ───────────────────────────────────────────────────────────
        fp = Path(filepath)
        if fp.suffix.lower() == ".pqd":
            adapter = ProntoAdapter(fp, session=params.get("session"))
        elif _PQDIF_AVAILABLE:
            adapter = PQDIFAdapter(fp)
        else:
            raise RuntimeError(
                "pqdifpy is not installed and this is not a .pqd file.\n"
                "pip install pqdifpy  or use a .pqd Pronto file."
            )

        ds = extract_dataset(adapter, ChannelMapper())
        if ds.df.empty:
            raise RuntimeError("DataFrame is empty after extraction — check channel matching.")

        # ── Analysis ──────────────────────────────────────────────────────────
        df = ds.df
        volt_result         = check_voltage_compliance(df, thresh)
        thd_result          = check_thd(df, thresh)
        pf_result           = check_power_factor(df, thresh)
        imb_result          = check_voltage_imbalance(df, thresh)
        curr_imb_result     = check_current_imbalance(df, thresh)
        demand_result       = check_demand(df, thresh)
        harm_result         = check_individual_harmonics(df, thresh)
        volt_harm_result    = check_individual_voltage_harmonics(df, thresh)
        neutral_harm_result = check_neutral_harmonics(df, thresh)
        source_harm_result   = check_harmonic_sources(df, thresh)
        spectral_shape_result = check_spectral_shape(df, thresh, source_harm_result)
        direction_result      = check_harmonic_direction(ds, thresh)
        impedance_result      = check_source_impedance(df, thresh)
        stat_result         = check_harmonic_statistics(df, thresh)
        event_result        = detect_events(ds, thresh)
        neutral_health_result = check_neutral_health(ds, thresh)
        itic_result         = check_itic(event_result, thresh)
        ll_volt_result      = check_line_to_line_voltage(df, thresh)
        frequency_result    = check_frequency(df, thresh)
        flicker_result      = check_flicker(df, thresh)
        kfactor_result      = kfactor_by_phase(df)

        report = generate_report(
            ds, volt_result, thd_result, pf_result,
            imb_result, curr_imb_result, demand_result,
            harm_result, volt_harm_result, neutral_harm_result,
            source_harm_result, stat_result, event_result, thresh,
            neutral_health_result=neutral_health_result,
            spectral_shape_result=spectral_shape_result,
            direction_result=direction_result,
            impedance_result=impedance_result,
            itic_result=itic_result,
            ll_volt_result=ll_volt_result,
            frequency_result=frequency_result,
            flicker_result=flicker_result,
            kfactor_result=kfactor_result,
        )
        report["root_causes"] = analyze_root_causes(report, ds, thresh)

        # ── Export ────────────────────────────────────────────────────────────
        export_results(ds, report, outdir, stem=stem)

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_overview(ds, thresh, outdir=outdir, stem=stem)
        plot_voltage(df, volt_result, thresh, outdir=outdir, stem=stem)
        plot_thd(df, thd_result, thresh, outdir=outdir, stem=stem)
        plot_summary(df, imb_result, outdir=outdir, stem=stem)
        plot_harmonic_spectrum(df, thresh, outdir=outdir, stem=stem)
        plot_itic(event_result["events"], thresh, outdir=outdir, stem=stem)
        plot_neutral_health(ds, neutral_health_result, thresh, outdir=outdir, stem=stem)
        plot_demand_profile(df, thd_result, outdir=outdir, stem=stem)
        plot_harmonic_trend(df, outdir=outdir, stem=stem)
        plot_imbalance(df, imb_result, curr_imb_result, outdir=outdir, stem=stem)
        plot_pf_load(df, pf_result, outdir=outdir, stem=stem)
        plot_flicker(df, flicker_result, outdir=outdir, stem=stem)
        plot_waveform_capture(ds, thresh, outdir=outdir, stem=stem)

        # ── Word report ───────────────────────────────────────────────────────
        generate_word_report(
            report=report,
            thresh=thresh,
            ds=ds,
            site_name=params["site"] or stem,
            site_address=params["address"],
            engineer_name=params["engineer"],
            outdir=outdir,
            stem=stem,
            engineer_title=params["engineer_title"],
            engineer_email=params["engineer_email"],
        )

        # Separate customer-facing letter, residential only.
        letter = generate_customer_letter(
            report=report,
            thresh=thresh,
            site_address=params["address"] or params["site"] or stem,
            engineer_name=params["engineer"],
            outdir=outdir,
            stem=stem,
            engineer_title=params["engineer_title"],
            engineer_email=params["engineer_email"],
        )
        if letter:
            self._log_write(
                "\nCustomer document written alongside the internal "
                "engineering report.\n")
        else:
            # Anything else means the letter could not be produced. Saying
            # "residential-only" here would send the reader to whatever letter
            # is in the folder, which is then from an earlier run.
            self._log_write(
                "\nCustomer letter was NOT written — see the error above.\n",
                tag="error")

        self._log_write(
            "\nDone.  Internal engineering report, customer document and plots "
            "saved to pq_output/\n", tag="done")
        self._open_documents(stem)

    def _open_documents(self, stem: str):
        """Open both documents a run produces, not just the internal one.

        The customer document is opened first and the internal report second,
        so the internal one ends up in front -- it is the one the engineer
        reads first, and it was the only one that opened before this. Both are
        on screen either way, which is the point: the pair is written to be
        read against each other, and a customer document that has to be found
        by hand tends not to be read at all before it is sent.
        """
        outdir = _SCRIPT.parent / "pq_output"
        documents = [
            ("customer document", outdir / f"{stem}_customer_letter.docx"),
            ("internal engineering report",
             outdir / f"{stem}_internal_engineering_report.docx"),
        ]
        opened = 0
        for label, path in documents:
            if not path.exists():
                self._log_write(
                    f"\nThe {label} was not written, so it could not be "
                    "opened. See the log above.\n", tag="error")
                continue
            try:
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                elif sys.platform == "win32":
                    subprocess.Popen(["start", "", str(path)], shell=True)
                else:
                    subprocess.Popen(["xdg-open", str(path)])
                opened += 1
            except OSError as exc:
                # Failing to launch Word must not lose the run: the files are
                # written and the folder button still reaches them.
                self._log_write(f"\nCould not open the {label} ({exc}). "
                                f"It is saved at {path}.\n", tag="error")
        if opened:
            self.after(0, lambda: self._open_btn.config(state="normal"))

    def _reset_run_btn(self):
        self._run_btn.config(state="normal", text="Run Analysis")
        self._open_btn.config(state="normal")
        self._running = False

    # ── Feedback dialog ───────────────────────────────────────────────────

    def _show_feedback(self):
        import urllib.parse

        win = tk.Toplevel(self)
        win.title("Send Feedback")
        win.configure(bg=_BG)
        win.resizable(False, False)
        win.grab_set()

        hdr = tk.Frame(win, bg=_BTN_RUN)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Send Feedback", bg=_BTN_RUN, fg="white",
                 font=_FONT_UI_B, pady=10, padx=16).pack(anchor="w")

        body = tk.Frame(win, bg=_BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        tk.Label(body, text="What happened, what you expected, or what would help:",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(anchor="w")

        txt = tk.Text(body, width=56, height=8, font=_FONT_MONO,
                      relief="solid", bd=1, padx=6, pady=4, wrap="word")
        txt.pack(fill="both", expand=True, pady=(4, 0))
        txt.focus_set()

        tk.Label(body, text="Include the .pqd filename and a screenshot if relevant.",
                 bg=_BG, fg="#888888", font=_FONT_UI_S).pack(anchor="w", pady=(4, 8))

        btn_row = tk.Frame(body, bg=_BG)
        btn_row.pack(fill="x")

        def _send():
            note = txt.get("1.0", "end").strip()
            file_path = self._file_var.get()
            body_text = note
            if file_path:
                body_text += f"\n\n---\nFile: {Path(file_path).name}"
            params = urllib.parse.urlencode({
                "subject": "PQ Analyzer Feedback",
                "body":    body_text,
            }, quote_via=urllib.parse.quote)
            webbrowser.open(f"mailto:jacobbyronwhitaker@gmail.com?{params}")
            win.destroy()

        ttk.Button(btn_row, text="Send via Email", command=_send,
                   style="Run.TButton", cursor="hand2",
                   ).pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=win.destroy,
                   style="Quiet.TButton", cursor="hand2",
                   ).pack(side="left", padx=(8, 0))

    # ── Help window ───────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("PQ Analyzer — Reference Guide")
        win.configure(bg=_BG)
        win.resizable(True, True)
        win.minsize(640, 560)

        win.minsize(760, 620)

        # Header bar
        hdr = tk.Frame(win, bg=_BTN_RUN)
        hdr.pack(fill="x")
        tk.Label(hdr, text="PQ Analyzer — Reference & Standards",
                 bg=_BTN_RUN, fg="white", font=_FONT_UI_B,
                 pady=10, padx=16).pack(anchor="w")

        # One notebook page per topic rather than one scroll of everything.
        # The guide covers three different kinds of thing -- what to enter and
        # why, what the standards say, and how to work a job -- and a reader
        # who wants one of them should not have to scroll past the other two.
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True, padx=12, pady=(10, 6))

        _f0, _fs = _FONT_UI[0], _FONT_UI[1]
        _link_map = {}
        _current = {}

        class _CurrentText:
            """Writes to whichever page is being built."""
            def __getattr__(self, name):
                return getattr(_current["page"], name)

        txt = _CurrentText()

        def page(tab_title):
            """A scrollable pane, added as a tab, and returned to be selected."""
            frame = tk.Frame(nb, bg=_BG)
            nb.add(frame, text=f"  {tab_title}  ")
            t = tk.Text(frame, bg=_BG, fg=_LABEL_FG, font=_FONT_UI,
                        relief="flat", wrap="word", cursor="arrow",
                        state="normal", padx=10, pady=8)
            sb = ttk.Scrollbar(frame, command=t.yview)
            t["yscrollcommand"] = sb.set
            sb.pack(side="right", fill="y")
            t.pack(side="left", fill="both", expand=True)

            t.tag_config("h1",   font=(_f0, _fs, "bold"), foreground=_BTN_RUN,
                         spacing1=10, spacing3=2)
            t.tag_config("rule", font=(_f0, 8),  foreground="#cccccc")
            t.tag_config("h2",   font=(_f0, _fs, "bold"), foreground="#333333",
                         spacing1=8, spacing3=1, lmargin1=12, lmargin2=12)
            t.tag_config("body", font=_FONT_UI,  foreground="#555555",
                         lmargin1=24, lmargin2=24, spacing3=2)
            t.tag_config("lead", font=_FONT_UI, foreground="#444444",
                         lmargin1=12, lmargin2=12, spacing3=3)
            t.tag_config("link", font=(_f0, _fs-1), foreground=_BTN_RUN,
                         underline=True)
            t.tag_config("note", font=(_f0, _fs-1), foreground="#999999",
                         lmargin1=24, lmargin2=24)
            return t

        def use(t):
            _current["page"] = t

        def lead(body):
            """An orienting paragraph under a page's title, before the entries."""
            for line in body.splitlines():
                txt.insert("end", f"  {line}\n", "lead")
            txt.insert("end", "\n")

        def _add_link(label, url):
            tag = f"_lnk{len(_link_map)}"
            _link_map[tag] = url
            target = _current["page"]
            target.insert("end", label, ("link", tag))
            target.tag_bind(tag, "<Enter>",
                            lambda e, w=target: w.config(cursor="hand2"))
            target.tag_bind(tag, "<Leave>",
                            lambda e, w=target: w.config(cursor="arrow"))
            target.tag_bind(tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))

        def section(title):
            txt.insert("end", f"\n{title}\n", "h1")
            txt.insert("end", "─" * 60 + "\n", "rule")

        def std(name, desc, url):
            txt.insert("end", f"  {name}\n", "h2")
            for line in desc.splitlines():
                txt.insert("end", f"  {line}\n", "body")
            txt.insert("end", "  ")
            _add_link("Open on IEEE Xplore ↗", url)
            txt.insert("end", "\n\n")

        def concept(title, body):
            txt.insert("end", f"  {title}\n", "h2")
            for line in body.splitlines():
                txt.insert("end", f"  {line}\n", "body")
            txt.insert("end", "\n")

        # Pages are created here in the order the tabs should appear; the
        # content below is emitted in whatever order it already sat in, by
        # switching pages with use(). Tab order and file order are separate.
        pg_start     = page("Start here")
        pg_standard  = page("Which standard")
        pg_tariff    = page("PSCo tariff")
        pg_refs      = page("Standards")
        pg_workflow  = page("Investigating a job")
        pg_concepts  = page("Concepts")
        pg_neutral   = page("Neutral integrity")
        pg_methods   = page("Methods")

        # ── Start here ─────────────────────────────────────────────────────
        use(pg_start)
        section("Start here")
        lead(
            "This tool measures a recording and grades it against whichever\n"
            "standard governs the service. Most of that is automatic. Three\n"
            "things are not, because nothing in a .pqd file carries them, and\n"
            "getting any of the three wrong changes the answer rather than\n"
            "just the wording."
        )

        concept(
            "1. What is physically behind the meter",
            "Set Power flow to match the site, not the customer's rate schedule:\n"
            "\n"
            "    Load only                consumes only\n"
            "    Load + generation        an array or battery in parallel\n"
            "    Generation only          a producer's plant, no load to speak of\n"
            "\n"
            "This decides whether a negative fundamental means the CTs are on\n"
            "backwards or the site is exporting — opposite conclusions from the\n"
            "same measurement.  See \"Which standard\" for which schedules mean\n"
            "there is hardware on site; several solar schedules do not.",
        )

        concept(
            "2. Rated generation and average load demand",
            "Needed on any site with generation.  IEEE 519-2022 Figure 1 compares\n"
            "them to decide whether 519 governs the service at all or whether\n"
            "IEEE 1547 does, and the two standards differ by three times in the\n"
            "aggregate current limit.\n"
            "\n"
            "Both come from records: a nameplate and a year of billing.  Left\n"
            "blank, the report says the standard is undetermined rather than\n"
            "guessing — which is honest, but it is not an answer you can send.",
        )

        concept(
            "3. IL from billing",
            "IEEE 519 defines IL as the twelve previous months' 15- or 30-minute\n"
            "maximum demands, averaged.  That is a billing quantity; no week-long\n"
            "recording can produce it.\n"
            "\n"
            "Leave it blank and the tool substitutes the largest fundamental in\n"
            "the recording and says so on the page.  That is fine for a look, but\n"
            "IL is a denominator: a slow week shrinks it and inflates every\n"
            "percentage measured against it.  Before telling a customer they are\n"
            "outside a limit, get the twelve-month figure.",
        )

        concept(
            "What the tool will not guess",
            "A recurring principle rather than a list of quirks.  Where a quantity\n"
            "is knowable only from records or from the field, the tool asks for it\n"
            "and states what it did without it:\n"
            "\n"
            "  · the primary line-to-line nominal on a primary-metered service\n"
            "  · which way power flows at the meter\n"
            "  · the plant rating and the annual average load demand\n"
            "  · IL from billing\n"
            "  · the conductor and run length for an expected-impedance check\n"
            "\n"
            "Anything derived from a substitute is labelled as such in the report.\n"
            "If a number looks surprising, read the sentence next to it first —\n"
            "it usually says which input it was resting on.",
        )

        concept(
            "Where to go next",
            "  Which standard        519 or 1547, and which schedules mean\n"
            "                        generation is actually on site\n"
            "  PSCo tariff           the PF and phase clauses, and what they\n"
            "                        actually say (this catches people out)\n"
            "  Standards             the reference shelf, with links\n"
            "  Investigating a job   what to check first, by customer class\n"
            "  Concepts              the ideas the checks are built on\n"
            "  Neutral integrity     split-phase open-neutral theory and tests\n"
            "  Methods               how each individual check is computed",
        )

        # ── IEEE Standards ─────────────────────────────────────────────────
        use(pg_refs)
        section("IEEE / ANSI Standards")
        lead(
            "The reference shelf. What each standard covers and where to read it;\n"
            "the limits and tables the tool applies are on the other pages."
        )

        std(
            "IEEE 519-2022 — Harmonic Control in Electric Power Systems",
            "Primary harmonic compliance standard used by this tool.  Defines current TDD\n"
            "and voltage THD limits at the PCC based on the ISC/IL ratio (short-circuit\n"
            "strength relative to load).  Replaces the 2014 edition.",
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            "?newsearch=true&queryText=IEEE+519-2022",
        )

        std(
            "IEEE 1159-2019 — Monitoring Electric Power Quality",
            "Establishes standard terminology and measurement methods for PQ phenomena:\n"
            "sags, swells, interruptions, harmonics, flicker, and transients.  Essential\n"
            "reference for interpreting meter data and writing PQ reports.",
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            "?newsearch=true&queryText=IEEE+1159-2019",
        )

        std(
            "IEEE 1453-2022 — Voltage Fluctuations on Power Systems (Flicker)",
            "Defines flicker severity indices Pst (short-term) and Plt (long-term) and\n"
            "the limits for fluctuating loads such as arc furnaces, welders, and large\n"
            "motor starts.",
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            "?newsearch=true&queryText=IEEE+1453-2022",
        )

        std(
            "IEEE 1250-2011 — Guide for Identifying and Improving Voltage Quality",
            "Practical troubleshooting guide covering harmonic resonance, capacitor bank\n"
            "interactions, voltage notching, and neutral conductor sizing for nonlinear\n"
            "loads.",
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            "?newsearch=true&queryText=IEEE+1250-2011",
        )

        txt.insert("end", "  ANSI C84.1-2016 — Electric Power Systems Voltage Ratings\n", "h2")
        txt.insert("end",
                   "  Defines Range A (normal operating band, ±5% of nominal) and Range B\n"
                   "  (for voltages arising from practical design and operating\n"
                   "  conditions, to be corrected within a reasonable time).\n"
                   "  120 V system, service voltage: Range A = 114–126 V, Range B = 110–127 V.\n"
                   "  Over 600 V (2.4–34.5 kV): Range A = 97.5–105%, Range B = 95–105.8%.\n"
                   "  Published by NEMA — not available on IEEE Xplore.\n", "body")
        txt.insert("end", "  Available at ", "note")
        _add_link("webstore.ansi.org ↗",
                  "https://webstore.ansi.org/search#q=C84.1&t=All")
        txt.insert("end", "\n\n")

        txt.insert("end", "  ITI (CBEMA) Curve Application Note — Voltage Tolerance Envelope\n", "h2")
        txt.insert("end",
                   "  Published by the Information Technology Industry Council (ITIC), 2000.\n"
                   "  Defines the voltage tolerance envelope that information technology\n"
                   "  equipment must be able to ride through without disruption:\n"
                   "\n"
                   "    Duration < 20 ms    Any voltage (interruption acceptable)\n"
                   "    20 ms – 500 ms      ≥ 70 % of nominal\n"
                   "    500 ms – 10 s       ≥ 80 % of nominal\n"
                   "    > 10 s (steady)     ≥ 90 % of nominal  (aligns with ANSI C84.1 Range A)\n"
                   "\n"
                   "    Overvoltage limits: 110 % steady · 120 % up to 500 ms ·\n"
                   "                        140 % up to 20 ms · 200 % up to 3 ms · 500 % spikes\n"
                   "\n"
                   "  Superseded the CBEMA curve, which was originally referenced in\n"
                   "  ANSI/IEEE 446-1987 (Emergency and Standby Power Systems).\n"
                   "  IEEE 1159-2019 references the ITIC curve as the standard voltage\n"
                   "  tolerance boundary for power quality event classification.\n"
                   "  Applicable to 120 V nominal (120/208 V and 120/240 V, 60 Hz systems).\n", "body")
        txt.insert("end", "  Curve document at ", "note")
        _add_link("itic.org ↗", "https://www.itic.org/technical-activities/tech-committees/power-quality")
        txt.insert("end", "\n\n")

        # ── PSCo Tariff Reference ──────────────────────────────────────────
        # ── Which harmonic standard applies ────────────────────────────────
        use(pg_standard)
        section("Which Harmonic Standard Applies — 519 or 1547")
        lead(
            "Two questions, in order. Is there generation physically behind this\n"
            "meter? And if so, is it big enough relative to the load that IEEE 519\n"
            "hands the site to IEEE 1547? The second question is the one people\n"
            "miss, and the schedules make the first one harder than it looks."
        )

        concept(
            "519 does not cover every service",
            "IEEE 519-2022 limits its own scope, in Clause 5.2, to \"a user's PCC\n"
            "primarily with harmonic producing loads\", and directs installations that\n"
            "are primarily inverter-based to another standard.  A site with solar or\n"
            "storage may therefore not be a 519 site at all, and grading it against\n"
            "Table 2 would quote a limit three times looser than the one that governs.\n"
            "\n"
            "Figure 1 of 519-2022 is the decision tree, and this tool follows it:\n"
            "\n"
            "    Does the installation have a DER or IBR?\n"
            "      No  ─────────────────────────────► IEEE 519 limits at the PCC\n"
            "      Yes ─► Is combined site rated generation\n"
            "             < 10% of annual average load demand?\n"
            "               Yes ─────────────────────► IEEE 519 limits at the PCC\n"
            "               No  ─────────────────────► IEEE 1547 (or 2800)\n"
            "\n"
            "Both quantities in that test come from records, not from the recording:\n"
            "a nameplate and a year of billing.  Enter them in the Power flow row.\n"
            "Without them the tool says the standard is undetermined rather than\n"
            "guessing, because the two answers are not close together.",
        )

        concept(
            "What changes when 1547 applies",
            "1547 is not 519 with different numbers.  Three things differ:\n"
            "\n"
            "  Denominator   519 uses IL, the maximum demand load current.  1547 uses\n"
            "                I_rated, the DER unit's rated current capacity — a\n"
            "                nameplate, transformed to the reference point of\n"
            "                applicability if a transformer sits between.  Nothing in a\n"
            "                recording establishes it.\n"
            "\n"
            "  Limits        519's limits scale with ISC/IL: a stiff service earns a\n"
            "                load more headroom.  1547's are fixed.  A strong system\n"
            "                buys a plant nothing.\n"
            "\n"
            "  The metric    519 grades TDD, which specifically excludes interharmonics.\n"
            "                1547 grades TRD, which includes them.  They are different\n"
            "                measurements even where the denominators agree.\n"
            "\n"
            "                TRD = √(I_rms² − I₁²) ÷ I_rated × 100%   (1547 Eq. 2)\n"
            "\n"
            "IEEE 1547-2018 Clause 7.3, Table 26 — odd orders, % of I_rated:\n"
            "\n"
            "    h < 11    11 ≤ h < 17   17 ≤ h < 23   23 ≤ h < 35   35 ≤ h < 50   TRD\n"
            "     4.0%        2.0%          1.5%          0.6%          0.3%       5.0%\n"
            "\n"
            "Table 27 — even orders:  h=2 → 1.0%   h=4 → 2.0%   h=6 → 3.0%\n"
            "                         8 ≤ h < 50 → the odd range above\n"
            "\n"
            "Note the even limits are *looser* than 519's blanket 25%-of-odd rule.\n"
            "1547's rationale is that the 25% figure was researched and not found to\n"
            "be supported for a DER, though the 2nd harmonic was not relaxed.",
        )

        concept(
            "Two caveats that travel with every 1547 number",
            "Both are printed in the report, and neither is a formality:\n"
            "\n"
            "  Background exclusion   1547 sets its limits \"exclusive of any harmonic\n"
            "                         currents due to harmonic voltage distortion\n"
            "                         present in the Area EPS without the DER\n"
            "                         connected\".  Strictly, that needs a measurement\n"
            "                         taken with the plant offline.  A single site\n"
            "                         visit does not have one, so the figures include\n"
            "                         whatever the background drives through the\n"
            "                         inverters — conservative against the plant.\n"
            "\n"
            "  Interharmonics         TRD includes them; our meters report integer\n"
            "                         orders only.  The true TRD is therefore at least\n"
            "                         what we print, so a narrow pass is not clearance.",
        )

        concept(
            "IL is a billing quantity, not a measurement",
            "This one catches everyone.  519-2022 defines maximum demand load current\n"
            "as:\n"
            "\n"
            "    the sum of the rms currents corresponding to the 15 min or 30 min\n"
            "    maximum demand during each of the twelve previous months divided\n"
            "    by 12\n"
            "\n"
            "That is a year of billing history.  No week-long recording can produce it.\n"
            "If you leave the IL field blank, this tool substitutes the largest\n"
            "fundamental current in the recording and says so on the page — but that\n"
            "is a stand-in, and it moves with the weather, the season and the shift\n"
            "pattern that happened to run while the meter was on.\n"
            "\n"
            "It matters because IL is a denominator.  A recording taken in a slow week\n"
            "gives a small IL, which inflates every percentage measured against it and\n"
            "can manufacture a violation that a year of data would not support.  If\n"
            "you are about to tell a customer they are outside IEEE 519, get the\n"
            "twelve-month figure from billing first.",
        )

        concept(
            "Which schedules mean generation is present",
            "PSCo has seven renewable schedules and only some put hardware behind the\n"
            "meter.  The Figure 1 test is about what is physically there, not what the\n"
            "customer is billed under:\n"
            "\n"
            "    Generation on site        No generation on site\n"
            "    ─────────────────────     ──────────────────────────────────────\n"
            "    NM    net metering        OS-NM   array on other property\n"
            "    PV    rooftop/on-site     RC/RCF  Renewable*Connect subscription\n"
            "    RE    recycled energy     SRCS    Solar*Rewards Community share\n"
            "    AVPP  aggregated DERs\n"
            "\n"
            "The right column bills like solar and measures like an ordinary load, so\n"
            "\"the customer is net metered\" is not enough to go on — ask what is on\n"
            "site.  Two easy mistakes:\n"
            "\n"
            "  · RE is waste-heat generation, not solar, and it is real parallel\n"
            "    generation.  AVPP is batteries, which export on dispatch rather than\n"
            "    on sunlight, so there is no quiet period to reason from.\n"
            "  · Schedule SRCS names the *subscribers* who buy a community array's\n"
            "    output.  The array itself is the \"SRCS Producer\" and takes service\n"
            "    separately, on the Company's own production meter.  If you are\n"
            "    metering the field, you are not metering an SRCS customer.",
        )

        std(
            "IEEE 1547-2018 — Interconnection and Interoperability of DER",
            "Governs current distortion for inverter-based installations, which\n"
            "519-2022 Figure 1 hands over whenever site rated generation reaches 10%\n"
            "of annual average load demand.  Clause 7.3 and Tables 26–27 carry the\n"
            "limits; Clause 7.4 covers overvoltage contribution.",
            "https://ieeexplore.ieee.org/search/searchresult.jsp"
            "?newsearch=true&queryText=IEEE+1547-2018",
        )

        use(pg_tariff)
        section("PSCo Electric Tariff — PQ Requirements")
        lead(
            "The tariff clauses are scoped by which section of the Rules and\n"
            "Regulations they sit in, not by rate schedule. That is the opposite\n"
            "of how they are usually quoted, including by earlier versions of this\n"
            "tool, so the wording below is taken from the filed tariff."
        )

        concept(
            "The PF clauses are scoped by rules section, not by schedule",
            "This is easy to get backwards, and the tool had it backwards until it was\n"
            "checked against the filed tariff.  The Rules and Regulations are divided\n"
            "into GENERAL, RESIDENTIAL and COMMERCIAL AND INDUSTRIAL parts, and the two\n"
            "power factor clauses live in different ones:\n"
            "\n"
            "  Sheet R73   Rules and Regulations — GENERAL, \"Customer's Installation\"\n"
            "              \"Company's rates contemplate Customer's use of service at a\n"
            "              Power Factor, at the point where service is metered, of not\n"
            "              less than ninety percent (90%) lagging... Company reserves\n"
            "              the right to discontinue service to any Customer not\n"
            "              complying herewith.\"\n"
            "              → General section, so it reaches EVERY class, residential\n"
            "                included.  It also requires PF correction on inherently\n"
            "                low-PF equipment (neon, fluorescent and the like).\n"
            "\n"
            "  Sheet R121  Rules and Regulations — COMMERCIAL AND INDUSTRIAL\n"
            "              \"Customer, at all times, will maintain at Company's Point of\n"
            "              Delivery a Power Factor as near unity as practicable.\"\n"
            "              → All of C, SG and PG, not PG alone.  Where correction\n"
            "                equipment is fitted, the customer must also install a\n"
            "                relay or switch to control it and prevent excessive\n"
            "                voltage variation on the Company's lines.\n"
            "\n"
            "So a C&I customer is under both at once, and a residential customer is\n"
            "under R73.  The Residential rules section contains no power factor clause\n"
            "of its own — which is not the same as residential being exempt.  What is\n"
            "true is that no reactive billing applies there, so the tool does not raise\n"
            "power factor as a finding on Schedule R.",
        )

        concept(
            "The 15% phase clause is a billing provision, not a limit",
            "Sheet R123, Commercial and Industrial, under \"Billing Demands will be\n"
            "determined as set forth in the applicable rate schedule, subject to the\n"
            "following provisions\":\n"
            "\n"
            "    If three-phase service is provided and Customer's equipment is so\n"
            "    connected that at the Point of Delivery the load on any one (1) phase\n"
            "    exceeds the load on any other phase by more than fifteen percent\n"
            "    (15%), the Company MAY TAKE AS THE BILLING DEMAND the three-phase\n"
            "    equivalent of the maximum kilovolt-amperes in any phase adjusted to a\n"
            "    ninety percent (90%) Power Factor.\n"
            "\n"
            "Read it carefully: nothing there forbids imbalance.  It says what the\n"
            "Company may charge for if imbalance exceeds 15%.  Telling a customer they\n"
            "are \"outside Sheet R121's 15% limit\" is wrong twice over — wrong sheet,\n"
            "and wrong about what the clause does.  The defensible sentence is that\n"
            "above 15% their billing demand may be computed from the worst phase.\n"
            "\n"
            "The 10% figure the tool flags against is the PSCo Blue Book and NEMA MG1\n"
            "guidance for equipment health, which is a separate matter from billing.",
        )

        concept(
            "Harmonics are not in the tariff at all",
            "No PSCo schedule carries a harmonic clause.  Enforcement runs through the\n"
            "PSCo Blue Book, which references IEEE 519 — or, for an installation with\n"
            "generation, IEEE 1547 by way of 519-2022 Figure 1.  See the section above.",
        )

        concept(
            "Tariff Sheet Reference Summary",
            "  Sheet R73   PF ≥ 0.90 lagging at the metering point.  GENERAL rules, so\n"
            "              all classes.  Right to discontinue service for non-compliance.\n"
            "\n"
            "  Sheet R121  PF as near unity as practicable at the point of delivery.\n"
            "              COMMERCIAL AND INDUSTRIAL rules, so C, SG and PG.\n"
            "\n"
            "  Sheet R123  Billing demand provisions, including the 15% phase clause.\n"
            "              A billing remedy, not a limit on the customer.\n"
            "\n"
            "  Harmonics   No tariff clause; Blue Book → IEEE 519-2022 / IEEE 1547-2018.\n"
            "\n"
            "Verified against the filed tariff (COLO. PUC No. 8 Electric) on 2026-08-13.\n"
            "Sheet numbering moves with tariff revisions — re-check before citing a\n"
            "sheet number to a customer.",
        )

        # ── Investigation Guidance by Customer Class ───────────────────────
        use(pg_workflow)
        section("Investigation Guidance by Customer Class")
        lead(
            "What to look at first when a job lands, by who the customer is. This\n"
            "is procedure, not limits -- the numbers live on the other pages."
        )

        concept(
            "Schedule R — Residential",
            "Common complaints: lights flickering, appliances or electronics resetting,\n"
            "breakers tripping.\n"
            "\n"
            "Most likely causes and what to check first:\n"
            "  1. ANSI C84.1 compliance — if voltage is below Range A, that is a utility\n"
            "     responsibility.  Check for low secondary voltage, long service runs,\n"
            "     undersized conductors, or transformer tap set too low.\n"
            "  2. Voltage trend by hour-of-day — low voltage that tracks load (peaks at\n"
            "     noon or evening) points to secondary conductor sizing or a tap issue.\n"
            "  3. Voltage sag events vs ITIC curve — if events fall inside the ITIC\n"
            "     immunity envelope, the customer's equipment should not be tripping;\n"
            "     the problem is equipment sensitivity, not your system.\n"
            "  4. Flicker (Pst) — motor starts on shared transformers (well pumps, large\n"
            "     HVAC) can cause neighbor complaints even when voltage stays in Range A.\n"
            "  5. Split-phase imbalance — one leg consistently lower than the other\n"
            "     suggests unbalanced loading or a neutral issue.\n"
            "\n"
            "Utility vs customer split: sustained low voltage = utility.  Equipment\n"
            "tripping on normal transients = likely equipment sensitivity.  Neutral issues\n"
            "require field investigation to determine responsibility.",
        )

        concept(
            "Schedule C — Small Commercial  (< 50 kW)",
            "Common complaints: POS systems crashing, LED lighting flickering, HVAC\n"
            "controls locking out, unexplained equipment restarts.\n"
            "\n"
            "Most likely causes and what to check first:\n"
            "  1. ANSI C84.1 compliance — same first stop as residential.\n"
            "  2. Voltage sag events vs ITIC curve — most commercial equipment sensitivity\n"
            "     complaints are explained here.  If the sag is inside ITIC, the equipment\n"
            "     is not immune enough for a normal utility system.\n"
            "  3. Harmonic signature — H3-dominant = SMPS loads (customer's computers,\n"
            "     LED drivers, switching supplies) polluting the shared neutral.\n"
            "     H5/H7-dominant = small VFDs on HVAC or refrigeration equipment.\n"
            "  4. Power factor — small motors and compressors.  Cite Sheet R73 if below\n"
            "     0.90 lagging.\n"
            "  5. Voltage trend by hour — separates 'our feeder is weak at 5 PM' from\n"
            "     'their own load is causing the event'.\n"
            "\n"
            "Utility vs customer split: if TDD or THD is elevated, identify the load\n"
            "signature first — it almost always points back to the customer's own\n"
            "equipment.  Voltage sags from utility-side faults are typically short\n"
            "(<10 cycles) with sharp recovery.",
        )

        concept(
            "Schedule SG — C&I Secondary  (≥ 50 kW)",
            "Most complex class.  Think manufacturers, warehouses, food processing —\n"
            "multiple VFDs, large motors, mixed loads.  The customer's own equipment\n"
            "is the most likely source of problems.\n"
            "\n"
            "Most likely causes and what to check first:\n"
            "  1. TDD vs IEEE 519 limit — the most important flag.  If over limit,\n"
            "     it is almost always their VFDs or rectifiers.  Use the harmonic\n"
            "     signature to identify the load type.\n"
            "  2. Harmonic load signature — H5/H7 ratio identifies 6-pulse VFDs;\n"
            "     H11/H13 dominant = 12-pulse rectifier; H3 dominant = SMPS/computers;\n"
            "     high interval-to-interval variability = arc or welder load.\n"
            "  3. Voltage sag profile — large motor starts show as a correlated voltage\n"
            "     drop + current spike.  If the sag originates on their panel (voltage\n"
            "     drops at meter during their own motor start), it is their system.\n"
            "     Plot against ITIC to show whether their other equipment should be\n"
            "     immune to their own starts.\n"
            "  4. Current imbalance — unbalanced single-phase loads spread across a\n"
            "     3-phase panel.  Over 10% warrants action (Blue Book / NEMA MG1).  Above\n"
"     15%, Sheet R123 lets billing demand be taken from the worst phase —\n"
"     that is a charge, not a breach.\n"
            "  5. Power factor — large motor loads.  Cite Sheet R73.  Note that\n"
            "     adding capacitor banks for PF correction can create harmonic resonance\n"
            "     — check for amplified harmonic orders after correction is installed.\n"
            "  6. Transformer K-factor — high harmonic load may be overheating a\n"
            "     standard transformer even below nameplate kVA rating.\n"
            "\n"
            "Utility vs customer split: high TDD = customer's loads.  Low steady-state\n"
            "voltage before any events = utility tap or conductor.  Self-inflicted sags\n"
            "from their own motor starts are their responsibility to mitigate.",
        )

        concept(
            "Schedule PG — C&I Primary",
            "Largest customers; own their substation.  Your metering is upstream of\n"
            "their transformer — you are measuring what they inject into your system,\n"
            "not their internal PQ.  This makes attribution cleaner: if TDD is high\n"
            "at primary metering, they own it definitively.\n"
            "\n"
            "Most likely causes and what to check first:\n"
            "  1. Power factor — cite Sheet R121 (near unity, all C&I) and Sheet R73\n"
"     (0.90 lagging, all classes).  Large lagging PF is visible on the\n"
            "     feeder and depresses voltage for neighboring customers.  Common cause:\n"
            "     bulk capacitor banks undersized or switched off-peak.\n"
            "  2. Flicker (Pst/Plt) — arc furnaces and welders cause flicker that\n"
            "     propagates upstream.  This is the most likely way a PG customer\n"
            "     affects your other customers on the same feeder.  Compare Pst to\n"
            "     IEC 61000-3-7 / IEEE 1453 planning levels.\n"
            "  3. Per-order harmonic spectrum — capacitor bank resonance at primary\n"
            "     voltage can amplify specific harmonic orders dramatically.  An H7 or\n"
            "     H11 spike that is disproportionate to the load signature is the tell.\n"
            "  4. Current imbalance — large 3-phase industrial with unbalanced\n"
            "     single-phase loads.  Above 15% Sheet R123 allows billing demand from\n"
"     the worst phase; the 10% action threshold is Blue Book / NEMA MG1.\n"
            "  5. Demand profile — spikes in peak demand that pull your feeder voltage\n"
            "     down affect all other customers.  Use as context for any voltage\n"
            "     complaint investigations on the same feeder.\n"
            "\n"
            "Note: because metering is at primary voltage, sag events and harmonic\n"
            "data are upstream of the customer's transformer.  A sag that appears\n"
            "minor at primary (4 kV or 12 kV) may be amplified at their secondary\n"
            "if their transformer is near saturation.",
        )

        # ── Key Concepts ───────────────────────────────────────────────────
        use(pg_concepts)
        section("Key Concepts")
        lead(
            "How each check works and why it is built the way it is. Read this when\n"
            "a result is surprising and you want to know what produced it."
        )

        concept(
            "THD vs TDD",
            "THD (Total Harmonic Distortion) expresses harmonic current as a percentage\n"
            "of the fundamental at the moment of measurement.  It rises when load drops,\n"
            "even if absolute harmonic amps are unchanged.\n"
            "\n"
            "TDD (Total Demand Distortion) uses the maximum demand load current (IL) as\n"
            "the denominator — the same value regardless of instantaneous load.  IEEE 519\n"
            "uses TDD for current limits, which prevents a lightly-loaded VFD from\n"
            "appearing non-compliant simply because it is running at 20% load.",
        )

        concept(
            "ISC / IL Ratio — What It Drives",
            "ISC is the available short-circuit current at the PCC; IL is the maximum\n"
            "12-month demand current.  A higher ratio means a stiffer source, which can\n"
            "absorb more harmonic current without voltage distortion.  IEEE 519-2022\n"
            "Table 2 current TDD limits by ISC/IL:\n"
            "\n"
            "   < 20    →  TDD ≤  5%  (most residential / small commercial)\n"
            "   20–50   →  TDD ≤  8%\n"
            "   50–100  →  TDD ≤ 12%\n"
            "   100–1000 → TDD ≤ 15%\n"
            "   > 1000  →  TDD ≤ 20%\n"
            "\n"
            "If ISC is unknown, this tool falls back to a flat 5% THD limit.",
        )

        concept(
            "ANSI C84.1 Voltage Bands",
            "Range A: steady-state service voltage should remain within ±5% of nominal\n"
            "(114–126 V on a 120 V system; 456–504 V on a 480 V system).  Utilities are\n"
            "expected to supply within Range A under normal conditions.\n"
            "\n"
            "Range B: 110–127 V on a 120 V base — −8.33% below nominal but only +5.83%\n"
            "above it.  It covers voltages that result from practical design and\n"
            "operating conditions.  C84.1 asks that excursions into Range B be limited in\n"
            "extent, frequency and duration, and corrected within a reasonable time, so a\n"
            "sustained Range B voltage is a finding even though it is not outside the\n"
            "standard.\n"
            "\n"
            "Systems over 600 V (2.4–34.5 kV) get their own row, and its lower limits are\n"
            "tighter: Range A 97.5–105%, Range B 95–105.8%.  The upper limits are the\n"
            "same.  The extra 2.5% on the low side is reserved for the drop through the\n"
            "customer's own transformation, which sits below a primary meter.\n"
            "\n"
            "Both are *service* voltage, measured at the point of delivery, which is where\n"
            "this meter sits.  C84.1 states a second, wider set for utilization voltage at\n"
            "the equipment terminals; that one allows for the customer's wiring drop and\n"
            "is not what this tool applies.",
        )

        concept(
            "Split-Phase Service (120/240 V)",
            "A residential or small-commercial split-phase service has two energized\n"
            "conductors (L1 and L2), each 120 V to neutral, and 240 V L1-to-L2.\n"
            "The Pronto meter records Van (L1-N), Vbn (L2-N), Ia (L1 current),\n"
            "Ib (L2 current), and In (neutral current) as separate channels.\n"
            "\n"
            "Enter 120 V as the nominal voltage — the tool automatically recognizes\n"
            "the split-phase topology and applies the correct ANSI bands.",
        )

        concept(
            "Harmonic Load Signatures",
            "Different load types produce characteristic harmonic patterns:\n"
            "\n"
            "  6-pulse VFD / rectifier: H5 dominant, H7 second (6k±1 pattern);\n"
            "     very low H3.  H5/H7 ratio ≈ 1.5–3.\n"
            "\n"
            "  12-pulse rectifier: H11/H13 dominant; H5/H7 largely cancelled.\n"
            "\n"
            "  Single-phase SMPS (PCs, LED drivers): H3 > H5; strong triplens;\n"
            "     elevated neutral current (In ≈ Ia).\n"
            "\n"
            "  Arc furnace / welder: high interval-to-interval variability;\n"
            "     significant even harmonics (H2, H4).\n"
            "\n"
            "  Saturated transformer: H3 and H5 dominant on all three phases\n"
            "     simultaneously; H3/H5 > 1.5.",
        )

        # ── Analysis Methods & Diagnostics ────────────────────────────────
        # ── Neutral integrity ──────────────────────────────────────────────
        # Written to be read start to finish. The neutral is the one part of a
        # service where the correct interpretation flips depending on the
        # secondary configuration, and where the same measurement can mean
        # "healthy" or "failing" depending on which service you are looking at.
        use(pg_neutral)
        section("Neutral Integrity — Theory and Diagnostics")
        lead(
            "Split-phase only, and the one failure mode that damages equipment\n"
            "rather than merely annoying people. The theory first, then what the\n"
            "tool measures and what each reading rules in or out."
        )

        concept(
            "1. What the neutral actually carries",
            "The neutral is the return path for whatever the phase conductors do not\n"
            "return to each other.  It does not carry the arithmetic sum of the phase\n"
            "currents — it carries their vector sum, and the angles between the phases\n"
            "decide the result.  Almost everything confusing about neutrals follows\n"
            "from that one fact.\n"
            "\n"
            "Three secondary configurations matter here, and they behave differently:\n"
            "\n"
            "  Split phase 120/240   One center-tapped single-phase transformer.\n"
            "                        The two legs are 180° apart.\n"
            "  Three phase 120/208   A three-phase transformer, all three phases taken.\n"
            "                        The phases are 120° apart.\n"
            "  Single phase 120/208  The same three-phase transformer, but the customer\n"
            "                        pulls only two legs.  Still 120° apart.\n"
            "\n"
            "The last two share a transformer.  The difference is only how many wires\n"
            "the customer pulls — which is exactly why the neutral behaves differently.",
        )

        concept(
            "2. Why balanced load does not mean zero neutral current",
            "Take equal current I on every leg and add the phasors:\n"
            "\n"
            "  Split phase, legs 180° apart\n"
            "     I∠0° + I∠180° = 0.        Neutral current ≈ 0.\n"
            "     The legs oppose, so a balanced service returns nothing on the neutral.\n"
            "\n"
            "  Three phase, three legs 120° apart\n"
            "     I∠0° + I∠120° + I∠240° = 0.   Neutral current ≈ 0.\n"
            "     All three cancel — the classic balanced wye result.\n"
            "\n"
            "  Single phase 120/208, two legs 120° apart\n"
            "     I∠0° + I∠120° = I∠60°, magnitude I.   Neutral current ≈ I.\n"
            "     Two of three do not cancel.  The neutral carries a full leg's worth\n"
            "     of current even when the load is perfectly balanced.\n"
            "\n"
            "This is the single most misread result in this tool.  A 120/208 service\n"
            "with a neutral at 100% of leg current is behaving exactly as designed.\n"
            "The same reading on a 120/240 service means something is badly wrong.\n"
            "\n"
            "Practical consequence: on a 120/208 single-phase service the neutral must\n"
            "be sized as a full current-carrying conductor.  It is not a \"return only\n"
            "the imbalance\" conductor the way a split-phase neutral is.",
        )

        concept(
            "3. Harmonics — why triplens break the cancellation",
            "Phase angle multiplies with harmonic order.  A conductor whose fundamental\n"
            "sits at angle θ has its h-th harmonic at h × θ.  Work that through:\n"
            "\n"
            "  Split phase, legs 180° apart\n"
            "     3rd harmonic: 3 × 180° = 540° ≡ 180°.\n"
            "     Still opposed, so triplens subtract in the neutral just as the\n"
            "     fundamental does.  A split-phase neutral does not accumulate triplens.\n"
            "\n"
            "  Any wye, legs 120° apart\n"
            "     3rd harmonic: 3 × 120° = 360° ≡ 0°.\n"
            "     The triplens land in phase with each other and ADD arithmetically.\n"
            "     H3, H9, H15 are \"zero-sequence\" for exactly this reason.\n"
            "\n"
            "So on any wye-derived service — full three-phase or two legs of one —\n"
            "single-phase nonlinear loads (computers, LED drivers, switching supplies)\n"
            "pile their 3rd harmonic into the shared neutral.  Neutral current can\n"
            "exceed phase current, which is why NEC requires the neutral of such a\n"
            "circuit to be treated as a current-carrying conductor.\n"
            "\n"
            "Reading it: on a 120/208 service, neutral at roughly leg current is the\n"
            "geometry.  Neutral ABOVE leg current is the harmonics on top of it.",
        )

        concept(
            "4. The open neutral — what physically happens",
            "An open or high-resistance neutral is the failure this section exists to\n"
            "find, and the mechanism is worth understanding because it explains every\n"
            "symptom.\n"
            "\n"
            "With the neutral intact, each leg's load is fed from its own 120 V source.\n"
            "The legs are independent, and the neutral carries the difference.\n"
            "\n"
            "Open the neutral and the two loads are suddenly in SERIES across the\n"
            "line-to-line voltage.  The point where they meet — what used to be the\n"
            "neutral — is no longer tied to anything.  It floats to wherever the two\n"
            "load impedances divide the L-L voltage.\n"
            "\n"
            "  The lightly loaded leg (high impedance) takes most of the voltage and\n"
            "     rises toward the full L-L value.\n"
            "  The heavily loaded leg (low impedance) collapses toward zero.\n"
            "\n"
            "That is the damage mechanism: equipment on the lightly loaded side sees\n"
            "an overvoltage that can approach 240 V (or 208 V) on a 120 V circuit.\n"
            "It also explains the signatures below — as load shifts between legs, the\n"
            "floating midpoint moves, so one leg rises exactly as the other falls.",
        )

        concept(
            "5. The five diagnostics, and what each can and cannot see",
            "The tool combines five independent indicators.  None is conclusive alone;\n"
            "the value is in which ones agree.\n"
            "\n"
            "  a) Cross-leg correlation (Pearson r between L1 and L2)\n"
            "     Healthy: both legs move together as the transformer loads and unloads,\n"
            "       so r > 0.8.\n"
            "     Open neutral: the floating midpoint means one leg rises as the other\n"
            "       falls, driving r toward −1.\n"
            "     This is the primary open-neutral test, and it works on both 120/240\n"
            "       and 120/208 services.\n"
            "     Blind spot: needs load variation.  A perfectly steady service gives\n"
            "       no correlation to measure, and the tool reports it as not computable\n"
            "       rather than guessing.\n"
            "\n"
            "  b) Voltage sum (L1 + L2) — and this one depends on the service\n"
            "     On 120/240, the legs are collinear, so L1 + L2 = the line-to-line\n"
            "       voltage, 240 V.  Open the neutral and the loads sit in series across\n"
            "       that same 240 V — so the sum is STILL 240 V.  On a split-phase\n"
            "       service the sum tells you nothing about the neutral.  A rock-steady\n"
            "       240 V is not evidence the neutral is sound.\n"
            "     On 120/208, a healthy service reads 120 + 120 = 240 V, but an open\n"
            "       neutral puts the loads in series across the line-to-line 208 V, so\n"
            "       the sum COLLAPSES toward 208.  Here the sum is a real discriminator.\n"
            "     The tool applies this per configuration and says which case it is in.\n"
            "\n"
            "  c) Voltage asymmetry |L1 − L2|\n"
            "     Sustained asymmetry means the legs are unequally loaded, the neutral\n"
            "       has resistance, or both.  Useful, but it cannot separate ordinary\n"
            "       unbalanced loading from a degrading neutral on its own.\n"
            "\n"
            "  d) Neutral-to-earth voltage (Vne)\n"
            "     The most direct measurement of neutral impedance available, when the\n"
            "       meter records it.  Current through a resistive neutral develops a\n"
            "       voltage along it, and Vne is what that looks like from the far end.\n"
            "     Under 0.5 V is normal.  Above 2 V indicates significant impedance.\n"
            "       Above 5 V is a shock hazard — the \"grounded\" metal in the premises\n"
            "       is no longer at earth potential.\n"
            "\n"
            "  e) Coincident opposing sag/swell\n"
            "     One leg drops below 90% while the other simultaneously exceeds 110%.\n"
            "     This is the open-neutral signature in its clearest form and needs\n"
            "       cycle-level (adaptive) records to catch.",
        )

        concept(
            "6. Assessing a neutral in the field",
            "A practical order of operations:\n"
            "\n"
            "  1. Establish the configuration first.  Everything above depends on it.\n"
            "     Set Service Type before reading any neutral result.\n"
            "\n"
            "  2. Ask whether the neutral current is explained by geometry.\n"
            "     120/240: balanced load should give a near-zero neutral.\n"
            "     120/208 two-leg: balanced load gives a neutral near full leg current.\n"
            "     Only current beyond that needs an explanation.\n"
            "\n"
            "  3. If the neutral is high, separate imbalance from harmonics.\n"
            "     Compare the neutral's H3 content against the phases.  Triplen-dominated\n"
            "     neutral current is a harmonics problem — the fix is load-side, not a\n"
            "     wiring fault.  Broadband neutral current tracking load imbalance is a\n"
            "     balancing problem.\n"
            "\n"
            "  4. Suspect an open neutral when the legs move in opposition.\n"
            "     Negative cross-leg correlation, coincident opposing sag/swell, or\n"
            "     elevated Vne.  Any one warrants a physical inspection; two together\n"
            "     make it likely.\n"
            "\n"
            "  5. Inspect from the meter socket outward.  Open and high-resistance\n"
            "     neutrals are overwhelmingly connection failures — socket jaws, the\n"
            "     service-entrance lug, the drop connector — not conductor failures.",
        )

        concept(
            "7. What this analysis cannot tell you",
            "Stated plainly so the results are not over-read:\n"
            "\n"
            "  It cannot locate the fault.  Every indicator is measured at one point,\n"
            "    so it describes the neutral between the meter and the source as a\n"
            "    whole.  It cannot distinguish a bad socket jaw from a bad drop.\n"
            "\n"
            "  It cannot see a neutral problem that never manifests.  A high-resistance\n"
            "    neutral only shows up under load imbalance.  A recording taken during\n"
            "    a quiet week may look clean on a neutral that fails in the evening.\n"
            "\n"
            "  On a 120/240 service the voltage sum contributes nothing, so the finding\n"
            "    rests on correlation, asymmetry and Vne.  If load was steady and Vne\n"
            "    was not recorded, the tool has very little to go on and says so.\n"
            "\n"
            "  On a 120/208 service it sees only two of the three phases, so it cannot\n"
            "    assess unbalance at the transformer itself — only the difference\n"
            "    between the two legs this customer is served from.\n"
            "\n"
            "  Severity here is an engineering judgment built from indicators that\n"
            "    agree, not a standards compliance result.  There is no ANSI or IEEE\n"
            "    limit for \"neutral health\"; the thresholds are practical ones.",
        )

        use(pg_methods)
        section("Analysis Methods & Diagnostics")
        lead(
            "How individual checks are computed, in the order the report presents\n"
            "them. Useful when you need to defend a number or explain one."
        )

        concept(
            "Voltage Compliance — ANSI C84.1",
            "Reports what percentage of recording intervals fall inside Range A, inside\n"
            "Range B, or outside both.  On a 120 V base the service-voltage ranges are\n"
            "114–126 V (Range A, ±5%) and 110–127 V (Range B, −8.33%/+5.83%); both scale\n"
            "with the nominal.  Range B is not symmetric — the standard tolerates the\n"
            "drop of a long or loaded feeder further than it tolerates overvoltage.\n"
            "\n"
            "The verdict is taken from the interval average.  C84.1 rates sustained\n"
            "service voltage: an excursion shorter than one interval is a sag or swell,\n"
            "and those are graded on depth and duration against the ITIC envelope under\n"
            "Voltage Events.  Where the meter's max-min record exists, the within-interval\n"
            "extremes are still read and reported beside the averages, labelled as such.\n"
            "\n"
            "Split-phase services (120/240 V, no voltage_c channel) are automatically\n"
            "detected and voltage_a / voltage_b are evaluated independently against 120 V\n"
            "bands.  Three-phase services evaluate all three phases against the nominal\n"
            "L-N voltage derived from the entered nominal value.  A primary-metered\n"
            "service takes its nominal from the Primary voltage field instead: nothing in\n"
            "the meter file names the primary voltage, so it is entered, not inferred.\n"
            "\n"
            "Above 600 V, C84.1 Table 1 uses a separate group, and it is TIGHTER below\n"
            "nominal, not looser: Range A is 97.5–105% and Range B is 95–105.8%.  A\n"
            "primary-metered customer still has their own transformation between the\n"
            "meter and their equipment, and the standard reserves that headroom for the\n"
            "drop through it.  Table 1 stops at 34.5 kV; above that no C84.1 range is\n"
            "claimed.",
        )

        concept(
            "THD / TDD — IEEE 519-2022 Basic Check",
            "Evaluates the average THD (voltage) and TDD (current) over the full recording\n"
            "against the applicable IEEE 519-2022 limits.\n"
            "\n"
            "Current TDD limit is determined by the ISC/IL ratio entered in the tool:\n"
            "  ISC/IL < 20     →  TDD ≤  5%\n"
            "  ISC/IL 20–50    →  TDD ≤  8%\n"
            "  ISC/IL 50–100   →  TDD ≤ 12%\n"
            "  ISC/IL 100–1000 →  TDD ≤ 15%\n"
            "  ISC/IL > 1000   →  TDD ≤ 20%\n"
            "\n"
            "If ISC is not entered, the tool falls back to a flat 5% limit (most\n"
            "conservative).  Enter ISC in amps to get the correct limit for the service.",
        )

        concept(
            "IEEE 519 Statistical Compliance (P95 / P99)",
            "IEEE 519-2022 Clause 5 specifies compliance is measured statistically, not\n"
            "by instantaneous values.  This tool evaluates three windows:\n"
            "\n"
            "  ST weekly (primary):  P95 of 7-day window ≤ 1.0× limit\n"
            "                        P99 of 7-day window ≤ 1.5× limit\n"
            "  VST daily:            daily P99 ≤ 2.0× limit\n"
            "\n"
            "The meter exports 5-minute interval averages, used here as a proxy for the\n"
            "standard's 10-minute Short Time (ST) measurement.  True 3-second Very Short\n"
            "Time (VST) data is not available from this format; daily P99 of 5-minute\n"
            "data is a conservative lower bound but may miss sub-minute peaks.\n"
            "\n"
            "If the recording is shorter than 7 days, percentiles are computed over the\n"
            "full recording period and noted as such.  This check requires ISC to be\n"
            "entered.",
        )

        concept(
            "Per-Order Harmonic Spectrum",
            "IEEE 519-2022 Table 1 limits per-order voltage harmonics to 5% of nominal\n"
            "for systems below 1 kV.  The tool checks each available harmonic order\n"
            "(H3 through H13, or higher if present) against this limit.\n"
            "\n"
            "Table 2 individual current harmonic limits are also checked per order.\n"
            "Failing orders are listed with their mean value, limit, and margin.\n"
            "\n"
            "Per-order harmonic columns in the meter data are named h5_current_a,\n"
            "h7_current_b, h3_voltage_a, etc.  Not all meters export per-order data;\n"
            "if only thd_current_* is available, per-order checks are skipped.",
        )

        concept(
            "Neutral Harmonic Analysis",
            "Analyzes triplen harmonic current (H3, H9, H15 — zero-sequence orders) in\n"
            "the neutral conductor.  In 4-wire wye systems, triplen harmonics from each\n"
            "phase add rather than cancel in the neutral, so neutral current can reach or\n"
            "exceed phase current at heavy SMPS/LED loads.\n"
            "\n"
            "Outputs:\n"
            "  Triplen fraction: what share of total neutral harmonic current is triplen\n"
            "  Dominant order: which triplen order (H3, H9, H15) is largest\n"
            "  Accumulation factor: ratio of neutral harmonic sum to phase H3 — values\n"
            "    significantly above 1.0 confirm neutral accumulation is occurring\n"
            "\n"
            "Requires h3_current_neutral (and h9/h15 if available) in the meter data.",
        )

        concept(
            "Harmonic Source Attribution (Impedance Method)",
            "Estimates the apparent harmonic impedance Z_h at each order:\n"
            "\n"
            "  Z_h = mean(V_h) / mean(I_h)   [Ω, per order, averaged across phases]\n"
            "\n"
            "A purely inductive (utility) source has Z proportional to harmonic order\n"
            "(Z_h = a×h).  The tool fits this linear model and computes a Z-ratio:\n"
            "\n"
            "  Z_ratio = Z_h / (a × h)\n"
            "\n"
            "  Z_ratio > 2.5 at any order → parallel resonance suspected\n"
            "\n"
            "Attribution uses Pearson correlation between the V_h and I_h time series:\n"
            "  corr > 0.50 → 'customer'  (V and I vary together → load drives both)\n"
            "  corr ≤ 0.50 → 'indeterminate'\n"
            "\n"
            "This is an indicative heuristic only.  Exact source direction requires\n"
            "waveform phasor data.  Requires both per-order voltage and current harmonic\n"
            "channels at the same orders.",
        )

        concept(
            "Harmonic Load Signature Detection",
            "Scores the measured harmonic spectrum against 14 reference load-type\n"
            "signatures using a combined similarity metric:\n"
            "\n"
            "  55%  cosine similarity on spectral shape (H3/H5/H7/H9/H11/H13)\n"
            "  30%  log-ratio match on H5/H7 ratio (best discriminator for VFDs)\n"
            "  15%  log-ratio match on H3/H5 ratio (SMPS vs VFD separator)\n"
            "\n"
            "A variability modifier adjusts the score based on H5 inter-interval\n"
            "coefficient of variation (CV).  Steady-state load types (VFDs, SMPS) are\n"
            "penalised if CV > 0.30; intermittent types (welders, arc furnaces) are\n"
            "penalised if CV < 0.25.\n"
            "\n"
            "Matches above 75% similarity are reported; matches above 85% are 'medium'\n"
            "confidence and above 95% are 'high' confidence.  Up to three matches\n"
            "are shown (best match + contributing loads).\n"
            "\n"
            "Recognized load types:\n"
            "  6-pulse VFD with reactor    6-pulse VFD without reactor\n"
            "  12-pulse rectifier          18-pulse / active front-end drive\n"
            "  SMPS (computers/servers)    Fluorescent (magnetic ballast)\n"
            "  LED drivers (no PFC)        EV charger (Level 2)\n"
            "  UPS (6-pulse double-conv.)  Arc welder / resistance welder\n"
            "  Electric arc furnace        Transformer saturation\n"
            "  DC fast charger (DCFC)      Mixed VFD + SMPS",
        )

        concept(
            "K-Factor — Transformer Derating for Harmonic Loads",
            "K-factor quantifies how much a nonlinear load will thermally stress a\n"
            "standard (K-1) transformer:\n"
            "\n"
            "  K = Σ(Ih² × h²) / Σ(Ih²)\n"
            "\n"
            "where Ih is the harmonic current amplitude at order h as a fraction of\n"
            "fundamental.  A pure sinusoidal load gives K = 1.  VFD-heavy loads\n"
            "typically produce K = 4–8; SMPS-dominant loads can reach K = 13+.\n"
            "\n"
            "A standard transformer should be derated when the measured K-factor\n"
            "exceeds its design rating (usually 1).  K-rated transformers (K-4, K-13,\n"
            "K-20) are designed for harmonic-heavy loads.  The kfactor_meter channel\n"
            "in the Pronto meter records K-factor directly.",
        )

        concept(
            "Flicker — Pst and Plt",
            "Flicker severity is measured by IEC 61000-4-15 and evaluated against\n"
            "IEEE 1453-2022 / IEC 61000-3-7 planning levels.\n"
            "\n"
            "  Pst (short-term, 10-minute):  limit 1.0 at the PCC\n"
            "  Plt (long-term, 2-hour):       limit 0.65 at the PCC\n"
            "\n"
            "The tool reports the 95th-percentile Pst and Plt over the recording,\n"
            "the number of intervals exceeding the limit, and the worst-case value.\n"
            "Common sources of flicker: large motor starts, arc welders, arc furnaces,\n"
            "wind turbines, and intermittently switched capacitor banks.",
        )

        concept(
            "Voltage & Current Imbalance",
            "Voltage imbalance uses the NEMA MG1 definition:\n"
            "  Vu = max |Vphase − Vavg| / Vavg × 100  (%)\n"
            "\n"
            "IEEE 1159 recommends flagging above 3%.  Motor nameplate derating begins\n"
            "at 1% and accelerates rapidly above 3%.\n"
            "\n"
            "Current imbalance uses the PSCo procedure:\n"
            "  Iu = max |Iphase − Iavg| / Iavg × 100  (%)\n"
            "\n"
            "PSCo Blue Book limit: 10%.  Tariff Sheet R123 does not limit imbalance: above\n"
            "15% phase-to-phase it allows the Company to take billing demand from the\n"
            "worst phase.  Both thresholds are evaluated, but only the first is a limit.",
        )

        concept(
            "Demand & Transformer Loading",
            "Reports peak demand (kW), average demand, load factor (avg/peak), and\n"
            "estimated transformer utilization as a percentage of nameplate kVA.\n"
            "\n"
            "Transformer nameplate kVA is entered in the tool's threshold settings.\n"
            "Loading above 80% of nameplate is flagged as high; above 100% is critical.\n"
            "Both of these thresholds are before any K-factor derating — a K-rated\n"
            "transformer may carry more; a K-1 serving harmonic loads is effectively\n"
            "derated below nameplate.",
        )

        concept(
            "Event Detection",
            "Detects discrete voltage and current events from the interval data or,\n"
            "when available, from the adaptive (cycle-level) record.\n"
            "\n"
            "Detected event types:\n"
            "  Voltage sag    — instantaneous voltage < 90% of nominal\n"
            "  Voltage swell  — instantaneous voltage > 110% of nominal\n"
            "  Current step   — sudden change in current magnitude between intervals\n"
            "\n"
            "The adaptive record (when present) provides cycle-resolution (~16.7 ms)\n"
            "data and enables detection of shorter events that 5-minute averaging\n"
            "would hide.  Events are plotted against the ITIC curve to assess whether\n"
            "equipment immunity standards require the load to tolerate the event.",
        )

        concept(
            "Engineering Assessment (Likely Causes)",
            "After all individual checks run, the tool synthesizes findings into a\n"
            "likely-cause list ranked by severity.  Each finding includes:\n"
            "\n"
            "  Category     — what domain (voltage, harmonics, power factor, etc.)\n"
            "  Severity      — critical, warning, or info\n"
            "  Finding       — what was measured and why it matters\n"
            "  Cause         — probable physical explanation\n"
            "  Responsibility — customer, utility, or shared\n"
            "  Recommendation — specific corrective action\n"
            "\n"
            "Harmonic signature matches are folded into root cause findings.  If a\n"
            "load type is identified with medium or high confidence, the finding text\n"
            "references that specific load type and its recommended mitigation.",
        )

        for _pg in (pg_start, pg_standard, pg_tariff, pg_refs, pg_workflow,
                    pg_concepts, pg_neutral, pg_methods):
            _pg.config(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy,
                   style="Plain.TButton", cursor="hand2").pack(pady=(0, 14))

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log_write(self, text, tag=None):
        def _write():
            self._log.config(state="normal")
            if tag:
                self._log.insert("end", text, tag)
            else:
                self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _write)

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


def _claim_windows_taskbar_identity() -> None:
    """Give this app its own taskbar identity on Windows.

    The taskbar groups windows by AppUserModelID and draws that group's icon.
    A script launched through pythonw.exe inherits Python's ID, so the taskbar
    button shows the Python logo no matter what icon the window itself carries
    -- which is what "the icon works on the Mac but not on Windows" looks like.

    Has to run before the first window is mapped, so it is called here rather
    than from inside the window's own setup.

    No-op everywhere else, and never fatal: an unrecognised shell32 or a locked
    down host costs an icon, not a session.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "XcelEnergy.PQAnalyzer")
    except Exception as exc:
        _logging.getLogger(__name__).debug(
            "Could not set the taskbar AppUserModelID (%s); the taskbar may "
            "show the Python icon.", exc)


if __name__ == "__main__":
    _claim_windows_taskbar_identity()
    app = PQApp()
    app.mainloop()

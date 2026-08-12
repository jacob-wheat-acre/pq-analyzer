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
_ISC_FG    = "#1a6fbf"   # blue for auto-populated ISC
_ISC_NONE  = "#888888"   # grey when no ISC resolved

# ── PSCo tariff schedule → CLI key mapping ───────────────────────────────────
_SCHEDULE_KEY = {
    "Schedule R — Residential":               "r",
    "Schedule C — Small Commercial  (< 50 kW)": "c",
    "Schedule SG — C&I Secondary  (≥ 50 kW)":  "sg",
    "Schedule PG — C&I Primary":              "pg",
}

# ── Which version is running ─────────────────────────────────────────────────
# Read on its own, ahead of the engine import below and out of its try/except:
# when that import fails, the version is the first thing anyone asks for, and a
# copy that cannot say what it is cannot be told apart from a current one.
# pq_constants is pure data with no third-party imports, so it loads even on an
# install where python-docx or matplotlib is missing.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from pq_constants import __version__ as _ENGINE_VERSION
except Exception:
    _ENGINE_VERSION = "unknown"

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
                "_eng_phone_var", "_eng_email_var"}


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
        self._set_icon()
        self._build_ui()
        self._running = False

    def _set_icon(self):
        icon_dir = Path(__file__).parent
        try:
            if sys.platform == "win32":
                ico = icon_dir / "icon.ico"
                if ico.exists():
                    self.iconbitmap(str(ico))
            else:
                png = icon_dir / "icon.png"
                if png.exists():
                    from PIL import Image, ImageTk
                    img = Image.open(png).resize((64, 64), Image.LANCZOS)
                    self._tk_icon = ImageTk.PhotoImage(img)
                    self.iconphoto(True, self._tk_icon)
        except Exception:
            pass  # icon is cosmetic — never block startup

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
        tk.Button(file_frame, text="Browse…", command=self._browse,
                  font=_FONT_UI).pack(side="left")

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
                 text="(R: no PF clause  |  C/SG: ≥ 0.90 Sheet R73  |  PG: near unity Sheet R121)",
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

        # ── Service conductor row ──────────────────────────────────────────────
        # The run between that transformer and the meter. With it the measured
        # service impedance gets an expected value to be compared against;
        # without it the measurement still stands, uncompared.
        cond_frame = tk.Frame(self, bg=_BG)
        cond_frame.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(cond_frame, text="Service conductor", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")

        self._conductor_labels = {label: key for key, label in conductor_options()}
        self._conductor_var = tk.StringVar(value=_CONDUCTOR_NONE)
        self._conductor_combo = ttk.Combobox(
            cond_frame, textvariable=self._conductor_var, state="readonly",
            values=[_CONDUCTOR_NONE] + list(self._conductor_labels.keys()),
            width=32, font=_FONT_UI,
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

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=12, pady=(4, 0))

        # ── Report details section (collapsible) ──────────────────────────────
        det_hdr = tk.Frame(self, bg=_BG)
        det_hdr.pack(fill="x", padx=12, pady=(4, 0))

        self._details_open = tk.BooleanVar(value=False)
        self._det_toggle_btn = tk.Button(
            det_hdr, text="▶  Report Details (site address, engineer, feeder…)",
            command=self._toggle_details,
            bg=_BG, fg="#555555", font=_FONT_UI_S,
            relief="flat", cursor="hand2", anchor="w",
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

        # ── Site details ───────────────────────────────────────────────────────
        tk.Label(self._details_frame,
                 text="  Site", bg=_BG, fg="#555555", font=_FONT_UI_S,
                 ).pack(anchor="w", padx=12, pady=(6, 0))

        self._meter_id_var = tk.StringVar()
        self._feeder_var   = tk.StringVar()
        self._subst_var    = tk.StringVar()

        _detail_row("Meter / Account #", self._meter_id_var, "(Pronto meter ID or account #)")
        _detail_row("Feeder / Circuit", self._feeder_var,   "(e.g. FDR-4203)")
        _detail_row("Substation",       self._subst_var,    "(e.g. Sheridan 115/13 kV)")

        ttk.Separator(self._details_frame, orient="horizontal").pack(
            fill="x", padx=12, pady=(6, 0))

        # ── Engineer / sign-off ────────────────────────────────────────────────
        tk.Label(self._details_frame,
                 text="  Engineer sign-off", bg=_BG, fg="#555555", font=_FONT_UI_S,
                 ).pack(anchor="w", padx=12, pady=(4, 0))

        self._eng_name_var  = tk.StringVar()
        self._eng_title_var = tk.StringVar()
        self._eng_phone_var = tk.StringVar()
        self._eng_email_var = tk.StringVar()

        _detail_row("Name",  self._eng_name_var,  "(e.g. Jacob Whitaker)")
        _detail_row("Title", self._eng_title_var, "(default: Electric Area Engineer)")
        _detail_row("Phone", self._eng_phone_var, "(e.g. 303-555-0100)")
        _detail_row("Email", self._eng_email_var, "(e.g. jwhitaker@xcelenergy.com)")

        tk.Frame(self._details_frame, bg=_BG, height=6).pack()  # bottom padding

        # ── Divider + Run button ───────────────────────────────────────────────
        self._sep_before_run = ttk.Separator(self, orient="horizontal")
        self._sep_before_run.pack(fill="x", padx=12, pady=4)

        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill="x", padx=12, pady=4)

        self._run_btn = tk.Button(
            btn_frame, text="Run Analysis",
            command=self._run,
            bg=_BTN_RUN, fg=_BTN_TXT, activebackground="#155a9e",
            font=_FONT_UI_B,
            relief="flat", cursor="hand2", padx=20, pady=8,
        )
        self._run_btn.pack(side="left")

        self._open_btn = tk.Button(
            btn_frame, text="Open Output Folder",
            command=self._open_folder,
            font=_FONT_UI, relief="flat", cursor="hand2",
            bg=_BG, padx=12, pady=8,
        )
        self._open_btn.pack(side="left", padx=(12, 0))
        self._open_btn.config(state="disabled")

        tk.Button(
            btn_frame, text="Clear All",
            command=self._clear_all,
            font=_FONT_UI, relief="flat", cursor="hand2",
            bg=_BG, padx=12, pady=8,
        ).pack(side="left", padx=(8, 0))

        tk.Button(
            btn_frame, text="? Help",
            command=self._show_help,
            font=_FONT_UI, relief="flat", cursor="hand2",
            bg=_BG, fg="#555555", padx=12, pady=8,
        ).pack(side="right")

        tk.Button(
            btn_frame, text="✉ Feedback",
            command=self._show_feedback,
            font=_FONT_UI, relief="flat", cursor="hand2",
            bg=_BG, fg="#555555", padx=12, pady=8,
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

        The engineer's own name, title, phone and email are left alone: they
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
                "the transformer and conductor pickers, and the site details "
                "— back to their defaults.\n\nYour engineer name, title, phone "
                "and email are kept.\n\nClear the rest?"):
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
                text="▶  Report Details (site address, engineer, feeder…)")
        else:
            self._details_frame.pack(fill="x", before=self._sep_before_run)
            self._details_open.set(True)
            self._det_toggle_btn.config(
                text="▼  Report Details (site address, engineer, feeder…)")

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

    def _refresh_kva_options(self):
        """Rebuild kVA combo list for the current type + nominal voltage."""
        key = self._xfmr_type_key

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
        tk.Button(btns, text="Show the error details",
                  command=self._show_import_error,
                  font=_FONT_UI_S).pack(side="left")
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

        tk.Button(win, text="Copy to clipboard", command=_copy,
                  font=_FONT_UI).pack(pady=(0, 8))

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

        # Transformer kVA
        xfmr_key = self._xfmr_type_key
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
            "site":           self._site_var.get().strip(),
            "address":        self._address_var.get().strip(),
            "meter_id":       self._meter_id_var.get().strip(),
            "feeder":         self._feeder_var.get().strip(),
            "substation":     self._subst_var.get().strip(),
            "engineer":       self._eng_name_var.get().strip(),
            "engineer_title": self._eng_title_var.get().strip(),
            "engineer_phone": self._eng_phone_var.get().strip(),
            "engineer_email": self._eng_email_var.get().strip(),
            "xfmr_key":       xfmr_key,
            "kva":            kva,
            "isc_amps":       isc_amps,
            # The service-type and topology pickers decide how many phases the
            # report and its charts describe; without these the plots fall back
            # to guessing from which channels happen to be present.
            "topology":       self._topo_var.get(),
            "conductor_key":  self._conductor_labels.get(self._conductor_var.get()),
            "run_length_ft":  run_length_ft,
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
            isc_amps=isc_amps,
            isc_source=isc_source,
            transformer_kva=kva,
            service_type=params.get("xfmr_key"),
            topology=params.get("topology", "auto"),
            conductor_key=params.get("conductor_key"),
            run_length_ft=params.get("run_length_ft"),
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
            engineer_contact="",
            outdir=outdir,
            stem=stem,
            meter_id=params["meter_id"],
            feeder=params["feeder"],
            substation=params["substation"],
            engineer_title=params["engineer_title"],
            engineer_phone=params["engineer_phone"],
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
            engineer_phone=params["engineer_phone"],
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

        tk.Button(btn_row, text="Send via Email", command=_send,
                  font=_FONT_UI, relief="flat", cursor="hand2",
                  bg=_BTN_RUN, fg=_BTN_TXT, padx=14, pady=7,
                  ).pack(side="left")
        tk.Button(btn_row, text="Cancel", command=win.destroy,
                  font=_FONT_UI, relief="flat", cursor="hand2",
                  bg=_BG, fg="#555555", padx=14, pady=7,
                  ).pack(side="left", padx=(8, 0))

    # ── Help window ───────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self)
        win.title("PQ Analyzer — Reference Guide")
        win.configure(bg=_BG)
        win.resizable(True, True)
        win.minsize(640, 560)

        # Header bar
        hdr = tk.Frame(win, bg=_BTN_RUN)
        hdr.pack(fill="x")
        tk.Label(hdr, text="PQ Analyzer — Reference & Standards",
                 bg=_BTN_RUN, fg="white", font=_FONT_UI_B,
                 pady=10, padx=16).pack(anchor="w")

        # Scrollable content
        outer = tk.Frame(win, bg=_BG)
        outer.pack(fill="both", expand=True, padx=16, pady=10)

        _f0, _fs = _FONT_UI[0], _FONT_UI[1]
        txt = tk.Text(outer, bg=_BG, fg=_LABEL_FG, font=_FONT_UI,
                      relief="flat", wrap="word", cursor="arrow",
                      state="normal", padx=6, pady=4)
        sb = ttk.Scrollbar(outer, command=txt.yview)
        txt["yscrollcommand"] = sb.set
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)

        txt.tag_config("h1",   font=(_f0, _fs, "bold"), foreground=_BTN_RUN,
                       spacing1=10, spacing3=2)
        txt.tag_config("rule", font=(_f0, 8),  foreground="#cccccc")
        txt.tag_config("h2",   font=(_f0, _fs, "bold"), foreground="#333333",
                       spacing1=8, spacing3=1, lmargin1=12, lmargin2=12)
        txt.tag_config("body", font=_FONT_UI,  foreground="#555555",
                       lmargin1=24, lmargin2=24, spacing3=2)
        txt.tag_config("link", font=(_f0, _fs-1), foreground=_BTN_RUN, underline=True)
        txt.tag_config("note", font=(_f0, _fs-1), foreground="#999999",
                       lmargin1=24, lmargin2=24)

        _link_map = {}

        def _add_link(label, url):
            tag = f"_lnk{len(_link_map)}"
            _link_map[tag] = url
            txt.insert("end", label, ("link", tag))
            txt.tag_bind(tag, "<Enter>",    lambda e: txt.config(cursor="hand2"))
            txt.tag_bind(tag, "<Leave>",    lambda e: txt.config(cursor="arrow"))
            txt.tag_bind(tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))

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

        # ── IEEE Standards ─────────────────────────────────────────────────
        section("IEEE / ANSI Standards")

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

        txt.insert("end", "  ANSI C84.1-2020 — Electric Power Systems Voltage Ratings\n", "h2")
        txt.insert("end",
                   "  Defines Range A (normal operating band, ±5% of nominal) and Range B\n"
                   "  (occasional excursions).  120 V system: Range A = 114–126 V.\n"
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
        section("PSCo Electric Tariff — PQ Requirements")

        concept(
            "Schedule R — Residential",
            "No power factor clause.  Residential customers are not contractually required\n"
            "to maintain any particular power factor.  Values in the 0.85–0.95 range are\n"
            "typical and expected.  No IEEE 519 harmonic clause exists in the tariff;\n"
            "harmonic enforcement for all classes is via the PSCo Blue Book and IEEE 519.",
        )

        concept(
            "Schedule C — Small Commercial  (< 50 kW demand)",
            "Power Factor: PSCo Electric Tariff Sheet R73 requires the customer to maintain\n"
            "power factor of not less than 90% lagging (0.90).  The Company reserves the\n"
            "right to install metering and bill a reactive demand charge, or to discontinue\n"
            "service, if the customer does not comply.\n"
            "\n"
            "Harmonics: No specific harmonic clause in the tariff.  Enforcement is through\n"
            "the PSCo Blue Book standard, which references IEEE 519-2022.",
        )

        concept(
            "Schedule SG — C&I Secondary  (≥ 50 kW demand)",
            "Power Factor: Same as Schedule C — Sheet R73 requires PF ≥ 0.90 lagging.\n"
            "The Company reserves the right to discontinue service to any customer not\n"
            "complying herewith.  Reactive demand charges may also be assessed.\n"
            "\n"
            "Phase Balance: Sheet R121 requires that load in any one phase shall not exceed\n"
            "the load in any other phase by more than 15% for three-phase services.\n"
            "\n"
            "Harmonics: No specific harmonic clause in the tariff.  Enforcement is through\n"
            "the PSCo Blue Book standard, which references IEEE 519-2022.",
        )

        concept(
            "Schedule PG — C&I Primary",
            "Power Factor: Sheet R121 requires Primary service customers to maintain power\n"
            "factor as near unity as practicable.  There is no explicit numeric threshold\n"
            "stated, but 0.90 lagging is the practical enforcement floor consistent with\n"
            "Sheet R73 for secondary customers.\n"
            "\n"
            "Phase Balance: Sheet R121 requires that load in any one phase shall not exceed\n"
            "the load in any other phase by more than 15% for three-phase services.\n"
            "\n"
            "Harmonics: No specific harmonic clause in the tariff.  Enforcement is through\n"
            "the PSCo Blue Book standard, which references IEEE 519-2022.",
        )

        concept(
            "Tariff Sheet Reference Summary",
            "  Sheet R73  — Power factor clause for Secondary customers (Schedules C, SG)\n"
            "               Minimum 0.90 lagging; right to discontinue service.\n"
            "\n"
            "  Sheet R121 — Requirements for Primary service (Schedule PG)\n"
            "               PF near unity; phase imbalance ≤ 15% between phases.\n"
            "\n"
            "  Harmonics  — No tariff clause; governed by PSCo Blue Book → IEEE 519-2022.\n"
            "\n"
            "Note: Sheet numbers reference the PSCo Electric Service Rules and Regulations\n"
            "(Tariff) as filed with the Colorado PUC.  Sheet numbering may change with\n"
            "tariff revisions — verify against the current filed tariff when citing.",
        )

        # ── Investigation Guidance by Customer Class ───────────────────────
        section("Investigation Guidance by Customer Class")

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
            "     3-phase panel.  Cite Sheet R121 (≤ 15%).  Over 10% warrants action.\n"
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
            "  1. Power factor — cite Sheet R121.  Large lagging PF is visible on the\n"
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
            "     single-phase loads.  Cite Sheet R121 (≤ 15% between phases).\n"
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
        section("Key Concepts")

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
            "(e.g., 114–126 V on a 120 V system).  Utilities are expected to supply\n"
            "within Range A under normal conditions.\n"
            "\n"
            "Range B: occasional short-duration excursions outside Range A are tolerated\n"
            "during abnormal system conditions.  Sustained Range B voltage requires a\n"
            "corrective action plan.",
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
        section("Neutral Integrity — Theory and Diagnostics")

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

        section("Analysis Methods & Diagnostics")

        concept(
            "Voltage Compliance — ANSI C84.1",
            "Reports what percentage of 5-minute intervals fall inside Range A (±5% of\n"
            "nominal), Range B (±8.3%), or outside both bands.  When the meter's max-min\n"
            "max-min record is available, peak and minimum voltage within each interval\n"
            "are used to detect momentary exceedances that the interval average would mask.\n"
            "\n"
            "Split-phase services (120/240 V, no voltage_c channel) are automatically\n"
            "detected and voltage_a / voltage_b are evaluated independently against 120 V\n"
            "bands.  Three-phase services evaluate all three phases against the nominal\n"
            "L-N voltage derived from the entered nominal value.",
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
            "PSCo Blue Book limit: 10%.  Tariff Sheet R121 requires three-phase loads\n"
            "to remain within 15% phase-to-phase.  Both thresholds are evaluated.",
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

        txt.config(state="disabled")

        tk.Button(win, text="Close", command=win.destroy,
                  font=_FONT_UI, relief="flat", padx=20, pady=6,
                  bg="#dddddd", cursor="hand2").pack(pady=(0, 14))

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


if __name__ == "__main__":
    app = PQApp()
    app.mainloop()

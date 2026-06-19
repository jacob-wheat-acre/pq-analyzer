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

# ── Transformer / Blue Book data ──────────────────────────────────────────────
# Import the same lookup tables used by the analysis engine so the GUI always
# stays in sync with what pq_analyzer.py will actually compute.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from pq_analyzer import (
        _BLUE_BOOK_ISC,
        _SERVICE_TYPE_LABEL,
        _infer_secondary_v,
        _lookup_isc,
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
        check_harmonic_sources,
        check_harmonic_statistics,
        detect_events,
        analyze_root_causes,
        generate_report,
        export_results,
        generate_word_report,
        plot_voltage,
        plot_thd,
        plot_summary,
        plot_harmonic_spectrum,
    )
    _BOOK_AVAILABLE = True
except Exception as _import_exc:
    import traceback
    _log_path = Path(__file__).parent / "import_error.log"
    _log_path.write_text(traceback.format_exc())
    _BLUE_BOOK_ISC = {}
    _SERVICE_TYPE_LABEL = {}
    _BOOK_AVAILABLE = False

# Display labels for the type picker (ordered for the dropdown)
_TYPE_ORDER = [
    "1ph-overhead",
    "1ph-padmount",
    "3ph-padmount",
    "3ph-overhead-wye",
    "3ph-open-delta",
    "3ph-closed-delta",
]
_TYPE_DISPLAY = {k: _SERVICE_TYPE_LABEL.get(k, k) for k in _TYPE_ORDER}
# Sentinel for primary-metered services (no Blue Book kVA/ISC lookup)
_PRIMARY_KEY    = "__primary__"
_PRIMARY_LABEL  = "Primary metered"


def _resolve_secondary_v(svc_type: str, nominal_v: float) -> int:
    """Convert nominal L-N voltage to the secondary (L-L) voltage used as a Blue Book key.

    Single-phase services with 120 V L-N use 240 V L-L as their secondary voltage
    (120/240 V split-phase).  If the Blue Book has no 120 V entries for the given
    service type, fall back to 240 V automatically.
    """
    try:
        sv = _infer_secondary_v(svc_type, nominal_v)
    except Exception:
        return 240
    if sv == 120 and not any(k[0] == svc_type and k[2] == 120 for k in _BLUE_BOOK_ISC):
        return 240
    return sv


def _kva_options(svc_type: str, nominal_v: float) -> list:
    """Return sorted list of kVA sizes available in the Blue Book for this type/voltage."""
    if svc_type == _PRIMARY_KEY or not svc_type:
        return []
    sec_v = _resolve_secondary_v(svc_type, nominal_v)
    return sorted({k[1] for k in _BLUE_BOOK_ISC if k[0] == svc_type and k[2] == sec_v})


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
        self.title("PQ Analyzer")
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

        # ── File row ──────────────────────────────────────────────────────────
        file_frame = tk.Frame(self, bg=_BG)
        file_frame.pack(fill="x", **pad)

        tk.Label(file_frame, text="PQD File", width=16, anchor="w",
                 bg=_BG, fg=_LABEL_FG, font=_FONT_UI).pack(side="left")
        self._file_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self._file_var, font=_FONT_UI,
                 width=40).pack(side="left", padx=(0, 6), fill="x", expand=True)
        tk.Button(file_frame, text="Browse…", command=self._browse,
                  font=_FONT_UI).pack(side="left")

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

        # ISC auto-label
        self._isc_auto_var = tk.StringVar(value="")
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
            self._kva_combo.config(state="disabled", values=[])
            self._kva_var.set("")
            self._isc_auto_var.set("")
            return

        try:
            nominal = float(self._nominal_var.get())
        except ValueError:
            nominal = 120.0

        sizes = _kva_options(key, nominal)
        if not sizes:
            self._kva_combo.config(state="disabled", values=[])
            self._kva_var.set("")
            self._isc_auto_var.set("No Blue Book entries for this type/voltage combination")
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
        )

        # ── Adapter ───────────────────────────────────────────────────────────
        fp = Path(filepath)
        if fp.suffix.lower() == ".pqd":
            adapter = ProntoAdapter(fp)
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
        source_harm_result  = check_harmonic_sources(df, thresh)
        stat_result         = check_harmonic_statistics(df, thresh)
        event_result        = detect_events(ds, thresh)

        report = generate_report(
            ds, volt_result, thd_result, pf_result,
            imb_result, curr_imb_result, demand_result,
            harm_result, volt_harm_result, neutral_harm_result,
            source_harm_result, stat_result, event_result, thresh,
        )
        report["root_causes"] = analyze_root_causes(report, ds, thresh)

        # ── Export ────────────────────────────────────────────────────────────
        export_results(ds, report, outdir, stem=stem)

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_voltage(df, volt_result, thresh, outdir=outdir)
        plot_thd(df, thd_result, thresh, outdir=outdir)
        plot_summary(df, imb_result, outdir=outdir)
        plot_harmonic_spectrum(df, thresh, outdir=outdir)

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

        self._log_write("\nDone.  Word report and plots saved to pq_output/\n", tag="done")
        self._open_report(stem)

    def _open_report(self, stem: str):
        report = _SCRIPT.parent / "pq_output" / f"{stem}_report.docx"
        if report.exists():
            self.after(0, lambda: self._open_btn.config(state="normal"))
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(report)])
            elif sys.platform == "win32":
                subprocess.Popen(["start", "", str(report)], shell=True)
            else:
                subprocess.Popen(["xdg-open", str(report)])

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
        section("Analysis Methods & Diagnostics")

        concept(
            "Voltage Compliance — ANSI C84.1",
            "Reports what percentage of 5-minute intervals fall inside Range A (±5% of\n"
            "nominal), Range B (±8.3%), or outside both bands.  When the meter's max-min\n"
            "record (obs[24]) is available, peak and minimum voltage within each interval\n"
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
            "Root Cause Analysis",
            "After all individual checks run, the tool synthesizes findings into a\n"
            "root cause list ranked by severity.  Each finding includes:\n"
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

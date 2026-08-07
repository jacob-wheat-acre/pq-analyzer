from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pq_constants import (
    Thresholds,
    _h519_limit,
    _itic_upper_v,
    _itic_lower_v,
    _ITIC_UPPER_MS_STEP,
    _ITIC_UPPER_PCT_STEP,
    _ITIC_LOWER_MS_STEP,
    _ITIC_LOWER_PCT_STEP,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Colorblind-safe phase palette (Okabe–Ito): blue / orange / green.
# Validated for protan/deutan/tritan separation — do not swap for the old
# Material blue/orange/green, whose orange↔green pair is indistinguishable
# under protanopia.
_PH_A = "#0072B2"
_PH_B = "#E69F00"
_PH_C = "#009E73"
_PHASE_CLR  = {"a": _PH_A, "b": _PH_B, "c": _PH_C,
               "A": _PH_A, "B": _PH_B, "C": _PH_C}
_NEUTRAL_CLR = "#666666"


#: Power channels are carried in watts and VAR throughout (see PQDataset in
#: pq_adapter), while every label and every reported figure is in kW and kVAR.
#: The analysis code divides by 1000 at each use; the plots did not, so a 3.2 kW
#: house was drawn as 3,200 on an axis labelled kW.
_W_PER_KW = 1000.0


def _to_kilo(series: pd.Series) -> pd.Series:
    """Convert a watt/VAR channel to kW/kVAR for display."""
    return series / _W_PER_KW


def service_phases(df: pd.DataFrame, thresh: Optional[Thresholds] = None) -> list:
    """The phases this service actually has, as (key, label, colour).

    A 120/240 V split-phase service has two legs, not three, and labelling them
    "Phase A/B/C" invents a conductor that does not exist -- the harmonic
    spectrum chart was drawing an empty "Phase C" series and legend entry for
    every house.

    Resolution order puts the engineer's own picker first, because channel
    presence alone cannot tell a genuinely single-phase service from a
    three-phase one whose C phase is missing from the export:

      1. ``service_type`` from the transformer picker ("1ph-..." is split-phase)
      2. ``topology`` if explicitly set to something other than "auto"
      3. otherwise, whether a C-phase channel was read at all
    """
    split = None
    if thresh is not None:
        svc = (thresh.service_type or "")
        if svc.startswith("1ph"):
            split = True
        elif svc.startswith("3ph"):
            split = False
        elif thresh.topology == "split-phase":
            split = True
        elif thresh.topology == "3ph-wye":
            split = False
    if split is None:
        split = "current_c" not in df.columns and "voltage_c" not in df.columns

    if split:
        # L1/L2 is what the legs are called on a split-phase service.
        return [("a", "L1", _PH_A), ("b", "L2", _PH_B)]
    return [("a", "Phase A", _PH_A), ("b", "Phase B", _PH_B), ("c", "Phase C", _PH_C)]


def _plot_name(stem: str, name: str) -> str:
    """Stem-prefixed plot filename so multi-site output dirs never collide."""
    return f"{stem}_{name}" if stem else name


def _fmt_time_axis(ax, index: pd.DatetimeIndex) -> None:
    """Time-axis tick format: include the date once the span exceeds a day,
    otherwise plain clock time."""
    span_h = ((index[-1] - index[0]).total_seconds() / 3600) if len(index) > 1 else 0
    fmt = "%m/%d %H:%M" if span_h > 30 else "%H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _save(fig, outdir: Optional[Path], stem: str, name: str) -> None:
    if outdir:
        fname = _plot_name(stem, name)
        fig.savefig(outdir / fname, dpi=150)
        log.info("Saved plot → %s/%s", outdir, fname)
    else:
        plt.show()
    plt.close(fig)

def _shade_violations(ax, violation_ts: pd.DatetimeIndex, df_index: pd.DatetimeIndex):
    """Shade violation windows on an axis as translucent red bands."""
    if violation_ts.empty:
        return
    resolution = (df_index[1] - df_index[0]) if len(df_index) > 1 else pd.Timedelta("1s")
    in_viol = False
    v_start = None
    for ts in df_index:
        is_viol = ts in violation_ts
        if is_viol and not in_viol:
            v_start = ts
            in_viol = True
        elif not is_viol and in_viol:
            ax.axvspan(v_start, ts, color="red", alpha=0.15, linewidth=0)
            in_viol = False
    if in_viol and v_start is not None:
        ax.axvspan(v_start, df_index[-1], color="red", alpha=0.15, linewidth=0)


def _gap_spans(index: pd.DatetimeIndex) -> list:
    """Spans where the meter recorded nothing, as (start, end) pairs.

    A line drawn straight across a gap reads as data that was never measured,
    which is the opposite of what an overview chart is for.
    """
    if len(index) < 3:
        return []
    step = pd.Series(index).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return []
    limit = step * 2.5
    gaps = []
    for prev, nxt in zip(index[:-1], index[1:]):
        if (nxt - prev) > limit:
            gaps.append((prev, nxt))
    return gaps


def _break_at_gaps(s: pd.Series, gaps: list) -> pd.Series:
    """Insert a NaN inside each gap so the line breaks instead of spanning it."""
    if not gaps:
        return s
    breaks = pd.Series(
        np.nan,
        index=pd.DatetimeIndex([start + (end - start) / 2 for start, end in gaps],
                               tz=s.index.tz),
    )
    return pd.concat([s, breaks]).sort_index()


def plot_overview(
    ds,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Voltage and current over the whole recording, as one sanity check.

    This is the chart that answers "did the tool read the file correctly?"
    before any compliance question is asked: the full period, every phase, the
    raw measured series with nothing filtered or scaled. Voltage and current are
    on separate stacked panels sharing one time axis rather than on twin y-axes,
    because two scales on one frame make the crossings and relative heights an
    artifact of the scaling choice.
    """
    df = ds.df
    v_cols = [c for c in ["voltage_a", "voltage_b", "voltage_c"] if c in df.columns]
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    has_neutral = "current_neutral" in df.columns
    if not v_cols and not i_cols:
        log.warning("Overview plot needs voltage or current channels — skipping.")
        return

    phases   = service_phases(df, thresh)
    labels   = {ph: label for ph, label, _ in phases}
    is_split = len(phases) == 2

    panels = [p for p in (("v", v_cols), ("i", i_cols)) if p[1]]
    gaps = _gap_spans(df.index)

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(14, 3.2 * len(panels) + 1.0),
        sharex=True, squeeze=False,
    )
    axes = [a[0] for a in axes]

    for ax, (kind, cols) in zip(axes, panels):
        for col in cols:
            ph = col[-1]
            series = _break_at_gaps(df[col].dropna(), gaps)
            ax.plot(series.index, series.values, color=_PHASE_CLR.get(ph, "gray"),
                    lw=0.9, label=labels.get(ph, col))
        if kind == "i" and has_neutral:
            # Dashed as well as gray: the neutral has to stay identifiable
            # where the phase hues are the only other cue.
            series = _break_at_gaps(df["current_neutral"].dropna(), gaps)
            ax.plot(series.index, series.values, color=_NEUTRAL_CLR, lw=0.9,
                    ls="--", label="Neutral")
        if kind == "v":
            # The allowed band, in gray rather than a status hue: on this chart
            # the colors carry phase identity, and a red limit line beside an
            # orange phase trace would read as one more series.
            vmin = thresh.nominal_voltage * (1 - thresh.volt_tolerance)
            vmax = thresh.nominal_voltage * (1 + thresh.volt_tolerance)
            ax.axhspan(vmin, vmax, color="#333333", alpha=0.06, linewidth=0,
                       zorder=0, label=f"Allowed range ({vmin:.0f}–{vmax:.0f} V)")
            ax.axhline(thresh.nominal_voltage, color="gray", ls=":", lw=0.8, alpha=0.6)
            # Keep the band in frame so a flat trace is read against the margin
            # it has, not against an axis zoomed to its own noise.
            lo = min([df[c].min() for c in cols] + [vmin])
            hi = max([df[c].max() for c in cols] + [vmax])
            pad = max((hi - lo) * 0.05, 1.0)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel("RMS Voltage (V)")
            ax.set_title("Recording Overview — measured voltage and current, "
                         "full period", fontsize=12)
        else:
            ax.set_ylabel("RMS Current (A)")
            ax.set_ylim(bottom=0)

        for start, end in gaps:
            ax.axvspan(start, end, color=_NEUTRAL_CLR, alpha=0.12, linewidth=0)
        ax.grid(True, alpha=0.3)
        if len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(fontsize=8, loc="upper right", ncol=4)

    axes[-1].set_xlabel("Time")
    _fmt_time_axis(axes[-1], df.index)
    fig.autofmt_xdate()

    # The caption under the panels is the sanity check itself: what the tool
    # believes it read, in the same frame as the data it drew.
    interval = ds.meta.get("interval_minutes", 5)
    span = f"{df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M}"
    subtitle = (f"{span}   ·   {ds.duration_hours:.1f} h "
                f"({ds.duration_hours / 24:.1f} days)   ·   "
                f"{len(df):,} intervals of {interval:g} min")
    if gaps:
        subtitle += f"   ·   {len(gaps)} recording gap(s), shaded"
    fig.text(0.5, 0.005, subtitle, ha="center", fontsize=8, color="#444444")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, stem, "overview.png")


def plot_voltage(
    df: pd.DataFrame,
    volt_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    v_cols = [c for c in ["voltage_a", "voltage_b", "voltage_c"] if c in df.columns]
    if not v_cols:
        log.warning("No voltage columns to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"voltage_a": _PH_A, "voltage_b": _PH_B, "voltage_c": _PH_C}
    phases   = service_phases(df, thresh)
    is_split = len(phases) == 2
    if is_split:
        labels = {"voltage_a": "L1-N", "voltage_b": "L2-N"}
        topo_title = "Split-Phase Voltage (L-N)"
    else:
        labels = {f"voltage_{ph}": label for ph, label, _ in phases}
        topo_title = "Three-Phase Voltage"

    for col in v_cols:
        ax.plot(df.index, df[col], color=colors.get(col, "gray"),
                lw=0.8, label=labels.get(col, col))

    vmin = thresh.nominal_voltage * (1 - thresh.volt_tolerance)
    vmax = thresh.nominal_voltage * (1 + thresh.volt_tolerance)
    ax.axhline(vmin, color="red",    ls="--", lw=1.0, label=f"ANSI lower ({vmin:.1f} V)")
    ax.axhline(vmax, color="orange", ls="--", lw=1.0, label=f"ANSI upper ({vmax:.1f} V)")
    ax.axhline(thresh.nominal_voltage, color="gray", ls=":", lw=0.8, alpha=0.6)

    viol_ts = volt_result.get("violation_timestamps", pd.DatetimeIndex([]))
    _shade_violations(ax, viol_ts, df.index)

    if not viol_ts.empty:
        ax.legend(handles=ax.lines[:], loc="upper right", fontsize=8)
        ax.legend(
            [Patch(facecolor="red", alpha=0.3)],
            ["Voltage violation"],
            loc="upper left", fontsize=8,
        )

    _fmt_time_axis(ax, df.index)
    fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel("RMS Voltage (V)")
    ax.set_title(f"{topo_title} — ANSI C84.1 Compliance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, outdir, stem, "voltage.png")


def plot_thd(
    df: pd.DataFrame,
    thd_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Voltage THD and current TDD (or THD fallback) timelines vs IEEE 519 limits.

    When tdd_info is present in thd_result, the current panel shows the same
    per-interval TDD series the compliance check evaluates:
    TDD(t) = THD%(t) × Irms(t) / IL.
    """
    thd_v = [c for c in ["thd_voltage_a", "thd_voltage_b", "thd_voltage_c"] if c in df.columns]
    thd_i = [c for c in ["thd_current_a", "thd_current_b", "thd_current_c"] if c in df.columns]

    if not thd_v and not thd_i:
        log.warning("No THD columns to plot.")
        return

    tdd_info = thd_result.get("tdd_info", {})
    il_amps  = tdd_info.get("il_amps")
    use_tdd  = bool(tdd_info) and il_amps
    c_limit  = thd_result["current"].get("limit_pct", thresh.thd_current_limit) \
               if thd_result.get("current", {}).get("available") else thresh.thd_current_limit
    v_limit  = thd_result["voltage"].get("limit_pct", thresh.thd_voltage_limit) \
               if thd_result.get("voltage", {}).get("available") else thresh.thd_voltage_limit

    i_label = "Current TDD (%)" if use_tdd else "Current THD (%)"
    if use_tdd and not tdd_info.get("isc_provided", True):
        i_lim_label = f"Conservative limit ({c_limit:.1f}%, class assumed)"
    else:
        i_lim_label = f"IEEE 519 limit ({c_limit:.1f}%)"

    n_plots = int(bool(thd_v)) + int(bool(thd_i))
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    for cols, limit, label, lim_label, key in [
        (thd_v, v_limit, "Voltage THD (%)", f"IEEE 519 limit ({v_limit:.1f}%)", "voltage"),
        (thd_i, c_limit, i_label, i_lim_label, "current"),
    ]:
        if not cols:
            continue
        ax = axes[plot_idx]
        plot_idx += 1
        colors_map = {
            f"thd_{key}_a": _PH_A, f"thd_{key}_b": _PH_B, f"thd_{key}_c": _PH_C
        }
        for col in cols:
            phase = col.split("_")[-1]
            series = df[col]
            if key == "current" and use_tdd:
                i_col = f"current_{phase}"
                if i_col in df.columns:
                    aligned = df[[col, i_col]].dropna()
                    series  = aligned[col] * aligned[i_col] / il_amps
            ax.plot(series.index, series, color=colors_map.get(col, "gray"),
                    lw=0.8, label=phase.upper())
        ax.axhline(limit, color="red", ls="--", lw=1.0, label=lim_label)

        viol_ts_list = thd_result[key].get("violation_timestamps", [])
        if viol_ts_list is not None and len(viol_ts_list):
            viol_idx = pd.DatetimeIndex(viol_ts_list)
            _shade_violations(ax, viol_idx, df.index)

        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        metric_word = label.split(" ")[1]
        ax.set_title(f"{label.split(' ')[0]} {metric_word} — IEEE 519 Compliance")

    _fmt_time_axis(axes[-1], df.index)
    fig.autofmt_xdate()
    axes[-1].set_xlabel("Time")
    fig.tight_layout()

    _save(fig, outdir, stem, "thd.png")


def plot_summary(
    df: pd.DataFrame,
    imb_result: dict,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Four-panel summary: voltage imbalance, power factor, real/reactive power."""
    panels = []
    if "imbalance_series" in imb_result:
        panels.append(("Voltage Imbalance (%)", imb_result["imbalance_series"],
                        imb_result.get("limit_pct"), "#9C27B0"))
    if "power_factor" in df.columns:
        panels.append(("Power Factor", df["power_factor"], None, "#009688"))
    if "power_real" in df.columns:
        panels.append(("Real Power (kW)", _to_kilo(df["power_real"]), None, "#F44336"))
    if "power_reactive" in df.columns:
        panels.append(("Reactive Power (kVAR)", _to_kilo(df["power_reactive"]), None, "#FF5722"))

    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (ylabel, series, limit, color) in zip(axes, panels):
        ax.plot(series.index, series.values, color=color, lw=0.8)
        if limit is not None:
            ax.axhline(limit, color="red", ls="--", lw=1.0, label=f"Limit ({limit})")
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    _fmt_time_axis(axes[-1], df.index)
    fig.autofmt_xdate()
    axes[-1].set_xlabel("Time")
    fig.suptitle("Power Quality Summary", fontsize=11)
    fig.tight_layout()

    _save(fig, outdir, stem, "summary.png")


def plot_harmonic_spectrum(
    df: pd.DataFrame,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Bar chart of median H3–H13 per phase (% of IL).

    Shows the three-phase harmonic spectrum with IEEE 519-2022 per-order limits
    where ISC/IL is known.  Harmonics are stored as Amps; divided by IL here to
    display as % of IL for direct comparison against limits.
    """
    orders = [3, 5, 7, 9, 11, 13]
    phases = service_phases(df, thresh)

    # Build per-phase median harmonic % of IL
    il_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    if not il_cols:
        return
    il_amps = float(df[il_cols].max(axis=1).max())
    if il_amps <= 0:
        return

    data = {}
    for ph, _, _ in phases:
        row = []
        for h in orders:
            col = f"h{h}_current_{ph}"
            if col in df.columns:
                row.append(float(df[col].median()) / il_amps * 100)
            else:
                row.append(0.0)
        data[ph] = row

    if all(all(v == 0 for v in data[ph]) for ph, _, _ in phases):
        log.warning("No individual harmonic data to plot.")
        return

    x = np.arange(len(orders))
    # Width and offset both follow the phase count so a two-leg service gets a
    # centred pair rather than three slots with one left empty.
    n_ph  = len(phases)
    width = 0.66 / n_ph
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (ph, label, color) in enumerate(phases):
        offset = (i - (n_ph - 1) / 2) * width
        ax.bar(x + offset, data[ph], width, label=label, color=color, alpha=0.85)

    # IEEE 519-2022 limits — horizontal segment spanning only the orders each limit covers
    if thresh.isc_amps is not None:
        isc_il = thresh.isc_amps / il_amps
        # Map each limit value → x-indices of the orders it applies to
        limit_groups: dict[float, list[int]] = {}
        for xi, h in enumerate(orders):
            lim = _h519_limit(h, isc_il)
            if lim > 0:
                limit_groups.setdefault(lim, []).append(xi)
        pad = width * 1.5 + 0.05
        for lim, x_idxs in sorted(limit_groups.items(), reverse=True):
            xmin = min(x_idxs) - pad
            xmax = max(x_idxs) + pad
            label = f"IEEE 519 limit {lim:.1f}% (ISC/IL={isc_il:.0f})"
            ax.hlines(lim, xmin, xmax, colors="red", linestyles="--", lw=1.2, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in orders])
    ax.set_ylabel("% of IL (max demand current)")
    ax.set_title("Current Harmonic Spectrum — Median Over Recording Period")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    _save(fig, outdir, stem, "harmonic_spectrum.png")


# ─────────────────────────────────────────────────────────────────────────────
# ITIC CURVE
# Reference: "ITI (CBEMA) Curve Application Note," Information Technology
# Industry Council (ITIC), 2000.  Superseded the CBEMA curve originally
# referenced in ANSI/IEEE 446-1987.  Referenced by IEEE 1159-2019 as the
# standard voltage tolerance envelope for information technology equipment.
# Applicable to 120 V nominal (120/208 V and 120/240 V, 60 Hz systems).
# ─────────────────────────────────────────────────────────────────────────────

def plot_itic(
    events: pd.DataFrame,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """ITIC voltage tolerance curve with sag/swell events plotted as (duration, magnitude) points.

    Requires event records with duration_ms populated — available from adaptive
    (cycle-level) data but not from 5-minute interval averages.
    """
    vol_events = (
        events[events["type"].isin(["voltage_sag", "voltage_swell"])].copy()
        if events is not None and not events.empty
        else pd.DataFrame()
    )
    has_duration = (
        not vol_events.empty
        and "duration_ms" in vol_events.columns
        and vol_events["duration_ms"].notna().any()
    )
    if not has_duration:
        log.warning(
            "ITIC plot requires event-level duration data (adaptive/waveform records). "
            "Not available from 5-minute interval averages — skipping."
        )
        return

    vol_events = vol_events.dropna(subset=["duration_ms", "value_v"])
    nominal = thresh.nominal_voltage
    vol_events["pct"] = vol_events["value_v"] / nominal * 100.0

    fig, ax = plt.subplots(figsize=(10, 7))

    x_fill = np.logspace(-3, 5, 2000)
    upper  = _itic_upper_v(x_fill)
    lower  = _itic_lower_v(x_fill)

    ax.fill_between(x_fill, upper, 600,  color="#ff9999", alpha=0.45, linewidth=0)
    ax.fill_between(x_fill, 0,    lower, color="#ff9999", alpha=0.45, linewidth=0,
                    label="ITIC prohibited zone")
    ax.fill_between(x_fill, lower, upper, color="#d4edda", alpha=0.40, linewidth=0,
                    label="ITIC no-disruption zone")

    ax.plot(_ITIC_UPPER_MS_STEP, _ITIC_UPPER_PCT_STEP, "r-", lw=1.5)
    ax.plot(_ITIC_LOWER_MS_STEP, _ITIC_LOWER_PCT_STEP, "r-", lw=1.5, label="ITIC boundary")
    ax.axhline(100, color="#666666", ls=":", lw=0.8, alpha=0.7, label="100% nominal")

    phase_colors = {"A": _PH_A, "B": _PH_B, "C": _PH_C}
    for phase, color in phase_colors.items():
        s = vol_events[(vol_events["type"] == "voltage_sag")    & (vol_events["phase"] == phase)]
        if not s.empty:
            ax.scatter(s["duration_ms"], s["pct"], marker="v", color=color, s=60,
                       zorder=5, edgecolors="white", linewidths=0.5,
                       label=f"Sag Ph-{phase} (n={len(s)})")
        sw = vol_events[(vol_events["type"] == "voltage_swell") & (vol_events["phase"] == phase)]
        if not sw.empty:
            ax.scatter(sw["duration_ms"], sw["pct"], marker="^", color=color, s=60,
                       zorder=5, edgecolors="white", linewidths=0.5,
                       label=f"Swell Ph-{phase} (n={len(sw)})")

    ax.set_xscale("log")
    ax.set_xlim(0.001, 1e5)
    ax.set_ylim(0, 600)
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Voltage (% of nominal)")
    ax.set_title(
        "ITIC Voltage Tolerance Curve\n"
        "ITI (CBEMA) Curve Application Note, ITIC 2000  ·  Referenced in IEEE 1159-2019"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    x_ticks = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1 s", "10 s", "100 s"])

    fig.tight_layout()
    outpath = (outdir or Path(".")) / _plot_name(stem, "itic_curve.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ITIC curve plot saved → %s", outpath)


def plot_neutral_health(
    ds,
    neutral_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Four-panel neutral health plot for split-phase services."""
    if not neutral_result.get("available"):
        return

    df = ds.df
    if "voltage_a" not in df.columns or "voltage_b" not in df.columns:
        log.warning("plot_neutral_health: voltage_a/voltage_b not in dataset.")
        return

    va = df["voltage_a"].dropna()
    vb = df["voltage_b"].dropna()
    aligned = pd.concat([va, vb], axis=1, join="inner").dropna()
    if aligned.empty:
        return

    has_vne = (
        neutral_result.get("vne_available")
        and ds.has_adaptive
        and ds.adaptive_df is not None
        and "vne_v" in ds.adaptive_df.columns
    )
    n_rows = 4 if has_vne else 3
    nom    = thresh.nominal_voltage

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=False)

    # ── Panel 0: L1 and L2 voltage ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(aligned.index, aligned["voltage_a"],
            color=_PH_A, lw=0.8, label="L1-N (voltage_a)")
    ax.plot(aligned.index, aligned["voltage_b"],
            color=_PH_B, lw=0.8, label="L2-N (voltage_b)")
    vmin = nom * (1 - thresh.volt_tolerance)
    vmax = nom * (1 + thresh.volt_tolerance)
    ax.axhline(vmin, color="red", ls="--", lw=0.8, alpha=0.7, label=f"ANSI lower ({vmin:.1f} V)")
    ax.axhline(vmax, color="red", ls="--", lw=0.8, alpha=0.7, label=f"ANSI upper ({vmax:.1f} V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("L1-N and L2-N Voltages")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _fmt_time_axis(ax, aligned.index)

    # ── Panel 1: Voltage sum (L1 + L2) ───────────────────────────────────────
    ax = axes[1]
    vsum     = aligned["voltage_a"] + aligned["voltage_b"]
    exp_sum  = nom * 2
    ax.plot(vsum.index, vsum, color="#9C27B0", lw=0.8, label="L1 + L2 sum")
    ax.axhline(exp_sum, color="green", ls="--", lw=1.0, alpha=0.7,
               label=f"Expected {exp_sum:.0f} V")
    ax.axhspan(exp_sum * 0.97, exp_sum * 1.03, alpha=0.08, color="green", label="±3% band")
    sum_std = neutral_result.get("sum_std_v", 0.0)
    ax.set_ylabel("L1 + L2 (V)")
    ax.set_title(f"Voltage Sum Stability  [std = {sum_std:.2f} V]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _fmt_time_axis(ax, aligned.index)

    # ── Panel 2: Voltage asymmetry |L1 − L2| ─────────────────────────────────
    ax = axes[2]
    asym     = (aligned["voltage_a"] - aligned["voltage_b"]).abs()
    asym_pct = neutral_result.get("asym_pct", 0.0)
    ax.plot(asym.index, asym, color="#F44336", lw=0.8, label="|L1 − L2|")
    ax.axhline(nom * 0.02, color="orange", ls="--", lw=0.8,
               label=f"2% ({nom * 0.02:.1f} V)")
    ax.axhline(nom * 0.05, color="red",    ls="--", lw=0.8,
               label=f"5% ({nom * 0.05:.1f} V)")
    ax.set_ylabel("|L1 − L2| (V)")
    ax.set_title(
        f"Leg Asymmetry  "
        f"[mean = {neutral_result.get('asym_mean_v', 0):.1f} V  ({asym_pct:.1f}% of nominal)]"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _fmt_time_axis(ax, aligned.index)

    # ── Panel 3 (optional): Neutral-to-earth Vne ─────────────────────────────
    if has_vne:
        ax = axes[3]
        vne = ds.adaptive_df["vne_v"].dropna().abs()
        ax.plot(vne.index, vne, color="#607D8B", lw=0.8, label="Vne (neutral-to-earth)")
        ax.axhline(0.5, color="goldenrod", ls="--", lw=0.8, label="0.5 V caution")
        ax.axhline(2.0, color="orange",    ls="--", lw=0.8, label="2.0 V warning")
        ax.axhline(5.0, color="red",       ls="--", lw=0.8, label="5.0 V critical")
        ax.set_ylabel("Vne (V)")
        ax.set_title(
            f"Neutral-to-Earth Voltage  "
            f"[max = {neutral_result.get('vne_max_v', 0):.2f} V]"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        _fmt_time_axis(ax, vne.index)

    sev_colors = {"normal": "green", "caution": "goldenrod",
                  "warning": "orange", "critical": "red"}
    sev = neutral_result.get("severity", "unknown")
    fig.suptitle(
        f"Neutral Health Assessment — Severity: {sev.upper()}",
        fontsize=11, fontweight="bold",
        color=sev_colors.get(sev, "black"),
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    outpath = (outdir or Path(".")) / _plot_name(stem, "neutral_health.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Neutral health plot saved → %s", outpath)


# ─────────────────────────────────────────────────────────────────────────────
# DEMAND PROFILE / HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_demand_profile(
    df: pd.DataFrame,
    thd_result: dict,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Time-of-day demand pattern.

    Recordings spanning ≥ 2 days: hour-of-day × date heatmap of demand, plus a
    TDD heatmap when computable — answers "when does the load (and distortion)
    peak" at a glance.  Single-day recordings: hourly demand profile
    (mean line + min–max band) instead, since a one-column heatmap is unreadable.
    """
    i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    if "power_real" in df.columns and df["power_real"].notna().any():
        demand = _to_kilo(df["power_real"].dropna())
        d_label = "Real power (kW)"
    elif i_cols:
        demand = df[i_cols].max(axis=1).dropna()
        d_label = "Worst-phase current (A)"
    else:
        log.warning("plot_demand_profile: no power or current channels.")
        return
    if demand.empty:
        return

    n_days = len(np.unique(demand.index.date))

    # Per-interval TDD series (same formula as check_thd), worst phase
    tdd_info = thd_result.get("tdd_info", {})
    il_amps  = tdd_info.get("il_amps")
    tdd_worst = None
    if il_amps:
        tdd_series = []
        for ph in ("a", "b", "c"):
            t_col, i_col = f"thd_current_{ph}", f"current_{ph}"
            if t_col in df.columns and i_col in df.columns:
                aligned = df[[t_col, i_col]].dropna()
                if len(aligned):
                    tdd_series.append(aligned[t_col] * aligned[i_col] / il_amps)
        if tdd_series:
            tdd_worst = pd.concat(tdd_series, axis=1).max(axis=1).dropna()

    if n_days >= 2:
        panels = [("Demand", demand, d_label, "Blues")]
        if tdd_worst is not None and not tdd_worst.empty:
            panels.append(("Current TDD", tdd_worst, "TDD (%)", "Oranges"))
        fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, series, cbar_label, cmap) in zip(axes, panels):
            grid = series.groupby(
                [series.index.hour, series.index.date]
            ).mean().unstack()
            im = ax.imshow(grid.values, aspect="auto", cmap=cmap,
                           origin="lower", interpolation="nearest")
            ax.set_yticks(range(0, 24, 3))
            ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8)
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([d.strftime("%m/%d") for d in grid.columns],
                               fontsize=8, rotation=45)
            ax.set_ylabel("Hour of day")
            ax.set_title(f"{title} by Hour and Day")
            fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
        fig.tight_layout()
    else:
        hours  = demand.index.hour
        h_mean = demand.groupby(hours).mean()
        h_min  = demand.groupby(hours).min()
        h_max  = demand.groupby(hours).max()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(h_mean.index, h_min, h_max, color=_PH_A, alpha=0.15,
                        linewidth=0, label="Min–max range")
        ax.plot(h_mean.index, h_mean, color=_PH_A, lw=1.8, label="Hourly mean")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(d_label)
        ax.set_title("Demand Profile by Hour of Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    _save(fig, outdir, stem, "demand_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# HARMONIC TREND vs LOAD
# ─────────────────────────────────────────────────────────────────────────────

def plot_harmonic_trend(
    df: pd.DataFrame,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Worst-phase H3/H5/H7 harmonic current magnitudes over time, with total
    load current for context (all in amps — one axis).  Load-correlated
    harmonics point to customer load; load-independent harmonics point to
    background/system sources."""
    orders = [(3, _PH_A), (5, _PH_B), (7, _PH_C)]
    series = {}
    for h, _ in orders:
        cols = [f"h{h}_current_{p}" for p in "abc" if f"h{h}_current_{p}" in df.columns]
        if cols:
            s = df[cols].max(axis=1).dropna()
            if not s.empty:
                series[h] = s
    if not series:
        log.warning("plot_harmonic_trend: no individual harmonic channels.")
        return

    i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    load = df[i_cols].max(axis=1).dropna() if i_cols else None

    fig, ax = plt.subplots(figsize=(14, 4.5))
    if load is not None and not load.empty:
        ax.plot(load.index, load, color=_NEUTRAL_CLR, lw=0.8, ls="--", alpha=0.7,
                label="Load current (worst phase)")
    for h, color in orders:
        if h in series:
            ax.plot(series[h].index, series[h], color=color, lw=1.0, label=f"H{h}")

    _fmt_time_axis(ax, df.index)
    fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel("Current (A)")
    ax.set_title("Harmonic Current Magnitudes vs Load (worst phase per order)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, outdir, stem, "harmonic_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# IMBALANCE & NEUTRAL CURRENT
# ─────────────────────────────────────────────────────────────────────────────

def plot_imbalance(
    df: pd.DataFrame,
    volt_imb_result: dict,
    curr_imb_result: dict,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Stacked panels: voltage imbalance %, current imbalance %, and neutral
    current — each against its limit where one applies."""
    panels = []

    v_series = volt_imb_result.get("imbalance_series")
    if v_series is not None and len(v_series.dropna()):
        panels.append(("Voltage imbalance (%)", v_series.dropna(),
                       volt_imb_result.get("limit_pct"), "#6A3D9A"))

    i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    if len(i_cols) >= 2:
        idf = df[i_cols].dropna()
        if len(idf):
            avg = idf.mean(axis=1)
            dev = idf.subtract(avg, axis=0).abs().max(axis=1)
            ci  = pd.Series(np.where(avg > 1.0, dev / avg * 100, np.nan), index=idf.index).dropna()
            if len(ci):
                panels.append(("Current imbalance (%)", ci,
                               curr_imb_result.get("limit_pct"), _PH_B))

    if "current_neutral" in df.columns:
        nc = df["current_neutral"].dropna()
        if len(nc):
            panels.append(("Neutral current (A)", nc, None, _NEUTRAL_CLR))

    if not panels:
        log.warning("plot_imbalance: no imbalance data to plot.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (ylabel, series, limit, color) in zip(axes, panels):
        ax.plot(series.index, series, color=color, lw=0.8)
        if limit is not None:
            ax.axhline(limit, color="red", ls="--", lw=1.0, label=f"Limit ({limit:.0f}%)")
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    _fmt_time_axis(axes[-1], df.index)
    fig.autofmt_xdate()
    axes[-1].set_xlabel("Time")
    fig.suptitle("Voltage / Current Imbalance and Neutral Current", fontsize=11)
    fig.tight_layout()

    _save(fig, outdir, stem, "imbalance.png")


# ─────────────────────────────────────────────────────────────────────────────
# POWER FACTOR vs LOAD
# ─────────────────────────────────────────────────────────────────────────────

def plot_pf_load(
    df: pd.DataFrame,
    pf_result: dict,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Scatter of power factor vs load.  Shows whether low PF coincides with
    high load (tariff-relevant) or only light load (usually benign)."""
    if "power_factor" not in df.columns:
        return
    if "power_real" in df.columns and df["power_real"].notna().any():
        load, x_label = _to_kilo(df["power_real"]), "Real power (kW)"
    else:
        i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
        if not i_cols:
            return
        load, x_label = df[i_cols].max(axis=1), "Worst-phase current (A)"

    aligned = pd.concat([load.rename("load"), df["power_factor"].abs().rename("pf")],
                        axis=1).dropna()
    if aligned.empty:
        return

    limit = pf_result.get("limit") if pf_result.get("available") else None
    below = aligned["pf"] < limit if limit else pd.Series(False, index=aligned.index)

    fig, ax = plt.subplots(figsize=(9, 5))
    ok = aligned[~below]
    ax.scatter(ok["load"], ok["pf"], s=14, color=_PH_A, alpha=0.5,
               edgecolors="none", label="Interval")
    if below.any():
        bad = aligned[below]
        ax.scatter(bad["load"], bad["pf"], s=18, color="#CC0000", alpha=0.7,
                   edgecolors="white", linewidths=0.3,
                   label=f"Below limit (n={len(bad)})")
    if limit:
        ax.axhline(limit, color="red", ls="--", lw=1.0, label=f"Tariff limit ({limit:.2f})")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Power factor")
    ax.set_ylim(min(0.5, float(aligned["pf"].min()) - 0.02), 1.02)
    ax.set_title("Power Factor vs Load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, outdir, stem, "pf_load.png")


def plot_flicker(
    df: pd.DataFrame,
    flicker_result: dict,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Short- and long-term flicker severity over time, and how long each lasted.

    Three panels because the two timelines alone cannot answer the question a
    reader actually has. A Pst trace with one spike at 5.0 and a trace that
    sits over the limit all week look similarly alarming at a glance, and they
    are not the same finding. The third panel sorts every reading from worst
    to best against the share of the recording at or above it, so the width of
    the exceedance is visible: a curve that crosses the limit line near the
    left edge is an anomaly, one that crosses far to the right is a condition.
    The 95th-percentile mark is drawn on it because that is the point both
    IEC 61000-3-3 and IEEE 1453 actually assess, and the severity band in the
    report is graded there rather than at the maximum.
    """
    if not flicker_result.get("available"):
        return

    panels = [(kind, label, flicker_result[f"{kind}_limit"])
              for kind, label in (("pst", "Pst — short-term (10 min)"),
                                  ("plt", "Plt — long-term (2 h)"))
              if flicker_result.get(kind)]
    if not panels:
        return

    fig, axes = plt.subplots(len(panels) + 1, 1,
                             figsize=(14, 3.6 * (len(panels) + 1)))
    axes = np.atleast_1d(axes)

    for ax, (kind, label, limit) in zip(axes, panels):
        for phase, stats in sorted(flicker_result[kind].items()):
            col = stats["column"]
            if col not in df.columns:
                continue
            series = df[col].dropna()
            ax.plot(series.index, series, lw=0.9,
                    color=_PHASE_CLR.get(phase.lower(), "gray"),
                    label=f"{phase}  (95th pct {stats['p95']:.2f}, "
                          f"max {stats['max']:.2f})")
            over = series[series > limit]
            if not over.empty:
                ax.scatter(over.index, over, s=9, color="#CC0000",
                           zorder=3, edgecolors="none")
        ax.axhline(limit, color="red", ls="--", lw=1.0,
                   label=f"IEC 61000-3-3 limit ({limit:.2f})")
        if kind == "plt":
            # The level the supply system is held to, which is not the same
            # number as the equipment emission limit above.
            ax.axhline(0.8, color="#888888", ls=":", lw=1.0,
                       label="IEEE 1453 LV compatibility level (0.80)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _fmt_time_axis(ax, df.index)

    # ── how much of the recording each level occupied ────────────────────────
    ax = axes[-1]
    for kind, label, limit in panels:
        for phase, stats in sorted(flicker_result[kind].items()):
            col = stats["column"]
            if col not in df.columns:
                continue
            values = np.sort(df[col].dropna().to_numpy(float))[::-1]
            if not len(values):
                continue
            share = np.arange(1, len(values) + 1) / len(values) * 100.0
            ax.plot(share, values, lw=1.2,
                    ls="-" if kind == "pst" else "--",
                    color=_PHASE_CLR.get(phase.lower(), "gray"),
                    label=f"{kind.title()} {phase}")
        # Two limit lines share this panel, so each is named at the right edge
        # rather than leaving the reader to infer which is which.
        ax.axhline(limit, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.annotate(f"{kind.title()} limit {limit:.2f}", xy=(100, limit),
                    xytext=(-4, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color="#CC0000")

    ax.axvline(5.0, color="#333333", ls=":", lw=1.0)
    ax.annotate("95th percentile\n(what the standards assess)", xy=(5.0, ax.get_ylim()[1]),
                xytext=(8.0, ax.get_ylim()[1] * 0.92), fontsize=8, color="#333333")
    ax.set_xlabel("Share of the recording at or above this severity (%)")
    ax.set_ylabel("Flicker severity")
    ax.set_title("How long each level lasted — narrow at the left is an anomaly, "
                 "wide to the right is a condition")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, outdir, stem, "flicker.png")


# ─────────────────────────────────────────────────────────────────────────────
# POINT-ON-WAVE CAPTURE SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_capture(
    ds,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
    stem: str = "",
) -> None:
    """Instantaneous waveform snapshot of the most severe point-on-wave
    capture (largest half-cycle RMS deviation from nominal voltage)."""
    wfs = getattr(ds, "waveforms", None) or []
    if not wfs:
        return
    nominal = thresh.nominal_voltage

    def severity(wf) -> float:
        fs = wf.get("fs_hz") or 0
        if fs <= 0:
            return 0.0
        w = max(int(round(fs / 60.0 / 2)), 8)
        worst = 0.0
        for x in wf["voltages"].values():
            x = np.asarray(x, dtype=float)
            if len(x) < 2 * w:
                continue
            c = np.cumsum(np.concatenate(([0.0], x * x)))
            rms = np.sqrt((c[w:] - c[:-w]) / w)
            worst = max(worst, float(np.max(np.abs(rms - nominal))))
        return worst

    wf   = max(wfs, key=severity)
    t_ms = np.asarray(wf["t"], dtype=float) * 1000.0
    has_i = bool(wf.get("currents"))

    fig, axes = plt.subplots(2 if has_i else 1, 1,
                             figsize=(14, 7 if has_i else 4), sharex=True)
    if not has_i:
        axes = [axes]

    ax = axes[0]
    peak = nominal * np.sqrt(2)
    for ph, x in wf["voltages"].items():
        n = min(len(x), len(t_ms))
        ax.plot(t_ms[:n], x[:n], color=_PHASE_CLR.get(ph, "gray"), lw=0.9,
                label=f"V {ph.upper()}")
    ax.axhline(peak,  color="red", ls=":", lw=0.8, alpha=0.7,
               label=f"±nominal peak ({peak:.0f} V)")
    ax.axhline(-peak, color="red", ls=":", lw=0.8, alpha=0.7)
    ax.set_ylabel("Voltage (V)")
    ax.set_title(
        f"Worst Waveform Capture — {wf['timestamp']:%Y-%m-%d %H:%M:%S.%f}"[:60]
        + f"  ({len(wfs)} captures in recording)"
    )
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    if has_i:
        ax = axes[1]
        for ph, x in wf["currents"].items():
            n = min(len(x), len(t_ms))
            lbl = "I N" if ph == "n" else f"I {ph.upper()}"
            clr = _NEUTRAL_CLR if ph == "n" else _PHASE_CLR.get(ph, "gray")
            ax.plot(t_ms[:n], x[:n], color=clr, lw=0.9, label=lbl)
        ax.set_ylabel("Current (A)")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time from capture start (ms)")
    fig.tight_layout()

    _save(fig, outdir, stem, "waveform_worst.png")

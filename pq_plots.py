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

from pq_constants import Thresholds, _h519_limit

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

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


def plot_voltage(
    df: pd.DataFrame,
    volt_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
) -> None:
    v_cols = [c for c in ["voltage_a", "voltage_b", "voltage_c"] if c in df.columns]
    if not v_cols:
        log.warning("No voltage columns to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"voltage_a": "#2196F3", "voltage_b": "#FF9800", "voltage_c": "#4CAF50"}
    is_split = "voltage_c" not in df.columns
    if is_split:
        labels = {"voltage_a": "L1-N", "voltage_b": "L2-N"}
        topo_title = "Split-Phase Voltage (L-N)"
    else:
        labels = {"voltage_a": "Phase A", "voltage_b": "Phase B", "voltage_c": "Phase C"}
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

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel("RMS Voltage (V)")
    ax.set_title(f"{topo_title} — ANSI C84.1 Compliance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if outdir:
        fig.savefig(outdir / "voltage.png", dpi=150)
        log.info("Saved plot → %s/voltage.png", outdir)
    else:
        plt.show()
    plt.close(fig)


def plot_thd(
    df: pd.DataFrame,
    thd_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
) -> None:
    thd_v = [c for c in ["thd_voltage_a", "thd_voltage_b", "thd_voltage_c"] if c in df.columns]
    thd_i = [c for c in ["thd_current_a", "thd_current_b", "thd_current_c"] if c in df.columns]

    if not thd_v and not thd_i:
        log.warning("No THD columns to plot.")
        return

    n_plots = int(bool(thd_v)) + int(bool(thd_i))
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    for cols, limit, label, key in [
        (thd_v, thresh.thd_voltage_limit, "Voltage THD (%)", "voltage"),
        (thd_i, thresh.thd_current_limit, "Current THD (%)", "current"),
    ]:
        if not cols:
            continue
        ax = axes[plot_idx]
        plot_idx += 1
        colors_map = {
            f"thd_{key}_a": "#2196F3", f"thd_{key}_b": "#FF9800", f"thd_{key}_c": "#4CAF50"
        }
        for col in cols:
            ax.plot(df.index, df[col], color=colors_map.get(col, "gray"),
                    lw=0.8, label=col.split("_")[-1].upper())
        ax.axhline(limit, color="red", ls="--", lw=1.0, label=f"IEEE 519 limit ({limit}%)")

        viol_ts_list = thd_result[key].get("violation_timestamps", [])
        if viol_ts_list:
            viol_idx = pd.DatetimeIndex(viol_ts_list)
            _shade_violations(ax, viol_idx, df.index)

        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{label.split(' ')[0]} THD — IEEE 519 Compliance")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    axes[-1].set_xlabel("Time")
    fig.tight_layout()

    if outdir:
        fig.savefig(outdir / "thd.png", dpi=150)
        log.info("Saved plot → %s/thd.png", outdir)
    else:
        plt.show()
    plt.close(fig)


def plot_summary(
    df: pd.DataFrame,
    imb_result: dict,
    outdir: Optional[Path] = None,
) -> None:
    """Four-panel summary: voltage imbalance, power factor, real/reactive power."""
    panels = []
    if "imbalance_series" in imb_result:
        panels.append(("Voltage Imbalance (%)", imb_result["imbalance_series"],
                        imb_result.get("limit_pct"), "#9C27B0"))
    if "power_factor" in df.columns:
        panels.append(("Power Factor", df["power_factor"], None, "#009688"))
    if "power_real" in df.columns:
        panels.append(("Real Power (kW)", df["power_real"], None, "#F44336"))
    if "power_reactive" in df.columns:
        panels.append(("Reactive Power (kVAR)", df["power_reactive"], None, "#FF5722"))

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

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    axes[-1].set_xlabel("Time")
    fig.suptitle("Power Quality Summary", fontsize=11)
    fig.tight_layout()

    if outdir:
        fig.savefig(outdir / "summary.png", dpi=150)
        log.info("Saved plot → %s/summary.png", outdir)
    else:
        plt.show()
    plt.close(fig)


def plot_harmonic_spectrum(
    df: pd.DataFrame,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
) -> None:
    """Bar chart of median H3–H13 per phase (% of IL).

    Shows the three-phase harmonic spectrum with IEEE 519-2022 per-order limits
    where ISC/IL is known.  Harmonics are stored as Amps; divided by IL here to
    display as % of IL for direct comparison against limits.
    """
    orders = [3, 5, 7, 9, 11, 13]
    phases = [("a", "Phase A", "#2196F3"), ("b", "Phase B", "#FF9800"), ("c", "Phase C", "#4CAF50")]

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
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (ph, label, color) in enumerate(phases):
        offset = (i - 1) * width
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

    if outdir:
        fig.savefig(outdir / "harmonic_spectrum.png", dpi=150)
        log.info("Saved plot → %s/harmonic_spectrum.png", outdir)
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ITIC CURVE
# Reference: "ITI (CBEMA) Curve Application Note," Information Technology
# Industry Council (ITIC), 2000.  Superseded the CBEMA curve originally
# referenced in ANSI/IEEE 446-1987.  Referenced by IEEE 1159-2019 as the
# standard voltage tolerance envelope for information technology equipment.
# Applicable to 120 V nominal (120/208 V and 120/240 V, 60 Hz systems).
# ─────────────────────────────────────────────────────────────────────────────

# Step-function boundary lines (duplicate x-values create vertical segments)
_ITIC_UPPER_MS_STEP  = np.array([0.001, 1,   1,   3,   3,   20,  20,  500, 500, 1e6])
_ITIC_UPPER_PCT_STEP = np.array([500,   500, 200, 200, 140, 140, 120, 120, 110, 110])
_ITIC_LOWER_MS_STEP  = np.array([0.001, 20,  20,  500, 500, 1e4, 1e4, 1e6])
_ITIC_LOWER_PCT_STEP = np.array([0,     0,   70,  70,  80,  80,  90,  90 ])


def _itic_upper_v(x: np.ndarray) -> np.ndarray:
    """ITIC upper boundary (% nominal) at each duration x (ms)."""
    r = np.full_like(x, 110.0, dtype=float)
    r[x < 500] = 120.0
    r[x < 20]  = 140.0
    r[x < 3]   = 200.0
    r[x < 1]   = 500.0
    return r


def _itic_lower_v(x: np.ndarray) -> np.ndarray:
    """ITIC lower boundary (% nominal) at each duration x (ms)."""
    r = np.full_like(x, 90.0, dtype=float)
    r[x < 10000] = 80.0
    r[x < 500]   = 70.0
    r[x < 20]    = 0.0
    return r


def plot_itic(
    events: pd.DataFrame,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
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

    phase_colors = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
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
    outpath = (outdir or Path(".")) / "itic_curve.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ITIC curve plot saved → %s", outpath)


def plot_neutral_health(
    ds,
    neutral_result: dict,
    thresh: Thresholds,
    outdir: Optional[Path] = None,
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
            color="#2196F3", lw=0.8, label="L1-N (voltage_a)")
    ax.plot(aligned.index, aligned["voltage_b"],
            color="#FF9800", lw=0.8, label="L2-N (voltage_b)")
    vmin = nom * (1 - thresh.volt_tolerance)
    vmax = nom * (1 + thresh.volt_tolerance)
    ax.axhline(vmin, color="red", ls="--", lw=0.8, alpha=0.7, label=f"ANSI lower ({vmin:.1f} V)")
    ax.axhline(vmax, color="red", ls="--", lw=0.8, alpha=0.7, label=f"ANSI upper ({vmax:.1f} V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("L1-N and L2-N Voltages")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

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

    outpath = (outdir or Path(".")) / "neutral_health.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Neutral health plot saved → %s", outpath)

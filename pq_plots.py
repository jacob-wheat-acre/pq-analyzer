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

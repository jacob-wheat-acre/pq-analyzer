"""
validate_maxmin.py — Diagnostic: verify obs[24] max-min parsing against a real Watkins .pqd file.

Usage:
    python validate_maxmin.py path/to/site.pqd

Prints:
  - Which branch the parser took (4×n full tuples vs 2×n values-only)
  - Per-channel comparison: avg (obs[23]) | peak (obs[24] max) | min (obs[24] min)
  - Plausibility flags: any row where peak < avg or min > avg
  - Channel-level stats: count of implausible rows and worst offender

A physically sound recording will always have:
    min ≤ avg ≤ peak   (for magnitudes like V, I, THD)

If peak < avg for many rows the 4×n stride is wrong — try 2×n.
If min > avg the quality-word interleaving is wrong — strides need checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make the pq_* modules importable regardless of working directory ──────────
sys.path.insert(0, str(Path(__file__).parent))

from pq_adapter import ProntoAdapter, ChannelMapper, extract_dataset


CHANNELS = [
    "voltage_a", "voltage_b", "voltage_c",
    "current_a", "current_b", "current_c",
    "current_neutral",
    "thd_current_a", "thd_current_b", "thd_current_c",
    "thd_voltage_a", "thd_voltage_b", "thd_voltage_c",
    "flicker_pst", "flicker_plt",
    "power_factor",
    "kfactor_meter",
]


def _banner(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def main(pqd_path: str) -> None:
    fp = Path(pqd_path)
    if not fp.exists():
        print(f"ERROR: file not found: {fp}")
        sys.exit(1)

    print(f"\nLoading {fp.name}  ({fp.stat().st_size / 1e6:.1f} MB) …")
    adapter = ProntoAdapter(fp)
    mapper  = ChannelMapper()
    ds      = extract_dataset(adapter, mapper)
    df      = ds.df

    # ── Branch detection ──────────────────────────────────────────────────────
    _banner("obs[24] branch detection")

    peaks = adapter.interval_peaks
    mins  = adapter.interval_mins

    if not peaks:
        print("  obs[24] record NOT found or produced no channels.")
        print("  Check _load_v2_maxmin() — the label search may need adjustment.")
        return

    # Infer which branch from a representative channel
    sample_ch   = next(iter(peaks))
    sample_arr  = peaks[sample_ch]
    avg_col     = sample_ch
    n_avg       = len(df[avg_col].dropna()) if avg_col in df.columns else 0
    n_peak      = len(sample_arr)

    print(f"  Interval avg row count (obs[23]) : {n_avg:,}")
    print(f"  Peak array length (obs[24])      : {n_peak:,}")
    print(f"  Min channels loaded              : {len(mins)}")
    print(f"  Peak channels loaded             : {len(peaks)}")

    if n_peak > 0 and n_avg > 0:
        ratio = n_peak / n_avg
        if abs(ratio - 1.0) < 0.05:
            print(f"  Branch taken: 2×n (values only — min will be NaN)")
        else:
            print(f"  Unexpected length ratio: {ratio:.3f} — investigate parser")
    else:
        print("  Could not determine branch (empty arrays)")

    # ── Per-channel comparison table ──────────────────────────────────────────
    _banner("Per-channel: avg | peak | min   (5-number summary, aligned rows)")

    header = f"{'Channel':<22}  {'avg_min':>8}  {'avg_p50':>8}  {'avg_max':>8}  " \
             f"{'peak_max':>9}  {'min_min':>8}  {'flags'}"
    print(header)
    print("-" * 80)

    plausibility_issues: list[tuple[str, int, int]] = []

    for ch in CHANNELS:
        avg_col  = ch
        peak_col = f"{ch}_peak"
        min_col  = f"{ch}_min"

        if avg_col not in df.columns:
            continue

        avg = df[avg_col].dropna()
        if avg.empty:
            continue

        avg_arr = avg.values
        flags: list[str] = []

        # Align peak/min to avg index
        peak_arr = min_arr = None
        if peak_col in df.columns:
            pk = df[peak_col].reindex(avg.index)
            peak_arr = pk.values
        if min_col in df.columns:
            mn = df[min_col].reindex(avg.index)
            min_arr = mn.values

        # Plausibility: peak ≥ avg (for positive-definite quantities)
        n_peak_bad = n_min_bad = 0
        if peak_arr is not None:
            mask = np.isfinite(avg_arr) & np.isfinite(peak_arr)
            n_peak_bad = int(np.sum(peak_arr[mask] < avg_arr[mask] - 0.01))
            if n_peak_bad > 0:
                flags.append(f"peak<avg({n_peak_bad})")
        if min_arr is not None:
            mask = np.isfinite(avg_arr) & np.isfinite(min_arr)
            n_min_bad  = int(np.sum(min_arr[mask] > avg_arr[mask] + 0.01))
            if n_min_bad > 0:
                flags.append(f"min>avg({n_min_bad})")

        if n_peak_bad or n_min_bad:
            plausibility_issues.append((ch, n_peak_bad, n_min_bad))

        peak_max_str = f"{np.nanmax(peak_arr):8.2f}" if peak_arr is not None else "     n/a"
        min_min_str  = f"{np.nanmin(min_arr):8.2f}"  if min_arr is not None  else "     n/a"

        print(
            f"{ch:<22}  "
            f"{np.nanmin(avg_arr):8.2f}  "
            f"{np.nanmedian(avg_arr):8.2f}  "
            f"{np.nanmax(avg_arr):8.2f}  "
            f"{peak_max_str}  "
            f"{min_min_str}  "
            f"{'  '.join(flags) or 'OK'}"
        )

    # ── Plausibility summary ──────────────────────────────────────────────────
    _banner("Plausibility summary")

    if not plausibility_issues:
        print("  All channels: peak ≥ avg and min ≤ avg. Parsing looks correct.")
    else:
        print("  PROBLEMS FOUND — obs[24] stride may be wrong:\n")
        for ch, n_pk, n_mn in plausibility_issues:
            n_avg_rows = len(df[ch].dropna())
            print(f"  {ch:<22}  peak<avg: {n_pk:4d}/{n_avg_rows}  min>avg: {n_mn:4d}/{n_avg_rows}")
        print("\n  Diagnostic: try toggling the 4×n vs 2×n branch threshold in _load_v2_maxmin()")

    # ── Sample rows — spot-check the first and last 5 ─────────────────────────
    _banner("Spot-check: voltage_a avg | peak | min (first 10 rows with data)")

    ch = "voltage_a"
    if ch in df.columns:
        cols = [c for c in [ch, f"{ch}_peak", f"{ch}_min"] if c in df.columns]
        spot = df[cols].dropna(subset=[ch]).head(10)
        print(spot.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        print("  voltage_a not present in DataFrame")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_maxmin.py path/to/site.pqd")
        sys.exit(1)
    main(sys.argv[1])

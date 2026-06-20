from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from pq_constants import (
    Thresholds,
    _H519_ORDERS,
    _LOAD_SIGNATURES,
    _SERVICE_TYPE_LABEL,
    _h519_limit,
    _impedance_range,
    _lookup_isc,
    _tdd_class,
    _tdd_limit,
)
from pq_adapter import PQDataset

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _require(df: pd.DataFrame, *cols: str) -> bool:
    """Return True if all cols exist in df and have at least one finite value."""
    for c in cols:
        if c not in df.columns or df[c].dropna().empty:
            return False
    return True


def check_voltage_compliance(
    df: pd.DataFrame, thresh: Thresholds
) -> dict:
    """ANSI C84.1 voltage compliance check.

    Returns per-phase statistics and the union of all violation timestamps.
    """
    v_cols = ["voltage_a", "voltage_b", "voltage_c"]
    if not any(c in df.columns for c in v_cols):
        return {
            "available":              False,
            "error":                  "No voltage channels found.",
            "phases":                 {},
            "total_pct_out_of_bounds": None,
            "violation_timestamps":   pd.DatetimeIndex([]),
        }

    vmin = thresh.nominal_voltage * (1 - thresh.volt_tolerance)
    vmax = thresh.nominal_voltage * (1 + thresh.volt_tolerance)
    result = {
        "available":              True,
        "error":                  None,
        "nominal_v":              thresh.nominal_voltage,
        "range_v":                (vmin, vmax),
        "phases":                 {},
        "violation_timestamps":   pd.DatetimeIndex([]),
    }

    all_violations = pd.Series(False, index=df.index)
    for col in v_cols:
        if col not in df.columns:
            continue
        s    = df[col].dropna()
        smin = df[f"{col}_min"].reindex(s.index).fillna(s)  if f"{col}_min"  in df.columns else s
        smax = df[f"{col}_peak"].reindex(s.index).fillna(s) if f"{col}_peak" in df.columns else s
        under = smin < vmin
        over  = smax > vmax
        viol  = under | over
        all_violations.loc[viol.index[viol]] = True
        result["phases"][col] = {
            "pct_out_of_bounds": float(viol.mean() * 100),
            "pct_under":         float(under.mean() * 100),
            "pct_over":          float(over.mean() * 100),
            "min_v":             float(smin.min()),
            "max_v":             float(smax.max()),
            "mean_v":            float(s.mean()),
            "used_interval_extremes": smin is not s,
        }

    result["violation_timestamps"] = df.index[all_violations]
    result["total_pct_out_of_bounds"] = float(all_violations.mean() * 100)
    return result


def check_thd(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """IEEE 519-2022 compliance: voltage THD and current TDD.

    Voltage: standard THD (relative to fundamental), limit from thresh.thd_voltage_limit.
    Current: TDD (relative to maximum demand current IL) when thresh.isc_amps is set.
      TDD(t) = THD%(t) × Irms(t) / IL   where IL = max demand current in recording.
      The TDD class limit is selected from IEEE 519-2022 Table 2 via the ISC/IL ratio.
    Falls back to plain THD vs thresh.thd_current_limit when isc_amps is not provided.
    """
    result = {
        "available":            False,
        "error":                None,
        "voltage":              {"available": False},
        "current":              {"available": False},
        "tdd_info":             {},
        "violation_timestamps": pd.DatetimeIndex([]),
    }

    # ── Determine current limit and IL ────────────────────────────────────────
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    il_amps: Optional[float] = None
    if i_cols:
        il_amps = float(df[i_cols].max(axis=1).max())

    if thresh.isc_amps is not None and il_amps and il_amps > 0:
        isc_il      = thresh.isc_amps / il_amps
        current_limit = _tdd_limit(isc_il)
        use_tdd     = True
        result["tdd_info"] = {
            "isc_amps":      thresh.isc_amps,
            "il_amps":       round(il_amps, 1),
            "isc_il_ratio":  round(isc_il, 1),
            "tdd_class":     _tdd_class(isc_il),
            "tdd_limit_pct": current_limit,
            "isc_source":    thresh.isc_source,
        }
        log.info(
            "IEEE 519 TDD: ISC=%.0f A  IL=%.0f A  ISC/IL=%.1f  class %s  limit=%.1f%%",
            thresh.isc_amps, il_amps, isc_il, _tdd_class(isc_il), current_limit,
        )
    else:
        isc_il        = None
        current_limit = thresh.thd_current_limit
        use_tdd       = False
        if i_cols:
            log.warning(
                "No --isc provided; using THD fallback limit %.1f%%. "
                "Pass --isc <amps> for accurate IEEE 519-2022 TDD class.",
                current_limit,
            )

    # ── Voltage THD ───────────────────────────────────────────────────────────
    v_thd_cols = [c for c in ["thd_voltage_a", "thd_voltage_b", "thd_voltage_c"]
                  if c in df.columns]
    if v_thd_cols:
        worst = df[v_thd_cols].max(axis=1).dropna()
        exceed = worst > thresh.thd_voltage_limit
        result["voltage"] = {
            "available":        True,
            "limit_pct":        thresh.thd_voltage_limit,
            "max_thd_pct":      float(worst.max()),
            "mean_thd_pct":     float(worst.mean()),
            "pct_exceeding":    float(exceed.mean() * 100),
            "violation_timestamps": worst.index[exceed].tolist(),
        }
        result["available"] = True

    # ── Current TDD (or THD fallback) ─────────────────────────────────────────
    i_thd_cols = [c for c in ["thd_current_a", "thd_current_b", "thd_current_c"]
                  if c in df.columns]
    if i_thd_cols:
        if use_tdd:
            # TDD(t) = thd_pct(t) × Irms(t) / IL_max
            # Using per-phase pairing where possible; fall back to THD col alone
            tdd_cols: List[pd.Series] = []
            for col in i_thd_cols:
                phase   = col[-1]
                i_col   = f"current_{phase}"
                aligned = df[[col, i_col]].dropna() if i_col in df.columns else None
                if aligned is not None and len(aligned):
                    tdd_cols.append(aligned[col] * aligned[i_col] / il_amps)
                else:
                    tdd_cols.append(df[col].dropna())  # graceful degradation
            worst = pd.concat(tdd_cols, axis=1).max(axis=1).dropna()
            metric = "tdd"
        else:
            worst  = df[i_thd_cols].max(axis=1).dropna()
            metric = "thd"

        exceed = worst > current_limit
        result["current"] = {
            "available":        True,
            "metric":           metric,
            "limit_pct":        current_limit,
            "max_thd_pct":      float(worst.max()),
            "mean_thd_pct":     float(worst.mean()),
            "pct_exceeding":    float(exceed.mean() * 100),
            "violation_timestamps": worst.index[exceed].tolist(),
        }
        result["available"] = True

        # Peak TDD — uses per-interval maximum THD from obs[24] if available
        pk_thd_cols = [f"{c}_peak" for c in i_thd_cols if f"{c}_peak" in df.columns]
        if pk_thd_cols and use_tdd and il_amps:
            pk_tdd_series: List[pd.Series] = []
            for col in pk_thd_cols:
                base_col = col.replace("_peak", "")
                phase    = base_col[-1]
                i_col    = f"current_{phase}"
                aligned  = df[[col, i_col]].dropna() if i_col in df.columns else None
                if aligned is not None and len(aligned):
                    pk_tdd_series.append(aligned[col] * aligned[i_col] / il_amps)
                else:
                    pk_tdd_series.append(df[col].dropna())
            pk_worst  = pd.concat(pk_tdd_series, axis=1).max(axis=1).dropna()
            pk_exceed = pk_worst > current_limit
            result["current"]["peak_max_tdd_pct"]   = round(float(pk_worst.max()), 2)
            result["current"]["peak_pct_exceeding"] = round(float(pk_exceed.mean() * 100), 2)

    v_viol = set(result["voltage"].get("violation_timestamps", []))
    i_viol = set(result["current"].get("violation_timestamps", []))
    result["violation_timestamps"] = pd.DatetimeIndex(sorted(v_viol | i_viol))
    return result


def check_power_factor(df: pd.DataFrame, thresh: Thresholds) -> dict:
    if "power_factor" not in df.columns:
        return {
            "available":            False,
            "error":                "No power factor channel found.",
            "pct_below_limit":      None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }
    pf = df["power_factor"].dropna()
    low = pf < thresh.power_factor_limit
    return {
        "available":            True,
        "error":                None,
        "limit":                thresh.power_factor_limit,
        "min_pf":               float(pf.min()),
        "mean_pf":              float(pf.mean()),
        "pct_below_limit":      float(low.mean() * 100),
        "violation_timestamps": pf.index[low],
    }


def check_individual_harmonics(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """
    IEEE 519-2022 Table 2 per-order harmonic check.
    Requires individual harmonic columns h{n}_current_{a/b/c} in df.
    Returns per-phase, per-order results with pass/fail and worst-case % of IL.
    Only runs when thresh.isc_amps is set (needed for ISC/IL class).
    """
    result: dict = {"available": False, "phases": {}, "worst_order": None,
                    "worst_pct_of_il": 0.0, "overall_pass": True}

    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    if not i_cols:
        return result

    il_amps = float(df[i_cols].max(axis=1).max())
    if il_amps <= 0:
        return result

    if thresh.isc_amps is None:
        result["note"] = "Pass --isc to enable per-order IEEE 519 check"
        return result

    h_cols = [c for c in df.columns if c.startswith("h") and "_current_" in c]
    if not h_cols:
        result["note"] = "Meter did not record individual harmonic orders (only THD totals available)"
        return result

    isc_il = thresh.isc_amps / il_amps
    result["available"] = True
    result["il_amps"] = round(il_amps, 1)
    result["isc_il_ratio"] = round(isc_il, 1)

    worst_pct = 0.0
    worst_order = None

    for ph in ("a", "b", "c"):
        ph_result = {}
        for h in _H519_ORDERS:
            col = f"h{h}_current_{ph}"
            if col not in df.columns:
                continue
            ih = df[col].dropna()
            if len(ih) == 0:
                continue
            limit_pct = _h519_limit(h, isc_il)
            if limit_pct == 0:
                continue
            pct_of_il = ih / il_amps * 100
            max_pct   = float(pct_of_il.max())
            mean_pct  = float(pct_of_il.mean())
            exceeds   = float((pct_of_il > limit_pct).mean() * 100)
            passes    = exceeds == 0
            ph_result[h] = {
                "max_pct_il":   round(max_pct, 2),
                "mean_pct_il":  round(mean_pct, 2),
                "limit_pct_il": limit_pct,
                "pct_exceeding": round(exceeds, 2),
                "pass": passes,
            }
            if not passes:
                result["overall_pass"] = False
            if max_pct > worst_pct:
                worst_pct = max_pct
                worst_order = (h, ph)

        result["phases"][ph] = ph_result

    result["worst_order"] = worst_order
    result["worst_pct_of_il"] = round(worst_pct, 2)
    return result


_NEUTRAL_HARMONIC_ORDERS = (3, 5, 7, 9, 11, 13)
_TRIPLEN_ORDERS           = frozenset({3, 9, 15})

_PST_LIMIT = 1.0    # IEC 61000-3-3 short-term flicker severity limit
_PLT_LIMIT = 0.65   # IEC 61000-3-3 long-term flicker severity limit


def check_neutral_harmonics(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Neutral harmonic analysis — zero-sequence triplen accumulation diagnostic.

    In a 4-wire wye system, H3/H9/H15 (zero-sequence triplens) from single-phase
    nonlinear loads add arithmetically in the neutral rather than canceling.
    This function quantifies:
      - Per-order neutral harmonic current (Amps, mean and max)
      - Triplen vs non-triplen split
      - Accumulation factor: H3_neutral / mean(H3_a, H3_b, H3_c)
          ≈ 0     → H3 cancels (balanced 3-phase, near-zero neutral H3)
          ≈ 1     → one phase dominates
          ≈ 3     → equal H3 from all three phases accumulates fully
          > 3     → resonance amplification
    """
    avail = [c for c in df.columns
             if c.startswith("h") and c.endswith("_current_neutral")]
    if not avail:
        return {"available": False, "note": "No neutral harmonic channels in dataset"}

    result: dict = {
        "available": True,
        "orders":               {},
        "triplen_sum_mean_a":   0.0,
        "nontriplen_sum_mean_a": 0.0,
        "triplen_pct":          0.0,
        "triplen_dominant":     False,
        "accumulation_factor":  None,
    }

    triplen_sum     = pd.Series(0.0, index=df.index)
    nontriplen_sum  = pd.Series(0.0, index=df.index)

    for h in _NEUTRAL_HARMONIC_ORDERS:
        col = f"h{h}_current_neutral"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        result["orders"][h] = {
            "mean_a": round(float(s.mean()), 3),
            "max_a":  round(float(s.max()), 3),
            "is_triplen": h in _TRIPLEN_ORDERS,
        }
        aligned = s.reindex(df.index).fillna(0.0)
        if h in _TRIPLEN_ORDERS:
            triplen_sum = triplen_sum.add(aligned)
        else:
            nontriplen_sum = nontriplen_sum.add(aligned)

    t_mean  = float(triplen_sum.mean())
    nt_mean = float(nontriplen_sum.mean())
    total   = t_mean + nt_mean

    result["triplen_sum_mean_a"]    = round(t_mean, 3)
    result["nontriplen_sum_mean_a"] = round(nt_mean, 3)
    result["triplen_pct"]           = round(t_mean / total * 100, 1) if total > 0 else 0.0
    result["triplen_dominant"]      = total > 0 and t_mean / total > 0.5

    h3_n_col    = "h3_current_neutral"
    h3_ph_cols  = [f"h3_current_{p}" for p in "abc" if f"h3_current_{p}" in df.columns]
    if h3_n_col in df.columns and h3_ph_cols:
        h3_n_mean  = float(df[h3_n_col].dropna().mean())
        h3_ph_mean = float(df[h3_ph_cols].mean(axis=1).mean())
        if h3_ph_mean > 0.01:
            result["accumulation_factor"] = round(h3_n_mean / h3_ph_mean, 2)

    return result


_SOURCE_ORDERS       = (3, 5, 7, 11, 13)   # orders where both V_h and I_h exist in Pronto
_RESONANCE_THRESHOLD = 2.5                  # Z_h/Z_trend > this → parallel resonance suspect
_CUSTOMER_CORR       = 0.50                 # Pearson r > this → customer-injection attribution
_MIN_CORR_PERIODS    = 20                   # minimum non-NaN pairs for reliable correlation


def check_harmonic_sources(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Apparent harmonic impedance Z_h and source attribution per harmonic order.

    For each order in {3, 5, 7, 11, 13} where h{n}_voltage_{ph} and
    h{n}_current_{ph} columns are present in df, computes:

      Z_h   = mean(V_h_ph) / mean(I_h_ph)   [Ω], averaged across available phases
      corr_h = Pearson r between interval V_h and I_h time series
      Z_ratio = Z_h / Z_linear_h             where Z_linear_h = a × h fits through origin

    Resonance: Z_ratio > 2.5 at any order → parallel resonance suspect.
    Attribution heuristic (indicative — exact direction requires phasor data):
      corr > 0.50 → 'customer'  (V and I co-vary → load injection drives both)
      else        → 'indeterminate'
    """
    orders_with_data: dict[int, dict] = {}

    for h in _SOURCE_ORDERS:
        z_per_phase: list[float] = []
        v_series_list: list[pd.Series] = []
        i_series_list: list[pd.Series] = []

        for ph in ("a", "b", "c"):
            cv, ci = f"h{h}_voltage_{ph}", f"h{h}_current_{ph}"
            if cv not in df.columns or ci not in df.columns:
                continue
            v = df[cv].dropna()
            i = df[ci].dropna()
            aligned = v.align(i, join="inner")
            v_al, i_al = aligned[0], aligned[1]
            valid = (v_al > 0) & (i_al > 0)
            if valid.sum() < 3:
                continue
            z_val = float(v_al[valid].mean() / i_al[valid].mean())
            z_per_phase.append(z_val)
            v_series_list.append(v_al[valid])
            i_series_list.append(i_al[valid])

        if not z_per_phase:
            continue

        z_mean = float(np.mean(z_per_phase))

        # Phase-averaged time series for correlation (align on common index)
        corr_r: Optional[float] = None
        if v_series_list and i_series_list:
            v_avg = pd.concat(v_series_list).groupby(level=0).mean()
            i_avg = pd.concat(i_series_list).groupby(level=0).mean()
            v_a2, i_a2 = v_avg.align(i_avg, join="inner")
            if len(v_a2) >= _MIN_CORR_PERIODS:
                corr_r = round(float(v_a2.corr(i_a2)), 3)

        orders_with_data[h] = {
            "z_ohm":  round(z_mean, 4),
            "corr":   corr_r,
            "phases_used": len(z_per_phase),
        }

    if not orders_with_data:
        return {
            "available": False,
            "note": "No orders with both voltage and current harmonic channels",
        }

    # Fit Z_linear(h) = a × h through origin — expected for purely inductive source
    h_arr = np.array(list(orders_with_data.keys()), dtype=float)
    z_arr = np.array([orders_with_data[h]["z_ohm"] for h in h_arr.astype(int)], dtype=float)
    a_fit = float(np.dot(h_arr, z_arr) / np.dot(h_arr, h_arr)) if len(h_arr) >= 2 else None

    resonant_orders: list[int] = []

    for h, od in orders_with_data.items():
        z_linear = round(a_fit * h, 4) if a_fit is not None else None
        ratio    = round(od["z_ohm"] / z_linear, 2) if z_linear and z_linear > 0 else None

        corr_r = od["corr"]
        if corr_r is not None and corr_r > _CUSTOMER_CORR:
            attribution = "customer"
        else:
            attribution = "indeterminate"

        if ratio is not None and ratio > _RESONANCE_THRESHOLD:
            attribution = "resonance_suspect"
            resonant_orders.append(h)

        od["z_linear_ohm"] = z_linear
        od["z_ratio"]      = ratio
        od["attribution"]  = attribution

    # Overall summary
    attrs = [od["attribution"] for od in orders_with_data.values()]
    customer_count = attrs.count("customer")
    resonance_count = len(resonant_orders)

    if resonance_count > 0:
        overall = "resonance_suspect"
    elif customer_count == len(attrs):
        overall = "customer"
    elif customer_count > 0:
        overall = "mixed"
    else:
        overall = "indeterminate"

    return {
        "available":      True,
        "orders":         orders_with_data,
        "linear_slope_a": round(a_fit, 5) if a_fit is not None else None,
        "resonant_orders": resonant_orders,
        "overall":         overall,
        "note": (
            "Attribution is indicative — Pearson r between V_h and I_h interval series. "
            "Exact source direction requires waveform phasor measurements."
        ),
    }


def check_individual_voltage_harmonics(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """IEEE 519-2022 Table 1 per-order voltage harmonic check.

    For buses < 1.0 kV the individual harmonic limit is 5% of nominal voltage.
    Channels h{n}_voltage_{ph} must be in absolute Volts (from Aac block).
    """
    _V_ORDERS = (3, 5, 7, 11, 13)
    INDIV_LIMIT = 5.0  # % of nominal for service voltage < 1 kV

    result: dict = {
        "available": False, "phases": {}, "worst_order": None,
        "worst_pct_nom": 0.0, "overall_pass": True, "limit_pct": INDIV_LIMIT,
    }

    v_nom = thresh.nominal_voltage
    if v_nom <= 0:
        return result

    v_h_cols = [c for c in df.columns if c.startswith("h") and "_voltage_" in c]
    if not v_h_cols:
        result["note"] = "No per-order voltage harmonic channels available"
        return result

    result["available"] = True
    result["nominal_v"] = v_nom

    worst_pct = 0.0
    worst_order = None

    for ph in ("a", "b", "c"):
        ph_result: dict = {}
        for h in _V_ORDERS:
            col = f"h{h}_voltage_{ph}"
            if col not in df.columns:
                continue
            vh = df[col].dropna()
            if len(vh) == 0:
                continue
            pct_nom = vh / v_nom * 100
            max_pct  = float(pct_nom.max())
            mean_pct = float(pct_nom.mean())
            passes   = max_pct <= INDIV_LIMIT
            ph_result[h] = {
                "max_pct_nom":  round(max_pct, 2),
                "mean_pct_nom": round(mean_pct, 2),
                "limit_pct":    INDIV_LIMIT,
                "pass":         passes,
            }
            if not passes:
                result["overall_pass"] = False
            if max_pct > worst_pct:
                worst_pct = max_pct
                worst_order = (h, ph)
        result["phases"][ph] = ph_result

    result["worst_order"] = worst_order
    result["worst_pct_nom"] = round(worst_pct, 2)
    return result


def check_harmonic_statistics(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """
    IEEE 519-2022 Clause 5 statistical compliance evaluation.

    Three evaluation windows per the standard:
    - ST weekly:   P95 over 7-day period vs 1.0× limit  (primary compliance)
                   P99 over 7-day period vs 1.5× limit
    - VST daily:   daily P99 vs 2.0× limit

    5-minute interval data is used as a proxy for IEC 61000-4-30 Short Time
    (10-min) measurements.  True VST (3-second) data is not available from this
    export format; daily P99 of 5-minute data is a conservative lower bound
    (5-min P99 ≤ true 3-second P99) but may miss short-duration peaks.

    Voltage harmonics: per-order values not available in this meter format;
    voltage THD check only.  Per IEEE 519-2022, voltage harmonics exclude the
    P99 short-time check.
    """
    result: dict = {
        "available": False,
        "method_note": (
            "5-min interval data used as IEC 61000-4-30 ST (10-min) proxy. "
            "Daily VST P99 approximated from 5-min data — conservative but may "
            "not capture sub-minute harmonic peaks."
        ),
    }

    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    if not i_cols:
        return result

    il_amps = float(df[i_cols].max(axis=1).max())
    if il_amps <= 0:
        return result

    if thresh.isc_amps is None:
        result["note"] = "ISC not provided — statistical harmonic check requires --isc"
        return result

    h_cols = [c for c in df.columns if c.startswith("h") and "_current_" in c]
    thd_cols = [c for c in df.columns if c.startswith("thd_current_")]
    if not h_cols and not thd_cols:
        result["note"] = "No harmonic channels available"
        return result

    isc_il = thresh.isc_amps / il_amps
    period_days = (df.index[-1] - df.index[0]).total_seconds() / 86400

    result.update({
        "available": True,
        "il_amps": round(il_amps, 1),
        "isc_il_ratio": round(isc_il, 1),
        "period_days": round(period_days, 2),
        "period_note": (
            f"Recording {period_days:.1f} d (< 7-day window); "
            "percentiles computed over full period."
        ) if period_days < 7 else (
            f"Recording {period_days:.1f} d; worst 7-day window reported."
        ),
    })

    def _weekly(s: pd.Series, lim: float, exclude_p99: bool = False) -> dict:
        vals = s.dropna()
        if len(vals) < 5:
            return {}
        if period_days >= 7:
            chunks = [g for _, g in vals.resample("7D") if len(g) >= 10]
            if not chunks:
                chunks = [vals]
            p95 = float(max(g.quantile(0.95) for g in chunks))
            p99 = float(max(g.quantile(0.99) for g in chunks))
        else:
            p95 = float(vals.quantile(0.95))
            p99 = float(vals.quantile(0.99))
        p95_pass = bool(p95 <= lim)
        p99_pass = bool(p99 <= 1.5 * lim) if not exclude_p99 else None
        return {
            "p95": round(p95, 3), "p99": round(p99, 3),
            "limit": round(lim, 2), "limit_1p5x": round(1.5 * lim, 2),
            "p95_pass": p95_pass, "p99_pass": p99_pass,
            "p95_margin": round(lim - p95, 3),
            "p99_margin": round(1.5 * lim - p99, 3) if not exclude_p99 else None,
        }

    def _daily(s: pd.Series, lim_2x: float) -> dict:
        vals = s.dropna()
        if len(vals) < 5:
            return {}
        daily_p99 = vals.groupby(vals.index.date).quantile(0.99)
        worst = float(daily_p99.max())
        return {
            "worst_day": str(daily_p99.idxmax()),
            "p99": round(worst, 3),
            "limit_2x": round(lim_2x, 2),
            "pass": bool(worst <= lim_2x),
            "margin": round(lim_2x - worst, 3),
        }

    weekly: dict = {}
    daily_vst: dict = {}
    overall_pass = True

    for h in _H519_ORDERS:
        lim = _h519_limit(h, isc_il)
        if lim == 0:
            continue
        key = f"h{h}"
        weekly[key] = {}
        daily_vst[key] = {}
        for ph in ("a", "b", "c"):
            col = f"h{h}_current_{ph}"
            if col not in df.columns:
                continue
            s = df[col].dropna() / il_amps * 100
            w = _weekly(s, lim)
            if w:
                weekly[key][ph] = w
                if not w["p95_pass"] or w["p99_pass"] is False:
                    overall_pass = False
            d = _daily(s, 2.0 * lim)
            if d:
                daily_vst[key][ph] = d
                if not d["pass"]:
                    overall_pass = False

    tdd_lim = _tdd_limit(isc_il)
    weekly["thd"] = {}
    daily_vst["thd"] = {}
    for ph in ("a", "b", "c"):
        col = f"thd_current_{ph}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        w = _weekly(s, tdd_lim)
        if w:
            weekly["thd"][ph] = w
            if not w["p95_pass"] or w["p99_pass"] is False:
                overall_pass = False
        d = _daily(s, 2.0 * tdd_lim)
        if d:
            daily_vst["thd"][ph] = d
            if not d["pass"]:
                overall_pass = False

    result.update({
        "weekly": weekly, "daily_vst": daily_vst,
        "overall_pass": overall_pass,
        "tdd_limit": round(tdd_lim, 1),
        "isc_class": _tdd_class(isc_il),
    })
    return result


def check_voltage_imbalance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """NEMA MG1 voltage unbalance = max_phase_deviation / average_voltage × 100.

    IEEE 1159-2009 recommends flagging above 3 %.
    """
    v_cols = [c for c in ["voltage_a", "voltage_b", "voltage_c"] if c in df.columns]
    if len(v_cols) < 2:
        return {
            "available":            False,
            "error":                "Need at least two voltage phases for imbalance calculation.",
            "pct_exceeding":        None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }

    vdf = df[v_cols].dropna()
    avg  = vdf.mean(axis=1)
    dev  = (vdf.subtract(avg, axis=0)).abs().max(axis=1)
    imbalance = np.where(avg > 0, dev / avg * 100, np.nan)
    imb_series = pd.Series(imbalance, index=vdf.index)
    exceed = imb_series > thresh.imbalance_limit

    return {
        "available":            True,
        "error":                None,
        "limit_pct":            thresh.imbalance_limit,
        "max_imbalance_pct":    float(np.nanmax(imbalance)),
        "mean_imbalance_pct":   float(np.nanmean(imbalance)),
        "pct_exceeding":        float(exceed.mean() * 100),
        "imbalance_series":     imb_series,
        "violation_timestamps": imb_series.index[exceed],
    }


def check_current_imbalance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Current imbalance check per PSC procedure (limit: 10 %).

    Imbalance = max phase deviation from 3-phase average / average × 100 %.
    Also reports neutral current statistics if the channel is present —
    high neutral current indicates either load imbalance or triplen harmonics.
    """
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    if len(i_cols) < 2:
        return {
            "available":            False,
            "error":                "Need at least two current phases for imbalance calculation.",
            "pct_exceeding":        None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }

    idf = df[i_cols].dropna()
    avg = idf.mean(axis=1)
    dev = idf.subtract(avg, axis=0).abs().max(axis=1)
    # Skip rows where average current is negligible (avoids divide-near-zero noise)
    imbalance = np.where(avg > 1.0, dev / avg * 100, np.nan)
    imb_series = pd.Series(imbalance, index=idf.index)
    exceed = imb_series > thresh.current_imbalance_limit

    result: dict = {
        "available":            True,
        "error":                None,
        "limit_pct":            thresh.current_imbalance_limit,
        "max_imbalance_pct":    float(np.nanmax(imbalance)),
        "mean_imbalance_pct":   float(np.nanmean(imbalance)),
        "pct_exceeding":        float(exceed.mean() * 100),
        "violation_timestamps": imb_series.index[exceed],
    }

    if "current_neutral" in df.columns:
        In = df["current_neutral"].dropna()
        avg_phase = avg.reindex(In.index)
        in_pct = np.where(avg_phase > 1.0, In.values / avg_phase.values * 100, np.nan)
        result["neutral_current"] = {
            "mean_amps":            round(float(In.mean()), 1),
            "max_amps":             round(float(In.max()), 1),
            "mean_pct_of_phase":    round(float(np.nanmean(in_pct)), 1),
            "max_pct_of_phase":     round(float(np.nanmax(in_pct)), 1),
        }

    return result


def check_demand(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Transformer loading and demand analysis.

    Computes:
    - Peak and mean apparent power (kVA)
    - 8-hour rolling peak demand — compared to transformer nameplate if provided
      (per Xcel loading guide: transformers may exceed nameplate for 8-hour peaks
      if load falls below nameplate in off-peak hours)
    - Load factor = mean demand / peak demand
    - Real and reactive power summaries
    """
    result: dict = {"available": False, "error": None}

    # ── Apparent power ─────────────────────────────────────────────────────────
    if "power_real" in df.columns and "power_reactive" in df.columns:
        apparent = np.sqrt(df["power_real"] ** 2 + df["power_reactive"] ** 2).dropna()
    elif "power_real" in df.columns and "power_factor" in df.columns:
        pf = df["power_factor"].replace(0, np.nan)
        apparent = (df["power_real"] / pf).dropna()
    else:
        apparent = pd.Series(dtype=float)

    if len(apparent) > 0:
        peak_kva  = float(apparent.max()) / 1000
        mean_kva  = float(apparent.mean()) / 1000
        load_factor = mean_kva / peak_kva if peak_kva > 0 else float("nan")

        # 8-hour rolling mean: window width in samples
        if len(df.index) > 1:
            interval_min = (df.index[1] - df.index[0]).total_seconds() / 60
        else:
            interval_min = 5.0
        win_8h = max(1, int(round(8 * 60 / interval_min)))
        peak_8h_kva = float(apparent.rolling(win_8h, min_periods=1).mean().max()) / 1000

        result["apparent_power"] = {
            "peak_kva":    round(peak_kva, 1),
            "mean_kva":    round(mean_kva, 1),
            "peak_8h_kva": round(peak_8h_kva, 1),
            "load_factor": round(load_factor, 3) if not np.isnan(load_factor) else None,
        }

        if thresh.transformer_kva is not None:
            pct = peak_8h_kva / thresh.transformer_kva * 100
            result["transformer"] = {
                "nameplate_kva": thresh.transformer_kva,
                "peak_8h_kva":   round(peak_8h_kva, 1),
                "pct_nameplate": round(pct, 1),
                "overloaded":    pct > 100,
            }

    if "power_real" in df.columns:
        kw = df["power_real"].dropna() / 1000
        result["real_power"] = {
            "peak_kw": round(float(kw.max()), 1),
            "mean_kw": round(float(kw.mean()), 1),
        }

    if "power_reactive" in df.columns:
        kvar = df["power_reactive"].dropna() / 1000
        result["reactive_power"] = {
            "peak_kvar": round(float(kvar.max()), 1),
            "mean_kvar": round(float(kvar.mean()), 1),
        }

    # True interval peak current from obs[24] max-min record
    pk_i_cols = [f"current_{ph}_peak" for ph in ("a", "b", "c")
                 if f"current_{ph}_peak" in df.columns]
    if pk_i_cols:
        pk_i = df[pk_i_cols].max(axis=1).dropna()
        result["peak_current"] = {
            "max_a":  round(float(pk_i.max()), 1),
            "mean_a": round(float(pk_i.mean()), 1),
            "phases": {
                col.split("_")[1]: round(float(df[col].dropna().max()), 1)
                for col in pk_i_cols
            },
        }

    # Mark available if any sub-results were populated
    data_keys = {"apparent_power", "real_power", "reactive_power", "transformer", "peak_current"}
    if any(k in result for k in data_keys):
        result["available"] = True
    else:
        result["error"] = "No real or reactive power channels found."

    return result


def detect_events(ds: PQDataset, thresh: Thresholds) -> dict:
    """Threshold-based event detection.

    When ``ds.has_adaptive`` is True, uses cycle-level (≈17 ms) adaptive
    records for higher-fidelity sag/swell detection and adds IEC 61000-3-3
    flicker events from PST/PLT channels.  Falls back to 5-minute interval
    averages (augmented by obs[24] min/peak columns) when adaptive data is
    absent.

    Detects:
      - voltage_sag   : V < 90 % nominal (leading edge)
      - voltage_swell : V > 110 % nominal (leading edge)
      - voltage_spike : |ΔV| > event_delta_pct × nominal in one sample
      - flicker_pst   : adap_pst > 1.0  (adaptive only)
      - flicker_plt   : adap_plt > 0.65 (adaptive only)
      - current_step  : |ΔI| > 25 % of mean current
    """
    events: list = []
    nominal      = thresh.nominal_voltage
    sag_thresh   = 0.90 * nominal
    swell_thresh = 1.10 * nominal
    delta_v      = thresh.event_delta_pct * nominal

    use_adaptive = ds.has_adaptive
    data_source  = "adaptive" if use_adaptive else "interval"

    if use_adaptive:
        adf = ds.adaptive_df
        assert adf is not None

        # ── Voltage sag/swell at cycle resolution ─────────────────────────────
        for vcol, phase in [("van_v", "A"), ("vbn_v", "B"), ("vcn_v", "C")]:
            if vcol not in adf.columns:
                continue
            s = adf[vcol].dropna()
            s_vals  = s.values
            s_idx   = s.index
            pos_map = {ts: i for i, ts in enumerate(s_idx)}
            sample_ms = (
                (s_idx[1] - s_idx[0]).total_seconds() * 1000
                if len(s_idx) > 1 else 16.7
            )

            sag_starts   = s[(s < sag_thresh)   & (s.shift(1) >= sag_thresh)].index
            swell_starts = s[(s > swell_thresh)  & (s.shift(1) <= swell_thresh)].index

            for ts in sag_starts:
                loc = pos_map[ts]
                end = loc
                while end + 1 < len(s_vals) and s_vals[end + 1] < sag_thresh:
                    end += 1
                events.append({
                    "timestamp":   ts,
                    "type":        "voltage_sag",
                    "phase":       phase,
                    "value_v":     float(np.min(s_vals[loc: end + 1])),
                    "duration_ms": (s_idx[end] - ts).total_seconds() * 1000 + sample_ms,
                })

            for ts in swell_starts:
                loc = pos_map[ts]
                end = loc
                while end + 1 < len(s_vals) and s_vals[end + 1] > swell_thresh:
                    end += 1
                events.append({
                    "timestamp":   ts,
                    "type":        "voltage_swell",
                    "phase":       phase,
                    "value_v":     float(np.max(s_vals[loc: end + 1])),
                    "duration_ms": (s_idx[end] - ts).total_seconds() * 1000 + sample_ms,
                })
            diffs = s.diff().abs()
            for ts in diffs[diffs > delta_v].index:
                events.append({"timestamp": ts, "type": "voltage_spike", "phase": phase,
                               "delta_v": float(diffs.loc[ts])})

        # ── Flicker events (IEC 61000-3-3) ────────────────────────────────────
        if "adap_pst" in adf.columns:
            pst = adf["adap_pst"].dropna()
            for ts in pst[(pst > _PST_LIMIT) & (pst.shift(1) <= _PST_LIMIT)].index:
                events.append({"timestamp": ts, "type": "flicker_pst", "phase": "A",
                               "value": float(pst.loc[ts])})

        if "adap_plt" in adf.columns:
            plt_ = adf["adap_plt"].dropna()
            for ts in plt_[(plt_ > _PLT_LIMIT) & (plt_.shift(1) <= _PLT_LIMIT)].index:
                events.append({"timestamp": ts, "type": "flicker_plt", "phase": "A",
                               "value": float(plt_.loc[ts])})

        # ── Current step changes at cycle resolution ───────────────────────────
        for icol, phase in [("ia_a", "A"), ("ib_a", "B"), ("ic_a", "C")]:
            if icol not in adf.columns:
                continue
            s = adf[icol].dropna()
            mean_i  = s.mean()
            delta_i = 0.25 * mean_i if mean_i > 0 else 5.0
            diffs = s.diff().abs()
            for ts in diffs[diffs > delta_i].index:
                events.append({"timestamp": ts, "type": "current_step", "phase": phase,
                               "delta_a": float(diffs.loc[ts])})

    else:
        # ── Interval fallback (5-min averages + obs[24] min/peak) ─────────────
        df = ds.df
        for col in ["voltage_a", "voltage_b", "voltage_c"]:
            if col not in df.columns:
                continue
            s     = df[col].dropna()
            s_low = df[f"{col}_min"].reindex(s.index).fillna(s)  if f"{col}_min"  in df.columns else s
            s_hi  = df[f"{col}_peak"].reindex(s.index).fillna(s) if f"{col}_peak" in df.columns else s
            phase = col.split("_")[1].upper()

            sag_starts   = s_low[(s_low < sag_thresh)   & (s_low.shift(1) >= sag_thresh)].index
            swell_starts = s_hi[ (s_hi  > swell_thresh)  & (s_hi.shift(1)  <= swell_thresh)].index
            for ts in sag_starts:
                events.append({"timestamp": ts, "type": "voltage_sag",   "phase": phase,
                               "value_v": float(s_low.loc[ts])})
            for ts in swell_starts:
                events.append({"timestamp": ts, "type": "voltage_swell", "phase": phase,
                               "value_v": float(s_hi.loc[ts])})

            diffs = s.diff().abs()
            for ts in diffs[diffs > delta_v].index:
                events.append({"timestamp": ts, "type": "voltage_spike", "phase": phase,
                               "delta_v": float(diffs.loc[ts])})

        for col in ["current_a", "current_b", "current_c"]:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            phase = col.split("_")[1].upper()
            mean_i  = s.mean()
            delta_i = 0.25 * mean_i if mean_i > 0 else 5.0
            diffs = s.diff().abs()
            for ts in diffs[diffs > delta_i].index:
                events.append({"timestamp": ts, "type": "current_step", "phase": phase,
                               "delta_a": float(diffs.loc[ts])})

    events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True) \
        if events else pd.DataFrame(columns=["timestamp", "type", "phase"])
    return {
        "event_count": len(events_df),
        "events":      events_df,
        "data_source": data_source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8a. NEUTRAL HEALTH  (split-phase open-neutral detection)
# ─────────────────────────────────────────────────────────────────────────────

def check_neutral_health(ds: PQDataset, thresh: Thresholds) -> dict:
    """
    Assess split-phase neutral integrity. Only meaningful for split-phase topology.

    Combines four independent indicators:
    - Voltage sum stability  : Van + Vbn should hold near 240 V
    - Cross-leg correlation  : healthy legs track together (r > 0.8);
                               open neutral causes opposition (r → −1)
    - Neutral-to-earth Vne   : elevated Vne indicates neutral impedance
    - Coincident opposing events: one leg sags while the other swells
    """
    topology = ds.meta.get("topology", "unknown")
    if topology != "split-phase":
        return {"available": False, "reason": "not split-phase topology"}

    df = ds.df
    if "voltage_a" not in df.columns or "voltage_b" not in df.columns:
        return {
            "available": False,
            "reason": "split-phase topology but missing L1 or L2 voltage channel",
        }

    va = df["voltage_a"].dropna()
    vb = df["voltage_b"].dropna()
    aligned = pd.concat([va, vb], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return {"available": False, "reason": "insufficient overlapping voltage samples"}

    va_a = aligned["voltage_a"]
    vb_a = aligned["voltage_b"]
    nom  = thresh.nominal_voltage

    # ── 1. Voltage sum ────────────────────────────────────────────────────────
    vsum     = va_a + vb_a
    sum_mean = float(vsum.mean())
    sum_std  = float(vsum.std())

    # ── 2. Cross-leg Pearson correlation ─────────────────────────────────────
    leg_corr = float(va_a.corr(vb_a))

    # ── 3. Voltage asymmetry |L1 − L2| ───────────────────────────────────────
    asym      = (va_a - vb_a).abs()
    asym_mean = float(asym.mean())
    asym_max  = float(asym.max())
    asym_pct  = asym_mean / nom * 100 if nom > 0 else 0.0

    # ── 4. Neutral-to-earth voltage ───────────────────────────────────────────
    vne_available = False
    vne_mean      = 0.0
    vne_max       = 0.0
    if ds.has_adaptive and ds.adaptive_df is not None and "vne_v" in ds.adaptive_df.columns:
        vne_raw = ds.adaptive_df["vne_v"].dropna().abs()
        if len(vne_raw) > 0:
            vne_mean      = float(vne_raw.mean())
            vne_max       = float(vne_raw.max())
            vne_available = True

    # ── 5. Coincident opposing sag/swell ──────────────────────────────────────
    n_coincident = 0
    if ds.has_adaptive and ds.adaptive_df is not None:
        adf = ds.adaptive_df
        if "van_v" in adf.columns and "vbn_v" in adf.columns:
            both = pd.concat(
                [adf["van_v"].dropna(), adf["vbn_v"].dropna()], axis=1, join="inner"
            ).dropna()
            if len(both) > 0:
                sag_thr   = nom * 0.90
                swell_thr = nom * 1.10
                n_coincident = int(
                    (
                        ((both["van_v"] < sag_thr) & (both["vbn_v"] > swell_thr)) |
                        ((both["vbn_v"] < sag_thr) & (both["van_v"] > swell_thr))
                    ).sum()
                )

    # ── Severity ──────────────────────────────────────────────────────────────
    if n_coincident >= 3 or (vne_available and vne_max > 5.0) or leg_corr < -0.3:
        severity = "critical"
    elif n_coincident >= 1 or (vne_available and vne_max > 2.0) or leg_corr < 0.0 or sum_std > 5.0:
        severity = "warning"
    elif (vne_available and vne_max > 0.5) or leg_corr < 0.5 or sum_std > 2.0 or asym_pct > 3.0:
        severity = "caution"
    else:
        severity = "normal"

    # ── Plain-language findings ────────────────────────────────────────────────
    findings: List[str] = []

    if n_coincident >= 1:
        s = "s" if n_coincident > 1 else ""
        findings.append(
            f"Detected {n_coincident} coincident opposing sag/swell event{s}: "
            f"one leg below {nom * 0.90:.0f} V while the other exceeded {nom * 1.10:.0f} V "
            "simultaneously. This is a hallmark signature of an open or high-resistance neutral."
        )

    if vne_available:
        if vne_max > 5.0:
            findings.append(
                f"Neutral-to-earth voltage reached {vne_max:.1f} V (mean {vne_mean:.1f} V). "
                "Above 2 V indicates significant neutral impedance; above 5 V is a safety hazard — "
                "investigate immediately."
            )
        elif vne_max > 2.0:
            findings.append(
                f"Neutral-to-earth voltage elevated: max {vne_max:.1f} V, mean {vne_mean:.1f} V. "
                "Investigate neutral conductor connections and the grounding electrode system."
            )
        elif vne_max > 0.5:
            findings.append(
                f"Neutral-to-earth voltage mildly elevated: max {vne_max:.1f} V (normal < 0.5 V). "
                "Monitor and investigate if increasing."
            )
        else:
            findings.append(f"Neutral-to-earth voltage is normal (max {vne_max:.2f} V).")

    if leg_corr < 0.0:
        findings.append(
            f"Cross-leg voltage correlation is negative (r = {leg_corr:.3f}). "
            "When L1 rises, L2 falls — a strong indicator of the neutral floating between legs."
        )
    elif leg_corr < 0.5:
        findings.append(
            f"Cross-leg voltage correlation is weak (r = {leg_corr:.3f}; healthy > 0.80). "
            "Legs are not tracking the source together — investigate neutral continuity."
        )

    if sum_std > 3.0:
        findings.append(
            f"Voltage sum (L1 + L2) is unstable: mean {sum_mean:.1f} V, std {sum_std:.1f} V. "
            "A solid neutral holds L1 + L2 near 240 V with std < 1 V."
        )
    elif sum_std > 1.0:
        findings.append(
            f"Voltage sum (L1 + L2) shows moderate variation: "
            f"mean {sum_mean:.1f} V, std {sum_std:.2f} V."
        )

    if asym_pct > 5.0:
        findings.append(
            f"Sustained voltage asymmetry: mean |L1 − L2| = {asym_mean:.1f} V "
            f"({asym_pct:.1f}% of nominal), max {asym_max:.1f} V. "
            "Investigate load balance and neutral continuity."
        )
    elif asym_pct > 2.0:
        findings.append(
            f"Moderate voltage asymmetry: mean |L1 − L2| = {asym_mean:.1f} V "
            f"({asym_pct:.1f}% of nominal)."
        )

    if not findings:
        findings.append(
            f"Neutral appears healthy: L1 + L2 = {sum_mean:.1f} V (std {sum_std:.2f} V), "
            f"leg correlation r = {leg_corr:.3f}, asymmetry {asym_mean:.1f} V ({asym_pct:.1f}%)."
        )

    return {
        "available":         True,
        "topology":          "split-phase",
        "sample_count":      len(aligned),
        "sum_mean_v":        round(sum_mean, 2),
        "sum_std_v":         round(sum_std, 3),
        "leg_correlation":   round(leg_corr, 3),
        "asym_mean_v":       round(asym_mean, 2),
        "asym_max_v":        round(asym_max, 2),
        "asym_pct":          round(asym_pct, 2),
        "vne_available":     vne_available,
        "vne_mean_v":        round(vne_mean, 2),
        "vne_max_v":         round(vne_max, 2),
        "coincident_events": n_coincident,
        "severity":          severity,
        "findings":          findings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8b. ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Each finding is a plain dict with these keys:
#   category     : str  — "harmonics" | "imbalance" | "voltage" | "demand" | "power_factor"
#   severity     : str  — "critical" | "warning" | "info"
#   title        : str  — headline (~5 words)
#   finding      : str  — what was measured (quantitative)
#   cause        : str  — likely explanation
#   responsibility: str — "utility" | "customer" | "shared" | "unknown"
#   recommendation: str — specific action(s)
#   confidence   : str  — "high" | "medium" | "low"
#   evidence     : dict — key metrics that triggered the rule


def _harmonic_means(df: pd.DataFrame, orders, phases=("a", "b", "c")) -> Dict[int, float]:
    """Return mean per-order harmonic current (amps, averaged across phases) over recording."""
    result = {}
    for h in orders:
        cols = [f"h{h}_current_{p}" for p in phases if f"h{h}_current_{p}" in df.columns]
        if cols:
            result[h] = float(df[cols].values.mean())
    return result


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two non-negative vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _detect_harmonic_signature(df: pd.DataFrame, il_amps: float) -> List[dict]:
    """
    Score each entry in _LOAD_SIGNATURES against the measured harmonic spectrum
    using cosine similarity, then return the top matches as finding dicts.

    Cosine similarity measures spectral *shape* — the absolute THD level does not
    affect which load type wins.  A variability modifier adjusts scores up/down based
    on whether the measured H5 inter-interval CV matches the load type's expected
    stability (steady-state vs. intermittent).
    """
    _ORDERS = [3, 5, 7, 9, 11, 13]
    h_mean = _harmonic_means(df, _ORDERS)
    if len(h_mean) < 3:
        return []

    measured = np.array([h_mean.get(h, 0.0) for h in _ORDERS], dtype=float)
    if np.linalg.norm(measured) == 0:
        return []

    # H5 inter-interval variability (coefficient of variation)
    h5_cols = [f"h5_current_{p}" for p in "abc" if f"h5_current_{p}" in df.columns]
    h5_cv = 0.0
    if h5_cols:
        s = df[h5_cols].values.flatten()
        s = s[~np.isnan(s)]
        if len(s) > 0 and s.mean() > 0:
            h5_cv = float(s.std() / s.mean())

    # Key ratio features from the measured spectrum
    h3 = h_mean.get(3, 0.0)
    h5 = h_mean.get(5, 0.0)
    h7 = h_mean.get(7, 0.0)
    h5_h7 = h5 / max(h7, 0.001)
    h3_h5 = h3 / max(h5, 0.001)

    def _log_ratio_match(r_measured: float, r_ref: float, sigma: float = 0.6) -> float:
        """Log-Gaussian similarity: 1.0 when ratios match, decays as they diverge."""
        if r_ref <= 0 or r_measured <= 0:
            return 0.5
        return float(np.exp(-0.5 * (np.log(r_measured / r_ref) / sigma) ** 2))

    # Score each signature
    scored = []
    for sig in _LOAD_SIGNATURES:
        ref = np.array(sig["spectrum"], dtype=float)
        ref_h3, ref_h5, ref_h7 = ref[0], ref[1], ref[2]

        # Cosine similarity on full spectral vector (shape match)
        cos = _cosine_sim(measured, ref)

        # Ratio match on H5/H7 and H3/H5 — these are the two most discriminating ratios
        # and are not adequately captured by cosine similarity alone when H3 ≈ H5
        m_h5_h7 = _log_ratio_match(h5_h7, ref_h5 / max(ref_h7, 0.001))
        m_h3_h5 = _log_ratio_match(h3_h5, ref_h3 / max(ref_h5, 0.001))

        # Combined score: spectral shape 55%, H5/H7 ratio 30%, H3/H5 ratio 15%
        combined = 0.55 * cos + 0.30 * m_h5_h7 + 0.15 * m_h3_h5

        # Variability modifier — penalise mismatches between expected and observed stability
        ev = sig["variability"]
        if ev == "low" and h5_cv > 0.30:
            combined *= 0.80
        elif ev == "high" and h5_cv < 0.25:
            combined *= 0.65
        elif ev == "medium" and h5_cv > 0.40:
            combined *= 0.85

        scored.append((combined, sig))

    scored.sort(key=lambda t: -t[0])

    # Diagnostic ratios for finding text (h3/h5/h7/h5_h7/h3_h5 already set above)
    h_pcts = {h: round(h_mean.get(h, 0.0) / il_amps * 100, 2) for h in _ORDERS}
    spectrum_str = ", ".join(
        f"H{h}={h_pcts[h]:.1f}%" for h in _ORDERS if h_pcts.get(h, 0) > 0.05
    )

    findings = []
    rank_labels = ["Best match", "Contributing load", "Contributing load"]
    for rank, (sim, sig) in enumerate(scored[:3]):
        if sim < 0.75:
            break
        conf = "high" if sim >= 0.95 else ("medium" if sim >= 0.85 else "low")
        findings.append({
            "category":       "harmonics",
            "severity":       "info",
            "title":          f"{rank_labels[rank]}: {sig['title']}",
            "finding":        (
                f"Spectral similarity {sim:.0%}. Measured spectrum: {spectrum_str}. "
                f"H5/H7={h5_h7:.2f}, H3/H5={h3_h5:.2f}, H5 variability (CV)={h5_cv:.2f}."
            ),
            "cause":          sig["cause"],
            "responsibility": sig["responsibility"],
            "recommendation": sig["recommendation"],
            "confidence":     conf,
            "evidence":       {
                "similarity":   round(sim, 3),
                "signature_id": sig["id"],
                "rank":         rank + 1,
                "h5_h7_ratio":  round(h5_h7, 2),
                "h3_h5_ratio":  round(h3_h5, 2),
                "h5_cv":        round(h5_cv, 3),
                **{f"h{h}_pct_il": h_pcts[h] for h in _ORDERS},
            },
        })

    return findings


def analyze_root_causes(report: dict, ds: PQDataset, thresh: Thresholds) -> List[dict]:
    """
    Analyze all compliance results and dataset to produce root cause findings
    with likely causes, responsibility assignments, and specific recommendations.
    """
    df = ds.df
    findings: List[dict] = []

    pf_flags = report["pass_fail"]
    thd      = report["thd_compliance"]
    ci       = report["current_imbalance"]
    volt_imb = report["voltage_imbalance"]
    volt     = report["voltage_compliance"]
    dem      = report["demand"]
    pfr      = report["power_factor"]
    ih       = report["individual_harmonics"]
    tdd_info = thd.get("tdd_info", {})

    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    il_amps = float(df[i_cols].max(axis=1).max()) if i_cols else 0.0

    # ── Harmonic signature detection ──────────────────────────────────────────
    if il_amps > 0 and any(f"h5_current_{p}" in df.columns for p in "abc"):
        findings.extend(_detect_harmonic_signature(df, il_amps))

    # ── TDD approaching limit (marginal compliance warning) ──────────────────
    c_thd = thd.get("current", {})
    if c_thd.get("available") and tdd_info:
        tdd_max  = c_thd.get("max_thd_pct", 0)
        tdd_lim  = tdd_info.get("tdd_limit_pct", 100)
        utilization = tdd_max / tdd_lim if tdd_lim > 0 else 0
        if 0.75 <= utilization < 1.0:
            findings.append({
                "category":       "harmonics",
                "severity":       "warning",
                "title":          "TDD approaching IEEE 519-2022 limit",
                "finding":        (f"Maximum TDD is {tdd_max:.2f}% against a {tdd_lim:.1f}% limit "
                                   f"({utilization*100:.0f}% of limit consumed). "
                                   "No violation, but limited margin for load growth."),
                "cause":          ("Current harmonic load is close to the IEEE 519-2022 class limit. "
                                   "Additional VFDs, rectifiers, or other nonlinear loads could "
                                   "push TDD over the limit."),
                "responsibility": "shared",
                "recommendation": ("Document existing harmonic sources. Before adding significant "
                                   "nonlinear loads, perform a harmonic study to verify continued "
                                   "compliance. Consider adding input reactors to existing VFDs to "
                                   "create headroom."),
                "confidence":     "high",
                "evidence":       {"tdd_max_pct":  round(tdd_max, 2),
                                   "tdd_limit_pct": tdd_lim,
                                   "pct_of_limit":  round(utilization*100, 1)},
            })

    # ── Neutral current — triplen / imbalance diagnostic ─────────────────────
    neutral_harm = report.get("neutral_harmonics", {})
    if "neutral_current" in ci and il_amps > 0:
        nc      = ci["neutral_current"]
        in_pct  = nc["mean_pct_of_phase"]
        in_max  = nc["max_pct_of_phase"]

        # Prefer actual neutral harmonic data when available; fall back to
        # inferring from phase H3 averages when neutral channels are absent.
        nh_avail       = neutral_harm.get("available", False)
        triplen_pct    = neutral_harm.get("triplen_pct", 0.0)      if nh_avail else None
        acc_factor     = neutral_harm.get("accumulation_factor")    if nh_avail else None
        triplen_dom    = neutral_harm.get("triplen_dominant", False) if nh_avail else None
        h3_n_mean      = neutral_harm.get("orders", {}).get(3, {}).get("mean_a", 0.0) if nh_avail else 0.0

        if not nh_avail:
            h3_phase_mean = _harmonic_means(df, [3]).get(3, 0.0)
            h3_pct_il = h3_phase_mean / il_amps * 100 if il_amps > 0 else 0
            triplen_dom = h3_pct_il > 2.0
        else:
            h3_pct_il = None

        if in_pct > 10:
            ci_mean = ci.get("mean_imbalance_pct", 0)

            if triplen_dom:
                if nh_avail and acc_factor is not None and acc_factor > 3.0:
                    cause_text = (
                        f"Neutral harmonic current is dominated by triplens ({triplen_pct:.0f}% "
                        "of neutral harmonic content) with an accumulation factor of "
                        f"{acc_factor:.1f}× — exceeding the 3× theoretical maximum for balanced "
                        "single-phase loads, which indicates harmonic resonance. At resonance, the "
                        "system impedance amplifies triplen-order (zero-sequence) currents "
                        "beyond what the loads inject."
                    )
                    rec_text = (
                        "Conduct a harmonic impedance scan to identify the resonant frequency. "
                        "Detune existing capacitor banks or add series reactors to shift resonance. "
                        "Verify neutral conductor sizing can withstand resonance-amplified currents."
                    )
                elif nh_avail:
                    cause_text = (
                        f"Neutral harmonic current is dominated by triplens ({triplen_pct:.0f}% "
                        f"of neutral harmonic content, mean H3-neutral = {h3_n_mean:.2f} A, "
                        f"accumulation factor {acc_factor:.1f}×). In 4-wire wye systems, H3, H9, "
                        "and H15 are zero-sequence and add arithmetically in the neutral. "
                        "Sources: single-phase switched-mode power supplies, LED drivers, "
                        "electronic ballasts, and EV chargers."
                    )
                    rec_text = (
                        "Verify neutral conductor sizing handles full triplen harmonic current "
                        f"(mean {nc['mean_amps']:.1f} A, peak {nc['max_amps']:.1f} A). "
                        "Consider a K-rated or harmonic-mitigating transformer. "
                        "Inventory single-phase nonlinear loads by phase to identify dominant sources."
                    )
                else:
                    cause_text = (
                        f"Elevated neutral current ({in_pct:.1f}% of phase average) is consistent "
                        f"with triplen harmonic accumulation (H3 ≈ {h3_pct_il:.1f}% of IL in phases). "
                        "H3, H9, H15 are zero-sequence and add in the neutral rather than canceling."
                    )
                    rec_text = (
                        "Verify neutral conductor sizing. Consider a K-rated transformer. "
                        "Identify single-phase nonlinear loads (SMPS, LED drivers, EV chargers)."
                    )
            else:
                cause_text = (
                    f"Neutral current ({in_pct:.1f}% of phase average) is primarily driven by "
                    f"load imbalance across phases (current imbalance mean {ci_mean:.1f}%). "
                    "Unequal single-phase loads on A, B, C phases produce a residual neutral current."
                )
                rec_text = (
                    "Redistribute single-phase loads to balance current across phases. "
                    "Target < 5% current imbalance to bring neutral current below 5% of phase current."
                )

            evidence: dict = {
                "neutral_mean_pct": round(in_pct, 1),
                "neutral_max_pct":  round(in_max, 1),
            }
            if nh_avail:
                evidence["triplen_pct"]         = triplen_pct
                evidence["accumulation_factor"]  = acc_factor
                evidence["h3_neutral_mean_a"]    = round(h3_n_mean, 3)
            else:
                evidence["h3_pct_il"] = round(h3_pct_il, 2) if h3_pct_il is not None else None

            findings.append({
                "category":       "imbalance",
                "severity":       "warning" if in_pct > 20 else "info",
                "title":          "Elevated neutral current",
                "finding":        (f"Mean neutral current {nc['mean_amps']:.1f} A "
                                   f"({in_pct:.1f}% of phase average). "
                                   f"Peak {nc['max_amps']:.1f} A ({in_max:.1f}%)."),
                "cause":          cause_text,
                "responsibility": "customer",
                "recommendation": rec_text,
                "confidence":     "high" if nh_avail else "medium",
                "evidence":       evidence,
            })

    # ── Harmonic source attribution — resonance detection ────────────────────
    source_harm = report.get("harmonic_sources", {})
    if source_harm.get("available"):
        resonant = source_harm.get("resonant_orders", [])
        if resonant:
            h_res_str = ", ".join(f"H{h} ({h * 60} Hz)" for h in sorted(resonant))
            z_evidence = {
                f"H{h}_z_ratio": source_harm["orders"][h].get("z_ratio")
                for h in resonant
            }
            # Try to estimate resonant order from capacitor reactive power if available
            cap_note = ""
            if "power_reactive" in df.columns:
                kvar_mean = float(df["power_reactive"].dropna().mean()) / 1000
                if kvar_mean < -0.5:
                    cap_note = (
                        f" The site draws {abs(kvar_mean):.1f} kVAR leading — "
                        "capacitor banks are likely present and are a probable resonance source."
                    )
            findings.append({
                "category":       "harmonics",
                "severity":       "warning",
                "title":          f"Parallel resonance suspected at {h_res_str}",
                "finding":        (
                    f"Harmonic impedance at {h_res_str} is {max(source_harm['orders'][h].get('z_ratio', 0) for h in resonant):.1f}× "
                    "higher than the linear inductive trend extrapolated from other orders. "
                    "This signature is consistent with a parallel LC resonance between system "
                    "inductance and capacitor banks at that harmonic frequency."
                ),
                "cause":          (
                    "Parallel resonance forms when the system (transformer + feeder) inductance "
                    "resonates with power factor correction or harmonic filter capacitors. "
                    "At the resonant order, even small harmonic currents produce large "
                    f"harmonic voltages, amplifying both V_h and I_h at that order.{cap_note}"
                ),
                "responsibility": "shared",
                "recommendation": (
                    "Commission a harmonic impedance frequency sweep to confirm the resonant "
                    f"frequency (target: {', '.join(str(h * 60) for h in sorted(resonant))} Hz). "
                    "Detune existing capacitor banks by adding series reactors (typically 5–7% "
                    "of bank kVAR), or switch to a harmonic filter bank tuned below H5 (282 Hz). "
                    "Until resolved, do not add more capacitors without a harmonic study."
                ),
                "confidence":     "medium",
                "evidence":       z_evidence,
            })

        overall = source_harm.get("overall", "indeterminate")
        if overall == "customer" and not resonant:
            # Customer injection confirmed — reinforce any existing harmonic signature finding
            h5_pct = source_harm["orders"].get(5, {}).get("z_ohm", 0)
            corr_vals = [od.get("corr") or 0.0 for od in source_harm["orders"].values()
                         if od.get("corr") is not None]
            mean_corr = round(float(np.mean(corr_vals)), 2) if corr_vals else None
            if mean_corr is not None and mean_corr > 0.60:
                findings.append({
                    "category":       "harmonics",
                    "severity":       "info",
                    "title":          "Harmonic currents confirmed as customer-side injection",
                    "finding":        (
                        f"Voltage and current harmonics are strongly correlated across all "
                        f"measured orders (mean Pearson r = {mean_corr:.2f}). "
                        "This confirms harmonics originate from loads on this service rather "
                        "than from background utility voltage distortion."
                    ),
                    "cause":          (
                        "Customer-side nonlinear loads (VFDs, rectifiers, SMPS) inject "
                        "harmonic currents that develop harmonic voltages across the source "
                        "impedance — causing V_h and I_h to rise and fall together."
                    ),
                    "responsibility": "customer",
                    "recommendation": (
                        "Focus mitigation on customer loads. Options: input reactors on VFDs, "
                        "active harmonic filters, or 12-pulse / 18-pulse drive upgrades. "
                        "Utility-side action (capacitor detuning) is not required at this stage."
                    ),
                    "confidence":     "medium",
                    "evidence":       {"mean_pearson_r": mean_corr,
                                       "orders_tested": list(source_harm["orders"].keys())},
                })

    # ── Current imbalance — utility vs. customer ──────────────────────────────
    if pf_flags.get("current_imbalance") is False:
        vi_mean = volt_imb.get("mean_imbalance_pct", 0)
        ci_mean = ci.get("mean_imbalance_pct", 0)
        if vi_mean < 1.0:
            findings.append({
                "category":       "imbalance",
                "severity":       "warning",
                "title":          "Current imbalance — customer load origin",
                "finding":        (f"Current imbalance mean {ci_mean:.1f}%, "
                                   f"max {ci.get('max_imbalance_pct', 0):.1f}% "
                                   f"(limit 10%). Voltage imbalance is low ({vi_mean:.2f}%), "
                                   "indicating balanced supply voltage."),
                "cause":          ("The utility supply voltage is well balanced but current is "
                                   "not, indicating the imbalance originates from unequal "
                                   "single-phase load distribution on the customer side."),
                "responsibility": "customer",
                "recommendation": ("Survey single-phase loads (lighting, small appliances, "
                                   "plug loads, HVAC controls) and redistribute to balance "
                                   "phase currents. Target < 5% imbalance to reduce neutral "
                                   "current and improve motor efficiency."),
                "confidence":     "high",
                "evidence":       {"current_imb_mean_pct":  round(ci_mean, 2),
                                   "voltage_imb_mean_pct":  round(vi_mean, 2)},
            })
        else:
            findings.append({
                "category":       "imbalance",
                "severity":       "warning",
                "title":          "Current imbalance — investigate supply voltage",
                "finding":        (f"Current imbalance mean {ci_mean:.1f}%, "
                                   f"voltage imbalance mean {vi_mean:.2f}%. "
                                   "Both are elevated; supply may be contributing."),
                "cause":          ("Both voltage and current are imbalanced. Unbalanced supply "
                                   "voltage (from the utility distribution system) will induce "
                                   "current imbalance in motor loads. Customer load imbalance "
                                   "may also be a contributing factor."),
                "responsibility": "shared",
                "recommendation": ("Measure voltage imbalance with all customer loads "
                                   "disconnected to isolate the utility contribution. "
                                   "If voltage imbalance exceeds 1% at no-load, Xcel Energy "
                                   "will investigate the distribution system."),
                "confidence":     "medium",
                "evidence":       {"current_imb_mean_pct": round(ci_mean, 2),
                                   "voltage_imb_mean_pct": round(vi_mean, 2)},
            })

    # ── Voltage imbalance — utility-side ─────────────────────────────────────
    if pf_flags.get("voltage_imbalance") is False:
        vi_max = volt_imb.get("max_imbalance_pct", 0)
        findings.append({
            "category":       "voltage",
            "severity":       "warning",
            "title":          "Voltage imbalance — utility responsibility",
            "finding":        (f"Voltage imbalance max {vi_max:.2f}%, "
                               f"mean {volt_imb.get('mean_imbalance_pct', 0):.2f}% "
                               f"(limit 3%)."),
            "cause":          ("Steady-state voltage imbalance exceeding 3% is typically caused "
                               "by unbalanced distribution transformer loading, asymmetric "
                               "line impedances, a blown capacitor fuse on one phase, "
                               "or an open delta transformer configuration."),
            "responsibility": "utility",
            "recommendation": ("Xcel Energy will investigate. Measure with all customer loads "
                               "disconnected to confirm utility-side origin. Check feeder "
                               "capacitor bank fuses and transformer loading on adjacent services."),
            "confidence":     "high",
            "evidence":       {"voltage_imb_max_pct":  round(vi_max, 2)},
        })

    # ── Voltage — low at high load (secondary drop) ───────────────────────────
    if "phases" in volt and i_cols:
        volt_series = df[[c for c in ["voltage_a","voltage_b","voltage_c"]
                           if c in df.columns]].mean(axis=1)
        load_series = df[i_cols].mean(axis=1)
        if len(volt_series) > 10 and load_series.std() > 5:
            corr = float(volt_series.corr(load_series))
            if corr < -0.5:
                phases = volt["phases"]
                min_v  = min(v["min_v"] for v in phases.values())
                findings.append({
                    "category":       "voltage",
                    "severity":       "warning",
                    "title":          "Voltage drops with increasing load",
                    "finding":        (f"Voltage-to-load correlation = {corr:.2f}. "
                                       f"Minimum recorded voltage {min_v:.1f} V. "
                                       "Voltage tends to decrease as current increases."),
                    "cause":          ("Negative voltage-load correlation indicates resistive "
                                       "voltage drop in the secondary service conductors or "
                                       "transformer impedance. This is more pronounced on long "
                                       "secondary runs or undersized conductors."),
                    "responsibility": "utility" if corr < -0.7 else "shared",
                    "recommendation": ("Review secondary conductor sizing and length. "
                                       "Calculate secondary voltage drop at peak load. "
                                       "If drop exceeds design criteria, conductor upgrade "
                                       "or transformer tap adjustment may be needed."),
                    "confidence":     "medium" if corr > -0.7 else "high",
                    "evidence":       {"volt_load_correlation": round(corr, 3),
                                       "min_voltage_v":         round(min_v, 1)},
                })

    # ── Power factor — inductive load, quantify correction needed ─────────────
    # Residential customers are not subject to the PF tariff clause; skip.
    if (thresh.customer_class != "r"
            and pf_flags.get("power_factor") is False
            and "power_reactive" in df.columns):
        q_mean = float(df["power_reactive"].mean()) / 1000   # kVAR
        p_mean = float(df["power_real"].mean()) / 1000       # kW
        pf_mean = pfr.get("mean_pf", 0)
        import math
        target_pf = thresh.power_factor_limit
        # kVAR needed to correct from current PF to target PF
        kvar_needed = p_mean * (math.tan(math.acos(pf_mean)) - math.tan(math.acos(target_pf)))
        findings.append({
            "category":       "power_factor",
            "severity":       "warning",
            "title":          "Low power factor — capacitor correction needed",
            "finding":        (f"Mean PF {pf_mean:.3f} lagging (limit {target_pf:.2f}). "
                               f"Mean reactive power {q_mean:.1f} kVAR. "
                               f"Estimated correction needed: {kvar_needed:.0f} kVAR."),
            "cause":          ("Lagging power factor is caused by inductive reactive loads — "
                               "primarily motors, transformers, and inductive ballasts drawing "
                               "magnetizing current. VFDs with active front ends may improve "
                               "PF at the drive level but do not eliminate motor reactive demand."),
            "responsibility": "customer",
            "recommendation": (f"Install approximately {kvar_needed:.0f} kVAR of power factor "
                               f"correction capacitors to achieve PF ≥ {target_pf:.2f}. "
                               "Size capacitors in switched steps to avoid over-correction at "
                               "light load. Verify capacitor placement does not excite harmonic "
                               "resonance (consult IEEE 1036 for PF correction in harmonic "
                               "environments)."),
            "confidence":     "high",
            "evidence":       {"mean_pf":     round(pf_mean, 4),
                               "mean_q_kvar": round(q_mean, 1),
                               "kvar_needed": round(kvar_needed, 0)},
        })

    # ── Neutral health — open/high-resistance neutral ────────────────────────
    nh = report.get("neutral_health", {})
    if nh.get("available") and nh.get("severity") in ("warning", "critical"):
        sev         = nh["severity"]
        n_coinc     = nh.get("coincident_events", 0)
        leg_r       = nh.get("leg_correlation", 1.0)
        sum_std     = nh.get("sum_std_v", 0.0)
        vne_max     = nh.get("vne_max_v", 0.0)
        vne_avail   = nh.get("vne_available", False)

        # Build the finding text from whichever indicators triggered
        evidence_parts: List[str] = []
        if n_coinc >= 1:
            evidence_parts.append(
                f"{n_coinc} coincident opposing sag/swell event{'s' if n_coinc > 1 else ''}"
            )
        if vne_avail and vne_max > 0.5:
            evidence_parts.append(f"Vne max {vne_max:.1f} V")
        if leg_r < 0.5:
            evidence_parts.append(f"leg correlation r = {leg_r:.3f}")
        if sum_std > 2.0:
            evidence_parts.append(f"L1+L2 sum std = {sum_std:.1f} V")

        findings.append({
            "category":       "voltage",
            "severity":       sev,
            "title":          "Open or high-resistance neutral suspected",
            "finding":        (
                "Split-phase neutral health indicators point to a compromised neutral: "
                + "; ".join(evidence_parts) + ". "
                "Voltage is redistributing between legs through the neutral impedance."
            ),
            "cause":          (
                "An open or high-resistance neutral causes the two 120 V legs to float relative "
                "to each other. Heavily loaded legs pull voltage below 120 V while the lightly "
                "loaded leg rises — voltage redistribution proportional to the load imbalance. "
                "Common causes: loose neutral wire at meter socket, corroded split-bolt connector, "
                "failed utility neutral splice, or broken neutral conductor."
            ),
            "responsibility": "utility",
            "recommendation": (
                "Inspect and tighten all neutral connections from the meter socket through the "
                "service entrance to the main panel. Check for corrosion at split-bolt connectors "
                "and wire nut splices. Measure neutral-to-ground voltage at the panel — > 1 V "
                "under load confirms neutral resistance. If the service neutral is overhead, "
                "inspect the drip loop and weatherhead connection. File a trouble call with "
                "Xcel Energy to inspect the utility secondary and meter socket neutral lug."
            ),
            "confidence":     "high" if n_coinc >= 2 or (vne_avail and vne_max > 2.0) else "medium",
            "evidence":       {
                "severity":          sev,
                "coincident_events": n_coinc,
                "leg_correlation":   leg_r,
                "sum_std_v":         sum_std,
                "vne_max_v":         vne_max if vne_avail else None,
                "asym_mean_v":       nh.get("asym_mean_v"),
            },
        })

    # ── Transformer loading — harmonic derating concern ───────────────────────
    if "transformer" in dem and ih.get("available") and il_amps > 0:
        tx      = dem["transformer"]
        pct_tx  = tx.get("pct_nameplate", 0)
        # Prefer meter-measured K-factor (includes all H1-H51) over estimated value
        if "kfactor_meter" in df.columns:
            k_factor = float(df["kfactor_meter"].median())
            k_source = "meter"
        else:
            h_means  = _harmonic_means(df, _H519_ORDERS)
            k_num    = sum((h_means.get(h, 0) / il_amps)**2 * h**2 for h in _H519_ORDERS)
            k_denom  = sum((h_means.get(h, 0) / il_amps)**2 for h in _H519_ORDERS)
            k_factor = k_num / k_denom if k_denom > 0 else 1.0
            k_source = "estimated"
        if pct_tx > 70 and k_factor > 4:
            findings.append({
                "category":       "demand",
                "severity":       "warning",
                "title":          "Transformer derating — harmonic K-factor",
                "finding":        (f"Transformer loaded at {pct_tx:.0f}% of nameplate. "
                                   f"Harmonic load K-factor = {k_factor:.1f} ({k_source}). "
                                   "Standard distribution transformers are rated K=1."),
                "cause":          ("Harmonic currents cause additional eddy-current losses in "
                                   "transformer windings beyond what the nameplate rating assumes. "
                                   f"A K-factor of {k_factor:.1f} means harmonic-related heating "
                                   "is significantly greater than for a sinusoidal load at "
                                   "the same kVA, increasing winding temperature and accelerating "
                                   "insulation degradation."),
                "responsibility": "customer",
                "recommendation": (f"Derate the transformer or replace with a K-{int(k_factor)+1} "
                                   "or higher rated unit. Alternatively, reduce harmonic content "
                                   "(AC line reactors on VFD inputs, or a passive harmonic filter) "
                                   "to lower the effective K-factor before the next capacity "
                                   "addition."),
                "confidence":     "high" if k_source == "meter" else "medium",
                "evidence":       {"pct_nameplate":   round(pct_tx, 1),
                                   "k_factor":        round(k_factor, 1),
                                   "k_source":        k_source},
            })

    return findings

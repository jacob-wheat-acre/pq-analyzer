from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from pq_constants import (
    measured as _m,
    measured_pct as _mp,
    ansi_bands,
    ansi_band_basis,
    VOLTAGE_BAND_LABEL,
    VOLTAGE_BAND_ORDER,
    SEVERITY_LABEL,
    SEVERITY_ORDER,
    SEVERITY_SEVERE_MARGIN,
    SEVERITY_SEVERE_MARGIN_ALONE,
    SEVERITY_SEVERE_PERSISTENCE,
    SEVERITY_SIGNIFICANT_MARGIN,
    SEVERITY_SIGNIFICANT_PERSISTENCE,
    LOAD_FAMILY_LABEL,
    LOAD_FAMILY_RECOMMENDATION,
    SEVERITY_WATCH_FRACTION,
    SEVERITY_WATCH_FRACTION_FLOOR,
    is_single_phase_208,
    ll_factor,
    SIGNATURE_ABSOLUTE_FLOOR,
    SIGNATURE_FAMILY_SEPARATION,
    SIGNATURE_MEMBER_SEPARATION,
    Thresholds,
    _DER_SHARE_FOR_1547,
    _H519_ORDERS,
    _H1547_ORDERS,
    _LOAD_SIGNATURES,
    _SERVICE_TYPE_LABEL,
    _TRD_LIMIT,
    _h519_limit,
    _h1547_limit,
    ride_through_region,
    frequency_ride_through_region,
    FREQ_ACTIVE_POWER_CAPABILITY,
    FREQ_CONTINUOUS_MAX_V_OVER_F,
    FREQ_CUMULATIVE_ALLOWANCE_S,
    FREQ_CUMULATIVE_WINDOW_S,
    _impedance_range,
    expected_service_impedance,
    _itic_lower_v,
    _itic_upper_v,
    _lookup_isc,
    _tdd_class,
    _tdd_limit,
)
from pq_adapter import PQDataset

#: Matches a per-order harmonic column (h3_current_a, h13_voltage_b, …) and
#: nothing else.  Aggregate channels like hrms_current_a must not be mistaken
#: for a single order.
_HARMONIC_COL = re.compile(r"h\d+_(current|voltage)_")

#: Voltage THD is only meaningful while the fundamental is intact.  Below this
#: fraction of nominal the measurement is a sag, an interruption or a meter
#: dropout, and V_h/V_1 reports distortion that is not physically there.  0.5 pu
#: is deliberately permissive: IEEE 1159 calls anything under 0.9 pu a sag, but
#: real distortion during a shallow sag is still worth seeing, whereas nothing
#: below half nominal is a valid steady-state THD reading.
_VTHD_VALID_PU = 0.5

#: A maximum this many times the 95th percentile is a spike rather than a
#: condition, and the report says so instead of leading with the number.
_VTHD_OUTLIER_RATIO = 3.0

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def grade_finding(
    passes: Optional[bool],
    measured: Optional[float] = None,
    limit: Optional[float] = None,
    persistence_pct: Optional[float] = None,
    confidence_notes: Optional[List[str]] = None,
    lower_is_worse: bool = False,
) -> dict:
    """Grade one compliance finding on severity, separately from pass/fail.

    Compliance answers "was this inside the limit"; severity answers "how much
    should anyone care".  Collapsing both into one red FAIL makes an isolated
    artifact look like a sustained overload, which is what makes a report
    alarming out of proportion to what it found.

    Severity comes from two measured quantities -- how far past the limit
    (margin) and how much of the recording was past it (persistence) -- and is
    then downgraded one band when the underlying number is known to be less
    trustworthy.  The downgrade reason is returned so the report can print it:
    a discount a reader cannot see is indistinguishable from hand-waving.

    ``lower_is_worse`` covers metrics like power factor, where falling below the
    limit is the failure and the margin therefore inverts.
    """
    notes = [n for n in (confidence_notes or []) if n]

    if passes is None:
        return {"band": "not_assessed", "label": SEVERITY_LABEL["not_assessed"],
                "reason": "", "margin": None, "downgraded": False}

    # ── Margin: how far past the limit, as a ratio ────────────────────────────
    margin: Optional[float] = None
    if measured is not None and limit is not None and limit > 0 and measured > 0:
        margin = (limit / measured) if lower_is_worse else (measured / limit)

    if passes:
        # Inside the limit.  Only distinguish "comfortably" from "close to it".
        band = "compliant"
        reason = ""
        watch_at = (SEVERITY_WATCH_FRACTION_FLOOR if lower_is_worse
                    else SEVERITY_WATCH_FRACTION)
        if margin is not None and margin >= watch_at:
            band = "watch"
            reason = (f"within the limit but only {_m((1 / margin - 1) * 100, '.0f', '%')} "
                      "above it" if lower_is_worse
                      else f"within the limit but at {_m(margin * 100, '.0f', '%')} of it")
        # A pass built on soft data is still worth qualifying -- a one-day
        # recording that meets a weekly statistical test has met less than the
        # test asks for, and the reader should be told without being alarmed.
        if notes:
            reason = (reason + "; " if reason else "") + "; ".join(notes)
        return {"band": band, "label": SEVERITY_LABEL[band], "reason": reason,
                "margin": margin, "downgraded": False}

    # ── Outside the limit: grade it ───────────────────────────────────────────
    p = persistence_pct if persistence_pct is not None else 0.0
    m = margin if margin is not None else 1.0

    if m >= SEVERITY_SEVERE_MARGIN_ALONE or (
            m >= SEVERITY_SEVERE_MARGIN and p >= SEVERITY_SEVERE_PERSISTENCE):
        band = "severe"
    elif m >= SEVERITY_SIGNIFICANT_MARGIN or p >= SEVERITY_SIGNIFICANT_PERSISTENCE:
        band = "significant"
    else:
        band = "minor"

    # ── Confidence downgrade ──────────────────────────────────────────────────
    downgraded = False
    if notes and band != "minor":
        band = SEVERITY_ORDER[SEVERITY_ORDER.index(band) - 1]
        downgraded = True

    bits = []
    if margin is not None:
        # "1.02x the limit" reads backwards for a floor like power factor, where
        # the failure is falling short of it rather than rising above it.
        bits.append(f"{_m((m - 1) * 100, '.0f', '%')} below the limit" if lower_is_worse
                    else f"{_m(m, '.2f', 'x')} the limit")
    if persistence_pct is not None:
        bits.append(f"{_mp(p, '.1f')} of the recording")
    reason = ", ".join(bits)
    if notes:
        reason = (reason + "; " if reason else "") + "; ".join(notes)
        if downgraded:
            reason += " (severity reduced one band)"

    return {"band": band, "label": SEVERITY_LABEL[band], "reason": reason,
            "margin": margin, "downgraded": downgraded}


def _require(df: pd.DataFrame, *cols: str) -> bool:
    """Return True if all cols exist in df and have at least one finite value."""
    for c in cols:
        if c not in df.columns or df[c].dropna().empty:
            return False
    return True


#: Service geometries this tool distinguishes.  The discriminator is the angle
#: between the legs, not how many there are, because that angle is what decides
#: whether the neutral sees a difference or a sum and whether triplen harmonics
#: cancel or accumulate:
#:
#:   "split-phase"  120/240 from a center tap.  Two legs 180 deg apart, so the
#:                  neutral carries |I1 - I2|.  For any odd harmonic h the leg
#:                  displacement is h*180 deg, which is 180 deg again, so odd
#:                  harmonics -- triplens included -- subtract in the neutral
#:                  exactly as the fundamental does.  There is no negative
#:                  sequence and no three-phase motor to derate.
#:   "two-leg-208"  Two legs of a 120/208 wye, 120 deg apart.  The neutral
#:                  carries a vector sum, and triplens are in phase (3*120 deg
#:                  = 360 deg) so they accumulate.  Only two of the three
#:                  phases are measured.
#:   "three-phase"  Full 3-phase wye.  NEMA MG1 unbalance and triplen
#:                  accumulation up to 3x both apply as written.
#:   "single-phase" One leg only; nothing to compare against.
SERVICE_GEOMETRIES = ("split-phase", "two-leg-208", "three-phase", "single-phase")


def service_geometry(thresh: Thresholds, columns) -> str:
    """Resolve which service geometry the checks should reason about.

    The engineer's picks win over channel presence, which cannot tell a
    120/240 service from two legs of a 120/208 one and cannot tell a genuinely
    single-phase service from a three-phase export that dropped a phase --
    see the note on `Thresholds.service_type`.  `check_neutral_health` resolved
    this inline before it was shared; this is the same precedence.
    """
    if is_single_phase_208(thresh.service_type):
        return "two-leg-208"
    if thresh.topology == "split-phase":
        return "split-phase"
    if thresh.topology == "3ph-wye":
        return "three-phase"
    cols = set(columns)
    if "current_c" in cols or "voltage_c" in cols:
        return "three-phase"
    if "current_b" in cols or "voltage_b" in cols:
        return "split-phase"
    return "single-phase"


def accumulates_triplens(geometry: str) -> bool:
    """True where triplen harmonics add in the neutral rather than subtract."""
    return geometry in ("two-leg-208", "three-phase")


def exports_power(thresh: Thresholds) -> bool:
    """True where the meter can see power flowing out of the premises."""
    return getattr(thresh, "service_role", "load") in ("mixed", "generation")


def is_generation_only(thresh: Thresholds) -> bool:
    """True at a plant with no load worth the name -- a producer's array."""
    return getattr(thresh, "service_role", "load") == "generation"


def applicable_current_standard(thresh: Thresholds) -> dict:
    """Which harmonic current standard governs this installation, per Figure 1.

    IEEE 519-2022 does not claim every service.  Clause 5.2 limits its own
    scope to "a user's PCC primarily with harmonic producing loads" and directs
    installations that are primarily inverter-based elsewhere; Figure 1 is the
    decision tree for the case in between:

        DER or IBR present?
          no  -> 519
          yes -> combined site rated generation < 10% of annual average
                 load demand?
                   yes -> 519
                   no  -> "a standard with an applicable scope such as
                           IEEE Std 1547 or IEEE Std 2800"

    Both quantities in that test are records quantities -- a nameplate and a
    year of billing -- so neither can be recovered from a recording, and the
    tree cannot be walked without them.  Rather than guess, an installation
    that declares generation but supplies neither is reported as undetermined
    and graded against 519 with that stated: the two standards differ by three
    times in the aggregate limit, which is not a difference to paper over.
    """
    out = {
        "standard":     "519",
        "branch":       "no_der",
        "der_share":    None,
        "determined":   True,
        "rated_ac_kw":  thresh.rated_ac_kw,
        "annual_avg_load_kw": thresh.annual_avg_load_kw,
    }
    if not exports_power(thresh):
        out["reason"] = ("The installation has no distributed energy resource, "
                         "so IEEE 519-2022 applies at the PCC.")
        return out

    rated = thresh.rated_ac_kw
    load  = thresh.annual_avg_load_kw
    if not rated or not load or load <= 0:
        missing = []
        if not rated:
            missing.append("the combined site rated generation")
        if not load or load <= 0:
            missing.append("the annual average load demand")
        out.update({
            "branch":     "undetermined",
            "determined": False,
            "reason": (
                "This installation has generation, so IEEE 519-2022 Figure 1 "
                "decides whether its limits apply here or whether IEEE 1547's "
                "do. That test compares the combined site rated generation "
                f"against the annual average load demand, and {' and '.join(missing)} "
                f"{'were' if len(missing) > 1 else 'was'} not supplied. "
                "The 519 limits are applied below, but they "
                "may not be the applicable ones: where generation reaches a "
                "tenth of average load demand the governing aggregate limit "
                "becomes 5.0% of the plant's rated current rather than a class "
                "limit read from ISC/IL."
            ),
        })
        return out

    share = rated / load
    out["der_share"] = share
    if share < _DER_SHARE_FOR_1547:
        out.update({
            "branch": "der_below_threshold",
            "reason": (
                f"The combined site rated generation ({rated:,.0f} kW) is "
                f"{share * 100:.1f}% of the annual average load demand "
                f"({load:,.0f} kW), below the 10% at which IEEE 519-2022 "
                "Figure 1 hands the installation to another standard, so the "
                "519 limits apply at the PCC."
            ),
        })
        return out

    out.update({
        "standard": "1547",
        "branch":   "der_at_or_above_threshold",
        "reason": (
            f"The combined site rated generation ({rated:,.0f} kW) is "
            f"{share * 100:.1f}% of the annual average load demand "
            f"({load:,.0f} kW). IEEE 519-2022 Figure 1 directs an installation "
            "at or above 10% to a standard with an applicable scope, so the "
            "current distortion limits below are IEEE 1547-2018 Clause 7.3 "
            "and are stated as a percentage of the plant's rated current, not "
            "of a maximum demand load current."
        ),
    })
    return out


def rated_output_amps(thresh: Thresholds, geometry: str) -> Optional[float]:
    """The plant's rated AC output as a current, or None if no rating was given.

    Inverters are rated in kW AC at unity power factor, so the current follows
    from the service geometry alone.  `nominal_voltage` is line-to-neutral
    throughout this module, which is why the three-phase divisor is 3·V_LN
    (that is sqrt(3)·V_LL) rather than sqrt(3)·V_LN.
    """
    kw = getattr(thresh, "rated_ac_kw", None)
    v_ln = thresh.nominal_voltage
    if not kw or kw <= 0 or not v_ln or v_ln <= 0:
        return None
    divisor = {
        "three-phase": 3.0 * v_ln,      # sqrt(3) x V_LL, and V_LL = sqrt(3) x V_LN
        "two-leg-208": np.sqrt(3.0) * v_ln,
        "split-phase": 2.0 * v_ln,      # rated across the 240 V leg pair
    }.get(geometry, v_ln)
    amps = kw * 1000.0 / divisor
    return amps if amps > 0 else None


#: Service classes that normally have the service transformer to themselves.
#: At PG the customer owns it outright; at SG a >50 kW service is built with a
#: dedicated pad.  Residential and small commercial share one transformer with
#: several neighbours, so a recording at one meter sees one contributor to its
#: load and not the load itself.
_DEDICATED_TRANSFORMER_CLASSES = frozenset({"sg", "pg"})


def has_dedicated_transformer(customer_class: Optional[str]) -> bool:
    """True when this service's demand is the whole of its transformer's load."""
    return (customer_class or "") in _DEDICATED_TRANSFORMER_CLASSES


#: Meter K-factor columns by phase label.  Phase A keeps the historical name.
KFACTOR_COLS = {"A": "kfactor_meter", "B": "kfactor_current_b",
                "C": "kfactor_current_c", "N": "kfactor_current_neutral"}

#: Flicker columns by phase label, for each IEC 61000-3-3 severity index.
FLICKER_COLS = {
    "pst": {"A": "flicker_pst", "B": "flicker_pst_b", "C": "flicker_pst_c"},
    "plt": {"A": "flicker_plt", "B": "flicker_plt_b", "C": "flicker_plt_c"},
}

#: Line-to-line voltage columns and the pair each represents.
LL_COLS = {"voltage_ab": "A-B", "voltage_bc": "B-C", "voltage_ca": "C-A"}

#: Standard ANSI C84.1 line-to-line nominal voltages, used to snap the inferred
#: nominal to a recognizable value for the report.
_STANDARD_LL_NOMINAL = [208, 240, 380, 400, 415, 480, 600, 2400, 4160, 4800,
                        12470, 13200, 13800, 22860, 24940, 34500]


#: Standard UL/IEEE C57.110 transformer K-ratings.  Nothing above K-50 is made.
STANDARD_K_RATINGS = (4, 9, 13, 20, 30, 40, 50)


def standard_k_rating(k_factor: float) -> Tuple[Optional[int], str]:
    """Map a measured K-factor onto a K-rating a transformer can actually be bought at.

    Returns (rating, wording).  Above K-50 there is no standard unit, and a
    K-factor that high in practice means the harmonic content was measured at
    very light load, where the index is dominated by the current resolution
    rather than by real harmonic heating -- 1.1 A of fundamental with harmonics
    rounded to 0.1 A can compute to K > 200.  Saying "a K-217 rated unit" would
    be a recommendation nobody can act on.
    """
    if k_factor <= 1.0:
        return 1, "K=1 rated (standard) transformer is adequate."
    for rating in STANDARD_K_RATINGS:
        if k_factor <= rating:
            severity = ("light" if rating <= 4 else
                        "moderate" if rating <= 9 else
                        "heavy" if rating <= 20 else "very high")
            return rating, (f"K-{rating} rated transformer recommended — "
                            f"{severity} harmonic load.")
    return None, (
        f"The measured K-factor of {k_factor:.0f} exceeds K-50, the highest "
        "standard rating available, which in practice means the harmonic content "
        "was measured at very light load where this index is not a meaningful "
        "sizing basis. Re-assess K-factor during a period of representative "
        "loading before specifying a transformer."
    )


def kfactor_by_phase(df: pd.DataFrame) -> dict:
    """Per-phase meter K-factor, and the phase that governs transformer rating.

    Harmonic heating is per-winding, so the K-rating a transformer needs is set
    by the worst phase -- reading phase A alone understated it by 2.1x on a real
    split-phase file (A=104, B=217). The neutral K-factor is reported but does
    not drive the rating: it describes neutral conductor heating rather than a
    transformer winding.
    """
    phases: Dict[str, dict] = {}
    for phase, col in KFACTOR_COLS.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        phases[phase] = {
            "column": col,
            "median": float(s.median()),
            "min":    float(s.min()),
            "max":    float(s.max()),
        }

    if not phases:
        return {"available": False, "phases": {}, "worst_phase": None,
                "note": "No meter K-factor channels available."}

    rating = {p: v for p, v in phases.items() if p != "N"} or phases
    worst = max(rating, key=lambda p: rating[p]["median"])
    return {
        "available":     True,
        "phases":        phases,
        "worst_phase":   worst,
        "median":        rating[worst]["median"],
        "min":           rating[worst]["min"],
        "max":           rating[worst]["max"],
        "phases_read":   sorted(phases),
        "phase_a_median": phases.get("A", {}).get("median"),
    }


def check_flicker(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """IEC 61000-3-3 flicker severity, on every phase the meter recorded.

    Previously only phase A was examined while the report asserted compliance
    for the service as a whole. On a real split-phase file phase B reached
    Pst 4.98 against phase A's 1.43, so the reported severity was 3.5x low, and
    a compliant phase A with a non-compliant phase B would have read as a pass.
    """
    out: dict = {"available": False, "pst": {}, "plt": {},
                 "pst_limit": _PST_LIMIT, "plt_limit": _PLT_LIMIT,
                 "worst_phase": None, "overall_pass": None}

    for kind, limit in (("pst", _PST_LIMIT), ("plt", _PLT_LIMIT)):
        for phase, col in FLICKER_COLS[kind].items():
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if s.empty:
                continue
            out[kind][phase] = {
                "column":       col,
                "max":          float(s.max()),
                "mean":         float(s.mean()),
                "median":       float(s.median()),
                "p95":          float(s.quantile(0.95)),
                "pct_exceeding": float((s > limit).mean() * 100),
                "pass":         bool(s.max() <= limit),
                # The meter carries one flicker value forward across several
                # intervals, so the number of readings is not the number of
                # measurement windows. Reported so nobody reads the interval
                # count as a sample size.
                "distinct_values": int(s.nunique()),
            }

    if not out["pst"] and not out["plt"]:
        out["note"] = "No flicker channels available."
        return out

    out["available"] = True
    # The governing phase is whichever comes closest to (or furthest past) its
    # own limit, so Pst and Plt are compared on equal footing.
    worst_ratio, worst_phase = -1.0, None
    for kind, limit in (("pst", _PST_LIMIT), ("plt", _PLT_LIMIT)):
        for phase, stats in out[kind].items():
            ratio = stats["max"] / limit if limit else 0.0
            if ratio > worst_ratio:
                worst_ratio, worst_phase = ratio, phase
    out["worst_phase"] = worst_phase
    out["worst_ratio_of_limit"] = round(worst_ratio, 3)
    out["pst_max"] = max((v["max"] for v in out["pst"].values()), default=None)
    out["plt_max"] = max((v["max"] for v in out["plt"].values()), default=None)
    out["pst_p95"] = max((v["p95"] for v in out["pst"].values()), default=None)
    out["plt_p95"] = max((v["p95"] for v in out["plt"].values()), default=None)
    # Which phase carries the worst of each, since a service-wide statement
    # that quotes phase A while phase B is over limit reads as a pass.
    for kind in ("pst", "plt"):
        if out[kind]:
            out[f"{kind}_worst_phase"] = max(
                out[kind], key=lambda p: out[kind][p]["max"])
    out["overall_pass"] = all(
        v["pass"] for kind in ("pst", "plt") for v in out[kind].values()
    )
    out["phases_read"] = sorted({p for kind in ("pst", "plt") for p in out[kind]})
    return out


def check_line_to_line_voltage(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """ANSI C84.1 compliance for line-to-line voltage.

    The line-to-neutral limits say nothing about line-to-line: the two differ by
    sqrt(3) on a wye service and by 2 on a split-phase one, and three-phase
    customer equipment is rated for the L-L value. The L-L nominal is inferred
    from the measured ratio to the L-N nominal rather than asking for another
    command-line argument, then snapped to a standard nominal when it is close.
    """
    ll_cols = [c for c in LL_COLS if c in df.columns and df[c].notna().any()]
    if not ll_cols:
        return {"available": False, "error": "No line-to-line voltage channels found.",
                "pairs": {}, "total_pct_out_of_bounds": None}

    ln_cols = [c for c in ("voltage_a", "voltage_b", "voltage_c")
               if c in df.columns and df[c].notna().any()]
    if not ln_cols:
        return {"available": False,
                "error": "Line-to-line channels present but no line-to-neutral "
                         "reference to infer the nominal from.",
                "pairs": {}, "total_pct_out_of_bounds": None}

    ln_median = float(np.nanmedian(df[ln_cols].to_numpy()))
    ll_median = float(np.nanmedian(df[ll_cols].to_numpy()))
    ratio = ll_median / ln_median if ln_median > 0 else 0.0

    if 1.60 <= ratio <= 1.85:
        factor, configuration = np.sqrt(3.0), "wye (L-L = √3 × L-N)"
    elif 1.90 <= ratio <= 2.10:
        factor, configuration = 2.0, "split-phase (L-L = 2 × L-N)"
    else:
        return {"available": False,
                "error": (f"Measured L-L/L-N ratio {ratio:.2f} matches neither a "
                          "wye (1.73) nor a split-phase (2.00) service; cannot "
                          "infer the line-to-line nominal."),
                "pairs": {}, "total_pct_out_of_bounds": None}

    # An entered primary nominal wins outright. Inference recovers the topology
    # -- whether the legs are 120 or 180 degrees apart -- but not the nominal
    # itself, and PSCo runs several primary voltages that no part of the file
    # names. Snapping a 13.2 kV service to the nearest table entry is a guess
    # dressed as a limit, so where the engineer has stated the voltage it is
    # used as stated and the source is recorded for the report.
    if thresh.primary_ll_voltage:
        nominal = float(thresh.primary_ll_voltage)
        nominal_source = "entered"
    else:
        nominal = thresh.nominal_voltage * factor
        snapped = min(_STANDARD_LL_NOMINAL, key=lambda v: abs(v - nominal))
        if abs(snapped - nominal) / nominal <= 0.02:
            nominal = float(snapped)
        nominal_source = "inferred"

    bands = ansi_bands(nominal)
    if not bands["range_a_evaluated"]:
        return {"available": False, "error": bands["range_b_note"],
                "pairs": {}, "total_pct_out_of_bounds": None}
    vmin, vmax = bands["a_min"], bands["a_max"]
    result: dict = {
        "available":     True,
        "error":         None,
        "nominal_v":     nominal,
        "nominal_source": nominal_source,
        "range_v":       (vmin, vmax),
        "range_b_v":     ((bands["b_min"], bands["b_max"])
                          if bands["range_b_evaluated"] else None),
        "range_b_evaluated": bands["range_b_evaluated"],
        "range_b_note":  bands["range_b_note"],
        "band_basis":    ansi_band_basis(bands),
        "nominal_group": bands["group"],
        "basis":         "interval average",
        "configuration": configuration,
        "ln_ll_ratio":   round(ratio, 3),
        "pairs":         {},
        "violation_timestamps": pd.DatetimeIndex([]),
    }

    all_violations = pd.Series(False, index=df.index)
    for col in ll_cols:
        s = df[col].dropna()
        # Reported, not judged on -- see check_voltage_compliance.
        smin = df[f"{col}_min"].reindex(s.index).fillna(s)  if f"{col}_min"  in df.columns else s
        smax = df[f"{col}_peak"].reindex(s.index).fillna(s) if f"{col}_peak" in df.columns else s
        under, over = s < vmin, s > vmax
        viol = under | over
        if bands["range_b_evaluated"]:
            out_b = (s < bands["b_min"]) | (s > bands["b_max"])
        else:
            out_b = pd.Series(False, index=s.index)
        in_b = viol & ~out_b

        if out_b.any():
            band = "outside_b"
        elif not viol.any():
            band = "range_a"
        else:
            band = "range_b" if bands["range_b_evaluated"] else "outside_a"

        all_violations.loc[viol.index[viol]] = True
        result["pairs"][LL_COLS[col]] = {
            "column":            col,
            "pct_out_of_bounds": float(viol.mean() * 100),
            "pct_under":         float(under.mean() * 100),
            "pct_over":          float(over.mean() * 100),
            "pct_range_b":       float(in_b.mean() * 100),
            "pct_outside_b":     float(out_b.mean() * 100),
            "band":              band,
            "min_v":             float(s.min()),
            "max_v":             float(s.max()),
            "mean_v":            float(s.mean()),
            "min_interval_v":    float(smin.min()),
            "max_interval_v":    float(smax.max()),
            "used_interval_extremes": smin is not s,
        }

    result["violation_timestamps"] = df.index[all_violations]
    result["total_pct_out_of_bounds"] = float(all_violations.mean() * 100)
    result["overall_pass"] = result["total_pct_out_of_bounds"] == 0
    result["band"] = max((p["band"] for p in result["pairs"].values()),
                         key=VOLTAGE_BAND_ORDER.index)
    result["total_pct_range_b"] = max(
        p["pct_range_b"] for p in result["pairs"].values())
    result["total_pct_outside_b"] = max(
        p["pct_outside_b"] for p in result["pairs"].values())
    return result


def check_frequency(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """System frequency deviation.

    ANSI C84.1 sets no frequency limit, so the default band is a practical one:
    +/-0.5 Hz, comparable to EN 50160's +/-1% for interconnected systems. On a
    healthy interconnection deviations are far smaller than this, so an
    exceedance points at an islanded service or a measurement problem rather
    than at ordinary grid regulation.
    """
    if "frequency" not in df.columns or df["frequency"].notna().sum() == 0:
        return {"available": False, "error": "No frequency channel available."}

    s = df["frequency"].dropna()
    nominal = thresh.frequency_nominal
    tolerance = thresh.frequency_tolerance_hz
    low, high = nominal - tolerance, nominal + tolerance
    out_of_band = (s < low) | (s > high)
    deviation = (s - nominal).abs()
    return {
        "available":          True,
        "error":              None,
        "nominal_hz":         nominal,
        "range_hz":           (low, high),
        "min_hz":             float(s.min()),
        "max_hz":             float(s.max()),
        "mean_hz":            float(s.mean()),
        "max_deviation_hz":   float(deviation.max()),
        "pct_out_of_band":    float(out_of_band.mean() * 100),
        "overall_pass":       bool(not out_of_band.any()),
        "violation_timestamps": s.index[out_of_band],
    }


#: Current-channel display resolution of the meter, in amps. Reported per-order
#: harmonic magnitudes are quantized to this.
CURRENT_RESOLUTION_A = 0.1

#: A spectrum is only classified when its dominant order stands this many
#: resolution steps clear of the quantization floor. At 5 steps the largest order
#: carries ~10% quantization error and the small orders far more, which is the
#: point where ratios like H5/H7 stop meaning anything.
_MIN_RESOLUTION_STEPS = 5

#: Lower bound on what counts as a loaded interval, in amps: ten resolution
#: steps. Nothing below this is meaningful load on any service.
_MIN_LOADED_AMPS = 1.0

#: Fewest loaded intervals worth drawing a spectral conclusion from.
_MIN_LOADED_INTERVALS = 20


#: Below this share of the largest |P| seen, an interval is too near the
#: crossover for its sign to say which way power was flowing. A service whose
#: generation is currently matching its load sits there, and filing those
#: intervals by the sign of a small residue would sort them by noise.
_FLOW_DEADBAND_FRACTION = 0.02


def primary_flow_direction(thresh: Thresholds) -> str:
    """The population a harmonic characterisation should describe by default.

    One answer used everywhere, so that the two direction methods and the
    spectrum gate cannot end up describing different halves of the same
    service and then be compared with each other.

    On a mixed service that is the importing intervals: the load is the half
    comparable with a service that does not generate, and the exporting half is
    reported separately on its own terms. On a plant there is only one
    population and it is the exporting one.
    """
    if is_generation_only(thresh):
        return "exporting"
    if exports_power(thresh):
        return "importing"
    return "all"


def flow_scope(df: pd.DataFrame, thresh: Thresholds, direction: str
               ) -> Tuple[Optional[pd.DataFrame], dict]:
    """The frame restricted to intervals flowing one way, and how that went.

    A service with generation behind it holds two populations that share a
    meter, and almost every harmonic statistic silently pools them. Grading a
    load's spectrum over intervals when its inverters were carrying the service
    describes the inverter; comparing a method that saw one population against
    a method that saw the other is not corroboration.

    `direction` is "importing", "exporting", or "all". Intervals within a
    deadband of zero are in neither: near the crossover the sign of real power
    is a residue, not a direction.

    Returns (frame or None, info). A None frame means the split could not be
    made -- no real-power channel, or too few intervals on that side -- and the
    caller should say so rather than quietly using everything.
    """
    info = {"direction": direction, "split": False, "intervals": len(df),
            "reason": ""}
    if direction == "all" or not exports_power(thresh):
        info["reason"] = "No generation at this service; all intervals used."
        return df, info

    if "power_real" not in df.columns or df["power_real"].notna().sum() == 0:
        info["reason"] = (
            "This service has generation, so its importing and exporting "
            "intervals describe different things, but the meter recorded no "
            "real-power channel to tell them apart. Every interval is pooled "
            "below, which mixes the two."
        )
        return df, info

    p = df["power_real"]
    scale = float(p.abs().max())
    deadband = scale * _FLOW_DEADBAND_FRACTION
    mask = (p > deadband) if direction == "importing" else (p < -deadband)
    kept = int(mask.sum())
    info.update({
        "split": True,
        "intervals": kept,
        "deadband_w": round(deadband, 1),
        "near_crossover": int((p.abs() <= deadband).sum()),
    })
    if kept < _MIN_LOADED_INTERVALS:
        info["reason"] = (
            f"Only {kept} interval(s) were {direction} by more than the "
            f"crossover deadband, too few to characterise separately."
        )
        return None, info
    info["reason"] = (
        f"{kept} of {len(df)} intervals were {direction}; the rest are "
        f"excluded here so the two directions are not pooled."
    )
    return df[mask], info


def harmonic_spectrum_significance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Whether the current harmonic spectrum carries usable shape information.

    Harmonic *shape* -- the ratios between orders that identify a load type or
    a resonance -- is only meaningful when the orders are well clear of the
    meter's 0.1 A reporting resolution and the service is actually loaded. On a
    1.1 A residential service with 0.2 A of third harmonic, each order carries
    25-50% quantization error, every ratio built from them is noise, and a
    classifier will nonetheless return a confident answer: this is what had a
    house reported as an electric arc furnace at 94% similarity.

    IEEE 519-2022 also specifies harmonic evaluation at maximum demand
    conditions, so light intervals are excluded from the spectrum rather than
    averaged into it.

    Returns a dict with ``usable``, a human ``reason``, the ``loaded`` boolean
    mask to compute the spectrum over, and the diagnostics behind the decision.
    """
    i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    out: dict = {"usable": False, "reason": "", "loaded": None,
                 "loaded_intervals": 0, "dominant_order_amps": 0.0,
                 "resolution_steps": 0.0, "load_verified": bool(i_cols)}

    # One direction of flow at a time. The floor below is a share of the peak,
    # so pooling both directions on a generating service lets the export peak
    # set a floor the load can never reach -- and the whole load half of the
    # service then drops out as "light load" without anything saying so.
    direction = primary_flow_direction(thresh)
    scoped, flow_info = flow_scope(df, thresh, direction)
    out["flow"] = flow_info
    if scoped is None:
        out["reason"] = flow_info["reason"]
        return out
    full_index, df = df.index, scoped

    # Restricting to loaded intervals is a refinement and needs an RMS current
    # channel. The resolution test below is the substantive one and stands on
    # its own, so a file without RMS current is still assessed rather than
    # declined outright.
    if i_cols:
        demand = df[i_cols].max(axis=1)
        il_amps = float(demand.max())
        floor = max(_MIN_LOADED_AMPS, il_amps * 0.10)
        loaded = demand >= floor
        # Callers index the original frame with this, so it goes back on the
        # full index: intervals dropped by the direction split are False here,
        # not missing.
        out["loaded"] = loaded.reindex(full_index, fill_value=False)
        out["loaded_intervals"] = int(loaded.sum())
        out["load_floor_amps"] = round(floor, 3)
        out["il_amps"] = round(il_amps, 2)

        if out["loaded_intervals"] < _MIN_LOADED_INTERVALS:
            out["reason"] = (
                f"Only {out['loaded_intervals']} interval(s) reached {floor:.1f} A "
                f"(10% of the {il_amps:.1f} A maximum demand); too little loaded "
                "data to characterize the harmonic spectrum."
            )
            return out
        scope = df[loaded]
    else:
        scope = df

    means = _harmonic_means(scope, (3, 5, 7, 9, 11, 13))
    dominant = max(means.values()) if means else 0.0
    steps = dominant / CURRENT_RESOLUTION_A if CURRENT_RESOLUTION_A else 0.0
    out["dominant_order_amps"] = round(dominant, 3)
    out["resolution_steps"] = round(steps, 1)

    if steps < _MIN_RESOLUTION_STEPS:
        out["reason"] = (
            f"The largest harmonic order averages {_m(dominant, '.2f', ' A')}, only "
            f"{_m(steps, '.1f')} times the meter's {CURRENT_RESOLUTION_A} A reporting "
            "resolution. The spectrum is dominated by quantization, so its shape "
            "cannot identify a load type or a resonance."
        )
        return out

    out["usable"] = True
    out["reason"] = (
        (f"{out['loaded_intervals']} loaded interval(s); " if i_cols
         else "load level not verified (no RMS current channel); ")
        + f"largest harmonic order {_m(dominant, '.2f', ' A')} "
        f"({_m(steps, '.0f', 'x')} the {CURRENT_RESOLUTION_A} A resolution)."
    )
    return out


def check_voltage_compliance(
    df: pd.DataFrame, thresh: Thresholds
) -> dict:
    """ANSI C84.1 voltage compliance check.

    The verdict is taken from the interval average, and each interval is placed
    in Range A, in Range B, or outside both.

    C84.1 rates *sustained* service voltage.  A dip lasting a few cycles inside
    a 30-second interval is a sag: IEEE 1159 grades it on depth and duration
    against the ITIC envelope, which this tool already does separately.  Judging
    Range A on the meter's within-interval extremes imported those events into
    the steady-state test, so a single 116 ms sag both failed C84.1 and was
    counted a second time as an ITIC violation -- one event, two findings, and
    the C84.1 one attributed to a standard that does not cover it.

    The extremes are still reported.  They are the most useful thing in the file
    for seeing that an event happened at all, and dropping them would lose that.
    They are returned under their own keys, as within-interval figures, and they
    do not decide compliance.
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

    bands = ansi_bands(thresh.nominal_voltage)
    if not bands["range_a_evaluated"]:
        # Above 34.5 kV Table 1 hands off to another standard. Reporting this as
        # unavailable, with the reason, is the only honest answer -- a pass
        # against a band the standard does not define is worse than no result.
        return {
            "available":              False,
            "error":                  bands["range_b_note"],
            "phases":                 {},
            "total_pct_out_of_bounds": None,
            "violation_timestamps":   pd.DatetimeIndex([]),
        }
    vmin, vmax = bands["a_min"], bands["a_max"]
    result = {
        "available":              True,
        "error":                  None,
        "nominal_v":              thresh.nominal_voltage,
        "range_v":                (vmin, vmax),
        "range_b_v":              ((bands["b_min"], bands["b_max"])
                                   if bands["range_b_evaluated"] else None),
        "range_b_evaluated":      bands["range_b_evaluated"],
        "range_b_note":           bands["range_b_note"],
        "band_basis":             ansi_band_basis(bands),
        "nominal_group":          bands["group"],
        "basis":                  "interval average",
        "phases":                 {},
        "phases_missing_data":    [],
        "violation_timestamps":   pd.DatetimeIndex([]),
    }

    all_violations = pd.Series(False, index=df.index)
    for col in v_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            # Column present but no usable samples -- don't silently fold this
            # phase into "no violations found" (it means "not measured", not
            # "compliant"). Track it so the report can say so explicitly.
            result["phases_missing_data"].append(col)
            continue
        # Within-interval extremes: reported, never the basis of the verdict.
        smin = df[f"{col}_min"].reindex(s.index).fillna(s)  if f"{col}_min"  in df.columns else s
        smax = df[f"{col}_peak"].reindex(s.index).fillna(s) if f"{col}_peak" in df.columns else s

        under = s < vmin
        over  = s > vmax
        viol  = under | over
        if bands["range_b_evaluated"]:
            under_b = s < bands["b_min"]
            over_b  = s > bands["b_max"]
        else:
            under_b = over_b = pd.Series(False, index=s.index)
        out_b = under_b | over_b
        # Range B is the band *between* the two, so an interval counts to it
        # only where it left Range A without leaving Range B as well.
        in_b = viol & ~out_b

        if out_b.any():
            band = "outside_b"
        elif not viol.any():
            band = "range_a"
        else:
            band = "range_b" if bands["range_b_evaluated"] else "outside_a"

        all_violations.loc[viol.index[viol]] = True
        result["phases"][col] = {
            "pct_out_of_bounds": float(viol.mean() * 100),
            "pct_under":         float(under.mean() * 100),
            "pct_over":          float(over.mean() * 100),
            "pct_range_b":       float(in_b.mean() * 100),
            "pct_outside_b":     float(out_b.mean() * 100),
            # Split the same way against the Range B edges, so a cell reporting
            # an outside-B share can break it down against the band it named
            # rather than against Range A's.
            "pct_under_b":       float(under_b.mean() * 100),
            "pct_over_b":        float(over_b.mean() * 100),
            "band":              band,
            "min_v":             float(s.min()),
            "max_v":             float(s.max()),
            "mean_v":            float(s.mean()),
            # The meter's within-interval extremes, for context only.
            "min_interval_v":    float(smin.min()),
            "max_interval_v":    float(smax.max()),
            "used_interval_extremes": smin is not s,
        }

    if not result["phases"]:
        # Every phase came back empty -- there's nothing to report a pass/fail on.
        # available=False (not True) so pass_fail correctly resolves to None/N/A
        # instead of a comparison against None silently reading as "failed".
        result["available"] = False
        result["error"] = "Voltage channels present but no phase had usable samples."
        result["total_pct_out_of_bounds"] = None
        return result

    result["violation_timestamps"] = df.index[all_violations]
    result["total_pct_out_of_bounds"] = float(all_violations.mean() * 100)
    # The service is graded on its worst phase: a supply that holds Range A on
    # two legs and leaves it on the third has still left it.
    result["band"] = max((p["band"] for p in result["phases"].values()),
                         key=VOLTAGE_BAND_ORDER.index)
    result["total_pct_range_b"] = max(
        p["pct_range_b"] for p in result["phases"].values())
    result["total_pct_outside_b"] = max(
        p["pct_outside_b"] for p in result["phases"].values())
    return result


#: Meter-aggregate harmonic RMS columns, by phase suffix.
HRMS_CURRENT_COLS = {"a": "hrms_current_a", "b": "hrms_current_b",
                     "c": "hrms_current_c", "neutral": "hrms_current_neutral"}


def harmonic_current_rms(
    df: pd.DataFrame, phase: str
) -> Tuple[Optional[pd.Series], str]:
    """Harmonic RMS current for one phase, in amps, with its provenance.

    Prefers the meter's own aggregate (hrms_current_*).  It is computed inside
    the instrument at full precision, whereas summing the reported per-order
    magnitudes loses whatever the display resolution rounded away: on a real
    light-load file the per-order sum gives 0.245 A against the meter's 0.300 A,
    a 22% understatement that propagates directly into TDD.  The gap closes at
    heavier load, where each order is large relative to the 0.1 A resolution.

    Returns (series, source) where source is 'meter', 'per-order sum', or ''.
    """
    col = HRMS_CURRENT_COLS.get(phase)
    if col and col in df.columns:
        s = df[col].dropna()
        if not s.empty:
            return s, "meter"

    orders = [c for c in df.columns
              if _HARMONIC_COL.match(c) and c.endswith(f"_current_{phase}")]
    if orders:
        squared = df[orders].pow(2).sum(axis=1, min_count=1).dropna()
        if not squared.empty:
            return np.sqrt(squared), "per-order sum"
    return None, ""


def fundamental_current(
    df: pd.DataFrame, phase: str, harm_rms: Optional[pd.Series]
) -> Optional[pd.Series]:
    """Fundamental current for one phase, from the true RMS and the harmonic RMS.

    I1 = sqrt(Irms² − Ih²).  IEEE 519-2022 defines both THD and the demand
    current IL against the fundamental, so the RMS channel on its own is not
    the right denominator -- it is larger by sqrt(1 + THD²).
    """
    col = f"current_{phase}"
    if col not in df.columns:
        return None
    rms = df[col].dropna()
    if rms.empty:
        return None
    if harm_rms is None:
        return rms
    aligned = pd.concat([rms.rename("rms"), harm_rms.rename("h")],
                        axis=1, join="inner").dropna()
    if aligned.empty:
        return rms
    # Noise can push the reported harmonic RMS above the total at very light
    # load; clamp rather than take the root of a negative number.
    squared = (aligned["rms"] ** 2 - aligned["h"] ** 2).clip(lower=0.0)
    return np.sqrt(squared)


def demand_current_il(
    df: pd.DataFrame,
    thresh: Optional[Thresholds] = None,
) -> Tuple[Optional[float], Dict[str, pd.Series], Dict[str, pd.Series],
           Dict[str, str]]:
    """IL, and the per-phase harmonic and fundamental series behind it.

    IEEE 519-2022 defines IL as the maximum demand current at the fundamental,
    so it is derived per phase from the RMS and the harmonic RMS rather than
    read straight off the RMS channel, which is larger by sqrt(1 + THD²).

    Every check that normalises against IL has to use this same one.  Grading
    TDD against a fundamental IL while the per-order table divides by an RMS IL
    puts two different denominators in one report and invites exactly the
    comparison a reader cannot make.

    At a generation-only site there is no demand load to take IL from, so an
    entered AC rating is used when there is one.  Without it IL falls back to
    the largest export actually measured, which grades the plant against the
    week it happened to have rather than against what it can do -- a cloudy
    recording then shrinks the denominator and inflates every percentage taken
    against it.  Which of the two was used is reported alongside the number.
    """
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    harm_rms: Dict[str, pd.Series] = {}
    harm_source: Dict[str, str] = {}
    fundamental: Dict[str, pd.Series] = {}
    for ph in (c[-1] for c in i_cols):
        h, src = harmonic_current_rms(df, ph)
        if h is not None:
            harm_rms[ph], harm_source[ph] = h, src
        f1 = fundamental_current(df, ph, h)
        if f1 is not None and not f1.empty:
            fundamental[ph] = f1

    # Order of preference, strongest reference first. Billing IL is the only
    # one that is the standard's own quantity rather than a stand-in for it.
    if thresh is not None and thresh.il_amps_billing:
        return float(thresh.il_amps_billing), harm_rms, fundamental, harm_source

    rated = (rated_output_amps(thresh, service_geometry(thresh, df.columns))
             if thresh is not None and is_generation_only(thresh) else None)
    if rated:
        return rated, harm_rms, fundamental, harm_source

    if fundamental:
        il_amps = float(max(f1.max() for f1 in fundamental.values()))
    elif i_cols:
        il_amps = float(df[i_cols].max(axis=1).max())
    else:
        return None, harm_rms, fundamental, harm_source
    return ((il_amps if il_amps > 0 else None),
            harm_rms, fundamental, harm_source)


def tdd_by_phase(
    df: pd.DataFrame,
    il_amps: float,
    harm_rms: Dict[str, pd.Series],
    fundamental: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """Per-phase TDD(t) = 100 × Ih(t) / IL, in percent.

    TDD divides by IL, a fixed number; THD divides by the fundamental at that
    instant.  On a service whose output falls to nothing -- a solar site at
    night -- the THD denominator approaches zero and the ratio runs away while
    the harmonic amperes behind it stay trivial.  That is why the standard
    grades current on TDD, and why a THD channel must never be handed to a TDD
    limit unconverted.
    """
    out: Dict[str, pd.Series] = {}
    if not il_amps or il_amps <= 0:
        return out
    for ph in ("a", "b", "c"):
        h = harm_rms.get(ph)
        if h is not None and not h.empty:
            out[ph] = h * 100.0 / il_amps
            continue
        # No harmonic RMS for this phase: recover the harmonic amperes from the
        # THD channel and the fundamental (Ih = THD% × I1 / 100) rather than
        # dropping the phase, which would silently shrink the assessment.
        col  = f"thd_current_{ph}"
        base = fundamental.get(ph)
        if col not in df.columns or base is None:
            continue
        aligned = pd.concat([df[col].rename("thd"), base.rename("i1")],
                            axis=1, join="inner").dropna()
        if not aligned.empty:
            out[ph] = aligned["thd"] * aligned["i1"] / il_amps
    return out


def check_thd(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """IEEE 519-2022 compliance: voltage THD and current TDD.

    Voltage: standard THD (relative to fundamental), limit from thresh.thd_voltage_limit.
    Current: TDD (relative to maximum demand current IL) whenever RMS current
      channels are present — ISC is not required to compute TDD itself:
      TDD(t) = 100 × Ih(t) / IL, where Ih is the harmonic RMS current (the
      meter's own aggregate where available, else the per-order sum) and IL is
      the maximum demand *fundamental* current in the recording.
      The TDD class limit is selected from IEEE 519-2022 Table 2 via the ISC/IL
      ratio when thresh.isc_amps is set; without ISC the most restrictive class
      (ISC/IL < 20, 5.0%) is assumed — conservative, since the true limit can
      only be equal or higher.
    Falls back to plain THD vs thresh.thd_current_limit only when no RMS current
    channels exist to derive IL from.
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
    # IEEE 519-2022 defines IL as the maximum demand load current at the
    # fundamental frequency, so it is derived per phase from the RMS and the
    # harmonic RMS rather than read straight off the RMS channel.
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    il_amps, harm_rms, fundamental, harm_source = demand_current_il(df, thresh)

    # Where IL came from. At a generation site the two answers differ by
    # however cloudy the week was, so the page has to say which it used rather
    # than printing a bare denominator.
    if thresh.il_amps_billing:
        il_basis = "billing"
    elif is_generation_only(thresh):
        il_basis = ("rated_output"
                    if rated_output_amps(thresh, service_geometry(thresh, df.columns))
                    else "measured_export")
    else:
        il_basis = "measured_demand"

    if il_amps and il_amps > 0:
        use_tdd = True
        if thresh.isc_amps is not None:
            isc_il        = thresh.isc_amps / il_amps
            current_limit = _tdd_limit(isc_il)
            result["tdd_info"] = {
                "isc_amps":      thresh.isc_amps,
                "il_amps":       round(il_amps, 1),
                "isc_il_ratio":  round(isc_il, 1),
                "tdd_class":     _tdd_class(isc_il),
                "tdd_limit_pct": current_limit,
                "isc_source":    thresh.isc_source,
                "isc_provided":  True,
                "il_basis":      il_basis,
                "rated_ac_kw":   thresh.rated_ac_kw,
            }
            log.info(
                "IEEE 519 TDD: ISC=%.0f A  IL=%.0f A  ISC/IL=%.1f  class %s  limit=%.1f%%",
                thresh.isc_amps, il_amps, isc_il, _tdd_class(isc_il), current_limit,
            )
        else:
            # TDD needs only IL; ISC is needed only to select the Table 2 limit
            # class. Assume the most restrictive class (ISC/IL < 20) — the true
            # limit can only be equal or higher.
            isc_il        = None
            current_limit = _tdd_limit(0.0)
            result["tdd_info"] = {
                "isc_amps":      None,
                "il_amps":       round(il_amps, 1),
                "isc_il_ratio":  None,
                "tdd_class":     "< 20 (assumed)",
                "tdd_limit_pct": current_limit,
                "isc_source":    None,
                "isc_provided":  False,
                "il_basis":      il_basis,
                "rated_ac_kw":   thresh.rated_ac_kw,
            }
            log.info(
                "IEEE 519 TDD: ISC not provided — IL=%.0f A, most restrictive class "
                "assumed (ISC/IL < 20, limit %.1f%%). Pass --isc for the true class limit.",
                il_amps, current_limit,
            )
    else:
        isc_il        = None
        current_limit = thresh.thd_current_limit
        use_tdd       = False

    # ── Voltage THD ───────────────────────────────────────────────────────────
    v_thd_cols = [c for c in ["thd_voltage_a", "thd_voltage_b", "thd_voltage_c"]
                  if c in df.columns]
    if v_thd_cols:
        worst = df[v_thd_cols].max(axis=1).dropna()
        if worst.empty:
            result["voltage"] = {
                "available": False,
                "error": "Voltage THD channel(s) present but no usable samples.",
            }
        else:
            # Voltage THD is V_h/V_1 — the same ratio structure that makes
            # current THD explode at light load.  When the fundamental collapses
            # (a sag, a momentary interruption, a meter dropout at the edges of
            # the recording) the denominator approaches zero and THD runs to
            # tens of percent.  Those samples are measurement artifacts, not
            # distortion, and IEEE 519-2022 evaluates against normal operating
            # conditions.  Drop them before judging anything.
            raw_n = len(worst)
            v_cols = [c for c in ["voltage_a", "voltage_b", "voltage_c"]
                      if c in df.columns]
            sag_floor = thresh.nominal_voltage * _VTHD_VALID_PU
            artifact_samples = 0
            if v_cols:
                v_min = df[v_cols].min(axis=1).reindex(worst.index)
                valid = v_min.notna() & (v_min >= sag_floor)
                if valid.any():
                    artifact_samples = int((~valid).sum())
                    worst = worst[valid]

            exceed = worst > thresh.thd_voltage_limit
            # IEEE 519-2022 Clause 5 judges voltage THD on the 95th percentile of
            # short-time values, not on every sample.  A single artifact sample
            # must not fail a site, so the percentile is the verdict and the
            # maximum is reported as context.
            p95 = float(worst.quantile(0.95))
            p99 = float(worst.quantile(0.99))
            v_max = float(worst.max())
            result["voltage"] = {
                "available":        True,
                "limit_pct":        thresh.thd_voltage_limit,
                "max_thd_pct":      v_max,
                "mean_thd_pct":     float(worst.mean()),
                "p95_thd_pct":      p95,
                "p99_thd_pct":      p99,
                "p95_limit_pct":    thresh.thd_voltage_limit,
                "p99_limit_pct":    thresh.thd_voltage_limit * 1.5,
                "p95_pass":         p95 <= thresh.thd_voltage_limit,
                "p99_pass":         p99 <= thresh.thd_voltage_limit * 1.5,
                "pct_exceeding":    float(exceed.mean() * 100),
                "sample_count":     len(worst),
                "artifact_samples": artifact_samples,
                "artifact_floor_v": round(sag_floor, 1),
                # A maximum far above the 95th percentile is a spike, not a
                # condition; the report says so rather than leading with it.
                "max_is_outlier":   bool(p95 > 0 and v_max > p95 * _VTHD_OUTLIER_RATIO
                                         and p95 <= thresh.thd_voltage_limit),
                "violation_timestamps": worst.index[exceed].tolist(),
            }
            result["available"] = True
            if artifact_samples:
                log.info(
                    "Voltage THD: %d of %d samples dropped — measured voltage "
                    "below %.1f V (%.0f%% of nominal), where the THD ratio is "
                    "not meaningful.",
                    artifact_samples, raw_n, sag_floor, _VTHD_VALID_PU * 100,
                )

    # ── Current TDD (or THD fallback) ─────────────────────────────────────────
    i_thd_cols = [c for c in ["thd_current_a", "thd_current_b", "thd_current_c"]
                  if c in df.columns]
    hrms_sources = sorted(set(harm_source.values()))
    if i_thd_cols or (harm_rms and use_tdd):
        if use_tdd and harm_rms:
            # TDD(t) = 100 × Ih(t) / IL, straight from the IEEE 519-2022
            # definition. The previous form, THD%(t) × Irms(t) / IL, was exact
            # only while the RMS current channel actually held the fundamental;
            # against a true RMS channel it overstates TDD by sqrt(1 + THD²).
            worst = pd.concat(
                [h * 100.0 / il_amps for h in harm_rms.values()], axis=1
            ).max(axis=1).dropna()
            metric = "tdd"
        elif use_tdd:
            # No harmonic RMS at all: fall back to the THD channel, but against
            # the derived fundamental rather than the RMS current.
            tdd_cols: List[pd.Series] = []
            for col in i_thd_cols:
                phase = col[-1]
                base = fundamental.get(phase)
                if base is None:
                    tdd_cols.append(df[col].dropna())
                    continue
                aligned = pd.concat([df[col].rename("thd"), base.rename("i1")],
                                    axis=1, join="inner").dropna()
                if len(aligned):
                    tdd_cols.append(aligned["thd"] * aligned["i1"] / il_amps)
                else:
                    tdd_cols.append(df[col].dropna())
            worst = pd.concat(tdd_cols, axis=1).max(axis=1).dropna()
            metric = "tdd"
        else:
            log.warning(
                "No RMS current channels — cannot derive IL for TDD. "
                "Evaluating raw THD against the %.1f%% fallback limit; "
                "light-load intervals may inflate THD.", current_limit,
            )
            worst  = df[i_thd_cols].max(axis=1).dropna()
            # Filter out light-load intervals: at < 10% of peak demand the THD%
            # denominator (I₁) approaches zero and produces meaningless large values.
            # IEEE 519-2022 §2.1 specifies evaluation at maximum demand conditions.
            light_load_filtered = False
            if il_amps and il_amps > 0 and i_cols:
                load_mask = df[i_cols].max(axis=1).reindex(worst.index).fillna(0) >= il_amps * 0.10
                if load_mask.any():
                    worst = worst[load_mask]
                    light_load_filtered = True
            metric = "thd"

        if worst.empty:
            result["current"] = {
                "available": False,
                "error": "Current THD/TDD channel(s) present but no usable samples.",
            }
        else:
            exceed = worst > current_limit
            result["current"] = {
                "available":              True,
                "metric":                 metric,
                "limit_pct":              current_limit,
                "max_thd_pct":            float(worst.max()),
                "mean_thd_pct":           float(worst.mean()),
                "pct_exceeding":          float(exceed.mean() * 100),
                "light_load_filtered":    light_load_filtered if not use_tdd else False,
                "harmonic_rms_source":    ", ".join(hrms_sources) or None,
                "il_amps":                round(il_amps, 2) if il_amps else None,
                "violation_timestamps":   worst.index[exceed].tolist(),
            }
            result["available"] = True

            # Peak TDD within each interval.  Prefer the harmonic RMS peak where
            # the meter recorded it; otherwise use the peak THD channel.
            pk_hrms = {ph: f"{HRMS_CURRENT_COLS[ph]}_peak" for ph in harm_rms
                       if f"{HRMS_CURRENT_COLS.get(ph, '')}_peak" in df.columns}
            if pk_hrms and use_tdd and il_amps:
                pk_worst = pd.concat(
                    [df[c].dropna() * 100.0 / il_amps for c in pk_hrms.values()],
                    axis=1,
                ).max(axis=1).dropna()
                if not pk_worst.empty:
                    pk_exceed = pk_worst > current_limit
                    result["current"]["peak_max_tdd_pct"] = round(float(pk_worst.max()), 2)
                    result["current"]["peak_pct_exceeding"] = round(
                        float(pk_exceed.mean() * 100), 2)
            pk_thd_cols = [f"{c}_peak" for c in i_thd_cols if f"{c}_peak" in df.columns]
            if (pk_thd_cols and use_tdd and il_amps
                    and "peak_max_tdd_pct" not in result["current"]):
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
                if not pk_worst.empty:
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


def check_trd(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """IEEE 1547-2018 Clause 7.3 current distortion, for a DER installation.

        %TRD = sqrt(I_rms² − I₁²) / I_rated × 100

    The numerator is the harmonic RMS current this module already derives; what
    makes TRD a different measurement from TDD is the denominator and the
    limits.  Irated is the plant's nameplate, so unlike IL it cannot be
    approximated from the recording at all -- without it there is no assessment
    to make, and the check declines rather than substituting a measured peak
    and calling the result 1547.

    Two caveats travel with every number here and are carried in the result so
    the report cannot state one without them:

      * 1547 sets these limits "exclusive of any harmonic currents due to
        harmonic voltage distortion present in the Area EPS without the DER
        connected".  Separating the plant's own injection from what the
        background voltage drives through it needs a measurement taken with
        the plant offline, which a single site visit does not have.
      * TRD includes interharmonics.  A meter reporting only integer orders
        understates it, so a pass close to the limit is not a comfortable one.
    """
    result: dict = {
        "available": False, "orders": {}, "trd_pct": None,
        "trd_limit_pct": _TRD_LIMIT, "overall_pass": True,
        "irated_amps": None, "worst_order": None, "worst_margin": None,
        "excludes_background": False, "interharmonics_measured": False,
    }

    geometry = service_geometry(thresh, df.columns)
    irated = rated_output_amps(thresh, geometry)
    if not irated:
        result["note"] = (
            "IEEE 1547 states its limits as a percentage of the plant's rated "
            "current, which is a nameplate quantity: no recording establishes "
            "it. Enter the combined site rated generation to assess against "
            "1547."
        )
        return result

    il_amps, harm_rms, fundamental, _src = demand_current_il(df, thresh)
    if not harm_rms and not fundamental:
        result["note"] = ("No current channels to measure harmonic distortion "
                          "from.")
        return result

    result["irated_amps"] = round(irated, 1)
    result["available"] = True

    # Aggregate TRD, on the worst phase: the limit applies per phase, so an
    # average across three would hide the one that fails.
    trd_by_phase = {ph: (h * 100.0 / irated) for ph, h in harm_rms.items()}
    if trd_by_phase:
        worst_phase = max(trd_by_phase, key=lambda p: float(trd_by_phase[p].max()))
        series = trd_by_phase[worst_phase].dropna()
        result["trd_pct"]   = round(float(series.max()), 2)
        result["trd_phase"] = worst_phase
        result["trd_mean_pct"] = round(float(series.mean()), 2)
        result["trd_pass"] = result["trd_pct"] <= _TRD_LIMIT
        result["pct_exceeding"] = round(float((series > _TRD_LIMIT).mean() * 100), 2)
        if not result["trd_pass"]:
            result["overall_pass"] = False

    # Per order, against Table 26 and Table 27.
    worst_margin = 0.0
    for h in _H1547_ORDERS:
        limit = _h1547_limit(h)
        if limit <= 0:
            continue
        pcts = []
        for ph in ("a", "b", "c"):
            col = f"h{h}_current_{ph}"
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    pcts.append(float(s.max()) * 100.0 / irated)
        if not pcts:
            continue
        worst = max(pcts)
        passes = worst <= limit
        result["orders"][h] = {
            "max_pct_irated": round(worst, 3),
            "limit_pct":      limit,
            "pass":           passes,
            "even":           h % 2 == 0,
        }
        if not passes:
            result["overall_pass"] = False
        ratio = worst / limit
        if ratio > worst_margin:
            worst_margin = ratio
            result["worst_order"]  = h
            result["worst_margin"] = round(ratio, 3)

    result["excludes_background"] = False
    result["interharmonics_measured"] = False
    result["caveats"] = [
        "IEEE 1547 excludes harmonic current caused by voltage distortion "
        "already present without the plant connected. That background was not "
        "measured with the plant offline, so the figures here include whatever "
        "part of the distortion the system drives through the inverters and "
        "are, to that extent, conservative against the plant.",
        "TRD includes interharmonics. This meter reports integer harmonic "
        "orders only, so the true TRD is at least the figure given and a "
        "narrow pass should not be read as clearance.",
    ]
    return result


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

    il_amps, _harm_rms, _fundamental, _harm_source = demand_current_il(df, thresh)
    if not il_amps:
        return result

    if thresh.isc_amps is None:
        result["note"] = "Pass --isc to enable per-order IEEE 519 check"
        return result

    # Per-order channels only: h3_current_a and friends. A loose startswith("h")
    # test also swallows aggregate channels such as hrms_current_a, which would
    # then be squared into the harmonic sum as if it were a single order.
    h_cols = [c for c in df.columns if _HARMONIC_COL.match(c) and "_current_" in c]
    if not h_cols:
        result["note"] = "Meter did not record individual harmonic orders (only THD totals available)"
        return result

    isc_il = thresh.isc_amps / il_amps
    result["available"] = True
    result["il_amps"] = round(il_amps, 1)
    result["isc_il_ratio"] = round(isc_il, 1)

    worst_pct = 0.0
    worst_order = None
    worst_margin = 0.0
    worst_margin_order = None
    worst_margin_pct = 0.0
    worst_margin_limit = None

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
            # Severity depends on the worst *margin*, not the worst magnitude:
            # the per-order limits fall steeply with h, so a small high-order
            # current can be further past its limit than a large H3.
            ratio = max_pct / limit_pct
            if ratio > worst_margin:
                worst_margin       = ratio
                worst_margin_order = (h, ph)
                worst_margin_pct   = max_pct
                worst_margin_limit = limit_pct

        result["phases"][ph] = ph_result

    result["worst_order"] = worst_order
    result["worst_pct_of_il"] = round(worst_pct, 2)
    result["worst_margin"] = round(worst_margin, 3) if worst_margin else None
    result["worst_margin_order"] = worst_margin_order
    result["worst_margin_pct_of_il"] = (round(worst_margin_pct, 2)
                                        if worst_margin_order else None)
    result["worst_limit_pct"] = worst_margin_limit if worst_margin_order else None
    return result


_NEUTRAL_HARMONIC_ORDERS = (3, 5, 7, 9, 11, 13)
_TRIPLEN_ORDERS           = frozenset({3, 9, 15})

_PST_LIMIT = 1.0    # IEC 61000-3-3 short-term flicker severity limit
_PLT_LIMIT = 0.65   # IEC 61000-3-3 long-term flicker severity limit


def check_neutral_harmonics(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Neutral harmonic analysis.

    Per-order neutral harmonic current (Amps, mean and max) is measured on any
    service that reports a neutral channel -- neutral heating is real whatever
    the geometry.

    The *accumulation* diagnostic is not.  Triplens add in the neutral because
    3 x 120 deg = 360 deg puts them in phase, which is a property of a 120 deg
    system.  On a 120/240 split-phase service the legs are 180 deg apart, and
    h x 180 deg is 180 deg again for every odd h, so triplens subtract in the
    neutral exactly as the fundamental does.  There is nothing to accumulate,
    and the factor's own scale (0 cancels / 1 one phase dominates / 3 full
    accumulation) is defined over three phases.  So on split-phase the
    accumulation factor and the zero-sequence reading are withheld rather than
    computed over two legs and interpreted against a three-phase scale.

    Where it does apply -- 120/208 two-leg and full three-phase --
      Accumulation factor: H3_neutral / mean(H3_a, H3_b, H3_c)
          ≈ 0     → H3 cancels (balanced 3-phase, near-zero neutral H3)
          ≈ 1     → one phase dominates
          ≈ 3     → equal H3 from all three phases accumulates fully
          > 3     → resonance amplification
    """
    avail = [c for c in df.columns
             if _HARMONIC_COL.match(c) and c.endswith("_current_neutral")]
    if not avail:
        return {"available": False, "note": "No neutral harmonic channels in dataset"}

    geometry = service_geometry(thresh, df.columns)
    triplens_accumulate = accumulates_triplens(geometry)

    result: dict = {
        "available": True,
        "geometry":             geometry,
        "triplens_accumulate":  triplens_accumulate,
        "accumulation_note":    None if triplens_accumulate else (
            "Triplen accumulation is not evaluated on a 120/240 split-phase "
            "service. The two legs are 180 degrees apart, so odd harmonics -- "
            "triplens included -- subtract in the neutral as the fundamental "
            "does, rather than adding as they would on a 120-degree system. "
            "The neutral harmonic currents below are still measured."),
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
    # "Dominant" drives the zero-sequence narrative downstream, so it only
    # means anything where triplens can accumulate in the first place.
    result["triplen_dominant"]      = (triplens_accumulate and total > 0
                                       and t_mean / total > 0.5)

    h3_n_col    = "h3_current_neutral"
    h3_ph_cols  = [f"h3_current_{p}" for p in "abc" if f"h3_current_{p}" in df.columns]
    if triplens_accumulate and h3_n_col in df.columns and h3_ph_cols:
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
      corr_h = correlation between the interval harmonic voltage and current
      Z_ratio = Z_h / Z_linear_h             where Z_linear_h = a × h fits through origin

    Resonance: Z_ratio > 2.5 at any order → parallel resonance suspect.
    Attribution heuristic (indicative — exact direction requires phasor data):
      corr > 0.50 → 'customer'  (V and I co-vary → load injection drives both)
      else        → 'indeterminate'
    """
    # Z_h and the ratios built from it are only interpretable when the harmonic
    # orders stand clear of the meter's reporting resolution. Below that the
    # impedance estimate is a ratio of two quantized near-zero numbers, which is
    # what produced a 13x "resonance" on a 1.1 A residential service.
    significance = harmonic_spectrum_significance(df, thresh)

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

    if not significance["usable"]:
        # Keep the measured impedances as data, but draw no conclusion from them.
        for od in orders_with_data.values():
            od["attribution"] = "not_assessed"
        return {
            "available":      True,
            "orders":         orders_with_data,
            "linear_slope_a": round(a_fit, 5) if a_fit is not None else None,
            "resonant_orders": [],
            "overall":         "not_assessed",
            "significance":    significance,
            "note": (
                "Source indication and resonance screening were not performed: "
                + significance["reason"]
                + " The impedances above are reported as measured data only."
            ),
        }

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
        "significance":    significance,
        "note": (
            "This indicates the direction distortion appears to come from, not responsibility for it. Based on the correlation between the harmonic voltage and harmonic current over the recording. "
            "Exact source direction requires waveform phasor measurements."
        ),
    }


def check_spectral_shape(df: pd.DataFrame, thresh: Thresholds, source_harm: dict) -> dict:
    """Single-visit classification: does the measured voltage harmonic spectrum look
    like broadband, multi-source injection (elevated and flat across many orders) or
    a narrowband condition (concentrated at one or two orders)?

    IMPORTANT: this classifies the spectral *shape* of this one recording. It is not
    a trend measurement and does not claim distortion is "rising" over time -- that
    would require comparing this result against repeat visits to the same site over
    months or years, which is out of scope here.

    Two conditions must both hold to classify as "broadband_consistent":
      1. Elevation: mean voltage THD is a meaningful fraction of the IEEE 519 limit
         (thresh.spectral_elevation_ratio), not just noise-level distortion.
      2. Flatness: the per-order voltage harmonic spectrum (H3, H5, H7, H11, H13) has
         a low relative variability (coefficient of variation) (thresh.spectral_flatness_cv) -- no single
         order dominates.

    Never returns "broadband_consistent" when check_harmonic_sources already flagged
    a resonance_suspect order at this site -- the two checks are complementary
    (narrowband vs. broadband explanations), not competing votes on the same finding.
    """
    significance = harmonic_spectrum_significance(df, thresh)
    if not significance["usable"]:
        return {
            "available": False,
            "error": "Spectral shape not classified: " + significance["reason"],
            "significance": significance,
        }

    _ORDERS = (3, 5, 7, 11, 13)
    v_cols_by_order = {
        h: [c for c in (f"h{h}_voltage_a", f"h{h}_voltage_b", f"h{h}_voltage_c") if c in df.columns]
        for h in _ORDERS
    }
    available_orders = [h for h in _ORDERS if v_cols_by_order[h]]
    if len(available_orders) < 3:
        return {
            "available": False,
            "error": "Need at least 3 harmonic voltage orders for spectral shape classification.",
        }

    order_means = {h: float(df[v_cols_by_order[h]].values.mean()) for h in available_orders}
    spectrum = np.array([order_means[h] for h in available_orders])
    if not np.isfinite(spectrum).all() or spectrum.mean() <= 0:
        return {"available": False, "error": "No measurable harmonic voltage content."}

    cv = float(np.std(spectrum) / np.mean(spectrum))

    thd_v_cols = [c for c in ("thd_voltage_a", "thd_voltage_b", "thd_voltage_c") if c in df.columns]
    if not thd_v_cols:
        return {"available": False, "error": "No thd_voltage channel available for elevation check."}
    mean_vthd_pct = float(df[thd_v_cols].values.mean())
    elevation_ratio = (
        round(mean_vthd_pct / thresh.thd_voltage_limit, 2) if thresh.thd_voltage_limit else None
    )

    is_elevated  = elevation_ratio is not None and elevation_ratio >= thresh.spectral_elevation_ratio
    is_flat      = cv < thresh.spectral_flatness_cv
    has_resonance = bool(source_harm.get("available") and source_harm.get("resonant_orders"))

    if has_resonance:
        classification = "resonance_present"
    elif is_elevated and is_flat:
        classification = "broadband_consistent"
    elif is_elevated and not is_flat:
        classification = "elevated_uneven"
    else:
        classification = "not_elevated"

    return {
        "available":        True,
        "mean_vthd_pct":    round(mean_vthd_pct, 2),
        "elevation_ratio":  elevation_ratio,
        "flatness_cv":      round(cv, 3),
        "order_means_v":    {h: round(order_means[h], 3) for h in available_orders},
        "classification":   classification,
        "note": (
            "Single-visit spectral-shape classification, not a measured trend. "
            "'broadband_consistent' means this recording's shape is consistent with many "
            "small distributed sources (DER, EV chargers, LED drivers, etc.) rather than "
            "one dominant order -- it is not evidence that distortion is increasing over time."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic source direction
#
# Two independent readings of the same question -- is the dominant harmonic
# source at the point of common coupling on the utility side or the customer
# side -- because the data supports two and they fail in different ways.
#
#   * The interval method regresses harmonic voltage on harmonic current over
#     the whole recording.  It needs no phase angles, so it works on every
#     file, and its intercept is a direct measurement of the distortion
#     present when the customer is drawing no harmonic current at all.
#   * The waveform method takes the sign of harmonic power from point-on-wave
#     captures, which carry V and I sampled simultaneously and so do carry the
#     angle.  It is the physically direct measurement, but only at the instants
#     the meter happened to capture.
#
# Neither is proof.  The power-direction sign is a screening indicator whose
# validity depends on the relative impedances either side of the PCC (Xu, Liu
# & Liu, "On the validity of the power direction method for harmonic source
# determination", IEEE Trans. Power Delivery 18(1), 2003), and the regression
# infers direction from covariance rather than measuring it.  Both are
# reported as evidence of where distortion appears to originate, never as an
# attribution of responsibility.
# ─────────────────────────────────────────────────────────────────────────────

#: Pearson r at or above this, with a positive slope, means the harmonic
#: voltage genuinely tracks the customer's harmonic current.
_DIRECTION_MIN_R = 0.50

#: Fewer aligned intervals than this and the regression is fitting noise.
_DIRECTION_MIN_POINTS = 20

#: Share of the harmonic voltage at high load that must be attributable to the
#: customer's own current before the indication is called downstream.
_DIRECTION_LOAD_SHARE = 0.60

#: Intervals in the lowest decile of customer harmonic current: what the PCC
#: distortion looks like when the customer is contributing least.
_BACKGROUND_QUANTILE = 0.10

#: A capture whose RMS voltage is this far from nominal is an event capture.
#: Harmonic direction during a sag describes the sag, not the steady state.
_WAVE_EVENT_PU = 0.10

#: Floor on the harmonic current in a capture before its angle means anything.
_WAVE_MIN_HARMONIC_A = 0.1

#: Fraction of the fundamental below which a harmonic phasor is angle noise.
_WAVE_MIN_HARMONIC_FRACTION = 0.005

#: Captures shorter than this cannot resolve the orders being asked about.
#: Set at three rather than a full ten because Pronto's event captures are
#: short by design -- the sub-two-cycle transient captures are still excluded,
#: but the 53 ms captures at 19.2 kHz carry three usable cycles.
_WAVE_MIN_CYCLES = 3.0

#: Phase-captures an order needs before its sign is a finding rather than an
#: anecdote: one capture on three phases is three readings of one instant.
_WAVE_MIN_SAMPLES = 3

#: Share of usable capture-phases that must agree on a sign before the order
#: gets a direction rather than "mixed".
_WAVE_AGREEMENT = 0.70


def _phasor(x: np.ndarray, window: np.ndarray, fs: float, freq: float) -> complex:
    """Complex peak-amplitude phasor of *x* at exactly *freq*.

    Projected onto the reference at the measured frequency rather than read
    out of an FFT bin: the capture is not an integer number of cycles, so no
    bin lands on 60 Hz, and a bin-quantized angle is wrong by tens of degrees
    at the higher orders -- which is the whole measurement here.  The same
    window multiplies V and I, so the angle *difference* is unaffected by it,
    and dividing by the window sum restores the amplitude.
    """
    n = len(x)
    ref = np.exp(-2j * np.pi * freq * np.arange(n) / fs)
    return complex(2.0 * np.sum(x * window * ref) / np.sum(window))


def _fundamental_hz(x: np.ndarray, fs: float, nominal: float = 60.0
                    ) -> Tuple[float, bool]:
    """Measured fundamental frequency of a capture, and whether to trust it.

    A capture is a fraction of a second, so an FFT bin is several hertz wide
    and the bin index alone is far too coarse: the reference the phasors are
    projected onto has to sit on the real fundamental.  The bin peak is
    therefore only a starting point, refined by maximising the projected
    magnitude itself.

    An estimate outside a few percent of nominal means the estimator was
    handed a transient or an interruption rather than a steady fundamental.
    Nominal is returned instead, and the caller is told, rather than every
    harmonic reference being built on a frequency the system never ran at.
    """
    n = len(x)
    if n < 16 or fs <= 0:
        return nominal, False
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft((x - x.mean()) * window))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs > nominal * 0.8) & (freqs < nominal * 1.2)
    if not band.any() or spec.max() <= 0:
        return nominal, False

    bin_hz = fs / n
    coarse = float(freqs[int(np.argmax(np.where(band, spec, 0.0)))])
    grid = np.arange(coarse - bin_hz, coarse + bin_hz, bin_hz / 50.0)
    grid = grid[grid > 0]
    if len(grid):
        magnitudes = [abs(_phasor(x, window, fs, f)) for f in grid]
        coarse = float(grid[int(np.argmax(magnitudes))])

    if abs(coarse / nominal - 1.0) > 0.05:
        return nominal, False
    return coarse, True


def harmonic_direction_from_intervals(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Where the harmonic voltage at the PCC comes from, from magnitudes alone.

    For each order, the interval harmonic voltage is regressed on the interval
    harmonic current:

        V_h = Z_h · I_h + V_bg

    The slope is the apparent harmonic impedance seen looking upstream, so
    Z_h · I_h is the part of the PCC distortion the customer's own current
    accounts for.  The intercept is the part that is there regardless -- the
    background distortion arriving from the system -- and it is checked
    against a direct measurement: the mean harmonic voltage over the intervals
    in the lowest decile of harmonic current, when the customer is
    contributing least.

    This runs on the whole recording rather than on the handful of instants a
    waveform capture covers, which is what makes it worth having alongside the
    phasor method even though it infers direction instead of measuring it.
    """
    # The same population the capture method reads, so that the agreement
    # between the two means something. Left unscoped, this method regressed
    # over whichever intervals survived the light-load gate -- the exporting
    # ones on a solar service -- while the captures were split to the importing
    # ones, and the two were then reported as corroborating each other.
    direction = primary_flow_direction(thresh)
    scoped, flow_info = flow_scope(df, thresh, direction)
    if scoped is None:
        return {"available": False, "orders": {}, "overall": "not_assessed",
                "flow": flow_info, "note": flow_info["reason"]}
    df = scoped

    significance = harmonic_spectrum_significance(df, thresh)
    orders: Dict[int, dict] = {}

    for h in _SOURCE_ORDERS:
        v_parts, i_parts = [], []
        for ph in ("a", "b", "c"):
            cv, ci = f"h{h}_voltage_{ph}", f"h{h}_current_{ph}"
            if cv in df.columns and ci in df.columns:
                v_al, i_al = df[cv].align(df[ci], join="inner")
                valid = v_al.notna() & i_al.notna() & (i_al > 0)
                if valid.sum():
                    v_parts.append(v_al[valid])
                    i_parts.append(i_al[valid])
        if not v_parts:
            continue

        v = pd.concat(v_parts).groupby(level=0).mean()
        i = pd.concat(i_parts).groupby(level=0).mean()
        v, i = v.align(i, join="inner")
        if len(v) < _DIRECTION_MIN_POINTS:
            continue

        slope, intercept = np.polyfit(i.to_numpy(float), v.to_numpy(float), 1)
        r = float(v.corr(i))
        i_high = float(i.quantile(0.95))
        v_from_load = max(float(slope) * i_high, 0.0)
        v_background = max(float(intercept), 0.0)
        total = v_from_load + v_background
        load_share = (v_from_load / total) if total > 0 else None

        # The intercept is an extrapolation; this is the same quantity read
        # straight off the data, and the two disagreeing is worth seeing.
        quiet = i <= i.quantile(_BACKGROUND_QUANTILE)
        v_quiet = float(v[quiet].mean()) if quiet.any() else None

        orders[h] = {
            "slope_ohm":      round(float(slope), 4),
            "intercept_v":    round(float(intercept), 4),
            "corr":           round(r, 3) if np.isfinite(r) else None,
            "points":         int(len(v)),
            "i_p95_a":        round(i_high, 3),
            "v_from_load_v":  round(v_from_load, 4),
            "v_background_v": round(v_background, 4),
            "v_at_quiet_v":   round(v_quiet, 4) if v_quiet is not None else None,
            "v_mean_v":       round(float(v.mean()), 4),
            "load_share":     round(load_share, 3) if load_share is not None else None,
        }

    if not orders:
        return {"available": False, "flow": flow_info,
                "note": "No order has both a harmonic voltage and a harmonic "
                        "current channel with enough aligned intervals."}

    if not significance["usable"]:
        for od in orders.values():
            od["indication"] = "not_assessed"
        return {
            "available": True, "orders": orders, "overall": "not_assessed",
            "significance": significance, "flow": flow_info,
            "note": ("Direction was not assessed: " + significance["reason"]
                     + " The regressions above are reported as measured data only."),
        }

    for od in orders.values():
        r = od["corr"]
        tracks_load = (r is not None and r >= _DIRECTION_MIN_R
                       and od["slope_ohm"] > 0
                       and od["load_share"] is not None
                       and od["load_share"] >= _DIRECTION_LOAD_SHARE)
        # Distortion that stays put while the customer's harmonic current
        # falls away has to be arriving from somewhere else.
        background_dominates = (
            od["v_mean_v"] > 0
            and od["v_background_v"] >= 0.5 * od["v_mean_v"]
            and (r is None or r < _DIRECTION_MIN_R)
        )
        if tracks_load:
            od["indication"] = "downstream"
        elif background_dominates:
            od["indication"] = "upstream"
        else:
            od["indication"] = "indeterminate"

    return {
        "available":    True,
        "orders":       orders,
        "overall":      _direction_consensus(
            [od["indication"] for od in orders.values()]),
        "significance": significance,
        "flow":         flow_info,
        "note": (
            "Direction inferred from how the harmonic voltage at the meter "
            "moves with the harmonic current drawn through it, over "
            + ("the whole recording. " if not flow_info["split"] else
               f"the {flow_info['direction']} intervals of the recording "
               f"({flow_info['intervals']} of {len(df)}), which is the same "
               "population the capture method reads so the two are comparable. ")
            + "It indicates where the distortion appears to originate, "
            "not responsibility for it."
        ),
    }


def _direction_orders(samples: Dict[int, List[dict]], sign: float) -> Dict[int, dict]:
    """Grade per-order harmonic power samples into an up/downstream indication.

    `sign` flips the whole set for reversed CTs.  It is +1 wherever the CT
    orientation is taken on faith rather than measured, because a harmonic
    power sign already means "the way the CT arrow points"; the inversion
    exists only to undo clamps installed backwards.
    """
    orders: Dict[int, dict] = {}
    for h in _SOURCE_ORDERS:
        group = samples.get(h) or []
        if len(group) < _WAVE_MIN_SAMPLES:
            continue
        vals = [sign * s["p"] for s in group]
        toward_customer = sum(1 for p in vals if p > 0)
        toward_system = len(vals) - toward_customer
        share_out = toward_system / len(vals)
        if share_out >= _WAVE_AGREEMENT:
            indication = "downstream"
        elif (1.0 - share_out) >= _WAVE_AGREEMENT:
            indication = "upstream"
        else:
            indication = "mixed"
        orders[h] = {
            "samples":          len(vals),
            "toward_system":    toward_system,
            "toward_customer":  toward_customer,
            "median_p_w":       round(float(np.median(vals)), 4),
            "median_angle_deg": round(float(np.median(
                [s["angle_deg"] for s in group])), 1),
            "indication":       indication,
        }
    return orders


def harmonic_direction_from_waveforms(ds: PQDataset, thresh: Thresholds) -> dict:
    """Direction of harmonic power flow, measured from point-on-wave captures.

    Each capture carries voltage and current sampled simultaneously, so the
    harmonic phasors carry an angle and the harmonic active power

        P_h = ½ · Re(V_h · conj(I_h))

    has a sign.  Positive is power flowing the same way as the fundamental --
    into the customer, from a source upstream.  Negative is harmonic power
    leaving the customer for the system, which is what a downstream source
    looks like.

    The fundamental sets the sign convention: at a load service P₁ flows into
    the customer, so a negative P₁ means the CTs were installed backwards and
    every harmonic sign with them.  That is detected and corrected here rather
    than being left to invert the conclusion silently.

    Captures taken during a sag or swell are excluded: their distortion
    describes the disturbance, not the steady state.
    """
    captures = list(getattr(ds, "waveforms", None) or [])
    out: dict = {
        "available": False, "captures_total": len(captures), "captures_used": 0,
        "excluded_event": 0, "excluded_light_load": 0, "excluded_short": 0,
        "excluded_no_fundamental": 0, "orders": {}, "overall": "indeterminate",
    }
    if not captures:
        out["note"] = ("No point-on-wave captures in this file, so harmonic "
                       "power direction could not be measured.")
        return out

    nominal = thresh.nominal_voltage
    # (order, phase-capture) -> P_h, before the polarity convention is fixed.
    samples: Dict[int, List[dict]] = {h: [] for h in (1,) + tuple(_SOURCE_ORDERS)}
    f0_seen: List[float] = []

    for cap in captures:
        fs = cap.get("fs_hz")
        voltages = cap.get("voltages") or {}
        currents = cap.get("currents") or {}
        phases = [p for p in ("a", "b", "c") if p in voltages and p in currents]
        if not fs or fs <= 0 or not phases:
            continue

        n = min(min(len(voltages[p]) for p in phases),
                min(len(currents[p]) for p in phases))
        if n / fs * thresh.frequency_nominal < _WAVE_MIN_CYCLES:
            out["excluded_short"] += 1
            continue

        # Ordered so each capture is counted under the first thing wrong with
        # it: a capture taken during a deep sag has no steady fundamental
        # either, and calling it an event says more than calling it unreadable.
        v_rms = float(np.sqrt(np.mean(
            np.asarray(voltages[phases[0]][:n], float) ** 2)))
        if nominal > 0 and abs(v_rms / nominal - 1.0) > _WAVE_EVENT_PU:
            out["excluded_event"] += 1
            continue

        f0, f0_ok = _fundamental_hz(np.asarray(voltages[phases[0]][:n], float),
                                    fs, thresh.frequency_nominal)
        if not f0_ok:
            # Without a fundamental to build the harmonic references on, every
            # angle below would be measured against the wrong frequency.
            out["excluded_no_fundamental"] += 1
            continue

        window = np.hanning(n)
        used_any = False
        for ph in phases:
            v = np.asarray(voltages[ph][:n], dtype=float)
            i = np.asarray(currents[ph][:n], dtype=float)
            v1 = _phasor(v, window, fs, f0)
            i1 = _phasor(i, window, fs, f0)
            if abs(i1) / np.sqrt(2.0) < _MIN_LOADED_AMPS:
                continue
            used_any = True
            # Carried onto every harmonic sample from this capture-phase: on a
            # generating service the two directions of fundamental flow have to
            # be separated, and that can only be done per capture, not from the
            # median over all of them.
            p1_ph = 0.5 * float(np.real(v1 * np.conj(i1)))
            samples[1].append({"p": p1_ph, "p1": p1_ph})
            for h in _SOURCE_ORDERS:
                vh = _phasor(v, window, fs, f0 * h)
                ih = _phasor(i, window, fs, f0 * h)
                floor = max(_WAVE_MIN_HARMONIC_A,
                            _WAVE_MIN_HARMONIC_FRACTION * abs(i1) / np.sqrt(2.0))
                if abs(ih) / np.sqrt(2.0) < floor:
                    continue
                samples[h].append({
                    "p": 0.5 * float(np.real(vh * np.conj(ih))),
                    "p1": p1_ph,
                    # Wrapped to ±180° by construction: an unwrapped
                    # difference of -249° is the same angle as +111° and
                    # medians of the two say opposite things.
                    "angle_deg": float(np.degrees(np.angle(vh * np.conj(ih)))),
                })
        if used_any:
            out["captures_used"] += 1
            f0_seen.append(f0)
        else:
            out["excluded_light_load"] += 1

    if not out["captures_used"]:
        # At a plant the same exclusion means the inverters were off, not that
        # a customer was drawing little -- there is no load there to be light.
        quiet = (f"{out['excluded_light_load']} while the plant was not "
                 f"producing (under {_MIN_LOADED_AMPS:.0f} A)"
                 if is_generation_only(thresh) else
                 f"{out['excluded_light_load']} at less than "
                 f"{_MIN_LOADED_AMPS:.0f} A of load")
        out["note"] = (
            f"None of the {len(captures)} capture(s) were usable: "
            f"{out['excluded_event']} were taken during a voltage event, "
            f"{quiet}, {out['excluded_short']} too short to resolve the orders, "
            f"{out['excluded_no_fundamental']} with no steady fundamental to "
            "measure against."
        )
        return out

    # CT polarity, from the fundamental. Which sign is the wrong one depends on
    # what the service is: a load imports, a plant exports, and a service that
    # does both cannot be told from reversed clamps at all. Only worth reading
    # when there is enough real power for its sign to mean anything.
    p1 = float(np.median([s["p"] for s in samples[1]])) if samples[1] else 0.0
    polarity_floor = 0.5 * _MIN_LOADED_AMPS * thresh.nominal_voltage
    polarity_verified = abs(p1) >= polarity_floor

    if is_generation_only(thresh):
        # A plant exports whenever it is running, so the expectation is simply
        # the load one with its sign flipped: P1 positive here is the anomaly.
        # The check is not lost at a generation site -- it is lost in the
        # middle, where both signs are legitimate.
        inverted = polarity_verified and p1 > 0
        orders   = _direction_orders(samples, -1.0 if inverted else 1.0)
    elif exports_power(thresh):
        # A generating service exports, so the sign of P1 no longer tells the
        # clamps apart from the flow: reversed CTs on an importing service and
        # correct CTs on an exporting one read the same. The inversion is
        # therefore not applied at all -- it would silently invert every
        # direction below on any capture taken while the site was generating.
        # Instead the captures are split on their own P1 and each half read on
        # its own terms, with the CTs assumed installed arrow-toward-load.
        # Sorted on their own P1, but only where that sign carries a direction.
        # A capture taken while generation was matching load has a fundamental
        # near zero, and filing it by the sign of the residue sorts it by
        # noise -- on an unbalanced service that can even put two phases of one
        # capture on opposite sides. The floor is the same one that decides
        # whether P1 is worth reading for polarity at all.
        importing = {h: [s for s in v if s["p1"] >= polarity_floor]
                     for h, v in samples.items()}
        exporting = {h: [s for s in v if s["p1"] <= -polarity_floor]
                     for h, v in samples.items()}
        near_zero = sum(1 for s in samples[1] if abs(s["p1"]) < polarity_floor)
        inverted = False
        polarity_verified = False
        orders = _direction_orders(importing, 1.0)
        out["export_split"] = {
            "importing": {
                "capture_phases": len(importing[1]),
                "orders":         orders,
                "overall":        _direction_consensus(
                    [od["indication"] for od in orders.values()]),
            },
            "exporting": {
                "capture_phases": len(exporting[1]),
                "orders":         (exp_orders := _direction_orders(exporting, 1.0)),
                "overall":        _direction_consensus(
                    [od["indication"] for od in exp_orders.values()]),
            },
            "near_crossover": near_zero,
            "deadband_w":     round(polarity_floor, 1),
        }
    else:
        inverted = polarity_verified and p1 < 0
        orders   = _direction_orders(samples, -1.0 if inverted else 1.0)

    out.update({
        # On a generating service the exporting captures are a result in their
        # own right, so the block is available on either half.
        "available":            bool(orders or (out.get("export_split") or {})
                                     .get("exporting", {}).get("orders")),
        "orders":               orders,
        "fundamental_hz":       round(float(np.median(f0_seen)), 3) if f0_seen else None,
        "ct_polarity_inverted": inverted,
        "ct_polarity_verified": polarity_verified,
        "median_p1_w":          round(abs(p1), 1),
        "overall":              _direction_consensus(
            [od["indication"] for od in orders.values()]),
        "note": (
            "Measured from the sign of harmonic power in "
            f"{out['captures_used']} point-on-wave capture(s). Captures are "
            "triggered snapshots, so this covers those instants and not the "
            "whole recording. The sign is a screening indicator: it identifies "
            "the side whose harmonic source dominates at the meter, and is "
            "least reliable when the impedances either side of the meter are "
            "comparable (Xu, Liu & Liu, IEEE T-PWRD 18(1), 2003)."
        ),
    })
    if is_generation_only(thresh):
        if inverted:
            out["polarity_note"] = (
                "The fundamental real power measured as flowing into the "
                "premises while the plant was producing, which at a generating "
                "site means the CTs were installed reversed. Every direction "
                "here is stated with that inversion corrected."
            )
        elif not polarity_verified:
            out["polarity_note"] = (
                f"The captures carried only {abs(p1):.0f} W of fundamental real "
                "power, too little for its sign to confirm the CT orientation, "
                "so the directions below assume the CTs were installed with "
                "the arrow toward the load. Reversed CTs would invert every one."
            )
        else:
            out["polarity_note"] = (
                "The fundamental real power measured as flowing out of the "
                "premises, which is what a plant that is producing should read "
                "and confirms the CTs are installed the right way round. Note "
                "that at a generating site the harmonic orders are expected to "
                "read as coming from the customer side: the plant is the only "
                "source behind the meter, so the direction below is a check on "
                "the measurement rather than a finding about who is "
                "responsible. What the orders are worth reading for is their "
                "size."
            )
    elif exports_power(thresh):
        split = out["export_split"]
        n_imp = split["importing"]["capture_phases"]
        n_exp = split["exporting"]["capture_phases"]
        out["polarity_note"] = (
            f"This service carries on-site generation, so the captures were "
            f"separated by the direction of fundamental flow at the instant "
            f"each was taken: {n_imp} while importing, {n_exp} while "
            f"exporting. The table states the importing captures, which are "
            f"the ones comparable with a service that does not generate; the "
            f"exporting captures are reported separately below and are not "
            f"sign-corrected against them. Because the site exports, the sign "
            f"of fundamental power cannot confirm the CT orientation, so both "
            f"readings assume the CTs were installed with the arrow toward "
            f"the load."
        )
        if split.get("near_crossover"):
            out["polarity_note"] += (
                f" A further {split['near_crossover']} carried less than "
                f"{split['deadband_w']:.0f} W of fundamental real power, too "
                f"little for its sign to say which way power was flowing, and "
                f"are in neither half."
            )
        if not n_imp:
            out["polarity_note"] += (
                " No capture was taken while the service was importing, so "
                "there is nothing here to compare against a non-generating "
                "service."
            )
    elif inverted:
        out["polarity_note"] = (
            "The fundamental real power measured as flowing out of the "
            "premises, which at a load service means the CTs were installed "
            "reversed. Every direction here is stated with that inversion "
            "corrected; if the service has on-site generation exporting during "
            "the captures, the correction is wrong and the directions invert."
        )
    elif not polarity_verified:
        out["polarity_note"] = (
            f"The captures carried only {abs(p1):.0f} W of fundamental real "
            "power, too little for its sign to confirm the CT orientation, so "
            "the directions below assume the CTs were installed with the "
            "arrow toward the load. Reversed CTs would invert every one."
        )
    if not orders:
        out["note"] = (
            f"{out['captures_used']} capture(s) were readable, but no harmonic "
            f"order reached {_WAVE_MIN_HARMONIC_A} A in at least "
            f"{_WAVE_MIN_SAMPLES} phase-captures, so no direction could be "
            "measured from them."
        )
    return out


def _direction_consensus(indications: List[str]) -> str:
    """One verdict from the per-order ones, without averaging away conflict."""
    real = [i for i in indications if i in ("downstream", "upstream")]
    if not real:
        return "indeterminate"
    if all(i == "downstream" for i in real):
        return "downstream"
    if all(i == "upstream" for i in real):
        return "upstream"
    return "mixed"


def check_harmonic_direction(ds: PQDataset, thresh: Thresholds) -> dict:
    """Both readings of harmonic source direction, and whether they agree.

    Agreement between a method that needs no angles and one that measures them
    is the strongest statement this data can support; disagreement is reported
    rather than resolved, because the two look at different things -- the whole
    recording versus a few triggered instants -- and either can be the more
    representative one.
    """
    intervals = harmonic_direction_from_intervals(ds.df, thresh)
    waveforms = harmonic_direction_from_waveforms(ds, thresh)

    agreement: Dict[int, str] = {}
    for h in _SOURCE_ORDERS:
        a = (intervals.get("orders") or {}).get(h, {}).get("indication")
        b = (waveforms.get("orders") or {}).get(h, {}).get("indication")
        decided = [x for x in (a, b) if x in ("downstream", "upstream")]
        if len(decided) == 2:
            agreement[h] = "agree" if a == b else "disagree"
        elif decided:
            agreement[h] = "single_method"

    overalls = [r.get("overall") for r in (intervals, waveforms)
                if r.get("available")]
    decided = [o for o in overalls if o in ("downstream", "upstream")]
    if len(set(decided)) == 1 and decided:
        overall = decided[0]
    elif len(set(decided)) > 1:
        overall = "conflicting"
    else:
        overall = "indeterminate"

    return {
        "available":  bool(intervals.get("available") or waveforms.get("available")),
        "interval":   intervals,
        "waveform":   waveforms,
        "agreement":  agreement,
        "overall":    overall,
        "methods_agree": (bool(agreement) and "disagree" not in agreement.values()
                          and "agree" in agreement.values()),
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

    v_h_cols = [c for c in df.columns if _HARMONIC_COL.match(c) and "_voltage_" in c]
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

    il_amps, harm_rms, fundamental, _harm_source = demand_current_il(df, thresh)
    if not il_amps:
        return result

    if thresh.isc_amps is None:
        result["note"] = "ISC not provided — statistical harmonic check requires --isc"
        return result

    h_cols = [c for c in df.columns if _HARMONIC_COL.match(c) and "_current_" in c]
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

    # TDD, not the raw THD channel.  This block previously graded thd_current_*
    # straight against the TDD limit with no conversion, so on a service that
    # falls to no output -- a solar site at night -- it reported the runaway THD
    # ratio as though it were TDD.  On one 3.7-day solar recording that printed
    # a P95 of 59.70% against a maximum TDD of 4.44% elsewhere in the same
    # report: two different series under one label, and an impossible comparison
    # a reader could not have caught.
    tdd_lim = _tdd_limit(isc_il)
    weekly["thd"] = {}
    daily_vst["thd"] = {}
    for ph, s in tdd_by_phase(df, il_amps, harm_rms, fundamental).items():
        s = s.dropna()
        if s.empty:
            continue
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

    # Name the case that actually binds.  "P95 within limits for all orders"
    # tells a reader nothing about how close the service came; the tightest
    # margin does, and it is what a follow-up measurement should watch.
    binding = None
    for key, per_phase in weekly.items():
        for ph, w in per_phase.items():
            if not w.get("limit"):
                continue
            ratio = w["p95"] / w["limit"]
            if binding is None or ratio > binding["ratio"]:
                binding = {
                    "order": key, "phase": ph,
                    "p95": w["p95"], "limit": w["limit"],
                    "ratio": round(ratio, 3),
                    "p95_pass": w["p95_pass"],
                }

    result.update({
        "weekly": weekly, "daily_vst": daily_vst,
        "overall_pass": overall_pass,
        "tdd_limit": round(tdd_lim, 1),
        "isc_class": _tdd_class(isc_il),
        "binding": binding,
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
    if vdf.empty:
        return {
            "available":            False,
            "error":                "No intervals with all voltage phases present simultaneously.",
            "pct_exceeding":        None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }

    if len(v_cols) >= 3:
        # NEMA MG1 as defined: max deviation from the average of the three line
        # voltages, over that average. The concern it encodes is negative-
        # sequence heating in three-phase motors.
        avg  = vdf.mean(axis=1)
        dev  = (vdf.subtract(avg, axis=0)).abs().max(axis=1)
        imbalance = np.where(avg > 0, dev / avg * 100, np.nan)
        metric, metric_label = "nema_mg1", "NEMA MG1 voltage unbalance"
        basis = ("Max deviation from the three-phase average, per NEMA MG1.")
        note = None
    else:
        # Two legs. NEMA MG1 is defined for three-phase systems and its formula
        # does not carry over: applied to two elements the deviation from their
        # own mean is half their difference, so a 4 V spread on 120 V legs
        # reported 1.67% where an engineer would say the legs are 3.3% apart.
        # There is also no negative sequence on a single-phase service and no
        # three-phase motor to derate, so what matters is the leg difference
        # itself -- an indicator of unequal loading or neutral impedance.
        diff = (vdf.iloc[:, 0] - vdf.iloc[:, 1]).abs()
        base = thresh.nominal_voltage if thresh.nominal_voltage > 0 else np.nan
        imbalance = (diff / base * 100).to_numpy()
        metric, metric_label = "leg_difference", "Leg-to-leg voltage difference"
        basis = ("Difference between the two legs as a percentage of nominal. "
                 "NEMA MG1 unbalance is defined for three-phase systems and is "
                 "not applicable to a single-phase service.")
        if is_single_phase_208(thresh.service_type):
            # Only two of the three wye phases are measured, so the source's
            # own unbalance cannot be separated from the customer's loading.
            note = ("This service takes two legs of a three-phase 120/208 "
                    "transformer, so only two of the three phases are "
                    "measured. True three-phase unbalance at the transformer "
                    "cannot be determined from this recording; the figure "
                    "below is the difference between the two legs served.")
        else:
            note = None

    imb_series = pd.Series(imbalance, index=vdf.index)
    exceed = imb_series > thresh.imbalance_limit

    return {
        "available":            True,
        "error":                None,
        "limit_pct":            thresh.imbalance_limit,
        "metric":               metric,
        "metric_label":         metric_label,
        "basis":                basis,
        "note":                 note,
        "max_imbalance_pct":    float(np.nanmax(imbalance)),
        "mean_imbalance_pct":   float(np.nanmean(imbalance)),
        "pct_exceeding":        float(exceed.mean() * 100),
        "imbalance_series":     imb_series,
        "violation_timestamps": imb_series.index[exceed],
    }


def check_current_imbalance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Current imbalance: a limit on three-phase service, a measurement on two legs.

    Three-phase: imbalance = max phase deviation from the average / average,
    against the 10% PSC procedure limit.

    Two legs: the same arithmetic reduces to |I1 - I2| / (I1 + I2), so the 10%
    limit would mean the legs differ by 20% of their average -- which is
    ordinary on a service where loads are assigned to legs by breaker position.
    There is no PSCo limit on split-phase leg imbalance and no standard one, so
    none is applied: the figure is reported as a measurement, and the verdict
    is left to the things that do have limits -- per-leg ANSI C84.1 voltage and
    `check_neutral_health`. Leg imbalance is what *causes* those; reporting it
    as a violation in its own right would be inventing a threshold.

    Also reports neutral current statistics if the channel is present.
    """
    geometry = service_geometry(thresh, df.columns)
    i_cols = [c for c in ["current_a", "current_b", "current_c"] if c in df.columns]
    if len(i_cols) < 2:
        return {
            "available":            False,
            "error":                "Need at least two current phases for imbalance calculation.",
            "pct_exceeding":        None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }

    idf = df[i_cols].dropna()
    if idf.empty:
        return {
            "available":            False,
            "error":                "No intervals with all current phases present simultaneously.",
            "pct_exceeding":        None,
            "violation_timestamps": pd.DatetimeIndex([]),
        }

    avg = idf.mean(axis=1)
    dev = idf.subtract(avg, axis=0).abs().max(axis=1)
    # Skip rows where average current is negligible (avoids divide-near-zero noise)
    imbalance = np.where(avg > 1.0, dev / avg * 100, np.nan)
    imb_series = pd.Series(imbalance, index=idf.index)

    two_leg = len(i_cols) < 3 or geometry in ("split-phase", "two-leg-208")
    if two_leg:
        limit = None
        exceed = pd.Series(False, index=idf.index)
        metric, metric_label = "leg_difference", "Leg current difference"
        basis = ("Difference between the legs as a percentage of their average "
                 "-- for two legs this is |I1 - I2| / (I1 + I2).")
        if geometry == "two-leg-208":
            note = ("This service takes two legs of a three-phase 120/208 "
                    "transformer, so the third phase is not measured and "
                    "three-phase imbalance at the transformer cannot be "
                    "determined from this recording. No limit is applied to "
                    "the difference between the two legs served; the figure "
                    "is a measurement, not a violation.")
        else:
            note = ("No limit is applied. Unequal loading of the two legs is "
                    "normal on a 120/240 service -- loads are assigned to legs "
                    "by breaker position and change through the day -- and "
                    "neither PSCo procedure nor any standard sets a limit on "
                    "it. This figure is a measurement, not a violation. What "
                    "it bears on is neutral current and the difference between "
                    "the two leg voltages, which are evaluated against ANSI "
                    "C84.1 and reported under neutral health.")
    else:
        limit = thresh.current_imbalance_limit
        exceed = imb_series > limit
        metric, metric_label = "nema_style", "Current imbalance"
        basis = ("Max phase deviation from the three-phase average, over that "
                 "average, per PSC procedure.")
        note = None

    result: dict = {
        "available":            True,
        "error":                None,
        "geometry":             geometry,
        "limit_pct":            limit,
        "metric":               metric,
        "metric_label":         metric_label,
        "basis":                basis,
        "note":                 note,
        "max_imbalance_pct":    float(np.nanmax(imbalance)),
        "mean_imbalance_pct":   float(np.nanmean(imbalance)),
        "pct_exceeding":        float(exceed.mean() * 100),
        "violation_timestamps": imb_series.index[exceed],
    }

    if "current_neutral" in df.columns:
        In = df["current_neutral"].dropna()
        if not In.empty:
            avg_phase = avg.reindex(In.index)
            in_pct = np.where(avg_phase > 1.0, In.values / avg_phase.values * 100, np.nan)
            # Only report phase-relative neutral stats when at least one interval
            # had a valid (non-negligible) phase average to compute a % against —
            # downstream consumers assume these fields are always real floats,
            # never None, whenever "neutral_current" is present at all.
            if np.isfinite(in_pct).any():
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
            dedicated = has_dedicated_transformer(thresh.customer_class)
            # `peak_8h_kva` is this service's demand. Where the transformer is
            # shared -- a residential pole or pad transformer feeds several
            # houses -- that demand is a *lower bound* on the transformer's
            # load, because the neighbours were not measured. The inference is
            # one-sided: above nameplate proves an overload whoever else is on
            # it, but below nameplate proves nothing at all. So the negative
            # case is None (not determinable) rather than False, and nothing
            # downstream may turn it into an all-clear.
            if pct > 100:
                overloaded = True
            elif dedicated:
                overloaded = False
            else:
                overloaded = None
            result["transformer"] = {
                "nameplate_kva": thresh.transformer_kva,
                "peak_8h_kva":   round(peak_8h_kva, 1),
                "pct_nameplate": round(pct, 1),
                "dedicated":     dedicated,
                "overloaded":    overloaded,
                "note": None if dedicated else (
                    "This transformer serves other customers whose load was "
                    "not measured. The figure below is this service's own "
                    "demand against the nameplate — its contribution to the "
                    "transformer's load, not that load. Total transformer "
                    "loading cannot be determined from a recording at one "
                    "meter."),
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

    # True interval peak current from the max-min record
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
    averages (augmented by the max-min min/peak columns) when adaptive data is
    absent.

    Detects:
      - voltage_sag   : V < 90 % nominal (leading edge)
      - voltage_swell : V > 110 % nominal (leading edge)
      - voltage_spike : |ΔV| > event_delta_pct × nominal in one sample
      - flicker_pst   : adap_pst > 1.0  (adaptive only)
      - flicker_plt   : adap_plt > 0.65 (adaptive only)
      - current_step  : |ΔI| > 25 % of mean current (5 A absolute floor)
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
            # 25% of mean, with a 5 A absolute floor — on lightly loaded services
            # (mean of a few amps) a pure percentage threshold flags every
            # appliance cycle as an event.
            delta_i = max(0.25 * mean_i, 5.0)
            diffs = s.diff().abs()
            for ts in diffs[diffs > delta_i].index:
                events.append({"timestamp": ts, "type": "current_step", "phase": phase,
                               "delta_a": float(diffs.loc[ts])})

    else:
        # ── Interval fallback (interval averages + min/peak columns) ──────────
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

    # ── Waveform-capture sag/swell events (½-cycle RMS per IEEE 1159) ─────────
    # Point-on-wave captures resolve events shorter than the adaptive record's
    # cycle-level resolution.  Events already seen in the adaptive data (same
    # phase within ±2 s) are skipped to avoid double counting.
    wf_events = _waveform_sag_swell_events(ds, thresh)
    if wf_events:
        # Waveform timestamps come from naive capture labels; align tz with any
        # existing events so comparison and sorting never mix aware and naive.
        tzinfo = next((getattr(e.get("timestamp"), "tzinfo", None) for e in events
                       if getattr(e.get("timestamp"), "tzinfo", None) is not None), None)
        if tzinfo is not None:
            for w in wf_events:
                w["timestamp"] = pd.Timestamp(w["timestamp"]).tz_localize(tzinfo)

        def _is_dup(w):
            for e in events:
                try:
                    close = abs((e["timestamp"] - w["timestamp"]).total_seconds()) <= 2.0
                except TypeError:
                    close = False
                if close and e["type"] == w["type"] and e.get("phase") == w["phase"]:
                    return True
            return False
        events.extend(w for w in wf_events if not _is_dup(w))

    events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True) \
        if events else pd.DataFrame(columns=["timestamp", "type", "phase"])
    return {
        "event_count":       len(events_df),
        "events":            events_df,
        "data_source":       data_source,
        "waveform_captures": len(getattr(ds, "waveforms", []) or []),
    }


def _waveform_sag_swell_events(ds: PQDataset, thresh: Thresholds) -> List[dict]:
    """Extract sag/swell events from point-on-wave captures via a sliding
    half-cycle RMS envelope (IEEE 1159 characterization).  Durations are
    clamped to the capture window, so they are lower bounds for events that
    outlast the capture."""
    import datetime as _dt

    nominal = thresh.nominal_voltage
    out: List[dict] = []
    for wf in getattr(ds, "waveforms", None) or []:
        t  = wf.get("t")
        fs = wf.get("fs_hz")
        if t is None or not fs or fs <= 0:
            continue
        w = max(int(round(fs / 60.0 / 2)), 8)          # half-cycle window
        for ph, x in wf.get("voltages", {}).items():
            n = min(len(x), len(t))
            if n < 2 * w:
                continue
            x = np.asarray(x[:n], dtype=float)
            c = np.cumsum(np.concatenate(([0.0], x * x)))
            rms = np.sqrt((c[w:] - c[:-w]) / w)         # rms[i] over x[i:i+w]
            for kind, mask in (
                ("voltage_sag",   rms < 0.9 * nominal),
                ("voltage_swell", rms > 1.1 * nominal),
            ):
                if not mask.any():
                    continue
                edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
                for s_i, e_i in zip(edges[::2], edges[1::2]):
                    seg = rms[s_i:e_i]
                    dur_ms = (t[min(e_i + w - 1, n - 1)] - t[s_i]) * 1000.0
                    if dur_ms < 1000.0 / 60.0 / 2:      # ignore < half a cycle
                        continue
                    extreme = float(seg.min() if kind == "voltage_sag" else seg.max())
                    out.append({
                        "timestamp":   wf["timestamp"] + _dt.timedelta(seconds=float(t[s_i])),
                        "type":        kind,
                        "phase":       ph.upper(),
                        "value_v":     round(extreme, 1),
                        "duration_ms": round(float(dur_ms), 1),
                        "source":      "waveform",
                    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8a. NEUTRAL HEALTH  (split-phase open-neutral detection)
# ─────────────────────────────────────────────────────────────────────────────

def check_neutral_health(ds: PQDataset, thresh: Thresholds) -> dict:
    """
    Assess split-phase neutral integrity. Only meaningful for split-phase topology.

    Combines five independent indicators:
    - Cross-leg correlation  : healthy legs track together (r > 0.8);
                               open neutral causes opposition (r → −1).
                               The primary test, and valid on both
                               configurations.
    - Voltage sum stability  : diagnostic only on a 120/208 two-leg service,
                               where an open neutral collapses the sum from
                               2 x nominal toward the line-to-line voltage.
                               On 120/240 the legs are collinear, so the sum
                               equals the line-to-line voltage whether the
                               neutral is intact or open -- see the note in
                               section 1 below.
    - Voltage asymmetry      : |L1 - L2| sustained above a few percent of
                               nominal indicates unequal loading, neutral
                               resistance, or both.
    - Neutral-to-earth Vne   : elevated Vne indicates neutral impedance
    - Coincident opposing events: one leg sags while the other swells
    """
    topology = ds.meta.get("topology", "unknown")
    two_leg_208 = is_single_phase_208(thresh.service_type)
    if topology != "split-phase" and not two_leg_208:
        return {"available": False, "reason": "not a two-leg service"}

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
    # What the sum can tell you differs by configuration, and the difference is
    # not cosmetic:
    #
    #   120/240 split-phase -- the legs are collinear (180 deg apart), so their
    #     scalar sum is the line-to-line voltage, 240 V.  Open the neutral and
    #     the two loads sit in series across that same 240 V, so the sum is
    #     still 240 V.  The sum therefore carries no open-neutral information
    #     here; it only confirms the measurement is sane.
    #
    #   120/208 two-leg -- healthy, the legs are 120 deg apart and each reads
    #     120 V, so the scalar sum is still 240 V.  But open the neutral and
    #     the loads sit in series across the *line-to-line* voltage, 208 V, so
    #     the sum collapses toward 208.  On this configuration the sum is a
    #     genuine open-neutral discriminator.
    vsum     = va_a + vb_a
    sum_mean = float(vsum.mean())
    sum_std  = float(vsum.std())
    healthy_sum = 2.0 * nom
    # The gate above already established which configuration this is, so read
    # the factor from that rather than from ll_factor's default -- with no
    # picker set ll_factor assumes three-phase, which would wrongly mark a
    # 120/240 service's sum as diagnostic.
    open_neutral_sum = nom * ((3.0 ** 0.5) if two_leg_208 else 2.0)
    sum_is_diagnostic = abs(healthy_sum - open_neutral_sum) > 0.05 * nom
    sum_toward_open = None
    if sum_is_diagnostic:
        # 1.0 means the sum sits where an open neutral would put it, 0.0 where
        # a healthy service would.
        span = healthy_sum - open_neutral_sum
        sum_toward_open = float((healthy_sum - sum_mean) / span) if span else None

    # ── 2. Cross-leg Pearson correlation ─────────────────────────────────────
    # Undefined when either leg is perfectly flat -- there is no variation to
    # correlate. That is not a fault, so report it as unavailable rather than
    # letting a NaN reach the page or read as a negative correlation.
    leg_corr = float(va_a.corr(vb_a))
    corr_available = not (np.isnan(leg_corr))
    if not corr_available:
        leg_corr = 1.0

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
                f"Neutral-to-earth voltage reached {_m(vne_max, '.1f', ' V')} (mean {_m(vne_mean, '.1f', ' V')}). "
                "Above 2 V indicates significant neutral impedance; above 5 V is a safety hazard — "
                "investigate immediately."
            )
        elif vne_max > 2.0:
            findings.append(
                f"Neutral-to-earth voltage elevated: max {_m(vne_max, '.1f', ' V')}, mean {_m(vne_mean, '.1f', ' V')}. "
                "Investigate neutral conductor connections and the grounding electrode system."
            )
        elif vne_max > 0.5:
            findings.append(
                f"Neutral-to-earth voltage mildly elevated: max {_m(vne_max, '.1f', ' V')} (normal < 0.5 V). "
                "Monitor and investigate if increasing."
            )
        else:
            findings.append(f"Neutral-to-earth voltage is normal (max {_m(vne_max, '.2f', ' V')}).")

    if leg_corr < 0.0:
        findings.append(
            f"Cross-leg voltage correlation is negative (r = {_m(leg_corr, '.3f')}). "
            "When L1 rises, L2 falls — a strong indicator of the neutral floating between legs."
        )
    elif leg_corr < 0.5:
        findings.append(
            f"Cross-leg voltage correlation is weak (r = {_m(leg_corr, '.3f')}; healthy > 0.80). "
            "Legs are not tracking the source together — investigate neutral continuity."
        )

    if sum_std > 3.0:
        findings.append(
            f"Voltage sum (L1 + L2) is unstable: mean {_m(sum_mean, '.1f', ' V')}, std {_m(sum_std, '.1f', ' V')}. "
            f"A solid neutral holds L1 + L2 near {healthy_sum:.0f} V with std < 1 V."
        )
    elif sum_std > 1.0:
        findings.append(
            f"Voltage sum (L1 + L2) shows moderate variation: "
            f"mean {_m(sum_mean, '.1f', ' V')}, std {_m(sum_std, '.2f', ' V')}."
        )

    # On a 120/208 two-leg service the sum separates a healthy neutral from an
    # open one; on a 120/240 service it cannot, and saying so stops a steady
    # 240 V reading being taken as evidence the neutral is sound.
    if sum_is_diagnostic:
        if sum_toward_open is not None and sum_toward_open > 0.5:
            findings.append(
                f"Voltage sum has collapsed toward the line-to-line value: "
                f"L1 + L2 = {_m(sum_mean, '.1f', ' V')} against {healthy_sum:.0f} V for a "
                f"healthy neutral and {open_neutral_sum:.0f} V if the two legs "
                f"were in series across the line-to-line voltage. On a "
                f"single-phase 120/208 service that is the open-neutral "
                f"signature."
            )
        else:
            findings.append(
                f"Voltage sum sits at {_m(sum_mean, '.1f', ' V')}, near the "
                f"{healthy_sum:.0f} V expected of a healthy neutral rather "
                f"than the {open_neutral_sum:.0f} V an open neutral would "
                f"produce on this 120/208 service."
            )
    else:
        findings.append(
            f"On a 120/240 service both legs are collinear, so L1 + L2 equals "
            f"the line-to-line voltage ({healthy_sum:.0f} V) whether the "
            f"neutral is intact or open. The sum confirms the measurement is "
            f"consistent but carries no open-neutral information here — the "
            f"cross-leg correlation and asymmetry below do."
        )

    if asym_pct > 5.0:
        findings.append(
            f"Sustained voltage asymmetry: mean |L1 − L2| = {_m(asym_mean, '.1f', ' V')} "
            f"({_m(asym_pct, '.1f', '%')} of nominal), max {_m(asym_max, '.1f', ' V')}. "
            "Investigate load balance and neutral continuity."
        )
    elif asym_pct > 2.0:
        findings.append(
            f"Moderate voltage asymmetry: mean |L1 − L2| = {_m(asym_mean, '.1f', ' V')} "
            f"({_m(asym_pct, '.1f', '%')} of nominal)."
        )

    if not findings:
        findings.append(
            f"Neutral appears healthy: L1 + L2 = {_m(sum_mean, '.1f', ' V')} (std {_m(sum_std, '.2f', ' V')}), "
            f"leg correlation r = {_m(leg_corr, '.3f')}, asymmetry {_m(asym_mean, '.1f', ' V')} ({_m(asym_pct, '.1f', '%')})."
        )

    return {
        "available":         True,
        "topology":          "1ph-208" if two_leg_208 else "split-phase",
        "sample_count":      len(aligned),
        "sum_mean_v":        round(sum_mean, 2),
        "sum_std_v":         round(sum_std, 3),
        "healthy_sum_v":     round(healthy_sum, 1),
        "open_neutral_sum_v": round(open_neutral_sum, 1),
        "sum_is_diagnostic": sum_is_diagnostic,
        "sum_toward_open":   (round(sum_toward_open, 3)
                              if sum_toward_open is not None else None),
        "leg_correlation":   round(leg_corr, 3) if corr_available else None,
        "leg_correlation_available": corr_available,
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
# Service impedance
#
# The same idea as the resonance screen, at the fundamental: voltage that
# falls as current rises is measuring the impedance between the source and the
# meter. Here it is quantified rather than correlated, and compared against
# what the picked transformer and service conductor ought to give.
#
# A high-impedance connection -- a corroded splice, a loose lug, an undersized
# or degraded conductor -- shows up three ways, and the first two need no
# conductor table at all:
#
#   * one phase reading materially higher than the others, which is what a
#     single bad connection looks like and is self-referencing;
#   * neutral-to-earth voltage rising with neutral current, which measures the
#     neutral connection directly;
#   * a total above what the transformer and conductor account for.
# ─────────────────────────────────────────────────────────────────────────────

#: Load steps needed before a slope through them means anything.
_IMPEDANCE_MIN_STEPS = 30

#: A step in current has to be this big to be this customer's load switching
#: rather than interval-to-interval noise: half an amp, or a twentieth of the
#: peak, whichever is larger.
_IMPEDANCE_STEP_MIN_A = 0.5
_IMPEDANCE_STEP_FRACTION = 0.05

#: Share of load steps whose voltage moves the opposite way, as series
#: impedance requires. At chance the fit is measuring something else.
_IMPEDANCE_MIN_CONSISTENCY = 0.65

#: The meter reports voltage to a tenth of a volt; a drop smaller than that
#: across the whole load range was never measurable.
_VOLTAGE_RESOLUTION_V = 0.1

#: Above this correlation between the real and reactive regressors, the power
#: factor did not move enough to separate R from X.
_IMPEDANCE_COLLINEAR_R = 0.98

#: Ratios of measured to expected impedance that change what the report says.
_IMPEDANCE_ELEVATED = 1.5
_IMPEDANCE_HIGH = 2.5

#: A phase whose resistance exceeds the lowest phase's by this much, and by
#: enough volts at peak load to matter, is the loose-connection signature.
_PHASE_ASYMMETRY_RATIO = 1.5
_PHASE_ASYMMETRY_VOLTS = 1.0

#: Neutral-to-earth volts at peak neutral current. Two is the level the
#: neutral integrity section already calls significant.
_NEUTRAL_RISE_VOLTS = 2.0


def _impedance_fit(v: pd.Series, i: pd.Series, pf: Optional[pd.Series],
                   rises_with_current: bool = False) -> Optional[dict]:
    """Series impedance from what each step in load does to the voltage.

    Fitted on differences between consecutive intervals, not on the levels.
    Regressing voltage on current directly measures the wrong thing at a
    residential service: the whole neighbourhood's air conditioning runs at
    the same time of day as this customer's, so feeder-wide droop is
    correlated with this load and lands in the slope. On a real 150 ft drop
    that read 0.29 Ω, against an expected 0.04 Ω -- a false positive on
    essentially every site.

    Differencing removes it. Feeder loading and regulator action drift over
    hours; a load switching on moves this meter's current and voltage in the
    same interval. So the estimator uses only intervals where the current
    stepped, and fits through the origin:

        ΔV = −R·Δ(I·cosφ) − X·Δ(I·sinφ)

    R and X separate only when the power factor moved; otherwise the
    effective magnitude is returned alone rather than a split that looks
    precise and is arbitrary. A last guard is the share of steps where
    voltage moved the opposite way to current: series impedance requires
    that of nearly all of them, and at chance the fit is reading something
    else entirely.
    """
    v, i = v.align(i, join="inner")
    valid = v.notna() & i.notna() & (v > 0) & (i > 0)
    if pf is not None:
        pf = pf.reindex(v.index)
        valid &= pf.notna() & (pf.abs() <= 1.0)
    v, i = v[valid], i[valid]
    if len(v) < _IMPEDANCE_MIN_STEPS:
        return None

    i_max = float(i.max())
    dv = v.diff().to_numpy(float)
    di = i.diff().to_numpy(float)
    floor = max(_IMPEDANCE_STEP_MIN_A, _IMPEDANCE_STEP_FRACTION * i_max)
    steps = np.isfinite(dv) & np.isfinite(di) & (np.abs(di) >= floor)
    n_steps = int(steps.sum())

    out: dict = {"points": int(len(v)), "i_max_a": round(i_max, 2),
                 "steps": n_steps, "step_floor_a": round(floor, 2)}

    if n_steps < _IMPEDANCE_MIN_STEPS:
        out.update({"identifiable": False, "reason": (
            f"Only {n_steps} interval(s) showed a load step of {floor:.1f} A "
            "or more, which is too few to measure a voltage drop against.")})
        return out

    # A voltage that never moves is not a failed fit, it is a measurement:
    # both real files record neutral-to-earth voltage as 0.0 or 0.1 V and
    # nothing else, which says the neutral connection is sound, not that the
    # data is bad. Reported as its own outcome so the report can say so.
    moved = float(np.mean(np.abs(dv[steps]) >= _VOLTAGE_RESOLUTION_V / 2))
    if moved < 0.2:
        out.update({"identifiable": False, "at_resolution": True,
                    "moved_share": round(moved, 3), "reason": (
                        f"Voltage moved by less than the meter's "
                        f"{_VOLTAGE_RESOLUTION_V} V resolution in "
                        f"{_m(1 - moved, '.0%')} of the {n_steps} load steps: the "
                        "drop across this impedance is below what the meter "
                        "can report.")})
        return out

    # A phase voltage falls as its load rises; neutral-to-earth voltage rises
    # with neutral current. Both are the same measurement with opposite signs.
    product = dv[steps] * di[steps]
    consistency = float(np.mean(product > 0 if rises_with_current else product < 0))
    out["consistency"] = round(consistency, 3)
    if consistency < _IMPEDANCE_MIN_CONSISTENCY:
        out.update({"identifiable": False, "reason": (
            f"Voltage moved against the load in only {_m(consistency, '.0%')} of the "
            f"{n_steps} load steps. A series impedance would show it in nearly "
            "all of them, so what varies this voltage is not this service's "
            "own current.")})
        return out

    out["identifiable"] = True

    if pf is not None:
        cos_phi = pf[valid].abs().to_numpy(float)
        sin_phi = np.sqrt(np.clip(1.0 - cos_phi ** 2, 0.0, 1.0))
        d_real = np.diff(i.to_numpy(float) * cos_phi, prepend=np.nan)
        d_reac = np.diff(i.to_numpy(float) * sin_phi, prepend=np.nan)
        usable = steps & np.isfinite(d_real) & np.isfinite(d_reac)
        if usable.sum() >= _IMPEDANCE_MIN_STEPS:
            collinear = abs(float(np.corrcoef(d_real[usable], d_reac[usable])[0, 1]))
            design = np.column_stack([d_real[usable], d_reac[usable]])
            coef, *_ = np.linalg.lstsq(design, dv[usable], rcond=None)
            r_ohm, x_ohm = -float(coef[0]), -float(coef[1])
            # A negative reactance is not a service impedance. Where the fit
            # returns one, the split did not hold up and only the magnitude
            # below is reported.
            if (np.isfinite(collinear) and collinear < _IMPEDANCE_COLLINEAR_R
                    and r_ohm > 0 and x_ohm >= 0):
                out.update({
                    "r_ohm": round(r_ohm, 5),
                    "x_ohm": round(x_ohm, 5),
                    "z_ohm": round(float(np.hypot(r_ohm, x_ohm)), 5),
                    "separated": True,
                    "pf_collinearity": round(collinear, 3),
                })
                if float(np.hypot(r_ohm, x_ohm)) * i_max < _VOLTAGE_RESOLUTION_V:
                    out.update({"identifiable": False, "reason": (
                        "The fitted impedance accounts for less than the "
                        f"meter's {_VOLTAGE_RESOLUTION_V} V resolution across "
                        "the whole load range.")})
                return out
            out["pf_collinearity"] = (round(collinear, 3)
                                      if np.isfinite(collinear) else None)

    # The median of the per-step ratios: robust to the steps where another
    # customer's load moved at the same moment as this one's.
    ratios = dv[steps] / di[steps]
    z_eff = float(np.median(ratios if rises_with_current else -ratios))
    out.update({"z_ohm": round(z_eff, 5), "separated": False})
    if z_eff * i_max < _VOLTAGE_RESOLUTION_V:
        out.update({"identifiable": False, "reason": (
            f"Across the whole {_m(i_max, '.1f', ' A')} load range the voltage change "
            f"attributable to this impedance is {_m(z_eff * i_max, '.2f', ' V')}, below "
            f"the meter's {_VOLTAGE_RESOLUTION_V} V reporting resolution. "
            "Nothing measurable is there to report.")})
    return out


def check_source_impedance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """Impedance from the source to the meter, per phase, measured and expected.

    The measurement stands on its own: the per-phase comparison and the
    neutral-to-earth rise are findings without any picked conductor, because
    both are read against the service's own other phases rather than against a
    table. The expected value, when the transformer and conductor are known,
    turns the total into a stated pass or fail.
    """
    pf = df["power_factor"] if "power_factor" in df.columns else None
    phases: Dict[str, dict] = {}
    for ph in ("a", "b", "c"):
        v_col, i_col = f"voltage_{ph}", f"current_{ph}"
        if v_col not in df.columns or i_col not in df.columns:
            continue
        fit = _impedance_fit(df[v_col], df[i_col], pf)
        if fit is not None:
            phases[ph] = fit

    if not phases:
        return {"available": False,
                "reason": ("No phase has both a voltage and a current channel "
                           "with enough intervals to fit a voltage drop.")}

    measured = {ph: f for ph, f in phases.items()
                if f.get("identifiable") and f.get("z_ohm") is not None}
    out: dict = {
        "available": True,
        "phases": phases,
        "power_factor_used": pf is not None,
    }

    # ── the neutral connection, measured directly ────────────────────────────
    if "voltage_neutral" in df.columns and "current_neutral" in df.columns:
        n_fit = _impedance_fit(df["voltage_neutral"], df["current_neutral"],
                               None, rises_with_current=True)
        if n_fit:
            if n_fit.get("identifiable"):
                r_n = float(n_fit["z_ohm"])
                i_n_peak = float(df["current_neutral"].max())
                rise = r_n * i_n_peak
                n_fit.update({
                    "r_ohm": round(r_n, 5),
                    "i_peak_a": round(i_n_peak, 2),
                    "rise_at_peak_v": round(rise, 2),
                    "elevated": bool(rise >= _NEUTRAL_RISE_VOLTS and r_n > 0),
                })
            out["neutral"] = n_fit

    # ── one phase against the others ─────────────────────────────────────────
    # Compared like for like: the resistance where the fit separated it, the
    # effective magnitude where it did not, never a mix of the two.
    key = ("r_ohm" if all(f.get("separated") for f in measured.values())
           else "z_ohm")
    resistances = {ph: f[key] for ph, f in measured.items()
                   if f.get(key) is not None and f[key] > 0}
    if len(resistances) >= 2:
        worst = max(resistances, key=resistances.get)
        best = min(resistances, key=resistances.get)
        i_peak = max(float(df[f"current_{ph}"].max()) for ph in resistances)
        excess_v = (resistances[worst] - resistances[best]) * i_peak
        ratio = (resistances[worst] / resistances[best]
                 if resistances[best] > 0 else None)
        out["asymmetry"] = {
            "worst_phase": worst.upper(),
            "best_phase": best.upper(),
            "ratio": round(ratio, 2) if ratio else None,
            "excess_v_at_peak": round(excess_v, 2),
            "flagged": bool(ratio and ratio >= _PHASE_ASYMMETRY_RATIO
                            and excess_v >= _PHASE_ASYMMETRY_VOLTS),
        }

    # ── against what the picked service ought to give ────────────────────────
    expected = expected_service_impedance(thresh)
    out["expected"] = expected
    z_values = [f["z_ohm"] for f in measured.values()
                if f.get("z_ohm") is not None and f["z_ohm"] > 0]
    if z_values:
        out["measured_z_ohm"] = round(float(np.median(z_values)), 5)
    if expected.get("available") and z_values and expected.get("total_ohm"):
        z_meas = float(np.median(z_values))
        ratio = z_meas / float(expected["total_ohm"])
        i_peak = max(float(df[f"current_{ph}"].max()) for ph in measured)
        out["comparison"] = {
            "measured_ohm": round(z_meas, 5),
            "expected_ohm": round(float(expected["total_ohm"]), 5),
            "ratio": round(ratio, 2),
            "excess_v_at_peak": round(
                (z_meas - float(expected["total_ohm"])) * i_peak, 2),
            "i_peak_a": round(i_peak, 2),
            "verdict": ("high" if ratio > _IMPEDANCE_HIGH
                        else "elevated" if ratio > _IMPEDANCE_ELEVATED
                        else "below_expected" if ratio < 0.5
                        else "consistent"),
        }

    out["overall"] = _impedance_overall(out)
    return out


def _impedance_overall(result: dict) -> str:
    """The headline, worst-first, so a flag cannot hide behind a pass."""
    comparison = (result.get("comparison") or {}).get("verdict")
    if comparison == "high" or (result.get("asymmetry") or {}).get("flagged"):
        return "high_impedance_suspected"
    if comparison == "elevated" or (result.get("neutral") or {}).get("elevated"):
        return "elevated"
    if comparison == "consistent":
        return "consistent_with_expected"
    if result.get("measured_z_ohm") is not None:
        return "measured_only"
    return "not_measurable"


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


#: Smallest power-factor correction worth recommending as an action.  Below
#: this the required kVAR is under the smallest practical switched capacitor
#: step, so an "install correction" line is bulk rather than advice.
_MIN_ACTIONABLE_KVAR = 5.0

#: Readable service names for the signature findings.  The tariff codes mean
#: nothing to a customer reading the report.
_CLASS_SERVICE_NAME = {
    "r":  "residential",
    "c":  "small commercial",
    "sg": "commercial / industrial secondary",
    "pg": "commercial / industrial primary",
}


def _detect_harmonic_signature(df: pd.DataFrame, il_amps: float,
                               significance: Optional[dict] = None,
                               customer_class: Optional[str] = None) -> List[dict]:
    """
    Score each entry in _LOAD_SIGNATURES against the measured harmonic spectrum
    using cosine similarity, then return the top matches as finding dicts.

    Cosine similarity measures spectral *shape* — the absolute THD level does not
    affect which load type wins.  A variability modifier adjusts scores up/down based
    on whether the measured H5 inter-interval CV matches the load type's expected
    stability (steady-state vs. intermittent).

    The candidate set is restricted to load types that are plausible for the
    service class.  Cosine similarity will always return a nearest neighbour, so
    an unrestricted library named an arc furnace as the best match for a house —
    a shape match against equipment that cannot be there, which discredits the
    rest of the report.  A house is scored against air conditioning, EV charging
    and rooftop PV; an arc furnace is only offered to a primary-metered service.
    """
    _ORDERS = [3, 5, 7, 9, 11, 13]
    # Matching a load signature is a statement about spectral shape, so it is
    # only made when the shape is measurable -- otherwise the library always
    # returns its nearest neighbour to noise, with a confident-looking score.
    if significance is not None and not significance["usable"]:
        log.info("Harmonic load-signature matching skipped: %s",
                 significance["reason"])
        return []
    scope = df[significance["loaded"]] if (
        significance and significance.get("loaded") is not None) else df
    h_mean = _harmonic_means(scope, _ORDERS)
    if len(h_mean) < 3:
        return []

    measured = np.array([h_mean.get(h, 0.0) for h in _ORDERS], dtype=float)
    if np.linalg.norm(measured) == 0:
        return []

    # H5 inter-interval variability (relative variability (coefficient of variation))
    h5_cols = [f"h5_current_{p}" for p in "abc" if f"h5_current_{p}" in scope.columns]
    h5_cv = 0.0
    if h5_cols:
        s = scope[h5_cols].values.flatten()
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

    # Restrict the library to equipment that can plausibly exist at this class
    # of service before scoring anything.
    candidates = [s for s in _LOAD_SIGNATURES
                  if not customer_class
                  or customer_class in s.get("classes", set())]
    if not candidates:
        log.info("No load signatures apply to customer class %r; "
                 "signature matching skipped.", customer_class)
        return []
    if customer_class:
        log.info("Load-signature matching against %d of %d reference loads "
                 "applicable to customer class %r.",
                 len(candidates), len(_LOAD_SIGNATURES), customer_class)

    # Score each signature
    scored = []
    for sig in candidates:
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
    class_note = ""
    if customer_class:
        class_note = (f" Scored against the {len(candidates)} load types "
                      f"plausible for a {_CLASS_SERVICE_NAME.get(customer_class, customer_class)} "
                      f"service; equipment that does not belong at this class of "
                      f"service was not considered.")

    measured_note = (
        f"Measured spectrum: {spectrum_str}. "
        f"H5/H7={_m(h5_h7, '.2f')}, H3/H5={_m(h3_h5, '.2f')}, H5 variability (CV)={_m(h5_cv, '.2f')}."
    )

    top_score, top_sig = scored[0]

    # ── Is the spectrum recognisable at all? ─────────────────────────────────
    # Nearest-neighbour scoring always returns something.  Against 20,000
    # random decaying spectra this library's top score had a median of 0.87 and
    # cleared 0.95 nearly a third of the time, so a match that merely "scores
    # well" is not evidence of anything.  Below the floor the report says so
    # rather than naming the nearest entry.
    if top_score < SIGNATURE_ABSOLUTE_FLOOR:
        log.info("No load signature recognised: best score %.3f is below the "
                 "%.2f floor.", top_score, SIGNATURE_ABSOLUTE_FLOOR)
        return [{
            "category":       "harmonics",
            "severity":       "info",
            # Routed to the report's last appendix, not the assessment.
            "experimental":   True,
            "title":          "No recognised load signature",
            # Informational: there is nothing here to act on, so this
            # must not become a line item in Recommended Actions.
            "no_action":      True,
            "finding":        (
                f"The measured harmonic spectrum does not correspond to any "
                f"reference load type closely enough to name one (best score "
                f"{_m(top_score, '.0%')}, below the {SIGNATURE_ABSOLUTE_FLOOR:.0%} "
                f"threshold required to report a match). {measured_note}"
            ),
            "cause": (
                "This usually means the service carries a mix of loads whose "
                "combined spectrum matches no single reference type, or that "
                "distortion is low enough that spectrum shape is dominated by "
                "measurement noise. It is not itself a problem finding."
            ),
            "origin_evidence": (
                "Reported because the analysis ran and found nothing it could "
                "name, which is different from the analysis not running."
            ),
            "recommendation": (
                "Interpret the measured spectrum directly, or identify loads by "
                "site survey. If a specific load type is suspected, a recording "
                "taken while that load is cycled on and off will separate its "
                "contribution far more reliably than spectrum matching."
            ),
            "confidence": "low",
            "evidence": {
                "similarity":       round(top_score, 3),
                "floor":            SIGNATURE_ABSOLUTE_FLOOR,
                "nearest_rejected": top_sig["id"],
                "h5_h7_ratio":      round(h5_h7, 2),
                "h3_h5_ratio":      round(h3_h5, 2),
                "h5_cv":            round(h5_cv, 3),
                **{f"h{h}_pct_il": h_pcts[h] for h in _ORDERS},
            },
        }]

    # ── Does it separate from the next *family*? ─────────────────────────────
    # Within a family the entries are the same topology and cannot be told
    # apart, so only the gap to a different family carries information.
    top_family = top_sig.get("family")
    next_family_score = next(
        (sc for sc, sg in scored if sg.get("family") != top_family), None)
    family_sep = (top_score - next_family_score
                  if next_family_score is not None else 1.0)

    if family_sep < SIGNATURE_FAMILY_SEPARATION:
        other = next(sg for sc, sg in scored if sg.get("family") != top_family)
        log.info("Load signature ambiguous across families: %s vs %s separated "
                 "by %.3f.", top_family, other.get("family"), family_sep)
        return [{
            "category":       "harmonics",
            "severity":       "info",
            # Routed to the report's last appendix, not the assessment.
            "experimental":   True,
            "title":          "No recognised load signature",
            # Informational: there is nothing here to act on, so this
            # must not become a line item in Recommended Actions.
            "no_action":      True,
            "finding":        (
                f"The measured spectrum sits between two unrelated load "
                f"families and cannot be assigned to either: "
                f"\"{LOAD_FAMILY_LABEL.get(top_family, top_family)}\" "
                f"({_m(top_score, '.0%')}) and "
                f"\"{LOAD_FAMILY_LABEL.get(other.get('family'), other.get('family'))}\" "
                f"({_m(next_family_score, '.0%')}) are separated by only "
                f"{_m(family_sep * 100, '.1f')} points. {measured_note}"
            ),
            "cause": (
                "A spectrum that matches two unrelated topologies equally well "
                "is usually a blend of several loads rather than one dominant "
                "source."
            ),
            "origin_evidence": (
                "Reported rather than resolved: naming either family would "
                "present a coin toss as a conclusion."
            ),
            "recommendation": (
                "Identify loads by site survey, or record again while "
                "individual loads are switched, rather than relying on spectrum "
                "shape alone."
            ),
            "confidence": "low",
            "evidence": {
                "similarity":     round(top_score, 3),
                "family_a":       top_family,
                "family_b":       other.get("family"),
                "family_separation": round(family_sep, 3),
                **{f"h{h}_pct_il": h_pcts[h] for h in _ORDERS},
            },
        }]

    # ── The family is as far as a single-point measurement goes ─────────────
    # Naming the individual entry was measured against synthetic mixtures of
    # two or three library loads sharing one service: on secondary service it
    # named a load that was not present 45% of the time, on primary 32%.  See
    # test_pq.py::TestLoadSignatureMixtures.  The meter sees the sum of
    # everything behind it, so a spectrum sitting nearest one entry is not
    # evidence that entry is drawing the current -- a blend of two loads lands
    # near a third entry that neither parent resembles.  Only the family, which
    # is a statement about converter topology, survives that.  Confidence is
    # capped at medium for the same reason: nothing measured here supports a
    # high-confidence equipment claim.
    same_family = [(sc, sg) for sc, sg in scored
                   if sg.get("family") == top_family]
    member_sep = (same_family[0][0] - same_family[1][0]
                  if len(same_family) > 1 else 1.0)

    title = (f"Possible load family: "
             f"{LOAD_FAMILY_LABEL.get(top_family, top_family)}")
    if len(same_family) == 1:
        detail = (f" This family has one reference entry "
                  f"(\"{top_sig['title']}\") for this class of service, so the "
                  f"family and that entry coincide here. It remains a match on "
                  f"spectral shape, not an identification of the equipment "
                  f"present.")
    else:
        peers = ", ".join(f'"{sg["title"]}"' for _, sg in same_family[:3])
        near = ("scoring within "
                f"{member_sep * 100:.1f} points of each other"
                if member_sep < SIGNATURE_MEMBER_SEPARATION
                else "and a service carrying more than one load routinely "
                     "scores nearest an entry it does not contain")
        detail = (f" The individual load type is not reported -- {peers} share "
                  f"this topology, {near} -- so the family is reported rather "
                  f"than a specific piece of equipment.")
    conf = "medium" if family_sep >= 0.15 else "low"

    findings.append({
        "category":       "harmonics",
        "severity":       "info",
        "experimental":   True,
        "title":          title,
        "finding":        (
            f"Spectral similarity {_m(top_score, '.0%')}, separated from the next load "
            f"family by {_m(family_sep * 100, '.1f')} points. {measured_note}"
            f"{detail}{class_note}"
        ),
        "cause":          top_sig["cause"],
        "origin_evidence": (
            "Load signatures describe equipment downstream of the meter. "
            "This is a match on spectral shape against a reference library, "
            "not a measurement of direction and not an identification of the "
            "equipment present. Treat it as a hypothesis to confirm on site, "
            "not as a finding in its own right."
        ),
        # Family-level, not the nearest entry's: the finding declines to
        # name the member, and member advice contradicts itself inside a
        # family. See LOAD_FAMILY_RECOMMENDATION.
        "recommendation": LOAD_FAMILY_RECOMMENDATION.get(
            top_family, top_sig["recommendation"]),
        "confidence":     conf,
        "evidence":       {
            "similarity":        round(top_score, 3),
            "signature_id":      top_sig["id"],   # nearest entry, not reported
            "family":            top_family,
            "family_separation": round(family_sep, 3),
            "member_separation": round(member_sep, 3),
            "resolved_to_member": False,
            "rank":              1,
            "h5_h7_ratio":       round(h5_h7, 2),
            "h3_h5_ratio":       round(h3_h5, 2),
            "h5_cv":             round(h5_cv, 3),
            **{f"h{h}_pct_il": h_pcts[h] for h in _ORDERS},
        },
    })

    return findings


def check_itic(event_result: dict, thresh: Thresholds) -> dict:
    """Evaluate detected sag/swell events against the ITIC (CBEMA) voltage
    tolerance envelope.

    Each event is a (duration, magnitude) point; points outside the envelope
    (below the lower boundary or above the upper boundary) are disruptions the
    ITI curve says IT equipment is not required to tolerate.  Requires
    event-level durations, which come from adaptive (cycle-level) or waveform
    records — 5-minute interval averages cannot resolve event duration.
    """
    ev = event_result.get("events")
    if ev is None or len(ev) == 0:
        return {
            "available":    True,
            "n_events":     0,
            "n_violations": 0,
            "overall_pass": True,
            "note":         "No voltage sag/swell events detected during the recording.",
            "violations":   [],
        }

    vs = ev[ev["type"].isin(["voltage_sag", "voltage_swell"])].copy()
    if (vs.empty or "duration_ms" not in vs.columns
            or not vs["duration_ms"].notna().any()):
        if vs.empty:
            return {
                "available":    True,
                "n_events":     0,
                "n_violations": 0,
                "overall_pass": True,
                "note":         "No voltage sag/swell events detected during the recording.",
                "violations":   [],
            }
        return {
            "available": False,
            "note": ("Sag/swell events were detected but event durations are not "
                     "available from this recording's data (requires cycle-level "
                     "adaptive or waveform records)."),
        }

    vs = vs.dropna(subset=["duration_ms", "value_v"])
    dur = vs["duration_ms"].to_numpy(dtype=float)
    pct = vs["value_v"].to_numpy(dtype=float) / thresh.nominal_voltage * 100.0
    viol_mask = (pct > _itic_upper_v(dur)) | (pct < _itic_lower_v(dur))

    violations = []
    for (_, row), is_viol, p in zip(vs.iterrows(), viol_mask, pct):
        if not is_viol:
            continue
        violations.append({
            "timestamp":   row.get("timestamp"),
            "type":        row["type"],
            "phase":       row.get("phase"),
            "value_v":     round(float(row["value_v"]), 1),
            "pct_nominal": round(float(p), 1),
            "duration_ms": round(float(row["duration_ms"]), 1),
        })
    violations.sort(key=lambda v: abs(v["pct_nominal"] - 100), reverse=True)

    return {
        "available":    True,
        "n_events":     int(len(vs)),
        "n_violations": int(viol_mask.sum()),
        "overall_pass": bool(viol_mask.sum() == 0),
        "worst":        violations[0] if violations else None,
        "violations":   violations[:20],
    }


#: PSCo Sheet R123: the spread between the highest and lowest phase at which
#: the Company may recompute billing demand from the worst phase.
_R123_SPREAD_PCT = 15.0

#: The power factor the same clause converts kVA to kW at, whatever the
#: customer's actual one is. The figure matches Sheet R73's "rates contemplate
#: ... not less than ninety percent (90%) lagging".
_R123_ASSUMED_PF = 0.90


def check_billing_demand_imbalance(df: pd.DataFrame, thresh: Thresholds) -> dict:
    """PSCo Sheet R123's phase-spread provision, evaluated at the peak interval.

    Not a power quality finding and not a compliance test -- nothing here is
    being violated. It is a cost the customer may already be paying without
    knowing why, and which balancing the panel would remove.

    The clause reads: where the load on any one phase exceeds the load on any
    other by more than fifteen percent, the Company *may* take as the billing
    demand "the three-phase equivalent of the maximum kilovolt-amperes in any
    phase adjusted to a ninety percent (90%) Power Factor" -- that is,
    3 x the worst phase's kVA x 0.90.

    Only the peak matters. Billing demand is set by the maximum demand interval
    of the month, so imbalance at 3 a.m. costs nothing and imbalance at the
    peak costs the whole uplift. Averaging the spread across a recording would
    answer a question nobody is billed on, which is why this looks only at the
    interval that set the peak, and reports the recording-wide count separately
    as context rather than as the finding.

    The uplift works out to the worst phase over the mean phase, so the trigger
    and the cost are measured differently: 100/100/120 A trips a 20% spread but
    carries a 12.5% uplift.
    """
    result: dict = {"available": False, "applies": False}

    geometry = service_geometry(thresh, df.columns)
    if geometry != "three-phase":
        result["note"] = ("Sheet R123's phase provision applies to three-phase "
                          "service.")
        return result
    # Schedule C has no demand charge -- only service, facility and energy --
    # so there is no billing demand for the clause to recompute. Raising it
    # with a customer who cannot be billed on it would be a false alarm.
    if thresh.customer_class not in ("sg", "pg"):
        result["note"] = ("This schedule carries no demand charge, so there is "
                          "no billing demand for Sheet R123 to recompute.")
        return result

    i_cols = [c for c in ("current_a", "current_b", "current_c") if c in df.columns]
    if len(i_cols) < 3:
        result["note"] = "Three-phase currents are needed to evaluate the spread."
        return result

    # The peak demand interval, on the same apparent-power basis check_demand
    # uses, so the two sections cannot disagree about when the peak was.
    if "power_real" in df.columns and "power_reactive" in df.columns:
        apparent = np.sqrt(df["power_real"] ** 2 + df["power_reactive"] ** 2)
    elif "power_real" in df.columns and "power_factor" in df.columns:
        apparent = df["power_real"] / df["power_factor"].replace(0, np.nan)
    else:
        apparent = df[i_cols].sum(axis=1) * thresh.nominal_voltage
    apparent = apparent.dropna()
    if apparent.empty:
        result["note"] = "No demand data to locate the peak interval."
        return result

    peak_ts = apparent.abs().idxmax()
    phases = df.loc[peak_ts, i_cols].astype(float)
    if phases.isna().any() or float(phases.min()) <= 0:
        result["note"] = "The peak interval has no usable per-phase current."
        return result

    hi, lo = float(phases.max()), float(phases.min())
    spread = (hi - lo) / lo * 100.0
    mean_a = float(phases.mean())

    # How often the spread is exceeded at all. Context, not the finding: a
    # different month's peak may land on one of these intervals.
    all_hi, all_lo = df[i_cols].max(axis=1), df[i_cols].min(axis=1)
    valid = all_lo > 0
    over = ((all_hi - all_lo) / all_lo * 100.0)[valid] > _R123_SPREAD_PCT

    v_ln = thresh.nominal_voltage
    measured_kva = float(phases.sum()) * v_ln / 1000.0
    clause_kva   = 3.0 * hi * v_ln / 1000.0

    result.update({
        "available":       True,
        "applies":         spread > _R123_SPREAD_PCT,
        "peak_timestamp":  peak_ts,
        "phase_amps":      {c[-1]: round(float(phases[c]), 1) for c in i_cols},
        "spread_pct":      round(spread, 1),
        "threshold_pct":   _R123_SPREAD_PCT,
        "worst_phase":     str(phases.idxmax())[-1],
        "measured_kva":    round(measured_kva, 1),
        "measured_kw":     round(measured_kva * _R123_ASSUMED_PF, 1),
        "clause_kva":      round(clause_kva, 1),
        "clause_kw":       round(clause_kva * _R123_ASSUMED_PF, 1),
        # The uplift is the worst phase over the mean phase, which is why the
        # trigger (max against min) and the cost do not match.
        "uplift":          round(hi / mean_a, 3),
        "uplift_pct":      round((hi / mean_a - 1.0) * 100.0, 1),
        "assumed_pf":      _R123_ASSUMED_PF,
        "intervals_over":  int(over.sum()),
        "intervals_total": int(valid.sum()),
        "caveats": [
            "Billing demand is set by the peak demand interval, so this is "
            "evaluated there rather than averaged over the recording. A month "
            "whose peak falls on a different interval may read differently.",
            "This recording's intervals may not be the interval the bill is "
            "computed on, and the clause is discretionary \u2014 it says the "
            "Company may take billing demand this way, not that it does.",
        ],
    })
    return result


def check_ride_through(event_result: dict, thresh: Thresholds) -> dict:
    """Measured voltage events against IEEE 1547-2018 Clause 6.4.2.

    The counterpart of `check_itic` for a service that generates, and it asks
    the opposite question.  ITIC asks whether a customer's equipment should
    have survived a dip.  Clause 6.4.2 asks whether the *plant* was required to
    stay on through it, and 6.4.2.1 makes failing to do so the plant's
    non-compliance rather than the system's.  So this does not produce a
    pass/fail against the utility: it says, for each event, which region of the
    table the plant was in and what that region required of it.

    Every event is reported rather than only the exceedances, because the
    useful finding here is usually the ordinary-looking one -- a shallow dip
    inside the continuous region that the plant tripped on anyway.

    Two limits are worth stating with the result.  The tables' durations are
    cumulative within a disturbance, and consecutive disturbances have their
    own requirement in Table 17 that this does not evaluate; and the standard
    measures the phase with the least voltage against the nominal, so a
    per-phase event list is what it wants and what it gets here.
    """
    result: dict = {
        "available": False, "category": thresh.der_category,
        "events": [], "counts": {}, "n_events": 0,
        "n_required_to_ride_through": 0, "n_beyond_requirement": 0,
    }

    if not exports_power(thresh):
        result["note"] = ("Clause 6 applies to a distributed energy resource. "
                          "This service has none.")
        return result

    if not thresh.der_category:
        result["note"] = (
            "IEEE 1547-2018 Clause 6.4.2 grades voltage disturbances against "
            "one of three abnormal operating performance categories, and the "
            "standard leaves the choice to the Area EPS operator rather than "
            "to the recording: the ride-through a plant owes at 0.75 p.u. is "
            "0.9 s under Category I and 20 s under Category III. Enter the "
            "category from the interconnection agreement to assess it."
        )
        return result

    ev = (event_result or {}).get("events")
    if ev is None or len(ev) == 0:
        result.update({
            "available": True,
            "note": "No voltage sag or swell events were detected in this recording.",
        })
        return result

    vs = ev[ev["type"].isin(["voltage_sag", "voltage_swell"])].copy()
    if vs.empty:
        result.update({
            "available": True,
            "note": "No voltage sag or swell events were detected in this recording.",
        })
        return result
    if "duration_ms" not in vs.columns or not vs["duration_ms"].notna().any():
        result["note"] = (
            "Voltage events were detected, but Clause 6.4.2 is a voltage "
            "against duration requirement and this recording carries no event "
            "durations. Cycle-level adaptive or waveform records are needed to "
            "resolve them."
        )
        return result

    vs = vs.dropna(subset=["duration_ms", "value_v"])
    counts: Dict[str, int] = {}
    events: List[dict] = []
    for _, row in vs.iterrows():
        v_pu = float(row["value_v"]) / thresh.nominal_voltage
        seconds = float(row["duration_ms"]) / 1000.0
        region = ride_through_region(thresh.der_category, v_pu)
        if region is None:
            continue
        minimum = region["min_ride_s"]

        # What the region required, and whether this event stayed inside it.
        if region["mode"] == "continuous":
            required, within = True, True
        elif minimum is None:                      # cease-to-energize row
            required, within = False, False
        else:
            required = True
            within = seconds <= minimum
        counts[region["mode"]] = counts.get(region["mode"], 0) + 1
        events.append({
            "timestamp":   row.get("timestamp"),
            "type":        row["type"],
            "phase":       row.get("phase"),
            "v_pu":        round(v_pu, 3),
            "pct_nominal": round(v_pu * 100.0, 1),
            "duration_s":  round(seconds, 3),
            "mode":        region["mode"],
            "region":      region["label"],
            "min_ride_s":  (None if minimum is None else
                            (None if minimum == math.inf else round(minimum, 3))),
            "continuous":  region["mode"] == "continuous",
            # True where the plant was obliged not to trip on this event.
            "must_not_trip": required and within,
        })

    # Deepest first: the one an operator will ask about is the worst one.
    events.sort(key=lambda e: abs(e["pct_nominal"] - 100), reverse=True)
    obliged = [e for e in events if e["must_not_trip"]]
    result.update({
        "available": True,
        "events": events,
        "counts": counts,
        "n_events": len(events),
        "n_required_to_ride_through": len(obliged),
        "n_beyond_requirement": len(events) - len(obliged),
        "worst": events[0] if events else None,
        "worst_obliged": obliged[0] if obliged else None,
        "caveats": [
            "The durations in Tables 14 to 16 are cumulative within a "
            "disturbance, and consecutive disturbances carry their own "
            "requirement in Table 17. Neither is evaluated here; each event is "
            "taken on its own.",
            "Clause 6.4.2 measures the phase with the least voltage against "
            "nominal, so events are listed per phase rather than combined.",
        ],
    })
    return result


def check_frequency_ride_through(ds: PQDataset, thresh: Thresholds) -> dict:
    """Measured frequency against IEEE 1547-2018 Clause 6.5.2 and Table 19.

    The frequency counterpart of `check_ride_through`, and it needs a different
    kind of data.  Table 19 is a minutes-long requirement -- 299 s minimum
    times, continuous operation indefinitely -- so unlike the voltage side it
    is not defeated by a coarse record outright.  What defeats it is averaging:
    twenty seconds at 57.5 Hz vanishes inside a five-minute mean of 60.0, and
    the standard counts cumulative time outside the band, which a mean cannot
    reconstruct.

    So which record this ran on is part of the answer and is reported with it.
    The variable-rate record carries frequency sample by sample and supports
    the clause properly; interval averages can show the averages never left the
    continuous band, which is a weaker statement and is labelled as one rather
    than being allowed to read as a pass.
    """
    result: dict = {
        "available": False, "source": None, "excursions": [],
        "n_excursions": 0, "n_required_to_ride_through": 0,
        "n_beyond_requirement": 0,
    }

    if not exports_power(thresh):
        result["note"] = ("Clause 6.5 applies to a distributed energy resource. "
                          "This service has none.")
        return result

    adf = ds.adaptive_df
    series = None
    if adf is not None and "adap_freq" in getattr(adf, "columns", []):
        series = adf["adap_freq"].dropna()
        source = "variable-rate"
    if series is None or len(series) < 2:
        if "frequency" not in ds.df.columns or ds.df["frequency"].notna().sum() == 0:
            result["note"] = "No frequency channel in this recording."
            return result
        series = ds.df["frequency"].dropna()
        source = "interval-average"
    result["source"] = source

    lo, hi = 58.8, 61.2
    outside = (series < lo) | (series > hi)

    if source == "interval-average":
        # An average cannot rule an excursion out, so it does not get to say
        # one did not happen -- only that nothing survived the averaging.
        result.update({
            "available": True,
            "assessable": False,
            "n_intervals_outside": int(outside.sum()),
            "min_hz": round(float(series.min()), 3),
            "max_hz": round(float(series.max()), 3),
            "note": (
                "This recording carries frequency only as interval averages. "
                "IEEE 1547 Clause 6.5.2 counts cumulative time outside "
                f"{lo}-{hi} Hz within a ten-minute window, which an average "
                "cannot reconstruct: a twenty-second excursion to 57.5 Hz "
                "leaves a five-minute mean sitting at 60.0. The averages "
                + ("stayed within the continuous operation band, which is "
                   "consistent with compliance but does not establish it."
                   if not outside.any() else
                   f"left the continuous band in {int(outside.sum())} "
                   "interval(s), which is enough to warrant a variable-rate "
                   "recording but not enough to grade against Table 19.")
            ),
        })
        return result

    # Variable-rate: the clause can be applied as written.
    idx = series.index
    step = pd.Series(idx).diff().dt.total_seconds()
    step.iloc[0] = step.iloc[1] if len(step) > 1 else 1.0
    step = step.to_numpy()

    # Cumulative seconds outside the band in any ten-minute window, counted
    # separately below and above: 6.5.2.3.1 and 6.5.2.4.1 each say "cumulative
    # duration below 58.8 Hz" / "above 61.2 Hz", not the two combined.
    def _rolling_seconds(mask) -> np.ndarray:
        secs = pd.Series(np.where(mask.to_numpy(), step, 0.0), index=idx)
        return secs.rolling(f"{int(FREQ_CUMULATIVE_WINDOW_S)}s").sum().to_numpy()

    below, above = series < lo, series > hi
    roll_below, roll_above = _rolling_seconds(below), _rolling_seconds(above)

    # Contiguous runs outside the band become excursions.
    excursions: List[dict] = []
    run_start = None
    arr, values = outside.to_numpy(), series.to_numpy()
    for i, flag in enumerate(list(arr) + [False]):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            sl = slice(run_start, i)
            worst_i = (int(np.argmin(values[sl])) if below.to_numpy()[run_start]
                       else int(np.argmax(values[sl]))) + run_start
            hz = float(values[worst_i])
            region = frequency_ride_through_region(hz)
            direction = "below" if hz < lo else "above"
            cumulative = float((roll_below if direction == "below"
                                else roll_above)[max(sl.stop - 1, 0)])
            within_allowance = cumulative < FREQ_CUMULATIVE_ALLOWANCE_S
            must_not_trip = region["mode"] == "mandatory" and within_allowance
            excursions.append({
                "timestamp":     idx[run_start],
                "direction":     direction,
                "extreme_hz":    round(hz, 3),
                "duration_s":    round(float(step[sl].sum()), 3),
                "cumulative_s":  round(cumulative, 1),
                "within_allowance": within_allowance,
                "mode":          region["mode"],
                "region":        region["label"],
                "must_not_trip": must_not_trip,
            })
            run_start = None

    excursions.sort(key=lambda e: abs(e["extreme_hz"] - 60.0), reverse=True)
    obliged = [e for e in excursions if e["must_not_trip"]]

    # 6.5.2.2 puts a second condition on the continuous band that frequency
    # alone does not establish.
    vf = _voltage_over_frequency(ds, thresh)

    result.update({
        "available":   True,
        "assessable":  True,
        "excursions":  excursions,
        "n_excursions": len(excursions),
        "n_required_to_ride_through": len(obliged),
        "n_beyond_requirement": len(excursions) - len(obliged),
        "worst":       excursions[0] if excursions else None,
        "min_hz":      round(float(series.min()), 3),
        "max_hz":      round(float(series.max()), 3),
        "v_over_f":    vf,
        "active_power_capability":
            FREQ_ACTIVE_POWER_CAPABILITY.get(thresh.der_category or ""),
        "note": (
            "Measured from the variable-rate record, so cumulative time "
            "outside the continuous band is counted as Clause 6.5.2 defines "
            "it." if excursions else
            "Frequency stayed within the 58.8-61.2 Hz continuous operation "
            "band for the whole recording, measured from the variable-rate "
            "record rather than from averages."),
        "caveats": [
            "Table 19 is the same for all three performance categories; the "
            "category changes only how much active power the plant must hold "
            "up during the excursion, per Table 20.",
            "The 299 s in Table 19 is a condition on the requirement, not a "
            "limit on the plant: past that much cumulative time outside the "
            "band in a ten-minute window, the obligation to ride through "
            "lapses and tripping is permitted.",
        ],
    })
    return result


def _voltage_over_frequency(ds: PQDataset, thresh: Thresholds) -> Optional[dict]:
    """The per-unit V/f ratio 6.5.2.2 attaches to continuous operation.

    Continuous operation is not frequency alone: the clause requires the ratio
    to stay at or below 1.1 as well, and a report that checked only the band
    would be claiming more than it measured.
    """
    v_cols = [c for c in ("voltage_a", "voltage_b", "voltage_c")
              if c in ds.df.columns]
    if not v_cols or "frequency" not in ds.df.columns or not thresh.nominal_voltage:
        return None
    v_pu = ds.df[v_cols].max(axis=1) / thresh.nominal_voltage
    f_pu = ds.df["frequency"] / thresh.frequency_nominal
    ratio = (v_pu / f_pu).dropna()
    if ratio.empty:
        return None
    worst = float(ratio.max())
    return {
        "max_ratio": round(worst, 3),
        "limit":     FREQ_CONTINUOUS_MAX_V_OVER_F,
        "within":    worst <= FREQ_CONTINUOUS_MAX_V_OVER_F,
    }


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
    il_amps = demand_current_il(df, thresh)[0] or 0.0
    geometry = service_geometry(thresh, df.columns)

    # ── Harmonic signature detection ──────────────────────────────────────────
    if il_amps > 0 and any(f"h5_current_{p}" in df.columns for p in "abc"):
        findings.extend(_detect_harmonic_signature(
            df, il_amps, harmonic_spectrum_significance(df, thresh),
            customer_class=thresh.customer_class))

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
                "finding":        (f"Maximum TDD is {_m(tdd_max, '.2f', '%')} against a {tdd_lim:.1f}% limit "
                                   f"({_m(utilization*100, '.0f', '%')} of limit consumed). "
                                   "No violation, but limited margin for load growth."),
                "cause":          ("Current harmonic load is close to the IEEE 519-2022 class limit. "
                                   "Additional VFDs, rectifiers, or other nonlinear loads could "
                                   "push TDD over the limit."),
                "origin_evidence": (
                    "Harmonic current is injected by load while the resulting "
                    "voltage distortion depends on supply impedance; interval "
                    "data alone does not separate the two."
                ),
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
            elif geometry == "split-phase":
                # A house has no A/B/C phases to redistribute across, and the
                # neutral here carries the difference between two legs rather
                # than a three-phase residual.
                cause_text = (
                    f"Neutral current ({in_pct:.1f}% of phase average) is the "
                    f"difference between the two 120 V legs (leg difference "
                    f"mean {ci_mean:.1f}%). On a 120/240 service the legs are "
                    "180 degrees apart, so the neutral carries whatever the "
                    "legs do not share. Unequal loading of the legs is the "
                    "ordinary cause and is not a fault in itself."
                )
                rec_text = (
                    "Confirm the neutral is sound before reading this as load "
                    "distribution — the neutral health assessment bears on "
                    "that. Where the neutral is intact, moving 120 V loads "
                    "between the two legs at the panel reduces both the "
                    "neutral current and the voltage difference between legs."
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
                "finding":        (f"Neutral current averages {_m(in_pct, '.1f', '%')} of phase current, "
                                   f"peaking at {_m(in_max, '.1f', '%')} (see Voltage and Current Imbalance "
                                   f"above for measured amps)."),
                "cause":          cause_text,
                "origin_evidence": (
                    "Neutral current is the difference between the two legs, "
                    "so it follows from how load is split between them and "
                    "from the harmonic content each leg draws, downstream of "
                    "the meter."
                    if geometry == "split-phase" else
                    "Neutral current is the sum of the phase currents, so it "
                    "follows from how load is distributed across the legs and "
                    "from triplen harmonic content downstream of the meter."
                ),
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
                        f" The site draws {_m(abs(kvar_mean), '.1f', ' kVAR')} leading — "
                        "capacitor banks are likely present and are a probable resonance source."
                    )
            # A commissioned impedance sweep is not a proportionate ask of a
            # house or a corner store; both get the review wording.
            if thresh.customer_class in ("r", "c"):
                res_cause = (
                    "Parallel resonance forms when feeder or service-entrance inductance resonates "
                    "with capacitance on the system (power-factor correction banks, a neighboring "
                    "customer's capacitors, or cable charging capacitance). Residential "
                    "services do not typically include capacitor banks, which bears on where "
                    f"the capacitance is likely to sit.{cap_note}"
                )
                res_rec = (
                    "If confirmed, this would warrant a distribution engineering review of "
                    "feeder capacitor banks and line-side PF correction equipment for a "
                    f"resonance at {', '.join(str(h * 60) for h in sorted(resonant))} Hz. "
                    "A harmonic impedance frequency sweep at the service entrance would "
                    "confirm the resonant frequency and locate the source."
                )
            else:
                res_cause = (
                    "Parallel resonance forms when the system (transformer + feeder) inductance "
                    "resonates with power factor correction or harmonic filter capacitors. "
                    "At the resonant order, even small harmonic currents produce large "
                    f"harmonic voltages, amplifying both V_h and I_h at that order.{cap_note}"
                )
                res_rec = (
                    "Commission a harmonic impedance frequency sweep to confirm the resonant "
                    f"frequency (target: {', '.join(str(h * 60) for h in sorted(resonant))} Hz). "
                    "Detune existing capacitor banks by adding series reactors (typically 5–7% "
                    "of bank kVAR), or switch to a harmonic filter bank tuned below H5 (282 Hz). "
                    "Until resolved, do not add more capacitors without a harmonic study."
                )

            findings.append({
                "category":       "harmonics",
                "severity":       "warning",
                "title":          f"Parallel resonance suspected at {h_res_str}",
                "finding":        (
                    f"Harmonic impedance at {h_res_str} is {_m(max(source_harm['orders'][h].get('z_ratio', 0) for h in resonant), '.1f', '×')} "
                    "higher than the linear inductive trend extrapolated from other orders. "
                    "This signature is consistent with a parallel LC resonance between system "
                    "inductance and capacitor banks at that harmonic frequency."
                ),
                "cause":          res_cause,
                "origin_evidence": (
                    "A parallel resonance requires both system inductance and "
                    "capacitance. Capacitor banks on the distribution system "
                    "and power-factor correction within the facility are both "
                    "candidates; the impedance signature does not distinguish "
                    "them."
                ),
                "recommendation": res_rec,
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
                    "title":          "Harmonic voltage and current rise together",
                    "finding":        (
                        f"Voltage and current harmonics are strongly correlated across all "
                        f"measured orders (mean Pearson r = {_m(mean_corr, '.2f')}). "
                        "This confirms harmonics originate from loads on this service rather "
                        "than from background utility voltage distortion."
                    ),
                    "cause":          (
                        "Customer-side nonlinear loads (VFDs, rectifiers, SMPS) inject "
                        "harmonic currents that develop harmonic voltages across the source "
                        "impedance — causing V_h and I_h to rise and fall together."
                    ),
                    "origin_evidence": (
                        "Voltage and current at these orders rise and fall "
                        "together, which is the pattern expected when a "
                        "downstream load injects the harmonic current. "
                        "Confirming direction requires phasor measurement."
                    ),
                    "recommendation": (
                        "Focus mitigation on customer loads. Options: input reactors on VFDs, "
                        "active harmonic filters, or 12-pulse / 18-pulse drive upgrades. "
                        "Utility-side action (capacitor detuning) is not required at this stage."
                    ),
                    "confidence":     "medium",
                    "evidence":       {"mean_pearson_r": mean_corr,
                                       "orders_tested": list(source_harm["orders"].keys())},
                })

    # ── Current imbalance, with the discriminating evidence ───────────────────
    if pf_flags.get("current_imbalance") is False:
        vi_mean = volt_imb.get("mean_imbalance_pct", 0)
        ci_mean = ci.get("mean_imbalance_pct", 0)
        if vi_mean < 1.0:
            findings.append({
                "category":       "imbalance",
                "severity":       "warning",
                "title":          "Current imbalance with balanced supply voltage",
                "finding":        (f"Current imbalance mean {_m(ci_mean, '.1f', '%')}, "
                                   f"max {_m(ci.get('max_imbalance_pct', 0), '.1f', '%')} "
                                   f"(limit 10%). Voltage imbalance is low ({_m(vi_mean, '.2f', '%')}), "
                                   "indicating balanced supply voltage."),
                "cause":          ("Supply voltage is well balanced while current is not, "
                                   "which is the pattern produced by unequal single-phase "
                                   "load distribution rather than by a supply asymmetry."),
                "origin_evidence": (
                    "Supply voltage is balanced while current is not. A "
                    "balanced voltage with unbalanced current is the "
                    "signature of unequal single-phase load distribution "
                    "rather than a supply condition."
                ),
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
                "finding":        (f"Current imbalance mean {_m(ci_mean, '.1f', '%')}, "
                                   f"voltage imbalance mean {_m(vi_mean, '.2f', '%')}. "
                                   "Both are elevated; supply may be contributing."),
                "cause":          ("Both voltage and current are imbalanced. Unbalanced supply "
                                   "voltage (from the utility distribution system) will induce "
                                   "current imbalance in motor loads. Customer load imbalance "
                                   "may also be a contributing factor."),
                "origin_evidence": (
                    "Both voltage and current are unbalanced, so a supply "
                    "asymmetry and an unbalanced load would produce the same "
                    "measurement. Measuring with load disconnected separates "
                    "them."
                ),
                "recommendation": ("Measure voltage imbalance with all customer loads "
                                   "disconnected to isolate the utility contribution. "
                                   "Voltage imbalance still above 1% at no-load would "
                                   "point upstream of the meter and warrant a "
                                   "distribution review."),
                "confidence":     "medium",
                "evidence":       {"current_imb_mean_pct": round(ci_mean, 2),
                                   "voltage_imb_mean_pct": round(vi_mean, 2)},
            })

    # ── Voltage imbalance, with the discriminating evidence ──────────────────
    # Only reached when current imbalance did *not* fail. This finding's title
    # and origin evidence both assert that load current is balanced, and they
    # used to assert it without checking: on the residential fixture it fired
    # alongside "Current imbalance -- investigate supply voltage", one saying
    # both were elevated and the other saying current was low, about the same
    # recording.
    if (pf_flags.get("voltage_imbalance") is False
            and pf_flags.get("current_imbalance") is not False):
        vi_max = volt_imb.get("max_imbalance_pct", 0)
        if geometry in ("split-phase", "two-leg-208"):
            # Two legs, so there is no third phase to lose a fuse on and no
            # delta to open. The leg difference is a voltage divider across the
            # neutral impedance.
            title = "Leg voltage difference with balanced leg currents"
            cause = ("On a two-leg service a sustained difference between the "
                     "leg voltages comes from unequal loading acting across the "
                     "impedance of the shared neutral path, or from added "
                     "resistance in that path. With the leg currents balanced, "
                     "the neutral path itself is the more likely of the two.")
            origin = ("The legs differ in voltage while drawing comparable "
                      "current. Unequal loading would show in the currents, so "
                      "this pattern points to the shared neutral or supply "
                      "rather than to how load is split between the legs.")
            rec = ("Read this together with the neutral health assessment. "
                   "Inspect the neutral path from the transformer through the "
                   "service drop and meter socket to the panel, and measure "
                   "neutral-to-ground voltage under load.")
        else:
            title = "Voltage imbalance with balanced load current"
            cause = ("Steady-state voltage imbalance exceeding 3% is typically caused "
                     "by unbalanced distribution transformer loading, asymmetric "
                     "line impedances, a blown capacitor fuse on one phase, "
                     "or an open delta transformer configuration.")
            origin = ("Voltage imbalance is present while current imbalance is low, "
                      "which is the pattern expected of an asymmetry upstream of "
                      "the meter rather than of load distribution.")
            rec = ("Confirming the origin requires measurement with all customer "
                   "loads disconnected. Feeder capacitor bank fuses and "
                   "transformer loading on adjacent services are worth checking.")
        findings.append({
            "category":       "voltage",
            "severity":       "warning",
            "title":          title,
            "finding":        (f"{volt_imb.get('metric_label', 'Voltage imbalance')} "
                               f"max {_m(vi_max, '.2f', '%')}, "
                               f"mean {_m(volt_imb.get('mean_imbalance_pct', 0), '.2f', '%')} "
                               f"(limit {volt_imb.get('limit_pct', 3.0):.0f}%)."),
            "cause":          cause,
            "origin_evidence": origin,
            "recommendation": rec,
            "confidence":     "high",
            "evidence":       {"voltage_imb_max_pct":  round(vi_max, 2),
                               "geometry":             geometry},
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
                    "finding":        (f"Voltage-to-load correlation = {_m(corr, '.2f')}. "
                                       f"Minimum recorded voltage {_m(min_v, '.1f', ' V')}. "
                                       "Voltage tends to decrease as current increases."),
                    "cause":          ("Negative voltage-load correlation indicates resistive "
                                       "voltage drop in the secondary service conductors or "
                                       "transformer impedance. This is more pronounced on long "
                                       "secondary runs or undersized conductors."),
                    "origin_evidence": (
                        "Voltage falls as load rises, which is the signature "
                        "of series impedance between the source and the "
                        "meter. The impedance may lie in the service "
                        "conductors, the connections, or the transformer."
                    ),
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
        # Below a few kVAR the correction is smaller than the smallest practical
        # switched capacitor step, so recommending an install is not actionable
        # advice -- it just adds a line to the report.
        if kvar_needed < _MIN_ACTIONABLE_KVAR:
            log.info("Power factor correction of %.1f kVAR is below the %.0f kVAR "
                     "actionable floor; not raised as a recommendation.",
                     kvar_needed, _MIN_ACTIONABLE_KVAR)
            findings.append({
                "category":   "power_factor",
                "severity":   "info",
                "title":      "Power factor below tariff limit by a small margin",
                "finding":    (f"Mean PF {_m(pf_mean, '.3f')} lagging against a "
                               f"{target_pf:.2f} limit. Correcting to the limit "
                               f"needs about {kvar_needed:.1f} kVAR, which is "
                               f"below the smallest practical switched capacitor "
                               f"step."),
                "cause":      ("The shortfall is small enough that it is more "
                               "likely to reflect light-load operation than an "
                               "uncorrected inductive load."),
                "origin_evidence": ("Reactive demand is set by the load's own "
                                    "characteristics."),
                "recommendation": ("No correction equipment is warranted at this "
                                   "margin. Re-evaluate if load grows."),
                "no_action":  True,
                "confidence": "medium",
                "evidence":   {"mean_pf": round(pf_mean, 4),
                               "kvar_needed": round(kvar_needed, 1)},
            })
        else:
            findings.append({
            "category":       "power_factor",
            "severity":       "warning",
            "title":          "Low power factor — capacitor correction needed",
            "finding":        (f"Mean PF {_m(pf_mean, '.3f')} lagging (limit {target_pf:.2f}). "
                               f"Mean reactive power {_m(q_mean, '.1f', ' kVAR')}. "
                               f"Estimated correction needed: {kvar_needed:.0f} kVAR."),
            "cause":          ("Lagging power factor is caused by inductive reactive loads — "
                               "primarily motors, transformers, and inductive ballasts drawing "
                               "magnetizing current. VFDs with active front ends may improve "
                               "PF at the drive level but do not eliminate motor reactive demand."),
            "origin_evidence": (
                "Reactive demand is set by the load's own characteristics, "
                "and correction equipment is normally installed on the load "
                "side of the meter."
            ),
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
            evidence_parts.append(f"Vne max {_m(vne_max, '.1f', ' V')}")
        if leg_r < 0.5:
            evidence_parts.append(f"leg correlation r = {_m(leg_r, '.3f')}")
        if sum_std > 2.0:
            evidence_parts.append(f"L1+L2 sum std = {_m(sum_std, '.1f', ' V')}")

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
            "origin_evidence": (
                "Cross-leg voltage behaviour of this kind arises from a "
                "neutral path shared by both legs. That path runs from the "
                "transformer through the service drop and meter socket to the "
                "main panel, so it spans both utility and customer equipment."
            ),
            "recommendation": (
                "Inspect and tighten all neutral connections from the meter socket through the "
                "service entrance to the main panel. Check for corrosion at split-bolt connectors "
                "and wire nut splices. Measure neutral-to-ground voltage at the panel — > 1 V "
                "under load confirms neutral resistance. If the service neutral is overhead, "
                "inspect the drip loop and weatherhead connection. Because the neutral path "
                "spans both sides of the meter, inspection of the utility secondary and the "
                "meter socket neutral lug is also indicated."
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
        # Prefer meter-measured K-factor (includes all H1-H51) over estimated value --
        # but only when the channel actually has valid data, otherwise fall through
        # to the estimated branch instead of silently losing this finding to a NaN.
        # Sized on the worst phase: heating is per-winding, so the phase with the
        # highest K-factor sets the rating the transformer needs.
        kf = kfactor_by_phase(df)
        if kf["available"]:
            k_factor = kf["median"]
            k_phase = kf["worst_phase"]
            k_source = f"meter, phase {k_phase}"
        else:
            h_means  = _harmonic_means(df, _H519_ORDERS)
            k_num    = sum((h_means.get(h, 0) / il_amps)**2 * h**2 for h in _H519_ORDERS)
            k_denom  = sum((h_means.get(h, 0) / il_amps)**2 for h in _H519_ORDERS)
            k_factor = k_num / k_denom if k_denom > 0 else 1.0
            k_phase = None
            k_source = "estimated"
        _k_rating, _k_wording = standard_k_rating(k_factor)
        # A K-rating recommendation is only actionable when the harmonic content
        # behind it was measured at real load; at 1.1 A the meter's K-factor runs
        # to the hundreds and means nothing for sizing.
        _k_significant = harmonic_spectrum_significance(df, thresh)["usable"]
        if pct_tx > 70 and k_factor > 4 and _k_significant:
            findings.append({
                "category":       "demand",
                "severity":       "warning",
                "title":          "Transformer derating — harmonic K-factor",
                "finding":        (f"Transformer loaded at {_m(pct_tx, '.0f', '%')} of nameplate. "
                                   f"Harmonic load K-factor = {_m(k_factor, '.1f')} ({k_source}). "
                                   + (", ".join(f"phase {p} = {_m(v['median'], '.1f')}"
                                                for p, v in sorted(kf["phases"].items()))
                                      + ". " if kf["available"] and len(kf["phases"]) > 1 else "")
                                   + "Standard distribution transformers are rated K=1."),
                "cause":          ("Harmonic currents cause additional eddy-current losses in "
                                   "transformer windings beyond what the nameplate rating assumes. "
                                   f"A K-factor of {_m(k_factor, '.1f')} means harmonic-related heating "
                                   "is significantly greater than for a sinusoidal load at "
                                   "the same kVA, increasing winding temperature and accelerating "
                                   "insulation degradation."),
                "origin_evidence": (
                    "Transformer heating follows the harmonic current the "
                    "load draws; the transformer's rating is a separate "
                    "matter from what causes the heating."
                ),
                "recommendation": (
                    (f"Derate the transformer or replace with a K-{_k_rating} "
                     "or higher rated unit. " if _k_rating else _k_wording + " ")
                    + "Alternatively, reduce harmonic content "
                    "(AC line reactors on VFD inputs, or a passive harmonic filter) "
                    "to lower the effective K-factor before the next capacity "
                    "addition."),
                "confidence":     "high" if k_phase else "medium",
                "evidence":       {"pct_nameplate":   round(pct_tx, 1),
                                   "k_factor":        round(k_factor, 1),
                                   "k_source":        k_source,
                                   "k_worst_phase":   k_phase,
                                   "k_by_phase":      {p: round(v["median"], 1)
                                                       for p, v in kf["phases"].items()}
                                                      if kf["available"] else None},
            })

    return findings

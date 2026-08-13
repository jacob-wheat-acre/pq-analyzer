from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as _np

__version__ = "0.40.2"


@dataclass
class Thresholds:
    """All engineering limits in one place — pass to PQAnalyzer."""
    nominal_voltage: float = 120.0        # V (line-to-neutral)
    volt_tolerance: float = 0.05          # ±5 % → ANSI C84.1 Range A
    # A primary-metered service is metered on the high side, and PSCo runs a
    # variety of primary voltages. Nothing in the file identifies which one, and
    # inferring it from the measured L-L/L-N ratio only recovers the topology,
    # not the nominal -- so the engineer enters it and the ANSI bands are built
    # from it. Left unset, the L-L nominal is inferred as it always was.
    primary_ll_voltage: Optional[float] = None  # V line-to-line, primary metering
    thd_voltage_limit: float = 8.0        # % → IEEE 519 Table 2 (≤1 kV bus)
    thd_current_limit: float = 5.0        # % THD fallback when no RMS current channels (TDD unavailable)
    power_factor_limit: float = 0.90      # lagging — flag below this
    imbalance_limit: float = 3.0          # % voltage unbalance — NEMA MG1 / IEEE 1159
    current_imbalance_limit: float = 10.0 # % current unbalance — per PSC procedure
    event_delta_pct: float = 0.10         # spike detection: step > 10 % of nominal
    # ANSI C84.1 does not limit frequency; ±0.5 Hz is a practical flag band,
    # comparable to EN 50160's ±1 % for interconnected systems.
    frequency_nominal: float = 60.0       # Hz
    frequency_tolerance_hz: float = 0.5   # Hz — flag deviation beyond this
    isc_amps: Optional[float] = None      # short-circuit current at PCC (A) — from Blue Book
    isc_source: Optional[str] = None     # human-readable note on how ISC was determined
    transformer_kva: Optional[float] = None  # service transformer nameplate (kVA)
    customer_class: str = "sg"            # "r" | "c" | "sg" | "pg"  (PSCo tariff schedules)
    # The engineer picks these at the start; they resolve how many phases the
    # service actually has, which channel presence alone can get wrong when a
    # phase is simply missing from the export.
    service_type: Optional[str] = None    # e.g. "1ph-padmount", "3ph-padmount"
    topology: str = "auto"                # "auto" | "3ph-wye" | "split-phase"
    # The run between the transformer and the meter, picked at the start. Both
    # are needed before a measured impedance can be compared with an expected
    # one; without them the measurement still stands on its own.
    conductor_key: Optional[str] = None   # key into _CONDUCTOR_TABLE
    run_length_ft: Optional[float] = None # transformer → meter, one way
    # Where the transformer does not land at this meter but on a secondary main
    # shared with the neighbours, that main is in this customer's path too and
    # its drop belongs in the expected impedance. Left unset, the service is
    # taken to be a dedicated run from the transformer, which is what a
    # dedicated-transformer class already is.
    shared_secondary_key: Optional[str] = None   # key into _CONDUCTOR_TABLE
    shared_secondary_ft: Optional[float] = None  # transformer → service tap
    # A primary-metered service is metered on the high side, so the transformer
    # and the secondary run are the customer's, not the utility's, and neither
    # is in the path being measured. What is in it is the primary line from the
    # source to the metering point, entered as sequence impedances in ohms.
    # Z1 carries balanced load current and is what the report compares against;
    # Z0 is optional and used only where the physics asks for it -- triplen
    # harmonics, which are zero-sequence, and unbalanced current returning
    # through earth. Z2 is not asked for: for a passive line it equals Z1.
    primary_metered: bool = False
    primary_r1_ohm: Optional[float] = None
    primary_x1_ohm: Optional[float] = None
    primary_r0_ohm: Optional[float] = None
    primary_x0_ohm: Optional[float] = None
    # Spectral-shape ("broadband vs. resonance") classifier -- heuristic starting
    # points, not yet empirically validated across many sites. See check_spectral_shape().
    spectral_elevation_ratio: float = 0.4  # mean VTHD / thd_voltage_limit above this = "elevated"
    spectral_flatness_cv: float = 0.6      # per-order spectrum CV below this = "flat" (broadband-like)


# ── Marking measured values ──────────────────────────────────────────────────
# Prose puts four kinds of number in one sentence -- what the meter recorded,
# what a standard allows, what a nameplate says, and what the engineer entered
# -- and only the first was measured here. A value is marked where it is
# written, because nothing downstream can tell "the measured 0.0226 Ω" from
# "the expected 0.0249 Ω" by looking at the text. The Word layer renders marked
# values bold; every other output strips the markers.
#
# This lives in the constants module because both the analysis layer, which
# writes findings, and the report layer, which writes sections, have to speak
# it, and the analysis layer cannot import the report.
MEASURED_OPEN  = ""      # private use area: never occurs in report text
MEASURED_CLOSE = ""


def measured(value, spec: str = "", unit: str = "") -> str:
    """Mark ``value`` as something this recording measured.

    ``spec`` is an ordinary format spec, so ``measured(v, '.1f', ' V')`` reads
    at the call site the way ``f"{v:.1f} V"`` did. The unit travels with the
    number: a bold figure beside a plain unit scans worse than either alone.
    """
    text = format(value, spec) if spec else f"{value}"
    return f"{MEASURED_OPEN}{text}{unit}{MEASURED_CLOSE}"


def pct_text(value, spec: str = ".1f", mark: bool = False) -> str:
    """Format a percentage, and never round a real exceedance down to zero.

    A share of the recording is almost always printed beside a claim that a
    limit was passed, so "0.0% of the recording" next to "Exceeded" reads as a
    contradiction: the reader is shown a flag and, in the same row, arithmetic
    that appears to refute it.  What actually happened is that a handful of
    intervals out of tens of thousands rounded away.  A share that is genuinely
    zero still prints as zero -- that is not a rounding artifact and has to stay
    distinguishable from one -- while anything too small for the format to show
    prints as less than the smallest figure that format can carry.

    ``mark`` marks the figure as measured; leave it off for the outputs that
    cannot render marks (see ``measured``).
    """
    def _fmt(v) -> str:
        return measured(v, spec, "%") if mark else f"{format(v, spec)}%"

    try:
        v = float(value)
    except (TypeError, ValueError):
        return _fmt(value) if not mark else measured(value, spec, "%")
    # Only step in where the format itself is what erased the number.
    if v == 0 or float(format(v, spec)) != 0:
        return _fmt(v)
    digits = 0
    if "." in spec:
        tail = spec.rsplit(".", 1)[1].rstrip("f%")
        if tail.isdigit():
            digits = int(tail)
    smallest = 10.0 ** -digits
    # The comparison sign is not itself a measured quantity, so it stays outside
    # the marks that the Word layer renders bold.
    return ("<" if v > 0 else ">-") + _fmt(smallest)


def measured_pct(value, spec: str = ".1f") -> str:
    """``pct_text`` for prose that marks what this recording measured."""
    return pct_text(value, spec, mark=True)


def strip_marks(text: str) -> str:
    """Drop the markers, for every output that cannot render them."""
    if not isinstance(text, str):
        return text
    return text.replace(MEASURED_OPEN, "").replace(MEASURED_CLOSE, "")


# ── ANSI C84.1 voltage ranges ────────────────────────────────────────────────
# C84.1-2016 Table 1 states the ranges for service voltage (measured at the
# point of delivery, which is where this meter sits) and separately for
# utilization voltage (at the equipment terminals, after the customer's own
# drop). This tool measures at the meter, so the service voltage column is the
# applicable one. Table 1 splits it into two groups, and they do NOT share a
# lower limit:
#
#                       Range B min   Range A min   Range A max   Range B max
#   120 V – 600 V          91.67 %       95.0 %        105 %        105.83 %
#   2.4 kV – 34.5 kV       95.0 %        97.5 %        105 %        105.8 %
#
# The low-voltage figures are stated in the table as volts on a 120 V base
# (Range A 114-126, Range B 110-127) and scale exactly to its published rows:
# 208 -> 197/218 and 191/220, 240 -> 228/252 and 220/254, 480 -> 456/504 and
# 440/508. The over-600 V figures are stated as percentages.
#
# Two asymmetries matter, and both are deliberate:
#
#   * Range B is wider below nominal than above it. The standard tolerates the
#     drop of a long or heavily loaded feeder further than it tolerates the
#     overvoltage that shortens equipment life.
#   * The over-600 V group is TIGHTER on the low side, not looser -- Range A
#     bottoms out at 97.5 % rather than 95 %. A primary-metered customer still
#     has their own transformation between this meter and their equipment, and
#     the standard reserves that headroom for the drop through it. Applying the
#     low-voltage -5 % to a 13.2 kV service puts the limit 330 V below where
#     C84.1 sets it, which passes a real undervoltage.
RANGE_A_UNDER = 114.0 / 120.0   # 0.95
RANGE_A_OVER  = 126.0 / 120.0   # 1.05
RANGE_B_UNDER = 110.0 / 120.0   # 0.91667
RANGE_B_OVER  = 127.0 / 120.0   # 1.05833

RANGE_A_UNDER_MV = 0.975
RANGE_A_OVER_MV  = 1.05
RANGE_B_UNDER_MV = 0.95
RANGE_B_OVER_MV  = 1.058

#: Where Table 1 changes groups. Compared against whatever nominal is handed in,
#: which is the line-to-neutral figure for the per-phase check and the
#: line-to-line figure for the L-L check. That works for every system this tool
#: supports because the two groups do not overlap on either basis: the largest
#: low-voltage system is 600 V L-L (346 V L-N) and the smallest medium-voltage
#: one is 2400 V L-L (1386 V L-N).
MV_NOMINAL_FLOOR_V = 600.0

#: Table 1 stops at 34.5 kV; above that is transmission and a different standard.
#: Compared line-to-line, so a 34.5 kV system is inside it on both bases.
MV_NOMINAL_CEILING_V = 34500.0


def ansi_bands(nominal_v: float) -> dict:
    """The C84.1 service-voltage ranges for one nominal.

    Returns both bands in volts, the Table 1 group they came from, and
    ``range_b_evaluated``.  Callers must not fall back to the Range A band when
    a band is unavailable: "we did not evaluate this" and "this passed" are
    different answers, and the report has to be able to say which one it holds.
    """
    nominal_v = float(nominal_v)
    if nominal_v > MV_NOMINAL_CEILING_V:
        # Above 34.5 kV C84.1 hands off to another standard. Returning the
        # medium-voltage band here would be inventing a limit for a system the
        # table does not cover.
        return {
            "nominal_v":         nominal_v,
            "group":             "out_of_scope",
            "a_min":             None,
            "a_max":             None,
            "b_min":             None,
            "b_max":             None,
            "range_a_evaluated": False,
            "range_b_evaluated": False,
            "range_b_note": (
                f"ANSI C84.1-2016 Table 1 does not cover {nominal_v:,.0f} V: it "
                "runs to 34.5 kV, above which voltage ratings are set by another "
                "standard. No C84.1 range is evaluated for this service."),
        }

    mv = nominal_v > MV_NOMINAL_FLOOR_V
    a_lo, a_hi, b_lo, b_hi = (
        (RANGE_A_UNDER_MV, RANGE_A_OVER_MV, RANGE_B_UNDER_MV, RANGE_B_OVER_MV)
        if mv else
        (RANGE_A_UNDER, RANGE_A_OVER, RANGE_B_UNDER, RANGE_B_OVER))
    return {
        "nominal_v":         nominal_v,
        "group":             "over_600v" if mv else "under_600v",
        "a_min":             nominal_v * a_lo,
        "a_max":             nominal_v * a_hi,
        "b_min":             nominal_v * b_lo,
        "b_max":             nominal_v * b_hi,
        "range_a_evaluated": True,
        "range_b_evaluated": True,
        "range_b_note":      "",
    }


def ansi_band_basis(bands: dict) -> str:
    """One line naming which Table 1 group a set of bands came from.

    Worth printing because the two groups differ only in their lower limits, so
    a reader checking a primary service's 97.5 % floor against the ±5 % they
    know from secondary work would otherwise think the tool had it wrong.
    """
    if bands.get("group") == "over_600v":
        return ("ANSI C84.1-2016 Table 1, service voltage, 2.4–34.5 kV group: "
                "Range A 97.5–105% of nominal, Range B 95–105.8%. The lower "
                "limits are tighter than the 120–600 V group's because a "
                "primary-metered customer's own transformation still lies "
                "between this meter and their equipment.")
    if bands.get("group") == "under_600v":
        return ("ANSI C84.1-2016 Table 1, service voltage, 120–600 V group: "
                "Range A ±5% of nominal, Range B −8.33%/+5.83%.")
    return bands.get("range_b_note", "")


#: The states a steady-state voltage reading can be in, worst last. "outside_a"
#: is the honest answer where Range B was not evaluated: the reading left Range
#: A, and how far past it went is not something this tool can grade.
VOLTAGE_BAND_ORDER = ["range_a", "range_b", "outside_a", "outside_b"]

VOLTAGE_BAND_LABEL = {
    "range_a":   "Within Range A",
    "range_b":   "Range B",
    "outside_a": "Outside Range A",
    "outside_b": "Outside Range B",
}


# ── Finding severity ─────────────────────────────────────────────────────────
# Compliance is binary because the standards are: a value is inside the limit or
# it is not, and that determination stays quotable.  Severity is a second,
# separate axis answering how much the exceedance matters, so that a single
# artifact interval and a sustained 2x overload stop sharing one red FAIL.
#
# These are engineering judgment, not values from any standard.  They are named
# here rather than buried in the report so they can be tuned against real files.

#: Bands, ordered least to most serious.  "watch" is inside the limit.
SEVERITY_ORDER = ["compliant", "watch", "minor", "significant", "severe"]

SEVERITY_LABEL = {
    "compliant":    "Compliant",
    "watch":        "Watch",
    "minor":        "Minor",
    "significant":  "Significant",
    "severe":       "Severe",
    "not_assessed": "Not assessed",
}

#: Inside the limit but this close to it -> "Watch".  Nothing is wrong yet, so
#: this must not render in a warning colour; it flags drift worth tracking.
SEVERITY_WATCH_FRACTION = 0.85

#: Tighter band for floor metrics like power factor.  PF is bounded above at
#: 1.0, so its whole usable range above a 0.90 limit spans about 11% -- a 15%
#: "close to the limit" rule would mark every power factor, including 0.99, as
#: Watch.  0.95 flags roughly PF < 0.945 instead.
SEVERITY_WATCH_FRACTION_FLOOR = 0.95

#: Over the limit by this ratio -> at least "Significant".
SEVERITY_SIGNIFICANT_MARGIN = 1.20
#: Over the limit for this share of the recording (%) -> at least "Significant".
SEVERITY_SIGNIFICANT_PERSISTENCE = 25.0

#: "Severe" needs both a large margin and persistence -- a brief 1.6x excursion
#: is not the same finding as one that holds for a quarter of the week.
SEVERITY_SEVERE_MARGIN = 1.50
SEVERITY_SEVERE_PERSISTENCE = 25.0
#: ...except that this far over the limit is severe however briefly it happened.
SEVERITY_SEVERE_MARGIN_ALONE = 2.00


# IEEE 519-2022 Table 2: TDD limits indexed by ISC/IL ratio
_TDD_TABLE = [(20, 5.0), (50, 8.0), (100, 12.0), (1000, 15.0)]

# IEEE 519-2022 Table 2: individual harmonic limits (% of IL) by ISC/IL class
# Rows: (h_min, h_max_exclusive, [limit_<20, limit_20-50, limit_50-100, limit_100-1000, limit_>=1000])
_H519_LIMITS: List[Tuple[int, int, List[float]]] = [
    (2,  11, [4.0,  7.0, 10.0, 12.0, 15.0]),   # h < 11
    (11, 17, [2.0,  3.5,  4.5,  5.5,  7.0]),   # 11 ≤ h < 17
    (17, 23, [1.5,  2.5,  4.0,  5.0,  6.0]),   # 17 ≤ h < 23
    (23, 35, [0.6,  1.0,  1.5,  2.0,  2.5]),   # 23 ≤ h < 35
    (35, 51, [0.3,  0.5,  0.7,  1.0,  1.4]),   # 35 ≤ h ≤ 50
]

# Odd harmonic orders to check per IEEE 519-2022 (even harmonics limited to 25% of odd limits)
_H519_ORDERS = [3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 35, 37, 47, 49]

# ── Load-signature families and match thresholds ─────────────────────────────
# Several library entries describe the same electrical topology and therefore
# have near-identical spectra -- a 6-pulse VFD, a 6-pulse UPS and a DC fast
# charger all rectify three-phase power the same way.  Nothing in a harmonic
# spectrum can separate them, so when the top candidates come from one family
# the family is reported rather than a specific piece of equipment.
LOAD_FAMILY_LABEL = {
    "six_pulse":              "Three-phase 6-pulse rectifier load "
                              "(VFD, UPS, or DC fast charger)",
    "multipulse":             "Multi-pulse or active-front-end drive "
                              "(12-pulse, 18-pulse, or AFE)",
    "single_phase_switchmode": "Single-phase switch-mode or electronic "
                              "lighting load",
    "triplen_saturation":     "Transformer saturation",
    "arcing":                 "Arcing load (welding or arc furnace)",
    "mixed_three_phase":      "Mixed three-phase: rectifier plus "
                              "single-phase nonlinear load",
    "mixed_single_phase":     "Single-phase drive or charger",
    "near_linear":            "Near-linear or inverter-interfaced load",
}

#: Candidate actions, written per *family* rather than per entry.
#:
#: The finding names the family and refuses to name the member, so attaching one
#: member's advice would smuggle the member claim back in through the body --
#: and inside a family that advice contradicts itself.  "six_pulse" held one
#: entry saying "verify existing input reactors are in service" and another
#: saying "add reactors"; "multipulse" held "verify the phase-shifting
#: transformer" against "no action required".  Which one printed turned on a
#: member score gap the same finding calls meaningless.
#:
#: Each entry is therefore verification-first: it says how to tell the members
#: apart on site, and what follows from each answer.  Where the advice depends
#: on triplens accumulating in the neutral it says so, because these families
#: reach residential services where the legs are collinear and triplens
#: subtract instead -- see `service_geometry` in pq_analysis.
LOAD_FAMILY_RECOMMENDATION = {
    "six_pulse": (
        "Identify the 6-pulse rectifier loads at this service — variable "
        "frequency drives, double-conversion UPS units and DC fast chargers "
        "all present this way — and establish whether input reactors or DC bus "
        "chokes are fitted and in service. H11 and H13 elevated relative to H5 "
        "indicates reactors absent or bypassed; with 3–5% impedance reactors "
        "fitted, H11/H13 typically fall to around a fifth of H5. Where "
        "reactors are missing, adding them reduces H5/H7 by 30–50% and extends "
        "input diode life. Where they are already in service and TDD remains "
        "above the limit, 12-pulse, 18-pulse or active front-end equipment, or "
        "a harmonic filter, are the next steps. Current DC fast chargers "
        "commonly use 12-pulse or active PFC front ends — confirm against "
        "equipment specifications rather than assuming a 6-pulse one."
    ),
    "multipulse": (
        "Multi-pulse and active-front-end equipment is already a harmonic "
        "mitigation measure, so this family is not usually the thing to "
        "change. If TDD remains above the limit, confirm which is installed "
        "and check it is working as designed: on a 12-pulse drive, verify the "
        "phase-shifting transformer is balanced between the two rectifier "
        "bridges, since imbalance between bridges reintroduces the 5th and 7th "
        "the topology exists to cancel. On an 18-pulse or active front-end "
        "unit there is normally nothing to correct, and other loads at this "
        "service are the more likely contributors. An active harmonic filter "
        "is the remaining option after that."
    ),
    "single_phase_switchmode": (
        "Survey the single-phase electronic loads at this service — computers "
        "and servers, electronic ballasts and LED drivers share this "
        "signature. The action depends on which dominates. Magnetic-ballast "
        "fluorescent fixtures are worth retrofitting to electronic ballasts or "
        "LED. Budget LED drivers without active power factor correction are "
        "worth addressing on procurement: specify PF > 0.90 and THD < 20%, and "
        "note IEC 61000-3-2 Class C applies to lighting. Where the load is "
        "concentrated computing equipment the equipment itself is not usually "
        "what changes — verify instead that the neutral conductor is rated for "
        "the harmonic current it carries, since on a three-phase four-wire "
        "system triplens from these loads add in the neutral rather than "
        "cancelling, and consider a K-rated or isolation transformer."
    ),
    "mixed_single_phase": (
        "Establish which single-phase electronic load is present: a Level 2 "
        "charger's on-board rectifier and an inverter-driven heat pump or "
        "mini-split produce a similar signature, and both are cyclic. Neither "
        "normally justifies mitigation at a single residence. Where several "
        "units share one transformer the aggregate matters more than any one "
        "of them — check the transformer's total loading, and on a three-phase "
        "four-wire secondary whether H3 is accumulating in the neutral. For "
        "clustered or fleet charging, managed charging and transformer sizing "
        "are the levers, and three-phase DC fast chargers with active PFC "
        "front ends are preferable to co-located single-phase units."
    ),
    "arcing": (
        "Arc loads are intermittent by nature and the response scales with "
        "size. For welding, scheduling operations to avoid coincident peaks is "
        "usually the first step, with a series reactor or active power filter "
        "for large or continuous loads. For furnace-scale arc loads a "
        "dedicated harmonic study and dynamic compensation — a static VAR "
        "compensator or STATCOM — are normally required rather than optional. "
        "Either way, read the flicker measurement alongside this: arc loads "
        "act on voltage flicker at least as much as on harmonics, and Pst "
        "above 1.0 warrants coordination with Xcel Energy before mitigation is "
        "specified. Arc furnace installations are subject to pre-approval "
        "under tariff."
    ),
    "near_linear": (
        "Neither load in this family is normally a significant distortion "
        "source and mitigation is rarely warranted. Establish which is present "
        "before acting. Distortion that follows the solar day points to an "
        "interconnected PV inverter — confirm the interconnection, review the "
        "inverter's IEEE 1547 test report, and check the firmware is current, "
        "since harmonic and anti-islanding behaviour are commonly improved in "
        "later revisions. Distortion that tracks cooling or heating demand "
        "points to compressor load instead, where the usual complaint is "
        "starting voltage dip rather than harmonics, addressed by service "
        "conductor size, transformer sizing or a soft start rather than by "
        "filters."
    ),
    # Single-entry families: the family and the member coincide, so these carry
    # the member's advice. The attribution sentence that used to close the
    # saturation entry ("may be a shared or utility responsibility") is gone --
    # the tool states evidence and the engineer assigns responsibility.
    "triplen_saturation": (
        "Check the supply voltage level. Sustained voltage above +5% of "
        "nominal drives transformer saturation and the harmonic injection that "
        "follows from it, so the voltage measurement rather than the harmonic "
        "one is where this is resolved. The relevant evidence is whether the "
        "over-voltage is present at no load."
    ),
    "mixed_three_phase": (
        "Two sources are indicated and both are worth addressing. For the "
        "three-phase rectifier component, establish whether input reactors are "
        "fitted and in service, and consider multi-pulse or active front-end "
        "equipment if TDD remains above the limit. For the single-phase "
        "component — computers, LED lighting — verify neutral conductor sizing "
        "against the harmonic current it carries, noting that triplens add in "
        "the neutral on a three-phase four-wire system. A K-rated transformer "
        "is worth considering where the measured K-factor exceeds 4."
    ),
}

# Thresholds below were set from a measured null distribution, not chosen by
# eye.  Scoring 20,000 randomly generated decaying spectra (the shape real
# harmonic spectra take) against this library gave a median top score of 0.87,
# with 71% above 0.75 and 29% above 0.95.  In other words the old 0.75 gate
# admitted noise most of the time, and pure noise reached "high confidence"
# roughly a third of the time.  See test_pq.py::TestLoadSignatureFloor.

#: Minimum score before any load type is named at all.  Below this the report
#: says it does not recognise the spectrum instead of naming a nearest
#: neighbour.  Engineering judgment informed by the null distribution above.
SIGNATURE_ABSOLUTE_FLOOR = 0.90

#: Minimum gap to the next *family* before a match is considered resolved.
#: Within-family gaps are meaningless -- those entries are the same topology.
SIGNATURE_FAMILY_SEPARATION = 0.05

#: Gap to the next entry inside the winning family, below which those entries
#: are described as indistinguishable.  This no longer gates anything: the
#: family is now always what gets reported, because naming the individual entry
#: was measured naming a load that was not present 45% of the time on secondary
#: and 32% on primary service (test_pq.py::TestLoadSignatureMixtures).  It only
#: selects how the finding words the omission.
SIGNATURE_MEMBER_SEPARATION = 0.05


# ── Harmonic load-signature reference library ─────────────────────────────────
# Spectrum vectors are [H3, H5, H7, H9, H11, H13] as typical % of fundamental
# at rated/normal operating load.  Only the *shape* matters — vectors are
# normalized to unit length at scoring time, so absolute THD is irrelevant.
#
# variability: expected inter-interval CV of H5
#   "low"    → continuous steady-state load (VFDs, lighting, office equipment)
#   "medium" → cyclic or partially intermittent (EV chargers, batch processes)
#   "high"   → strongly intermittent (arc furnace, arc welder during operation)
_LOAD_SIGNATURES: List[Dict] = [
    {
        "id": "vfd_6pulse_reactor",
        "family": "six_pulse",
        "classes": {"c", "sg", "pg"},
        "title": "6-pulse VFD / rectifier (with input reactor)",
        "spectrum": [2, 23, 9, 1, 5, 4],
        "variability": "low",
        "cause": (
            "H5-dominant spectrum with H5/H7 ≈ 2.5 and low H3 is the classic signature "
            "of 6-pulse rectifier loads with 3–5% AC line reactors. Common sources: "
            "variable frequency drives (VFDs), UPS systems, and DC motor drives."
        ),
        "recommendation": (
            "Inventory all 6-pulse rectifier loads. To reduce harmonic injection: "
            "(1) verify existing input reactors are in service, (2) upgrade to 12-pulse "
            "or 18-pulse drives where THD is critical, or (3) add a passive harmonic filter."
        ),
        "responsibility": "customer",
    },
    {
        "id": "vfd_6pulse_no_reactor",
        "family": "six_pulse",
        "classes": {"c", "sg", "pg"},
        "title": "6-pulse VFD / rectifier (no input reactor)",
        "spectrum": [2, 30, 13, 2, 12, 9],
        "variability": "low",
        "cause": (
            "6-pulse pattern with elevated H11/H13 relative to H5 indicates VFDs or "
            "rectifiers running without input reactors or DC bus chokes. With reactors, "
            "H11/H13 are typically suppressed to roughly H5/5."
        ),
        "recommendation": (
            "Add 3–5% impedance AC line reactors to VFD inputs. This reduces H5/H7 "
            "by 30–50%, suppresses H11/H13, and extends VFD input diode life."
        ),
        "responsibility": "customer",
    },
    {
        "id": "rectifier_12pulse",
        "family": "multipulse",
        "classes": {"sg", "pg"},
        "title": "12-pulse rectifier / drive",
        "spectrum": [1, 3, 2, 1, 14, 11],
        "variability": "low",
        "cause": (
            "H5 and H7 near-cancellation with H11/H13 dominant is the characteristic "
            "signature of 12-pulse converter loads — typically large VFDs (>100 hp) using "
            "dual 6-pulse converters fed by a 30° phase-shifting transformer."
        ),
        "recommendation": (
            "12-pulse drives are already a harmonic mitigation measure. If TDD remains high, "
            "verify the phase-shifting transformer is balanced between the two rectifier bridges. "
            "Consider an active harmonic filter if further reduction is needed."
        ),
        "responsibility": "customer",
    },
    {
        "id": "drive_18pulse_afe",
        "family": "multipulse",
        "classes": {"sg", "pg"},
        "title": "18-pulse or active front-end (AFE) drive",
        "spectrum": [1, 1, 1, 1, 1, 1],
        "variability": "low",
        "cause": (
            "Near-flat, very low harmonic spectrum across all orders is characteristic of "
            "18-pulse drives or VFDs with active front-end rectifiers. These are premium "
            "'low-harmonic' drives designed to meet IEEE 519 at the equipment level."
        ),
        "recommendation": (
            "No action required — this load type is already low-harmonic. If TDD is still "
            "non-compliant, other load types at this service are the primary contributors."
        ),
        "responsibility": "customer",
    },
    {
        "id": "smps",
        "family": "single_phase_switchmode",
        "classes": {"r", "c", "sg", "pg"},
        "title": "Switched-mode power supplies (computers / servers / office equipment)",
        "spectrum": [35, 18, 9, 5, 3, 2],
        "variability": "low",
        "cause": (
            "H3-dominant spectrum with rapidly decaying odd harmonics is the signature of "
            "single-phase SMPS loads: computers, monitors, servers, and electronic ballasts. "
            "In 4-wire wye systems, triplen harmonics (H3, H9, H15) accumulate in the neutral."
        ),
        "recommendation": (
            "Survey single-phase nonlinear loads. Verify neutral conductor is rated for "
            "harmonic current (173% of phase conductor for heavily loaded SMPS environments). "
            "Consider K-rated or isolation transformers for concentrated SMPS loads."
        ),
        "responsibility": "customer",
    },
    {
        "id": "fluorescent_magnetic",
        "family": "single_phase_switchmode",
        "classes": {"c", "sg", "pg"},
        "title": "Fluorescent lighting (magnetic ballast)",
        "spectrum": [30, 12, 5, 2, 1, 1],
        "variability": "low",
        "cause": (
            "H3-dominant spectrum with steeper decay than SMPS is characteristic of "
            "fluorescent fixtures with magnetic (core-and-coil) ballasts — increasingly rare "
            "as T12/T8 magnetic ballasts are replaced with electronic ballasts or LEDs."
        ),
        "recommendation": (
            "Retrofit magnetic ballast fixtures with electronic ballasts or LED replacements. "
            "Reduces harmonic injection and improves energy efficiency simultaneously."
        ),
        "responsibility": "customer",
    },
    {
        "id": "led_poor_pf",
        "family": "single_phase_switchmode",
        "classes": {"r", "c", "sg", "pg"},
        "title": "LED drivers (poor power factor / no active PFC)",
        "spectrum": [40, 10, 4, 2, 1, 1],
        "variability": "low",
        "cause": (
            "Extremely H3-dominant spectrum with steep geometric decay is the signature of "
            "budget LED drivers lacking active power factor correction (PFC). Common in "
            "retrofit lamps, low-cost commercial fixtures, and residential LED bulbs."
        ),
        "recommendation": (
            "Specify LED fixtures with active PFC drivers (PF > 0.90, THD < 20%). "
            "IEC 61000-3-2 Class C applies to lighting — verify compliance on procurement."
        ),
        "responsibility": "customer",
    },
    {
        "id": "ev_charger_l2",
        "family": "mixed_single_phase",
        "classes": {"r", "c", "sg"},
        "title": "EV charger (Level 2 / AC charging)",
        "spectrum": [20, 15, 8, 4, 3, 2],
        "variability": "medium",
        "cause": (
            "Mixed triplen and 6k±1 signature reflects the single-phase on-board charger "
            "in most L2 EV charging (the EVSE is passive; the rectifier is in the vehicle). "
            "H3 contribution varies with charger design and battery state of charge."
        ),
        "recommendation": (
            "For co-located L2 chargers, consider managed charging and transformer sizing "
            "for harmonic current. For large EV fleets, evaluate 3-phase DC fast chargers "
            "with active PFC front-ends."
        ),
        "responsibility": "customer",
    },
    {
        "id": "ups_6pulse",
        "family": "six_pulse",
        "classes": {"c", "sg", "pg"},
        "title": "UPS (6-pulse double-conversion)",
        "spectrum": [2, 22, 10, 1, 4, 3],
        "variability": "low",
        "cause": (
            "6-pulse rectifier spectrum nearly identical to VFD-with-reactor. "
            "Double-conversion UPS units present a 6-pulse rectifier load on the utility "
            "input at all times, regardless of the downstream UPS output load."
        ),
        "recommendation": (
            "Verify UPS input filtering is in service. Modern UPS units with active PFC "
            "front-ends produce significantly lower input harmonic current — consult "
            "manufacturer specifications for input THD at rated load."
        ),
        "responsibility": "customer",
    },
    {
        "id": "welder_arc",
        "family": "arcing",
        "classes": {"c", "sg", "pg"},
        "title": "Arc welder / resistance welder",
        "spectrum": [10, 8, 6, 5, 4, 3],
        "variability": "high",
        "cause": (
            "Relatively flat harmonic spectrum with no single dominant order, combined with "
            "high inter-interval variability, is characteristic of arc welding equipment. "
            "Arc loads also generate even harmonics and subharmonics."
        ),
        "recommendation": (
            "Identify and schedule welding operations to minimize peak harmonic loading. "
            "For large welding loads, consider a series reactor or active power filter. "
            "If arc loads cause voltage flicker (PST > 1.0), coordinate with Xcel Energy."
        ),
        "responsibility": "customer",
    },
    {
        "id": "arc_furnace",
        "family": "arcing",
        "classes": {"pg"},
        "title": "Electric arc furnace (EAF) / plasma load",
        "spectrum": [15, 12, 9, 7, 5, 4],
        "variability": "high",
        "cause": (
            "Broad harmonic spectrum with very high variability and near-equal harmonic "
            "magnitudes across orders is characteristic of electric arc furnaces. EAFs also "
            "produce significant even harmonics, interharmonics, and voltage flicker."
        ),
        "recommendation": (
            "Large arc loads typically require a dedicated harmonic study and a static VAR "
            "compensator (SVC) or STATCOM. Coordinate with Xcel Energy — arc furnace "
            "installations require pre-approval under tariff requirements."
        ),
        "responsibility": "customer",
    },
    {
        "id": "transformer_saturation",
        "family": "triplen_saturation",
        "classes": {"r", "c", "sg", "pg"},
        "title": "Transformer saturation (overvoltage-induced)",
        "spectrum": [35, 8, 3, 1, 1, 1],
        "variability": "low",
        "cause": (
            "Very high H3 with rapidly decaying higher orders, correlated with elevated "
            "supply voltage rather than load magnitude, indicates transformer core saturation. "
            "Unlike SMPS-generated H3, saturation-sourced H3 is a utility-side phenomenon."
        ),
        "recommendation": (
            "Check supply voltage level — if consistently above +5% of nominal, contact "
            "Xcel Energy. Voltage regulation issues may be driving transformer saturation "
            "and harmonic injection. This may be a shared or utility responsibility."
        ),
        "responsibility": "shared",
    },
    {
        "id": "dc_fast_charger",
        "family": "six_pulse",
        "classes": {"c", "sg", "pg"},
        "title": "DC fast charger (DCFC / Level 3, 6-pulse front-end)",
        "spectrum": [3, 25, 10, 1, 5, 4],
        "variability": "medium",
        "cause": (
            "6-pulse rectifier spectrum with slightly elevated H3 (vs. VFD) is typical of "
            "DC fast chargers using 6-pulse front-end rectifiers. Variability is medium — "
            "charger power varies with battery state of charge over the session."
        ),
        "recommendation": (
            "Modern DCFC units use 12-pulse or active PFC front-ends — verify equipment "
            "specifications before installation. For high-power charger clusters, commission "
            "a harmonic study to assess PCC impact."
        ),
        "responsibility": "customer",
    },
    {
        "id": "mixed_vfd_smps",
        "family": "mixed_three_phase",
        "classes": {"c", "sg", "pg"},
        "title": "Mixed load: 6-pulse VFDs + single-phase nonlinear loads",
        "spectrum": [15, 20, 8, 2, 4, 3],
        "variability": "low",
        "cause": (
            "H5 dominant over H3 (6k±1 VFD pattern) combined with a significant H3 component "
            "(triplen from SMPS/computers/lighting) indicates a mixed load environment. "
            "This is the most common harmonic profile for commercial and light-industrial "
            "customers: 3-phase VFDs or rectifiers plus single-phase office equipment."
        ),
        "recommendation": (
            "Address both harmonic sources: (1) add input reactors or upgrade to multi-pulse "
            "drives for 3-phase VFD loads, and (2) verify neutral conductor sizing for triplen "
            "harmonic current from single-phase loads (computers, LED lighting). "
            "Consider a K-rated transformer if K-factor exceeds 4."
        ),
        "responsibility": "customer",
    },
    # ── Residential and small-commercial loads ───────────────────────────────
    # The library was built around industrial equipment, so a residential
    # service had nothing plausible to match and was handed the nearest
    # industrial neighbour.  The three below cover what actually drives
    # distortion on a house: air conditioning, EV charging (above), and
    # rooftop PV.
    #
    # Their spectra are engineering estimates from the topology of each load,
    # not measurements from a validated dataset -- the same standing as the
    # rest of this table.  Treat a match as a hypothesis to check on site.
    {
        "id": "pv_inverter",
        "family": "near_linear",
        "classes": {"r", "c", "sg"},
        "title": "Rooftop PV inverter (grid-following DER)",
        "spectrum": [8, 6, 4, 2, 2, 1],
        "variability": "high",
        "cause": (
            "Low-order odd harmonics with no single dominant order, varying "
            "strongly through the day, is consistent with a grid-following PV "
            "inverter. IEEE 1547 holds inverters to under 5% TDD at rated "
            "output, but the percentage rises at low irradiance because the "
            "fundamental falls while the harmonic floor does not — so early "
            "morning and late afternoon read worst. Distortion that tracks "
            "daylight and disappears overnight points here rather than to a "
            "load."
        ),
        "recommendation": (
            "Confirm whether DER is interconnected at this service and review "
            "the inverter's IEEE 1547 test report. Check that measured "
            "distortion follows the solar day; if it does not, the inverter is "
            "probably not the source. Verify the inverter firmware is current, "
            "as harmonic and anti-islanding behaviour are often improved in "
            "later revisions."
        ),
        "responsibility": "customer",
    },
    {
        "id": "ac_compressor_single_phase",
        "family": "near_linear",
        "classes": {"r", "c"},
        "title": "Air conditioning / heat pump (fixed-speed compressor)",
        "spectrum": [6, 4, 2, 1, 1, 1],
        "variability": "medium",
        "cause": (
            "Low distortion dominated by H3, cycling on and off over tens of "
            "minutes, is characteristic of a fixed-speed single-phase "
            "compressor — a largely linear inductive load whose modest "
            "harmonics come from motor and any transformer saturation. Expect "
            "the current to step rather than ramp, and a starting inrush that "
            "can depress voltage briefly at each start."
        ),
        "recommendation": (
            "Normally no harmonic mitigation is warranted; this load type is "
            "not a significant distortion source. If starting dips are the "
            "complaint, evaluate service conductor size and transformer "
            "sizing, or a soft-start on the compressor, rather than filters."
        ),
        "responsibility": "customer",
    },
    {
        "id": "heat_pump_inverter",
        "family": "mixed_single_phase",
        "classes": {"r", "c"},
        "title": "Inverter-driven heat pump / mini-split",
        "spectrum": [12, 14, 6, 2, 3, 2],
        "variability": "medium",
        "cause": (
            "Comparable H3 and H5 with a modest H7 tail is consistent with a "
            "variable-speed heat pump or ductless mini-split, which is a small "
            "single-phase drive: a rectifier front end gives it the H5 of a "
            "VFD, while its single-phase supply keeps H3 present. Unlike a "
            "fixed-speed compressor it modulates continuously rather than "
            "cycling."
        ),
        "recommendation": (
            "No mitigation is normally justified at residential scale. Where "
            "several units share one transformer, check that H3 is not "
            "accumulating on the neutral, since triplens from single-phase "
            "loads add rather than cancel."
        ),
        "responsibility": "customer",
    },
]


def _h519_class_idx(isc_il: float) -> int:
    """Return 0-based class index into _H519_LIMITS sublists."""
    for i, (threshold, _) in enumerate(_TDD_TABLE):
        if isc_il < threshold:
            return i
    return 4


def _h519_limit(h: int, isc_il: float) -> float:
    """Return per-order IEEE 519-2022 limit (% of IL) for harmonic h at given ISC/IL."""
    cls = _h519_class_idx(isc_il)
    for h_min, h_max, limits in _H519_LIMITS:
        if h_min <= h < h_max:
            return limits[cls]
    return 0.0  # harmonic order out of scope


def _tdd_limit(isc_il: float) -> float:
    """Return IEEE 519-2022 TDD limit (%) for the given ISC/IL ratio."""
    for threshold, limit in _TDD_TABLE:
        if isc_il < threshold:
            return limit
    return 20.0


def _tdd_class(isc_il: float) -> str:
    """Return the ISC/IL class label string for display."""
    for threshold, _ in _TDD_TABLE:
        if isc_il < threshold:
            return f"< {threshold}"
    return "≥ 1000"


# ── Xcel Energy Blue Book fault current tables ────────────────────────────────
# Source: "Standard for Electric Installation and Use", effective 2026-02-15.
# All values = RMS symmetrical fault current (A) at the transformer secondary
# terminals. No source or secondary conductor impedance is included — values
# represent the maximum (worst-case for equipment rating) ISC.
# Key: (service_type, kva, secondary_line_voltage) → isc_amps

_BLUE_BOOK_ISC: Dict[Tuple[str, int, int], int] = {
    # ── Table IA: Single-phase overhead transformers ─────────────────────────
    # 120 V secondary (%Z = 1.9)
    ("1ph-overhead",  10, 120):  4_300,
    ("1ph-overhead",  15, 120):  6_500,
    ("1ph-overhead",  25, 120): 10_900,
    ("1ph-overhead",  50, 120): 21_700,
    ("1ph-overhead",  75, 120): 32_600,
    ("1ph-overhead", 100, 120): 43_400,
    ("1ph-overhead", 150, 120): 65_100,
    ("1ph-overhead", 167, 120): 72_500,
    # 240 V secondary (%Z = 1.4)
    ("1ph-overhead",  10, 240):  2_900,
    ("1ph-overhead",  15, 240):  4_400,
    ("1ph-overhead",  25, 240):  7_400,
    ("1ph-overhead",  50, 240): 14_800,
    ("1ph-overhead",  75, 240): 22_200,
    ("1ph-overhead", 100, 240): 29_600,
    ("1ph-overhead", 150, 240): 44_400,
    ("1ph-overhead", 167, 240): 49_400,

    # ── Table IB: Single-phase pad-mounted transformers ──────────────────────
    # 240 V secondary (%Z = 1.4)
    ("1ph-padmount",  25, 240):  7_400,
    ("1ph-padmount",  50, 240): 14_800,
    ("1ph-padmount", 100, 240): 29_600,
    ("1ph-padmount", 150, 240): 44_400,
    ("1ph-padmount", 167, 240): 49_400,

    # ── Table II: Three-phase pad-mounted transformers ───────────────────────
    # 277/480 V secondary
    ("3ph-padmount",   75, 480):  5_600,
    ("3ph-padmount",  150, 480): 11_200,
    ("3ph-padmount",  300, 480): 22_500,
    ("3ph-padmount",  500, 480): 33_400,
    ("3ph-padmount",  750, 480): 16_900,   # higher %Z (5.32%) — non-monotonic
    ("3ph-padmount", 1000, 480): 22_600,
    ("3ph-padmount", 1500, 480): 33_900,
    ("3ph-padmount", 2000, 480): 45_200,
    ("3ph-padmount", 2500, 480): 56_500,
    # 120/240 V secondary
    ("3ph-padmount",   75, 240): 11_100,
    ("3ph-padmount",  150, 240): 21_800,
    ("3ph-padmount",  300, 240): 42_300,
    ("3ph-padmount",  500, 240): 60_900,
    ("3ph-padmount",  750, 240): 32_300,
    ("3ph-padmount", 1000, 240): 42_400,
    # 120/208 V secondary
    ("3ph-padmount",   75, 208): 13_000,
    ("3ph-padmount",  150, 208): 26_000,
    ("3ph-padmount",  300, 208): 52_000,
    ("3ph-padmount",  500, 208): 77_100,
    ("3ph-padmount",  750, 208): 39_100,
    ("3ph-padmount", 1000, 208): 52_100,
    ("3ph-padmount", 1500, 208): 78_200,

    # ── Table III: Three-phase overhead wye-connected transformer banks ──────
    # 277/480 V secondary (%Z = 1.4 for all)
    ("3ph-overhead-wye",  45, 480):  3_800,
    ("3ph-overhead-wye",  75, 480):  6_400,
    ("3ph-overhead-wye", 150, 480): 12_800,
    ("3ph-overhead-wye", 300, 480): 25_700,
    ("3ph-overhead-wye", 500, 480): 42_900,
    # 120/208 V secondary
    ("3ph-overhead-wye",  45, 208):  8_900,
    ("3ph-overhead-wye",  75, 208): 14_800,
    ("3ph-overhead-wye", 150, 208): 29_700,
    ("3ph-overhead-wye", 300, 208): 59_400,
    ("3ph-overhead-wye", 500, 208): 99_100,

    # ── Tables IV & V: Three-phase overhead delta banks ──────────────────────
    # Open delta (Table IV) and closed delta (Table V) use single-phase
    # overhead transformer impedance values from Table IA. Representative
    # totals for the most common balanced configurations are listed here;
    # unbalanced or mixed-size banks require manual calculation per Table IA.
    #
    # Open delta — 120/240 V and 240/480 V (power + lighting units)
    ("3ph-open-delta",  20, 240):  6_144,   # 10+10 kVA
    ("3ph-open-delta",  35, 240):  9_935,   # 10+25 kVA
    ("3ph-open-delta",  60, 240): 16_944,   # 10+50 kVA
    ("3ph-open-delta",  85, 240): 24_166,   # 10+75 kVA
    ("3ph-open-delta", 110, 240): 30_724,   # 10+100 kVA
    ("3ph-open-delta", 117, 240): 31_455,   # 10+107? skip; use 25+50=75 below
    ("3ph-open-delta",  50, 240): 15_362,   # 25+25 kVA
    ("3ph-open-delta",  75, 240): 21_514,   # 25+50 kVA
    ("3ph-open-delta", 100, 240): 28_253,   # 25+75 kVA
    ("3ph-open-delta", 125, 240): 35_244,   # 25+100 kVA
    ("3ph-open-delta", 192, 240): 54_468,   # 25+167 kVA
    ("3ph-open-delta", 100, 240): 30_724,   # 50+50 kVA (overwritten by 25+75 — use explicit key if needed)
    ("3ph-open-delta", 334, 240): 102_618,  # 167+167 kVA
    # Open delta 480 V (half of 240 V values)
    ("3ph-open-delta",  20, 480):  3_072,
    ("3ph-open-delta",  35, 480):  4_968,
    ("3ph-open-delta",  60, 480):  8_472,
    ("3ph-open-delta",  85, 480): 12_083,
    ("3ph-open-delta", 334, 480): 51_309,
    # Closed delta — 120/240 V (Table V)
    ("3ph-closed-delta",  20, 240):  6_782,   # 10+10+10 kVA
    ("3ph-closed-delta",  35, 240): 12_757,   # 10+10+25 kVA → use 25+25+25 below
    ("3ph-closed-delta",  75, 240): 25_515,   # 10+10+50 or 25+25+25
    ("3ph-closed-delta", 150, 240): 51_031,   # 10+10+100 or 50+50+50
    ("3ph-closed-delta", 251, 240): 85_221,   # 10+10+167
    ("3ph-closed-delta", 300, 240): 67_826,   # 100+100+100 kVA
    ("3ph-closed-delta", 501, 240): 113_270,  # 167+167+167 kVA
    # Closed delta 480 V (half of 240 V)
    ("3ph-closed-delta",  20, 480):  3_391,
    ("3ph-closed-delta",  75, 480): 12_758,
    ("3ph-closed-delta", 150, 480): 25_516,
    ("3ph-closed-delta", 300, 480): 33_913,
    ("3ph-closed-delta", 501, 480): 56_635,
}

# Table IX: typical transformer impedance ranges — (min_pct, max_pct)
_BLUE_BOOK_IMPEDANCE: Dict[str, List[Tuple[int, int, float, float]]] = {
    # (kva_min, kva_max, z_min_pct, z_max_pct)
    "1ph-overhead": [
        (10,  75,  1.6, 2.4),
        (100, 100, 1.6, 2.8),
        (167, 167, 1.8, 3.2),
        (250, 333, 5.3, 6.2),
    ],
    "1ph-padmount": [
        (10,  75,  1.4, 2.4),
        (100, 100, 1.6, 2.4),
        (167, 167, 1.7, 2.8),
        (250, 250, 5.3, 6.2),
    ],
    "3ph-overhead-wye": [
        (10,   75,  1.6, 2.4),
        (150,  150, 1.6, 2.4),
        (500,  500, 1.8, 3.2),
        (750, 2500, 5.3, 6.2),
    ],
    "3ph-padmount": [
        (10,   75,  1.6, 2.4),
        (150,  150, 1.6, 2.4),
        (300,  300, 1.6, 2.8),
        (500,  500, 1.8, 3.2),
        (750, 2500, 5.3, 6.2),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SERVICE CONDUCTOR IMPEDANCE
#
# GENERIC PUBLISHED VALUES, NOT PSCo BLUE BOOK VALUES. Resistance is NEC
# Chapter 9 Table 8 (stranded, uncoated, 75 °C); reactance is a typical value
# for the construction, since Table 9's conduit figures do not describe a
# triplex drop. Every expected-impedance figure the report prints from this
# table is labelled as generic, so nobody reads the comparison as tighter than
# it is.
#
# To replace a row with a Blue Book figure, edit the two numbers here: they are
# ohms per 1000 ft per conductor at 75 °C and 60 Hz, and nothing else in the
# code carries conductor constants.
# ─────────────────────────────────────────────────────────────────────────────

#: key → (label, R Ω/1000 ft, X Ω/1000 ft)
_CONDUCTOR_TABLE: Dict[str, Tuple[str, float, float]] = {
    # Overhead triplex/quadruplex aluminum — the usual residential drop.
    "al-2-triplex":    ("#2 AL triplex (overhead drop)",        0.319, 0.035),
    "al-1-0-triplex":  ("1/0 AL triplex (overhead drop)",       0.201, 0.035),
    "al-2-0-triplex":  ("2/0 AL triplex (overhead drop)",       0.159, 0.035),
    "al-4-0-triplex":  ("4/0 AL triplex (overhead drop)",       0.100, 0.035),
    # Underground residential distribution — direct-buried aluminum.
    "al-2-urd":        ("#2 AL URD (underground service)",      0.319, 0.030),
    "al-1-0-urd":      ("1/0 AL URD (underground service)",     0.201, 0.030),
    "al-4-0-urd":      ("4/0 AL URD (underground service)",     0.100, 0.030),
    "al-350-urd":      ("350 kcmil AL URD (underground)",       0.0611, 0.030),
    "al-500-urd":      ("500 kcmil AL URD (underground)",       0.0424, 0.030),
    # Copper, for older services.
    "cu-4":            ("#4 CU service conductor",              0.308, 0.050),
    "cu-2":            ("#2 CU service conductor",              0.194, 0.050),
    "cu-1-0":          ("1/0 CU service conductor",             0.122, 0.050),
}


def conductor_options() -> List[Tuple[str, str]]:
    """(key, label) pairs for the picker, in table order."""
    return [(key, label) for key, (label, _r, _x) in _CONDUCTOR_TABLE.items()]


def conductor_impedance(key: str, length_ft: float,
                        return_path: bool = True) -> Optional[Tuple[float, float]]:
    """(R, X) in ohms for *length_ft* of this conductor, as the load sees it.

    A line-to-neutral load's current goes out on a phase conductor and back on
    the neutral, so the impedance in its voltage drop is both conductors, not
    one. ``return_path=False`` gives the one-way value, which is what a
    balanced three-phase load sees because its neutral carries almost nothing.
    """
    row = _CONDUCTOR_TABLE.get(key)
    if row is None or not length_ft or length_ft <= 0:
        return None
    _label, r_per_kft, x_per_kft = row
    conductors = 2.0 if return_path else 1.0
    scale = conductors * length_ft / 1000.0
    return r_per_kft * scale, x_per_kft * scale


def conductor_label(key: Optional[str]) -> Optional[str]:
    row = _CONDUCTOR_TABLE.get(key or "")
    return row[0] if row else None


def primary_line_impedance(thresh: "Thresholds") -> Optional[dict]:
    """Sequence impedance of the primary line, for a primary-metered service.

    Entered rather than derived: a primary line's impedance comes off a
    planning model or a fault study, and there is no table here that could
    reproduce one honestly from a conductor size alone.

    Positive sequence is what balanced load current flows in, so Z1 is what the
    measured impedance is compared against and the only part that is required.
    Zero sequence is optional and carried through for the two places it is the
    right number rather than Z1: triplen harmonics, which are zero-sequence on
    a balanced system, and the earth-return path of unbalanced current. Negative
    sequence is not asked for because a passive line has Z2 = Z1.
    """
    r1, x1 = thresh.primary_r1_ohm, thresh.primary_x1_ohm
    if r1 is None and x1 is None:
        return None
    r1 = float(r1 or 0.0)
    x1 = float(x1 or 0.0)
    out = {
        "r1_ohm": r1,
        "x1_ohm": x1,
        "z1_ohm": float(_np.hypot(r1, x1)),
    }
    r0, x0 = thresh.primary_r0_ohm, thresh.primary_x0_ohm
    if r0 is not None or x0 is not None:
        r0 = float(r0 or 0.0)
        x0 = float(x0 or 0.0)
        out.update({
            "r0_ohm": r0,
            "x0_ohm": x0,
            "z0_ohm": float(_np.hypot(r0, x0)),
        })
        if out["z1_ohm"] > 0:
            out["z0_over_z1"] = out["z0_ohm"] / out["z1_ohm"]
        # A single-phase load tapped off this line sees the phase out and the
        # earth back, which is (2*Z1 + Z0)/3 rather than Z1 alone.
        out["single_phase_loop_ohm"] = float(
            _np.hypot((2 * r1 + r0) / 3.0, (2 * x1 + x0) / 3.0))
    return out


# Service type → human label for report display
_SERVICE_TYPE_LABEL: Dict[str, str] = {
    "1ph-overhead":      "Single-phase overhead",
    "1ph-padmount":      "Single-phase pad-mounted",
    # Two legs of a 208Y/120 wye, not a center-tapped single-phase transformer.
    # Common in condos and apartments fed from a three-phase service.
    "1ph-208":       "Single-phase 120/208 (two legs of a three-phase transformer)",
    "3ph-padmount":      "Three-phase pad-mounted",
    "3ph-overhead-wye":  "Three-phase overhead wye bank",
    "3ph-open-delta":    "Three-phase overhead open delta bank",
    "3ph-closed-delta":  "Three-phase overhead closed delta bank",
}


# Nominal voltage as entered → the secondary line voltages that could key it,
# best first. Only 120 and 277 are line-to-neutral readings of a wye secondary
# (208Y/120, 480Y/277); 208, 240 and 480 are already the secondary line voltage,
# since no PSCo distribution secondary is √3 above them. A 120 V pick on a delta
# bank is the center-tapped leg of a 120/240 V secondary, hence the 240 fallback.
_THREE_PHASE_SECONDARY: Dict[int, Tuple[int, ...]] = {
    120: (208, 240),
    208: (208, 240),
    240: (240,),
    277: (480,),
    480: (480,),
}


def _secondary_candidates(service_type: str, nominal_v: float) -> Tuple[int, ...]:
    """Secondary line voltages that could key this service, best match first."""
    if service_type in SINGLE_PHASE_208_TYPES:
        # Two legs of a wye: the secondary is the wye's line voltage (208), not
        # the 240 of a center-tapped single-phase can.
        nearest = min(_THREE_PHASE_SECONDARY, key=lambda v: abs(v - nominal_v))
        return _THREE_PHASE_SECONDARY[nearest]
    if service_type.startswith("1ph"):
        # A 120 V L-N service is one leg of a 120/240 V split-phase secondary.
        return (120, 240) if nominal_v <= 150 else (240,)
    nearest = min(_THREE_PHASE_SECONDARY, key=lambda v: abs(v - nominal_v))
    return _THREE_PHASE_SECONDARY[nearest]


#: Single-phase services taken from two legs of a three-phase 120/208
#: transformer, as opposed to a center-tapped 120/240 single-phase one.  The
#: legs sit 120 degrees apart rather than 180, so line-to-line is sqrt(3) x
#: line-to-neutral (208 V) instead of 2x (240 V), and the neutral carries the
#: vector sum of the two legs rather than their difference.
SINGLE_PHASE_208_TYPES = frozenset({"1ph-208"})

#: A single-phase 120/208 service is fed from the same three-phase transformer
#: as a full three-phase 120/208 service -- the customer simply pulls fewer
#: wires -- so its Blue Book fault current comes from the three-phase rows.
_1PH_208_ISC_PROXY = "3ph-padmount"


def is_single_phase_208(service_type: Optional[str]) -> bool:
    """True for a single-phase service taken from a three-phase wye."""
    return (service_type or "") in SINGLE_PHASE_208_TYPES


def ll_factor(service_type: Optional[str] = None, topology: str = "auto") -> float:
    """Ratio of line-to-line to line-to-neutral voltage for this service.

    2.0 for a center-tapped single-phase secondary (120/240), sqrt(3) for
    anything derived from a wye -- including a single-phase service taken from
    two legs of one (120/208).  Getting this wrong misreports a 208 V service
    as a 240 V service, a 13% error that reads as a severe undervoltage.
    """
    svc = service_type or ""
    if is_single_phase_208(svc):
        return 3.0 ** 0.5
    if svc.startswith("1ph") or topology == "split-phase":
        return 2.0
    return 3.0 ** 0.5


def isc_lookup_type(service_type: Optional[str]) -> str:
    """Blue Book row family to read for this service.

    A single-phase 120/208 service has no rows of its own: it is served by the
    same transformer as a three-phase 120/208 service, so it reads the same
    rows.  Both the ISC lookup and the GUI's kVA list go through here so they
    cannot disagree.
    """
    svc = service_type or ""
    return _1PH_208_ISC_PROXY if is_single_phase_208(svc) else svc


def _infer_secondary_v(service_type: str, nominal_v: float) -> int:
    """Convert the entered nominal voltage to the secondary voltage used as table key.

    Returns the first candidate the Blue Book actually carries rows for, so the
    kVA list, the ISC label and the analysis run all read the same table entry.
    """
    available = {k[2] for k in _BLUE_BOOK_ISC if k[0] == service_type}
    candidates = _secondary_candidates(service_type, nominal_v)
    for cand in candidates:
        if cand in available:
            return cand
    return candidates[0]


def _lookup_isc(service_type: str, kva: float, nominal_v: float) -> Optional[Tuple[int, str]]:
    """
    Look up ISC from Blue Book tables.
    Returns (isc_amps, note_string) or None if not found.
    The note identifies which table entry was used.
    """
    # A single-phase 120/208 service shares its transformer with a three-phase
    # 120/208 service, so its available fault current comes from the three-phase
    # rows -- reading the single-phase (120/240) rows would give the wrong ISC
    # and therefore the wrong IEEE 519 Table 2 class.
    lookup_type = isc_lookup_type(service_type)
    secondary_v = _infer_secondary_v(lookup_type, nominal_v)
    kva_int = int(round(kva))
    key = (lookup_type, kva_int, secondary_v)
    isc = _BLUE_BOOK_ISC.get(key)
    if isc is not None:
        label = _SERVICE_TYPE_LABEL.get(service_type, service_type)
        note = (f"Blue Book Table — {label}, {kva_int} kVA, "
                f"{secondary_v}V secondary → {isc:,} A at transformer terminals")
        return isc, note

    # Try finding the nearest kVA in the same service type / voltage
    candidates = {k[1]: v for k, v in _BLUE_BOOK_ISC.items()
                  if k[0] == lookup_type and k[2] == secondary_v}
    if candidates:
        nearest_kva = min(candidates, key=lambda k: abs(k - kva_int))
        isc = candidates[nearest_kva]
        label = _SERVICE_TYPE_LABEL.get(service_type, service_type)
        note = (f"Blue Book Table (nearest kVA={nearest_kva}) — {label}, "
                f"{secondary_v}V secondary → {isc:,} A at transformer terminals")
        return isc, note

    return None


def expected_service_impedance(thresh: "Thresholds") -> dict:
    """What the impedance from the source to the meter ought to be.

    Built from what the engineer already picks at the start, in two parts:

    * everything upstream of the service conductors. Where the Blue Book ISC
      is known it is the honest number for this, because a fault current at
      the transformer terminals already contains the primary system and the
      transformer together: Z = V_LN / ISC. Without it, the transformer's own
      impedance range is used and the total is a floor rather than an
      estimate, since the primary system is then missing from it.
    * the service conductors, from the picked type and run length.

    Returns the parts as well as the total: an engineer comparing a measured
    figure against this needs to see which term dominates it.
    """
    v_ln = thresh.nominal_voltage
    single_phase = (thresh.topology == "split-phase"
                    or (thresh.service_type or "").startswith("1ph"))
    out: dict = {
        "available": False,
        "single_phase": single_phase,
        "conductor_label": conductor_label(thresh.conductor_key),
        "run_length_ft": thresh.run_length_ft,
        "generic_conductor_constants": True,
        "primary_metered": bool(thresh.primary_metered),
    }

    # ── metered on the high side ─────────────────────────────────────────────
    # The transformer and everything below it belong to the customer and sit
    # downstream of the meter, so neither is in the path this recording sees.
    # What is in it is the primary line, which is entered rather than looked up.
    if thresh.primary_metered:
        primary = primary_line_impedance(thresh)
        if primary is None:
            out["reason"] = (
                "No expected impedance: this service is metered on the primary, "
                "so the expected value is the primary line impedance to the "
                "metering point, and no R1/X1 was entered."
            )
            return out
        out["primary"] = primary
        out["sequence_used"] = "positive"
        out["available"] = True
        out["upstream_ohm"] = primary["z1_ohm"]
        out["upstream_source"] = (
            f"the primary line to the metering point, as entered: "
            f"R1 {primary['r1_ohm']:.4f} Ω, X1 {primary['x1_ohm']:.4f} Ω"
        )
        out["total_ohm"] = primary["z1_ohm"]
        return out

    # ── upstream of the service conductors ───────────────────────────────────
    z_upstream: Optional[float] = None
    z_transformer: Optional[Tuple[float, float]] = None
    if thresh.transformer_kva and thresh.service_type:
        pct = _impedance_range(thresh.service_type, thresh.transformer_kva)
        if pct:
            # V²/S on the base the meter sees: for a 120/240 service the L-N
            # path is through half the winding, which is what V_LN²/S gives.
            s_phase = thresh.transformer_kva * 1000.0 / (1.0 if single_phase else 3.0)
            z_transformer = tuple(p / 100.0 * v_ln ** 2 / s_phase for p in pct)
            out["transformer_pct"] = pct
            out["transformer_ohm_range"] = z_transformer

    if thresh.isc_amps:
        z_upstream = v_ln / float(thresh.isc_amps)
        out["upstream_ohm"] = z_upstream
        out["upstream_source"] = (
            f"V_LN / ISC ({thresh.isc_amps:,.0f} A) — the primary system and "
            "the transformer together, as the Blue Book fault current has them"
        )
    elif z_transformer:
        z_upstream = sum(z_transformer) / 2.0
        out["upstream_ohm"] = z_upstream
        out["upstream_source"] = (
            "the transformer impedance range alone; no ISC was resolved, so "
            "the primary system is not included and the total below is a floor"
        )
        out["upstream_is_floor"] = True

    # ── the service conductors ───────────────────────────────────────────────
    conductor = conductor_impedance(thresh.conductor_key or "",
                                    thresh.run_length_ft or 0.0,
                                    return_path=single_phase)
    if conductor:
        r_c, x_c = conductor
        out["conductor_r_ohm"] = r_c
        out["conductor_x_ohm"] = x_c
        out["conductor_z_ohm"] = float(_np.hypot(r_c, x_c))
        out["conductor_path"] = (
            "phase and neutral, since a line-to-neutral load's current returns "
            "on the neutral" if single_phase else
            "one way, since a balanced three-phase load's neutral carries "
            "almost nothing"
        )

    # ── the shared secondary main, where the service taps one ────────────────
    # This customer's current flows through it on the way to the transformer,
    # so its drop is in the measurement whether or not anyone entered it. It is
    # counted the same way as the service conductor, since the return path is
    # the same shared neutral.
    shared = conductor_impedance(thresh.shared_secondary_key or "",
                                 thresh.shared_secondary_ft or 0.0,
                                 return_path=single_phase)
    if shared:
        r_s, x_s = shared
        out["shared_secondary_r_ohm"] = r_s
        out["shared_secondary_x_ohm"] = x_s
        out["shared_secondary_z_ohm"] = float(_np.hypot(r_s, x_s))
        out["shared_secondary_label"] = conductor_label(thresh.shared_secondary_key)
        out["shared_secondary_ft"] = thresh.shared_secondary_ft

    if z_upstream is None and conductor is None and shared is None:
        out["reason"] = (
            "No expected impedance: it needs the transformer kVA or the "
            "short-circuit current, and the service conductor type and run "
            "length."
        )
        return out

    out["available"] = True
    out["total_ohm"] = ((z_upstream or 0.0)
                        + (out.get("shared_secondary_z_ohm") or 0.0)
                        + (out.get("conductor_z_ohm") or 0.0))
    missing = []
    if z_upstream is None:
        missing.append("the transformer and primary system")
    if conductor is None:
        missing.append("the service conductors")
    if missing:
        out["partial"] = " and ".join(missing)
    return out


def _impedance_range(service_type: str, kva: float) -> Optional[Tuple[float, float]]:
    """Return (z_min_pct, z_max_pct) from Table IX for the given service type and kVA."""
    rows = _BLUE_BOOK_IMPEDANCE.get(service_type, [])
    for kva_min, kva_max, z_min, z_max in rows:
        if kva_min <= kva <= kva_max:
            return z_min, z_max
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ITIC (CBEMA) VOLTAGE TOLERANCE ENVELOPE
# Reference: "ITI (CBEMA) Curve Application Note," Information Technology
# Industry Council (ITIC), 2000.  Referenced by IEEE 1159-2019 as the standard
# voltage tolerance envelope for information technology equipment.
# Applicable to 120 V nominal (120/208 V and 120/240 V, 60 Hz systems).
# ─────────────────────────────────────────────────────────────────────────────

# Step-function boundary lines (duplicate x-values create vertical segments)
_ITIC_UPPER_MS_STEP  = _np.array([0.001, 1,   1,   3,   3,   20,  20,  500, 500, 1e6])
_ITIC_UPPER_PCT_STEP = _np.array([500,   500, 200, 200, 140, 140, 120, 120, 110, 110])
_ITIC_LOWER_MS_STEP  = _np.array([0.001, 20,  20,  500, 500, 1e4, 1e4, 1e6])
_ITIC_LOWER_PCT_STEP = _np.array([0,     0,   70,  70,  80,  80,  90,  90 ])


def _itic_upper_v(x: "_np.ndarray") -> "_np.ndarray":
    """ITIC upper boundary (% nominal) at each duration x (ms)."""
    r = _np.full_like(x, 110.0, dtype=float)
    r[x < 500] = 120.0
    r[x < 20]  = 140.0
    r[x < 3]   = 200.0
    r[x < 1]   = 500.0
    return r


def _itic_lower_v(x: "_np.ndarray") -> "_np.ndarray":
    """ITIC lower boundary (% nominal) at each duration x (ms)."""
    r = _np.full_like(x, 90.0, dtype=float)
    r[x < 10000] = 80.0
    r[x < 500]   = 70.0
    r[x < 20]    = 0.0
    return r

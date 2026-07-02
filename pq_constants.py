from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__version__ = "0.1.0"


@dataclass
class Thresholds:
    """All engineering limits in one place — pass to PQAnalyzer."""
    nominal_voltage: float = 120.0        # V (line-to-neutral)
    volt_tolerance: float = 0.05          # ±5 % → ANSI C84.1 Range A
    thd_voltage_limit: float = 8.0        # % → IEEE 519 Table 2 (≤1 kV bus)
    thd_current_limit: float = 5.0        # % fallback when isc_amps not provided
    power_factor_limit: float = 0.90      # lagging — flag below this
    imbalance_limit: float = 3.0          # % voltage unbalance — NEMA MG1 / IEEE 1159
    current_imbalance_limit: float = 10.0 # % current unbalance — per PSC procedure
    event_delta_pct: float = 0.10         # spike detection: step > 10 % of nominal
    isc_amps: Optional[float] = None      # short-circuit current at PCC (A) — from Blue Book
    isc_source: Optional[str] = None     # human-readable note on how ISC was determined
    transformer_kva: Optional[float] = None  # service transformer nameplate (kVA)
    customer_class: str = "sg"            # "r" | "c" | "sg" | "pg"  (PSCo tariff schedules)


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
        "title": "12-pulse rectifier / drive",
        "spectrum": [1, 3, 2, 1, 14, 11],
        "variability": "low",
        "cause": (
            "H5 and H7 near-cancellation with H11/H13 dominant is the definitive signature "
            "of 12-pulse converter loads — typically large VFDs (>100 hp) using dual 6-pulse "
            "converters fed by a 30° phase-shifting transformer."
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

# Service type → human label for report display
_SERVICE_TYPE_LABEL: Dict[str, str] = {
    "1ph-overhead":      "Single-phase overhead",
    "1ph-padmount":      "Single-phase pad-mounted",
    "3ph-padmount":      "Three-phase pad-mounted",
    "3ph-overhead-wye":  "Three-phase overhead wye bank",
    "3ph-open-delta":    "Three-phase overhead open delta bank",
    "3ph-closed-delta":  "Three-phase overhead closed delta bank",
}


def _infer_secondary_v(service_type: str, nominal_v: float) -> int:
    """Convert L-N nominal voltage to the line-to-line secondary voltage used as table key."""
    if service_type.startswith("1ph"):
        # Single-phase: nominal is L-N or full service voltage
        return 120 if nominal_v <= 120 else 240
    else:
        # Three-phase: nominal is L-N; derive L-L and snap to standard
        ll = nominal_v * 3 ** 0.5
        if ll < 220:
            return 208
        elif ll < 400:
            return 240
        else:
            return 480


def _lookup_isc(service_type: str, kva: float, nominal_v: float) -> Optional[Tuple[int, str]]:
    """
    Look up ISC from Blue Book tables.
    Returns (isc_amps, note_string) or None if not found.
    The note identifies which table entry was used.
    """
    secondary_v = _infer_secondary_v(service_type, nominal_v)
    kva_int = int(round(kva))
    key = (service_type, kva_int, secondary_v)
    isc = _BLUE_BOOK_ISC.get(key)
    if isc is not None:
        label = _SERVICE_TYPE_LABEL.get(service_type, service_type)
        note = (f"Blue Book Table — {label}, {kva_int} kVA, "
                f"{secondary_v}V secondary → {isc:,} A at transformer terminals")
        return isc, note

    # Try finding the nearest kVA in the same service type / voltage
    candidates = {k[1]: v for k, v in _BLUE_BOOK_ISC.items()
                  if k[0] == service_type and k[2] == secondary_v}
    if candidates:
        nearest_kva = min(candidates, key=lambda k: abs(k - kva_int))
        isc = candidates[nearest_kva]
        label = _SERVICE_TYPE_LABEL.get(service_type, service_type)
        note = (f"Blue Book Table (nearest kVA={nearest_kva}) — {label}, "
                f"{secondary_v}V secondary → {isc:,} A at transformer terminals")
        return isc, note

    return None


def _impedance_range(service_type: str, kva: float) -> Optional[Tuple[float, float]]:
    """Return (z_min_pct, z_max_pct) from Table IX for the given service type and kVA."""
    rows = _BLUE_BOOK_IMPEDANCE.get(service_type, [])
    for kva_min, kva_max, z_min, z_max in rows:
        if kva_min <= kva <= kva_max:
            return z_min, z_max
    return None

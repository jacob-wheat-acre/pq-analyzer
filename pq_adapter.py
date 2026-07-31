from __future__ import annotations

import logging
import re
import struct
import sys
import warnings
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import pqdif
from pq_constants import _H519_ORDERS

# PQDIF parsing is handled by pqdif.py (this repo), written directly from
# IEEE Std 1159.3-2019 -- see ProntoAdapter.  pqdifpy is optional and only
# backs the unused PQDIFAdapter class below; nothing requires it.
try:
    import pqdifpy
    _PQDIF_AVAILABLE = True
except ImportError:
    _PQDIF_AVAILABLE = False

# rapidfuzz improves fuzzy channel-name matching; stdlib difflib is the fallback.
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ = True
except ImportError:
    import difflib as _difflib
    _RAPIDFUZZ = False

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHANNEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Every channel we care about is given a stable canonical name.
# The mapper resolves any device channel to one of these names.
CANONICAL = [
    "voltage_a", "voltage_b", "voltage_c",
    "current_a", "current_b", "current_c",
    "current_neutral",
    "power_real", "power_reactive", "power_factor",
    "thd_voltage_a", "thd_voltage_b", "thd_voltage_c",
    "thd_current_a", "thd_current_b", "thd_current_c",
    # Individual harmonic current magnitudes (Amps) — IEEE 519-2022 per-order checks
    *[f"h{h}_current_{ph}" for ph in ("a", "b", "c") for h in _H519_ORDERS],
    # Individual harmonic voltage magnitudes (Volts) — key odd orders
    *[f"h{h}_voltage_{ph}" for ph in ("a", "b", "c") for h in (3, 5, 7, 11, 13)],
    # Neutral current harmonics — triplens accumulate in neutral for zero-sequence diagnosis
    *[f"h{h}_current_neutral" for h in (3, 5, 7, 9, 11, 13)],
    # Line-to-line and neutral-to-earth voltages
    "voltage_ab", "voltage_bc", "voltage_ca", "voltage_neutral",
    # Apparent power and system frequency
    "power_apparent", "frequency",
    # Total harmonic RMS per phase, as computed by the meter at full precision
    # (more accurate than summing the rounded per-order magnitudes)
    "hrms_voltage_a", "hrms_voltage_b", "hrms_voltage_c",
    "hrms_current_a", "hrms_current_b", "hrms_current_c", "hrms_current_neutral",
    # Neutral distortion
    "thd_current_neutral", "thd_voltage_neutral",
    # Meter-reported unbalance (cross-check against our own NEMA computation)
    "unbalance_voltage_meter", "unbalance_current_meter", "unbalance_voltage_nps",
    # Meter-measured transformer K-factor and IEC flicker severity
    "kfactor_meter", "kfactor_current_b", "kfactor_current_c",
    "kfactor_current_neutral",
    "flicker_pst_b", "flicker_plt_b", "flicker_pst_c", "flicker_plt_c",
    "flicker_pst",
    "flicker_plt",
]

# ── PQDIF tag dictionaries ────────────────────────────────────────────────────
# PQDIF encodes channel identity with three fields (gemstone naming in parens):
#
#   quantity_type  (QuantityMeasured, uint32)
#       The physical quantity: Voltage, Current, Power … pqdifpy exposes this
#       on the ChannelDefinition as cd.quantity_type (attribute name varies by
#       library).  In the gemstone model this is QuantityMeasuredIDTag.
#
#   quantity_measured  (QuantityCharacteristic, GUID)
#       How the quantity is represented: RMS, THD, Peak, FlkrPST …  Lives on
#       the SeriesDefinition (sd.quantity_measured in pqdifpy).  gemstone calls
#       this QuantityCharacteristicIDTag.  Not to be confused with gemstone's
#       QuantityTypeIDTag (WaveForm / ValueLog / Phasor …) which describes the
#       data shape and is NOT used for channel identity matching here.
#
#   phase  (Phase, uint32 enum)
#       Which conductor: AN=1, BN=2, CN=3, NG=4, Net=9, Total=13 …  pqdifpy
#       returns the enum name lowercased, e.g. "an", "ng", "total".
#
# The sets below list every normalised string we accept for each field.
# See _normalise_tag() for how raw enums / GUIDs are converted to strings.

_TAG_MAP: Dict[str, Dict[str, Set[str]]] = {
    #  canonical        quantity_type (QuantityMeasured)    quantity_measured (QuantityCharacteristic)    phase
    "voltage_a":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"an", "a", "phase_a"}},
    "voltage_b":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"bn", "b", "phase_b"}},
    "voltage_c":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"cn", "c", "phase_c"}},
    "current_a":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"an", "a", "phase_a"}},
    "current_b":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"bn", "b", "phase_b"}},
    "current_c":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"cn", "c", "phase_c"}},
    # Phase.Total=13, Phase.Net=9, Phase.Residual=8; "three_phase"/"aggregate" are not PQDIF enum values
    "power_real":     {"qt": {"power", "watts"}, "qm": {"real", "watts", "active", "p"}, "ph": {"total", "net", "residual", ""}},
    "power_reactive": {"qt": {"power"},          "qm": {"reactive", "var", "q"},          "ph": {"total", "net", "residual", ""}},
    "power_factor":   {"qt": {"power", "powerfactor"}, "qm": {"powerfactor", "pf", "factor"}, "ph": {"total", "net", "residual", ""}},
    # THD: TotalTHD = % of fundamental; TotalTHDRMS = % of total RMS — both accepted
    "thd_voltage_a":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"an", "a", "phase_a"}},
    "thd_voltage_b":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"bn", "b", "phase_b"}},
    "thd_voltage_c":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"cn", "c", "phase_c"}},
    "thd_current_a":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"an", "a", "phase_a"}},
    "thd_current_b":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"bn", "b", "phase_b"}},
    "thd_current_c":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalthd", "totalthdrms", "totalharmdist", "thdpercent"}, "ph": {"cn", "c", "phase_c"}},
    # Phase.NG=4 (neutral-to-ground); "neutral" retained for ProntoAdapter internal strings
    "current_neutral":{"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"ng", "neutral", "n", "in", "i4", "phase_n"}},
    # Individual harmonic currents — one entry per order × phase.
    # Note: standard PQDIF files use QuantityCharacteristic.Spectra with SeriesNominalQuantity
    # for the harmonic order; the h{n}/harmonic{n} strings below match Pronto label-derived channels.
    **{f"h{h}_current_a": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"an","a","phase_a"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_b": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"bn","b","phase_b"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_c": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"cn","c","phase_c"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_neutral": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"ng","neutral","n","in","i4","phase_n"}}
       for h in (3, 5, 7, 9, 11, 13)},
    # Individual harmonic voltages
    **{f"h{h}_voltage_a": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"an","a","phase_a"}}
       for h in (3, 5, 7, 11, 13)},
    **{f"h{h}_voltage_b": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"bn","b","phase_b"}}
       for h in (3, 5, 7, 11, 13)},
    **{f"h{h}_voltage_c": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}", "spectra"}, "ph": {"cn","c","phase_c"}}
       for h in (3, 5, 7, 11, 13)},
    # Line-to-line voltages (ANSI C84.1 applies to these as well as L-N)
    "voltage_ab":     {"qt": {"voltage"},       "qm": {"rms"},                         "ph": {"ab"}},
    "voltage_bc":     {"qt": {"voltage"},       "qm": {"rms"},                         "ph": {"bc"}},
    "voltage_ca":     {"qt": {"voltage"},       "qm": {"rms"},                         "ph": {"ca", "ac"}},
    # Neutral-to-earth voltage — the primary open-neutral indicator
    "voltage_neutral":{"qt": {"voltage"},       "qm": {"rms"},                         "ph": {"ng", "neutral"}},
    # Apparent power as measured (includes distortion power, unlike sqrt(P²+Q²))
    "power_apparent": {"qt": {"power"},         "qm": {"apparent", "va", "s"},         "ph": {"total", "net", "residual", ""}},
    "frequency":      {"qt": {"frequency"},     "qm": {"frequency"},                   "ph": {"total", "net", ""}},
    # Total harmonic RMS (aggregate, not a single order)
    "hrms_voltage_a": {"qt": {"voltageharmonics"}, "qm": {"hrms"}, "ph": {"an", "a", "phase_a"}},
    "hrms_voltage_b": {"qt": {"voltageharmonics"}, "qm": {"hrms"}, "ph": {"bn", "b", "phase_b"}},
    "hrms_voltage_c": {"qt": {"voltageharmonics"}, "qm": {"hrms"}, "ph": {"cn", "c", "phase_c"}},
    "hrms_current_a": {"qt": {"currentharmonics"}, "qm": {"hrms"}, "ph": {"an", "a", "phase_a"}},
    "hrms_current_b": {"qt": {"currentharmonics"}, "qm": {"hrms"}, "ph": {"bn", "b", "phase_b"}},
    "hrms_current_c": {"qt": {"currentharmonics"}, "qm": {"hrms"}, "ph": {"cn", "c", "phase_c"}},
    "hrms_current_neutral": {"qt": {"currentharmonics"}, "qm": {"hrms"}, "ph": {"ng", "neutral", "n"}},
    # Neutral distortion
    "thd_current_neutral": {"qt": {"currentharmonics"}, "qm": {"thd", "totalthd"}, "ph": {"ng", "neutral", "n"}},
    "thd_voltage_neutral": {"qt": {"voltageharmonics"}, "qm": {"thd", "totalthd"}, "ph": {"ng", "neutral", "n"}},
    # Meter-reported unbalance.  AVG_IMBAL is the max-deviation-from-average
    # definition; S2S1 is the IEC negative/positive sequence ratio.
    "unbalance_voltage_meter": {"qt": {"voltage"}, "qm": {"avgimbal"},  "ph": {"total", "net", ""}},
    "unbalance_current_meter": {"qt": {"current"}, "qm": {"avgimbal"},  "ph": {"total", "net", ""}},
    "unbalance_voltage_nps":   {"qt": {"voltage"}, "qm": {"s2s1"},     "ph": {"total", "net", ""}},
    # Transformer K-factor
    "kfactor_meter": {"qt": {"kfactor"}, "qm": {"kfactor"},        "ph": {"an", "a", "total", "net", "residual", ""}},
    "kfactor_current_b": {"qt": {"kfactor"}, "qm": {"kfactor"},    "ph": {"bn", "b"}},
    "kfactor_current_c": {"qt": {"kfactor"}, "qm": {"kfactor"},    "ph": {"cn", "c"}},
    "kfactor_current_neutral": {"qt": {"kfactor"}, "qm": {"kfactor"}, "ph": {"ng", "neutral", "n"}},
    # Flicker: FlkrPST / FlkrPLT are QuantityCharacteristic GUIDs → normalised to "flkrpst"/"flkrplt"
    "flicker_pst":   {"qt": {"flicker", "voltage"}, "qm": {"pst", "flkrpst"}, "ph": {"an", "a", "phase_a", "total", "net", ""}},
    "flicker_plt":   {"qt": {"flicker", "voltage"}, "qm": {"plt", "flkrplt"}, "ph": {"an", "a", "phase_a", "total", "net", ""}},
    "flicker_pst_b": {"qt": {"flicker", "voltage"}, "qm": {"pst", "flkrpst"}, "ph": {"bn", "b", "phase_b"}},
    "flicker_plt_b": {"qt": {"flicker", "voltage"}, "qm": {"plt", "flkrplt"}, "ph": {"bn", "b", "phase_b"}},
    "flicker_pst_c": {"qt": {"flicker", "voltage"}, "qm": {"pst", "flkrpst"}, "ph": {"cn", "c", "phase_c"}},
    "flicker_plt_c": {"qt": {"flicker", "voltage"}, "qm": {"plt", "flkrplt"}, "ph": {"cn", "c", "phase_c"}},
}

# ── Fuzzy name patterns (fallback when tags are absent or non-standard) ───────
# Each list entry is a regex pattern matched against the channel label (lowercased).
_NAME_PATTERNS: Dict[str, List[str]] = {
    # THD entries come first so "THD Va" / "THD Ia" labels don't fall through to
    # the base voltage/current patterns, which also match the trailing "Va" / "Ia".
    "thd_voltage_a":  [r"thd[_\s]?v[_\s]?a", r"v[_\s]?thd[_\s]?a", r"voltage[_\s]?thd[_\s]?a"],
    "thd_voltage_b":  [r"thd[_\s]?v[_\s]?b", r"v[_\s]?thd[_\s]?b", r"voltage[_\s]?thd[_\s]?b"],
    "thd_voltage_c":  [r"thd[_\s]?v[_\s]?c", r"v[_\s]?thd[_\s]?c", r"voltage[_\s]?thd[_\s]?c"],
    "thd_current_a":  [r"thd[_\s]?i[_\s]?a", r"i[_\s]?thd[_\s]?a", r"current[_\s]?thd[_\s]?a"],
    "thd_current_b":  [r"thd[_\s]?i[_\s]?b", r"i[_\s]?thd[_\s]?b", r"current[_\s]?thd[_\s]?b"],
    "thd_current_c":  [r"thd[_\s]?i[_\s]?c", r"i[_\s]?thd[_\s]?c", r"current[_\s]?thd[_\s]?c"],
    # \bvan\b / \bvbn\b / \bvcn\b match Pronto-style "Van RMS" / "Vbn RMS" labels
    "voltage_a":      [r"\bvan\b", r"v[_\s]?a\b", r"va\b", r"vrms[_\s]?a", r"ph[ase]*[_\s]?a[_\s]?v", r"v1\b"],
    "voltage_b":      [r"\bvbn\b", r"v[_\s]?b\b", r"vb\b", r"vrms[_\s]?b", r"ph[ase]*[_\s]?b[_\s]?v", r"v2\b"],
    "voltage_c":      [r"\bvcn\b", r"v[_\s]?c\b", r"vc\b", r"vrms[_\s]?c", r"ph[ase]*[_\s]?c[_\s]?v", r"v3\b"],
    "current_a":      [r"i[_\s]?a\b", r"ia\b", r"irms[_\s]?a", r"ph[ase]*[_\s]?a[_\s]?i", r"i1\b", r"a[_\s]?rms"],
    "current_b":      [r"i[_\s]?b\b", r"ib\b", r"irms[_\s]?b", r"ph[ase]*[_\s]?b[_\s]?i", r"i2\b"],
    "current_c":      [r"i[_\s]?c\b", r"ic\b", r"irms[_\s]?c", r"ph[ase]*[_\s]?c[_\s]?i", r"i3\b"],
    "power_real":     [r"kw\b", r"real[_\s]?pow", r"active[_\s]?pow", r"p[_\s]?total", r"watts"],
    "power_reactive": [r"kvar\b", r"react[ive]*[_\s]?pow", r"q[_\s]?total"],
    "power_factor":   [r"\bpf\b", r"power[_\s]?fac", r"pf[_\s]?total"],
}


def _normalise_tag(value) -> str:
    """Convert a PQDIF tag (enum, GUID, or string) to a lowercase plain string.

    pqdifpy may expose tags as:
      - Python enums  → use .name or .value
      - UUID objects  → map via a known GUID table (extend as needed)
      - Strings       → just lowercase

    Extend the GUID table below with values you observe in your files.
    Print raw tag objects with --list-channels to see what your library returns.
    """
    if value is None:
        return ""
    # If it's an enum
    if hasattr(value, "name"):
        return str(value.name).lower().replace(" ", "_")
    # If it's a UUID
    s = str(value).lower().strip("{}")
    # Two separate GUID spaces can appear here depending on which tag pqdifpy
    # returns as a raw UUID rather than an enum:
    #
    #   QuantityTypeID (on ChannelDefinition) — describes the DATA SHAPE.
    #   QuantityCharacteristicID (on SeriesDefinition) — describes HOW MEASURED.
    #
    # Source: gemstone/pqdif QuantityType.cs and QuantityCharacteristic.cs
    _GUID_NAMES = {
        # ── QuantityType (data shape) ──────────────────────────────────────
        "67f6af80-f753-11cf-9d89-0080c72e70a3": "waveform",     # Point-on-wave
        "67f6af82-f753-11cf-9d89-0080c72e70a3": "valuelog",     # Time-logged averages (most interval data)
        "67f6af81-f753-11cf-9d89-0080c72e70a3": "phasor",       # Time-domain phasor
        "67f6af85-f753-11cf-9d89-0080c72e70a3": "response",     # Frequency-domain
        "67f6af83-f753-11cf-9d89-0080c72e70a3": "flash",
        "67f6af87-f753-11cf-9d89-0080c72e70a3": "histogram",
        "67f6af88-f753-11cf-9d89-0080c72e70a3": "histogram3d",
        "67f6af89-f753-11cf-9d89-0080c72e70a3": "cpf",
        "67f6af8a-f753-11cf-9d89-0080c72e70a3": "xy",
        "67f6af8b-f753-11cf-9d89-0080c72e70a3": "magdur",       # Magnitude+duration (sag/swell event records)
        "67f6af8c-f753-11cf-9d89-0080c72e70a3": "xyz",
        "67f6af8d-f753-11cf-9d89-0080c72e70a3": "magdurtime",
        "67f6af8e-f753-11cf-9d89-0080c72e70a3": "magdurcount",
        # ── QuantityCharacteristic (how measured) ──────────────────────────
        "a6b31ae5-b451-11d1-ae17-0060083a2628": "rms",
        "a6b31ae2-b451-11d1-ae17-0060083a2628": "peak",
        "a6b31adc-b451-11d1-ae17-0060083a2628": "hrms",
        "a6b31add-b451-11d1-ae17-0060083a2628": "instantaneous",
        "a6b31ae9-b451-11d1-ae17-0060083a2628": "spectra",
        "07ef68af-9ff5-11d2-b30b-006008b37183": "frequency",
        "a6b31aec-b451-11d1-ae17-0060083a2628": "totalthd",     # THD normalised to fundamental
        "f3d216e0-2aa5-11d5-a4b3-444553540000": "totalthdrms",  # THD normalised to RMS
        "a6b31ad4-b451-11d1-ae17-0060083a2628": "eventhd",
        "a6b31ae0-b451-11d1-ae17-0060083a2628": "oddthd",
        "a6b31ae7-b451-11d1-ae17-0060083a2628": "s0s1",         # Zero-sequence unbalance
        "a6b31ae8-b451-11d1-ae17-0060083a2628": "s2s1",         # Negative-sequence unbalance
        "515bf320-71ca-11d4-a4b3-444553540000": "flkrpst",      # IEC flicker Pst
        "515bf321-71ca-11d4-a4b3-444553540000": "flkrplt",      # IEC flicker Plt
        "8786ca11-9113-11d3-b930-0050da2b1f4d": "kfactor",      # Transformer K-factor
        "f3d216e7-2aa5-11d5-a4b3-444553540000": "tdd",          # Total demand distortion
        "07ef68a0-9ff5-11d2-b30b-006008b37183": "rmsdemand",
    }
    if s in _GUID_NAMES:
        return _GUID_NAMES[s]
    return re.sub(r"[^a-z0-9]", "", s)


@dataclass
class RawChannelInfo:
    """Metadata for one channel as returned by the PQDIF adapter."""
    index: int                 # position in the DataSource channel list
    label: str                 # device-assigned text name
    quantity_type: str         # normalised: 'voltage', 'current', 'power' …
    quantity_measured: str     # normalised: 'rms', 'thd', 'average' …
    phase: str                 # normalised: 'a', 'b', 'c', 'total' …
    unit: str                  # 'V', 'A', 'kW' etc. (informational)

    def debug_str(self) -> str:
        return (
            f"  [{self.index:3d}] label={self.label!r:30s}  "
            f"qt={self.quantity_type:20s}  qm={self.quantity_measured:15s}  "
            f"ph={self.phase:10s}  unit={self.unit}"
        )


class ChannelMapper:
    """Map raw device channels to canonical engineering names.

    Resolution order:
      1. PQDIF tag match (quantity_type + quantity_measured + phase)
      2. Regex pattern match on label
      3. Fuzzy string match on label (requires rapidfuzz or difflib)
    """

    def __init__(self, fuzzy_threshold: float = 0.70):
        self.fuzzy_threshold = fuzzy_threshold

    def resolve(self, channels: List[RawChannelInfo]) -> Dict[str, RawChannelInfo]:
        """Return {canonical_name: RawChannelInfo} for every channel matched."""
        result: Dict[str, RawChannelInfo] = {}
        unmatched: List[RawChannelInfo] = []

        for ch in channels:
            name = self._match_by_tags(ch)
            if name is None:
                name = self._match_by_regex(ch.label)
            if name is None:
                unmatched.append(ch)
                continue
            if name not in result:
                result[name] = ch
                log.debug("  %s → %s (tag/regex)", ch.label, name)

        # Fuzzy pass for anything still unmatched
        for ch in unmatched:
            name = self._match_fuzzy(ch.label, set(result.keys()))
            if name:
                result[name] = ch
                log.debug("  %s → %s (fuzzy)", ch.label, name)
            else:
                log.debug("  %s → (no match)", ch.label)

        return result

    def _match_by_tags(self, ch: RawChannelInfo) -> Optional[str]:
        qt = ch.quantity_type.lower().replace(" ", "")
        qm = ch.quantity_measured.lower().replace(" ", "")
        ph = ch.phase.lower().replace(" ", "")
        for canonical, tags in _TAG_MAP.items():
            if qt in tags["qt"] and qm in tags["qm"] and ph in tags["ph"]:
                return canonical
        return None

    def _match_by_regex(self, label: str) -> Optional[str]:
        lbl = label.lower()
        for canonical, patterns in _NAME_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, lbl):
                    return canonical
        return None

    def _match_fuzzy(self, label: str, already_found: Set[str]) -> Optional[str]:
        remaining = [c for c in CANONICAL if c not in already_found]
        if not remaining:
            return None
        lbl = label.lower()
        best_name, best_score = None, 0.0
        for canonical in remaining:
            # Compare against the canonical name and each of its regex terms
            candidates = [canonical.replace("_", " ")]
            candidates += [p.replace(r"\b", "").replace("[_\\s]?", "")
                           for p in _NAME_PATTERNS.get(canonical, [])]
            for candidate in candidates:
                if _RAPIDFUZZ:
                    score = _rfuzz.partial_ratio(lbl, candidate) / 100.0
                else:
                    score = _difflib.SequenceMatcher(None, lbl, candidate).ratio()
                if score > best_score:
                    best_score, best_name = score, canonical
        if best_score >= self.fuzzy_threshold:
            return best_name
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PQDIF FILE ADAPTER
# ─────────────────────────────────────────────────────────────────────────────
#
# The adapter is the only layer that touches pqdifpy directly.
# If pqdifpy's API differs from what is shown here, fix ONLY this class.
#
# Expected behaviour of the adapter:
#   .list_channels()  → List[RawChannelInfo]  (one entry per channel)
#   .iter_observations(wanted_indices)        → yields (timestamps, {idx: ndarray})
#
# --list-channels calls list_channels() and prints each RawChannelInfo.debug_str().
# This is the fastest way to see exactly what your library exposes.
# ─────────────────────────────────────────────────────────────────────────────

class PQDIFAdapter:
    """Thin wrapper around pqdifpy that exposes only what pq_analyzer needs.

    pqdifpy API assumptions (version ≥0.x):
        import pqdifpy
        reader = pqdifpy.PQDIFReader(filepath)   # or pqdifpy.Reader(...)
        for record in reader:
            record.record_type                   # pqdifpy.RecordType enum
            # DataSource record:
            record.channel_definitions           # iterable
                cd.label                         # str
                cd.quantity_type                 # enum / GUID / str
                cd.phase                         # enum / str
                cd.series_definitions            # list
                    sd.quantity_measured         # enum / str
                    sd.units                     # str (optional)
            # Observation record:
            record.start_time                    # datetime
            record.channel_instances             # iterable
                ci.channel_definition_index      # int
                ci.series                        # list of series objects
                    s.values                     # numpy array
                    s.sample_count               # int (for equally-spaced data)
                    s.time_increment             # float seconds (for EQ data)
                    s.time_stamps                # array of datetime64 (for non-EQ)

    If your pqdifpy version uses different attribute names, search for the
    comment "ADAPT:" below and adjust those lines.
    """

    def __init__(self, filepath: str | Path):
        if not _PQDIF_AVAILABLE:
            raise ImportError(
                "pqdifpy is not installed.\n"
                "  pip install pqdifpy\n"
                "Or run with --demo for synthetic data."
            )
        self.filepath = Path(filepath)
        self._channel_defs: List[RawChannelInfo] = []
        self._reader = None
        self._scan_channel_defs()

    def _scan_channel_defs(self):
        """First pass: read DataSource records only to build the channel index."""
        log.info("Scanning channel definitions …")
        # ADAPT: adjust class/attribute names to match your pqdifpy version.
        reader = pqdifpy.PQDIFReader(str(self.filepath))  # ADAPT if needed
        idx = 0
        for record in reader:
            rt = str(getattr(record, "record_type", "")).lower()
            if "datasource" not in rt and "data_source" not in rt:
                continue
            channel_defs = getattr(record, "channel_definitions",
                                   getattr(record, "channels", []))
            for cd in channel_defs:
                label = str(getattr(cd, "label", getattr(cd, "name", f"ch_{idx}")))
                qt    = _normalise_tag(getattr(cd, "quantity_type", None))
                phase = _normalise_tag(getattr(cd, "phase", None))
                # quantity_measured lives on the series definition in PQDIF
                sd_list = getattr(cd, "series_definitions",
                                  getattr(cd, "series", []))
                qm   = _normalise_tag(
                    getattr(sd_list[0], "quantity_measured", None)
                    if sd_list else None
                )
                unit = str(getattr(
                    sd_list[0] if sd_list else cd, "units",
                    getattr(cd, "unit", "")
                ))
                self._channel_defs.append(
                    RawChannelInfo(idx, label, qt, qm, phase, unit)
                )
                idx += 1
        log.info("Found %d channel definitions.", len(self._channel_defs))

    def list_channels(self) -> List[RawChannelInfo]:
        return self._channel_defs

    def iter_observations(
        self, wanted_indices: Set[int]
    ):
        """Yield (timestamps_array, {channel_idx: values_array}) per observation.

        timestamps_array : np.ndarray[datetime64[ns]]
        values           : np.ndarray[float64], same length as timestamps
        """
        reader = pqdifpy.PQDIFReader(str(self.filepath))  # ADAPT if needed
        obs_count = 0
        for record in reader:
            rt = str(getattr(record, "record_type", "")).lower()
            if "observation" not in rt:
                continue
            obs_count += 1

            start_time: datetime = getattr(record, "start_time",
                                           getattr(record, "trigger_time",
                                                   datetime.now()))
            channel_instances = getattr(record, "channel_instances",
                                        getattr(record, "channels", []))
            timestamps = None
            data: Dict[int, np.ndarray] = {}

            for ci in channel_instances:
                # ADAPT: attribute name may be channel_definition_index or channel_index
                cidx = int(getattr(ci, "channel_definition_index",
                                   getattr(ci, "channel_index", -1)))
                if cidx not in wanted_indices:
                    continue

                series_list = getattr(ci, "series",
                                      getattr(ci, "series_instances", []))
                if not series_list:
                    continue
                s = series_list[0]  # take the first series (usually RMS or the primary value)

                values = np.asarray(getattr(s, "values",
                                            getattr(s, "data", [])), dtype=float)
                if len(values) == 0:
                    continue

                # Reconstruct timestamps.  PQDIF supports two timestamp schemes:
                #   Equally spaced: start_time + n * time_increment
                #   Explicit:       each sample has a timestamp
                if hasattr(s, "time_stamps") and s.time_stamps is not None:
                    ts = np.asarray(s.time_stamps, dtype="datetime64[ns]")
                else:
                    n = len(values)
                    increment_sec = float(getattr(s, "time_increment",
                                                  getattr(s, "sample_interval", 1.0)))
                    base = np.datetime64(
                        start_time.replace(tzinfo=None), "ns"
                    )
                    ts = base + np.arange(n) * np.timedelta64(
                        int(increment_sec * 1e9), "ns"
                    )

                data[cidx] = values
                if timestamps is None or len(ts) > len(timestamps):
                    timestamps = ts

            if timestamps is not None and data:
                yield timestamps, data

        log.info("Read %d observation records.", obs_count)


class ProntoAdapter:
    """
    Reader for Pronto PQDIF files (Xcel Energy metering system).

    Pronto's .pqd exports are fully compliant with IEEE Std 1159.3-2019, so the
    primary path (``_load_spec``) resolves everything structurally through
    pqdif.py: channel identity comes from the file's own series definitions
    (quantity measured, quantity characteristic, phase), timestamps come from
    each observation's tagTimeStart, and value arrays come from tagSeriesValues.
    No byte offsets, channel orderings or label spellings are assumed, which is
    what makes it robust across firmware versions and service topologies.

    Two Pronto conventions are *not* in the standard and so are detected rather
    than assumed:

      - Each interval is written as a step pair -- the same value at the
        interval's start and end time -- which _step_pair_stride() detects from
        the time series and deduplicates.
      - Interval data is split across two observation records that share one
        time base: 'Interval (avg)' carries derived quantities (THD, harmonics,
        power, flicker) and 'Interval (max-min)' carries the true RMS voltages
        and currents.  Both are pooled.

    A legacy reverse-engineered reader (``_load_legacy`` and the ``_load_v2*``
    methods) is retained as a fallback for anything that fails the PQDIF
    signature check.  Nothing in the repository exercises it any more -- the
    test_data/ fixtures are generated from the standard by make_test_pqd.py --
    and no file a Pronto meter produces should reach it.  It is kept only in
    case an older firmware writes something that is not valid PQDIF, since we
    have no sample of such a file to verify against.

    Channels exposed (match CANONICAL names via _TAG_MAP):
        voltage_a/b/c (V), current_a/b/c (A),
        power_real (W), power_reactive (VAR), power_factor,
        thd_current_a (%)

    Note: obs[0] contains 460+ individual harmonic channels (H2–H50 for each
    phase of voltage and current) which are not exposed here but are accessible
    by extending _O0_CHAN_* constants and adding entries to ch_defs in _load().
    """

    _TAG_OBSERVATION = 0x8973861A

    # ── obs[0]: harmonic / power-quality channels ─────────────────────────────
    _O0_FIRST  = 13328   # first channel block offset in decompressed body
    _O0_STRIDE = 5052    # bytes per channel
    _O0_VALREL = 2632    # value series data offset relative to channel block start

    # obs[0] channel positions (0-indexed).  Identified by value-signature
    # analysis; DS channel name labels at these positions are known-incorrect
    # firmware labels in the Pronto exporter (positions 52–58).
    _O0_APPAR_PWR  = 3   # apparent power (VA)
    _O0_REACT_PWR  = 4   # reactive power (VAR)
    _O0_PF         = 6   # power factor (dimensionless 0–1)
    _O0_FREQ       = 7   # frequency (Hz)  — not in CANONICAL, skip for now
    _O0_THD_IA     = 8   # current THD phase A (%)

    # ── obs[1]: five-minute RMS interval channels ─────────────────────────────
    _O1_FIRST  = 536     # first channel block offset in decompressed body
    _O1_STRIDE = 10012   # bytes per channel (4 series: ts, max, min, avg)
    _O1_TSREL  = 236     # timestamp series data offset relative to channel block
    _O1_AVGREL = 7592    # average value series offset relative to channel block

    # obs[1] channel positions (DS indices 39–49 in order)
    _O1_VAN = 0   # RMS Van – V (phase A line-to-neutral)
    _O1_VBN = 1   # RMS Vbn – V (phase B line-to-neutral)
    _O1_VCN = 2   # RMS Vcn – V (phase C line-to-neutral)
    _O1_IA  = 7   # RMS Ia – A  (phase A current)
    _O1_IB  = 8   # RMS Ib – A  (phase B current)
    _O1_IC  = 9   # RMS Ic – A  (phase C current)
    _O1_IN  = 10  # RMS In – A  (neutral current / I4)

    # ── obs[0]: individual harmonic blocks ───────────────────────────────────
    # Each block: ch[BASE] = fundamental (H1), ch[BASE+h-1] = H_h magnitude.
    # Layout confirmed by value-signature analysis on real Pronto PQDIF files.
    _O0_VA_BLOCK  = 58   # Va fundamental (V); H_h at BASE+h-1
    _O0_IA_BLOCK  = 109  # Ia fundamental (A); H_h at BASE+h-1
    _O0_VB_BLOCK  = 160  # Vb fundamental (V)
    _O0_IB_BLOCK  = 211  # Ib fundamental (A)
    _O0_VC_BLOCK  = 262  # Vc fundamental (V)
    _O0_IC_BLOCK  = 313  # Ic fundamental (A)

    # ── new-format (v2): 30+ obs records, pointer-chain channel structure ─────
    # Channels are discovered by reading DataSource labels and mapping them through
    # the ChannelInstances table in the Interval (avg) obs body.
    # entry+20: u32 absolute offset of channel block start in obs body
    # Each channel block:
    #   +_V2_TS_REL:  u32 count + count×f64 timestamps (dedup every other)
    #   +data_rel:    u32 count + count×f64 measurements (dedup every other)
    #   data_rel = _V2_TS_REL + 4 + ts_count_raw×8 + 32  (computed dynamically)
    _V2_ENTRY_SIZE   = 28    # bytes per channel entry in the ChannelInstances table
    _V2_BODY_OFF_REL = 20    # offset within each entry of the abs channel-block pointer
    _V2_TS_REL       = 180   # offset from channel block start to timestamp count+data

    # ── PQDIF element tag GUIDs (first 4 bytes, little-endian) ───────────────
    # Used by _pqdif_elements / _build_label_map for DataSource label discovery.
    _TAG_DATASOURCE  = 0x89738619   # DataSource record type
    _ELEM_CHAN_DEFS  = 0xB48D858D   # ChannelDefinitions collection in DataSource
    _ELEM_CHAN_LABEL = 0xB48D8590   # channel label string in each ChannelDefinition
    _ELEM_CHAN_INSTS = 0x3D786F91   # ChannelInstances collection in obs body
    _ELEM_DS_IDX     = 0xB48D858F   # DS channel index (inline u32) in ChannelInstance

    # v2 "Variable Adaptive" obs record — 29-channel layout confirmed from Pronto viewer.
    # Entry table order follows VIEWER GROUP ORDER (not C-number order):
    #   Voltage AC (C=1-7) → Current AC (C=8-11) → Unbalance (C=25-28) →
    #   Power (C=20-24) → Frequency (C=29) → Harmonic Group THD (C=12-19)
    # Timestamps and data are SINGLE float64s (no quality-pair interleaving, unlike interval).
    _ADAP_TS_REL = 236   # ts_count u32 at ch_abs+236; ts array at ch_abs+240
    # Entry-table start AND channel order both vary across export versions and
    # service topologies (split-phase vs three-phase), so both are discovered
    # per file at load time: _scan_entry_table locates the table by pattern,
    # and _identify_adaptive_channels assigns names by signature.

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._raw_channels: List[RawChannelInfo] = []
        self._obs_ts: Optional[np.ndarray] = None
        self._obs_data: Dict[int, np.ndarray] = {}
        self._adaptive_df: Optional[pd.DataFrame] = None
        self._waveforms: List[dict] = []
        self._load()

    # ── Public interface (same contract as PQDIFAdapter) ──────────────────────

    def list_channels(self) -> List[RawChannelInfo]:
        return self._raw_channels

    def iter_observations(self, wanted_indices: Set[int]):
        """Yield (timestamps_array, {channel_idx: values_array})."""
        data = {i: v for i, v in self._obs_data.items() if i in wanted_indices}
        if data:
            yield self._obs_ts, data

    @property
    def adaptive_df(self) -> Optional[pd.DataFrame]:
        """High-resolution variable-rate DataFrame from the adaptive obs record, or None."""
        return self._adaptive_df

    @property
    def waveforms(self) -> List[dict]:
        """Decoded point-on-wave capture records (empty when absent)."""
        return self._waveforms

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self):
        """Read the file, preferring the spec-compliant path.

        Pronto's exports are fully IEEE 1159.3-2019 compliant, so pqdif.py
        handles them without any offset guessing.  Falling back to the legacy
        offset reader means the file failed the PQDIF signature check, which no
        Pronto export should; the message says which check failed.
        """
        try:
            self._spec = pqdif.PQDIFFile(self.filepath)
        except pqdif.PQDIFError as exc:
            log.info(
                "ProntoAdapter: %s is not standard PQDIF (%s); using the "
                "legacy offset reader.", self.filepath.name, exc,
            )
            self._spec = None

        if self._spec is not None:
            self._load_spec()
            self.data_quality = {
                "missing_bytes": self._spec.missing_bytes,
                "unreadable_observations": len(self._spec.unreadable_observations),
            }
            return
        self._load_legacy()

    def _load_legacy(self):
        raw = self.filepath.read_bytes()
        recs = self._walk_records(raw)

        obs_recs = [r for r in recs if r['tag'] == self._TAG_OBSERVATION]
        if len(obs_recs) >= 4:
            # Extended Pronto format: waveform captures + Interval (avg) obs record.
            # Covers both the original v2 format (30+ obs) and the newer Pronto-to-PQDIF
            # export format (~26 obs records).  The old proprietary format has exactly 3
            # obs records, so >= 4 safely routes all export-format files here.
            self._load_v2(obs_recs, recs)
            return

        if len(obs_recs) < 2:
            raise ValueError(
                f"ProntoAdapter: expected ≥2 Observation records, found {len(obs_recs)}. "
                "Is this a Pronto PQDIF file?"
            )

        try:
            obs0_body = zlib.decompress(obs_recs[0]['raw'])
            obs1_body = zlib.decompress(obs_recs[1]['raw'])
        except zlib.error as exc:
            raise ValueError(f"ProntoAdapter: zlib decompression failed — {exc}") from exc

        base_date = self._parse_date()
        n, self._obs_ts = self._build_timestamps(obs1_body, base_date)

        def read_o1(ch_idx: int) -> np.ndarray:
            off = self._O1_FIRST + ch_idx * self._O1_STRIDE + self._O1_AVGREL
            return self._load_dedup(obs1_body, off, n)

        def read_o0(ch_idx: int) -> np.ndarray:
            off = self._O0_FIRST + ch_idx * self._O0_STRIDE + self._O0_VALREL
            return self._load_dedup(obs0_body, off, n)

        volt_a = read_o1(self._O1_VAN)
        volt_b = read_o1(self._O1_VBN)
        volt_c = read_o1(self._O1_VCN)
        curr_a = read_o1(self._O1_IA)
        curr_b = read_o1(self._O1_IB)
        curr_c = read_o1(self._O1_IC)
        curr_n = read_o1(self._O1_IN)

        appar  = read_o0(self._O0_APPAR_PWR)
        react  = read_o0(self._O0_REACT_PWR)
        pf     = read_o0(self._O0_PF)
        thd_ia = read_o0(self._O0_THD_IA)
        real   = appar * pf

        ch_defs = [
            # (index, label, quantity_type, quantity_measured, phase, unit)
            (0,  'Van RMS',       'voltage',          'rms',         'an',      'V'  ),
            (1,  'Vbn RMS',       'voltage',          'rms',         'bn',      'V'  ),
            (2,  'Vcn RMS',       'voltage',          'rms',         'cn',      'V'  ),
            (3,  'Ia RMS',        'current',          'rms',         'an',      'A'  ),
            (4,  'Ib RMS',        'current',          'rms',         'bn',      'A'  ),
            (5,  'Ic RMS',        'current',          'rms',         'cn',      'A'  ),
            (6,  'Real Power',    'watts',            'watts',       'total',   'W'  ),
            (7,  'Reactive Power','power',            'reactive',    'total',   'VAR'),
            (8,  'Power Factor',  'powerfactor',      'powerfactor', 'total',   ''   ),
            (9,  'THD Ia',        'currentharmonics', 'thd',         'an',      '%'  ),
            (10, 'In RMS',        'current',          'rms',         'neutral', 'A'  ),
        ]
        arrays = [volt_a, volt_b, volt_c, curr_a, curr_b, curr_c,
                  real, react, pf, thd_ia, curr_n]

        # ── Individual harmonic magnitudes from obs[0] ─────────────────────
        # Blocks: ch[BASE] = H1 (fundamental); ch[BASE+h-1] = H_h magnitude.
        ph_map = [('an', self._O0_VA_BLOCK, self._O0_IA_BLOCK),
                  ('bn', self._O0_VB_BLOCK, self._O0_IB_BLOCK),
                  ('cn', self._O0_VC_BLOCK, self._O0_IC_BLOCK)]
        idx = 11
        for ph_code, v_base, i_base in ph_map:
            for h in _H519_ORDERS:
                ch_defs.append((idx, f'H{h} I_{ph_code}', 'currentharmonics', f'h{h}', ph_code, 'A'))
                arrays.append(read_o0(i_base + h - 1))
                idx += 1
            for h in (3, 5, 7, 11, 13):
                ch_defs.append((idx, f'H{h} V_{ph_code}', 'voltageharmonics', f'h{h}', ph_code, 'V'))
                arrays.append(read_o0(v_base + h - 1))
                idx += 1

        self._raw_channels = [
            RawChannelInfo(idx, label, qt, qm, phase, unit)
            for (idx, label, qt, qm, phase, unit) in ch_defs
        ]
        self._obs_data = {cd[0]: arr for cd, arr in zip(ch_defs, arrays)}

        log.info(
            "ProntoAdapter: loaded %d channels, %d 5-min intervals (%s → %s)",
            len(self._raw_channels), n,
            pd.Timestamp(self._obs_ts[0]).strftime('%Y-%m-%d %H:%M') if n else '–',
            pd.Timestamp(self._obs_ts[-1]).strftime('%Y-%m-%d %H:%M') if n else '–',
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Spec-compliant path (IEEE Std 1159.3-2019, via pqdif.py)
    # ─────────────────────────────────────────────────────────────────────────

    # (quantity_measured, characteristic, phase) → (label, canonical, tag qt,
    # tag qm, tag phase).  The key is read straight out of the file's own series
    # definitions, so this table does not depend on channel names, service
    # topology or firmware version.  The tag fields feed _TAG_MAP via
    # ChannelMapper, which derives the same canonical name recorded here.
    _SPEC_CHANNELS: Dict[Tuple[str, str, str], Tuple[str, str, str, str, str]] = {
        # ── True RMS, not the fundamental ────────────────────────────────
        # Characteristic RMS is the quantity ANSI C84.1 limits apply to.
        # 'Harm 1 of Van' is characteristic SPECTRA_HGROUP -- the H1 magnitude,
        # which reads low by a factor of sqrt(1 + THD²) (0.3 % at 7.5 % THD)
        # and must never be used as the voltage trend.
        ('voltage', 'RMS', 'an'): ('Van RMS', 'voltage_a', 'voltage', 'rms', 'an'),
        ('voltage', 'RMS', 'bn'): ('Vbn RMS', 'voltage_b', 'voltage', 'rms', 'bn'),
        ('voltage', 'RMS', 'cn'): ('Vcn RMS', 'voltage_c', 'voltage', 'rms', 'cn'),
        ('current', 'RMS', 'an'): ('Ia RMS', 'current_a', 'current', 'rms', 'an'),
        ('current', 'RMS', 'bn'): ('Ib RMS', 'current_b', 'current', 'rms', 'bn'),
        ('current', 'RMS', 'cn'): ('Ic RMS', 'current_c', 'current', 'rms', 'cn'),
        ('current', 'RMS', 'ng'): ('In RMS', 'current_neutral',
                                   'current', 'rms', 'neutral'),
        # ── Power ─────────────────────────────────────────────────────────
        ('power', 'P',  'none'): ('Real Power', 'power_real',
                                  'watts', 'watts', 'total'),
        ('power', 'Q',  'none'): ('Reactive Power', 'power_reactive',
                                  'power', 'reactive', 'total'),
        ('power', 'PF', 'none'): ('Power Factor', 'power_factor',
                                  'powerfactor', 'powerfactor', 'total'),
        # ── Distortion, as measured by the instrument ─────────────────────
        ('voltage', 'TOTAL_THD', 'an'): ('THD Van', 'thd_voltage_a',
                                         'voltageharmonics', 'thd', 'an'),
        ('voltage', 'TOTAL_THD', 'bn'): ('THD Vbn', 'thd_voltage_b',
                                         'voltageharmonics', 'thd', 'bn'),
        ('voltage', 'TOTAL_THD', 'cn'): ('THD Vcn', 'thd_voltage_c',
                                         'voltageharmonics', 'thd', 'cn'),
        ('current', 'TOTAL_THD', 'an'): ('THD Ia', 'thd_current_a',
                                         'currentharmonics', 'thd', 'an'),
        ('current', 'TOTAL_THD', 'bn'): ('THD Ib', 'thd_current_b',
                                         'currentharmonics', 'thd', 'bn'),
        ('current', 'TOTAL_THD', 'cn'): ('THD Ic', 'thd_current_c',
                                         'currentharmonics', 'thd', 'cn'),
        # ── Meter-computed K-factor and IEC flicker, per phase ────────────
        ('current', 'K_FACTOR', 'an'): ('K-Factor', 'kfactor_meter',
                                        'kfactor', 'kfactor', 'an'),
        ('current', 'K_FACTOR', 'bn'): ('K-Factor Ib', 'kfactor_current_b',
                                        'kfactor', 'kfactor', 'bn'),
        ('current', 'K_FACTOR', 'cn'): ('K-Factor Ic', 'kfactor_current_c',
                                        'kfactor', 'kfactor', 'cn'),
        ('current', 'K_FACTOR', 'ng'): ('K-Factor In', 'kfactor_current_neutral',
                                        'kfactor', 'kfactor', 'neutral'),
        ('voltage', 'FLKR_PST', 'an'): ('Flicker PST', 'flicker_pst',
                                        'flicker', 'pst', 'an'),
        ('voltage', 'FLKR_PLT', 'an'): ('Flicker PLT', 'flicker_plt',
                                        'flicker', 'plt', 'an'),
        ('voltage', 'FLKR_PST', 'bn'): ('Flicker PST Vbn', 'flicker_pst_b',
                                        'flicker', 'pst', 'bn'),
        ('voltage', 'FLKR_PLT', 'bn'): ('Flicker PLT Vbn', 'flicker_plt_b',
                                        'flicker', 'plt', 'bn'),
        ('voltage', 'FLKR_PST', 'cn'): ('Flicker PST Vcn', 'flicker_pst_c',
                                        'flicker', 'pst', 'cn'),
        ('voltage', 'FLKR_PLT', 'cn'): ('Flicker PLT Vcn', 'flicker_plt_c',
                                        'flicker', 'plt', 'cn'),
        # ── Line-to-line and neutral-to-earth voltage ─────────────────────
        ('voltage', 'RMS', 'ab'): ('Vab RMS', 'voltage_ab', 'voltage', 'rms', 'ab'),
        ('voltage', 'RMS', 'bc'): ('Vbc RMS', 'voltage_bc', 'voltage', 'rms', 'bc'),
        ('voltage', 'RMS', 'ca'): ('Vca RMS', 'voltage_ca', 'voltage', 'rms', 'ca'),
        ('voltage', 'RMS', 'ng'): ('Vne RMS', 'voltage_neutral',
                                   'voltage', 'rms', 'neutral'),
        # ── Apparent power and frequency ──────────────────────────────────
        # The meter's S includes distortion power, so it is not sqrt(P² + Q²).
        ('power', 'S', 'none'): ('Apparent Power', 'power_apparent',
                                 'power', 'apparent', 'total'),
        ('voltage', 'FREQUENCY', 'none'): ('Frequency', 'frequency',
                                           'frequency', 'frequency', 'total'),
        # ── Aggregate harmonic RMS ────────────────────────────────────────
        # Computed inside the meter at full precision; summing the reported
        # per-order magnitudes understates it, badly at light load where the
        # individual orders round to the display resolution.
        ('voltage', 'HRMS', 'an'): ('Hrms Van', 'hrms_voltage_a',
                                    'voltageharmonics', 'hrms', 'an'),
        ('voltage', 'HRMS', 'bn'): ('Hrms Vbn', 'hrms_voltage_b',
                                    'voltageharmonics', 'hrms', 'bn'),
        ('voltage', 'HRMS', 'cn'): ('Hrms Vcn', 'hrms_voltage_c',
                                    'voltageharmonics', 'hrms', 'cn'),
        ('current', 'HRMS', 'an'): ('Hrms Ia', 'hrms_current_a',
                                    'currentharmonics', 'hrms', 'an'),
        ('current', 'HRMS', 'bn'): ('Hrms Ib', 'hrms_current_b',
                                    'currentharmonics', 'hrms', 'bn'),
        ('current', 'HRMS', 'cn'): ('Hrms Ic', 'hrms_current_c',
                                    'currentharmonics', 'hrms', 'cn'),
        ('current', 'HRMS', 'ng'): ('Hrms In', 'hrms_current_neutral',
                                    'currentharmonics', 'hrms', 'neutral'),
        # ── Neutral distortion ────────────────────────────────────────────
        ('current', 'TOTAL_THD', 'ng'): ('THD In', 'thd_current_neutral',
                                         'currentharmonics', 'thd', 'neutral'),
        ('voltage', 'TOTAL_THD', 'ng'): ('THD Vne', 'thd_voltage_neutral',
                                         'voltageharmonics', 'thd', 'neutral'),
        # ── Unbalance as the meter computes it ────────────────────────────
        ('voltage', 'AVG_IMBAL', 'none'): ('V Unbalance',
                                           'unbalance_voltage_meter',
                                           'voltage', 'avgimbal', 'total'),
        ('current', 'AVG_IMBAL', 'none'): ('I Unbalance',
                                           'unbalance_current_meter',
                                           'current', 'avgimbal', 'total'),
        ('voltage', 'S2S1', 'none'): ('V Unbalance NPS/PPS',
                                      'unbalance_voltage_nps',
                                      'voltage', 's2s1', 'total'),
    }

    #: Characteristics that are instrument housekeeping rather than power
    #: quality, so their absence from _SPEC_CHANNELS is deliberate and does not
    #: warrant a log line.
    _SPEC_IGNORED = {'STATUS'}

    #: Characteristics Pronto overloads across several distinct channels, where
    #: the metadata alone is not enough to tell them apart.  HRMS is used for
    #: the total harmonic RMS *and* for the odd, even and triplen subtotals
    #: ('Hrms Van (V1)', 'Odds Van (V1)', 'Evens Van (V1)', 'Triplens Van' all
    #: carry ID_QC_HRMS, where the standard offers HRMS_ODD / HRMS_EVEN /
    #: HRMS_TRIPLEN).  Requiring a name prefix keeps the subtotals from being
    #: silently picked up as the total, which file ordering alone decided.
    _SPEC_NAME_PREFIX = {'HRMS': 'hrms'}

    #: Harmonic channel names, e.g. 'Harm 13 of Van'.  The order lives only in
    #: the name -- Pronto gives each order its own channel definition rather
    #: than using one definition with per-instance frequency, so there is no
    #: metadata field to read it from.
    _HARM_NAME = re.compile(r'^Harm (\d+) of (V[a-z]{2}|I[a-z])$')

    #: Which orders are reported per phase group, matching CANONICAL.
    _HARM_REPORT = {
        'Van': (3, 5, 7, 11, 13), 'Vbn': (3, 5, 7, 11, 13),
        'Vcn': (3, 5, 7, 11, 13),
        'Ia': _H519_ORDERS, 'Ib': _H519_ORDERS, 'Ic': _H519_ORDERS,
        'In': (3, 5, 7, 9, 11, 13),
    }

    #: (quantity_measured, characteristic, phase) → adaptive DataFrame column.
    _SPEC_ADAPTIVE = {
        ('voltage', 'RMS', 'an'): 'van_v',
        ('voltage', 'RMS', 'bn'): 'vbn_v',
        ('voltage', 'RMS', 'cn'): 'vcn_v',
        ('voltage', 'RMS', 'ng'): 'vne_v',
        ('voltage', 'RMS', 'ab'): 'vab_v',
        ('voltage', 'RMS', 'bc'): 'vbc_v',
        ('voltage', 'RMS', 'ca'): 'vac_v',
        ('current', 'RMS', 'an'): 'ia_a',
        ('current', 'RMS', 'bn'): 'ib_a',
        ('current', 'RMS', 'cn'): 'ic_a',
        ('current', 'RMS', 'ng'): 'in_a',
        ('voltage', 'TOTAL_THD', 'an'): 'thd_van_pct',
        ('voltage', 'TOTAL_THD', 'bn'): 'thd_vbn_pct',
        ('voltage', 'TOTAL_THD', 'cn'): 'thd_vcn_pct',
        ('current', 'TOTAL_THD', 'an'): 'thd_ia_pct',
        ('current', 'TOTAL_THD', 'bn'): 'thd_ib_pct',
        ('current', 'TOTAL_THD', 'cn'): 'thd_ic_pct',
        ('power', 'P',  'none'): 'kw_w',
        ('power', 'Q',  'none'): 'kvar_var',
        ('power', 'S',  'none'): 'kva_va',
        ('power', 'PF', 'none'): 'adap_pf',
        ('voltage', 'FREQUENCY', 'none'): 'adap_freq',
        ('voltage', 'AVG_IMBAL', 'none'): 'v_imbalance_pct',
        ('current', 'AVG_IMBAL', 'none'): 'i_imbalance_pct',
        ('voltage', 'S2S1', 'none'): 'v_nps_pps_pct',
        ('current', 'TOTAL_THD', 'ng'): 'thd_in_pct',
        # Deliberately absent: ('voltage', 'TOTAL_THD', 'ng') -- THD of the
        # neutral-to-earth voltage.  Divided by a reference near zero it is
        # numerical noise (median 171 % on the files checked), and because this
        # record logs a sample per change it is the single largest channel here
        # -- half a million points, a fifth of the whole record -- which would
        # inflate the union index for nothing.  It is still exposed on the
        # interval grid as thd_voltage_neutral, where it costs nothing.
        ('voltage', 'FLKR_PST', 'an'): 'pst_van',
        ('voltage', 'FLKR_PST', 'bn'): 'pst_vbn',
        ('voltage', 'FLKR_PST', 'cn'): 'pst_vcn',
        ('voltage', 'FLKR_PLT', 'an'): 'plt_van',
        ('voltage', 'FLKR_PLT', 'bn'): 'plt_vbn',
        ('voltage', 'FLKR_PLT', 'cn'): 'plt_vcn',
        ('power', 'Q_FUND', 'none'): 'kvar_fund_var',
    }

    def _load_spec(self) -> None:
        """Load a standards-compliant PQDIF file by structural traversal."""
        spec = self._spec
        assert spec is not None
        interval_obs, adaptive_obs, waveform_obs = self._classify_observations()

        if not interval_obs:
            raise ValueError(
                f"{self.filepath.name}: no interval (uniform time base) "
                f"observation record found among {len(spec.observations)} "
                "observations. Observation names: "
                + ", ".join(repr(o.name) for o in spec.observations[:8])
            )

        log.info(
            "ProntoAdapter (spec): PQDIF %d.%d, %d channel definitions, "
            "%d interval / %d adaptive / %d waveform observations",
            spec.version[0], spec.version[1], len(spec.definitions),
            len(interval_obs), 1 if adaptive_obs else 0, len(waveform_obs),
        )

        self._load_spec_interval(interval_obs)
        if adaptive_obs is not None:
            self._load_spec_adaptive(adaptive_obs)
        self._load_spec_waveforms(waveform_obs)

    def _classify_observations(self):
        """Split observations by structure rather than by label text.

        Three kinds occur, distinguished by what the file itself says:

        * waveform captures  -- channels of quantity type WAVEFORM
        * interval trends    -- every channel shares one common time base
        * variable adaptive  -- each channel carries its own time base

        Naming ('Interval (avg)', 'Variable Adaptive') is a Pronto convention,
        not part of the standard, so it is used only for logging.
        """
        interval, waveform = [], []
        adaptive = None
        for obs in self._spec.observations:
            if not obs.channels:
                continue
            if any(c.quantity_type == 'WAVEFORM' for c in obs.channels):
                waveform.append(obs)
            elif self._shares_one_time_base(obs):
                interval.append(obs)
            elif adaptive is None or len(obs.channels) > len(adaptive.channels):
                # More than one per-channel-time-base record would be unusual;
                # keep the richest and note the rest.
                if adaptive is not None:
                    log.warning(
                        "ProntoAdapter (spec): ignoring extra variable-rate "
                        "observation %r (%d channels)",
                        adaptive.name, len(adaptive.channels),
                    )
                adaptive = obs
        return interval, adaptive, waveform

    @staticmethod
    def _shares_one_time_base(obs) -> bool:
        """True when every channel in the observation has the same TIME series."""
        reference: Optional[np.ndarray] = None
        for channel in obs.channels:
            t = channel.time
            if t is None or len(t) == 0:
                return False
            if reference is None:
                reference = t
                continue
            if len(t) != len(reference):
                return False
            # Compare endpoints and midpoint rather than the whole array: a
            # shared time base is written once and referenced, so any two
            # channels either match everywhere or diverge immediately.
            probe = (0, len(t) // 2, len(t) - 1)
            if any(t[i] != reference[i] for i in probe):
                return False
        return reference is not None

    @staticmethod
    def _step_pair_stride(t: np.ndarray) -> int:
        """Detect Pronto's step-pair interval encoding.

        Each interval is written as two points with the same value, one at the
        interval start and one at its end, so a viewer can draw a horizontal
        segment::

            t = [0, 64.095, 64.095001, 184.095, 184.095001, …]
            v = [122.2, 122.2, 122.0,   122.0,   121.8, …]

        The gap *between* pairs is ~1 µs while the gap *within* a pair is the
        interval length, which makes the encoding unambiguous.  Returns 2 when
        the series is step-paired (take every other point) and 1 otherwise.
        This is a Pronto convention, not part of IEEE 1159.3, so it is detected
        rather than assumed.
        """
        if len(t) < 4 or len(t) % 2 != 0:
            return 1
        gaps = np.diff(t)
        within = float(np.median(gaps[0::2]))
        between = float(np.median(gaps[1::2]))
        if within > 0 and 0 <= between <= 1e-3 * within:
            return 2
        return 1

    #: PQDIF epoch (Annex A): timestamps count days from here.
    _PQDIF_EPOCH = datetime(1900, 1, 1)

    def _spec_times(self, obs, t: np.ndarray, units: int) -> np.ndarray:
        """Convert a channel TIME series to absolute datetime64[ns].

        The observation's tagTimeStart is the authoritative start of the
        record.  Note that Pronto's *label* dates disagree with tagTimeStart by
        a fixed two days on the files checked, which is why the old reader --
        which scraped the date out of a waveform label -- dated every report
        two days early.

        ``units`` is the TIME series' ID_QU_* value: seconds relative to
        tagTimeStart (the normal case), cycles relative to it, or absolute.
        """
        if units == pqdif.UNITS_TIMESTAMP:
            base = np.datetime64(self._PQDIF_EPOCH, 'ns')
            return base + (t * 1e9).astype('int64').view('timedelta64[ns]')

        start = obs.start_time
        if start is None:
            log.warning(
                "ProntoAdapter (spec): observation %r has no tagTimeStart; "
                "timestamps will be relative to 1900-01-01.", obs.name,
            )
            start = self._PQDIF_EPOCH

        # Cycles are relative to tagTimeStart too, at the nominal frequency.
        seconds = (t / 60.0) if units == pqdif.UNITS_CYCLES else t
        base = np.datetime64(start.replace(tzinfo=None), 'ns')
        return base + (seconds * 1e9).astype('int64').view('timedelta64[ns]')

    def _load_spec_interval(self, interval_obs: List) -> None:
        """Build the interval channel set from every uniform-grid observation.

        Pronto splits interval data across two records that share one time
        base: 'Interval (avg)' holds the derived quantities (THD, harmonics,
        power, flicker) and 'Interval (max-min)' holds the true RMS voltages
        and currents with their per-interval MAX and MIN.  Both are needed, so
        channels are pooled across them.
        """
        # The largest observation defines the grid; any observation whose grid
        # matches contributes channels to the same pool.
        interval_obs = sorted(interval_obs, key=lambda o: -len(o.channels))
        primary = interval_obs[0]
        reference = primary.channels[0].time
        stride = self._step_pair_stride(reference)

        obs_times = self._spec_times(primary, reference[0::stride],
                                     primary.channels[0].time_units_id)
        self._obs_ts = obs_times
        n = len(obs_times)

        pooled: List = []
        for obs in interval_obs:
            t = obs.channels[0].time
            if len(t) != len(reference) or t[0] != reference[0] or t[-1] != reference[-1]:
                log.warning(
                    "ProntoAdapter (spec): observation %r has %d samples on a "
                    "different time base than %r (%d); its channels are skipped.",
                    obs.name, len(t), primary.name, len(reference),
                )
                continue
            pooled.extend(obs.channels)

        log.info(
            "ProntoAdapter (spec): %d interval channels pooled from %d "
            "observation(s), %d intervals%s",
            len(pooled), len(interval_obs), n,
            " (step-paired encoding)" if stride == 2 else "",
        )

        def measured(channel) -> Optional[np.ndarray]:
            """The channel's average value series, deduplicated and padded.

            Only AVG and VAL are accepted: a channel carrying just MAX and MIN
            has no average, and substituting an extreme for one would silently
            bias every downstream statistic.
            """
            values = channel.series.get('AVG', channel.series.get('VAL'))
            if values is None:
                return None
            values = np.asarray(values, dtype=float)[0::stride]
            if len(values) < n:
                values = np.pad(values, (0, n - len(values)),
                                constant_values=np.nan)
            return values[:n]

        ch_defs: List[Tuple] = []
        arrays: List[np.ndarray] = []
        canonical_index: Dict[str, int] = {}

        def add(label: str, canonical: Optional[str], qt: str, qm: str,
                phase: str, unit: str, values: np.ndarray) -> None:
            index = len(ch_defs)
            ch_defs.append((index, label, qt, qm, phase, unit))
            arrays.append(values)
            if canonical:
                canonical_index[canonical] = index

        # ── Scalar quantities, keyed on the file's own series metadata ────
        seen: Set[Tuple[str, str, str]] = set()
        harmonics: Dict[Tuple[str, int], np.ndarray] = {}
        peaks: Dict[str, np.ndarray] = {}
        mins: Dict[str, np.ndarray] = {}

        unmapped: Dict[Tuple[str, str, str], List[str]] = {}

        for channel in pooled:
            key = (channel.quantity_measured, channel.characteristic,
                   channel.phase)
            entry = self._SPEC_CHANNELS.get(key)
            prefix = self._SPEC_NAME_PREFIX.get(channel.characteristic)
            if entry is not None and prefix is not None \
                    and not channel.name.lower().startswith(prefix):
                entry = None
            if entry is not None:
                if key in seen:
                    continue
                values = measured(channel)
                if values is None:
                    continue
                seen.add(key)
                label, canonical, qt, qm, phase = entry
                add(label, canonical, qt, qm, phase, channel.units, values)

                # Per-interval extremes, where the instrument recorded them.
                for value_type, sink in (('MAX', peaks), ('MIN', mins)):
                    extreme = channel.series.get(value_type)
                    if extreme is None:
                        continue
                    extreme = np.asarray(extreme, dtype=float)[0::stride]
                    if len(extreme) < n:
                        extreme = np.pad(extreme, (0, n - len(extreme)),
                                         constant_values=np.nan)
                    sink[canonical] = extreme[:n]
                continue

            # ── Individual harmonic orders ────────────────────────────────
            match = self._HARM_NAME.match(channel.name)
            if match and channel.characteristic in ('SPECTRA_HGROUP', 'SPECTRA',
                                                    'HRMS'):
                order, group = int(match.group(1)), match.group(2)
                if (group, order) not in harmonics:
                    values = measured(channel)
                    if values is not None:
                        harmonics[(group, order)] = values
                continue

            # Nothing recognised this channel.  It parsed fine -- we simply have
            # no canonical column for it -- so record it and report once below.
            # This is how a new firmware's additions announce themselves instead
            # of quietly going missing from the analysis.
            if channel.characteristic not in self._SPEC_IGNORED:
                unmapped.setdefault(key, []).append(channel.name)

        if unmapped:
            log.info(
                "ProntoAdapter (spec): %d channel group(s) in this file have no "
                "canonical column and are not analysed. Add to _SPEC_CHANNELS "
                "to expose them: %s",
                len(unmapped),
                "; ".join(
                    f"{k[0]}/{k[1]}/phase={k[2]} ({names[0]!r}"
                    + (f" +{len(names) - 1} more)" if len(names) > 1 else ")")
                    for k, names in sorted(unmapped.items())
                ),
            )

        # Emit the reported harmonic orders in a stable order.
        for group, orders in self._HARM_REPORT.items():
            qt = 'voltageharmonics' if group.startswith('V') else 'currentharmonics'
            phase = {'Van': 'an', 'Vbn': 'bn', 'Vcn': 'cn',
                     'Ia': 'an', 'Ib': 'bn', 'Ic': 'cn', 'In': 'neutral'}[group]
            unit = 'V' if group.startswith('V') else 'A'
            for order in orders:
                values = harmonics.get((group, order))
                if values is None:
                    continue
                canonical = (f"h{order}_voltage_{phase[0]}"
                             if group.startswith('V')
                             else (f"h{order}_current_neutral"
                                   if phase == 'neutral'
                                   else f"h{order}_current_{phase[0]}"))
                add(f'H{order} {group}', canonical, qt, f'h{order}', phase,
                    unit, values)

        # ── Fall back to computed THD only where the meter reported none ──
        # IEEE 519 defines THD against the fundamental, so H1 is the correct
        # denominator.  The instrument's own TOTAL_THD channel is preferred:
        # its characteristic is stated in the file, so there is no need to
        # second-guess the label.
        for group, phase, label, canonical, qt in (
            ('Ia', 'an', 'THD Ia', 'thd_current_a', 'currentharmonics'),
            ('Ib', 'bn', 'THD Ib', 'thd_current_b', 'currentharmonics'),
            ('Ic', 'cn', 'THD Ic', 'thd_current_c', 'currentharmonics'),
        ):
            if canonical in canonical_index:
                continue
            h1 = harmonics.get((group, 1))
            if h1 is None:
                continue
            # Prefer the meter's harmonic RMS aggregate for the numerator: it is
            # computed at full precision inside the instrument, whereas summing
            # the reported orders loses whatever the display resolution rounded
            # off -- badly at light load, where each order is comparable to the
            # resolution itself.
            hrms_index = canonical_index.get(f'hrms_current_{phase}')
            if hrms_index is not None:
                numerator, source = arrays[hrms_index], 'meter harmonic RMS'
            else:
                orders = [harmonics[(group, h)] ** 2
                          for h in range(2, 51) if (group, h) in harmonics]
                if not orders:
                    continue
                numerator = np.sqrt(sum(orders))
                source = f'{len(orders)} harmonic orders'
            denominator = np.where(h1 > 0.01, h1, np.nan)
            log.info(
                "ProntoAdapter (spec): %s not reported by the meter; "
                "computed from %s.", label, source,
            )
            add(label, canonical, qt, 'thd', phase, '%',
                numerator / denominator * 100.0)

        self._raw_channels = [
            RawChannelInfo(index, label, qt, qm, phase, unit)
            for (index, label, qt, qm, phase, unit) in ch_defs
        ]
        self._obs_data = {cd[0]: arr for cd, arr in zip(ch_defs, arrays)}
        self._interval_peaks = peaks
        self._interval_mins = mins

        interval_minutes = (
            float(np.median(np.diff(obs_times).astype('int64'))) / 60e9
            if n >= 2 else 0.0
        )
        log.info(
            "ProntoAdapter (spec): %d channels, %d intervals of %.1f min "
            "(%s → %s), %d peak / %d min series",
            len(self._raw_channels), n, interval_minutes,
            pd.Timestamp(obs_times[0]).strftime('%Y-%m-%d %H:%M') if n else '–',
            pd.Timestamp(obs_times[-1]).strftime('%Y-%m-%d %H:%M') if n else '–',
            len(peaks), len(mins),
        )

    def _load_spec_adaptive(self, obs) -> None:
        """Build the variable-rate DataFrame.

        Every channel has its own time base here, so each becomes a pandas
        Series on its own index and the result is the outer join.  Channels are
        named from their series metadata, which replaces the old approach of
        correlating each unnamed channel against the interval averages.
        """
        columns: Dict[str, pd.Series] = {}
        unmapped: Dict[Tuple[str, str, str], List[str]] = {}
        for channel in obs.channels:
            key = (channel.quantity_measured, channel.characteristic,
                   channel.phase)
            column = self._SPEC_ADAPTIVE.get(key)
            prefix = self._SPEC_NAME_PREFIX.get(channel.characteristic)
            if column is not None and prefix is not None \
                    and not channel.name.lower().startswith(prefix):
                column = None
            if column is None or column in columns:
                if (column is None
                        and channel.characteristic not in self._SPEC_IGNORED):
                    unmapped.setdefault(key, []).append(channel.name)
                continue
            t = channel.time
            values = channel.series.get('AVG', channel.series.get('VAL'))
            if t is None or values is None or len(t) == 0:
                continue
            size = min(len(t), len(values))
            index = pd.DatetimeIndex(
                self._spec_times(obs, t[:size], channel.time_units_id))
            series = pd.Series(np.asarray(values[:size], dtype=float),
                               index=index)
            series = series[~series.index.duplicated(keep='first')]
            columns[column] = series

        if not columns:
            log.warning(
                "ProntoAdapter (spec): no recognised channels in variable-rate "
                "observation %r; skipped.", obs.name,
            )
            return

        df = pd.concat([s.rename(name) for name, s in columns.items()], axis=1)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        self._adaptive_df = df

        span_hours = ((df.index[-1] - df.index[0]).total_seconds() / 3600
                      if len(df) > 1 else 0.0)
        log.info(
            "ProntoAdapter (spec): variable-rate record %d rows × %d channels, "
            "%.1f h span",
            len(df), len(df.columns), span_hours,
        )
        if unmapped:
            log.info(
                "ProntoAdapter (spec): %d variable-rate channel group(s) have no "
                "column and are not analysed. Add to _SPEC_ADAPTIVE to expose "
                "them: %s",
                len(unmapped),
                "; ".join(
                    f"{k[0]}/{k[1]}/phase={k[2]} ({names[0]!r})"
                    for k, names in sorted(unmapped.items())
                ),
            )

    def _load_spec_waveforms(self, waveform_obs: List) -> None:
        """Decode point-on-wave captures.

        Each capture channel states its own phase and whether it is a voltage
        or a current, so no amplitude-based guessing is needed.
        """
        for obs in waveform_obs:
            voltages: Dict[str, np.ndarray] = {}
            currents: Dict[str, np.ndarray] = {}
            vne: Optional[np.ndarray] = None
            times: Optional[np.ndarray] = None

            for channel in obs.channels:
                samples = channel.value('VAL', 'INSTANTANEOUS')
                if samples is None or channel.time is None:
                    continue
                samples = np.asarray(samples, dtype=float)
                if times is None or len(channel.time) > len(times):
                    times = np.asarray(channel.time, dtype=float)
                phase = channel.phase
                if channel.quantity_measured == 'voltage':
                    if phase == 'ng':
                        vne = samples
                    elif phase in ('an', 'bn', 'cn'):
                        voltages[phase[0]] = samples
                elif channel.quantity_measured == 'current':
                    currents['n' if phase == 'ng' else phase[0]] = samples

            if not voltages or times is None or len(times) < 2:
                continue
            dt = float(np.median(np.diff(times)))
            self._waveforms.append({
                "timestamp": obs.start_time,
                "label":     obs.name,
                "t":         times,
                "fs_hz":     (1.0 / dt) if dt > 0 else None,
                "voltages":  voltages,
                "vne":       vne,
                "currents":  currents,
            })

        if self._waveforms:
            self._waveforms.sort(key=lambda w: w["timestamp"])
            first = self._waveforms[0]
            log.info(
                "ProntoAdapter (spec): %d point-on-wave captures "
                "(%d V / %d I channels, %.1f kHz, ~%.0f samples/cycle)",
                len(self._waveforms), len(first["voltages"]),
                len(first["currents"]), (first["fs_hz"] or 0) / 1000,
                (first["fs_hz"] or 0) / 60.0,
            )

    def _build_timestamps(
        self, obs1_body: bytes, base_date: datetime
    ) -> Tuple[int, np.ndarray]:
        off = self._O1_FIRST + self._O1_TSREL
        ts_sec = self._read_f64(obs1_body, off)
        if ts_sec is None or len(ts_sec) < 2:
            raise ValueError("ProntoAdapter: cannot read timestamps from obs[1].")
        ts_dedup = ts_sec[0::2]
        n = len(ts_dedup)
        base_ns = np.datetime64(base_date.replace(tzinfo=None), 'ns')
        ts = np.array(
            [base_ns + np.timedelta64(int(t * 1e9), 'ns') for t in ts_dedup],
            dtype='datetime64[ns]',
        )
        return n, ts

    @staticmethod
    def _pqdif_elements(body: bytes, col_off: int) -> List[Dict]:
        """Parse a PQDIF element-list: [u32 count][count × 28-byte elements].
        Each element: 16-byte GUID (first 4 bytes used as key), 4-byte type,
        4-byte offset, 4-byte size. Zero-size scalars store value inline in offset."""
        if col_off + 4 > len(body):
            return []
        count = struct.unpack_from('<I', body, col_off)[0]
        if count > 100_000:
            return []
        out: List[Dict] = []
        for i in range(count):
            base = col_off + 4 + i * 28
            if base + 28 > len(body):
                break
            out.append({
                'guid4': struct.unpack_from('<I', body, base)[0],
                'type':  struct.unpack_from('<I', body, base + 16)[0],
                'off':   struct.unpack_from('<I', body, base + 20)[0],
                'sz':    struct.unpack_from('<I', body, base + 24)[0],
            })
        return out

    @staticmethod
    def _find_channel_instances_off(obs_body: bytes) -> Optional[int]:
        """Structural offset of the ChannelInstances element-list within an obs
        body, read directly from the top-level PQDIF element table (guid4
        0x3D786F91) rather than guessed from label length. Both the DS-label
        map (_build_label_map) and the per-channel data pointers (_load_v2)
        key off the same 'ci' index into this same table, so both must resolve
        addresses through it -- see _load_v2's entry_start cross-check."""
        for e in ProntoAdapter._pqdif_elements(obs_body, 0):
            if e['guid4'] == 0x3D786F91:
                return e['off']
        return None

    def _build_label_map(self, all_recs: List[Dict], obs_body: bytes) -> Dict[str, int]:
        """Build {label → obs_ci} from DataSource channel names and ChannelInstances.
        Labels decoded with latin-1 to preserve all byte values (CP1253 phi=0xF8)."""
        ds_body: Optional[bytes] = None
        for r in all_recs:
            if r['tag'] == 0x89738619:
                try:
                    ds_body = zlib.decompress(r['raw'])
                except zlib.error:
                    ds_body = r['raw']
                break
        if ds_body is None:
            return {}

        ds_top = self._pqdif_elements(ds_body, 0)
        cd_off: Optional[int] = None
        for e in ds_top:
            if e['guid4'] == 0xB48D858D:
                cd_off = e['off']
                break
        if cd_off is None:
            return {}

        ds_label: Dict[int, str] = {}
        for ds_ci, e in enumerate(self._pqdif_elements(ds_body, cd_off)):
            if e['type'] != 1:
                continue
            for s in self._pqdif_elements(ds_body, e['off']):
                if s['guid4'] == 0xB48D8590 and s['sz'] > 4:
                    raw = ds_body[s['off'] + 4 : s['off'] + s['sz']]
                    lbl = raw.rstrip(b'\x00').decode('latin-1').strip()
                    if lbl:
                        ds_label[ds_ci] = lbl
                    break

        ci_off = self._find_channel_instances_off(obs_body)
        if ci_off is None:
            return {}

        label_map: Dict[str, int] = {}
        for obs_ci, e in enumerate(self._pqdif_elements(obs_body, ci_off)):
            if e['type'] != 1:
                continue
            for s in self._pqdif_elements(obs_body, e['off']):
                if s['guid4'] == 0xB48D858F:
                    ds_ci = s['off']   # DS channel index stored inline in offset field
                    lbl = ds_label.get(ds_ci)
                    if lbl:
                        label_map[lbl] = obs_ci
                    break

        return label_map

    _PRINTABLE_RUN = re.compile(rb'[\x20-\x7e]{4,}')

    @staticmethod
    def _describe_obs_label(body: bytes) -> str:
        """Best-effort human-readable label for a decompressed obs body, for error
        diagnostics only. Tries the expected label_length/label field at offset
        144/148 first (see _load_v2), then falls back to the longest run of
        printable text in the first 300 bytes in case the layout has shifted."""
        if len(body) > 148:
            length = struct.unpack_from('<I', body, 144)[0]
            if 1 <= length <= 200 and 148 + length <= len(body):
                candidate = body[148:148 + length].rstrip(b'\x00')
                if candidate:
                    return repr(candidate.decode('latin-1'))
        runs = ProntoAdapter._PRINTABLE_RUN.findall(body[:300])
        if runs:
            return repr(max(runs, key=len).decode('latin-1')) + ' (fallback scan)'
        return '<no printable label found in first 300 bytes>'

    @classmethod
    def _describe_obs_records(cls, obs_recs: List[Dict]) -> str:
        """One diagnostic line per Observation record: decompressed size and its
        apparent label. Emitted in the 'Interval (avg) not found' error so a user's
        traceback alone is enough to tell us what labels this file actually uses,
        without needing the file itself."""
        lines = []
        for i, rec in enumerate(obs_recs):
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error as exc:
                lines.append(f"  obs[{i}]: zlib decompression failed ({exc}), raw_len={len(rec['raw'])}")
                continue
            lines.append(f"  obs[{i}]: body_len={len(body)}, label={cls._describe_obs_label(body)}")
        return "\n".join(lines)

    def _load_v2(self, obs_recs: List[Dict], all_recs: List[Dict]) -> None:
        """Load new Pronto format using DataSource label-based channel discovery.

        Channel layout is derived by reading the DataSource record (which contains
        516 named channel definitions) and mapping each ChannelInstance in the
        Interval (avg) obs body to its DataSource label.  This replaces the old
        approach of hardcoded positional indices (which were firmware-specific).

        Key findings from binary analysis of Pronto PQDIF files:
          - Power labels use CP1253 encoding: the phi character (φ) is byte 0xF8,
            decoded here with latin-1 as '\\xf8'.
          - 'THD Ia (I1)' / 'Hrms Ia' DS labels have swapped meanings in the
            Pronto exporter firmware; THD is computed here from the harmonic block.
          - 'Harm 1 of Ia' (observed at a non-trivial obs_ci, e.g. 109 on a real
            3-phase commercial file) is the I1 fundamental in Amps, consistent
            with measured apparent power (VA/3/Vln).
        """
        interval_body: Optional[bytes] = None
        for rec in obs_recs:
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error:
                continue
            if b'Interval (avg)' in body[148:220]:
                interval_body = body
                break

        if interval_body is None:
            raise ValueError(
                "ProntoAdapter v2: could not find 'Interval (avg)' observation record. "
                "Is this a Pronto PQDIF file?\n"
                f"Found {len(obs_recs)} Observation record(s):\n"
                + self._describe_obs_records(obs_recs)
            )

        base_date = self._parse_v2_date(obs_recs)

        # ── Dynamic entry_start: read label_length from bytes 144-147 ────────
        label_length = struct.unpack_from('<I', interval_body, 144)[0]
        if not (1 <= label_length <= 512):
            raise ValueError(
                f"ProntoAdapter v2: unexpected label_length {label_length} in "
                "'Interval (avg)' obs body — unsupported Pronto PQDIF format."
            )
        heuristic_entry_start = 148 + ((label_length + 3) & ~3) + 28

        # ── Structural ChannelInstances table (authoritative) ────────────────
        # entry_start used to be derived only from label_length padding, which
        # happened to match this table's real start on every file seen so far.
        # Read the table's own pointer (same one _build_label_map keys 'ci'
        # against) instead of assuming that match always holds -- a per-channel
        # index that comes from that same table must resolve addresses through
        # it, not through a separately guessed offset.
        ci_off = self._find_channel_instances_off(interval_body)
        ci_elements = self._pqdif_elements(interval_body, ci_off) if ci_off is not None else []
        entry_start = (ci_off + 4) if ci_off is not None else heuristic_entry_start
        if ci_off is not None and entry_start != heuristic_entry_start:
            log.warning(
                "ProntoAdapter v2: ChannelInstances table starts at %d but the "
                "label_length=%d heuristic predicted %d -- using the structural "
                "table. If channel data still comes back empty, this file's "
                "layout needs more adapter support.",
                entry_start, label_length, heuristic_entry_start,
            )

        def channel_abs(ci: int) -> Optional[int]:
            if ci < len(ci_elements):
                return ci_elements[ci]['off']
            pos = entry_start + ci * self._V2_ENTRY_SIZE + self._V2_BODY_OFF_REL
            if pos + 4 > len(interval_body):
                return None
            return struct.unpack_from('<I', interval_body, pos)[0]

        # ── Dynamic DATA_REL ────────────────────────────────────────────────
        ch0_abs = channel_abs(0)
        if ch0_abs is None:
            raise ValueError(
                f"ProntoAdapter v2: entry_start={entry_start} + body_off="
                f"{self._V2_BODY_OFF_REL} exceeds body length {len(interval_body)}."
            )
        ts_abs = ch0_abs + self._V2_TS_REL
        if ts_abs + 4 > len(interval_body):
            raise ValueError(
                f"ProntoAdapter v2: ch0 pointer {ch0_abs} + TS_REL {self._V2_TS_REL} "
                f"= {ts_abs} exceeds body length {len(interval_body)}. "
                f"label_length={label_length}, entry_start={entry_start}. "
                "File may use an unsupported Pronto firmware version."
            )
        ts_count_raw = struct.unpack_from('<I', interval_body, ts_abs)[0]
        data_rel = self._V2_TS_REL + 4 + ts_count_raw * 8 + 32

        # Structural diagnostics only -- byte offsets and counts, never any
        # measured value. If every channel on a file comes back empty (see the
        # all-columns-NaN guard in extract_dataframe), these are the numbers
        # that pin down where the per-file layout assumption broke: data_rel is
        # computed once here and reused for every channel via read_v2() below,
        # so a wrong value here explains a total (not per-channel) failure.
        log.info(
            "ProntoAdapter v2 structure: label_length=%d entry_start=%d ch0_abs=%d "
            "ts_abs=%d ts_count_raw=%d data_rel=%d body_len=%d",
            label_length, entry_start, ch0_abs, ts_abs, ts_count_raw, data_rel,
            len(interval_body),
        )

        # The per-channel value count naturally equals ts_count_raw -- there's
        # one raw sample pair per timestamp, so this is the correct, expected
        # case, not a broken signature. The old fixed "> 15_000" ceiling here
        # was the actual bug: it silently rejected every channel (as if empty)
        # on any recording long enough for ts_count_raw to exceed 15,000 (e.g.
        # a 7-day 1-minute recording lands around 20,000) while a real 4064
        # from a ~3-day recording passed fine. Bound it against the file's own
        # ts_count_raw instead of a flat guess -- a channel can't legitimately
        # have more raw samples than there are timestamps.
        _count_ceiling = ts_count_raw

        _logged_call_count = [0]

        def read_v2(ci: int) -> np.ndarray:
            ch_abs = channel_abs(ci)
            if ch_abs is None:
                return np.array([np.nan])
            data_abs = ch_abs + data_rel
            count = struct.unpack_from('<I', interval_body, data_abs)[0]
            if _logged_call_count[0] < 5:
                _logged_call_count[0] += 1
                log.info(
                    "ProntoAdapter v2 channel ci=%d: ch_abs=%d data_abs=%d count=%d "
                    "(valid range is 1-%d; outside that -> treated as empty)",
                    ci, ch_abs, data_abs, count, _count_ceiling,
                )
            if count == 0 or count > _count_ceiling:
                return np.array([np.nan])
            raw = np.frombuffer(
                interval_body[data_abs + 4 : data_abs + 4 + count * 8], dtype='<f8'
            )
            vals = raw[0::2]
            return vals[np.isfinite(vals)]

        # ── Timestamps from channel 0 ───────────────────────────────────────
        ts_raw = np.frombuffer(
            interval_body[ts_abs + 4 : ts_abs + 4 + ts_count_raw * 8], dtype='<f8'
        )
        ts_secs = ts_raw[0::2]
        n = len(ts_secs)

        base_ns = np.datetime64(base_date.replace(tzinfo=None), 'ns')
        self._obs_ts = np.array(
            [base_ns + np.timedelta64(int(t * 1e9), 'ns') for t in ts_secs],
            dtype='datetime64[ns]',
        )

        def pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) < n:
                return np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
            return arr[:n]

        # ── Label-based channel discovery ───────────────────────────────────
        label_map = self._build_label_map(all_recs, interval_body)
        if len(label_map) < 5:
            log.warning(
                "ProntoAdapter v2: label map has only %d entries — "
                "DataSource record may be missing or unreadable.",
                len(label_map),
            )

        # Split-phase detection: no 'Harm 1 of Vcn' → no C phase
        is_split_phase = 'Harm 1 of Vcn' not in label_map

        # ── Build channel list from label map ───────────────────────────────
        ch_defs: List[Tuple] = []
        arrays:  List[np.ndarray] = []
        local_idx = 0

        def add(human: str, qt: str, qm: str, phase: str,
                unit: str, arr: np.ndarray) -> None:
            nonlocal local_idx
            ch_defs.append((local_idx, human, qt, qm, phase, unit))
            arrays.append(arr)
            local_idx += 1

        def rv(ci: Optional[int]) -> np.ndarray:
            return pad(read_v2(ci)) if ci is not None else np.full(n, np.nan)

        # Direct single-label → channel mappings.
        # Power labels: Pronto firmware stores the phi symbol as CP1253 byte 0xF8;
        # we decode all DS labels with latin-1 so 0xF8 → '\xf8' in both label and pattern.
        _DIRECT: List[Tuple[str, str, str, str, str, str]] = [
            ('Harm 1 of Van',         'Van RMS',      'voltage',          'rms',         'an',      'V'  ),
            ('Harm 1 of Vbn',         'Vbn RMS',      'voltage',          'rms',         'bn',      'V'  ),
            ('Harm 1 of Vcn',         'Vcn RMS',      'voltage',          'rms',         'cn',      'V'  ),
            ('Harm 1 of Vne',         'Vne RMS',      'voltage',          'rms',         'neutral', 'V'  ),
            ('Harm 1 of Ia',          'Ia RMS',       'current',          'rms',         'an',      'A'  ),
            ('Harm 1 of Ib',          'Ib RMS',       'current',          'rms',         'bn',      'A'  ),
            ('Harm 1 of Ic',          'Ic RMS',       'current',          'rms',         'cn',      'A'  ),
            ('Harm 1 of In',          'In RMS',       'current',          'rms',         'neutral', 'A'  ),
            ('3\xf8 4w Real Power',   'Real Power',   'watts',            'watts',       'total',   'W'  ),
            ('3\xf8 4w VA Reactive',  'React. Power', 'power',            'reactive',    'total',   'VAR'),
            ('3\xf8 4w Power Factor', 'Power Factor', 'powerfactor',      'powerfactor', 'total',   ''   ),
            ('THD Van (V1)',           'THD Van',      'voltageharmonics', 'thd',         'an',      '%'  ),
            ('THD Vbn (V2)',           'THD Vbn',      'voltageharmonics', 'thd',         'bn',      '%'  ),
            ('THD Vcn (V3)',           'THD Vcn',      'voltageharmonics', 'thd',         'cn',      '%'  ),
            ('K-Factor Ia',           'K-Factor',     'kfactor',          'kfactor',     'total',   ''   ),
            ('Flicker PST Van (V1)',   'Flicker PST',  'flicker',          'pst',         'an',      ''   ),
            ('Flicker PLT Van (V1)',   'Flicker PLT',  'flicker',          'plt',         'an',      ''   ),
        ]
        for ds_lbl, human, qt, qm, phase, unit in _DIRECT:
            ci = label_map.get(ds_lbl)
            if ci is not None:
                add(human, qt, qm, phase, unit, rv(ci))

        # Per-order harmonic channels — load all available orders H2-H50,
        # cache for THD computation, then add standard reporting orders to output.
        _HARM_BLOCKS: List[Tuple[str, str, str, str, Tuple[int, ...]]] = [
            ('Van', 'voltageharmonics', 'an',      'V', (3, 5, 7, 11, 13)  ),
            ('Vbn', 'voltageharmonics', 'bn',      'V', (3, 5, 7, 11, 13)  ),
            ('Vcn', 'voltageharmonics', 'cn',      'V', (3, 5, 7, 11, 13)  ),
            ('Ia',  'currentharmonics', 'an',      'A', _H519_ORDERS        ),
            ('Ib',  'currentharmonics', 'bn',      'A', _H519_ORDERS        ),
            ('Ic',  'currentharmonics', 'cn',      'A', _H519_ORDERS        ),
            ('In',  'currentharmonics', 'neutral', 'A', (3, 5, 7, 9, 11, 13)),
        ]
        _harm: Dict[Tuple[str, int], np.ndarray] = {}

        for ph_key, qt, phase, unit, report_orders in _HARM_BLOCKS:
            for h in range(2, 51):
                ci = label_map.get(f'Harm {h} of {ph_key}')
                if ci is not None:
                    _harm[(ph_key, h)] = rv(ci)
            for h in report_orders:
                arr = _harm.get((ph_key, h))
                if arr is not None:
                    add(f'H{h} {ph_key}', qt, f'h{h}', phase, unit, arr)

        # Computed THD for current: sqrt(ΣHn²) / H1 × 100 %.
        # The Pronto DS label 'THD Ia (I1)' actually stores a total-current aggregate
        # (not THD%) due to a firmware label error; compute from the harmonic block.
        for ph_key, phase in (('Ia', 'an'), ('Ib', 'bn'), ('Ic', 'cn')):
            h1_ci = label_map.get(f'Harm 1 of {ph_key}')
            if h1_ci is None:
                continue
            h1 = rv(h1_ci)
            harm_sq = [_harm[(ph_key, h)] ** 2
                       for h in range(2, 51) if (ph_key, h) in _harm]
            if not harm_sq:
                continue
            h1_safe = np.where(h1 > 0.01, h1, np.nan)
            thd_arr = np.sqrt(sum(harm_sq)) / h1_safe * 100.0
            _THD_LABELS = {'an': 'THD Ia', 'bn': 'THD Ib', 'cn': 'THD Ic', 'neutral': 'THD In'}
            add(_THD_LABELS[phase], 'currentharmonics', 'thd', phase, '%', thd_arr)

        self._raw_channels = [
            RawChannelInfo(idx, label, qt, qm, phase, unit)
            for (idx, label, qt, qm, phase, unit) in ch_defs
        ]
        self._obs_data = {cd[0]: arr for cd, arr in zip(ch_defs, arrays)}

        dt_min = round(float(np.median(np.diff(ts_secs))) / 60) if n >= 2 else 5
        topo = 'split-phase' if is_split_phase else '3-phase'
        log.info(
            "ProntoAdapter v2 (%s, label-map): %d channels, %d %d-min intervals (%s → %s)",
            topo, len(self._raw_channels), n, dt_min,
            pd.Timestamp(self._obs_ts[0]).strftime('%Y-%m-%d %H:%M') if n else '–',
            pd.Timestamp(self._obs_ts[-1]).strftime('%Y-%m-%d %H:%M') if n else '–',
        )
        self._load_adaptive(obs_recs, base_date)
        self._load_waveforms(obs_recs)
        self._load_v2_maxmin(obs_recs, n)

    def _load_v2_maxmin(self, obs_recs: List[Dict], n: int) -> None:
        """Parse obs[24] 'Interval (max-min)' record into interval_peaks / interval_mins.

        The maxmin record uses the same PQDIF framing as the avg record but has a longer
        label string, which shifts the entry-table start and the intra-block header by a
        few bytes.  Each channel block also contains three separate data blobs (max, min,
        and a third section — probably per-interval average) rather than one.

        Channel layout confirmed from binary inspection of a real 3-phase commercial .pqd file:
          ci=0  voltage_a (Van L-N)
          ci=1  voltage_b (Vbn L-N)
          ci=2  voltage_c (Vcn L-N)
          ci=3  unknown small value (~0.1 V — possibly neutral voltage or freq. deviation)
          ci=4  Vab L-L  ← not mapped (no canonical column in avg body)
          ci=5  Vbc L-L  ← not mapped
          ci=6  Vca L-L  ← not mapped
          ci=7  kfactor_meter
          ci=8  unknown (per-phase K-factor candidate)
          ci=9  unknown (per-phase K-factor candidate)
          ci=10 thd_current_a
        """
        maxmin_body: Optional[bytes] = None
        for rec in obs_recs:
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error:
                continue
            if b'Interval (max-min)' in body[148:220]:
                maxmin_body = body
                break
        if maxmin_body is None:
            self._interval_peaks: Dict[str, np.ndarray] = {}
            self._interval_mins:  Dict[str, np.ndarray] = {}
            return

        # entry_start: bytes 144-147 hold the label length (including null terminator).
        # The label field is padded to a 4-byte boundary, then followed by a fixed 28-byte
        # header block, then the channel entry table. Prefer the structural
        # ChannelInstances pointer (same lookup used in _load_v2) over that padding
        # arithmetic -- see the entry_start cross-check there for why the heuristic
        # alone isn't reliable on every file.
        label_length = struct.unpack_from('<I', maxmin_body, 144)[0]
        heuristic_entry_start = 148 + ((label_length + 3) & ~3) + 28
        ci_off = self._find_channel_instances_off(maxmin_body)
        ci_elements = self._pqdif_elements(maxmin_body, ci_off) if ci_off is not None else []
        entry_start = (ci_off + 4) if ci_off is not None else heuristic_entry_start
        if ci_off is not None and entry_start != heuristic_entry_start:
            log.warning(
                "ProntoAdapter v2: obs[24] ChannelInstances table starts at %d but the "
                "label_length=%d heuristic predicted %d -- using the structural table.",
                entry_start, label_length, heuristic_entry_start,
            )

        def _channel_abs(ci: int) -> Optional[int]:
            if ci < len(ci_elements):
                return ci_elements[ci]['off']
            pos = entry_start + ci * self._V2_ENTRY_SIZE + self._V2_BODY_OFF_REL
            if pos + 4 > len(maxmin_body):
                return None
            return struct.unpack_from('<I', maxmin_body, pos)[0]

        ch0_abs = _channel_abs(0)
        if ch0_abs is None:
            self._interval_peaks = {}
            self._interval_mins  = {}
            return

        # The maxmin channel block has more intra-block header bytes than the avg block
        # (extra sub-blob pointers for the min and third sections).  Find ts_count_raw
        # = 2*n by scanning the ch0 block rather than using the fixed _V2_TS_REL offset.
        ts_rel: Optional[int] = None
        for off in range(0, 512, 4):
            if ch0_abs + off + 4 > len(maxmin_body):
                break
            if struct.unpack_from('<I', maxmin_body, ch0_abs + off)[0] == 2 * n:
                ts_rel = off
                break
        if ts_rel is None:
            log.warning("ProntoAdapter v2: could not locate ts_count in obs[24] ch0 block")
            self._interval_peaks = {}
            self._interval_mins  = {}
            return

        # data_rel: offset from ch_abs to the MAX-values count field.
        # Each blob is: u32 count + count×f64 values, padded with a 32-byte inter-blob header.
        blob_size = 4 + (2 * n) * 8       # count u32 + n × (value, dup) f64 pairs
        data_rel  = ts_rel + 4 + (2 * n) * 8 + 32  # skip ts blob + 32-byte separator
        min_rel   = data_rel + blob_size + 32        # skip max blob + 32-byte separator

        def _read_section(ci: int, rel: int) -> Optional[np.ndarray]:
            ch_abs = _channel_abs(ci)
            if ch_abs is None:
                return None
            abs_off = ch_abs + rel
            if abs_off + 4 > len(maxmin_body):
                return None
            count = struct.unpack_from('<I', maxmin_body, abs_off)[0]
            if count == 0 or count > 60_000:
                return None
            end = abs_off + 4 + count * 8
            if end > len(maxmin_body):
                return None
            raw  = np.frombuffer(maxmin_body[abs_off + 4 : end], dtype='<f8')
            vals = raw[0::2][:n].copy()   # every-other dedup (same as avg body)
            vals[~np.isfinite(vals)] = np.nan
            return vals if not np.all(np.isnan(vals)) else None

        # Channel index map specific to the maxmin obs record.
        # These ci values are NOT the same as the avg body's _V2_CH_* constants.
        #
        # The maxmin record stores ALL channels at the moment of peak/dip VOLTAGE,
        # not independently-tracked per-channel maxima.  Only voltage channels are
        # mapped here — for those, section 1 IS the peak voltage and section 2 IS
        # the minimum voltage.  Other channels (k-factor, THD) at those same moments
        # are not meaningful as "peaks" or "mins" of those quantities.
        mm_map: Dict[str, int] = {
            'voltage_a': 0,
            'voltage_b': 1,
            'voltage_c': 2,
        }

        peaks: Dict[str, np.ndarray] = {}
        mins:  Dict[str, np.ndarray] = {}
        for canonical, ci in mm_map.items():
            maxv = _read_section(ci, data_rel)
            minv = _read_section(ci, min_rel)
            if maxv is not None:
                peaks[canonical] = maxv
            if minv is not None:
                mins[canonical]  = minv

        self._interval_peaks = peaks
        self._interval_mins  = mins
        log.info(
            "ProntoAdapter v2: obs[24] max-min loaded — %d peak / %d min channels",
            len(peaks), len(mins),
        )

    @property
    def interval_peaks(self) -> Dict[str, np.ndarray]:
        """Per-interval maximum values from obs[24]; keys are CANONICAL column names."""
        return getattr(self, '_interval_peaks', {})

    @property
    def interval_mins(self) -> Dict[str, np.ndarray]:
        """Per-interval minimum values from obs[24]; keys are CANONICAL column names."""
        return getattr(self, '_interval_mins', {})

    def _parse_v2_date(self, obs_recs: List[Dict]) -> datetime:
        """Parse recording start date from the first waveform obs label (MM/DD/YY format)."""
        for rec in obs_recs:
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error:
                continue
            label = body[148:220].decode('ascii', errors='replace')
            m = re.search(r'(\d{2})/(\d{2})/(\d{2})', label)
            if m:
                month, day, yr2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return datetime(2000 + yr2, month, day, 0, 0, 0)
        log.warning(
            "ProntoAdapter v2: cannot parse base date from obs labels; defaulting to 2000-01-01."
        )
        return datetime(2000, 1, 1)

    def _load_adaptive(self, obs_recs: List[Dict], base_date: datetime) -> None:
        """Parse Variable Adaptive obs record into self._adaptive_df.

        Unlike interval channels (paired float64 timestamps+quality), the adaptive
        record stores single float64 timestamps and single float64 values.  Each
        channel has its own independent timestamp array.  The result is a sparse
        DataFrame on a union DatetimeIndex; each column is NaN wherever that
        specific channel has no sample at a given timestamp.
        """
        adap_body: Optional[bytes] = None
        for rec in obs_recs:
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error:
                continue
            if b'Variable Adaptive' in body[148:220]:
                adap_body = body
                break
        if adap_body is None:
            return

        ts_rel = self._ADAP_TS_REL

        # The channel entry table's start offset drifts between Pronto export
        # versions (216 in the original format, 224 in newer exports), so
        # locate it by pattern instead of a fixed constant.
        ch_offsets = self._scan_entry_table(adap_body)
        if not ch_offsets:
            log.warning(
                "ProntoAdapter adaptive: could not locate channel entry table — "
                "adaptive record skipped."
            )
            return
        log.info("ProntoAdapter adaptive: entry table with %d channels", len(ch_offsets))

        def read_adap_ch(ci: int):
            if ci >= len(ch_offsets):
                return None, None
            ch_abs = ch_offsets[ci]
            ts_cnt_pos = ch_abs + ts_rel
            if ts_cnt_pos + 4 > len(adap_body):
                return None, None
            ts_cnt = struct.unpack_from('<I', adap_body, ts_cnt_pos)[0]
            if not (1 <= ts_cnt <= 200_000):
                return None, None
            ts_start = ts_cnt_pos + 4
            ts_end   = ts_start + ts_cnt * 8
            if ts_end > len(adap_body):
                return None, None
            ts_raw = np.frombuffer(adap_body[ts_start:ts_end], dtype='<f8')
            if not (np.isfinite(ts_raw[0]) and ts_raw[0] < 100):
                return None, None
            # data block follows gap at ts_end+32 (4-byte count, then values)
            data_cnt_pos = ts_end + 32
            if data_cnt_pos + 4 > len(adap_body):
                return None, None
            dcnt = struct.unpack_from('<I', adap_body, data_cnt_pos)[0]
            if not (1 <= dcnt <= 200_000):
                return None, None
            dstart = data_cnt_pos + 4
            dend   = dstart + dcnt * 8
            if dend > len(adap_body):
                return None, None
            d_raw = np.frombuffer(adap_body[dstart:dend], dtype='<f8').copy()
            d_raw[~np.isfinite(d_raw)] = np.nan
            d_raw[(d_raw < -1e6) | (d_raw > 1e6)] = np.nan
            n = min(len(ts_raw), len(d_raw))
            return ts_raw[:n], d_raw[:n]

        base_ns = np.int64(np.datetime64(base_date.replace(tzinfo=None), 'ns').view('int64'))

        def ch_series(ci: int) -> Optional[pd.Series]:
            ts, vals = read_adap_ch(ci)
            if ts is None or len(ts) == 0:
                return None
            abs_ns = (base_ns + (ts * 1e9).astype('int64')).astype('datetime64[ns]')
            return pd.Series(vals, index=pd.DatetimeIndex(abs_ns))

        parsed = [(ci, s) for ci in range(len(ch_offsets))
                  if (s := ch_series(ci)) is not None and len(s.dropna()) >= 10]
        if not parsed:
            return

        # Channel ORDER in the adaptive record differs between split-phase and
        # three-phase exports, so identify channels by signature (correlation
        # against the interval-average channels) instead of by position.
        named = self._identify_adaptive_channels(parsed)
        if not named:
            log.warning("ProntoAdapter adaptive: no channels identified — record skipped.")
            return

        series_list = [s.rename(col) for col, s in named.items()]
        df = pd.concat(series_list, axis=1).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        self._adaptive_df = df

        ts_span_h = (df.index[-1] - df.index[0]).total_seconds() / 3600
        log.info(
            "ProntoAdapter adaptive: %d variable-rate samples, %.1f h span, %d channels",
            len(df), ts_span_h, len(series_list),
        )

    def _load_waveforms(self, obs_recs: List[Dict]) -> None:
        """Decode point-on-wave 'Waveform' capture observations.

        Each capture is one observation record labeled
        ``<meter> - Waveform - MM/DD/YY HH:MM:SS.ffff``.  Its body carries one
        entry-table (located with _scan_entry_table) pointing at ~6 channel
        blocks.  Block layout: u32 sample count at +208, float64 per-sample
        times (seconds from capture start) at +212, then a 60-byte gap, a
        repeated u32 count, and float64 instantaneous samples in engineering
        units (V or A).  Typical captures: ~3 000 samples per channel at
        ≈19.2 kHz (~320 samples/cycle), 0.1–1.5 s per capture.

        Channels carry no identifying GUID, so they are classified by
        amplitude against the interval Van RMS median: voltage phases first
        (in table order), a near-zero block immediately after them as Vne,
        and the remainder as phase currents (last one neutral when present).
        """
        # Reference L-N voltage from the interval channels for classification
        ref_v = 120.0
        labels = {ch.label.strip().lower() for ch in self._raw_channels}
        for ch in self._raw_channels:
            if ch.label.strip().lower() == 'van rms' and ch.index in self._obs_data:
                vals = self._obs_data[ch.index]
                med = float(np.nanmedian(vals)) if len(vals) else 0.0
                if med > 1:
                    ref_v = med
                break
        # Split-phase meters have no C phase — the third current block is neutral
        is_split = 'ic rms' not in labels

        for rec in obs_recs:
            try:
                body = zlib.decompress(rec['raw'])
            except zlib.error:
                continue
            label = body[148:220].decode('ascii', errors='replace')
            if ' - Waveform - ' not in label:
                continue
            m = re.search(
                r'(\d{2})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d+)', label)
            if not m:
                continue
            mo, dy, yr, hh, mm, ss, frac = m.groups()
            try:
                cap_ts = datetime(2000 + int(yr), int(mo), int(dy),
                                  int(hh), int(mm), int(ss),
                                  int(float('0.' + frac) * 1e6))
            except ValueError:
                continue

            channels: List[Tuple[np.ndarray, np.ndarray]] = []
            for off in self._scan_entry_table(body):
                if off + 212 > len(body):
                    continue
                cnt = struct.unpack_from('<I', body, off + 208)[0]
                if not (64 <= cnt <= 2_000_000):
                    continue
                ts_end = off + 212 + cnt * 8
                if ts_end > len(body):
                    continue
                t = np.frombuffer(body[off + 212: ts_end], dtype='<f8')
                if not (np.all(np.isfinite(t)) and t[0] >= 0 and t[-1] < 600
                        and np.all(np.diff(t) >= 0)):
                    continue
                # data: u32 count (== cnt) within a small gap after ts array
                data = None
                for g in range(0, 68, 4):
                    if ts_end + g + 4 + cnt * 8 > len(body):
                        break
                    if struct.unpack_from('<I', body, ts_end + g)[0] == cnt:
                        data = np.frombuffer(
                            body[ts_end + g + 4: ts_end + g + 4 + cnt * 8],
                            dtype='<f8').copy()
                        break
                if data is None or not np.all(np.isfinite(data)):
                    continue
                channels.append((t, data))

            if not channels:
                continue

            # Classify by amplitude and table position: L-N voltage phases come
            # first, then (split-phase) a near-zero Vne block, then currents.
            # Median half-cycle RMS is the discriminator — robust to sags and
            # inrush spikes within the capture, unlike a simple peak test.
            t0 = channels[0][0]
            dt = float(np.median(np.diff(t0))) if len(t0) > 1 else 0.0
            w = max(int(round((1 / 60.0) / dt / 2)), 8) if dt > 0 else 32

            def _med_rms(x: np.ndarray) -> float:
                if len(x) < w * 2:
                    return float(np.sqrt(np.mean(x * x)))
                c = np.cumsum(np.concatenate(([0.0], x * x)))
                return float(np.median(np.sqrt((c[w:] - c[:-w]) / w)))

            voltages: Dict[str, np.ndarray] = {}
            currents: Dict[str, np.ndarray] = {}
            vne: Optional[np.ndarray] = None
            v_names = iter(('a', 'b') if is_split else ('a', 'b', 'c'))
            i_names = iter(('a', 'b', 'n') if is_split else ('a', 'b', 'c', 'n'))
            mode_v = True
            v_cap  = 2 if is_split else 3
            for _, data in channels:
                m = _med_rms(data)
                if mode_v and 0.4 * ref_v <= m <= 1.6 * ref_v and len(voltages) < v_cap:
                    voltages[next(v_names)] = data
                    continue
                if mode_v and voltages:
                    # first non-voltage block: split-phase exports place Vne here
                    mode_v = False
                    if m < 0.15 * ref_v and vne is None:
                        vne = data
                        continue
                try:
                    currents[next(i_names)] = data
                except StopIteration:
                    pass

            if not voltages:
                continue
            self._waveforms.append({
                "timestamp": cap_ts,
                "label":     label.split('\x00')[0].strip(),
                "t":         t0,
                "fs_hz":     (1.0 / dt) if dt > 0 else None,
                "voltages":  voltages,
                "vne":       vne,
                "currents":  currents,
            })

        if self._waveforms:
            self._waveforms.sort(key=lambda w: w["timestamp"])
            log.info(
                "ProntoAdapter waveforms: decoded %d point-on-wave captures "
                "(%s ch V / %s ch I typical, ~%.0f samples/cycle)",
                len(self._waveforms),
                len(self._waveforms[0]["voltages"]),
                len(self._waveforms[0]["currents"]),
                (self._waveforms[0]["fs_hz"] or 0) / 60.0,
            )

    # Interval-channel label → adaptive column name, used as identification
    # references for adaptive channels (matched by correlation, not position).
    _ADAP_REF_LABELS = {
        'van rms': 'van_v', 'vbn rms': 'vbn_v', 'vcn rms': 'vcn_v',
        'vne rms': 'vne_v',
        'ia rms': 'ia_a', 'ib rms': 'ib_a', 'ic rms': 'ic_a', 'in rms': 'in_a',
        'thd van': 'thd_van_pct', 'thd vbn': 'thd_vbn_pct', 'thd vcn': 'thd_vcn_pct',
        'real power': 'kw_w', 'reactive power': 'kvar_var',
        'apparent power': 'kva_va', 'power factor': 'adap_pf',
    }

    def _identify_adaptive_channels(
        self, parsed: List[Tuple[int, pd.Series]]
    ) -> Dict[str, pd.Series]:
        """Assign canonical names to adaptive channels by signature.

        Primary method: bin each adaptive series to the interval-average grid
        and correlate against the interval channels (Van RMS, Ia RMS, …) — the
        same physical quantity correlates ≈1 with its own interval average.
        Near-constant references (e.g. Vne at ~0.1 V) fall back to median
        matching; frequency/power-factor channels are identified by value
        signature; L-L voltages by their ratio to the identified L-N voltage.
        Unidentified channels are skipped."""
        by_ci = dict(parsed)
        assigned: Dict[str, pd.Series] = {}
        used_ci: Set[int] = set()

        if self._obs_ts is None or not self._obs_data or len(self._obs_ts) < 3:
            return {}
        ref_idx = pd.DatetimeIndex(self._obs_ts)
        td = pd.Series(ref_idx[1:] - ref_idx[:-1]).median()

        refs: Dict[str, pd.Series] = {}
        for ch in self._raw_channels:
            col = self._ADAP_REF_LABELS.get(ch.label.strip().lower())
            if col and ch.index in self._obs_data and col not in refs:
                vals = self._obs_data[ch.index]
                r = pd.Series(vals[:len(ref_idx)], index=ref_idx[:len(vals)]).dropna()
                if len(r) >= 20:
                    refs[col] = r.groupby(r.index.floor(td)).mean()

        binned = {ci: s.groupby(s.index.floor(td)).mean() for ci, s in parsed}

        # 1. Correlation scoring against interval references
        scores: List[Tuple[float, int, str]] = []
        for ci, _ in parsed:
            b = binned[ci]
            for col, rb in refs.items():
                al = pd.concat([b, rb], axis=1, join='inner').dropna()
                if len(al) < 20:
                    continue
                x, y = al.iloc[:, 0], al.iloc[:, 1]
                if x.std() == 0 or y.std() == 0:
                    continue
                corr = float(x.corr(y))
                mx, my = float(x.median()), float(y.median())
                ratio_ok = ((abs(my) < 1.0 and abs(mx) < 2.0)
                            or (my != 0 and 0.5 <= mx / my <= 2.0))
                if np.isfinite(corr) and corr >= 0.8 and ratio_ok:
                    scores.append((corr, ci, col))
        for corr, ci, col in sorted(scores, reverse=True):
            if ci in used_ci or col in assigned:
                continue
            assigned[col] = by_ci[ci]
            used_ci.add(ci)

        # 2. Median-ratio fallback for near-constant references (Vne, idle currents)
        for col, rb in refs.items():
            if col in assigned:
                continue
            my = float(rb.median())
            tol = max(abs(my) * 0.3, 0.5)
            cands = [(abs(float(binned[ci].median()) - my), ci)
                     for ci, _ in parsed
                     if ci not in used_ci and abs(float(binned[ci].median()) - my) <= tol]
            if cands:
                _, ci = min(cands)
                assigned[col] = by_ci[ci]
                used_ci.add(ci)

        # 3. Signature-only channels with no interval counterpart
        for ci, s in parsed:
            if ci in used_ci:
                continue
            med = float(s.median())
            if 'adap_freq' not in assigned and 55.0 <= med <= 65.0 and float(s.std()) < 1.0:
                assigned['adap_freq'] = s
                used_ci.add(ci)
                continue
            if ('adap_pf' not in assigned and 0.3 <= abs(med) <= 1.05
                    and float(s.abs().quantile(0.99)) <= 1.1):
                assigned['adap_pf'] = s
                used_ci.add(ci)

        # 4. L-L voltages: ~2× L-N (split-phase) or ~√3× (wye)
        ln = assigned.get('van_v')
        if ln is not None and float(ln.median()) > 0:
            ln_med = float(ln.median())
            ll_names = iter(('vab_v', 'vbc_v', 'vac_v'))
            for ci, s in parsed:
                if ci in used_ci:
                    continue
                if 1.65 <= float(s.median()) / ln_med <= 2.1:
                    try:
                        assigned[next(ll_names)] = s
                    except StopIteration:
                        break
                    used_ci.add(ci)

        n_un = len(parsed) - len(used_ci)
        log.info(
            "ProntoAdapter adaptive: identified %d of %d channels by signature%s",
            len(used_ci), len(parsed),
            f" ({n_un} unidentified, skipped)" if n_un else "",
        )
        return assigned

    @staticmethod
    def _scan_entry_table(body: bytes, lo: int = 150, hi: int = 600) -> List[int]:
        """Locate the per-channel entry table near the top of a decompressed
        observation body and return the channel block offsets, in table order.

        Entries are 28 bytes: u32 flag (=1), u32 absolute block offset, u32
        sub-header size, and a 16-byte type GUID.  The table start drifts
        between Pronto export versions, so scan for the longest run of valid
        entries (flag == 1, offsets strictly increasing and within the body)
        rather than trusting a fixed offset."""
        n = len(body)
        best: List[int] = []
        for start in range(lo, min(hi, n - 28), 2):
            offs: List[int] = []
            pos = start
            while pos + 28 <= n:
                flag, off = struct.unpack_from('<2I', body, pos)
                if flag == 1 and 0 < off < n and (not offs or off > offs[-1]):
                    offs.append(off)
                    pos += 28
                else:
                    break
            if len(offs) > len(best):
                best = offs
        return best if len(best) >= 3 else []

    def _load_dedup(self, body: bytes, off: int, n: int) -> np.ndarray:
        raw = self._read_f64(body, off)
        if raw is None:
            log.warning("ProntoAdapter: missing series at body offset %d", off)
            return np.full(n, np.nan)
        dedup = np.array(raw[0::2], dtype=float)
        if len(dedup) < n:
            return np.pad(dedup, (0, n - len(dedup)), constant_values=np.nan)
        return dedup[:n]

    def _parse_date(self) -> datetime:
        """Extract recording date from filename.  Pronto format: M-D-YYYY."""
        stem = self.filepath.stem
        m = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', stem)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(year, month, day, 0, 0, 0)
        log.warning(
            "ProntoAdapter: cannot parse date from filename %r; defaulting to 2000-01-01.",
            stem,
        )
        return datetime(2000, 1, 1)

    @staticmethod
    def _walk_records(data: bytes) -> List[Dict]:
        recs: List[Dict] = []
        pos = 0
        while pos + 48 <= len(data):
            tag  = struct.unpack_from('<I', data, pos + 16)[0]
            hdr  = struct.unpack_from('<I', data, pos + 32)[0]
            blen = struct.unpack_from('<I', data, pos + 36)[0]
            nxt  = struct.unpack_from('<I', data, pos + 40)[0]
            recs.append({'tag': tag, 'raw': data[pos + hdr: pos + hdr + blen]})
            if nxt == 0:
                break
            pos = nxt
        return recs

    @staticmethod
    def _read_f64(body: bytes, off: int) -> Optional[List[float]]:
        """Read a length-prefixed float64 array: u32 count + count × float64."""
        if off + 4 > len(body):
            return None
        count = struct.unpack_from('<I', body, off)[0]
        end = off + 4 + count * 8
        if count == 0 or count > 10_000 or end > len(body):
            return None
        return list(struct.unpack_from(f'<{count}d', body, off + 4))


class MockAdapter:
    """Generates realistic synthetic PQ data for testing without a real PQDIF file."""

    def __init__(self, duration_hours: float = 2.0, interval_sec: float = 1.0,
                 nominal: float = 120.0):
        self.nominal = nominal
        rng = np.random.default_rng(42)
        n = int(duration_hours * 3600 / interval_sec)
        t_start = np.datetime64("2024-01-15T08:00:00", "ns")
        self._ts = t_start + np.arange(n) * np.timedelta64(int(interval_sec * 1e9), "ns")

        # Simulate realistic 3-phase voltage with a sag around t=3000s
        base_v = nominal * (1 + 0.005 * np.sin(2 * np.pi * np.arange(n) / 3600))
        sag_mask = (np.arange(n) > 3000) & (np.arange(n) < 3060)
        va = base_v + rng.normal(0, 0.3, n)
        vb = base_v * 1.002 + rng.normal(0, 0.3, n)
        vc = base_v * 0.998 + rng.normal(0, 0.3, n)
        va[sag_mask] *= 0.88   # 12 % sag event
        # Inject an overvoltage near the end
        swell_mask = (np.arange(n) > 6000) & (np.arange(n) < 6020)
        va[swell_mask] *= 1.08

        ia = 50.0 + 5 * np.sin(2 * np.pi * np.arange(n) / 900) + rng.normal(0, 0.5, n)
        ib = 51.0 + 5 * np.sin(2 * np.pi * np.arange(n) / 900 + 0.1) + rng.normal(0, 0.5, n)
        ic = 49.5 + 5 * np.sin(2 * np.pi * np.arange(n) / 900 - 0.1) + rng.normal(0, 0.5, n)

        kw   = 18.0 + 2 * np.sin(2 * np.pi * np.arange(n) / 3600) + rng.normal(0, 0.2, n)
        kvar = 6.0  + 0.5 * np.sin(2 * np.pi * np.arange(n) / 3600) + rng.normal(0, 0.1, n)
        pf   = np.clip(kw / np.sqrt(kw**2 + kvar**2) + rng.normal(0, 0.005, n), 0.5, 1.0)

        # THD with a few exceedance periods
        thd_v = np.clip(3.0 + rng.normal(0, 0.8, n), 0.5, 20)
        thd_v[4000:4200] += 7.0  # exceedance window
        thd_i = np.clip(4.0 + rng.normal(0, 1.0, n), 0.5, 20)

        # Synthetic per-order harmonic currents — phases A/B/C (Amps absolute)
        # Representative of a mixed VFD + SMPS site
        h3a  = np.clip(3.5 + rng.normal(0, 0.3, n), 0.5, 10)
        h5a  = np.clip(5.0 + rng.normal(0, 0.5, n), 0.5, 15)
        h7a  = np.clip(2.0 + rng.normal(0, 0.3, n), 0.1, 8)
        h9a  = np.clip(0.8 + rng.normal(0, 0.1, n), 0.0, 4)
        h11a = np.clip(1.2 + rng.normal(0, 0.2, n), 0.1, 5)
        h13a = np.clip(0.9 + rng.normal(0, 0.1, n), 0.0, 4)
        h3b  = np.clip(h3a  * 0.97 + rng.normal(0, 0.15, n), 0.3, 10)
        h5b  = np.clip(h5a  * 0.98 + rng.normal(0, 0.25, n), 0.3, 15)
        h7b  = np.clip(h7a  * 0.96 + rng.normal(0, 0.15, n), 0.1, 8)
        h11b = np.clip(h11a * 0.97 + rng.normal(0, 0.10, n), 0.1, 5)
        h13b = np.clip(h13a * 0.98 + rng.normal(0, 0.08, n), 0.0, 4)
        h3c  = np.clip(h3a  * 1.02 + rng.normal(0, 0.15, n), 0.3, 10)
        h5c  = np.clip(h5a  * 1.01 + rng.normal(0, 0.25, n), 0.3, 15)
        h7c  = np.clip(h7a  * 1.03 + rng.normal(0, 0.15, n), 0.1, 8)
        h11c = np.clip(h11a * 1.01 + rng.normal(0, 0.10, n), 0.1, 5)
        h13c = np.clip(h13a * 1.02 + rng.normal(0, 0.08, n), 0.0, 4)
        # Neutral triplens accumulate from all three phases (≈ 2.8× phase H3)
        h3n  = np.clip(h3a * 2.8 + rng.normal(0, 0.4, n), 0.5, 30)
        h9n  = np.clip(h9a * 2.5 + rng.normal(0, 0.1, n), 0.0, 12)
        # Non-triplens in the neutral should be near zero for balanced 3-phase
        h5n  = np.clip(rng.normal(0, 0.15, n), 0.0, 1.0)
        h7n  = np.clip(rng.normal(0, 0.10, n), 0.0, 0.8)
        h11n = np.clip(rng.normal(0, 0.10, n), 0.0, 0.6)
        h13n = np.clip(rng.normal(0, 0.08, n), 0.0, 0.5)
        # Voltage harmonics — customer injection into stiff source (kZ ≈ 0.03 Ω/order)
        # V_h correlates with I_h → high Pearson r expected across orders
        kz = 0.03
        h3va  = np.clip(h3a  * 3  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h5va  = np.clip(h5a  * 5  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h7va  = np.clip(h7a  * 7  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h11va = np.clip(h11a * 11 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h13va = np.clip(h13a * 13 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h3vb  = np.clip(h3b  * 3  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h5vb  = np.clip(h5b  * 5  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h7vb  = np.clip(h7b  * 7  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h11vb = np.clip(h11b * 11 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h13vb = np.clip(h13b * 13 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h3vc  = np.clip(h3c  * 3  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h5vc  = np.clip(h5c  * 5  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h7vc  = np.clip(h7c  * 7  * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h11vc = np.clip(h11c * 11 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)
        h13vc = np.clip(h13c * 13 * kz + rng.normal(0, 0.01, n), 0.0, 5.0)

        # Synthetic adaptive DataFrame — cycle-level (~17 ms), simulates event bursts
        n_adap   = 300
        t_adap   = self._ts[0] + np.arange(n_adap) * np.timedelta64(17_000_000, "ns")
        adap_van = np.full(n_adap, nominal * 1.002)
        adap_vbn = np.full(n_adap, nominal * 0.999)
        adap_vcn = np.full(n_adap, nominal * 0.998)
        adap_van[50:80] *= 0.86          # 14 % sag — within-interval event
        adap_pst = np.full(n_adap, 0.6)
        adap_pst[100:130] = 1.4          # PST exceedance burst
        adap_plt = np.full(n_adap, 0.3)
        adap_ia  = np.full(n_adap, 50.0)
        adap_ia[150:] = 83.0             # current step at row 150
        adap_ib  = np.full(n_adap, 51.0)
        adap_ic  = np.full(n_adap, 49.5)
        self.adaptive_df: Optional[pd.DataFrame] = pd.DataFrame(
            {
                "van_v":    adap_van,
                "vbn_v":    adap_vbn,
                "vcn_v":    adap_vcn,
                "ia_a":     adap_ia,
                "ib_a":     adap_ib,
                "ic_a":     adap_ic,
                "adap_pst": adap_pst,
                "adap_plt": adap_plt,
            },
            index=pd.DatetimeIndex(t_adap),
        )

        self._channels = {
            "voltage_a": va, "voltage_b": vb, "voltage_c": vc,
            "current_a": ia, "current_b": ib, "current_c": ic,
            "power_real": kw, "power_reactive": kvar, "power_factor": pf,
            "thd_voltage_a": thd_v, "thd_voltage_b": thd_v * 0.95, "thd_voltage_c": thd_v * 1.02,
            "thd_current_a": thd_i, "thd_current_b": thd_i * 0.98, "thd_current_c": thd_i * 1.01,
            "h3_current_a": h3a,  "h5_current_a": h5a,  "h7_current_a": h7a,
            "h9_current_a": h9a,  "h11_current_a": h11a, "h13_current_a": h13a,
            "h3_current_b": h3b,  "h5_current_b": h5b,  "h7_current_b": h7b,
            "h11_current_b": h11b, "h13_current_b": h13b,
            "h3_current_c": h3c,  "h5_current_c": h5c,  "h7_current_c": h7c,
            "h11_current_c": h11c, "h13_current_c": h13c,
            "h3_current_neutral": h3n, "h5_current_neutral": h5n, "h7_current_neutral": h7n,
            "h9_current_neutral": h9n, "h11_current_neutral": h11n, "h13_current_neutral": h13n,
            "h3_voltage_a": h3va,  "h5_voltage_a": h5va,  "h7_voltage_a": h7va,
            "h11_voltage_a": h11va, "h13_voltage_a": h13va,
            "h3_voltage_b": h3vb,  "h5_voltage_b": h5vb,  "h7_voltage_b": h7vb,
            "h11_voltage_b": h11vb, "h13_voltage_b": h13vb,
            "h3_voltage_c": h3vc,  "h5_voltage_c": h5vc,  "h7_voltage_c": h7vc,
            "h11_voltage_c": h11vc, "h13_voltage_c": h13vc,
        }
        # Build synthetic RawChannelInfo objects for compatibility
        self._raw_channels = [
            RawChannelInfo(i, name, "", "", "", "") for i, name in enumerate(self._channels)
        ]

    def list_channels(self) -> List[RawChannelInfo]:
        return self._raw_channels

    def iter_observations(self, wanted_indices):
        yield self._ts, {i: arr for i, (name, arr) in enumerate(self._channels.items())}


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA EXTRACTION & ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PQDataset:
    """Unified container for all PQ data from a single recording.

    Attributes
    ----------
    df          : 5-minute interval averages (obs[23]) with obs[24] max/min columns
                  injected as ``{col}_peak`` / ``{col}_min`` suffixes.
    adaptive_df : Cycle-level event-triggered data (obs[25]).  Sparse DataFrame on
                  a union DatetimeIndex; None when the obs record is absent.
    meta        : Recording metadata — topology, interval_minutes, start/end times.
    """
    df:          pd.DataFrame
    adaptive_df: Optional[pd.DataFrame]
    meta:        dict
    waveforms:   List[dict] = field(default_factory=list)

    @property
    def duration_hours(self) -> float:
        if len(self.df) > 1:
            return (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
        return 0.0

    @property
    def has_maxmin(self) -> bool:
        return any(c.endswith("_peak") for c in self.df.columns)

    @property
    def has_adaptive(self) -> bool:
        return self.adaptive_df is not None and len(self.adaptive_df) > 0

    @property
    def has_waveforms(self) -> bool:
        return bool(self.waveforms)

    def catalog(self) -> str:
        """Human-readable inventory of every data source and channel group."""
        n   = len(self.df)
        dt  = self.meta.get("interval_minutes", 5)
        dt_str = f"{dt:.0f}-min" if dt >= 0.5 else f"{round(dt * 60)}-sec"
        lines = [
            f"PQDataset — {self.duration_hours:.1f} h  |  "
            f"{n:,} intervals ({dt_str} avg)  |  "
            f"topology: {self.meta.get('topology', 'unknown')}"
        ]

        avg_cols = [c for c in self.df.columns
                    if not c.endswith("_peak") and not c.endswith("_min")]
        pk_cols  = [c for c in self.df.columns if c.endswith("_peak")]
        mn_cols  = [c for c in self.df.columns if c.endswith("_min")]

        _GROUPS: List[Tuple[str, object]] = [
            ("voltage",    lambda c: c.startswith("voltage")),
            ("current",    lambda c: c.startswith("current") and not re.match(r"h\d", c)),
            ("power",      lambda c: c.startswith("power")),
            ("thd",        lambda c: c.startswith("thd")),
            ("I-harm",     lambda c: bool(re.match(r"h\d+_current_", c))),
            ("V-harm",     lambda c: bool(re.match(r"h\d+_voltage_", c))),
            ("flicker",    lambda c: c.startswith("flicker")),
            ("kfactor",    lambda c: c.startswith("kfactor")),
        ]
        group_counts: List[str] = []
        accounted: Set[str] = set()
        for label, pred in _GROUPS:
            matches = [c for c in avg_cols if pred(c) and c not in accounted]  # type: ignore[operator]
            if matches:
                group_counts.append(f"{label}({len(matches)})")
                accounted.update(matches)
        other = [c for c in avg_cols if c not in accounted]
        if other:
            group_counts.append(f"other({len(other)})")

        lines.append(
            f"  Interval avg               : {len(avg_cols):3d} ch  "
            f"[{', '.join(group_counts)}]"
        )
        if pk_cols or mn_cols:
            sample = ", ".join(pk_cols[:3]) + (" …" if len(pk_cols) > 3 else "")
            lines.append(
                f"  Interval max/min           : {len(pk_cols):3d} peak / "
                f"{len(mn_cols):3d} min  [{sample}]"
            )
        else:
            lines.append("  Interval max/min           : not present")

        if self.has_adaptive:
            adf = self.adaptive_df
            assert adf is not None
            adur = (
                (adf.index[-1] - adf.index[0]).total_seconds() / 3600
                if len(adf) > 1 else 0.0
            )
            lines.append(
                f"  Variable-rate (adaptive)   : {len(adf):,} rows  "
                f"[{len(adf.columns)} ch, cycle-level, {adur:.1f} h span]"
            )
        else:
            lines.append("  Variable-rate (adaptive)   : not present")

        if self.has_waveforms:
            wf = self.waveforms
            fs = wf[0].get("fs_hz") or 0
            lines.append(
                f"  Waveform captures          : {len(wf)} point-on-wave records  "
                f"[{len(wf[0]['voltages'])}V/{len(wf[0]['currents'])}I ch, "
                f"{fs/1000:.1f} kHz]"
            )
        else:
            lines.append("  Waveform captures          : not present")

        return "\n".join(lines)


def extract_dataset(
    adapter,
    mapper: "ChannelMapper",
    resample: Optional[str] = None,
) -> PQDataset:
    """Build a PQDataset from an adapter.

    Wraps extract_dataframe() and folds in obs[24] max-min columns and the
    adaptive DataFrame so callers work with a single unified object instead of
    three separate data sources.
    """
    df = extract_dataframe(adapter, mapper, resample=resample)

    # Fold obs[24] interval max/min into the main DataFrame as _peak / _min columns
    for _col, _arr in getattr(adapter, "interval_peaks", {}).items():
        if _col in df.columns:
            df[f"{_col}_peak"] = pd.Series(
                _arr[: len(df)], index=df.index[: len(_arr)]
            )
    for _col, _arr in getattr(adapter, "interval_mins", {}).items():
        if _col in df.columns:
            df[f"{_col}_min"] = pd.Series(
                _arr[: len(df)], index=df.index[: len(_arr)]
            )

    adaptive_df: Optional[pd.DataFrame] = getattr(adapter, "adaptive_df", None)

    # Infer interval duration from index spacing
    if len(df.index) > 1:
        median_ns = float(np.median(np.diff(df.index.view("int64"))))
        interval_minutes = round(median_ns / 60e9, 1)
    else:
        interval_minutes = 5.0

    # Infer topology from which current phases are present
    if "current_c" in df.columns:
        topology = "3-phase"
    elif "current_b" in df.columns:
        topology = "split-phase"
    else:
        topology = "single-phase"

    meta: dict = {
        "topology":         topology,
        "interval_minutes": interval_minutes,
        # Non-zero means the file was incomplete; a compliance report built from
        # it has to say so rather than read as though the record were whole.
        "data_quality":     getattr(adapter, "data_quality", {}) or {},
        "start_time":       df.index[0].isoformat() if len(df) else None,
        "end_time":         df.index[-1].isoformat() if len(df) else None,
    }

    ds = PQDataset(df=df, adaptive_df=adaptive_df, meta=meta,
                   waveforms=getattr(adapter, "waveforms", []) or [])
    log.info("\n%s", ds.catalog())
    return ds


def extract_dataframe(
    adapter,
    mapper: ChannelMapper,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """Pull filtered channels from the adapter into a time-aligned DataFrame.

    Parameters
    ----------
    adapter   : PQDIFAdapter or MockAdapter
    mapper    : ChannelMapper
    resample  : pandas offset string, e.g. '1s', '1min', '10min', or None

    Returns
    -------
    pd.DataFrame  — index is DatetimeTZIndex (UTC), columns are canonical names.
    Memory note: observations are processed one at a time; only matched channels
    are accumulated, so 500-channel files with large waveform data stay lean.
    """
    raw_channels = adapter.list_channels()
    log.info("Resolving %d device channels to canonical names …", len(raw_channels))
    resolved: Dict[str, RawChannelInfo] = mapper.resolve(raw_channels)

    if not resolved:
        raise ValueError(
            "No channels matched. Run with --list-channels to inspect channel names, "
            "then update _NAME_PATTERNS or _TAG_MAP in the script."
        )

    log.info("Matched channels: %s", sorted(resolved.keys()))
    wanted_indices: Set[int] = {ch.index for ch in resolved.values()}
    # reverse map: channel_index → canonical_name
    idx_to_name = {ch.index: name for name, ch in resolved.items()}

    # Collect all observations into lists for efficient concatenation.
    all_timestamps: List[np.ndarray] = []
    all_values: Dict[str, List[np.ndarray]] = {name: [] for name in resolved}

    for ts_arr, obs_data in adapter.iter_observations(wanted_indices):
        all_timestamps.append(ts_arr)
        for idx, values in obs_data.items():
            name = idx_to_name.get(idx)
            if name:
                # Ensure length alignment — pad/trim to match timestamps if needed
                n = len(ts_arr)
                if len(values) < n:
                    values = np.pad(values, (0, n - len(values)), constant_values=np.nan)
                elif len(values) > n:
                    values = values[:n]
                all_values[name].append(values)

    if not all_timestamps:
        raise ValueError("No observation data found in file.")

    ts_concat = np.concatenate(all_timestamps)
    # Build DataFrame — columns only for channels that actually had data
    columns = {}
    for name, arrays in all_values.items():
        if arrays:
            arr = np.concatenate(arrays)
            if len(arr) == len(ts_concat):
                columns[name] = arr

    df = pd.DataFrame(columns, index=pd.DatetimeIndex(ts_concat, tz="UTC"))
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    if resample:
        log.info("Resampling to %s …", resample)
        df = df.resample(resample).mean()

    log.info(
        "DataFrame: %d rows × %d columns  (%s → %s)",
        len(df), len(df.columns),
        df.index[0] if len(df) else "–",
        df.index[-1] if len(df) else "–",
    )

    # Channel labels can resolve correctly (so they show up as "matched" above)
    # while the underlying binary data pointer for every single channel is
    # broken for this file -- e.g. a per-file offset the adapter computes once
    # and reuses for every channel lookup. That produces a DataFrame with the
    # right shape and column names but zero real data anywhere, which would
    # otherwise sail through as a "successful" analysis with N/A everywhere
    # and no indication that nothing was actually read. Fail loudly instead.
    if len(df.columns) > 0:
        empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
        frac_empty = len(empty_cols) / len(df.columns)
        if frac_empty >= 0.9:
            raise ValueError(
                f"{len(empty_cols)}/{len(df.columns)} matched channels have zero valid "
                "samples -- channel labels resolved correctly but no underlying data was "
                "read for any of them. This points at a per-file binary offset the adapter "
                "computed incorrectly for this specific file's export format, not a missing "
                "or corrupt individual channel. Re-export this file if possible, or report it "
                "as a new Pronto format variant needing adapter support."
            )

    return df

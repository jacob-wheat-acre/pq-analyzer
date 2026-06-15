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

from pq_constants import _H519_ORDERS

# pqdifpy — the primary PQDIF parsing library.
# Install with: pip install pqdifpy
# If you are using a different library (openhistorian, custom parser, etc.)
# replace only the PQDIFAdapter class below; everything else is unaffected.
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
    # Meter-measured transformer K-factor and IEC flicker severity
    "kfactor_meter",
    "flicker_pst",
    "flicker_plt",
]

# ── PQDIF tag dictionaries ────────────────────────────────────────────────────
# PQDIF encodes channel identity with three structured tags:
#   quantity_type     : what physical quantity is measured (Voltage, Current …)
#   quantity_measured : how it is measured (RMS, Average, THD …)
#   phase             : which conductor (AN, BN, CN, TOTAL …)
#
# These are more reliable than device-assigned text labels.
# The values below are the string representations pqdifpy exposes; your version
# may use different casing or an enum — see _normalise_tag() below.

_TAG_MAP: Dict[str, Dict[str, Set[str]]] = {
    #  canonical        quantity_type          quantity_measured    phase
    "voltage_a":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"an", "a", "phase_a"}},
    "voltage_b":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"bn", "b", "phase_b"}},
    "voltage_c":      {"qt": {"voltage"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"cn", "c", "phase_c"}},
    "current_a":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"an", "a", "phase_a"}},
    "current_b":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"bn", "b", "phase_b"}},
    "current_c":      {"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"cn", "c", "phase_c"}},
    "power_real":     {"qt": {"power", "watts"}, "qm": {"real", "watts", "active", "p"}, "ph": {"total", "three_phase", "net", "aggregate", ""}},
    "power_reactive": {"qt": {"power"},          "qm": {"reactive", "var", "q"},          "ph": {"total", "three_phase", "net", "aggregate", ""}},
    "power_factor":   {"qt": {"power", "powerfactor"}, "qm": {"powerfactor", "pf", "factor"}, "ph": {"total", "three_phase", "net", "aggregate", ""}},
    "thd_voltage_a":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"an", "a", "phase_a"}},
    "thd_voltage_b":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"bn", "b", "phase_b"}},
    "thd_voltage_c":  {"qt": {"voltage", "voltageharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"cn", "c", "phase_c"}},
    "thd_current_a":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"an", "a", "phase_a"}},
    "thd_current_b":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"bn", "b", "phase_b"}},
    "thd_current_c":  {"qt": {"current", "currentharmonics", "harmonics"}, "qm": {"thd", "totalharmdist", "thdpercent"}, "ph": {"cn", "c", "phase_c"}},
    "current_neutral":{"qt": {"current"},       "qm": {"rms", "average", "rmsvalue"},  "ph": {"neutral", "n", "in", "i4", "phase_n"}},
    # Individual harmonic currents — one entry per order × phase
    **{f"h{h}_current_a": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"an","a","phase_a"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_b": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"bn","b","phase_b"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_c": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"cn","c","phase_c"}}
       for h in _H519_ORDERS},
    **{f"h{h}_current_neutral": {"qt": {"currentharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"neutral","n","in","i4","phase_n"}}
       for h in (3, 5, 7, 9, 11, 13)},
    # Individual harmonic voltages
    **{f"h{h}_voltage_a": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"an","a","phase_a"}}
       for h in (3, 5, 7, 11, 13)},
    **{f"h{h}_voltage_b": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"bn","b","phase_b"}}
       for h in (3, 5, 7, 11, 13)},
    **{f"h{h}_voltage_c": {"qt": {"voltageharmonics"}, "qm": {f"h{h}", f"harmonic{h}"}, "ph": {"cn","c","phase_c"}}
       for h in (3, 5, 7, 11, 13)},
    # Transformer K-factor (measured by meter, includes all harmonic orders)
    "kfactor_meter": {"qt": {"kfactor"}, "qm": {"kfactor"},   "ph": {"total", "net", "aggregate", ""}},
    # Flicker severity indices
    "flicker_pst":   {"qt": {"flicker"}, "qm": {"pst"},       "ph": {"an", "a", "phase_a"}},
    "flicker_plt":   {"qt": {"flicker"}, "qm": {"plt"},       "ph": {"an", "a", "phase_a"}},
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
    _GUID_NAMES = {
        "67f6af80-f753-11cf-9d89-0080c72e70a3": "voltage",
        "67f6af81-f753-11cf-9d89-0080c72e70a3": "current",
        "67f6af8b-f753-11cf-9d89-0080c72e70a3": "power",
        "67f6af85-f753-11cf-9d89-0080c72e70a3": "energy",
        # Add more GUIDs here if needed — use --list-channels to find them
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
    Direct binary reader for Pronto PQDIF files (Xcel Energy metering system).

    Drop-in replacement for PQDIFAdapter for .pqd files created by Xcel Energy
    Pronto power-quality meters.  Reads the non-standard IEEE 1159.3 variant
    binary format without any external PQDIF library.

    Confirmed structure (reverse-engineered from real Xcel Energy files):
      - 6 physical records: Container, DataSource, Unknown, 3 Observation
      - Record bodies: zlib-compressed (magic bytes 0x78 0xDA)
      - obs[0]: 468 harmonic/power-quality channels (DataSource index 50+)
      - obs[1]: 11 five-minute RMS channels (DataSource index 39-49)
      - obs[2]: waveform data (not extracted here)

    Each 5-minute interval is stored as a near-identical pair of values;
    this adapter deduplicates by taking every other sample (even indices).

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
    _ADAP_ENTRY_START = 216   # entry table offset in decompressed adaptive body
    _ADAP_TS_REL      = 236   # ts_count u32 at ch_abs+236; ts array at ch_abs+240
    # Voltage AC (C=1–7)
    _ADAP_CH_VAN  =  0   # Van RMS L-N (V)
    _ADAP_CH_VBN  =  1   # Vbn RMS L-N (V)
    _ADAP_CH_VCN  =  2   # Vcn RMS L-N (V)
    _ADAP_CH_VNE  =  3   # Vne neutral-to-earth RMS (V) — typically 0–0.1 V
    _ADAP_CH_VAB  =  4   # Vab L-L (V)
    _ADAP_CH_VBC  =  5   # Vbc L-L (V)
    _ADAP_CH_VAC  =  6   # Vac L-L (V)
    # Current AC (C=8–11) — event-sampled; includes pre- and post-step instants
    _ADAP_CH_IA   =  7   # Ia RMS (A)
    _ADAP_CH_IB   =  8   # Ib RMS (A)
    _ADAP_CH_IC   =  9   # Ic RMS (A)
    _ADAP_CH_IN   = 10   # In neutral RMS (A)
    # Unbalance (C=25–28)
    _ADAP_CH_VUNBAL  = 11   # 3φ voltage unbalance % (NEMA method)
    _ADAP_CH_NPS_PPS = 12   # 3φ voltage NPS/PPS ratio %
    _ADAP_CH_NPS_ANG = 13   # 3φ voltage NPS-PPS phase angle (degrees)
    _ADAP_CH_IUNBAL  = 14   # 3φ current unbalance % (NEMA method)
    # Power (C=20–24)
    _ADAP_CH_KW   = 15   # 3φ 4-wire real power (W)
    _ADAP_CH_KVAR = 16   # 3φ 4-wire reactive power (VAr)
    _ADAP_CH_KVARF= 17   # 3φ 4-wire reactive power fundamental (VAr)
    _ADAP_CH_KVA  = 18   # 3φ 4-wire apparent power (VA)
    _ADAP_CH_PF   = 19   # 3φ power factor
    # Frequency (C=29)
    _ADAP_CH_FREQ = 20   # AC frequency (Hz)
    # Harmonic Group THD (C=12–19; entry 24 = THD_Vne absent in this file)
    _ADAP_CH_THD_VAN = 21   # %THD Van
    _ADAP_CH_THD_VBN = 22   # %THD Vbn
    _ADAP_CH_THD_VCN = 23   # %THD Vcn
    # entry 24 = THD_Vne (C=15) — absent (gap in entry table)
    _ADAP_CH_THD_IA  = 25   # THD Ia in absolute Aac (not %; divide by Ia_RMS for %)
    _ADAP_CH_THD_IB  = 26   # THD Ib in absolute Aac
    _ADAP_CH_THD_IC  = 27   # THD Ic in absolute Aac
    # entry 28: unknown (~192 A, n=19346; possibly Ia+Ib+Ic vector sum or peak)

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._raw_channels: List[RawChannelInfo] = []
        self._obs_ts: Optional[np.ndarray] = None
        self._obs_data: Dict[int, np.ndarray] = {}
        self._adaptive_df: Optional[pd.DataFrame] = None
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

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self):
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

        obs_top = self._pqdif_elements(obs_body, 0)
        ci_off: Optional[int] = None
        for e in obs_top:
            if e['guid4'] == 0x3D786F91:
                ci_off = e['off']
                break
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
          - 'Harm 1 of Ia' (obs_ci=109 on Watkins) is the I1 fundamental in Amps,
            consistent with measured apparent power (VA/3/Vln).
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
                "Is this a Pronto PQDIF file?"
            )

        base_date = self._parse_v2_date(obs_recs)

        # ── Dynamic entry_start: read label_length from bytes 144-147 ────────
        label_length = struct.unpack_from('<I', interval_body, 144)[0]
        if not (1 <= label_length <= 512):
            raise ValueError(
                f"ProntoAdapter v2: unexpected label_length {label_length} in "
                "'Interval (avg)' obs body — unsupported Pronto PQDIF format."
            )
        entry_start = 148 + ((label_length + 3) & ~3) + 28

        # ── Dynamic DATA_REL ────────────────────────────────────────────────
        pos0 = entry_start + self._V2_BODY_OFF_REL
        if pos0 + 4 > len(interval_body):
            raise ValueError(
                f"ProntoAdapter v2: entry_start={entry_start} + body_off="
                f"{self._V2_BODY_OFF_REL} exceeds body length {len(interval_body)}."
            )
        ch0_abs = struct.unpack_from('<I', interval_body, pos0)[0]
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

        def read_v2(ci: int) -> np.ndarray:
            pos = entry_start + ci * self._V2_ENTRY_SIZE + self._V2_BODY_OFF_REL
            ch_abs = struct.unpack_from('<I', interval_body, pos)[0]
            data_abs = ch_abs + data_rel
            count = struct.unpack_from('<I', interval_body, data_abs)[0]
            if count == 0 or count > 15_000:
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
        self._load_v2_maxmin(obs_recs, n)

    def _load_v2_maxmin(self, obs_recs: List[Dict], n: int) -> None:
        """Parse obs[24] 'Interval (max-min)' record into interval_peaks / interval_mins.

        The maxmin record uses the same PQDIF framing as the avg record but has a longer
        label string, which shifts the entry-table start and the intra-block header by a
        few bytes.  Each channel block also contains three separate data blobs (max, min,
        and a third section — probably per-interval average) rather than one.

        Channel layout confirmed from binary inspection of a real Watkins .pqd file:
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
        # header block, then the channel entry table.
        label_length = struct.unpack_from('<I', maxmin_body, 144)[0]
        entry_start  = 148 + ((label_length + 3) & ~3) + 28

        pos0    = entry_start + self._V2_BODY_OFF_REL
        ch0_abs = struct.unpack_from('<I', maxmin_body, pos0)[0]

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
            pos = entry_start + ci * self._V2_ENTRY_SIZE + self._V2_BODY_OFF_REL
            if pos + 4 > len(maxmin_body):
                return None
            ch_abs  = struct.unpack_from('<I', maxmin_body, pos)[0]
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

        es     = self._ADAP_ENTRY_START
        ts_rel = self._ADAP_TS_REL
        entry_sz = self._V2_ENTRY_SIZE
        body_off = self._V2_BODY_OFF_REL

        def read_adap_ch(ci: int):
            pos = es + ci * entry_sz + body_off
            if pos + 4 > len(adap_body):
                return None, None
            ch_abs = struct.unpack_from('<I', adap_body, pos)[0]
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

        def ch_series(ci: int, col: str) -> Optional[pd.Series]:
            ts, vals = read_adap_ch(ci)
            if ts is None or len(ts) == 0:
                return None
            abs_ns = (base_ns + (ts * 1e9).astype('int64')).astype('datetime64[ns]')
            return pd.Series(vals, index=pd.DatetimeIndex(abs_ns), name=col)

        wanted = {
            self._ADAP_CH_VAN:     'van_v',
            self._ADAP_CH_VBN:     'vbn_v',
            self._ADAP_CH_VCN:     'vcn_v',
            self._ADAP_CH_VNE:     'vne_v',
            self._ADAP_CH_VAB:     'vab_v',
            self._ADAP_CH_VBC:     'vbc_v',
            self._ADAP_CH_VAC:     'vac_v',
            self._ADAP_CH_IA:      'ia_a',
            self._ADAP_CH_IB:      'ib_a',
            self._ADAP_CH_IC:      'ic_a',
            self._ADAP_CH_IN:      'in_a',
            self._ADAP_CH_VUNBAL:  'v_unbal_pct',
            self._ADAP_CH_NPS_PPS: 'nps_pps_pct',
            self._ADAP_CH_NPS_ANG: 'nps_ang_deg',
            self._ADAP_CH_IUNBAL:  'i_unbal_pct',
            self._ADAP_CH_KW:      'kw_w',
            self._ADAP_CH_KVAR:    'kvar_var',
            self._ADAP_CH_KVARF:   'kvarf_var',
            self._ADAP_CH_KVA:     'kva_va',
            self._ADAP_CH_PF:      'adap_pf',
            self._ADAP_CH_FREQ:    'adap_freq',
            self._ADAP_CH_THD_VAN: 'thd_van_pct',
            self._ADAP_CH_THD_VBN: 'thd_vbn_pct',
            self._ADAP_CH_THD_VCN: 'thd_vcn_pct',
            self._ADAP_CH_THD_IA:  'thd_ia_aac',
            self._ADAP_CH_THD_IB:  'thd_ib_aac',
            self._ADAP_CH_THD_IC:  'thd_ic_aac',
        }

        series_list = [s for ci, col in wanted.items()
                       if (s := ch_series(ci, col)) is not None]
        if not series_list:
            return

        df = pd.concat(series_list, axis=1).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        self._adaptive_df = df

        ts_span_h = (df.index[-1] - df.index[0]).total_seconds() / 3600
        log.info(
            "ProntoAdapter adaptive: %d variable-rate samples, %.1f h span, %d channels",
            len(df), ts_span_h, len(series_list),
        )

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
            f"  Interval avg     (obs[23]): {len(avg_cols):3d} ch  "
            f"[{', '.join(group_counts)}]"
        )
        if pk_cols or mn_cols:
            sample = ", ".join(pk_cols[:3]) + (" …" if len(pk_cols) > 3 else "")
            lines.append(
                f"  Interval max/min (obs[24]): {len(pk_cols):3d} peak / "
                f"{len(mn_cols):3d} min  [{sample}]"
            )
        else:
            lines.append("  Interval max/min (obs[24]): not present")

        if self.has_adaptive:
            adf = self.adaptive_df
            assert adf is not None
            adur = (
                (adf.index[-1] - adf.index[0]).total_seconds() / 3600
                if len(adf) > 1 else 0.0
            )
            lines.append(
                f"  Adaptive events  (obs[25]): {len(adf):,} rows  "
                f"[{len(adf.columns)} ch, cycle-level, {adur:.1f} h span]"
            )
        else:
            lines.append("  Adaptive events  (obs[25]): not present")

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
        "start_time":       df.index[0].isoformat() if len(df) else None,
        "end_time":         df.index[-1].isoformat() if len(df) else None,
    }

    ds = PQDataset(df=df, adaptive_df=adaptive_df, meta=meta)
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
    return df

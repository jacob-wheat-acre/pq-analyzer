#!/usr/bin/env python3
"""
make_test_pqd.py — Synthetic Pronto PQDIF test-file generator.

Produces one .pqd file per PSCo customer class, each with injected violations:

  test_residential.pqd        (r)   split-phase 120/240 V
    - Voltage sag: L1 drops to 108 V (−10 %) for 15 intervals
    - Open-neutral signature: anti-correlated legs, L1+L2 sum swings ±12 V
    - Power factor: 0.82 (below 0.90 limit)
    - Flicker PST: 1.3 during one window

  test_commercial_small.pqd   (c)   3-phase 120/208 V
    - Voltage THD: 9.5 % on L1 during peak hours (> 8 % limit)
    - Power factor: 0.78 (below 0.90 limit)
    - Current imbalance: L1 = 45 A, L2 = 20 A, L3 = 10 A → 80 % imbalance

  test_commercial_large.pqd   (sg)  3-phase 277/480 V
    - Voltage imbalance: L1 = 290 V, L2 = 268 V, L3 = 280 V → 4.0 % (> 3 %)
    - Current TDD: H5 = 8 A, H7 = 6 A at 100 A fund. → TDD ≈ 10.7 % (> 5 %)
    - Per-order H5 = 8 % → exceeds IEEE 519-2022 individual limit

  test_commercial_primary.pqd (pg)  3-phase 2400 V (4160 Y primary)
    - Voltage swell: 2640 V (+10 %) for 30 intervals
    - Voltage sag:   2160 V (−10 %) for 20 intervals
    - Voltage THD:   9 % (> 8 % limit)

Binary format: IEEE Std 1159.3-2019 (PQDIF), written by the serializer below.

These fixtures are deliberately faithful to a real Pronto export, because the
reader they exercise (pqdif.py + ProntoAdapter._load_spec) resolves everything
through the standard's structure.  In particular each file reproduces:

  * a container record declaring version 1.5 and record-level zlib compression,
    followed by a data source record and several observation records, chained by
    the absolute links in their 64-byte headers;
  * channel identity carried in the series definitions -- quantity measured,
    quantity characteristic, phase and units -- not in the channel name;
  * interval data split across two observations that share one time base:
    'Interval (avg)' with the derived quantities and 'Interval (max-min)' with
    the true RMS voltages and currents plus their per-interval MIN and MAX;
  * the step-pair encoding, where every interval is written twice (once at its
    start time and once at its end) with the same value;
  * 'Harm 1 of Van' as the *fundamental* (characteristic SPECTRA_HGROUP) and
    'RMS Van (V1)' as the true RMS (characteristic RMS), differing by the
    sqrt(1 + THD^2) factor exactly as a real meter reports them.  A reader that
    confuses the two produces voltages that are low by that factor.

Run this module to regenerate test_data/*.pqd.
"""

from __future__ import annotations

import struct
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import pqdif

# Recording start written into every observation's tagTimeStart.  The reader
# takes its time base from here, so nothing depends on the file name.
START_TIME = datetime(2025, 6, 25, 0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Annex B reverse lookups
# ─────────────────────────────────────────────────────────────────────────────
# Built by inverting pqdif.py's tables so the writer and the reader can never
# disagree about an identifier.

_CHARACTERISTIC_IDS = {v: k for k, v in pqdif.CHARACTERISTIC_NAMES.items()}
_VALUE_TYPE_IDS = {v: k for k, v in pqdif.VALUE_TYPE_NAMES.items()}
_QUANTITY_TYPE_IDS = {v: k for k, v in pqdif.QUANTITY_TYPE_NAMES.items()}
_MEASURED_IDS = {v: k for k, v in pqdif.QUANTITY_MEASURED_NAMES.items()}
_PHASE_IDS = {
    'none': 0, 'an': 1, 'bn': 2, 'cn': 3, 'ng': 4,
    'ab': 5, 'bc': 6, 'ca': 7, 'total': 13,
}
_UNIT_IDS = {'': 0, 's': 2, 'V': 6, 'A': 7, 'VA': 8, 'W': 9, 'VAR': 10,
             'Hz': 15, 'deg': 17, '%': 19}


# ─────────────────────────────────────────────────────────────────────────────
# Element-tree serializer (Annex A)
# ─────────────────────────────────────────────────────────────────────────────

class Collection:
    """A collection element: an ordered list of (tag, node) children."""

    def __init__(self, children=None):
        self.children: list[tuple[uuid.UUID, object]] = list(children or [])

    def add(self, tag: uuid.UUID, node) -> "Collection":
        self.children.append((tag, node))
        return self


class Scalar:
    """A single value of one physical type."""

    def __init__(self, physical_type: int, value):
        self.physical_type = physical_type
        self.value = value


class Vector:
    """An array of values of one physical type."""

    def __init__(self, physical_type: int, values):
        self.physical_type = physical_type
        self.values = values


def _scalar_bytes(physical_type: int, value) -> bytes:
    if physical_type == 60:                       # GUID
        return value.bytes_le
    if physical_type == 50:                       # TIMESTAMPPQDIF
        delta = value - datetime(1900, 1, 1)
        return struct.pack('<Id', delta.days,
                           delta.seconds + delta.microseconds / 1e6)
    if physical_type == 32:
        return struct.pack('<I', int(value))
    if physical_type == 41:
        return struct.pack('<d', float(value))
    raise ValueError(f'unsupported scalar physical type {physical_type}')


def _vector_bytes(physical_type: int, values) -> bytes:
    if physical_type == 10:                       # CHAR1 -- NUL-terminated
        raw = values.encode('latin-1') + b'\x00'
        return struct.pack('<I', len(raw)) + raw
    if physical_type == 41:
        array = np.asarray(values, dtype='<f8')
        return struct.pack('<I', len(array)) + array.tobytes()
    if physical_type == 32:
        array = np.asarray(values, dtype='<u4')
        return struct.pack('<I', len(array)) + array.tobytes()
    raise ValueError(f'unsupported vector physical type {physical_type}')


def serialize_body(root: Collection) -> bytes:
    """Lay out an element tree as a record body.

    The body begins with the root collection at offset 0 (clause 4.2.2), so its
    element table is reserved first and filled in once the children have been
    appended.  All links are relative to the start of the body and every payload
    is padded to a 4-byte multiple, as Annex A requires.
    """
    buf = bytearray()

    def reserve(size: int) -> int:
        while len(buf) % 4:
            buf.append(0)
        offset = len(buf)
        buf.extend(b'\x00' * size)
        return offset

    def append(payload: bytes) -> tuple[int, int]:
        while len(buf) % 4:
            buf.append(0)
        offset = len(buf)
        buf.extend(payload)
        padded = (len(payload) + 3) & ~3
        buf.extend(b'\x00' * (padded - len(payload)))
        return offset, padded

    def emit(node) -> tuple[int, int, bool, int, int]:
        """Return (element_type, physical_type, embedded, link, size)."""
        if isinstance(node, Collection):
            offset = write_collection(node)
            return pqdif.ELEMENT_COLLECTION, 0, False, offset, 4 + 28 * len(node.children)
        if isinstance(node, Vector):
            offset, size = append(_vector_bytes(node.physical_type, node.values))
            return pqdif.ELEMENT_VECTOR, node.physical_type, False, offset, size
        if isinstance(node, Scalar):
            raw = _scalar_bytes(node.physical_type, node.value)
            if len(raw) <= 8:
                # Annex A: a scalar of eight bytes or fewer lives in the
                # element's own link/size words (isEmbedded).
                padded = raw.ljust(8, b'\x00')
                link, size = struct.unpack('<II', padded)
                return pqdif.ELEMENT_SCALAR, node.physical_type, True, link, size
            offset, size = append(raw)
            return pqdif.ELEMENT_SCALAR, node.physical_type, False, offset, size
        raise TypeError(f'cannot serialize {node!r}')

    def write_collection(node: Collection) -> int:
        count = len(node.children)
        offset = reserve(4 + 28 * count)
        described = [(tag,) + emit(child) for tag, child in node.children]
        struct.pack_into('<I', buf, offset, count)
        for i, (tag, etype, ptype, embedded, link, size) in enumerate(described):
            base = offset + 4 + i * 28
            buf[base:base + 16] = tag.bytes_le
            struct.pack_into('<4b', buf, base + 16,
                             etype, ptype, 1 if embedded else 0, 0)
            struct.pack_into('<II', buf, base + 20, link, size)
        return offset

    assert write_collection(root) == 0, 'root collection must start the body'
    return bytes(buf)


def _record(tag: uuid.UUID, body: bytes, next_offset: int,
            compress: bool) -> bytes:
    """Wrap a body in a 64-byte record header (Annex A c_record_mainheader)."""
    payload = zlib.compress(body) if compress else body
    header = bytearray(pqdif.RECORD_HEADER_SIZE)
    header[0:16] = pqdif.RECORD_SIGNATURE.bytes_le
    header[16:32] = tag.bytes_le
    struct.pack_into('<IIII', header, 32,
                     pqdif.RECORD_HEADER_SIZE, len(payload), next_offset,
                     zlib.crc32(payload) & 0xFFFFFFFF)
    return bytes(header) + payload


# ─────────────────────────────────────────────────────────────────────────────
# Logical records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Series:
    """One series of a channel: its value type and the samples themselves."""
    value_type: str                  # TIME / AVG / MIN / MAX / VAL
    characteristic: str
    units: str
    values: np.ndarray


@dataclass
class Channel:
    """One channel definition and the single instance of it we write."""
    name: str
    phase: str
    measured: str
    quantity_type: str = 'VALUELOG'
    series: list[Series] = field(default_factory=list)


def _container(title: str) -> bytes:
    root = Collection()
    root.add(pqdif.TAG_VERSION_INFO, Vector(32, [1, 5, 1, 5]))
    root.add(pqdif._u('89738608-f1c3-11cf-9d89-0080c72e70a3'),   # tagFileName
             Vector(10, title))
    root.add(pqdif._u('89738609-f1c3-11cf-9d89-0080c72e70a3'),   # tagCreation
             Scalar(50, START_TIME))
    root.add(pqdif.TAG_COMPRESSION_STYLE, Scalar(32, 2))          # RECORDLEVEL
    root.add(pqdif.TAG_COMPRESSION_ALGORITHM, Scalar(32, 1))      # zlib
    return serialize_body(root)


def _data_source(name: str, channels: list[Channel]) -> bytes:
    root = Collection()
    root.add(pqdif.TAG_NAME_DS, Vector(10, name))
    root.add(pqdif._u('b48d8581-f5f5-11cf-9d89-0080c72e70a3'),   # tagDataSourceTypeID
             Scalar(60, pqdif._u('e2da5083-7fdb-11d3-9b39-0040052c2d28')))

    definitions = Collection()
    for channel in channels:
        defn = Collection()
        defn.add(pqdif.TAG_CHANNEL_NAME, Vector(10, channel.name))
        defn.add(pqdif.TAG_PHASE_ID, Scalar(32, _PHASE_IDS[channel.phase]))
        defn.add(pqdif.TAG_QUANTITY_TYPE_ID,
                 Scalar(60, _QUANTITY_TYPE_IDS[channel.quantity_type]))
        defn.add(pqdif.TAG_QUANTITY_MEASURED_ID,
                 Scalar(32, _MEASURED_IDS[channel.measured]))

        series_defns = Collection()
        for series in channel.series:
            sd = Collection()
            sd.add(pqdif.TAG_VALUE_TYPE_ID,
                   Scalar(60, _VALUE_TYPE_IDS[series.value_type]))
            sd.add(pqdif.TAG_QUANTITY_UNITS_ID,
                   Scalar(32, _UNIT_IDS[series.units]))
            sd.add(pqdif.TAG_QUANTITY_CHARACTERISTIC_ID,
                   Scalar(60, _CHARACTERISTIC_IDS[series.characteristic]))
            sd.add(pqdif.TAG_STORAGE_METHOD_ID, Scalar(32, pqdif.METHOD_VALUES))
            series_defns.add(pqdif.TAG_ONE_SERIES_DEFN, sd)
        defn.add(pqdif.TAG_SERIES_DEFNS, series_defns)
        definitions.add(pqdif.TAG_ONE_CHANNEL_DEFN, defn)

    root.add(pqdif.TAG_CHANNEL_DEFNS, definitions)
    return serialize_body(root)


def _observation(name: str, channels: list[Channel],
                 definition_indices: list[int],
                 start_time: datetime = None) -> bytes:
    start_time = start_time or START_TIME
    root = Collection()
    root.add(pqdif.TAG_OBSERVATION_NAME, Vector(10, name))
    root.add(pqdif.TAG_TIME_CREATE, Scalar(50, start_time))
    root.add(pqdif.TAG_TIME_START, Scalar(50, start_time))
    root.add(pqdif._u('3d786f8d-f76e-11cf-9d89-0080c72e70a3'),   # tagTriggerMethodID
             Scalar(32, 1))

    instances = Collection()
    for channel, defn_index in zip(channels, definition_indices):
        instance = Collection()
        instance.add(pqdif.TAG_CHANNEL_DEFN_IDX, Scalar(32, defn_index))
        series_instances = Collection()
        for series in channel.series:
            si = Collection()
            si.add(pqdif.TAG_SERIES_VALUES, Vector(41, series.values))
            series_instances.add(pqdif.TAG_ONE_SERIES_INSTANCE, si)
        instance.add(pqdif.TAG_SERIES_INSTANCES, series_instances)
        instances.add(pqdif.TAG_ONE_CHANNEL_INST, instance)

    root.add(pqdif.TAG_CHANNEL_INSTANCES, instances)
    return serialize_body(root)


def build_file(site: str, observations) -> bytes:
    """Assemble container + data source + observations into one PQDIF file.

    Channel definitions are pooled across observations, and each channel
    instance references its definition by index, which is the definition/
    instance split described in clause 5.4.

    Each observation is ``(name, channels)`` or ``(name, channels, start_time)``.
    Giving a start time is how a file comes to hold more than one recording
    session -- a meter reset in the field, downloaded as "all data" -- which
    clause 6 describes as chunking a log into observation records.
    """
    observations = [obs if len(obs) == 3 else (*obs, None)
                    for obs in observations]
    all_channels: list[Channel] = []
    indices: list[list[int]] = []
    for _name, channels, _start in observations:
        obs_indices = []
        for channel in channels:
            obs_indices.append(len(all_channels))
            all_channels.append(channel)
        indices.append(obs_indices)

    bodies = [
        (pqdif.TAG_CONTAINER, _container(f'{site}.pqd'), False),
        (pqdif.TAG_DATA_SOURCE, _data_source(site, all_channels), True),
    ]
    for (name, channels, start), obs_indices in zip(observations, indices):
        bodies.append((pqdif.TAG_OBSERVATION,
                       _observation(name, channels, obs_indices, start), True))

    # Two passes: the header of each record holds the absolute offset of the
    # next one, so the compressed sizes must be known before any link is set.
    payloads = [zlib.compress(body) if compress else body
                for _tag, body, compress in bodies]
    offsets, position = [], 0
    for payload in payloads:
        offsets.append(position)
        position += pqdif.RECORD_HEADER_SIZE + len(payload)

    out = bytearray()
    for i, (tag, body, compress) in enumerate(bodies):
        is_last = i == len(bodies) - 1
        out += _record(tag, body, 0 if is_last else offsets[i + 1], compress)
    return bytes(out)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario → channel specifications
# ─────────────────────────────────────────────────────────────────────────────

# Suffix used in Pronto channel names → (phase, RMS channel name, units).
_PHASE_SUFFIX = {
    'Van': ('an', 'RMS Van (V1)', 'V'), 'Vbn': ('bn', 'RMS Vbn (V2)', 'V'),
    'Vcn': ('cn', 'RMS Vcn (V3)', 'V'), 'Vne': ('ng', 'RMS Vne (V4)', 'V'),
    'Ia': ('an', 'RMS Ia (I1)', 'A'), 'Ib': ('bn', 'RMS Ib (I2)', 'A'),
    'Ic': ('cn', 'RMS Ic (I3)', 'A'), 'In': ('ng', 'RMS In (I4)', 'A'),
}

#: Characteristic and units for the non-harmonic interval channels, keyed by a
#: distinctive fragment of the Pronto channel name.
_NAMED_CHANNELS = [
    ('Real Power',     'power',   'P',        'W'),
    ('VA Reactive',    'power',   'Q',        'VAR'),
    ('Apparent Power', 'power',   'S',        'VA'),
    ('Power Factor',   'power',   'PF',       ''),
    ('K-Factor',       'current', 'K_FACTOR', ''),
    ('Flicker PST',    'voltage', 'FLKR_PST', ''),
    ('Flicker PLT',    'voltage', 'FLKR_PLT', ''),
]


#: Display resolution of a Pronto meter, in volts and amps alike -- real files
#: report 122.2 V and 0.3 A, never more precision than this.
METER_RESOLUTION = 0.1


def _step_pairs(values: np.ndarray) -> np.ndarray:
    """Repeat every sample, matching Pronto's step-pair interval encoding."""
    return np.repeat(np.asarray(values, dtype=float), 2)


def _step_times(t_sec: np.ndarray, interval: float) -> np.ndarray:
    """Interleave interval start and end times.

    The end of one interval sits 1 us before the start of the next, which is the
    gap ProntoAdapter._step_pair_stride() looks for to recognize the encoding.
    """
    starts = np.asarray(t_sec, dtype=float)
    ends = starts + interval - 1e-6
    return np.stack([starts, ends], axis=1).reshape(-1)


def scenario_channels(labels: list[str], arrays: list[np.ndarray],
                      t_sec: np.ndarray, interval: float):
    """Convert a scenario's label/array pairs into the two interval observations.

    Returns (avg_channels, maxmin_channels).  The derived quantities keep the
    names the scenario supplied; the true-RMS channels are synthesised the way a
    real meter reports them, so that 'Harm 1 of Van' (the fundamental) and
    'RMS Van (V1)' differ by exactly sqrt(1 + THD^2).
    """
    data = dict(zip(labels, arrays))
    times = _step_times(t_sec, interval)
    time_series = Series('TIME', 'INSTANTANEOUS', 's', times)

    avg: list[Channel] = []
    maxmin: list[Channel] = []

    def add_avg(name, phase, measured, characteristic, units, values,
                quantity_type='VALUELOG'):
        avg.append(Channel(name, phase, measured, quantity_type, [
            time_series,
            Series('AVG', characteristic, units, _step_pairs(values)),
        ]))

    def add_rms(name, phase, measured, units, values):
        # A real meter reports the interval average alongside the extremes
        # actually seen inside it, so min <= avg <= max always holds.
        values = np.asarray(values, dtype=float)
        spread = np.maximum(np.abs(values) * 0.004, 1e-3)
        maxmin.append(Channel(name, phase, measured, 'PHASOR', [
            time_series,
            Series('MAX', 'RMS', units, _step_pairs(values + spread)),
            Series('MIN', 'RMS', units, _step_pairs(values - spread)),
            Series('AVG', 'RMS', units, _step_pairs(values)),
        ]))

    # ── Harmonic magnitudes, and the THD the meter reports with them ──────
    harmonics: dict[tuple[str, int], np.ndarray] = {}
    for label, values in data.items():
        parts = label.split()
        if len(parts) == 4 and parts[0] == 'Harm' and parts[2] == 'of':
            harmonics[(parts[3], int(parts[1]))] = np.asarray(values, dtype=float)

    for label, values in data.items():
        parts = label.split()
        if len(parts) == 4 and parts[0] == 'Harm' and parts[2] == 'of':
            group, order = parts[3], int(parts[1])
            phase, _rms_name, units = _PHASE_SUFFIX[group]
            measured = 'voltage' if group.startswith('V') else 'current'
            # Reported per-order magnitudes are quantized to the display
            # resolution; the aggregates above are not. `harmonics` keeps the
            # unrounded values, which is what the meter computes from.
            series_values = np.round(np.asarray(values, dtype=float)
                                     / METER_RESOLUTION) * METER_RESOLUTION
            if order == 1 and measured == 'voltage':
                # The scenario's array is the true RMS; the fundamental is
                # smaller by sqrt(1 + THD^2).
                thd = _reported_thd(data, group)
                series_values = np.asarray(values, dtype=float) / np.sqrt(
                    1.0 + (thd / 100.0) ** 2)
            add_avg(label, phase, measured, 'SPECTRA_HGROUP', units,
                    series_values)
            continue

        if label.startswith('THD '):
            group = label.split()[1]
            phase, _rms_name, _units = _PHASE_SUFFIX[group]
            measured = 'voltage' if group.startswith('V') else 'current'
            add_avg(label, phase, measured, 'TOTAL_THD', '%', values)
            continue

        for fragment, measured, characteristic, units in _NAMED_CHANNELS:
            if fragment in label:
                phase = 'an' if 'Flicker' in fragment or 'K-Factor' in fragment \
                    else 'none'
                add_avg(label, phase, measured, characteristic, units, values)
                break
        else:
            raise ValueError(f'no channel specification for {label!r}')

    # ── True RMS channels, in the max-min observation ─────────────────────
    for group, (phase, rms_name, units) in _PHASE_SUFFIX.items():
        fundamental = harmonics.get((group, 1))
        if fundamental is None:
            continue
        measured = 'voltage' if group.startswith('V') else 'current'
        if measured == 'voltage':
            rms = np.asarray(fundamental, dtype=float)
        else:
            # RMS of the fundamental and every harmonic the fixture carries.
            squares = [np.asarray(v, dtype=float) ** 2
                       for (g, order), v in harmonics.items()
                       if g == group and order >= 2]
            rms = np.sqrt(np.asarray(fundamental, dtype=float) ** 2
                          + (sum(squares) if squares else 0.0))
        add_rms(rms_name, phase, measured, units, rms)

    # ── Quantities a real Pronto export always carries ────────────────────
    # These are synthesised rather than written into each scenario because they
    # follow from the voltages already there, and every service has them.

    n = len(times) // 2

    # System frequency: nominally 60 Hz with a little wander, well inside the
    # +/-0.5 Hz band so the fixtures pass unless a scenario says otherwise.
    add_avg('Frequency', 'none', 'voltage', 'FREQUENCY', 'Hz',
            60.0 + RNG.normal(0.0, 0.015, n))

    # Line-to-line voltages.  On a split-phase service the two legs are 180
    # degrees apart so the L-L voltage is their sum -- which is what makes the
    # open-neutral window visible as a steady 240 V across a swollen and a
    # sagging leg.  On a wye service the L-L voltage is sqrt(3) times the mean
    # of the two line-to-neutral voltages.
    van = harmonics.get(('Van', 1))
    vbn = harmonics.get(('Vbn', 1))
    vcn = harmonics.get(('Vcn', 1))
    if van is not None and vbn is not None:
        if vcn is None:
            add_rms('Calc RMS Vab', 'ab', 'voltage', 'V', van + vbn)
        else:
            root3 = np.sqrt(3.0)
            add_rms('Calc RMS Vab', 'ab', 'voltage', 'V', root3 * (van + vbn) / 2)
            add_rms('Calc RMS Vbc', 'bc', 'voltage', 'V', root3 * (vbn + vcn) / 2)
            add_rms('Calc RMS Vca', 'ca', 'voltage', 'V', root3 * (vcn + van) / 2)

    # Per-phase K-factor and flicker.  Phase B is deliberately the worst so the
    # fixtures exercise worst-phase selection: reading phase A alone understates
    # the transformer K-rating and can miss a flicker exceedance entirely.
    for source, factors in (
        ('K-Factor Ia', (('K-Factor Ib', 'bn', 1.8), ('K-Factor Ic', 'cn', 0.6))),
        ('Flicker PST Van (V1)', (('Flicker PST Vbn (V2)', 'bn', 1.5),
                                  ('Flicker PST Vcn (V3)', 'cn', 0.7))),
        ('Flicker PLT Van (V1)', (('Flicker PLT Vbn (V2)', 'bn', 1.5),
                                  ('Flicker PLT Vcn (V3)', 'cn', 0.7))),
    ):
        base = data.get(source)
        if base is None:
            continue
        characteristic = ('K_FACTOR' if 'K-Factor' in source else
                          'FLKR_PST' if 'PST' in source else 'FLKR_PLT')
        measured = 'current' if 'K-Factor' in source else 'voltage'
        for name, phase, factor in factors:
            # Only for phases this service actually has.
            if phase == 'cn' and vcn is None:
                continue
            add_avg(name, phase, measured, characteristic, '',
                    np.asarray(base, dtype=float) * factor)

    # ── Aggregate harmonic RMS, as the meter reports it ───────────────────
    # A real meter computes this internally at full precision but rounds the
    # per-order magnitudes to its display resolution, so summing the reported
    # orders gives slightly less than the reported aggregate. Emitting the
    # orders rounded (see METER_RESOLUTION above) and the aggregate unrounded
    # reproduces that automatically, and to the right degree: negligible when a
    # harmonic is many multiples of the resolution, large when it is comparable
    # to it, which is why the understatement bites at light load.
    for group, (phase, _rms_name, units) in _PHASE_SUFFIX.items():
        squares = [np.asarray(v, dtype=float) ** 2
                   for (g, order), v in harmonics.items()
                   if g == group and order >= 2]
        if not squares:
            continue
        measured = 'voltage' if group.startswith('V') else 'current'
        add_avg(f'Hrms {group}', phase, measured, 'HRMS', units,
                np.sqrt(sum(squares)))

    # ── Current THD, computed from the harmonics the fixture carries ──────
    for group in ('Ia', 'Ib', 'Ic', 'In'):
        fundamental = harmonics.get((group, 1))
        if fundamental is None or f'THD {group} (I1)' in data:
            continue
        squares = [np.asarray(v, dtype=float) ** 2
                   for (g, order), v in harmonics.items()
                   if g == group and order >= 2]
        if not squares:
            continue
        phase, _rms_name, _units = _PHASE_SUFFIX[group]
        denominator = np.where(np.asarray(fundamental) > 0.01, fundamental, np.nan)
        add_avg(f'THD {group}', phase, 'current', 'TOTAL_THD', '%',
                np.sqrt(sum(squares)) / denominator * 100.0)

    return avg, maxmin


def _reported_thd(data: dict, group: str) -> np.ndarray:
    """The THD channel the scenario supplied for a voltage phase, or zero."""
    for label, values in data.items():
        if label.startswith(f'THD {group}'):
            return np.asarray(values, dtype=float)
    return np.zeros(1)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ─────────────────────────────────────────────────────────────────────────────

# 5-minute intervals over 24 hours = 288 samples
N_SAMPLES   = 288
T_SEC       = np.arange(N_SAMPLES, dtype='<f8') * 300.0  # seconds since midnight
RNG         = np.random.default_rng(42)

# Phi character as CP1253 byte 0xF8 decoded with latin-1 (Pronto firmware encoding)
PHI = '\xf8'

# Current harmonic orders included (H2-H13); covers the orders that drive TDD
# and all IEEE 519-2022 per-order limits up through H13.
_H_CURR = list(range(2, 14))   # H2..H13
_H_VOLT = (3, 5, 7, 11, 13)    # voltage harmonic orders per IEEE 519-2022 Table 2


def _harm_labels(phase: str, orders: list[int]) -> list[str]:
    return [f'Harm {h} of {phase}' for h in orders]


def _noise(n: int, sigma: float = 0.3) -> np.ndarray:
    return RNG.normal(0.0, sigma, n)


# ── Residential (r): split-phase 120/240 V ───────────────────────────────────

def make_residential() -> tuple[list[str], list[np.ndarray]]:
    """Split-phase 120/240 V with open-neutral signature and voltage sag.

    Flags triggered:
      check_voltage_compliance  — L1 sag to 108 V (−10 %) at intervals 100-114
      check_neutral_health      — leg anti-correlation (open-neutral window),
                                  Van+Vbn sum swings to 238 V std > 5 V
      check_power_factor        — PF = 0.82 (below 0.90 limit)
      detect_events (flicker)   — Flicker PST = 1.3 at intervals 200-219
    """
    n = N_SAMPLES

    van = np.full(n, 121.0) + _noise(n, 0.3)
    vbn = np.full(n, 121.0) + _noise(n, 0.3)

    # Voltage sag on L1 (intervals 100-114)
    van[100:115] = 108.0

    # Open-neutral window (intervals 50-69): legs become anti-correlated —
    # load imbalance onto one leg raises it while the other drops.
    van[50:70] = 133.0 + _noise(20, 0.5)   # L1 swells
    vbn[50:70] = 107.0 + _noise(20, 0.5)   # L2 sags
    # Van + Vbn ≈ 240 V normally but here sum = 240, just wildly distributed.
    # std of (van+vbn) over the full dataset will be >> 5 V → "warning" severity.

    # Neutral-to-earth (interval data; adaptive Vne is not included here)
    vne = np.full(n, 0.04) + _noise(n, 0.01)
    vne[50:70] = 3.8 + _noise(20, 0.2)  # elevated during open-neutral

    # Currents: residential loads, slightly imbalanced legs
    ia = np.full(n, 16.0) + _noise(n, 0.4)
    ib = np.full(n, 11.0) + _noise(n, 0.4)
    in_ = np.abs(ia - ib) + _noise(n, 0.3)
    in_[50:70] += 8.0  # elevated neutral current during fault window

    # Power: ~3.2 kW, PF = 0.82 (lagging — below 0.90 limit)
    kw   = np.full(n, 3200.0) + _noise(n, 50.0)
    kvar = np.full(n, 2200.0) + _noise(n, 30.0)   # high reactive → low PF
    pf   = np.clip(kw / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.005), 0.5, 1.0)

    # Voltage THD: ~4 % (below 8 % limit, no flag intended)
    thd_van = np.clip(4.0 + _noise(n, 0.4), 0.5, 7.9)
    thd_vbn = np.clip(4.1 + _noise(n, 0.4), 0.5, 7.9)

    # Flicker PST: brief exceedance at intervals 200-219
    pst = np.full(n, 0.45) + _noise(n, 0.05)
    pst[200:220] = 1.3
    plt = np.full(n, 0.28) + _noise(n, 0.03)

    kfactor = np.full(n, 1.15) + _noise(n, 0.05)

    # Voltage harmonics (small values, typical residential)
    def vh(base: float, n: int) -> np.ndarray:
        return np.clip(base + _noise(n, 0.05), 0.01, 5.0)

    h3va = vh(1.4, n); h5va = vh(0.7, n); h7va = vh(0.3, n)
    h11va = vh(0.15, n); h13va = vh(0.10, n)
    h3vb = vh(1.4, n); h5vb = vh(0.7, n); h7vb = vh(0.3, n)
    h11vb = vh(0.15, n); h13vb = vh(0.10, n)

    # Current harmonics (H2-H13 in Amps absolute)
    # H1 (fundamental) is already ia/ib — THD will be computed from H2-H13.
    # Keeping THD modest for residential (~15 %, which may flag but is typical for SMPS loads)
    def ih(base: float, n: int) -> np.ndarray:
        return np.clip(base + _noise(n, 0.1), 0.01, 20.0)

    # Ia harmonics: dominant H3 and H5 (SMPS signature)
    ia_h = [ih(0.4, n), ih(2.2, n), ih(0.3, n), ih(1.6, n), ih(0.2, n),
            ih(0.7, n), ih(0.1, n), ih(0.3, n), ih(0.1, n), ih(0.3, n),
            ih(0.1, n), ih(0.2, n)]  # H2..H13
    # Ib harmonics (similar)
    ib_h = [ih(0.3, n), ih(1.8, n), ih(0.2, n), ih(1.3, n), ih(0.15, n),
            ih(0.6, n), ih(0.1, n), ih(0.2, n), ih(0.08, n), ih(0.25, n),
            ih(0.08, n), ih(0.18, n)]
    # In harmonics: triplens accumulate from both legs
    in_h = [ih(0.1, n), ih(4.5, n), ih(0.1, n), ih(0.5, n), ih(0.1, n),
            ih(0.2, n), ih(0.1, n), ih(0.6, n), ih(0.05, n), ih(0.1, n),
            ih(0.05, n), ih(0.15, n)]

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vne',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of In',
        f'2{PHI} 3w Real Power', f'2{PHI} 3w VA Reactive', f'2{PHI} 3w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vne,
        ia, ib, in_,
        kw, kvar, pf,
        thd_van, thd_vbn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        *ia_h, *ib_h, *in_h,
    ]
    return labels, arrays


# ── Commercial Small (c): 3-phase 120/208 V ──────────────────────────────────

def make_commercial_small() -> tuple[list[str], list[np.ndarray]]:
    """3-phase 120/208 V with low PF, voltage THD exceedance, and current imbalance.

    Flags triggered:
      check_voltage_thd         — THD Van = 9.5 % at intervals 144-215 (> 8 %)
      check_power_factor        — PF = 0.78 throughout (below 0.90 limit)
      check_current_imbalance   — L1=45 A, L2=20 A, L3=10 A → 80 % imbalance
    """
    n = N_SAMPLES

    van = np.full(n, 121.5) + _noise(n, 0.3)
    vbn = np.full(n, 121.2) + _noise(n, 0.3)
    vcn = np.full(n, 121.8) + _noise(n, 0.3)

    # Current imbalance: heavily loaded L1, lightly loaded L2/L3
    ia = np.full(n, 45.0) + _noise(n, 0.5)   # 45 A
    ib = np.full(n, 20.0) + _noise(n, 0.4)   # 20 A
    ic = np.full(n, 10.0) + _noise(n, 0.3)   # 10 A
    in_ = np.clip(ia - ib - ic, 0, 60) + _noise(n, 0.3)

    # Power: ~10 kW, PF = 0.78 (capacitive bank absent, high inductive load)
    kw   = np.full(n, 10_800.0) + _noise(n, 100.0)
    kvar = np.full(n,  8_600.0) + _noise(n, 80.0)   # high reactive
    pf   = np.clip(kw / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.005), 0.5, 1.0)

    # Voltage THD: 4 % baseline, 9.5 % during peak-load hours (intervals 144-215 = noon-6pm)
    thd_van = np.clip(4.0 + _noise(n, 0.5), 0.5, 9.9)
    thd_van[144:216] = np.clip(9.5 + _noise(72, 0.3), 8.1, 11.0)
    thd_vbn = np.clip(3.8 + _noise(n, 0.5), 0.5, 9.9)
    thd_vcn = np.clip(4.1 + _noise(n, 0.5), 0.5, 9.9)

    pst = np.full(n, 0.5) + _noise(n, 0.05)
    plt = np.full(n, 0.3) + _noise(n, 0.03)
    kfactor = np.full(n, 1.3) + _noise(n, 0.05)

    def vh(base: float) -> np.ndarray:
        return np.clip(base + _noise(n, 0.05), 0.01, 10.0)

    def vh_peak(base: float, peak: float) -> np.ndarray:
        arr = np.clip(base + _noise(n, 0.1), 0.01, 15.0)
        arr[144:216] = np.clip(peak + _noise(72, 0.1), 0.01, 15.0)
        return arr

    h3va = vh_peak(1.5, 3.5); h5va = vh_peak(2.0, 5.0); h7va = vh_peak(1.0, 2.5)
    h11va = vh(0.6); h13va = vh(0.4)
    h3vb = vh_peak(1.4, 3.3); h5vb = vh_peak(1.9, 4.8); h7vb = vh_peak(0.9, 2.4)
    h11vb = vh(0.5); h13vb = vh(0.4)
    h3vc = vh_peak(1.5, 3.4); h5vc = vh_peak(2.0, 4.9); h7vc = vh_peak(1.0, 2.4)
    h11vc = vh(0.6); h13vc = vh(0.4)

    def ih(vals: list[float]) -> list[np.ndarray]:
        return [np.clip(v + _noise(n, 0.05), 0.01, 50.0) for v in vals]

    ia_h = ih([0.5, 4.0, 0.5, 3.0, 0.3, 1.5, 0.2, 0.6, 0.15, 0.8, 0.12, 0.5])
    ib_h = ih([0.2, 1.8, 0.2, 1.4, 0.1, 0.7, 0.1, 0.3, 0.07, 0.4, 0.05, 0.25])
    ic_h = ih([0.1, 0.9, 0.1, 0.7, 0.07, 0.3, 0.05, 0.15, 0.04, 0.18, 0.03, 0.12])
    in_h = ih([0.1, 8.0, 0.1, 0.6, 0.1, 0.5, 0.1, 1.0, 0.05, 0.2, 0.04, 0.3])

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vcn',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of Ic', 'Harm 1 of In',
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)', 'THD Vcn (V3)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Vcn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('Ic',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vcn,
        ia, ib, ic, in_,
        kw, kvar, pf,
        thd_van, thd_vbn, thd_vcn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        h3vc, h5vc, h7vc, h11vc, h13vc,
        *ia_h, *ib_h, *ic_h, *in_h,
    ]
    return labels, arrays


# ── Commercial Large / C&I Secondary (sg): 3-phase 277/480 V ─────────────────

def make_commercial_large() -> tuple[list[str], list[np.ndarray]]:
    """3-phase 277/480 V with voltage imbalance, high TDD, and H5 per-order violation.

    Flags triggered:
      check_voltage_imbalance   — L1=290 V, L2=268 V, L3=280 V → 4.0 % (> 3 % limit)
      check_thd (TDD)           — H5=8 A, H7=6 A at 100 A fund. → TDD ≈ 10.7 % (> 5 %)
      check_individual_harmonics— H5 current = 8 % (may exceed IEEE 519-2022 class limit)
      check_voltage_compliance  — L2 at 268 V is 3.2 % below 277 V nominal → sag flag
    """
    n = N_SAMPLES

    # Persistent voltage imbalance due to single-phase load on one feeder
    van = np.full(n, 290.0) + _noise(n, 0.5)   # L1 high
    vbn = np.full(n, 268.0) + _noise(n, 0.5)   # L2 low  (−3.2 % from 277 V nominal)
    vcn = np.full(n, 280.0) + _noise(n, 0.5)   # L3 close to nominal
    # NEMA imbalance: avg = (290+268+280)/3 = 279.3, max_dev = 11.3 → 4.0 %

    # Currents: large VFD load, dominant H5 and H7
    ia = np.full(n, 100.0) + _noise(n, 1.0)    # 100 A fundamental
    ib = np.full(n,  98.0) + _noise(n, 1.0)
    ic = np.full(n, 101.0) + _noise(n, 1.0)
    in_ = np.full(n,   4.0) + _noise(n, 0.5)   # neutral small for balanced 3-phase

    # Power: ~80 kW, good PF
    kw   = np.full(n, 79_500.0) + _noise(n, 500.0)
    kvar = np.full(n, 12_000.0) + _noise(n, 200.0)
    pf   = np.clip(kw / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.003), 0.8, 1.0)

    # Voltage THD: moderate (5 %, below 8 % limit)
    thd_van = np.clip(5.0 + _noise(n, 0.4), 0.5, 7.9)
    thd_vbn = np.clip(4.9 + _noise(n, 0.4), 0.5, 7.9)
    thd_vcn = np.clip(5.1 + _noise(n, 0.4), 0.5, 7.9)

    pst = np.full(n, 0.4) + _noise(n, 0.04)
    plt = np.full(n, 0.25) + _noise(n, 0.03)
    kfactor = np.full(n, 2.8) + _noise(n, 0.1)

    def vh(base: float) -> np.ndarray:
        return np.clip(base + _noise(n, 0.1), 0.01, 15.0)

    # Voltage harmonics driven by high current distortion (V_h ≈ I_h × Z_source)
    h3va = vh(2.8); h5va = vh(4.2); h7va = vh(3.0); h11va = vh(1.2); h13va = vh(0.9)
    h3vb = vh(2.7); h5vb = vh(4.0); h7vb = vh(2.9); h11vb = vh(1.1); h13vb = vh(0.9)
    h3vc = vh(2.8); h5vc = vh(4.1); h7vc = vh(3.0); h11vc = vh(1.2); h13vc = vh(0.9)

    def ih(vals: list[float]) -> list[np.ndarray]:
        return [np.clip(v + _noise(n, 0.1), 0.01, 50.0) for v in vals]

    # VFD signature: strong H5 and H7, smaller H11/H13
    # H5=8A, H7=6A → TDD = sqrt(64+36+...)/100 = 10.7 %+ → flags
    ia_h = ih([0.5, 1.5, 0.5, 8.0, 0.4, 6.0, 0.3, 0.8, 0.3, 2.5, 0.2, 1.8])
    ib_h = ih([0.5, 1.4, 0.5, 7.8, 0.4, 5.9, 0.3, 0.8, 0.3, 2.4, 0.2, 1.7])
    ic_h = ih([0.5, 1.5, 0.5, 8.1, 0.4, 6.1, 0.3, 0.8, 0.3, 2.5, 0.2, 1.8])
    in_h = ih([0.1, 0.5, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2])

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vcn',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of Ic', 'Harm 1 of In',
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)', 'THD Vcn (V3)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Vcn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('Ic',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vcn,
        ia, ib, ic, in_,
        kw, kvar, pf,
        thd_van, thd_vbn, thd_vcn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        h3vc, h5vc, h7vc, h11vc, h13vc,
        *ia_h, *ib_h, *ic_h, *in_h,
    ]
    return labels, arrays


def make_solar_net_metered() -> tuple[list[str], list[np.ndarray]]:
    """3-phase 277/480 V on Schedule SG with an inverter: output collapses nightly.

    Schedule NM is a service element under every rate schedule, so this is an
    ordinary SG service that also generates -- the fixture is `--customer-class
    sg --net-metered`, not a class of its own.

    The shape is what breaks any statistic taken against the fundamental. At
    night the inverter is off and the service draws a few amps of house load,
    so the reported current THD runs to tens of percent while the harmonic
    amperes behind it stay near nothing; at noon 200 A of fundamental carries
    the same handful of amps of distortion. TDD, measured against a fixed IL,
    barely moves across the whole day.
    """
    n = N_SAMPLES
    hours = T_SEC / 3600.0
    # One clean solar day: nothing before 06:00 or after 18:00, peak at noon.
    day = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)

    # Voltage rises a little under generation, as it does at the end of a
    # feeder carrying export at midday.
    van = 279.0 + 4.0 * day + _noise(n, 0.4)
    vbn = 278.0 + 4.0 * day + _noise(n, 0.4)
    vcn = 279.5 + 4.0 * day + _noise(n, 0.4)

    # Net current at the meter: house load at night, inverter output by day.
    # The meter reports magnitude, so export and import look alike here -- it
    # is the captures that carry the direction.
    fund = 3.0 + 197.0 * day
    ia = fund + _noise(n, 0.4)
    ib = fund * 0.99 + _noise(n, 0.4)
    ic = fund * 1.01 + _noise(n, 0.4)
    in_ = np.full(n, 1.5) + _noise(n, 0.2)

    # Real power goes negative while exporting; the sign convention here is the
    # meter's, and the analysis reads magnitudes off these channels.
    kw   = -60_000.0 * day + 4_000.0 + _noise(n, 200.0)
    kvar = np.full(n, 3_000.0) + _noise(n, 100.0)
    pf   = np.clip(np.abs(kw) / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.003),
                   0.8, 1.0)

    thd_van = np.clip(3.2 + 0.8 * day + _noise(n, 0.2), 0.5, 7.9)
    thd_vbn = np.clip(3.1 + 0.8 * day + _noise(n, 0.2), 0.5, 7.9)
    thd_vcn = np.clip(3.3 + 0.8 * day + _noise(n, 0.2), 0.5, 7.9)

    pst = np.full(n, 0.35) + _noise(n, 0.04)
    plt = np.full(n, 0.22) + _noise(n, 0.03)
    kfactor = np.full(n, 2.1) + _noise(n, 0.1)

    def vh(base: float) -> np.ndarray:
        return np.clip(base * (0.4 + 0.6 * day) + _noise(n, 0.08), 0.01, 15.0)

    h3va = vh(1.6); h5va = vh(2.4); h7va = vh(1.7); h11va = vh(0.7); h13va = vh(0.5)
    h3vb = vh(1.5); h5vb = vh(2.3); h7vb = vh(1.6); h11vb = vh(0.7); h13vb = vh(0.5)
    h3vc = vh(1.6); h5vc = vh(2.4); h7vc = vh(1.7); h11vc = vh(0.7); h13vc = vh(0.5)

    def ih(vals: list[float]) -> list[np.ndarray]:
        # Harmonic amperes follow the inverter, with a small floor that does
        # not: the night-time residue is what makes the THD ratio explode while
        # the amperes behind it stay trivial.
        return [np.clip(0.15 + v * day + _noise(n, 0.05), 0.01, 50.0)
                for v in vals]

    # Inverter signature: H5 and H7 dominant, H3 small on a three-wire tie.
    ia_h = ih([0.2, 0.6, 0.2, 4.5, 0.2, 2.8, 0.1, 0.5, 0.1, 1.4, 0.1, 0.9])
    ib_h = ih([0.2, 0.6, 0.2, 4.4, 0.2, 2.7, 0.1, 0.5, 0.1, 1.4, 0.1, 0.9])
    ic_h = ih([0.2, 0.6, 0.2, 4.6, 0.2, 2.9, 0.1, 0.5, 0.1, 1.4, 0.1, 0.9])
    in_h = ih([0.05, 0.2, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1])

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vcn',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of Ic', 'Harm 1 of In',
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)', 'THD Vcn (V3)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Vcn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('Ic',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vcn,
        ia, ib, ic, in_,
        kw, kvar, pf,
        thd_van, thd_vbn, thd_vcn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        h3vc, h5vc, h7vc, h11vc, h13vc,
        *ia_h, *ib_h, *ic_h, *in_h,
    ]
    return labels, arrays


def make_producer_array() -> tuple[list[str], list[np.ndarray]]:
    """A Solar*Rewards Community producer's array: generation and nothing else.

    Schedule SRCS names the subscribers who buy the output; the array itself is
    the "SRCS Producer" and sits on the Company's own production meter. There
    is no load behind it beyond trackers and SCADA, so between dusk and dawn
    the recording is not a lightly loaded service -- it is a plant that is off.

    Rated 250 kW AC at 277/480 V, which is 301 A. The recording peaks at 240 A
    because the week was not a clear one, which is exactly the gap between
    grading against the nameplate and grading against the recording.
    """
    n = N_SAMPLES
    hours = T_SEC / 3600.0
    clear = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)
    # Cloud cover across the afternoon, so the peak sits below the nameplate.
    haze = 1.0 - 0.35 * np.clip((hours - 12.0) / 6.0, 0.0, 1.0)
    day = clear * haze

    van = 279.0 + 5.0 * day + _noise(n, 0.4)
    vbn = 278.5 + 5.0 * day + _noise(n, 0.4)
    vcn = 279.5 + 5.0 * day + _noise(n, 0.4)

    # Overnight is auxiliary load only: trackers parked, SCADA, inverter
    # standby. Under the 1 A floor, which is the correct answer for a ratio and
    # the wrong word for a plant.
    fund = 0.4 + 239.6 * day
    ia = fund + _noise(n, 0.3)
    ib = fund * 0.99 + _noise(n, 0.3)
    ic = fund * 1.01 + _noise(n, 0.3)
    in_ = np.full(n, 0.8) + _noise(n, 0.1)

    # Export throughout; the small positive term overnight is the auxiliaries.
    kw   = -195_000.0 * day + 600.0 + _noise(n, 300.0)
    kvar = np.full(n, 2_000.0) + _noise(n, 100.0)
    pf   = np.clip(np.abs(kw) / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.003),
                   0.8, 1.0)

    thd_van = np.clip(2.9 + 0.9 * day + _noise(n, 0.2), 0.5, 7.9)
    thd_vbn = np.clip(2.8 + 0.9 * day + _noise(n, 0.2), 0.5, 7.9)
    thd_vcn = np.clip(3.0 + 0.9 * day + _noise(n, 0.2), 0.5, 7.9)

    pst = np.full(n, 0.30) + _noise(n, 0.04)
    plt = np.full(n, 0.19) + _noise(n, 0.03)
    kfactor = np.full(n, 1.9) + _noise(n, 0.1)

    def vh(base: float) -> np.ndarray:
        return np.clip(base * (0.3 + 0.7 * day) + _noise(n, 0.08), 0.01, 15.0)

    h3va = vh(1.4); h5va = vh(2.2); h7va = vh(1.5); h11va = vh(0.6); h13va = vh(0.4)
    h3vb = vh(1.3); h5vb = vh(2.1); h7vb = vh(1.5); h11vb = vh(0.6); h13vb = vh(0.4)
    h3vc = vh(1.4); h5vc = vh(2.2); h7vc = vh(1.6); h11vc = vh(0.6); h13vc = vh(0.4)

    def ih(vals: list[float]) -> list[np.ndarray]:
        return [np.clip(0.05 + v * day + _noise(n, 0.04), 0.01, 50.0)
                for v in vals]

    ia_h = ih([0.2, 0.5, 0.2, 5.2, 0.2, 3.1, 0.1, 0.6, 0.1, 1.6, 0.1, 1.0])
    ib_h = ih([0.2, 0.5, 0.2, 5.1, 0.2, 3.0, 0.1, 0.6, 0.1, 1.6, 0.1, 1.0])
    ic_h = ih([0.2, 0.5, 0.2, 5.3, 0.2, 3.2, 0.1, 0.6, 0.1, 1.6, 0.1, 1.0])
    in_h = ih([0.05, 0.15, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1])

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vcn',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of Ic', 'Harm 1 of In',
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)', 'THD Vcn (V3)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Vcn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('Ic',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vcn,
        ia, ib, ic, in_,
        kw, kvar, pf,
        thd_van, thd_vbn, thd_vcn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        h3vc, h5vc, h7vc, h11vc, h13vc,
        *ia_h, *ib_h, *ic_h, *in_h,
    ]
    return labels, arrays


# ── Commercial Primary (pg): 3-phase 13,200 V (22.86 kV Y) ──────────────────

def make_commercial_primary() -> tuple[list[str], list[np.ndarray]]:
    """3-phase 13,200 V L-N (22.86 kV Y primary-metered) with voltage events and THD.

    Flags triggered:
      check_voltage_compliance  — Sag: 11,880 V (−10 %) at intervals 100-119
                                  Swell: 14,520 V (+10 %) at intervals 200-229
      check_thd (voltage THD)   — THD Van = 9 % throughout (> 8 % limit)
      check_individual_voltage_harmonics — H5 ≈ 6.3 % of fundamental (> 3 % limit)
      check_power_factor        — PF = 0.87 (below 0.90 limit)
    """
    n = N_SAMPLES
    nom = 13_200.0

    van = np.full(n, nom) + _noise(n, 25.0)
    vbn = np.full(n, nom) + _noise(n, 25.0)
    vcn = np.full(n, nom) + _noise(n, 25.0)

    # Voltage sag: −10 % (utility switching or fault)
    van[100:120] = nom * 0.90 + _noise(20, 15.0)
    vbn[100:120] = nom * 0.90 + _noise(20, 15.0)
    vcn[100:120] = nom * 0.90 + _noise(20, 15.0)

    # Voltage swell: +10 % (load rejection / capacitor bank switching)
    van[200:230] = nom * 1.10 + _noise(30, 15.0)
    vbn[200:230] = nom * 1.10 + _noise(30, 15.0)
    vcn[200:230] = nom * 1.10 + _noise(30, 15.0)

    # Currents: ~500 kW at 13,200 V L-N → I = 500 kW / (3 × 13,200 × PF) ≈ 15 A
    ia = np.full(n, 15.0) + _noise(n, 0.2)
    ib = np.full(n, 15.1) + _noise(n, 0.2)
    ic = np.full(n, 14.9) + _noise(n, 0.2)
    in_ = np.full(n, 0.6) + _noise(n, 0.05)

    # Power: ~500 kW, moderate reactive (PF ≈ 0.87 — below 0.90 limit)
    kw   = np.full(n, 500_000.0) + _noise(n, 5000.0)
    kvar = np.full(n, 290_000.0) + _noise(n, 3000.0)
    pf   = np.clip(kw / np.sqrt(kw**2 + kvar**2) + _noise(n, 0.003), 0.7, 1.0)

    # Voltage THD: 9 % (above 8 % limit) — arc furnace or rectifier influence
    thd_van = np.clip(9.0 + _noise(n, 0.4), 7.0, 12.0)
    thd_vbn = np.clip(8.8 + _noise(n, 0.4), 7.0, 12.0)
    thd_vcn = np.clip(9.1 + _noise(n, 0.4), 7.0, 12.0)

    pst = np.full(n, 0.55) + _noise(n, 0.05)
    pst[150:170] = 1.2   # flicker exceedance
    plt = np.full(n, 0.35) + _noise(n, 0.03)
    kfactor = np.full(n, 3.5) + _noise(n, 0.15)

    # Voltage harmonics scaled to 13,200 V:
    # H5 = 6.3 % × 13,200 = 832 V — exceeds IEEE 519-2022 3 % individual limit
    # H7 = 4.5 % × 13,200 = 594 V
    def vh(base_v: float) -> np.ndarray:
        return np.clip(base_v + _noise(n, base_v * 0.02), 0.0, nom * 0.2)

    h3va = vh(660.0); h5va = vh(832.0); h7va = vh(594.0)
    h11va = vh(264.0); h13va = vh(198.0)
    h3vb = vh(649.0); h5vb = vh(817.0); h7vb = vh(583.0)
    h11vb = vh(259.0); h13vb = vh(193.0)
    h3vc = vh(665.0); h5vc = vh(838.0); h7vc = vh(599.0)
    h11vc = vh(267.0); h13vc = vh(200.0)

    def ih(vals: list[float]) -> list[np.ndarray]:
        return [np.clip(v + _noise(n, 0.05), 0.01, 10.0) for v in vals]

    # Same harmonic signature as 2400 V scenario, scaled to 15 A fundamental:
    # TDD = sqrt(H2²+…+H13²) / 15 ≈ 9.5 % → flags > 8 % limit
    ia_h = ih([0.06, 0.32, 0.06, 1.10, 0.06, 0.75, 0.04, 0.13, 0.04, 0.32, 0.03, 0.21])
    ib_h = ih([0.06, 0.32, 0.06, 1.08, 0.06, 0.73, 0.04, 0.13, 0.04, 0.31, 0.03, 0.21])
    ic_h = ih([0.06, 0.32, 0.06, 1.11, 0.06, 0.75, 0.04, 0.13, 0.04, 0.32, 0.03, 0.21])
    in_h = ih([0.01, 0.04, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02])

    labels = [
        'Harm 1 of Van', 'Harm 1 of Vbn', 'Harm 1 of Vcn',
        'Harm 1 of Ia', 'Harm 1 of Ib', 'Harm 1 of Ic', 'Harm 1 of In',
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
        'THD Van (V1)', 'THD Vbn (V2)', 'THD Vcn (V3)',
        'K-Factor Ia', 'Flicker PST Van (V1)', 'Flicker PLT Van (V1)',
        *_harm_labels('Van', _H_VOLT),
        *_harm_labels('Vbn', _H_VOLT),
        *_harm_labels('Vcn', _H_VOLT),
        *_harm_labels('Ia',  _H_CURR),
        *_harm_labels('Ib',  _H_CURR),
        *_harm_labels('Ic',  _H_CURR),
        *_harm_labels('In',  _H_CURR),
    ]
    arrays = [
        van, vbn, vcn,
        ia, ib, ic, in_,
        kw, kvar, pf,
        thd_van, thd_vbn, thd_vcn,
        kfactor, pst, plt,
        h3va, h5va, h7va, h11va, h13va,
        h3vb, h5vb, h7vb, h11vb, h13vb,
        h3vc, h5vc, h7vc, h11vc, h13vc,
        *ia_h, *ib_h, *ic_h, *in_h,
    ]
    return labels, arrays


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

#: Interval length in seconds (T_SEC is spaced by this).
INTERVAL_SEC = 300.0

SCENARIOS = [
    ("test_residential.pqd",        make_residential),
    ("test_commercial_small.pqd",   make_commercial_small),
    ("test_commercial_large.pqd",   make_commercial_large),
    ("test_commercial_primary.pqd", make_commercial_primary),
    ("test_solar_net_metered.pqd",  make_solar_net_metered),
    ("test_producer_array.pqd",     make_producer_array),
]


# ─────────────────────────────────────────────────────────────────────────────
# Point-on-wave captures
# ─────────────────────────────────────────────────────────────────────────────
#
# Interval channels carry magnitudes only, so a file built from them alone can
# never exercise the half of the direction check that needs angles.  A capture
# is a separate observation whose channels are quantity type WAVEFORM, each
# holding its own time base and a VAL/INSTANTANEOUS series of samples -- which
# is what clause 5.4 calls an instantaneous-value channel and what the adapter
# looks for when it sorts observations into interval and waveform.

#: 19.2 kHz, the rate a Pronto records its steady-state captures at: 320
#: samples per 60 Hz cycle, comfortably above the H13 the report reads.
CAPTURE_FS_HZ = 19200.0
CAPTURE_CYCLES = 10


def _capture_samples(spectrum: dict[int, tuple[float, float]],
                     fs: float = CAPTURE_FS_HZ,
                     cycles: int = CAPTURE_CYCLES,
                     f0: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """One channel's samples, from {order: (peak amplitude, phase in degrees)}.

    Stating the spectrum as amplitude and angle rather than as a waveform is
    what makes a capture fixture legible: the sign of harmonic power depends on
    nothing but the angle between a voltage order and the current order at the
    same frequency, so the intended answer is visible in the numbers.
    """
    n = int(round(fs * cycles / f0))
    t = np.arange(n) / fs
    out = np.zeros(n)
    for order, (amp, deg) in spectrum.items():
        out += amp * np.cos(2 * np.pi * f0 * order * t + np.radians(deg))
    return t, out


def capture_channels(voltage: dict[str, dict], current: dict[str, dict],
                     fs: float = CAPTURE_FS_HZ,
                     cycles: int = CAPTURE_CYCLES) -> list[Channel]:
    """The channels of one capture, from a per-phase spectrum for V and I.

    *voltage* and *current* map a phase ('a', 'b', 'c') to a spectrum in the
    form `_capture_samples` takes.
    """
    channels: list[Channel] = []
    for measured, group, units, prefix in (("voltage", voltage, "V", "V"),
                                           ("current", current, "A", "I")):
        for phase, spectrum in group.items():
            t, samples = _capture_samples(spectrum, fs, cycles)
            name = (f"{prefix}{phase}n" if measured == "voltage"
                    else f"{prefix}{phase}")
            channels.append(Channel(name, f"{phase}n", measured, 'WAVEFORM', [
                Series('TIME', 'INSTANTANEOUS', 's', t),
                Series('VAL', 'INSTANTANEOUS', units, samples),
            ]))
    return channels


def _steady_capture(v_rms: float, i_rms: float, *, exporting: bool = False,
                    h5_v: float = 4.0, h5_i: float = 2.0,
                    phases=("a", "b", "c")) -> list[Channel]:
    """A capture whose H5 leaves the premises: the source is on the customer side.

    V5 and I5 are put in antiphase, so P5 = ½·Re(V5·I5*) is negative whichever
    way the fundamental runs.  With *exporting* set, the fundamental current is
    reversed against the voltage as it is on a generating service -- the case
    that reads as reversed CTs unless the service is declared net-metered.
    """
    root2 = np.sqrt(2.0)
    i1_deg = 180.0 if exporting else 0.0
    return capture_channels(
        {p: {1: (v_rms * root2, 0.0), 5: (h5_v, 180.0)} for p in phases},
        {p: {1: (i_rms * root2, i1_deg), 5: (h5_i, 0.0)} for p in phases},
    )


def _large_captures() -> list[tuple]:
    """An ordinary load service: every capture taken while importing."""
    return [
        (f"test_commercial_large - Waveform {i + 1}",
         _steady_capture(277.0, 100.0),
         START_TIME + timedelta(hours=3 * (i + 1)))
        for i in range(4)
    ]


def _solar_captures() -> list[tuple]:
    """A generating service, captured on both sides of the day.

    Two before the inverter starts and four across the middle of the day, so
    the file holds enough of each to grade: the split needs three phase-
    readings per order per half, and each capture carries three phases.

    Without the exporting captures the fixture would not reach the case that
    matters -- a negative fundamental that is generation and not a reversed CT.
    """
    hours_importing = (2, 4)
    hours_exporting = (10, 12, 13, 15)
    out = []
    for h in hours_importing:
        out.append((f"test_solar_net_metered - Waveform {h:02d}00 (import)",
                    _steady_capture(277.0, 6.0, exporting=False),
                    START_TIME + timedelta(hours=h)))
    for h in hours_exporting:
        out.append((f"test_solar_net_metered - Waveform {h:02d}00 (export)",
                    _steady_capture(281.0, 180.0, exporting=True),
                    START_TIME + timedelta(hours=h)))
    return out


#: Which fixtures carry point-on-wave captures, and what is in them. Kept apart
#: from SCENARIOS because a capture is a separate observation, not another
#: interval channel.
def _producer_captures() -> list[tuple]:
    """A plant: every usable capture taken while exporting.

    There is no importing half to compare against, which is the point -- the
    polarity check here is the load one with its sign flipped, not the split
    used on a mixed service.
    """
    return [
        (f"test_producer_array - Waveform {h:02d}00",
         _steady_capture(282.0, 230.0, exporting=True),
         START_TIME + timedelta(hours=h))
        for h in (9, 11, 13, 15)
    ]


CAPTURES = {
    "test_commercial_large.pqd":  _large_captures,
    "test_solar_net_metered.pqd": _solar_captures,
    "test_producer_array.pqd":    _producer_captures,
}


#: The second session's start, three days after the first: long enough that
#: nothing could mistake the two for one recording with a gap in it.
SECOND_SESSION_START = datetime(2025, 6, 28, 0, 0, 0)


def _write_two_session_file(out_dir: Path) -> None:
    """A file holding two recording sessions, as "download all data" gives.

    A meter reset or re-armed in the field starts a new session, and a download
    of everything on the meter carries every session it still holds. The reader
    analyses one of them, so there has to be a fixture where choosing the wrong
    one is visible: the sessions here differ in length (24 h against 12 h) and
    start three days apart.
    """
    labels, arrays = make_residential()
    first_avg, first_maxmin = scenario_channels(labels, arrays, T_SEC, INTERVAL_SEC)

    half = N_SAMPLES // 2
    short_arrays = [np.asarray(a, dtype=float)[:half] for a in arrays]
    short_t = T_SEC[:half]
    second_avg, second_maxmin = scenario_channels(
        labels, short_arrays, short_t, INTERVAL_SEC)

    site = "test_two_sessions"
    pqd_bytes = build_file(site, [
        (f"{site} (general) - Interval (avg)", first_avg, START_TIME),
        (f"{site} (general) - Interval (max-min)", first_maxmin, START_TIME),
        (f"{site} (general) - Interval (avg)", second_avg, SECOND_SESSION_START),
        (f"{site} (general) - Interval (max-min)", second_maxmin,
         SECOND_SESSION_START),
    ])
    path = out_dir / f"{site}.pqd"
    path.write_bytes(pqd_bytes)
    parsed = pqdif.PQDIFFile(path)
    print(f"  wrote {path}  ({len(pqd_bytes):,} bytes, "
          f"{parsed.observation_count} observations in 2 sessions: "
          f"{N_SAMPLES} and {half} intervals)")


def main():
    out_dir = Path(__file__).parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    for fname, builder in SCENARIOS:
        labels, arrays = builder()
        assert len(labels) == len(arrays), \
            f"{fname}: label count ({len(labels)}) != array count ({len(arrays)})"
        for i, (lbl, arr) in enumerate(zip(labels, arrays)):
            assert len(arr) == N_SAMPLES, \
                f"{fname}: channel {i} ({lbl!r}) has {len(arr)} samples, expected {N_SAMPLES}"

        site = Path(fname).stem
        avg, maxmin = scenario_channels(labels, arrays, T_SEC, INTERVAL_SEC)
        observations = [
            (f"{site} (general) - Interval (avg)", avg),
            (f"{site} (general) - Interval (max-min)", maxmin),
        ]
        observations.extend(CAPTURES.get(fname, lambda: [])())
        pqd_bytes = build_file(site, observations)
        path = out_dir / fname
        path.write_bytes(pqd_bytes)

        # Reading each file back is the only check that matters: it proves the
        # fixture is valid PQDIF and takes the spec path in the real adapter.
        parsed = pqdif.PQDIFFile(path)
        print(f"  wrote {path}  ({len(pqd_bytes):,} bytes, "
              f"{len(parsed.definitions)} channel definitions, "
              f"{len(avg)} avg + {len(maxmin)} max-min channels"
              + (f", {len(observations) - 2} captures"
                 if len(observations) > 2 else "") + ")")

    _write_two_session_file(out_dir)

    print(f"\nSample CLI commands (run from repo root):\n")
    print("  python pq_analyzer.py test_data/test_residential.pqd \\")
    print("    --nominal 120 --topology split-phase --customer-class r\n")
    print("  python pq_analyzer.py test_data/test_commercial_small.pqd \\")
    print("    --nominal 120 --customer-class c\n")
    print("  python pq_analyzer.py test_data/test_commercial_large.pqd \\")
    print("    --nominal 277 --customer-class sg --isc 5000\n")
    print("  python pq_analyzer.py test_data/test_commercial_primary.pqd \\")
    print("    --nominal 13200 --customer-class pg --isc 5000\n")


if __name__ == "__main__":
    main()

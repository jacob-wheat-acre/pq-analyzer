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

Binary format: Pronto proprietary extension of IEEE 1159.3 PQDIF.

Each file record wrapper (from _walk_records):
  [+16] u32 tag     0x89738619=DataSource, 0x8973861A=Observation
  [+32] u32 hdr     header size (body starts at pos+hdr)
  [+36] u32 blen    compressed body length
  [+40] u32 nxt     absolute offset of next record (0 = last)

DataSource body — PQDIF element tree (IEEE 1159.3 §5):
  top[0] ChannelDefinitions (guid4=0xB48D858D) → list of N ChannelDefinition elements
  each ChannelDefinition → sub-list with label element (guid4=0xB48D8590)
  label element: off → 4-byte length prefix + label bytes, sz = 4 + len(label)

"Interval (avg)" obs body (Pronto-specific layout):
  [0-31]    top PQDIF element: ChannelInstances (guid4=0x3D786F91) → ci_off
  [144-147] label_length (14 for "Interval (avg)")
  [148-163] b'Interval (avg)' + 2 pad bytes
  [164-187] zero padding
  [188=ci_off] u32 count = N channels   (ci_off = entry_start − 4)
  [192=entry_start] N × 28-byte ChannelInstance entries
    entry[ci]+20 = abs offset of channel block in obs body
  [channel blocks] for each ci:
    [+0 ]  PQDIF sub-list: count=1, element guid4=0xB48D858F, off=DS_ci (inline)
    [+32-179] zero padding
    [+180] u32 ts_count_raw = 2*n
    [+184] n × (f64 t_sec, f64 quality=0)  — timestamps
    [+184+16n] 32-byte gap
    [+216+16n] u32 data_count = 2*n
    [+220+16n] n × (f64 value, f64 quality=0)  — measurements

Stub obs bodies supply the base date parsed by _parse_v2_date (seeks MM/DD/YY in
bytes 148-220 of each decompressed observation body).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# ── PQDIF GUID4 constants (first 4 bytes of 16-byte GUID, little-endian) ────
# These identifiers are defined in the IEEE 1159.3-2019 PQDIF standard and
# the Pronto vendor extension to that standard.
_TAG_DS   = 0x89738619  # DataSource record (IEEE 1159.3 §6.2)
_TAG_OBS  = 0x8973861A  # Observation record (IEEE 1159.3 §6.3)
_CHAN_DEFS = 0xB48D858D  # ChannelDefinitions collection (IEEE 1159.3 §7.2.1)
_CHAN_LBL  = 0xB48D8590  # ChannelDefinition label (IEEE 1159.3 §7.2.1.3)
_CHAN_INST = 0x3D786F91  # ChannelInstances collection (IEEE 1159.3 §7.3.1)
_DS_IDX    = 0xB48D858F  # DataSource channel index (Pronto extension, inline scalar)

# ── Pronto obs-body layout constants (must match ProntoAdapter) ─────────────
_ENTRY_SZ  = 28   # bytes per ChannelInstance entry
_BODY_OFF  = 20   # offset within entry of channel-block abs pointer (= e['off'])
_TS_REL    = 180  # offset from channel-block start to ts_count_raw
_HDR_SZ    = 48   # PQDIF record header size


# ─────────────────────────────────────────────────────────────────────────────
# Binary primitives
# ─────────────────────────────────────────────────────────────────────────────

def _elem(guid4: int, etype: int, off: int, sz: int) -> bytes:
    """One 28-byte PQDIF element: GUID(16) | type(4) | off(4) | sz(4)."""
    return struct.pack('<I', guid4) + b'\x00' * 12 + struct.pack('<III', etype, off, sz)


def _elem_list(elements: list[bytes]) -> bytes:
    """PQDIF element list: u32 count + concatenated elements."""
    return struct.pack('<I', len(elements)) + b''.join(elements)


def _record(tag: int, body: bytes, next_off: int) -> bytes:
    """PQDIF record with 48-byte header."""
    hdr = bytearray(_HDR_SZ)
    struct.pack_into('<I', hdr, 16, tag)
    struct.pack_into('<I', hdr, 32, _HDR_SZ)
    struct.pack_into('<I', hdr, 36, len(body))
    struct.pack_into('<I', hdr, 40, next_off)
    return bytes(hdr) + body


# ─────────────────────────────────────────────────────────────────────────────
# DataSource record
# ─────────────────────────────────────────────────────────────────────────────

def _make_datasource(labels: list[str]) -> bytes:
    """Build DataSource body with N channel label definitions.

    Layout:
      [0 ]    top element list → ChannelDefinitions at cd_off=32
      [32]    ChannelDefinitions list (N elements → sub_list[i])
      [32+4+N*28]  N sub-lists (32 bytes each) → string data
      [32+4+N*28+N*32]  string data: u32 len + bytes (no null needed)
    """
    N = len(labels)
    cd_off         = 32
    sub_base       = cd_off + 4 + N * 28
    str_base       = sub_base + N * 32

    # Pre-compute string offsets and sizes
    str_offs: list[int] = []
    str_szs:  list[int] = []
    cur = str_base
    for lbl in labels:
        raw = lbl.encode('latin-1')
        str_offs.append(cur)
        str_szs.append(4 + len(raw))  # sz = 4-byte prefix + raw bytes
        cur += 4 + len(raw)

    body = bytearray()

    # Top-level: one ChannelDefinitions element
    body += _elem_list([_elem(_CHAN_DEFS, 1, cd_off, 0)])
    assert len(body) == cd_off

    # ChannelDefinitions list: N elements, each → its sub-list
    body += _elem_list([_elem(_CHAN_DEFS, 1, sub_base + i * 32, 32) for i in range(N)])
    assert len(body) == sub_base

    # Sub-lists: one label element per channel
    for s_off, s_sz in zip(str_offs, str_szs):
        body += _elem_list([_elem(_CHAN_LBL, 1, s_off, s_sz)])
    assert len(body) == str_base

    # String data: 4-byte length + label bytes (latin-1 preserves CP1253 φ=0xF8)
    for lbl in labels:
        raw = lbl.encode('latin-1')
        body += struct.pack('<I', len(raw)) + raw

    return bytes(body)


# ─────────────────────────────────────────────────────────────────────────────
# Observation records
# ─────────────────────────────────────────────────────────────────────────────

def _make_stub_obs(date_str: str = "06/25/25") -> bytes:
    """Minimal obs body with a date pattern in bytes 148–220 for _parse_v2_date.

    _parse_v2_date searches decompressed obs bodies for r'(\\d\\d)/(\\d\\d)/(\\d\\d)'
    in that byte window, treating it as MM/DD/YY.
    """
    label = f"Waveshape {date_str}".encode('ascii')
    body = bytearray(256)
    struct.pack_into('<I', body, 144, len(label))
    body[148:148 + len(label)] = label
    return bytes(body)


def _ch_block_size(n: int) -> int:
    """Byte size of one channel block for n samples."""
    # 180 (sub-list+pad) + 4 (ts_count) + 2n*8 (ts pairs) + 32 (gap) + 4 (data_count) + 2n*8
    return 180 + 4 + 2 * n * 8 + 32 + 4 + 2 * n * 8


def _make_channel_block(ds_ci: int, t_sec: np.ndarray, values: np.ndarray) -> bytes:
    """One channel data block inside the Interval (avg) obs body.

    The block must satisfy two independent readers in ProntoAdapter:
      _build_label_map  — reads PQDIF sub-element list at block start to find DS_ci
      read_v2           — reads timestamps at +_TS_REL=180 and data after the gap
    """
    n = len(values)
    block = bytearray()

    # PQDIF sub-element list: count=1, element with DS channel index (inline scalar)
    # type=0 (scalar) means the value is stored in the 'off' field directly (IEEE 1159.3 §5.2)
    block += _elem_list([_elem(_DS_IDX, 0, ds_ci, 0)])  # 32 bytes
    block += b'\x00' * (_TS_REL - len(block))            # pad to offset 180
    assert len(block) == _TS_REL

    # Timestamps: u32 ts_count_raw = 2*n, then n pairs of (t_sec f64, quality f64)
    ts_count_raw = 2 * n
    block += struct.pack('<I', ts_count_raw)
    ts_pairs = np.empty(ts_count_raw, dtype='<f8')
    ts_pairs[0::2] = t_sec
    ts_pairs[1::2] = 0.0
    block += ts_pairs.tobytes()

    block += b'\x00' * 32  # gap between timestamps and data

    # Data: u32 data_count = 2*n, then n pairs of (value f64, quality f64)
    data_count = 2 * n
    block += struct.pack('<I', data_count)
    data_pairs = np.empty(data_count, dtype='<f8')
    data_pairs[0::2] = values
    data_pairs[1::2] = 0.0
    block += data_pairs.tobytes()

    assert len(block) == _ch_block_size(n)
    return bytes(block)


def _make_interval_obs(labels: list[str], channel_arrays: list[np.ndarray],
                       t_sec: np.ndarray) -> bytes:
    """Build the 'Interval (avg)' observation body.

    The obs body is consumed two ways:
      _build_label_map: reads top PQDIF element (ChannelInstances → ci_off),
                        then enumerates ChannelInstance entries (obs_ci → ch_block),
                        then reads each block's PQDIF sub-list for DS_ci.
      _load_v2 read_v2: reads channel block pointer at entry_start + ci*28 + 20,
                        then timestamps at ch_block + _TS_REL, data after gap.
    Both readers share the same 28-byte ChannelInstance entries; the 'off' field
    at byte +20 of each entry is both the PQDIF pointer AND the raw channel-block
    absolute offset.
    """
    assert len(labels) == len(channel_arrays)
    N = len(labels)
    n = len(t_sec)

    obs_label      = b'Interval (avg)'   # searched by _load_v2
    label_len      = len(obs_label)      # 14
    aligned_len    = (label_len + 3) & ~3  # 16 (round up to 4-byte boundary)
    entry_start    = 148 + aligned_len + 28  # = 192
    ci_off         = entry_start - 4         # = 188  (count u32 lives here)

    ch_blk_sz      = _ch_block_size(n)
    ch_blocks_base = entry_start + N * _ENTRY_SZ   # channel blocks start here
    ch_offsets     = [ch_blocks_base + i * ch_blk_sz for i in range(N)]

    body = bytearray()

    # [0-31] top-level element: ChannelInstances → ci_off
    ci_block_sz = 4 + N * _ENTRY_SZ
    body += _elem_list([_elem(_CHAN_INST, 1, ci_off, ci_block_sz)])
    assert len(body) == 32

    # [32-143] zero padding
    body += b'\x00' * (144 - 32)

    # [144-147] label_length
    body += struct.pack('<I', label_len)

    # [148 .. 148+aligned_len-1] obs label + zero padding
    body += obs_label + b'\x00' * (aligned_len - label_len)
    assert len(body) == 148 + aligned_len  # = 164

    # [164-187] 24 bytes zero padding
    body += b'\x00' * 24

    # [188=ci_off] channel count (u32) — this is the "count" for _pqdif_elements(obs_body, ci_off)
    body += struct.pack('<I', N)
    assert len(body) == entry_start  # = 192

    # [entry_start .. entry_start+N*28] ChannelInstance entries
    # Each is a 28-byte PQDIF element where 'off' = absolute channel-block offset.
    # type=1 satisfies _build_label_map's `if e['type'] != 1: continue` guard.
    for ch_off in ch_offsets:
        body += _elem(_CHAN_INST, 1, ch_off, ch_blk_sz)
    assert len(body) == ch_blocks_base

    # Channel data blocks
    for i, values in enumerate(channel_arrays):
        body += _make_channel_block(i, t_sec, values)

    return bytes(body)


# ─────────────────────────────────────────────────────────────────────────────
# File assembler
# ─────────────────────────────────────────────────────────────────────────────

def _build_pqd(labels: list[str], arrays: list[np.ndarray],
               t_sec: np.ndarray, date_str: str = "06/25/25") -> bytes:
    """Assemble a complete Pronto PQDIF file from channel labels and data arrays.

    Record chain: DataSource → StubObs × 3 → IntervalObs
    _load_v2 requires ≥ 4 Observation records; the three stubs satisfy that
    while also providing the base date that _parse_v2_date extracts.
    """
    ds_body  = _make_datasource(labels)
    stub1    = zlib.compress(_make_stub_obs(date_str))
    stub2    = zlib.compress(_make_stub_obs())
    stub3    = zlib.compress(_make_stub_obs())
    avg_body = zlib.compress(_make_interval_obs(labels, arrays, t_sec))

    ds_sz    = _HDR_SZ + len(ds_body)
    s1_sz    = _HDR_SZ + len(stub1)
    s2_sz    = _HDR_SZ + len(stub2)
    s3_sz    = _HDR_SZ + len(stub3)

    off0 = 0
    off1 = off0 + ds_sz
    off2 = off1 + s1_sz
    off3 = off2 + s2_sz
    off4 = off3 + s3_sz

    return (
        _record(_TAG_DS,  ds_body, off1) +
        _record(_TAG_OBS, stub1,   off2) +
        _record(_TAG_OBS, stub2,   off3) +
        _record(_TAG_OBS, stub3,   off4) +
        _record(_TAG_OBS, avg_body, 0)
    )


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
        f'3{PHI} 4w Real Power', f'3{PHI} 4w VA Reactive', f'3{PHI} 4w Power Factor',
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

SCENARIOS = [
    ("test_residential.pqd",        make_residential,       "06/25/25"),
    ("test_commercial_small.pqd",   make_commercial_small,  "06/25/25"),
    ("test_commercial_large.pqd",   make_commercial_large,  "06/25/25"),
    ("test_commercial_primary.pqd", make_commercial_primary,"06/25/25"),
]


def main():
    out_dir = Path(__file__).parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    for fname, builder, date_str in SCENARIOS:
        labels, arrays = builder()
        assert len(labels) == len(arrays), \
            f"{fname}: label count ({len(labels)}) != array count ({len(arrays)})"
        for i, (lbl, arr) in enumerate(zip(labels, arrays)):
            assert len(arr) == N_SAMPLES, \
                f"{fname}: channel {i} ({lbl!r}) has {len(arr)} samples, expected {N_SAMPLES}"

        pqd_bytes = _build_pqd(labels, arrays, T_SEC, date_str)
        path = out_dir / fname
        path.write_bytes(pqd_bytes)
        print(f"  wrote {path}  ({len(pqd_bytes):,} bytes,  {len(labels)} channels)")

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

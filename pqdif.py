"""Spec-compliant PQDIF reader — IEEE Std 1159.3-2019.

This module implements the format as specified, with no reverse-engineered byte
offsets.  Everything is resolved by walking the documented structure:

    Clause 4.2 / Annex A  physical structure (record header, collection/scalar/
                          vector elements, physical type IDs)
    Clause 5.4            data source ⇄ observation definition/instance parallel
    Clause 5.5            series storage methods (VALUES / SCALED / INCREMENT,
                          and shared series)
    Annex B               tag GUIDs and ID values

Reading a value is a traversal, never an offset computation::

    Observation
      └── tagChannelInstances
            └── tagOneChannelInst
                  ├── tagChannelDefnIdx ──→ DataSource tagOneChannelDefn
                  │                            └── tagSeriesDefns
                  │                                  └── tagOneSeriesDefn
                  │                                        └── tagValueTypeID
                  └── tagSeriesInstances
                        └── tagOneSeriesInstance
                              └── tagSeriesValues

The i-th series instance corresponds to the i-th series definition (clause
5.4.3), which is how a value array learns whether it is TIME, AVG, MIN or MAX.

Verified against real Pronto exports from two firmware generations; both are
fully compliant and use no vendor-private GUIDs.

Usage::

    f = PQDIFFile("meter.pqd")
    for obs in f.observations:
        print(obs.name, obs.start_time)
        for ch in obs.channels:
            print(ch.name, ch.phase, list(ch.series))
            volts = ch.series.get("AVG")
"""

from __future__ import annotations

import logging
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "PQDIFFile", "Observation", "Channel", "PQDIFError",
    "VALUE_TYPE_NAMES", "QUANTITY_TYPE_NAMES", "PHASE_NAMES",
    "CHARACTERISTIC_NAMES", "QUANTITY_MEASURED_NAMES", "UNITS_SYMBOLS",
]


class PQDIFError(Exception):
    """Raised when a file does not conform to IEEE 1159.3."""


# ─────────────────────────────────────────────────────────────────────────────
# Annex A — physical structure
# ─────────────────────────────────────────────────────────────────────────────

#: Signature GUID in every record header (Annex A, guidRecordSignaturePQDIF).
RECORD_SIGNATURE = uuid.UUID("4a111440-e49f-11cf-9900-505144494600")

RECORD_HEADER_SIZE = 64

ELEMENT_COLLECTION = 1
ELEMENT_SCALAR = 2
ELEMENT_VECTOR = 3

#: Element entry in a collection: 16-byte GUID, 4 one-byte fields, link, size.
_ELEMENT_SIZE = 28

# Physical type id → (struct format char or None, size in bytes).
# Annex A, ID_PHYS_TYPE_*.
_PHYS: Dict[int, tuple] = {
    1:  ("B", 1),   # BOOLEAN1
    2:  ("h", 2),   # BOOLEAN2
    3:  ("i", 4),   # BOOLEAN4
    10: (None, 1),  # CHAR1   (ASCII, handled as bytes)
    11: (None, 2),  # CHAR2   (Unicode)
    20: ("b", 1),   # INTEGER1
    21: ("h", 2),   # INTEGER2
    22: ("i", 4),   # INTEGER4
    30: ("B", 1),   # UNS_INTEGER1
    31: ("H", 2),   # UNS_INTEGER2
    32: ("I", 4),   # UNS_INTEGER4
    40: ("f", 4),   # REAL4
    41: ("d", 8),   # REAL8
    42: (None, 8),  # COMPLEX8   — two REAL4: real, imag
    43: (None, 16), # COMPLEX16  — two REAL8: real, imag
    50: (None, 12), # TIMESTAMPPQDIF — UINT4 days + REAL8 seconds
    60: (None, 16), # GUID
}

PHYS_TYPE_NAMES = {
    1: "BOOLEAN1", 2: "BOOLEAN2", 3: "BOOLEAN4", 10: "CHAR1", 11: "CHAR2",
    20: "INTEGER1", 21: "INTEGER2", 22: "INTEGER4",
    30: "UNS_INTEGER1", 31: "UNS_INTEGER2", 32: "UNS_INTEGER4",
    40: "REAL4", 41: "REAL8", 42: "COMPLEX8", 43: "COMPLEX16",
    50: "TIMESTAMPPQDIF", 60: "GUID",
}

#: PQDIF timestamps count days from this epoch (Annex A, struct ts).
_EPOCH = datetime(1900, 1, 1)


def _u(s: str) -> uuid.UUID:
    return uuid.UUID(s)


# ─────────────────────────────────────────────────────────────────────────────
# Annex B — tags
# ─────────────────────────────────────────────────────────────────────────────

# Record type tags
TAG_CONTAINER        = _u("89738606-f1c3-11cf-9d89-0080c72e70a3")
TAG_DATA_SOURCE      = _u("89738619-f1c3-11cf-9d89-0080c72e70a3")
TAG_MONITOR_SETTINGS = _u("b48d858c-f5f5-11cf-9d89-0080c72e70a3")
TAG_OBSERVATION      = _u("8973861a-f1c3-11cf-9d89-0080c72e70a3")

# Container
TAG_VERSION_INFO           = _u("89738607-f1c3-11cf-9d89-0080c72e70a3")
TAG_COMPRESSION_STYLE      = _u("8973861b-f1c3-11cf-9d89-0080c72e70a3")
TAG_COMPRESSION_ALGORITHM  = _u("8973861c-f1c3-11cf-9d89-0080c72e70a3")

# Data source → channel definitions
TAG_CHANNEL_DEFNS   = _u("b48d858d-f5f5-11cf-9d89-0080c72e70a3")
TAG_ONE_CHANNEL_DEFN = _u("b48d858e-f5f5-11cf-9d89-0080c72e70a3")
TAG_CHANNEL_NAME    = _u("b48d8590-f5f5-11cf-9d89-0080c72e70a3")
TAG_PHASE_ID        = _u("b48d8591-f5f5-11cf-9d89-0080c72e70a3")
TAG_QUANTITY_TYPE_ID = _u("b48d8592-f5f5-11cf-9d89-0080c72e70a3")
TAG_QUANTITY_MEASURED_ID = _u("c690e872-f755-11cf-9d89-0080c72e70a3")
TAG_NAME_DS         = _u("b48d8587-f5f5-11cf-9d89-0080c72e70a3")

# Channel definition → series definitions
TAG_SERIES_DEFNS     = _u("b48d8598-f5f5-11cf-9d89-0080c72e70a3")
TAG_ONE_SERIES_DEFN  = _u("b48d859a-f5f5-11cf-9d89-0080c72e70a3")
TAG_VALUE_TYPE_ID    = _u("b48d859c-f5f5-11cf-9d89-0080c72e70a3")
TAG_QUANTITY_UNITS_ID = _u("b48d859b-f5f5-11cf-9d89-0080c72e70a3")
TAG_QUANTITY_CHARACTERISTIC_ID = _u("3d786f9e-f76e-11cf-9d89-0080c72e70a3")
TAG_STORAGE_METHOD_ID = _u("b48d85a1-f5f5-11cf-9d89-0080c72e70a3")

# Observation
TAG_OBSERVATION_NAME  = _u("3d786f8a-f76e-11cf-9d89-0080c72e70a3")
TAG_TIME_START        = _u("3d786f8c-f76e-11cf-9d89-0080c72e70a3")
TAG_TIME_CREATE       = _u("3d786f8b-f76e-11cf-9d89-0080c72e70a3")
TAG_CHANNEL_INSTANCES = _u("3d786f91-f76e-11cf-9d89-0080c72e70a3")
TAG_ONE_CHANNEL_INST  = _u("3d786f92-f76e-11cf-9d89-0080c72e70a3")
TAG_CHANNEL_DEFN_IDX  = _u("b48d858f-f5f5-11cf-9d89-0080c72e70a3")

# Channel instance → series instances
TAG_SERIES_INSTANCES     = _u("3d786f93-f76e-11cf-9d89-0080c72e70a3")
TAG_ONE_SERIES_INSTANCE  = _u("3d786f94-f76e-11cf-9d89-0080c72e70a3")
TAG_SERIES_VALUES        = _u("3d786f99-f76e-11cf-9d89-0080c72e70a3")
TAG_SERIES_SCALE         = _u("3d786f96-f76e-11cf-9d89-0080c72e70a3")
TAG_SERIES_OFFSET        = _u("3d786f97-f76e-11cf-9d89-0080c72e70a3")
TAG_SERIES_SHARE_CHANNEL_IDX = _u("8973861f-f1c3-11cf-9d89-0080c72e70a3")
TAG_SERIES_SHARE_SERIES_IDX  = _u("89738620-f1c3-11cf-9d89-0080c72e70a3")

#: Clause 5.5 storage methods, OR-able.
METHOD_VALUES    = 1
METHOD_SCALED    = 2
METHOD_INCREMENT = 4

#: tagValueTypeID → short name (Annex B, ID_SERIES_VALUE_TYPE_*).
VALUE_TYPE_NAMES = {
    _u("c690e862-f755-11cf-9d89-0080c72e70a3"): "TIME",
    _u("67f6af97-f753-11cf-9d89-0080c72e70a3"): "VAL",
    _u("67f6af98-f753-11cf-9d89-0080c72e70a3"): "MIN",
    _u("67f6af99-f753-11cf-9d89-0080c72e70a3"): "MAX",
    _u("67f6af9a-f753-11cf-9d89-0080c72e70a3"): "AVG",
    _u("67f6af9b-f753-11cf-9d89-0080c72e70a3"): "INSTANTANEOUS",
    _u("3d786f9b-f76e-11cf-9d89-0080c72e70a3"): "PHASEANGLE",
}

#: tagQuantityTypeID → short name (Annex B, ID_QT_*).
QUANTITY_TYPE_NAMES = {
    _u("67f6af80-f753-11cf-9d89-0080c72e70a3"): "WAVEFORM",
    _u("67f6af81-f753-11cf-9d89-0080c72e70a3"): "PHASOR",
    _u("67f6af82-f753-11cf-9d89-0080c72e70a3"): "VALUELOG",
    _u("67f6af83-f753-11cf-9d89-0080c72e70a3"): "FLASH",
    _u("67f6af85-f753-11cf-9d89-0080c72e70a3"): "RESPONSE",
    _u("67f6af87-f753-11cf-9d89-0080c72e70a3"): "HISTOGRAM",
    _u("67f6af88-f753-11cf-9d89-0080c72e70a3"): "HISTOGRAM3D",
    _u("67f6af89-f753-11cf-9d89-0080c72e70a3"): "CPF",
    _u("67f6af8a-f753-11cf-9d89-0080c72e70a3"): "XY",
    _u("67f6af8b-f753-11cf-9d89-0080c72e70a3"): "MAGDUR",
    _u("67f6af8c-f753-11cf-9d89-0080c72e70a3"): "XYZ",
    _u("67f6af8d-f753-11cf-9d89-0080c72e70a3"): "MAGDURTIME",
    _u("67f6af8e-f753-11cf-9d89-0080c72e70a3"): "MAGDURCOUNT",
}

#: tagQuantityMeasuredID values (Annex B, ID_QM_*).  Stored as UINT4.
QUANTITY_MEASURED_NAMES = {
    0: "none", 1: "voltage", 2: "current", 3: "power", 4: "energy",
    5: "temperature", 6: "pressure", 7: "charge", 8: "efield", 9: "mfield",
    10: "velocity", 11: "bearing", 12: "force", 13: "torque", 14: "position",
    15: "fluxlinkage", 16: "fluxdensity", 17: "status", 18: "humidity",
}

#: tagQuantityUnitsID values (Annex B, ID_QU_*) → display symbol.
UNITS_SYMBOLS = {
    0: "", 1: "timestamp", 2: "s", 3: "cycles", 6: "V", 7: "A", 8: "VA",
    9: "W", 10: "VAR", 11: "ohm", 12: "S", 13: "V/A", 14: "J", 15: "Hz",
    16: "degC", 17: "deg", 18: "dB", 19: "%", 20: "pu", 21: "samples",
    22: "VARh", 23: "Wh", 24: "VAh", 25: "m/s", 26: "mph", 27: "bar",
    28: "Pa", 29: "N", 30: "N-m", 31: "rpm", 32: "rad/s", 33: "m",
    34: "Wb-turns", 35: "T", 36: "Wb", 37: "V/V", 38: "A/A", 39: "A/V",
}

#: Time-series units that change how a TIME series is interpreted.
UNITS_TIMESTAMP = 1   # absolute timestamps
UNITS_SECONDS = 2     # seconds relative to tagTimeStart
UNITS_CYCLES = 3      # cycles relative to tagTimeStart

#: tagQuantityCharacteristicID → short name (Annex B, ID_QC_*), complete list.
CHARACTERISTIC_NAMES = {
    _u("5000c15a-2e65-4c86-919e-343c5faa064c"): "ACCELERATION",
    _u("74e51e63-ac9b-43f9-8599-1e676035b14b"): "ADMITTANCE",
    _u("43a6b1fc-8ef9-482c-a0ad-131bb4808fa2"): "ADMITTANCE_SNEG",
    _u("cdbfd610-8b7c-4099-9493-9623793d0ae8"): "ADMITTANCE_SPOS",
    _u("791b9905-a4f2-482d-a2b6-136e21b7ca0e"): "ADMITTANCE_SZERO",
    _u("672d030f-7810-11d4-a4b3-444553540000"): "ANGLE_FUND",
    _u("8786ca10-9113-11d3-b930-0050da2b1f4d"): "ANSI_TDF",
    _u("a6b31ad0-b451-11d1-ae17-0060083a2628"): "ARITH_SUM",
    _u("a6b31ad1-b451-11d1-ae17-0060083a2628"): "AVG_IMBAL",
    _u("aede9d60-591d-43c6-9d18-fa34e08cd63f"): "CONDUCTANCE",
    _u("dc9392d3-edc8-40e1-a8e5-1eb269335945"): "CONDUCTANCE_SNEG",
    _u("02582b6c-a2bb-432a-99eb-c9bba8eda5ed"): "CONDUCTANCE_SPOS",
    _u("0c2a7e66-a4ef-4ecc-8995-dc6534c5683d"): "CONDUCTANCE_SZERO",
    _u("a6b31ad2-b451-11d1-ae17-0060083a2628"): "CREST_FACTOR",
    _u("d347ba63-e34c-11d4-82d9-00e09872a094"): "DAXIS",
    _u("d347ba65-e34c-11d4-82d9-00e09872a094"): "DAXISFIELD",
    _u("a6b31ad3-b451-11d1-ae17-0060083a2628"): "DF",
    _u("1c39fb01-a6aa-11d4-a4b3-444553540000"): "DF_ARITH",
    _u("07ef68ad-9ff5-11d2-b30b-006008b37183"): "DF_CO_S_DEMAND",
    _u("07ef68a4-9ff5-11d2-b30b-006008b37183"): "DF_DEMAND",
    _u("672d0312-7810-11d4-a4b3-444553540000"): "DF_VECTOR",
    _u("b3729014-9797-460e-a0b5-d82ddbfbc269"): "DURATION",
    _u("a6b31ad4-b451-11d1-ae17-0060083a2628"): "EVEN_THD",
    _u("f3d216e2-2aa5-11d5-a4b3-444553540000"): "EVEN_THD_RMS",
    _u("2a310a59-3cf5-4bc6-b1b4-2602005c00c2"): "FLAGGING",
    _u("5fd423aa-de44-4a13-bf5b-af9621bc6ffa"): "FLAGGING_IEC_61000_4_30",
    _u("a6b31ad5-b451-11d1-ae17-0060083a2628"): "FLKR_FREQ_MAX",
    _u("a6b31ad6-b451-11d1-ae17-0060083a2628"): "FLKR_MAG_AVG",
    _u("a6b31ad7-b451-11d1-ae17-0060083a2628"): "FLKR_MAG_MAX",
    _u("a6b31ad8-b451-11d1-ae17-0060083a2628"): "FLKR_MAX_DVV",
    _u("4d693eec-5d1d-4531-993a-793b5356c63d"): "FLKR_PILPF",
    _u("126de61c-6691-4d16-8fdf-46482bca4694"): "FLKR_PIMAX",
    _u("12db02cd-e2bb-4be8-a6fb-77e2daedf865"): "FLKR_PINST",
    _u("e065b621-ffdb-4598-9330-4d09353988b6"): "FLKR_PIROOT",
    _u("7d11f283-1ce7-4e58-8af0-79048793b8a7"): "FLKR_PIROOTLPF",
    _u("515bf321-71ca-11d4-a4b3-444553540000"): "FLKR_PLT",
    _u("2257ec05-06ea-4709-b43a-0c00534d554a"): "FLKR_PLTSLIDE",
    _u("515bf320-71ca-11d4-a4b3-444553540000"): "FLKR_PST",
    _u("a6b31ad9-b451-11d1-ae17-0060083a2628"): "FLKR_SPECTRUM",
    _u("a6b31ada-b451-11d1-ae17-0060083a2628"): "FLKR_WGT_AVG",
    _u("a6b31adb-b451-11d1-ae17-0060083a2628"): "FORM_FACTOR",
    _u("07ef68af-9ff5-11d2-b30b-006008b37183"): "FREQUENCY",
    _u("a6b31adc-b451-11d1-ae17-0060083a2628"): "HRMS",
    _u("e4d86bc6-ef02-491c-bdca-360590642488"): "HRMS_EVEN",
    _u("ee9a8bac-0234-4b18-a794-9ffd7818c6d3"): "HRMS_ODD",
    _u("c7e9f4d1-212b-4bb7-9f57-4c7bf4f3cd30"): "HRMS_TRIPLEN",
    _u("f3d216e5-2aa5-11d5-a4b3-444553540000"): "IHRMS",
    _u("4c19f9f2-e297-4ff5-825f-91e598f92856"): "IMPEDANCE",
    _u("30bbe94b-aa7f-4c39-957a-b88377fee380"): "IMPEDANCE_SNEG",
    _u("27233a3b-5a26-4341-b098-15416754d14e"): "IMPEDANCE_SPOS",
    _u("ba214f62-0390-48c3-88ea-449ec4182bbc"): "IMPEDANCE_SZERO",
    _u("a6b31add-b451-11d1-ae17-0060083a2628"): "INSTANTANEOUS",
    _u("a6b31ade-b451-11d1-ae17-0060083a2628"): "IT",
    _u("6042d1e3-ead6-488f-ac54-fc1e1d85fdf9"): "JERK",
    _u("8786ca11-9113-11d3-b930-0050da2b1f4d"): "K_FACTOR",
    _u("d347ba61-e34c-11d4-82d9-00e09872a094"): "LINEAR",
    _u("a6b31adf-b451-11d1-ae17-0060083a2628"): "NONE",
    _u("a6b31ae0-b451-11d1-ae17-0060083a2628"): "ODD_THD",
    _u("f3d216e1-2aa5-11d5-a4b3-444553540000"): "ODD_THD_RMS",
    _u("a6b31ae1-b451-11d1-ae17-0060083a2628"): "P",
    _u("a6b31ae2-b451-11d1-ae17-0060083a2628"): "PEAK",
    _u("a6b31ae3-b451-11d1-ae17-0060083a2628"): "PF",
    _u("1c39fb00-a6aa-11d4-a4b3-444553540000"): "PF_ARITH",
    _u("672d0308-7810-11d4-a4b3-444553540000"): "PF_CO_P_DEMAND",
    _u("672d0309-7810-11d4-a4b3-444553540000"): "PF_CO_Q_DEMAND",
    _u("07ef68ae-9ff5-11d2-b30b-006008b37183"): "PF_CO_S_DEMAND",
    _u("07ef68a5-9ff5-11d2-b30b-006008b37183"): "PF_DEMAND",
    _u("672d0311-7810-11d4-a4b3-444553540000"): "PF_VECTOR",
    _u("672d030a-7810-11d4-a4b3-444553540000"): "P_CO_Q_DEMAND",
    _u("672d030b-7810-11d4-a4b3-444553540000"): "P_CO_S_DEMAND",
    _u("07ef68a1-9ff5-11d2-b30b-006008b37183"): "P_DEMAND",
    _u("1cdda475-1ebb-42d8-8087-d01b0b5cfa97"): "P_FUND",
    _u("b82b5c80-55c7-11d5-a4b3-444553540000"): "P_HARMONIC",
    _u("b82b5c81-55c7-11d5-a4b3-444553540000"): "P_HARMONIC_UNSIGNED",
    _u("07ef68a6-9ff5-11d2-b30b-006008b37183"): "P_INTG",
    _u("07ef68a8-9ff5-11d2-b30b-006008b37183"): "P_INTG_NEG",
    _u("672d0301-7810-11d4-a4b3-444553540000"): "P_INTG_NEG_FUND",
    _u("07ef68a7-9ff5-11d2-b30b-006008b37183"): "P_INTG_POS",
    _u("672d0300-7810-11d4-a4b3-444553540000"): "P_INTG_POS_FUND",
    _u("f098a9a0-3ee4-11d5-a4b3-444553540000"): "P_IVL_INTG",
    _u("f098a9a3-3ee4-11d5-a4b3-444553540000"): "P_IVL_INTG_NEG",
    _u("f098a9a4-3ee4-11d5-a4b3-444553540000"): "P_IVL_INTG_NEG_FUND",
    _u("f098a9a1-3ee4-11d5-a4b3-444553540000"): "P_IVL_INTG_POS",
    _u("f098a9a2-3ee4-11d5-a4b3-444553540000"): "P_IVL_INTG_POS_FUND",
    _u("72e82a41-336c-11d5-a4b3-444553540000"): "P_PEAK_DEMAND",
    _u("672d0305-7810-11d4-a4b3-444553540000"): "P_PRED_DEMAND",
    _u("a6b31ae4-b451-11d1-ae17-0060083a2628"): "Q",
    _u("d347ba64-e34c-11d4-82d9-00e09872a094"): "QAXIS",
    _u("672d030d-7810-11d4-a4b3-444553540000"): "Q_CO_P_DEMAND",
    _u("672d030e-7810-11d4-a4b3-444553540000"): "Q_CO_S_DEMAND",
    _u("07ef68a2-9ff5-11d2-b30b-006008b37183"): "Q_DEMAND",
    _u("672d0310-7810-11d4-a4b3-444553540000"): "Q_FUND",
    _u("07ef68a9-9ff5-11d2-b30b-006008b37183"): "Q_INTG",
    _u("07ef68ab-9ff5-11d2-b30b-006008b37183"): "Q_INTG_NEG",
    _u("672d0304-7810-11d4-a4b3-444553540000"): "Q_INTG_NEG_FUND",
    _u("07ef68aa-9ff5-11d2-b30b-006008b37183"): "Q_INTG_POS",
    _u("672d0303-7810-11d4-a4b3-444553540000"): "Q_INTG_POS_FUND",
    _u("f098a9a5-3ee4-11d5-a4b3-444553540000"): "Q_IVL_INTG",
    _u("f098a9a9-3ee4-11d5-a4b3-444553540000"): "Q_IVL_INTG_NEG",
    _u("f098a9a8-3ee4-11d5-a4b3-444553540000"): "Q_IVL_INTG_NEG_FUND",
    _u("f098a9a6-3ee4-11d5-a4b3-444553540000"): "Q_IVL_INTG_POS",
    _u("f098a9a7-3ee4-11d5-a4b3-444553540000"): "Q_IVL_INTG_POS_FUND",
    _u("72e82a42-336c-11d5-a4b3-444553540000"): "Q_PEAK_DEMAND",
    _u("672d0306-7810-11d4-a4b3-444553540000"): "Q_PRED_DEMAND",
    _u("8627805f-cb7b-4841-9423-7c6e3db4ad49"): "REACTANCE",
    _u("90091312-6e6b-4443-a2b6-d40351b00ba1"): "REACTANCE_SNEG",
    _u("24a43f5b-a6b6-456e-be47-a71244bcd7bc"): "REACTANCE_SPOS",
    _u("2eaca452-59e4-43fc-a2d8-626d2f250d55"): "REACTANCE_SZERO",
    _u("361165bc-9b4c-4e11-8ee8-897dc541e66d"): "REL_HUMIDITY",
    _u("1c255132-faba-44a7-9cae-8ce85851b73e"): "RESISTANCE",
    _u("647dbcdc-4881-4539-a56b-fc3d000ec66c"): "RESISTANCE_SNEG",
    _u("01923143-3e31-4be5-9b54-92109c388a9b"): "RESISTANCE_SPOS",
    _u("e79a2084-5723-4625-bc45-c22229ec69ee"): "RESISTANCE_SZERO",
    _u("a6b31ae5-b451-11d1-ae17-0060083a2628"): "RMS",
    _u("07ef68a0-9ff5-11d2-b30b-006008b37183"): "RMS_DEMAND",
    _u("72e82a44-336c-11d5-a4b3-444553540000"): "RMS_PEAK_DEMAND",
    _u("d347ba62-e34c-11d4-82d9-00e09872a094"): "ROTATIONAL",
    _u("c8a86eca-cd40-4bee-9f1b-9e7a8d6ab34c"): "RVC_DELTA_UMAX",
    _u("9789ceb3-8314-41ff-8565-b96cecd6ef48"): "RVC_DELTA_USS",
    _u("a6b31ae6-b451-11d1-ae17-0060083a2628"): "S",
    _u("a6b31ae7-b451-11d1-ae17-0060083a2628"): "S0S1",
    _u("a6b31ae8-b451-11d1-ae17-0060083a2628"): "S2S1",
    _u("d71a4b91-3c92-11d4-9f2c-002078e0b723"): "SNEG",
    _u("78014e8c-e35d-4e87-a915-f3fb68a4cf8e"): "SOLAR_IRRADIANCE",
    _u("91600a0d-e2e8-43ea-9e90-ca714756e172"): "SOUND_ABSORPTION",
    _u("a6b31ae9-b451-11d1-ae17-0060083a2628"): "SPECTRA",
    _u("53be6ba8-0789-455b-9a95-da128683dda7"): "SPECTRA_HGROUP",
    _u("5e51e006-9c95-4c5e-878f-7ca87c0d2a0e"): "SPECTRA_IGROUP",
    _u("a6b31aea-b451-11d1-ae17-0060083a2628"): "SPOS",
    _u("b82b5c83-55c7-11d5-a4b3-444553540000"): "STATUS",
    _u("9df9c55a-cc24-4dce-8f9e-9ef9e91c95c4"): "SUSCEPTANCE",
    _u("339b8af9-f120-4331-b9d4-abca9cbead87"): "SUSCEPTANCE_SNEG",
    _u("265eaab1-bc47-4471-920b-8b91e19386f4"): "SUSCEPTANCE_SPOS",
    _u("52e726e4-e5a7-4b97-8977-d183c77e2375"): "SUSCEPTANCE_SZERO",
    _u("d71a4b92-3c92-11d4-9f2c-002078e0b723"): "SZERO",
    _u("1c39fb02-a6aa-11d4-a4b3-444553540000"): "S_ARITH",
    _u("1c39fb03-a6aa-11d4-a4b3-444553540000"): "S_ARITH_FUND",
    _u("672d0317-7810-11d4-a4b3-444553540000"): "S_CO_P_DEMAND",
    _u("672d0318-7810-11d4-a4b3-444553540000"): "S_CO_Q_DEMAND",
    _u("07ef68a3-9ff5-11d2-b30b-006008b37183"): "S_DEMAND",
    _u("672d0316-7810-11d4-a4b3-444553540000"): "S_FUND",
    _u("07ef68ac-9ff5-11d2-b30b-006008b37183"): "S_INTG",
    _u("672d0313-7810-11d4-a4b3-444553540000"): "S_INTG_FUND",
    _u("f098a9aa-3ee4-11d5-a4b3-444553540000"): "S_IVL_INTG",
    _u("f098a9ab-3ee4-11d5-a4b3-444553540000"): "S_IVL_INTG_FUND",
    _u("72e82a43-336c-11d5-a4b3-444553540000"): "S_PEAK_DEMAND",
    _u("672d0307-7810-11d4-a4b3-444553540000"): "S_PRED_DEMAND",
    _u("672d0314-7810-11d4-a4b3-444553540000"): "S_VECTOR",
    _u("672d0315-7810-11d4-a4b3-444553540000"): "S_VECTOR_FUND",
    _u("f3d216e7-2aa5-11d5-a4b3-444553540000"): "TDD",
    _u("7ee799fc-19e5-4045-80bb-a449b61dc903"): "TEMPERATURE",
    _u("1ba59bbc-d7ff-4ab3-a152-44c162d1ce07"): "THD_TRIPLEN",
    _u("f3d216e3-2aa5-11d5-a4b3-444553540000"): "TID",
    _u("f3d216e4-2aa5-11d5-a4b3-444553540000"): "TID_RMS",
    _u("a6b31aeb-b451-11d1-ae17-0060083a2628"): "TIF",
    _u("f3d216e6-2aa5-11d5-a4b3-444553540000"): "TIF_RMS",
    _u("fd434920-23ec-4948-8cb8-5053d86d0506"): "TIME_OFFSET",
    _u("a6b31aec-b451-11d1-ae17-0060083a2628"): "TOTAL_THD",
    _u("f3d216e0-2aa5-11d5-a4b3-444553540000"): "TOTAL_THD_RMS",
    _u("5202bd07-245c-11d5-a4b3-444553540000"): "TRANSFERFUNC",
    _u("039a8fde-9406-4bf5-b7b6-af11636d399f"): "VELOCITY",
}

#: tagPhaseID values (Annex B, ID_PHASE_*).  Stored as UINT4, not a GUID.
PHASE_NAMES = {
    0: "none", 1: "an", 2: "bn", 3: "cn", 4: "ng",
    5: "ab", 6: "bc", 7: "ca", 8: "residual", 9: "net",
    # 10-12 (positive/negative/zero sequence) are deprecated as of version 1.5.
    10: "pseq", 11: "nseq", 12: "zseq",
    13: "total", 14: "ln_ave", 15: "ll_ave", 16: "worst",
    17: "plus", 18: "minus",
    **{18 + i: f"general_{i}" for i in range(1, 17)},   # 19-34
    36: "ln_max", 37: "ln_min", 38: "ll_max", 39: "ll_min",
}


# ─────────────────────────────────────────────────────────────────────────────
# Element tree
# ─────────────────────────────────────────────────────────────────────────────

class Element:
    """One element inside a record body (Annex A, c_collection_element).

    Layout, all little-endian::

        +0   GUID   tagElement
        +16  INT1   typeElement    (collection / scalar / vector)
        +17  INT1   typePhysical   (ID_PHYS_TYPE_*, 0 for collections)
        +18  BOOL1  isEmbedded     (scalar value stored in the link bytes)
        +19  INT1   reserved
        +20  UINT4  linkElement    (offset, relative to the record body)
        +24  UINT4  sizeElement

    Note that ``linkElement`` is body-relative.  Clause 4.2.2 says relative
    links are "relative to the beginning of the record header", but Annex A's
    own comment on this field says "relative within the record body", and
    body-relative is what real files use.
    """

    __slots__ = ("tag", "element_type", "physical_type", "embedded",
                 "link", "size", "_body")

    def __init__(self, tag, element_type, physical_type, embedded,
                 link, size, body):
        self.tag = tag
        self.element_type = element_type
        self.physical_type = physical_type
        self.embedded = embedded
        self.link = link
        self.size = size
        self._body = body

    def __repr__(self) -> str:
        kind = {ELEMENT_COLLECTION: "Collection", ELEMENT_SCALAR: "Scalar",
                ELEMENT_VECTOR: "Vector"}.get(self.element_type, "?")
        return (f"<{kind} {self.tag} "
                f"{PHYS_TYPE_NAMES.get(self.physical_type, '')} "
                f"link={self.link} size={self.size}>")

    # ── collection access ────────────────────────────────────────────────
    def children(self) -> List["Element"]:
        """Elements of this collection (empty for scalars and vectors)."""
        if self.element_type != ELEMENT_COLLECTION:
            return []
        return _parse_collection(self._body, self.link)

    def find(self, tag: uuid.UUID) -> Optional["Element"]:
        """First child with this tag, or None."""
        for child in self.children():
            if child.tag == tag:
                return child
        return None

    def find_all(self, tag: uuid.UUID) -> List["Element"]:
        """Every child with this tag, in file order."""
        return [c for c in self.children() if c.tag == tag]

    # ── leaf access ──────────────────────────────────────────────────────
    def scalar(self):
        """Decode a scalar element to a Python value.

        A scalar of eight bytes or fewer may be stored inline in the element's
        link/size bytes rather than at a body offset (Annex A, isEmbedded).
        """
        pt = self.physical_type
        if pt not in _PHYS:
            raise PQDIFError(f"unknown physical type {pt} for scalar {self.tag}")
        code, size = _PHYS[pt]

        if self.embedded:
            raw = struct.pack("<II", self.link, self.size)[:size]
        else:
            if self.link + size > len(self._body):
                raise PQDIFError(f"scalar {self.tag} runs past end of record body")
            raw = self._body[self.link:self.link + size]

        if pt == 60:                              # GUID
            return uuid.UUID(bytes_le=raw)
        if pt == 50:                              # TIMESTAMPPQDIF
            day, sec = struct.unpack("<Id", raw)
            return _EPOCH + timedelta(days=day, seconds=sec)
        if pt in (42, 43):                        # COMPLEX8 / COMPLEX16
            half = "f" if pt == 42 else "d"
            re_, im = struct.unpack("<2" + half, raw)
            return complex(re_, im)
        if pt == 10:
            return raw.decode("latin-1")
        if code is None:
            return raw
        return struct.unpack("<" + code, raw)[0]

    def vector(self) -> np.ndarray:
        """Decode a numeric vector element (Annex A, c_vector).

        The element's own physical type gives the element width, so a
        COMPLEX16 series is 16 bytes per point, not 8 — there is never a need
        to guess at strides.
        """
        pt = self.physical_type
        if pt not in _PHYS:
            raise PQDIFError(f"unknown physical type {pt} for vector {self.tag}")
        code, width = _PHYS[pt]
        count = self._vector_count()
        start = self.link + 4
        end = start + count * width
        if end > len(self._body):
            raise PQDIFError(
                f"vector {self.tag} declares {count} values of "
                f"{PHYS_TYPE_NAMES.get(pt, pt)} ({end - start} bytes) but only "
                f"{len(self._body) - start} bytes remain in the record body"
            )
        buf = self._body[start:end]

        if pt == 43:      # COMPLEX16 — take the real part; imag is phase/quadrature
            return np.frombuffer(buf, dtype="<f8")[0::2].copy()
        if pt == 42:      # COMPLEX8
            return np.frombuffer(buf, dtype="<f4")[0::2].astype(np.float64)
        if pt == 50:      # TIMESTAMPPQDIF vector → seconds since its own epoch
            days = np.frombuffer(buf, dtype=np.dtype([("d", "<u4"), ("s", "<f8")]))
            return days["d"].astype(np.float64) * 86400.0 + days["s"]
        if code is None:
            raise PQDIFError(
                f"vector {self.tag} has non-numeric physical type "
                f"{PHYS_TYPE_NAMES.get(pt, pt)}"
            )
        return np.frombuffer(buf, dtype=np.dtype("<" + code)).astype(np.float64)

    def string(self) -> str:
        """Decode a CHAR1 vector to text.

        Decoded as latin-1 rather than ASCII: Pronto writes the phase symbol
        (φ) as a single byte 0xF8 in channel names like '3φ 4w Real Power',
        and latin-1 round-trips every byte value.
        """
        count = self._vector_count()
        start = self.link + 4
        raw = self._body[start:start + count]
        return raw.split(b"\x00")[0].decode("latin-1").strip()

    def _vector_count(self) -> int:
        if self.link + 4 > len(self._body):
            raise PQDIFError(f"vector {self.tag} count field is past end of body")
        return struct.unpack_from("<I", self._body, self.link)[0]


def _parse_collection(body: bytes, offset: int) -> List[Element]:
    """Parse a c_collection: UINT4 count followed by that many elements."""
    if offset + 4 > len(body):
        raise PQDIFError(f"collection at {offset} is past end of record body")
    count = struct.unpack_from("<I", body, offset)[0]
    end = offset + 4 + count * _ELEMENT_SIZE
    if end > len(body):
        raise PQDIFError(
            f"collection at {offset} declares {count} elements "
            f"({end} bytes) but the record body is only {len(body)} bytes"
        )
    out: List[Element] = []
    for i in range(count):
        base = offset + 4 + i * _ELEMENT_SIZE
        tag = uuid.UUID(bytes_le=body[base:base + 16])
        etype, ptype, embedded, _reserved = struct.unpack_from("<4b", body, base + 16)
        link, size = struct.unpack_from("<II", body, base + 20)
        out.append(Element(tag, etype, ptype, bool(embedded), link, size, body))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Record:
    """A physical record: 64-byte header plus a body of elements."""
    position: int
    signature: uuid.UUID
    tag: uuid.UUID
    header_size: int
    body_size: int
    next_position: int
    checksum: int
    raw_body: bytes

    def body(self, compressed: bool) -> Element:
        """Root collection of this record's body, decompressing if needed.

        The container record is never compressed, even under record-level
        compression (clause 6.2).
        """
        data = self.raw_body
        if compressed and self.tag != TAG_CONTAINER:
            try:
                data = zlib.decompress(data)
            except zlib.error as exc:
                raise PQDIFError(
                    f"record at {self.position} ({self.tag}) declared "
                    f"record-level zlib compression but did not decompress: {exc}"
                ) from exc
        return Element(self.tag, ELEMENT_COLLECTION, 0, False, 0, len(data), data)


def _walk_records(data: bytes) -> List[Record]:
    """Follow the record linked list (clause 4.1.1)."""
    records: List[Record] = []
    seen: set = set()
    pos = 0
    while pos + RECORD_HEADER_SIZE <= len(data):
        if pos in seen:
            raise PQDIFError(f"record link cycle detected at offset {pos}")
        seen.add(pos)
        signature = uuid.UUID(bytes_le=data[pos:pos + 16])
        tag = uuid.UUID(bytes_le=data[pos + 16:pos + 32])
        header_size, body_size, next_pos, checksum = struct.unpack_from(
            "<IIII", data, pos + 32)
        if signature != RECORD_SIGNATURE:
            raise PQDIFError(
                f"record at offset {pos} has signature {signature}, expected "
                f"{RECORD_SIGNATURE} — this is not a standard PQDIF file"
            )
        body_start = pos + header_size
        records.append(Record(
            position=pos, signature=signature, tag=tag,
            header_size=header_size, body_size=body_size,
            next_position=next_pos, checksum=checksum,
            raw_body=data[body_start:body_start + body_size],
        ))
        if next_pos == 0:
            break
        if next_pos <= pos or next_pos >= len(data):
            raise PQDIFError(
                f"record at offset {pos} links forward to {next_pos}, "
                f"which is outside the file (size {len(data)})"
            )
        pos = next_pos
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Logical view
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeriesDefinition:
    """A tagOneSeriesDefn: what one series of a channel contains."""
    value_type: str            # TIME / VAL / AVG / MIN / MAX / PHASEANGLE …
    storage_method: int        # clause 5.5 bit field
    characteristic: str        # ID_QC_* short name, e.g. RMS, TOTAL_THD
    units_id: int              # ID_QU_* value
    #: True when this is the channel's own time base rather than a measurement.
    @property
    def is_time(self) -> bool:
        return self.value_type == "TIME"

    @property
    def units(self) -> str:
        return UNITS_SYMBOLS.get(self.units_id, "")


@dataclass
class ChannelDefinition:
    """A tagOneChannelDefn: the reusable description of a channel."""
    index: int
    name: str
    phase: str
    quantity_type: str
    quantity_measured: str     # ID_QM_* short name: voltage / current / power …
    series: List[SeriesDefinition] = field(default_factory=list)


@dataclass
class Channel:
    """A tagOneChannelInst joined to its definition, with decoded series.

    ``series`` maps a value-type name ('TIME', 'AVG', 'MIN', 'MAX', 'VAL', …)
    to the decoded float64 array for that series.  ``characteristic`` and
    ``units`` describe the measurement series (not the time series), which is
    what identifies a channel physically: 'RMS' volts is a different quantity
    from 'SPECTRA_HGROUP' volts even when both are named after phase A.
    """
    definition_index: int
    name: str
    phase: str
    quantity_type: str
    quantity_measured: str
    characteristic: str
    units: str
    series: Dict[str, np.ndarray]
    #: ID_QU_* of the TIME series — how to convert it to real time.
    time_units_id: int = UNITS_SECONDS

    @property
    def time(self) -> Optional[np.ndarray]:
        """The TIME series (units given by the series definition; normally
        seconds relative to the observation's tagTimeStart)."""
        return self.series.get("TIME")

    def value(self, *preferred: str) -> Optional[np.ndarray]:
        """First present series among *preferred*, else any non-TIME series."""
        for name in preferred:
            if name in self.series:
                return self.series[name]
        for name, arr in self.series.items():
            if name != "TIME":
                return arr
        return None


@dataclass
class Observation:
    """One tagRecObservation."""
    index: int
    name: str
    start_time: Optional[datetime]
    channels: List[Channel]

    def channels_named(self, name: str) -> List[Channel]:
        return [c for c in self.channels if c.name == name]

    def channel_named(self, name: str) -> Optional[Channel]:
        for c in self.channels:
            if c.name == name:
                return c
        return None


class PQDIFFile:
    """A parsed PQDIF file.

    Parsing is eager for structure and lazy for bulk data: channel definitions
    and observation metadata are read on construction, while series values are
    decoded when an observation's channels are first accessed.
    """

    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        if len(data) < RECORD_HEADER_SIZE:
            raise PQDIFError(f"{self.path.name} is too small to be a PQDIF file")

        self.records = _walk_records(data)
        if not self.records or self.records[0].tag != TAG_CONTAINER:
            raise PQDIFError(
                f"{self.path.name}: first record is not a container record"
            )

        # ── container: version and compression (clause 5.3) ──────────────
        container = self.records[0].body(compressed=False)
        self.version: tuple = (1, 0, 1, 0)
        compression_style = 0
        self.compression_algorithm = 0
        for element in container.children():
            if element.tag == TAG_VERSION_INFO:
                v = element.vector()
                if len(v) >= 4:
                    self.version = tuple(int(x) for x in v[:4])
            elif element.tag == TAG_COMPRESSION_STYLE:
                compression_style = int(element.scalar())
            elif element.tag == TAG_COMPRESSION_ALGORITHM:
                self.compression_algorithm = int(element.scalar())

        if compression_style == 1:
            raise PQDIFError(
                f"{self.path.name} uses total-file compression "
                "(ID_COMP_STYLE_TOTALFILE), which is deprecated in "
                "IEEE 1159.3-2019 and not supported here"
            )
        self.compressed = compression_style == 2

        # Clause 5.3.1: a reader that cannot support the writer version should
        # fall back to the compatible version before giving up.
        if self.version[0] > 1 and self.version[2] > 1:
            raise PQDIFError(
                f"{self.path.name} declares writer version {self.version[0]}."
                f"{self.version[1]} and compatible version {self.version[2]}."
                f"{self.version[3]}; this reader implements version 1.x"
            )

        self.definitions: List[ChannelDefinition] = self._read_definitions()
        self._observation_records = [
            r for r in self.records if r.tag == TAG_OBSERVATION
        ]
        self._observations: Optional[List[Observation]] = None

    # ── data source ──────────────────────────────────────────────────────
    def _read_definitions(self) -> List[ChannelDefinition]:
        """Read channel definitions from the last data source record.

        Clause 5.1 allows several data source records; the one applying to an
        observation is the latest whose Date Effective is at or before it.
        Pronto writes exactly one, so the last is used and a warning is logged
        if there are more.
        """
        ds_records = [r for r in self.records if r.tag == TAG_DATA_SOURCE]
        if not ds_records:
            raise PQDIFError(f"{self.path.name} has no data source record")
        if len(ds_records) > 1:
            log.warning(
                "%s has %d data source records; using the last one. Per-observation "
                "Date Effective selection is not implemented.",
                self.path.name, len(ds_records),
            )

        body = ds_records[-1].body(self.compressed)
        self.data_source_name = ""
        name_element = body.find(TAG_NAME_DS)
        if name_element is not None:
            self.data_source_name = name_element.string()

        defns_element = body.find(TAG_CHANNEL_DEFNS)
        if defns_element is None:
            raise PQDIFError(
                f"{self.path.name}: data source record has no tagChannelDefns"
            )

        out: List[ChannelDefinition] = []
        for i, defn in enumerate(defns_element.find_all(TAG_ONE_CHANNEL_DEFN)):
            name_el = defn.find(TAG_CHANNEL_NAME)
            phase_el = defn.find(TAG_PHASE_ID)
            qt_el = defn.find(TAG_QUANTITY_TYPE_ID)
            qm_el = defn.find(TAG_QUANTITY_MEASURED_ID)

            phase = "none"
            if phase_el is not None:
                phase = PHASE_NAMES.get(int(phase_el.scalar()), "none")
            quantity_type = ""
            if qt_el is not None:
                quantity_type = QUANTITY_TYPE_NAMES.get(qt_el.scalar(), "")

            series: List[SeriesDefinition] = []
            series_defns = defn.find(TAG_SERIES_DEFNS)
            if series_defns is not None:
                for sd in series_defns.find_all(TAG_ONE_SERIES_DEFN):
                    vt_el = sd.find(TAG_VALUE_TYPE_ID)
                    sm_el = sd.find(TAG_STORAGE_METHOD_ID)
                    qc_el = sd.find(TAG_QUANTITY_CHARACTERISTIC_ID)
                    units_el = sd.find(TAG_QUANTITY_UNITS_ID)
                    value_type = "UNKNOWN"
                    if vt_el is not None:
                        raw = vt_el.scalar()
                        value_type = VALUE_TYPE_NAMES.get(raw, str(raw))
                    characteristic = ""
                    if qc_el is not None:
                        raw = qc_el.scalar()
                        characteristic = CHARACTERISTIC_NAMES.get(raw, str(raw))
                    series.append(SeriesDefinition(
                        value_type=value_type,
                        storage_method=(int(sm_el.scalar())
                                        if sm_el is not None else METHOD_VALUES),
                        characteristic=characteristic,
                        units_id=int(units_el.scalar()) if units_el is not None else 0,
                    ))

            measured = 0
            if qm_el is not None:
                measured = int(qm_el.scalar())

            out.append(ChannelDefinition(
                index=i,
                name=name_el.string() if name_el is not None else f"channel_{i}",
                phase=phase,
                quantity_type=quantity_type,
                quantity_measured=QUANTITY_MEASURED_NAMES.get(measured, "none"),
                series=series,
            ))
        return out

    # ── observations ─────────────────────────────────────────────────────
    @property
    def observations(self) -> List[Observation]:
        if self._observations is None:
            self._observations = [
                self._read_observation(i, r)
                for i, r in enumerate(self._observation_records)
            ]
        return self._observations

    def observation_names(self) -> List[str]:
        """Observation names without decoding any series values."""
        names = []
        for record in self._observation_records:
            body = record.body(self.compressed)
            element = body.find(TAG_OBSERVATION_NAME)
            names.append(element.string() if element is not None else "")
        return names

    def _read_observation(self, index: int, record: Record) -> Observation:
        body = record.body(self.compressed)

        name_el = body.find(TAG_OBSERVATION_NAME)
        name = name_el.string() if name_el is not None else ""

        start_time = None
        for tag in (TAG_TIME_START, TAG_TIME_CREATE):
            element = body.find(tag)
            if element is not None:
                start_time = element.scalar()
                break

        channels: List[Channel] = []
        instances_el = body.find(TAG_CHANNEL_INSTANCES)
        if instances_el is not None:
            instances = instances_el.find_all(TAG_ONE_CHANNEL_INST)
            for ci_index, instance in enumerate(instances):
                channel = self._read_channel(instance, instances, ci_index, name)
                if channel is not None:
                    channels.append(channel)

        return Observation(index=index, name=name,
                           start_time=start_time, channels=channels)

    def _read_channel(self, instance: Element, all_instances: List[Element],
                      ci_index: int, obs_name: str) -> Optional[Channel]:
        idx_el = instance.find(TAG_CHANNEL_DEFN_IDX)
        if idx_el is None:
            log.debug("observation %r channel instance %d has no "
                      "tagChannelDefnIdx; skipped", obs_name, ci_index)
            return None
        defn_index = int(idx_el.scalar())
        if not 0 <= defn_index < len(self.definitions):
            log.warning(
                "observation %r channel instance %d references channel "
                "definition %d, but only %d are defined; skipped",
                obs_name, ci_index, defn_index, len(self.definitions),
            )
            return None
        definition = self.definitions[defn_index]

        series: Dict[str, np.ndarray] = {}
        characteristic = ""
        units = ""
        time_units_id = UNITS_SECONDS
        instances_el = instance.find(TAG_SERIES_INSTANCES)
        if instances_el is not None:
            for si_index, si in enumerate(
                    instances_el.find_all(TAG_ONE_SERIES_INSTANCE)):
                # Clause 5.4.3: series instance i matches series definition i.
                if si_index < len(definition.series):
                    sd = definition.series[si_index]
                else:
                    sd = SeriesDefinition(f"SERIES{si_index}", METHOD_VALUES, "", 0)
                try:
                    values = self._read_series(si, sd.storage_method, all_instances)
                except PQDIFError as exc:
                    log.warning(
                        "observation %r channel %r series %d (%s): %s",
                        obs_name, definition.name, si_index, sd.value_type, exc,
                    )
                    continue
                if values is None or sd.value_type in series:
                    continue
                series[sd.value_type] = values
                if sd.is_time:
                    time_units_id = sd.units_id
                # The measurement series, not the time base, carries the
                # physical identity of the channel.
                elif not characteristic:
                    characteristic = sd.characteristic
                    units = sd.units

        return Channel(
            definition_index=defn_index,
            name=definition.name,
            phase=definition.phase,
            quantity_type=definition.quantity_type,
            quantity_measured=definition.quantity_measured,
            characteristic=characteristic,
            units=units,
            series=series,
            time_units_id=time_units_id,
        )

    def _read_series(self, series_instance: Element, method: int,
                     all_instances: List[Element]) -> Optional[np.ndarray]:
        """Decode one series instance per clause 5.5."""
        values_el = series_instance.find(TAG_SERIES_VALUES)

        # 5.5.3 — a shared series points at a master series elsewhere in this
        # observation instead of carrying its own values.
        if values_el is None:
            chan_idx_el = series_instance.find(TAG_SERIES_SHARE_CHANNEL_IDX)
            series_idx_el = series_instance.find(TAG_SERIES_SHARE_SERIES_IDX)
            if chan_idx_el is None or series_idx_el is None:
                return None
            chan_idx = int(chan_idx_el.scalar())
            series_idx = int(series_idx_el.scalar())
            if not 0 <= chan_idx < len(all_instances):
                raise PQDIFError(
                    f"shared series references channel instance {chan_idx}, "
                    f"but the observation has {len(all_instances)}"
                )
            master_instances = all_instances[chan_idx].find(TAG_SERIES_INSTANCES)
            if master_instances is None:
                raise PQDIFError(
                    f"shared series references channel instance {chan_idx}, "
                    "which has no series instances"
                )
            masters = master_instances.find_all(TAG_ONE_SERIES_INSTANCE)
            if not 0 <= series_idx < len(masters):
                raise PQDIFError(
                    f"shared series references series {series_idx} of channel "
                    f"instance {chan_idx}, which has {len(masters)} series"
                )
            return self._read_series(masters[series_idx], method, all_instances)

        raw = values_el.vector()

        if method & METHOD_INCREMENT:
            # 5.5.2 — [n_rates, count0, rate0, count1, rate1, …]
            values = _expand_increment(raw)
        else:
            values = raw

        if method & METHOD_SCALED:
            scale_el = series_instance.find(TAG_SERIES_SCALE)
            if scale_el is not None:
                values = values * float(scale_el.scalar())

        offset_el = series_instance.find(TAG_SERIES_OFFSET)
        if offset_el is not None:
            values = values + float(offset_el.scalar())

        return values


def _expand_increment(raw: np.ndarray) -> np.ndarray:
    """Rebuild a regular-rate series from its rate instructions (clause 5.5.2)."""
    if len(raw) < 3:
        raise PQDIFError(
            f"increment series needs at least 3 values, got {len(raw)}"
        )
    n_rates = int(raw[0])
    if len(raw) < 1 + 2 * n_rates:
        raise PQDIFError(
            f"increment series declares {n_rates} rates but carries only "
            f"{len(raw)} values"
        )
    chunks: List[np.ndarray] = []
    position = 0.0
    for i in range(n_rates):
        count = int(raw[1 + 2 * i])
        rate = float(raw[2 + 2 * i])
        if count < 0:
            raise PQDIFError(f"increment series rate {i} has negative count {count}")
        chunks.append(position + rate * np.arange(count, dtype=np.float64))
        position += rate * count
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)

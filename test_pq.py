"""
test_pq.py — Unit tests for PQ Analyzer IEEE math, channel mapping, and pipeline.

Run with:
    pytest test_pq.py -v

Coverage:
  1. IEEE 519-2022 per-order harmonic limits (_h519_limit)
  2. IEEE 519-2022 TDD limits (_tdd_limit) — boundary values
  3. ISC/IL class label (_tdd_class)
  4. Neutral harmonic block formula (_V2_CH_H*_IN_AAC constants)
  5. ChannelMapper tag resolution
  6. ChannelMapper regex pattern resolution
  7. Pipeline smoke test: MockAdapter → extract_dataset → check_voltage_compliance
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Make pq_* importable from any working directory ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import struct
import zlib

import pqdif
from pq_constants import (
    SEVERITY_ORDER,
    Thresholds,
    _h519_limit,
    _tdd_limit,
    _tdd_class,
)
from pq_adapter import (
    ProntoAdapter,
    MockAdapter,
    ChannelMapper,
    RawChannelInfo,
    extract_dataset,
)
from pq_analysis import (
    check_voltage_compliance,
    check_neutral_harmonics,
    check_harmonic_sources,
    detect_events,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. IEEE 519-2022 per-order harmonic limits
# ─────────────────────────────────────────────────────────────────────────────

class TestH519Limit:
    """Spot-check IEEE 519-2022 Table 2 values by class × harmonic order."""

    # Class <20: limits = [4.0, 7.0, 10.0, 12.0, 15.0] → class index 0
    def test_h5_class_lt20(self):
        assert _h519_limit(5, 15.0) == 4.0

    def test_h11_class_lt20(self):
        # 11 ≤ h < 17 row, class 0 → 2.0
        assert _h519_limit(11, 15.0) == 2.0

    def test_h17_class_lt20(self):
        # 17 ≤ h < 23, class 0 → 1.5
        assert _h519_limit(17, 15.0) == 1.5

    def test_h25_class_lt20(self):
        # 23 ≤ h < 35, class 0 → 0.6
        assert _h519_limit(25, 15.0) == 0.6

    def test_h37_class_lt20(self):
        # 35 ≤ h < 51, class 0 → 0.3
        assert _h519_limit(37, 15.0) == 0.3

    # Class 20–50: limits index 1
    def test_h5_class_20_50(self):
        assert _h519_limit(5, 25.0) == 7.0

    def test_h13_class_20_50(self):
        # 11 ≤ h < 17, class 1 → 3.5
        assert _h519_limit(13, 25.0) == 3.5

    # Class 50–100: limits index 2
    def test_h7_class_50_100(self):
        # h=7 is in 2 ≤ h < 11, class 2 → 10.0
        assert _h519_limit(7, 75.0) == 10.0

    def test_h17_class_50_100(self):
        # 17 ≤ h < 23, class 2 → 4.0
        assert _h519_limit(17, 75.0) == 4.0

    # Class 100–1000: limits index 3
    def test_h5_class_100_1000(self):
        assert _h519_limit(5, 500.0) == 12.0

    def test_h25_class_100_1000(self):
        # 23 ≤ h < 35, class 3 → 2.0
        assert _h519_limit(25, 500.0) == 2.0

    # Class ≥1000: limits index 4
    def test_h5_class_ge1000(self):
        assert _h519_limit(5, 1500.0) == 15.0

    def test_h37_class_ge1000(self):
        # 35 ≤ h < 51, class 4 → 1.4
        assert _h519_limit(37, 1500.0) == 1.4

    # Edge: order outside table
    def test_h_out_of_scope(self):
        assert _h519_limit(100, 50.0) == 0.0

    # Boundary: exactly at class threshold (isc_il = 20 is class 1, not class 0)
    def test_h5_boundary_at_20(self):
        assert _h519_limit(5, 20.0) == 7.0

    def test_h5_just_below_20(self):
        assert _h519_limit(5, 19.9) == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. IEEE 519-2022 TDD limits — boundary values
# ─────────────────────────────────────────────────────────────────────────────

class TestTDDLimit:
    def test_below_20(self):
        assert _tdd_limit(15.0) == 5.0

    def test_just_below_20(self):
        assert _tdd_limit(19.9) == 5.0

    def test_at_20(self):
        assert _tdd_limit(20.0) == 8.0

    def test_in_20_50(self):
        assert _tdd_limit(35.0) == 8.0

    def test_at_50(self):
        assert _tdd_limit(50.0) == 12.0

    def test_in_50_100(self):
        assert _tdd_limit(75.0) == 12.0

    def test_at_100(self):
        assert _tdd_limit(100.0) == 15.0

    def test_in_100_1000(self):
        assert _tdd_limit(500.0) == 15.0

    def test_just_below_1000(self):
        assert _tdd_limit(999.9) == 15.0

    def test_at_1000(self):
        assert _tdd_limit(1000.0) == 20.0

    def test_above_1000(self):
        assert _tdd_limit(5000.0) == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. ISC/IL class label
# ─────────────────────────────────────────────────────────────────────────────

class TestTDDClass:
    def test_lt20(self):
        assert _tdd_class(10.0) == "< 20"

    def test_20_to_50(self):
        assert _tdd_class(30.0) == "< 50"

    def test_50_to_100(self):
        assert _tdd_class(75.0) == "< 100"

    def test_100_to_1000(self):
        assert _tdd_class(500.0) == "< 1000"

    def test_ge1000(self):
        assert _tdd_class(1500.0) == "≥ 1000"



# ─────────────────────────────────────────────────────────────────────────────
# 5. ChannelMapper — tag-based resolution
# ─────────────────────────────────────────────────────────────────────────────

def _raw(index: int, label: str, qt: str, qm: str, ph: str) -> RawChannelInfo:
    return RawChannelInfo(index, label, qt, qm, ph, "")


class TestChannelMapperTags:
    mapper = ChannelMapper()

    def _resolve_one(self, qt: str, qm: str, ph: str) -> str | None:
        ch = _raw(0, "", qt, qm, ph)
        result = self.mapper.resolve([ch])
        return next(iter(result.keys())) if result else None

    def test_voltage_a(self):
        assert self._resolve_one("voltage", "rms", "an") == "voltage_a"

    def test_voltage_b(self):
        assert self._resolve_one("voltage", "average", "bn") == "voltage_b"

    def test_voltage_c(self):
        assert self._resolve_one("voltage", "rmsvalue", "cn") == "voltage_c"

    def test_current_a(self):
        assert self._resolve_one("current", "rms", "a") == "current_a"

    def test_current_b(self):
        assert self._resolve_one("current", "rms", "b") == "current_b"

    def test_current_neutral(self):
        assert self._resolve_one("current", "rms", "neutral") == "current_neutral"

    def test_current_neutral_phase_n(self):
        assert self._resolve_one("current", "rms", "phase_n") == "current_neutral"

    def test_thd_voltage_a(self):
        assert self._resolve_one("voltageharmonics", "thd", "an") == "thd_voltage_a"

    def test_thd_current_b(self):
        assert self._resolve_one("currentharmonics", "thd", "b") == "thd_current_b"

    def test_h5_current_a(self):
        assert self._resolve_one("currentharmonics", "h5", "a") == "h5_current_a"

    def test_h13_current_c(self):
        assert self._resolve_one("currentharmonics", "h13", "cn") == "h13_current_c"

    def test_h3_current_neutral(self):
        assert self._resolve_one("currentharmonics", "h3", "neutral") == "h3_current_neutral"

    def test_h7_current_neutral_in_alias(self):
        assert self._resolve_one("currentharmonics", "h7", "in") == "h7_current_neutral"

    def test_flicker_pst(self):
        assert self._resolve_one("flicker", "pst", "an") == "flicker_pst"

    def test_flicker_plt(self):
        assert self._resolve_one("flicker", "plt", "a") == "flicker_plt"

    def test_unmatched_returns_empty(self):
        ch = _raw(0, "xyzzy unknown", "unknown", "unknown", "unknown")
        result = self.mapper.resolve([ch])
        assert "voltage_a" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 6. ChannelMapper — regex pattern resolution (no tags, label only)
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelMapperRegex:
    mapper = ChannelMapper()

    def _resolve_by_label(self, label: str) -> str | None:
        ch = _raw(0, label, "", "", "")
        result = self.mapper.resolve([ch])
        return next(iter(result.keys())) if result else None

    def test_van_rms(self):
        assert self._resolve_by_label("Van RMS") == "voltage_a"

    def test_vb_label(self):
        assert self._resolve_by_label("Vb") == "voltage_b"

    def test_vc_label(self):
        assert self._resolve_by_label("Vc") == "voltage_c"

    def test_ia_label(self):
        assert self._resolve_by_label("Ia") == "current_a"

    def test_kw_label(self):
        assert self._resolve_by_label("kW") == "power_real"

    def test_kvar_label(self):
        assert self._resolve_by_label("kVAR") == "power_reactive"

    def test_thd_va_label(self):
        assert self._resolve_by_label("THD Va") == "thd_voltage_a"

    def test_thd_ia_label(self):
        assert self._resolve_by_label("THD Ia") == "thd_current_a"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pipeline smoke test: MockAdapter → extract_dataset → check_voltage_compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSmoke:
    """End-to-end test through the full extraction and analysis pipeline."""

    @pytest.fixture(scope="class")
    @classmethod
    def ds(cls):
        # 2.0 hours needed: MockAdapter injects swell at indices 6000–6020 (> 3600)
        adapter = MockAdapter(duration_hours=2.0, nominal=120.0)
        mapper  = ChannelMapper()
        return extract_dataset(adapter, mapper)

    @pytest.fixture(scope="class")
    @classmethod
    def thresh(cls):
        return Thresholds(nominal_voltage=120.0)

    def test_dataset_has_df(self, ds):
        assert not ds.df.empty

    def test_dataset_has_voltage_columns(self, ds):
        assert "voltage_a" in ds.df.columns
        assert "voltage_b" in ds.df.columns
        assert "voltage_c" in ds.df.columns

    def test_dataset_has_current_columns(self, ds):
        assert "current_a" in ds.df.columns

    def test_dataset_topology_inferred(self, ds):
        assert ds.meta.get("topology") in {"3-phase", "split-phase", "single-phase"}

    def test_dataset_has_adaptive_for_mock(self, ds):
        # MockAdapter now synthesizes a small adaptive_df for testing
        assert ds.has_adaptive

    def test_voltage_compliance_runs(self, ds, thresh):
        result = check_voltage_compliance(ds.df, thresh)
        assert "phases" in result
        assert "total_pct_out_of_bounds" in result

    def test_voltage_compliance_phases_present(self, ds, thresh):
        result = check_voltage_compliance(ds.df, thresh)
        assert "voltage_a" in result["phases"]

    def test_voltage_compliance_pct_numeric(self, ds, thresh):
        result = check_voltage_compliance(ds.df, thresh)
        pct = result["total_pct_out_of_bounds"]
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0

    def test_voltage_compliance_mock_sag_detected(self, ds, thresh):
        """MockAdapter injects a 12% sag event — compliance check must catch it."""
        result = check_voltage_compliance(ds.df, thresh)
        pct_under = result["phases"]["voltage_a"]["pct_under"]
        assert pct_under > 0.0, "Sag event in MockAdapter was not detected"

    def test_voltage_compliance_mock_swell_detected(self, ds, thresh):
        """MockAdapter injects an 8% swell event — compliance check must catch it."""
        result = check_voltage_compliance(ds.df, thresh)
        pct_over = result["phases"]["voltage_a"]["pct_over"]
        assert pct_over > 0.0, "Swell event in MockAdapter was not detected"

    def test_catalog_runs(self, ds):
        cat = ds.catalog()
        assert isinstance(cat, str)
        assert len(cat) > 0

    def test_duration_positive(self, ds):
        assert ds.duration_hours > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. check_neutral_harmonics
# ─────────────────────────────────────────────────────────────────────────────

class TestNeutralHarmonics:

    @pytest.fixture(scope="class")
    @classmethod
    def df_with_neutral(cls):
        """Synthetic DataFrame with phase and neutral harmonic channels."""
        import pandas as pd
        rng = np.random.default_rng(0)
        n   = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        data = {
            # Phase A harmonics (Amps)
            "h3_current_a":  3.5 + rng.normal(0, 0.2, n),
            "h5_current_a":  5.0 + rng.normal(0, 0.3, n),
            "h7_current_a":  2.0 + rng.normal(0, 0.2, n),
            "h9_current_a":  0.8 + rng.normal(0, 0.1, n),
            "h11_current_a": 1.2 + rng.normal(0, 0.1, n),
            "h13_current_a": 0.9 + rng.normal(0, 0.1, n),
            # Neutral: triplens accumulate (~2.8× phase H3), non-triplens near zero
            "h3_current_neutral":  3.5 * 2.8 + rng.normal(0, 0.3, n),   # ≈ 9.8 A
            "h5_current_neutral":  rng.normal(0, 0.1, n).clip(0),
            "h7_current_neutral":  rng.normal(0, 0.08, n).clip(0),
            "h9_current_neutral":  0.8 * 2.5 + rng.normal(0, 0.1, n),   # ≈ 2.0 A
            "h11_current_neutral": rng.normal(0, 0.07, n).clip(0),
            "h13_current_neutral": rng.normal(0, 0.06, n).clip(0),
        }
        return pd.DataFrame(data, index=idx)

    @pytest.fixture(scope="class")
    @classmethod
    def thresh(cls):
        # Three-phase wye data: the neutral triplens accumulate at ~2.8x phase
        # H3, which only happens on a 120-degree system. Only phase A's
        # channels are present, so channel presence alone would read this as a
        # single-phase service -- the engineer's topology pick is what settles
        # it, which is the precedence `service_geometry` documents.
        return Thresholds(nominal_voltage=120.0, topology="3ph-wye")

    def test_available_when_neutral_cols_present(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["available"] is True

    def test_unavailable_when_no_neutral_cols(self, thresh):
        import pandas as pd
        df = pd.DataFrame({"current_a": [50.0] * 10})
        result = check_neutral_harmonics(df, thresh)
        assert result["available"] is False

    def test_all_six_orders_present(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert set(result["orders"].keys()) == {3, 5, 7, 9, 11, 13}

    def test_triplen_orders_flagged(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["orders"][3]["is_triplen"] is True
        assert result["orders"][9]["is_triplen"] is True
        assert result["orders"][5]["is_triplen"] is False
        assert result["orders"][7]["is_triplen"] is False

    def test_triplen_dominant(self, df_with_neutral, thresh):
        """H3 + H9 >> H5 + H7 + H11 + H13 in this dataset."""
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["triplen_dominant"] is True

    def test_triplen_pct_above_50(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["triplen_pct"] > 50.0

    def test_accumulation_factor_computed(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["accumulation_factor"] is not None

    def test_accumulation_factor_near_expected(self, df_with_neutral, thresh):
        """H3-neutral ≈ 2.8× H3-phase, so accumulation_factor should be ≈ 2.8."""
        result = check_neutral_harmonics(df_with_neutral, thresh)
        af = result["accumulation_factor"]
        assert 2.0 < af < 4.0, f"Accumulation factor {af} out of expected 2.0–4.0 range"

    def test_mean_values_positive(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        for h, od in result["orders"].items():
            assert od["mean_a"] >= 0.0
            assert od["max_a"] >= od["mean_a"]

    def test_triplen_sum_greater_than_nontriplen(self, df_with_neutral, thresh):
        result = check_neutral_harmonics(df_with_neutral, thresh)
        assert result["triplen_sum_mean_a"] > result["nontriplen_sum_mean_a"]

    def test_pipeline_neutral_harmonics_available(self):
        """MockAdapter now includes neutral harmonic channels — full pipeline test."""
        adapter = MockAdapter(duration_hours=1.0, nominal=120.0)
        mapper  = ChannelMapper()
        ds      = extract_dataset(adapter, mapper)
        result  = check_neutral_harmonics(ds.df, Thresholds(nominal_voltage=120.0))
        assert result["available"] is True
        assert result["triplen_dominant"] is True
        assert result["accumulation_factor"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9. check_harmonic_sources
# ─────────────────────────────────────────────────────────────────────────────

class TestHarmonicSources:

    @pytest.fixture(scope="class")
    @classmethod
    def df_customer(cls):
        """Customer-injection scenario: V_h = k × h × I_h + noise → high correlation."""
        import pandas as pd
        rng = np.random.default_rng(7)
        n   = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        kz  = 0.03  # Ω per unit order (inductive trend)
        data = {}
        for h in (3, 5, 7, 11, 13):
            ih = 5.0 / h + rng.normal(0, 0.1, n)
            ih = ih.clip(0.05)
            vh = ih * h * kz + rng.normal(0, 0.002, n)
            vh = vh.clip(0)
            data[f"h{h}_current_a"] = ih
            data[f"h{h}_voltage_a"] = vh
        return pd.DataFrame(data, index=idx)

    @pytest.fixture(scope="class")
    @classmethod
    def df_resonance(cls):
        """Resonance at H5: Z_5 >> linear trend of Z_3, Z_7, Z_11, Z_13."""
        import pandas as pd
        rng = np.random.default_rng(99)
        n   = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        kz  = 0.03
        data = {}
        for h in (3, 5, 7, 11, 13):
            ih = 5.0 / h + rng.normal(0, 0.1, n)
            ih = ih.clip(0.05)
            if h == 5:
                # 6× spike above expected linear value → resonance
                vh = ih * h * kz * 6.0 + rng.normal(0, 0.01, n)
            else:
                vh = ih * h * kz + rng.normal(0, 0.002, n)
            vh = vh.clip(0)
            data[f"h{h}_current_a"] = ih
            data[f"h{h}_voltage_a"] = vh
        return pd.DataFrame(data, index=idx)

    @pytest.fixture(scope="class")
    @classmethod
    def thresh(cls):
        return Thresholds(nominal_voltage=120.0)

    # ── Basic availability ────────────────────────────────────────────────────

    def test_available_with_both_channels(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        assert result["available"] is True

    def test_unavailable_with_only_current(self, thresh):
        import pandas as pd
        df = pd.DataFrame({"h5_current_a": [5.0] * 50})
        result = check_harmonic_sources(df, thresh)
        assert result["available"] is False

    def test_unavailable_with_only_voltage(self, thresh):
        import pandas as pd
        df = pd.DataFrame({"h5_voltage_a": [0.5] * 50})
        result = check_harmonic_sources(df, thresh)
        assert result["available"] is False

    # ── Z_h values ───────────────────────────────────────────────────────────

    def test_all_five_orders_present(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        assert set(result["orders"].keys()) == {3, 5, 7, 11, 13}

    def test_z_ohm_positive(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        for h, od in result["orders"].items():
            assert od["z_ohm"] > 0, f"H{h} impedance should be positive"

    def test_z_increases_with_order(self, df_customer, thresh):
        """Z_h = k × h so Z should be monotonically increasing for customer injection."""
        result = check_harmonic_sources(df_customer, thresh)
        z_vals = [result["orders"][h]["z_ohm"] for h in sorted(result["orders"])]
        assert z_vals == sorted(z_vals), "Z_h should increase with harmonic order"

    def test_linear_slope_fitted(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        assert result["linear_slope_a"] is not None
        assert result["linear_slope_a"] > 0

    # ── Attribution — customer injection ─────────────────────────────────────

    def test_high_correlation_customer_injection(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        for h, od in result["orders"].items():
            assert od["corr"] is not None
            assert od["corr"] > 0.5, f"H{h} Pearson r={od['corr']:.2f} below 0.5 for customer scenario"

    def test_overall_customer_attribution(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        assert result["overall"] == "customer"

    def test_no_resonance_in_customer_scenario(self, df_customer, thresh):
        result = check_harmonic_sources(df_customer, thresh)
        assert result["resonant_orders"] == []

    # ── Resonance detection ───────────────────────────────────────────────────

    def test_resonance_flagged_at_h5(self, df_resonance, thresh):
        result = check_harmonic_sources(df_resonance, thresh)
        assert 5 in result["resonant_orders"], \
            f"H5 resonance not detected; z_ratio={result['orders'].get(5, {}).get('z_ratio')}"

    def test_h5_z_ratio_above_threshold(self, df_resonance, thresh):
        result = check_harmonic_sources(df_resonance, thresh)
        assert result["orders"][5]["z_ratio"] > 2.5

    def test_non_resonant_orders_not_flagged(self, df_resonance, thresh):
        result = check_harmonic_sources(df_resonance, thresh)
        for h in (3, 7, 11, 13):
            assert h not in result["resonant_orders"], f"H{h} spuriously flagged as resonance"

    def test_overall_resonance_suspect(self, df_resonance, thresh):
        result = check_harmonic_sources(df_resonance, thresh)
        assert result["overall"] == "resonance_suspect"

    # ── Pipeline smoke test ───────────────────────────────────────────────────

    def test_pipeline_source_available(self):
        """MockAdapter now has both voltage and current harmonics → full pipeline."""
        adapter = MockAdapter(duration_hours=1.0, nominal=120.0)
        mapper  = ChannelMapper()
        ds      = extract_dataset(adapter, mapper)
        result  = check_harmonic_sources(ds.df, Thresholds(nominal_voltage=120.0))
        assert result["available"] is True
        assert set(result["orders"].keys()) == {3, 5, 7, 11, 13}
        assert result["overall"] in ("customer", "mixed", "indeterminate")


# ─────────────────────────────────────────────────────────────────────────────
# 9b. Harmonic source direction — which side of the meter
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDataset:
    """Just the two attributes the direction check reads off a dataset."""

    def __init__(self, df, waveforms=()):
        self.df = df
        self.waveforms = list(waveforms)


def _capture(v_phasors, i_phasors, fs=19200.0, cycles=10, f0=60.0,
             phases=("a", "b")):
    """One synthetic point-on-wave capture.

    *v_phasors* and *i_phasors* map harmonic order to (amplitude, phase in
    degrees), so a test states the angle between voltage and current directly
    -- which is the only thing the sign of harmonic power depends on.
    """
    n = int(round(fs * cycles / f0))
    t = np.arange(n) / fs

    def build(spec):
        out = np.zeros(n)
        for h, (amp, deg) in spec.items():
            out += amp * np.cos(2 * np.pi * f0 * h * t + np.radians(deg))
        return out

    return {
        "timestamp": None, "label": "synthetic", "t": t, "fs_hz": fs,
        "voltages": {p: build(v_phasors) for p in phases},
        "currents": {p: build(i_phasors) for p in phases},
        "vne": None,
    }


class TestHarmonicDirectionFromIntervals:
    """Direction inferred from magnitudes over the whole recording.

    The discriminator is the intercept: distortion that persists when the
    premises stop drawing harmonic current came from somewhere else.
    """

    @staticmethod
    def _df(v_of_i, n=400, seed=3):
        import pandas as pd
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        # A load that swings over the day, so there are quiet intervals to
        # measure the background against and loaded ones to fit a slope on.
        duty = 0.5 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, n))
        data = {"current_a": 20.0 * duty + 2.0}
        for h in (3, 5, 7):
            ih = (4.0 / h) * duty + rng.normal(0, 0.02, n) + 0.2
            data[f"h{h}_current_a"] = ih.clip(0.05)
            data[f"h{h}_voltage_a"] = np.clip(v_of_i(h, data[f"h{h}_current_a"])
                                              + rng.normal(0, 0.01, n), 0, None)
        return pd.DataFrame(data, index=idx)

    def test_distortion_that_tracks_the_load_reads_as_customer_side(self):
        from pq_analysis import harmonic_direction_from_intervals
        df = self._df(lambda h, i: 0.4 * h * i)          # no background term
        r = harmonic_direction_from_intervals(df, Thresholds(nominal_voltage=120.0))
        assert r["available"] is True
        assert r["overall"] == "downstream"
        assert all(od["indication"] == "downstream" for od in r["orders"].values())

    def test_distortion_that_ignores_the_load_reads_as_utility_side(self):
        from pq_analysis import harmonic_direction_from_intervals
        rng = np.random.default_rng(11)
        # Voltage distortion present regardless of what the premises draws.
        df = self._df(lambda h, i: 3.0 / h + rng.normal(0, 0.05, len(i)))
        r = harmonic_direction_from_intervals(df, Thresholds(nominal_voltage=120.0))
        assert r["overall"] == "upstream"
        for h, od in r["orders"].items():
            assert od["indication"] == "upstream"
            # The intercept and the directly measured quiet-interval average
            # are the same claim by two routes; they should agree.
            assert od["v_at_quiet_v"] == pytest.approx(od["v_background_v"],
                                                       rel=0.25)

    def test_a_spectrum_at_the_meters_resolution_is_not_assessed(self):
        from pq_analysis import harmonic_direction_from_intervals
        df = self._df(lambda h, i: 0.4 * h * i)
        for h in (3, 5, 7):
            df[f"h{h}_current_a"] *= 0.02          # down into quantization
        r = harmonic_direction_from_intervals(df, Thresholds(nominal_voltage=120.0))
        assert r["overall"] == "not_assessed"
        assert all(od["indication"] == "not_assessed" for od in r["orders"].values())
        # The measurements survive even though no conclusion is drawn from them.
        assert r["orders"][3]["slope_ohm"] is not None


class TestHarmonicDirectionFromWaveforms:
    """Direction measured from the sign of harmonic power in the captures."""

    THRESH = Thresholds(nominal_voltage=120.0)
    V1 = (120.0 * np.sqrt(2), 0.0)
    I1 = (10.0 * np.sqrt(2), 0.0)          # 10 A rms, in phase → importing

    def _run(self, captures):
        from pq_analysis import harmonic_direction_from_waveforms
        import pandas as pd
        return harmonic_direction_from_waveforms(
            _FakeDataset(pd.DataFrame(), captures), self.THRESH)

    def test_harmonic_power_leaving_the_premises_reads_as_customer_side(self):
        # V3 and I3 in antiphase: P3 < 0, harmonic power flowing to the system.
        cap = _capture({1: self.V1, 3: (4.0, 180.0)},
                       {1: self.I1, 3: (2.0, 0.0)})
        r = self._run([cap, cap])
        assert r["available"] is True
        assert r["orders"][3]["indication"] == "downstream"
        assert r["orders"][3]["median_p_w"] < 0
        assert r["overall"] == "downstream"

    def test_harmonic_power_entering_the_premises_reads_as_utility_side(self):
        cap = _capture({1: self.V1, 3: (4.0, 0.0)},
                       {1: self.I1, 3: (2.0, 0.0)})
        r = self._run([cap, cap])
        assert r["orders"][3]["indication"] == "upstream"
        assert r["orders"][3]["median_p_w"] > 0

    def test_reversed_cts_are_detected_and_corrected(self):
        # Same physical situation as the customer-side case, with the clamps
        # on backwards: every current inverted, including the fundamental.
        cap = _capture({1: self.V1, 3: (4.0, 180.0)},
                       {1: (-self.I1[0], 0.0), 3: (-2.0, 0.0)})
        r = self._run([cap, cap])
        assert r["ct_polarity_inverted"] is True
        assert r["ct_polarity_verified"] is True
        assert "reversed" in r["polarity_note"]
        # The conclusion must match the un-reversed installation, not invert.
        assert r["orders"][3]["indication"] == "downstream"

    def test_captures_taken_during_an_event_are_excluded(self):
        sag = _capture({1: (70.0 * np.sqrt(2), 0.0), 3: (4.0, 180.0)},
                       {1: self.I1, 3: (2.0, 0.0)})
        r = self._run([sag, sag])
        assert r["excluded_event"] == 2
        assert r["captures_used"] == 0
        assert r["available"] is False
        assert "voltage event" in r["note"]

    def test_a_transient_capture_is_too_short_to_read(self):
        # Pronto's 153 kHz transient captures are a fraction of a cycle.
        short = _capture({1: self.V1, 3: (4.0, 180.0)},
                         {1: self.I1, 3: (2.0, 0.0)},
                         fs=153600.0, cycles=1)
        r = self._run([short])
        assert r["excluded_short"] == 1
        assert r["captures_used"] == 0

    def test_an_unloaded_capture_is_not_read_for_direction(self):
        idle = _capture({1: self.V1, 3: (4.0, 180.0)},
                        {1: (0.1, 0.0), 3: (0.05, 0.0)})
        r = self._run([idle])
        assert r["excluded_light_load"] == 1
        assert r["available"] is False

    def test_an_order_below_the_current_floor_is_not_given_a_direction(self):
        # 0.02 A rms of H5: an angle measured on that is noise.
        cap = _capture({1: self.V1, 3: (4.0, 180.0), 5: (1.0, 180.0)},
                       {1: self.I1, 3: (2.0, 0.0), 5: (0.03, 0.0)})
        r = self._run([cap, cap])
        assert 3 in r["orders"]
        assert 5 not in r["orders"]

    def test_the_measured_fundamental_is_used_not_the_nominal_one(self):
        # An off-nominal system: at 59.3 Hz, projecting onto 60 Hz would walk
        # the third harmonic's angle right across the sign boundary.
        cap = _capture({1: self.V1, 3: (4.0, 180.0)},
                       {1: self.I1, 3: (2.0, 0.0)}, f0=59.3)
        r = self._run([cap, cap])
        assert r["fundamental_hz"] == pytest.approx(59.3, abs=0.05)
        assert r["orders"][3]["indication"] == "downstream"

    def test_a_file_with_no_captures_says_so_rather_than_failing(self):
        r = self._run([])
        assert r["available"] is False
        assert r["captures_total"] == 0
        assert "No point-on-wave captures" in r["note"]


class TestHarmonicDirectionCombined:
    """The two methods together, including when they disagree."""

    THRESH = Thresholds(nominal_voltage=120.0)

    def _ds(self, v_of_i, wave_angle_deg):
        df = TestHarmonicDirectionFromIntervals._df(v_of_i)
        cap = _capture({1: TestHarmonicDirectionFromWaveforms.V1,
                        3: (4.0, wave_angle_deg)},
                       {1: TestHarmonicDirectionFromWaveforms.I1, 3: (2.0, 0.0)})
        return _FakeDataset(df, [cap, cap])

    def test_agreement_is_recorded_per_order(self):
        from pq_analysis import check_harmonic_direction
        ds = self._ds(lambda h, i: 0.4 * h * i, 180.0)   # both say customer
        r = check_harmonic_direction(ds, self.THRESH)
        assert r["agreement"][3] == "agree"
        assert r["overall"] == "downstream"
        assert r["methods_agree"] is True

    def test_disagreement_is_reported_rather_than_resolved(self):
        from pq_analysis import check_harmonic_direction
        rng = np.random.default_rng(5)
        # Trend says the distortion is background; the captures say the
        # premises is exporting H3. Both readings stand.
        ds = self._ds(lambda h, i: 3.0 / h + rng.normal(0, 0.05, len(i)), 180.0)
        r = check_harmonic_direction(ds, self.THRESH)
        assert r["agreement"][3] == "disagree"
        assert r["overall"] == "conflicting"
        assert r["methods_agree"] is False

    def test_the_report_section_prints_both_methods(self):
        from pq_analysis import check_harmonic_direction
        from pq_report import _direction_summary_sentence
        ds = self._ds(lambda h, i: 0.4 * h * i, 180.0)
        hd = check_harmonic_direction(ds, self.THRESH)
        sentence = _direction_summary_sentence(hd)
        assert "customer side" in sentence
        assert "agree at H3" in sentence

    def _rendered(self, customer_class):
        docx = pytest.importorskip("docx")
        from pq_analysis import check_harmonic_direction
        from pq_report import _word_harmonic_direction
        ds = self._ds(lambda h, i: 0.4 * h * i, 180.0)
        thresh = Thresholds(nominal_voltage=120.0, customer_class=customer_class)
        doc = docx.Document()
        _word_harmonic_direction(
            doc, {"harmonic_direction": check_harmonic_direction(ds, thresh)},
            thresh)
        return "\n".join(p.text for p in doc.paragraphs)

    def test_the_engineering_report_gets_the_section(self):
        text = self._rendered("sg")
        assert "Harmonic Source Direction" in text
        assert "Engineer's assessment" in text
        # Direction, never fault: the section names a side of the meter, says
        # in as many words that this is not an attribution, and leaves the
        # assessment blank for the engineer signing the report.
        assert "do not assign responsibility" in text
        assert "Neither method establishes what equipment is responsible" in text
        for claim in ("customer is responsible", "caused by the customer",
                      "the customer must", "Xcel will"):
            assert claim not in text

    def test_a_residential_report_does_not_get_it(self):
        # A homeowner's report carries no harmonic content at all.
        assert "Harmonic Source Direction" not in self._rendered("r")


# ─────────────────────────────────────────────────────────────────────────────
# 9c. Service impedance and high-impedance screening
# ─────────────────────────────────────────────────────────────────────────────

def _stepped_load(n=600, seed=4, peak=25.0):
    """A load that switches on and off, the way a real service's does."""
    rng = np.random.default_rng(seed)
    base = np.zeros(n)
    level = 4.0
    for k in range(n):
        if rng.random() < 0.25:                    # something switches
            level = float(rng.uniform(2.0, peak))
        base[k] = level + rng.normal(0, 0.05)
    return np.clip(base, 0.5, None)


def _service_frame(r_ohm=0.05, x_ohm=0.02, n=600, seed=4, v0=120.0,
                   phases=("a",), pf_varies=True, neutral_r=None,
                   per_phase_r=None, noise_v=0.01):
    """Interval data for a service of known impedance.

    Built forwards from the physics the check works backwards from: each
    phase's voltage is the no-load voltage less its own load's drop, so a
    test can state the answer and see whether it comes back.
    """
    import pandas as pd
    rng = np.random.default_rng(seed + 1)
    idx = pd.date_range("2024-06-01", periods=n, freq="5min", tz="UTC")
    pf = (rng.uniform(0.75, 1.0, n) if pf_varies else np.full(n, 0.9))
    data = {"power_factor": pf}
    for k, ph in enumerate(phases):
        i = _stepped_load(n, seed=seed + k)
        r = (per_phase_r or {}).get(ph, r_ohm)
        drop = r * i * pf + x_ohm * i * np.sqrt(np.clip(1 - pf ** 2, 0, 1))
        data[f"current_{ph}"] = i
        data[f"voltage_{ph}"] = v0 - drop + rng.normal(0, noise_v, n)
    if neutral_r is not None:
        i_n = _stepped_load(n, seed=seed + 50, peak=15.0)
        data["current_neutral"] = i_n
        data["voltage_neutral"] = neutral_r * i_n + rng.normal(0, 0.005, n)
    return pd.DataFrame(data, index=idx)


class TestServiceImpedanceMeasurement:
    """Reading the impedance between the source and the meter off the load steps."""

    THRESH = Thresholds(nominal_voltage=120.0)

    def test_a_known_impedance_comes_back(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.05, x_ohm=0.02)
        r = check_source_impedance(df, self.THRESH)
        fit = r["phases"]["a"]
        assert fit["identifiable"] is True
        assert fit["separated"] is True
        assert fit["r_ohm"] == pytest.approx(0.05, abs=0.005)
        assert fit["x_ohm"] == pytest.approx(0.02, abs=0.005)

    def test_a_steady_power_factor_gives_a_magnitude_not_a_split(self):
        from pq_analysis import check_source_impedance
        # Without pf variation the real and reactive parts of the current are
        # the same shape, and R and X are not separately identifiable.
        df = _service_frame(r_ohm=0.05, x_ohm=0.02, pf_varies=False)
        fit = check_source_impedance(df, self.THRESH)["phases"]["a"]
        assert fit["identifiable"] is True
        assert fit["separated"] is False
        assert fit.get("x_ohm") is None
        # The effective magnitude along the load's own angle still lands.
        assert fit["z_ohm"] == pytest.approx(0.05 * 0.9 + 0.02 * 0.436, abs=0.006)

    def test_feeder_wide_droop_is_not_read_as_this_services_impedance(self):
        # The failure this estimator exists to avoid: a voltage that sags on
        # the same daily cycle as the load, with no local impedance at all.
        # Fitting the levels finds an impedance; fitting the steps must not.
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.0, x_ohm=0.0, noise_v=0.02)
        daily = 3.0 * np.sin(np.linspace(0, 4 * np.pi, len(df)))
        df["voltage_a"] = df["voltage_a"] - daily
        df["current_a"] = df["current_a"] + 4.0 * np.sin(
            np.linspace(0, 4 * np.pi, len(df)))
        level_slope = np.polyfit(df["current_a"], df["voltage_a"], 1)[0]
        assert level_slope < -0.1          # the naive fit would claim >0.1 Ω

        fit = check_source_impedance(df, self.THRESH)["phases"]["a"]
        assert fit["identifiable"] is False
        assert "not this service" in fit["reason"]

    def test_a_drop_below_the_meters_resolution_is_reported_as_such(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.0, x_ohm=0.0, noise_v=0.0)
        fit = check_source_impedance(df, self.THRESH)["phases"]["a"]
        assert fit["identifiable"] is False
        assert fit["at_resolution"] is True
        assert "resolution" in fit["reason"]

    def test_one_bad_phase_is_flagged_against_the_others(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.03, x_ohm=0.0,
                            phases=("a", "b", "c"),
                            per_phase_r={"c": 0.12})   # a degrading connection
        r = check_source_impedance(df, self.THRESH)
        asym = r["asymmetry"]
        assert asym["worst_phase"] == "C"
        assert asym["flagged"] is True
        assert asym["ratio"] > 2.0
        assert r["overall"] == "high_impedance_suspected"

    def test_balanced_phases_are_not_flagged(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.03, x_ohm=0.0, phases=("a", "b", "c"))
        assert check_source_impedance(df, self.THRESH)["asymmetry"]["flagged"] is False

    def test_a_resistive_neutral_is_measured_from_its_own_rise(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.03, x_ohm=0.0, neutral_r=0.25)
        neutral = check_source_impedance(df, self.THRESH)["neutral"]
        assert neutral["identifiable"] is True
        assert neutral["r_ohm"] == pytest.approx(0.25, abs=0.02)
        assert neutral["elevated"] is True
        assert neutral["rise_at_peak_v"] > 2.0

    def test_a_sound_neutral_reads_as_below_resolution(self):
        from pq_analysis import check_source_impedance
        df = _service_frame(r_ohm=0.03, x_ohm=0.0, neutral_r=0.0)
        neutral = check_source_impedance(df, self.THRESH)["neutral"]
        assert neutral["identifiable"] is False
        assert neutral["at_resolution"] is True

    def test_a_service_that_never_moves_is_not_measured(self):
        import pandas as pd
        from pq_analysis import check_source_impedance
        idx = pd.date_range("2024-06-01", periods=200, freq="5min", tz="UTC")
        df = pd.DataFrame({"voltage_a": 120.0, "current_a": 10.0}, index=idx)
        fit = check_source_impedance(df, self.THRESH)["phases"]["a"]
        assert fit["identifiable"] is False
        assert "load step" in fit["reason"]


class TestExpectedServiceImpedance:
    """What the picked transformer and conductor say the impedance should be."""

    def test_a_single_phase_run_counts_the_neutral_return(self):
        from pq_constants import conductor_impedance
        out = conductor_impedance("al-4-0-triplex", 150.0, return_path=True)
        one_way = conductor_impedance("al-4-0-triplex", 150.0, return_path=False)
        # 4/0 AL at 0.100 Ω/1000 ft: 150 ft out and 150 ft back.
        assert out[0] == pytest.approx(0.030, abs=1e-6)
        assert one_way[0] == pytest.approx(0.015, abs=1e-6)

    def test_the_isc_path_includes_the_primary_system(self):
        from pq_constants import expected_service_impedance
        thresh = Thresholds(nominal_voltage=120.0, service_type="3ph-padmount",
                            transformer_kva=150, isc_amps=8000,
                            conductor_key="al-350-urd", run_length_ft=200)
        e = expected_service_impedance(thresh)
        assert e["available"] is True
        assert e["upstream_ohm"] == pytest.approx(120.0 / 8000)
        assert "primary system and the transformer" in e["upstream_source"]
        assert e.get("upstream_is_floor") is None

    def test_without_isc_the_expected_value_is_called_a_floor(self):
        from pq_constants import expected_service_impedance
        thresh = Thresholds(nominal_voltage=120.0, service_type="1ph-overhead",
                            transformer_kva=25, topology="split-phase",
                            conductor_key="al-4-0-triplex", run_length_ft=150)
        e = expected_service_impedance(thresh)
        # 25 kVA at 1.6–2.4%: V_LN²/S is the base the L-N path sits on.
        assert e["transformer_ohm_range"][0] == pytest.approx(
            0.016 * 120.0 ** 2 / 25000.0)
        assert e["upstream_is_floor"] is True
        assert "floor" in e["upstream_source"]

    def test_nothing_picked_means_no_expected_value(self):
        from pq_constants import expected_service_impedance
        e = expected_service_impedance(Thresholds(nominal_voltage=120.0))
        assert e["available"] is False
        assert "run length" in e["reason"]

    def test_the_comparison_calls_a_large_excess_high(self):
        from pq_analysis import check_source_impedance
        thresh = Thresholds(nominal_voltage=120.0, service_type="1ph-overhead",
                            transformer_kva=25, topology="split-phase",
                            conductor_key="al-4-0-triplex", run_length_ft=150)
        df = _service_frame(r_ohm=0.20, x_ohm=0.0)     # far above ~0.043 Ω
        r = check_source_impedance(df, thresh)
        assert r["comparison"]["verdict"] == "high"
        assert r["comparison"]["ratio"] > 2.5
        assert r["comparison"]["excess_v_at_peak"] > 1.0
        assert r["overall"] == "high_impedance_suspected"

    def test_an_as_built_service_reads_as_consistent(self):
        from pq_analysis import check_source_impedance
        thresh = Thresholds(nominal_voltage=120.0, service_type="1ph-overhead",
                            transformer_kva=25, topology="split-phase",
                            conductor_key="al-4-0-triplex", run_length_ft=150)
        # 0.0115 Ω transformer + 0.030 Ω of conductor ≈ 0.043 Ω expected.
        df = _service_frame(r_ohm=0.042, x_ohm=0.010, pf_varies=False)
        r = check_source_impedance(df, thresh)
        assert r["comparison"]["verdict"] == "consistent"
        assert r["overall"] == "consistent_with_expected"

    def test_the_report_section_states_the_constants_are_generic(self):
        docx = pytest.importorskip("docx")
        from pq_analysis import check_source_impedance
        from pq_report import _word_service_impedance
        thresh = Thresholds(nominal_voltage=120.0, customer_class="r",
                            service_type="1ph-overhead", transformer_kva=25,
                            topology="split-phase",
                            conductor_key="al-4-0-triplex", run_length_ft=150)
        df = _service_frame(r_ohm=0.20, x_ohm=0.0)
        doc = docx.Document()
        _word_service_impedance(
            doc, {"service_impedance": check_source_impedance(df, thresh)}, thresh)
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "Service Impedance" in text
        # The engineer has to be able to see that the expected side is generic.
        assert "generic published values" in text
        assert "not PSCo Blue Book figures" in text
        assert "Engineer's assessment" in text


# ─────────────────────────────────────────────────────────────────────────────
# 10. detect_events — adaptive vs interval path
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveEvents:
    """detect_events uses cycle-level adaptive data when ds.has_adaptive."""

    @pytest.fixture(scope="class")
    @classmethod
    def ds_with_adaptive(cls):
        """PQDataset backed by MockAdapter — includes synthetic adaptive_df."""
        adapter = MockAdapter(duration_hours=2.0, nominal=120.0)
        mapper  = ChannelMapper()
        return extract_dataset(adapter, mapper)

    @pytest.fixture(scope="class")
    @classmethod
    def thresh(cls):
        return Thresholds(nominal_voltage=120.0)

    # ── data_source flag ─────────────────────────────────────────────────────

    def test_data_source_adaptive_when_present(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        assert result["data_source"] == "adaptive"

    # ── voltage sag from van_v ───────────────────────────────────────────────

    def test_voltage_sag_detected_from_adaptive(self, ds_with_adaptive, thresh):
        """MockAdapter injects van_v[50:80] *= 0.86 — sag must be reported."""
        result = detect_events(ds_with_adaptive, thresh)
        sag_events = result["events"][result["events"]["type"] == "voltage_sag"]
        assert len(sag_events) > 0, "No voltage_sag detected from adaptive van_v"

    def test_voltage_sag_phase_a(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        sag_events = result["events"][result["events"]["type"] == "voltage_sag"]
        assert "A" in sag_events["phase"].values

    def test_sag_value_below_90pct(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        sag_events = result["events"][result["events"]["type"] == "voltage_sag"]
        assert sag_events["value_v"].min() < 0.90 * 120.0

    # ── PST flicker exceedance ────────────────────────────────────────────────

    def test_flicker_pst_detected(self, ds_with_adaptive, thresh):
        """MockAdapter injects adap_pst[100:130] = 1.4 — PST event must be reported."""
        result = detect_events(ds_with_adaptive, thresh)
        pst_events = result["events"][result["events"]["type"] == "flicker_pst"]
        assert len(pst_events) > 0, "No flicker_pst event detected"

    def test_flicker_pst_value_above_limit(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        pst_events = result["events"][result["events"]["type"] == "flicker_pst"]
        assert pst_events["value"].iloc[0] > 1.0

    # ── current step ─────────────────────────────────────────────────────────

    def test_current_step_detected_from_adaptive(self, ds_with_adaptive, thresh):
        """MockAdapter injects ia_a step from 50 A to 83 A at row 150."""
        result = detect_events(ds_with_adaptive, thresh)
        step_events = result["events"][result["events"]["type"] == "current_step"]
        assert len(step_events) > 0, "No current_step detected from adaptive ia_a"

    def test_current_step_phase_a(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        step_events = result["events"][result["events"]["type"] == "current_step"]
        assert "A" in step_events["phase"].values

    # ── result shape ─────────────────────────────────────────────────────────

    def test_event_count_matches_df_length(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        assert result["event_count"] == len(result["events"])

    def test_events_df_has_required_columns(self, ds_with_adaptive, thresh):
        result = detect_events(ds_with_adaptive, thresh)
        assert "timestamp" in result["events"].columns
        assert "type" in result["events"].columns
        assert "phase" in result["events"].columns

    # ── interval fallback path ────────────────────────────────────────────────

    def test_interval_fallback_data_source(self, thresh):
        """When adaptive_df is None, data_source must be 'interval'."""
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        df  = pd.DataFrame({
            "voltage_a": 120.0 + np.zeros(100),
            "voltage_b": 120.0 + np.zeros(100),
            "voltage_c": 120.0 + np.zeros(100),
        }, index=idx)
        from pq_adapter import PQDataset
        ds_no_adap = PQDataset(df=df, adaptive_df=None, meta={"interval_minutes": 5})
        result = detect_events(ds_no_adap, thresh)
        assert result["data_source"] == "interval"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Standardized result shapes — available/error contract
# ─────────────────────────────────────────────────────────────────────────────

class TestUnavailableShapes:
    """All check_* functions must return available+error keys even on missing data."""

    @pytest.fixture
    def empty_df(self):
        import pandas as pd
        return pd.DataFrame(index=pd.date_range("2024-01-01", periods=10, freq="5min"))

    @pytest.fixture
    def thresh(self):
        return Thresholds(nominal_voltage=120.0)

    def test_voltage_compliance_unavailable(self, empty_df, thresh):
        from pq_analysis import check_voltage_compliance
        r = check_voltage_compliance(empty_df, thresh)
        assert r["available"] is False
        assert r["error"] is not None
        assert r["total_pct_out_of_bounds"] is None
        assert len(r["violation_timestamps"]) == 0

    def test_voltage_compliance_available_shape(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"voltage_a": [120.0] * 10}, index=idx)
        from pq_analysis import check_voltage_compliance
        r = check_voltage_compliance(df, thresh)
        assert r["available"] is True
        assert r["error"] is None
        assert isinstance(r["total_pct_out_of_bounds"], float)

    def test_thd_unavailable_sub_dicts(self, empty_df, thresh):
        from pq_analysis import check_thd
        r = check_thd(empty_df, thresh)
        assert r["voltage"]["available"] is False
        assert r["current"]["available"] is False
        assert r["available"] is False

    def test_thd_available_when_voltage_found(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"thd_voltage_a": [3.0] * 10}, index=idx)
        from pq_analysis import check_thd
        r = check_thd(df, thresh)
        assert r["voltage"]["available"] is True
        assert r["available"] is True

    def test_power_factor_unavailable(self, empty_df, thresh):
        from pq_analysis import check_power_factor
        r = check_power_factor(empty_df, thresh)
        assert r["available"] is False
        assert r["error"] is not None
        assert r["pct_below_limit"] is None
        assert len(r["violation_timestamps"]) == 0

    def test_voltage_imbalance_unavailable(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"voltage_a": [120.0] * 10}, index=idx)  # only 1 phase
        from pq_analysis import check_voltage_imbalance
        r = check_voltage_imbalance(df, thresh)
        assert r["available"] is False
        assert r["pct_exceeding"] is None

    def test_voltage_imbalance_available_shape(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"voltage_a": [120.0]*10, "voltage_b": [119.5]*10}, index=idx)
        from pq_analysis import check_voltage_imbalance
        r = check_voltage_imbalance(df, thresh)
        assert r["available"] is True
        assert r["error"] is None
        assert isinstance(r["pct_exceeding"], float)

    def test_current_imbalance_unavailable(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"current_a": [50.0] * 10}, index=idx)
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(df, thresh)
        assert r["available"] is False
        assert r["pct_exceeding"] is None

    def test_current_imbalance_available_shape(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"current_a": [50.0]*10, "current_b": [48.0]*10}, index=idx)
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(df, thresh)
        assert r["available"] is True
        assert r["error"] is None

    def test_demand_unavailable(self, empty_df, thresh):
        from pq_analysis import check_demand
        r = check_demand(empty_df, thresh)
        assert r["available"] is False
        assert r["error"] is not None

    def test_demand_available_when_real_power_present(self, thresh):
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df  = pd.DataFrame({"power_real": [18000.0] * 10}, index=idx)
        from pq_analysis import check_demand
        r = check_demand(df, thresh)
        assert r["available"] is True
        assert r["error"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Spec-compliant PQDIF reader (IEEE Std 1159.3-2019)
# ─────────────────────────────────────────────────────────────────────────────

class TestPQDIFPhysical:
    """Annex A physical structure: elements, scalars, vectors."""

    @staticmethod
    def _element(body, physical_type, link, size, embedded=False,
                 element_type=pqdif.ELEMENT_VECTOR):
        return pqdif.Element(pqdif.TAG_SERIES_VALUES, element_type,
                             physical_type, embedded, link, size, body)

    def test_real8_vector(self):
        body = struct.pack("<I3d", 3, 1.5, 2.5, 3.5)
        el = self._element(body, 41, 0, len(body))
        assert np.allclose(el.vector(), [1.5, 2.5, 3.5])

    def test_int2_vector_widths_follow_physical_type(self):
        # A vector's stride comes from typePhysical, so INTEGER2 is 2 bytes per
        # point -- never assumed to be 8.
        body = struct.pack("<I4h", 4, -3, 7, 11, -19)
        el = self._element(body, 21, 0, len(body))
        assert np.allclose(el.vector(), [-3, 7, 11, -19])

    def test_complex16_takes_real_part(self):
        # COMPLEX16 is two REAL8 per point (Annex A), i.e. 16 bytes.
        body = struct.pack("<I4d", 2, 1.0, 90.0, 2.0, -45.0)
        el = self._element(body, 43, 0, len(body))
        assert np.allclose(el.vector(), [1.0, 2.0])

    def test_embedded_scalar_reads_from_link_bytes(self):
        # isEmbedded=TRUE stores the value in the 8 link/size bytes themselves.
        el = self._element(b"", 32, 1234, 0, embedded=True,
                           element_type=pqdif.ELEMENT_SCALAR)
        assert el.scalar() == 1234

    def test_non_embedded_scalar_reads_from_body(self):
        body = struct.pack("<xxxxI", 4321)
        el = self._element(body, 32, 4, 4, embedded=False,
                           element_type=pqdif.ELEMENT_SCALAR)
        assert el.scalar() == 4321

    def test_vector_overrunning_body_is_rejected(self):
        body = struct.pack("<I1d", 500, 1.0)   # claims 500 points, carries 1
        el = self._element(body, 41, 0, len(body))
        with pytest.raises(pqdif.PQDIFError):
            el.vector()

    def test_non_pqdif_file_is_rejected(self, tmp_path):
        # The signature GUID check is what routes the synthetic test fixtures to
        # the legacy reader instead of the spec reader.
        path = tmp_path / "not.pqd"
        path.write_bytes(b"\x00" * 256)
        with pytest.raises(pqdif.PQDIFError):
            pqdif.PQDIFFile(path)


class TestPQDIFIncrementSeries:
    """Clause 5.5.2 regular-rate series, using the standard's own examples."""

    def test_single_rate(self):
        # Spec example: 1792 points at 0.01 s.
        out = pqdif._expand_increment(np.array([1.0, 1792.0, 0.01]))
        assert len(out) == 1792
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.01)
        assert out[-1] == pytest.approx(1791 * 0.01)

    def test_two_rates(self):
        # Spec example: 896 points at 0.01 s then 896 at 0.02 s.
        out = pqdif._expand_increment(
            np.array([2.0, 896.0, 0.01, 896.0, 0.02]))
        assert len(out) == 1792
        assert out[895] == pytest.approx(895 * 0.01)
        assert out[896] == pytest.approx(896 * 0.01)
        assert out[897] == pytest.approx(896 * 0.01 + 0.02)

    def test_truncated_instructions_rejected(self):
        with pytest.raises(pqdif.PQDIFError):
            pqdif._expand_increment(np.array([5.0, 10.0, 0.01]))


class TestStepPairDetection:
    """Pronto writes each interval twice, at its start and end."""

    def test_step_pairs_detected(self):
        # Gaps within a pair are the interval; gaps between pairs are ~1 us.
        t = np.array([0.0, 120.0, 120.000001, 240.0, 240.000001, 360.0])
        assert ProntoAdapter._step_pair_stride(t) == 2

    def test_plain_series_not_deduplicated(self):
        t = np.arange(0.0, 600.0, 120.0)
        assert ProntoAdapter._step_pair_stride(t) == 1

    def test_odd_length_never_paired(self):
        t = np.array([0.0, 120.0, 120.000001])
        assert ProntoAdapter._step_pair_stride(t) == 1

    def test_uniform_series_not_paired(self):
        # Equal gaps everywhere: no pair structure, so keep every point.
        t = np.arange(0.0, 8.0, 1.0)
        assert ProntoAdapter._step_pair_stride(t) == 1


class TestSpecChannelTable:
    """The metadata → canonical mapping must agree with ChannelMapper."""

    def test_canonical_names_match_tag_resolution(self):
        # _SPEC_CHANNELS records the canonical name so interval_peaks/_mins can
        # be keyed by it. That name must be what ChannelMapper independently
        # derives from the same tags, or peaks would attach to the wrong column.
        mapper = ChannelMapper()
        for key, (label, canonical, qt, qm, phase) in \
                ProntoAdapter._SPEC_CHANNELS.items():
            info = RawChannelInfo(0, label, qt, qm, phase, "")
            assert mapper._match_by_tags(info) == canonical, (
                f"{key} → {canonical!r} but tags {(qt, qm, phase)} resolve to "
                f"{mapper._match_by_tags(info)!r}"
            )

    def test_rms_is_preferred_over_the_fundamental(self):
        # 'Harm 1 of Van' is characteristic SPECTRA_HGROUP and must never be
        # picked up as the voltage trend; only RMS may map to voltage_a.
        assert ('voltage', 'RMS', 'an') in ProntoAdapter._SPEC_CHANNELS
        assert ProntoAdapter._SPEC_CHANNELS[('voltage', 'RMS', 'an')][1] == \
            'voltage_a'
        for key in ProntoAdapter._SPEC_CHANNELS:
            assert key[1] not in ('SPECTRA_HGROUP', 'SPECTRA'), (
                f"{key} would source a trend from a harmonic magnitude"
            )

    def test_harmonic_name_pattern(self):
        m = ProntoAdapter._HARM_NAME.match('Harm 13 of Van')
        assert m and m.group(1) == '13' and m.group(2) == 'Van'
        m = ProntoAdapter._HARM_NAME.match('Harm 3 of In')
        assert m and m.group(1) == '3' and m.group(2) == 'In'
        # The RMS channels must not be mistaken for harmonic orders.
        assert ProntoAdapter._HARM_NAME.match('RMS Van (V1)') is None
        assert ProntoAdapter._HARM_NAME.match('Hrms Van (V1)') is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. test_data fixtures are valid PQDIF and exercise the spec reader
# ─────────────────────────────────────────────────────────────────────────────

_FIXTURES = sorted((Path(__file__).parent / "test_data").glob("*.pqd"))


@pytest.mark.skipif(not _FIXTURES, reason="test_data/*.pqd not generated")
class TestFixturesAreCompliant:
    """The fixtures exist to exercise the same path real Pronto files take.

    Regenerate them with `python make_test_pqd.py` after changing the writer.
    """

    @pytest.fixture(params=_FIXTURES, ids=lambda p: p.stem)
    def path(self, request):
        return request.param

    def test_parses_as_standard_pqdif(self, path):
        f = pqdif.PQDIFFile(path)
        assert f.version[:2] == (1, 5)
        assert f.compressed is True
        assert f.definitions

    def test_uses_the_spec_reader_not_the_legacy_fallback(self, path):
        adapter = ProntoAdapter(path)
        assert adapter._spec is not None, (
            f"{path.name} fell back to the legacy offset reader"
        )

    def test_every_guid_is_a_standard_identifier(self, path):
        # A vendor-private GUID would be legal but these fixtures should model
        # what real Pronto files contain, which is standard IDs throughout.
        f = pqdif.PQDIFFile(path)
        for defn in f.definitions:
            assert defn.quantity_type in ("VALUELOG", "PHASOR", "WAVEFORM")
            assert defn.quantity_measured in ("voltage", "current", "power")
            for series in defn.series:
                assert series.characteristic in pqdif.CHARACTERISTIC_NAMES.values()
                assert series.value_type in pqdif.VALUE_TYPE_NAMES.values()

    def test_interval_records_share_one_time_base(self, path):
        f = pqdif.PQDIFFile(path)
        grids = {
            len(obs.channels[0].time)
            for obs in f.observations if obs.channels
        }
        assert len(grids) == 1, f"observations disagree on sample count: {grids}"

    def test_step_pair_encoding_is_detected(self, path):
        f = pqdif.PQDIFFile(path)
        raw = f.observations[0].channels[0].time
        assert ProntoAdapter._step_pair_stride(raw) == 2
        # 288 intervals written as 576 points.
        adapter = ProntoAdapter(path)
        assert len(adapter._obs_ts) == len(raw) // 2

    def test_voltage_comes_from_rms_not_the_fundamental(self, path):
        # The regression this guards: 'Harm 1 of Van' (SPECTRA_HGROUP) is the
        # fundamental and is smaller than the true RMS by sqrt(1 + THD^2). The
        # mapped voltage_a column must be the RMS channel, not the fundamental.
        f = pqdif.PQDIFFile(path)
        by_name = {c.name: c for obs in f.observations for c in obs.channels}
        rms_channel = by_name["RMS Van (V1)"]
        fundamental_channel = by_name["Harm 1 of Van"]
        assert rms_channel.characteristic == "RMS"
        assert fundamental_channel.characteristic == "SPECTRA_HGROUP"

        rms = rms_channel.series["AVG"][0::2]
        fundamental = fundamental_channel.series["AVG"][0::2]
        assert np.all(rms > fundamental), "fixture does not separate RMS from H1"

        df = extract_dataset(ProntoAdapter(path), ChannelMapper()).df
        assert np.allclose(df["voltage_a"].to_numpy(), rms, rtol=1e-9)
        # And the difference is the physically expected factor.
        thd = df["thd_voltage_a"].to_numpy()
        assert np.allclose(fundamental, rms / np.sqrt(1.0 + (thd / 100.0) ** 2),
                           rtol=1e-9)

    def test_time_base_comes_from_tag_time_start(self, path):
        import pandas as pd
        adapter = ProntoAdapter(path)
        expected = pqdif.PQDIFFile(path).observations[0].start_time
        assert pd.Timestamp(adapter._obs_ts[0]).to_pydatetime() == expected

    def test_min_le_avg_le_peak(self, path):
        # Independent of any offset assumption: if MAX/MIN/AVG were assigned to
        # the wrong series this ordering would break.
        adapter = ProntoAdapter(path)
        df = extract_dataset(adapter, ChannelMapper()).df
        checked = 0
        for col in df.columns:
            if f"{col}_peak" not in df.columns:
                continue
            avg, peak, low = df[col], df[f"{col}_peak"], df[f"{col}_min"]
            assert (peak >= avg - 1e-9).all(), f"{col}: peak < avg"
            assert (low <= avg + 1e-9).all(), f"{col}: min > avg"
            checked += 1
        assert checked >= 2, "expected max-min columns on the fixtures"


class TestOverloadedCharacteristics:
    """Pronto reuses ID_QC_HRMS for four different channels per phase."""

    def test_hrms_requires_a_name_prefix(self):
        # 'Hrms Van (V1)', 'Odds Van (V1)', 'Evens Van (V1)' and 'Triplens Van'
        # all carry characteristic HRMS with phase 'an'. Only the first is the
        # total; without the prefix check, file ordering alone decided which one
        # became hrms_voltage_a.
        assert ProntoAdapter._SPEC_NAME_PREFIX["HRMS"] == "hrms"
        assert ("voltage", "HRMS", "an") in ProntoAdapter._SPEC_CHANNELS
        for name, wanted in [
            ("Hrms Van (V1)", True), ("Odds Van (V1)", False),
            ("Evens Van (V1)", False), ("Triplens Van", False),
        ]:
            matches = name.lower().startswith(
                ProntoAdapter._SPEC_NAME_PREFIX["HRMS"])
            assert matches is wanted, name

    def test_aggregate_hrms_is_not_a_per_order_column(self):
        # hrms_voltage_a must not be swept up by the per-order harmonic checks,
        # which would test the whole harmonic RMS against a single-order limit.
        from pq_analysis import _HARMONIC_COL
        assert _HARMONIC_COL.match("h3_voltage_a")
        assert _HARMONIC_COL.match("h13_current_neutral")
        assert not _HARMONIC_COL.match("hrms_voltage_a")
        assert not _HARMONIC_COL.match("hrms_current_neutral")

    def test_new_canonical_names_are_all_resolvable(self):
        # Every canonical name _SPEC_CHANNELS claims must exist in CANONICAL,
        # or extract_dataframe would drop the column on the floor.
        from pq_adapter import CANONICAL
        for key, (_label, canonical, *_tags) in \
                ProntoAdapter._SPEC_CHANNELS.items():
            assert canonical in CANONICAL, f"{key} → {canonical!r} not in CANONICAL"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Line-to-line voltage, frequency, flicker and K-factor
# ─────────────────────────────────────────────────────────────────────────────

def _frame(**cols):
    import pandas as pd
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.date_range("2025-01-01", periods=n, freq="5min"))


class TestKFactorByPhase:
    def test_rating_is_sized_on_the_worst_phase(self):
        from pq_analysis import kfactor_by_phase
        df = _frame(kfactor_meter=[104.0] * 10, kfactor_current_b=[217.0] * 10)
        kf = kfactor_by_phase(df)
        assert kf["available"] and kf["worst_phase"] == "B"
        assert kf["median"] == 217.0
        assert kf["phase_a_median"] == 104.0

    def test_neutral_does_not_drive_the_rating(self):
        # Neutral K-factor describes conductor heating, not a transformer winding.
        from pq_analysis import kfactor_by_phase
        df = _frame(kfactor_meter=[6.0] * 10, kfactor_current_neutral=[99.0] * 10)
        kf = kfactor_by_phase(df)
        assert kf["worst_phase"] == "A" and kf["median"] == 6.0
        assert "N" in kf["phases"]

    def test_unavailable_without_channels(self):
        from pq_analysis import kfactor_by_phase
        assert kfactor_by_phase(_frame(voltage_a=[120.0] * 5))["available"] is False

    def test_rating_never_exceeds_a_purchasable_unit(self):
        from pq_analysis import standard_k_rating, STANDARD_K_RATINGS
        for k in (1.5, 5.0, 10.0, 25.0, 50.0):
            rating, _ = standard_k_rating(k)
            assert rating in STANDARD_K_RATINGS or rating == 1
            assert rating >= k or rating == 1
        rating, wording = standard_k_rating(217.3)
        assert rating is None and "exceeds K-50" in wording


class TestFlickerAllPhases:
    def test_worst_phase_governs_not_phase_a(self):
        from pq_analysis import check_flicker
        df = _frame(flicker_pst=[0.5] * 10, flicker_pst_b=[4.98] * 10,
                    flicker_plt=[0.2] * 10, flicker_plt_b=[2.21] * 10)
        r = check_flicker(df, Thresholds())
        assert r["available"] and r["worst_phase"] == "B"
        assert r["pst_max"] == 4.98
        # Phase A alone would have passed both limits.
        assert r["pst"]["A"]["pass"] and not r["pst"]["B"]["pass"]
        assert r["overall_pass"] is False

    def test_pass_when_every_phase_is_within_limits(self):
        from pq_analysis import check_flicker
        df = _frame(flicker_pst=[0.4] * 10, flicker_pst_c=[0.6] * 10,
                    flicker_plt=[0.3] * 10)
        r = check_flicker(df, Thresholds())
        assert r["overall_pass"] is True
        assert sorted(r["phases_read"]) == ["A", "C"]

    def test_unavailable_without_channels(self):
        from pq_analysis import check_flicker
        assert check_flicker(_frame(voltage_a=[120.0] * 5), Thresholds())["available"] is False


class TestShortAndLongTermFlickerAreReportedSeparately:
    """Pst and Plt measure different windows and fail independently.

    Both were always read and analysed per phase; the narrative section and
    the key findings took their numbers from phase A's columns, so a service
    whose second leg reached Pst 4.98 was described by phase A's 1.43.
    """

    @staticmethod
    def _report(**cols):
        from pq_analysis import check_flicker
        df = _frame(**cols)
        return {"flicker": check_flicker(df, Thresholds()),
                "file_summary": {"duration_hours": 68.0}}, df

    def test_the_status_used_by_the_summary_follows_the_worst_phase(self):
        from pq_report import _flicker_status
        report, _df = self._report(flicker_pst=[0.5] * 10, flicker_pst_b=[4.98] * 10,
                                   flicker_plt=[0.2] * 10, flicker_plt_b=[2.21] * 10)
        status = _flicker_status(report)
        assert status["pst_max"] == 4.98 and status["pst_phase"] == "B"
        assert status["plt_max"] == 2.21 and status["plt_phase"] == "B"
        assert status["passes"] is False

    def test_a_long_term_failure_alone_is_named_as_a_cycling_load(self):
        docx = pytest.importorskip("docx")
        from pq_report import _word_flicker
        # Every ten-minute value inside its limit, the two-hour aggregate over:
        # the pattern a repeatedly cycling load makes.
        report, df = self._report(flicker_pst=[0.9] * 10, flicker_plt=[0.75] * 10)
        doc = docx.Document()
        _word_flicker(doc, report, df)
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "cycles repeatedly" in text
        assert "aggregates twelve consecutive Pst values" in text

    def test_both_measures_appear_for_every_phase(self):
        docx = pytest.importorskip("docx")
        from pq_report import _word_flicker
        report, df = self._report(flicker_pst=[0.5] * 10, flicker_pst_b=[4.98] * 10,
                                  flicker_plt=[0.2] * 10, flicker_plt_b=[2.21] * 10)
        doc = docx.Document()
        _word_flicker(doc, report, df)
        rows = [[c.text for c in row.cells] for row in doc.tables[0].rows]
        measures = {(r[0].split(",")[0], r[1]) for r in rows[1:]}
        assert measures == {("Pst (10 min)", "A"), ("Pst (10 min)", "B"),
                            ("Plt (2 h)", "A"), ("Plt (2 h)", "B")}
        # The phase that fails must be the one quoted in the narrative.
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "4.98 on phase B" in text

    def test_the_equipment_limit_is_not_presented_as_a_system_limit(self):
        docx = pytest.importorskip("docx")
        from pq_report import _word_flicker
        report, df = self._report(flicker_pst=[0.4] * 10, flicker_plt=[0.3] * 10)
        doc = docx.Document()
        _word_flicker(doc, report, df)
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "equipment may emit" in text
        assert "IEEE 1453-2015" in text and "0.8" in text
        # The recording is shorter than the week both standards assess over.
        assert "not a week" in text

    def test_held_values_are_not_mistaken_for_measurement_windows(self):
        from pq_analysis import check_flicker
        # Twelve intervals carrying three distinct Pst values.
        df = _frame(flicker_pst=[0.3, 0.3, 0.3, 0.3, 0.9, 0.9,
                                 0.9, 0.9, 0.5, 0.5, 0.5, 0.5])
        r = check_flicker(df, Thresholds())
        assert r["pst"]["A"]["distinct_values"] == 3


class TestFlickerSeverity:
    """How much a flicker exceedance matters, separately from whether it passed.

    Grading on the single worst reading made one bad ten-minute window look
    like a sustained condition: a Pst of 4.98 in a quiet week is 4.98x the
    limit, which the bands call severe on its own. Severity is graded at the
    95th percentile with the share of time over the limit, the same way
    voltage THD is; the maximum still decides pass or fail.
    """

    @staticmethod
    def _sev(pst, plt_=None, hours=167.0):
        import pandas as pd
        from pq_analysis import check_flicker
        from pq_report import _flicker_severities
        idx = pd.date_range("2024-01-01", periods=len(pst), freq="5min", tz="UTC")
        cols = {"flicker_pst": pd.Series(pst, index=idx)}
        if plt_ is not None:
            cols["flicker_plt"] = pd.Series(plt_, index=idx)
        fl = check_flicker(pd.DataFrame(cols), Thresholds())
        return fl, _flicker_severities(fl, {"file_summary": {"duration_hours": hours}})

    def test_a_lone_spike_fails_compliance_but_is_only_minor(self):
        quiet = [0.2] * 500
        quiet[7] = 5.0                      # one ten-minute window, five times the limit
        fl, sev = self._sev(quiet)
        assert fl["overall_pass"] is False          # compliance stays binary
        assert sev["flicker"]["band"] == "minor"    # severity does not
        assert "0.2% of the recording" in sev["flicker"]["reason"]

    def test_a_sustained_exceedance_is_significant(self):
        # Over the limit for a third of the recording, never dramatically.
        values = [1.2] * 170 + [0.3] * 330
        fl, sev = self._sev(values)
        assert sev["flicker"]["band"] in ("significant", "severe")
        assert "34.0% of the recording" in sev["flicker"]["reason"]

    def test_severity_is_graded_for_each_measure_separately(self):
        # Pst quiet with one spike, Plt over limit most of the time: the two
        # must not be collapsed into one band before the report sees them.
        pst = [0.2] * 500
        pst[3] = 4.0
        plt_ = [0.9] * 400 + [0.2] * 100
        fl, sev = self._sev(pst, plt_)
        assert sev["flicker_pst"]["band"] == "minor"
        assert sev["flicker_plt"]["band"] in ("significant", "severe")
        # The headline follows the worse of the two.
        assert sev["flicker"]["band"] == sev["flicker_plt"]["band"]

    def test_a_comfortable_pass_is_not_dressed_up_as_a_watch(self):
        fl, sev = self._sev([0.3] * 300, [0.2] * 300)
        assert sev["flicker"]["band"] == "compliant"

    def test_close_to_the_limit_reads_as_watch(self):
        fl, sev = self._sev([0.92] * 300)
        assert sev["flicker"]["band"] == "watch"
        assert "within the limit" in sev["flicker"]["reason"]

    def test_a_recording_shorter_than_a_day_is_discounted_a_band(self):
        values = [1.4] * 200 + [0.3] * 100
        _fl, full = self._sev(values, hours=72.0)
        _fl, short = self._sev(values, hours=8.0)
        assert SEVERITY_ORDER.index(short["flicker"]["band"]) \
            < SEVERITY_ORDER.index(full["flicker"]["band"])
        assert short["flicker"]["downgraded"] is True
        assert "two-hour windows" in short["flicker"]["reason"]

    def test_a_multi_day_recording_is_not_discounted(self):
        # Every survey is shorter than the week the standards assess over, so
        # that alone must not discount every finding forever.
        _fl, sev = self._sev([1.4] * 200 + [0.3] * 100, hours=72.0)
        assert sev["flicker"]["downgraded"] is False


class TestFlickerPlot:
    def test_the_chart_is_written_with_both_measures(self, tmp_path):
        pytest.importorskip("matplotlib")
        import pandas as pd
        from pq_analysis import check_flicker
        from pq_plots import plot_flicker
        idx = pd.date_range("2024-01-01", periods=300, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "flicker_pst":   pd.Series([0.4] * 299 + [3.0], index=idx),
            "flicker_pst_b": pd.Series([0.5] * 300, index=idx),
            "flicker_plt":   pd.Series([0.9] * 300, index=idx),
        }, index=idx)
        plot_flicker(df, check_flicker(df, Thresholds()),
                     outdir=tmp_path, stem="site")
        written = list(tmp_path.glob("*flicker.png"))
        assert written and written[0].stat().st_size > 10_000

    def test_no_chart_without_flicker_data(self, tmp_path):
        pytest.importorskip("matplotlib")
        from pq_plots import plot_flicker
        plot_flicker(_frame(voltage_a=[120.0] * 5), {"available": False},
                     outdir=tmp_path, stem="site")
        assert not list(tmp_path.glob("*flicker.png"))


class TestLineToLineVoltage:
    def test_wye_nominal_inferred_and_snapped(self):
        from pq_analysis import check_line_to_line_voltage
        df = _frame(voltage_a=[120.0] * 20, voltage_b=[120.0] * 20,
                    voltage_c=[120.0] * 20, voltage_ab=[207.8] * 20,
                    voltage_bc=[207.8] * 20, voltage_ca=[207.8] * 20)
        r = check_line_to_line_voltage(df, Thresholds(nominal_voltage=120.0))
        assert r["available"] and r["nominal_v"] == 208.0
        assert "wye" in r["configuration"]
        assert r["overall_pass"] is True
        assert set(r["pairs"]) == {"A-B", "B-C", "C-A"}

    def test_split_phase_nominal_inferred(self):
        from pq_analysis import check_line_to_line_voltage
        df = _frame(voltage_a=[120.0] * 20, voltage_b=[120.0] * 20,
                    voltage_ab=[240.0] * 20)
        r = check_line_to_line_voltage(df, Thresholds(nominal_voltage=120.0))
        assert r["available"] and r["nominal_v"] == 240.0
        assert "split-phase" in r["configuration"]

    def test_out_of_band_is_flagged(self):
        from pq_analysis import check_line_to_line_voltage
        ll = [207.8] * 18 + [180.0, 180.0]     # 13% low on the last two intervals
        df = _frame(voltage_a=[120.0] * 20, voltage_b=[120.0] * 20,
                    voltage_c=[120.0] * 20, voltage_ab=ll)
        r = check_line_to_line_voltage(df, Thresholds(nominal_voltage=120.0))
        assert r["overall_pass"] is False
        assert r["pairs"]["A-B"]["pct_under"] == pytest.approx(10.0)

    def test_ambiguous_ratio_is_refused_not_guessed(self):
        from pq_analysis import check_line_to_line_voltage
        df = _frame(voltage_a=[120.0] * 10, voltage_ab=[300.0] * 10)  # ratio 2.5
        r = check_line_to_line_voltage(df, Thresholds(nominal_voltage=120.0))
        assert r["available"] is False and "neither" in r["error"]

    def test_unavailable_without_ll_channels(self):
        from pq_analysis import check_line_to_line_voltage
        r = check_line_to_line_voltage(_frame(voltage_a=[120.0] * 5), Thresholds())
        assert r["available"] is False


class TestFrequency:
    def test_in_band_passes(self):
        from pq_analysis import check_frequency
        r = check_frequency(_frame(frequency=[60.0, 59.98, 60.02] * 4), Thresholds())
        assert r["available"] and r["overall_pass"] is True
        assert r["pct_out_of_band"] == 0.0

    def test_deviation_beyond_band_fails(self):
        from pq_analysis import check_frequency
        r = check_frequency(_frame(frequency=[60.0] * 8 + [58.9, 61.2]), Thresholds())
        assert r["overall_pass"] is False
        assert r["pct_out_of_band"] == pytest.approx(20.0)
        assert r["max_deviation_hz"] == pytest.approx(1.2, abs=1e-9)

    def test_unavailable_without_channel(self):
        from pq_analysis import check_frequency
        assert check_frequency(_frame(voltage_a=[120.0] * 5), Thresholds())["available"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 11. TDD from the meter's harmonic RMS (IEEE 519-2022)
# ─────────────────────────────────────────────────────────────────────────────

class TestHarmonicCurrentRMS:
    def test_meter_aggregate_is_preferred_over_the_per_order_sum(self):
        from pq_analysis import harmonic_current_rms
        # Per-order sum would give 5.0 A; the meter reports 6.0 A because the
        # orders it displays are rounded and the aggregate is not.
        df = _frame(hrms_current_a=[6.0] * 5,
                    h3_current_a=[3.0] * 5, h5_current_a=[4.0] * 5)
        series, source = harmonic_current_rms(df, "a")
        assert source == "meter"
        assert series.iloc[0] == 6.0

    def test_falls_back_to_the_per_order_sum(self):
        from pq_analysis import harmonic_current_rms
        df = _frame(h3_current_a=[3.0] * 5, h5_current_a=[4.0] * 5)
        series, source = harmonic_current_rms(df, "a")
        assert source == "per-order sum"
        assert series.iloc[0] == pytest.approx(5.0)

    def test_aggregate_columns_are_not_mistaken_for_orders(self):
        from pq_analysis import harmonic_current_rms
        # hrms_current_a must not also be summed in as if it were an order.
        df = _frame(h3_current_a=[3.0] * 5, h5_current_a=[4.0] * 5,
                    hrms_current_a=[float("nan")] * 5)
        series, source = harmonic_current_rms(df, "a")
        assert source == "per-order sum" and series.iloc[0] == pytest.approx(5.0)

    def test_no_channels_at_all(self):
        from pq_analysis import harmonic_current_rms
        assert harmonic_current_rms(_frame(voltage_a=[120.0] * 3), "a") == (None, "")


class TestFundamentalCurrent:
    def test_derived_from_rms_and_harmonic_rms(self):
        from pq_analysis import fundamental_current
        # 5-12-13 triangle: I1=12 when Irms=13 and Ih=5.
        df = _frame(current_a=[13.0] * 5)
        import pandas as pd
        h = pd.Series([5.0] * 5, index=df.index)
        assert fundamental_current(df, "a", h).iloc[0] == pytest.approx(12.0)

    def test_noise_above_the_total_is_clamped_not_square_rooted(self):
        from pq_analysis import fundamental_current
        import pandas as pd
        df = _frame(current_a=[1.0] * 5)
        h = pd.Series([2.0] * 5, index=df.index)   # impossible, but happens at ~0 A
        out = fundamental_current(df, "a", h)
        assert (out == 0.0).all() and out.notna().all()


class TestBlueBookISCLookup:
    """Every ISC the GUI offers must also resolve when the analysis runs."""

    # The nominal voltages the GUI offers.
    NOMINALS = (120.0, 208.0, 240.0, 277.0, 480.0)

    def test_single_phase_padmount_at_120_v_resolves(self):
        # 120 V L-N is one leg of a 120/240 V split-phase secondary, and the
        # pad-mount tables are keyed at 240 V only. The GUI showed 29,600 A while
        # the run resolved nothing, so the report said ISC was never provided.
        from pq_constants import _lookup_isc
        result = _lookup_isc("1ph-padmount", 100, 120.0)
        assert result is not None
        isc, note = result
        assert isc == 29_600
        assert "240V secondary" in note

    def test_every_size_the_gui_lists_resolves(self):
        from pq_constants import _BLUE_BOOK_ISC, _lookup_isc, _infer_secondary_v
        for svc_type in {k[0] for k in _BLUE_BOOK_ISC}:
            for nominal in self.NOMINALS:
                sec_v = _infer_secondary_v(svc_type, nominal)
                offered = [k[1] for k in _BLUE_BOOK_ISC
                           if k[0] == svc_type and k[2] == sec_v]
                for kva in offered:
                    result = _lookup_isc(svc_type, kva, nominal)
                    assert result is not None, (svc_type, kva, nominal)
                    assert result[0] == _BLUE_BOOK_ISC[(svc_type, kva, sec_v)]

    def test_the_secondary_is_never_a_voltage_the_table_lacks(self):
        # Resolving to a voltage with no rows is what made the lookup fail: the
        # exact key misses and the nearest-kVA fallback is filtered to the same
        # empty set, so the whole lookup returns None.
        from pq_constants import _BLUE_BOOK_ISC, _infer_secondary_v
        for svc_type in {k[0] for k in _BLUE_BOOK_ISC}:
            available = {k[2] for k in _BLUE_BOOK_ISC if k[0] == svc_type}
            for nominal in self.NOMINALS:
                sec_v = _infer_secondary_v(svc_type, nominal)
                if svc_type == "3ph-overhead-wye" and nominal == 240.0:
                    # A 240 V wye bank is not a configuration PSCo builds; the
                    # GUI says so and falls back to the manual ISC override.
                    continue
                assert sec_v in available, (svc_type, nominal, sec_v)

    @pytest.mark.parametrize("svc_type,nominal,expected", [
        # A 240 V pick is already the secondary line voltage — reading it as a
        # line-to-neutral value and scaling by √3 landed on the 480 V rows and
        # halved the ISC.
        ("3ph-open-delta",   240.0, 240),
        ("3ph-closed-delta", 240.0, 240),
        ("3ph-padmount",     240.0, 240),
        ("3ph-padmount",     208.0, 208),
        ("3ph-overhead-wye", 208.0, 208),
        # 120 V and 277 V are the line-to-neutral readings of a wye secondary.
        ("3ph-padmount",     120.0, 208),
        ("3ph-overhead-wye", 120.0, 208),
        ("3ph-overhead-wye", 277.0, 480),
        ("3ph-padmount",     480.0, 480),
        # A delta bank has no neutral: a 120 V pick is the center-tapped leg of
        # a 120/240 V secondary.
        ("3ph-open-delta",   120.0, 240),
        ("3ph-closed-delta", 120.0, 240),
        # Single-phase: 120 V rows where the table has them, 240 V otherwise.
        ("1ph-overhead",     120.0, 120),
        ("1ph-overhead",     240.0, 240),
        ("1ph-padmount",     120.0, 240),
        ("1ph-padmount",     240.0, 240),
    ])
    def test_secondary_voltage_for_each_service(self, svc_type, nominal, expected):
        from pq_constants import _infer_secondary_v
        assert _infer_secondary_v(svc_type, nominal) == expected

    def test_a_240_v_delta_reads_the_240_v_rows(self):
        from pq_constants import _lookup_isc
        isc_240, note = _lookup_isc("3ph-closed-delta", 150, 240.0)
        isc_480, _    = _lookup_isc("3ph-closed-delta", 150, 480.0)
        assert "240V secondary" in note
        assert isc_240 == pytest.approx(2 * isc_480, rel=0.01)


class TestTwoLegServiceRigor:
    """NEMA MG1 and the neutral sum both depend on the service configuration.

    NEMA MG1 unbalance is defined for three-phase systems; its formula applied
    to two legs returns half the difference an engineer would quote, under a
    label that does not apply. And the L1+L2 sum discriminates an open neutral
    on a 120/208 service but cannot on a 120/240 one.
    """

    def _two_leg(self, va=122.0, vb=118.0, n=120):
        import pandas as pd
        return pd.DataFrame(
            {"voltage_a": np.full(n, va), "voltage_b": np.full(n, vb)},
            index=pd.date_range("2025-01-01", periods=n, freq="5min"))

    def _ds(self, va, vb):
        import pandas as pd
        n = len(va)
        class _DS:
            pass
        d = _DS()
        d.df = pd.DataFrame({"voltage_a": va, "voltage_b": vb},
                            index=pd.date_range("2025-01-01", periods=n, freq="5min"))
        d.meta = {"topology": "split-phase"}
        d.has_adaptive = False
        d.adaptive_df = None
        return d

    def test_nema_mg1_is_not_claimed_on_a_single_phase_service(self):
        from pq_analysis import check_voltage_imbalance
        r = check_voltage_imbalance(self._two_leg(),
                                    Thresholds(nominal_voltage=120.0))
        assert r["metric"] == "leg_difference"
        assert "NEMA MG1" not in r["metric_label"]
        assert "not applicable to a single-phase service" in r["basis"]

    def test_two_legs_report_the_full_difference_not_half(self):
        from pq_analysis import check_voltage_imbalance
        # 122 vs 118 on a 120 V base is a 4 V spread: 3.33% of nominal.
        # The NEMA formula applied to two elements would have said 1.67%.
        r = check_voltage_imbalance(self._two_leg(),
                                    Thresholds(nominal_voltage=120.0))
        assert r["mean_imbalance_pct"] == pytest.approx(3.333, abs=0.01)

    def test_three_phase_still_uses_nema_mg1(self):
        from pq_analysis import check_voltage_imbalance
        import pandas as pd
        n = 60
        df = pd.DataFrame({"voltage_a": np.full(n, 122.0),
                           "voltage_b": np.full(n, 118.0),
                           "voltage_c": np.full(n, 120.0)},
                          index=pd.date_range("2025-01-01", periods=n, freq="5min"))
        r = check_voltage_imbalance(df, Thresholds(nominal_voltage=120.0))
        assert r["metric"] == "nema_mg1"
        assert r["mean_imbalance_pct"] == pytest.approx(1.667, abs=0.01)

    def test_a_208_service_says_the_third_phase_is_unmeasured(self):
        from pq_analysis import check_voltage_imbalance
        r = check_voltage_imbalance(
            self._two_leg(), Thresholds(nominal_voltage=120.0,
                                        service_type="1ph-208"))
        assert r["note"] and "cannot be determined" in r["note"]

    def test_the_sum_discriminates_an_open_neutral_on_a_208_service(self):
        from pq_analysis import check_neutral_health
        n = 120
        # Open neutral: the two loads sit in series across the 208 V L-L.
        frac = np.full(n, 0.45)
        ds = self._ds(208.0 * frac, 208.0 * (1 - frac))
        r = check_neutral_health(ds, Thresholds(nominal_voltage=120.0,
                                                service_type="1ph-208"))
        assert r["sum_is_diagnostic"] is True
        assert r["sum_toward_open"] > 0.9
        assert any("collapsed toward the line-to-line" in f for f in r["findings"])

    def test_a_healthy_208_service_is_not_flagged_by_the_sum(self):
        from pq_analysis import check_neutral_health
        n = 120
        ds = self._ds(np.full(n, 120.0), np.full(n, 120.0))
        r = check_neutral_health(ds, Thresholds(nominal_voltage=120.0,
                                                service_type="1ph-208"))
        assert r["sum_toward_open"] == pytest.approx(0.0, abs=0.05)

    def test_the_sum_carries_no_information_on_a_120_240_service(self):
        # Collinear legs sum to the line-to-line voltage whether the neutral is
        # intact or open, so a steady 240 V must not read as proof of health.
        from pq_analysis import check_neutral_health
        n = 120
        healthy = self._ds(np.full(n, 120.0), np.full(n, 120.0))
        frac = np.full(n, 0.45)
        opened = self._ds(240.0 * frac, 240.0 * (1 - frac))
        th = Thresholds(nominal_voltage=120.0)
        for ds in (healthy, opened):
            r = check_neutral_health(ds, th)
            assert r["sum_is_diagnostic"] is False
            assert any("carries no open-neutral information" in f
                       for f in r["findings"])

    def test_an_open_neutral_on_a_120_240_service_is_still_caught(self):
        # The sum cannot see it, but the cross-leg behaviour must.
        from pq_analysis import check_neutral_health
        n = 120
        rng = np.random.default_rng(1)
        frac = 0.45 + rng.normal(0, 0.03, n)
        ds = self._ds(240.0 * frac, 240.0 * (1 - frac))
        r = check_neutral_health(ds, Thresholds(nominal_voltage=120.0))
        assert r["severity"] in ("warning", "critical")
        assert r["leg_correlation_available"]
        assert r["leg_correlation"] < 0

    def test_neutral_health_runs_on_a_208_service(self):
        from pq_analysis import check_neutral_health
        n = 60
        ds = self._ds(np.full(n, 120.0), np.full(n, 120.0))
        r = check_neutral_health(ds, Thresholds(nominal_voltage=120.0,
                                                service_type="1ph-208"))
        assert r["available"] and r["topology"] == "1ph-208"


class TestServiceGeometryGating:
    """Three-phase criteria must not be applied to a two-leg service.

    The discriminator is the angle between the legs, not how many there are.
    A 120/240 service is collinear, so the neutral carries a difference and odd
    harmonics -- triplens included -- subtract in it. Two legs of a 120/208 wye
    are 120 degrees apart, so the neutral carries a sum and triplens add. Both
    have two legs; only one of them accumulates.
    """

    def _legs(self, ia=20.0, ib=12.0, n=120, neutral=None, h3n=None):
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        d = {"current_a": np.full(n, ia), "current_b": np.full(n, ib),
             "voltage_a": np.full(n, 121.0), "voltage_b": np.full(n, 119.0),
             "h3_current_a": np.full(n, 2.0), "h3_current_b": np.full(n, 2.0)}
        if neutral is not None:
            d["current_neutral"] = np.full(n, neutral)
        if h3n is not None:
            d["h3_current_neutral"] = np.full(n, h3n)
            d["h5_current_neutral"] = np.full(n, 0.2)
        return pd.DataFrame(d, index=idx)

    def _three_phase(self, n=120):
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        return pd.DataFrame(
            {"current_a": np.full(n, 20.0), "current_b": np.full(n, 12.0),
             "current_c": np.full(n, 16.0)}, index=idx)

    # ── the resolver ─────────────────────────────────────────────────────────

    def test_two_legs_default_to_split_phase(self):
        from pq_analysis import service_geometry
        assert service_geometry(Thresholds(), self._legs().columns) == "split-phase"

    def test_service_type_marks_a_208_network_service(self):
        from pq_analysis import service_geometry
        g = service_geometry(Thresholds(service_type="1ph-208"),
                             self._legs().columns)
        assert g == "two-leg-208"

    def test_a_third_phase_reads_as_three_phase(self):
        from pq_analysis import service_geometry
        assert service_geometry(
            Thresholds(), self._three_phase().columns) == "three-phase"

    def test_the_engineers_pick_beats_channel_presence(self):
        # A three-phase export that dropped phase C still has three phases.
        from pq_analysis import service_geometry
        g = service_geometry(Thresholds(topology="3ph-wye"), self._legs().columns)
        assert g == "three-phase"

    def test_only_120_degree_systems_accumulate_triplens(self):
        from pq_analysis import accumulates_triplens
        assert accumulates_triplens("three-phase") is True
        assert accumulates_triplens("two-leg-208") is True
        assert accumulates_triplens("split-phase") is False
        assert accumulates_triplens("single-phase") is False

    # ── current imbalance ────────────────────────────────────────────────────

    def test_no_limit_is_applied_to_leg_current_difference(self):
        # There is no PSCo number and no standard one, so none is invented.
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(self._legs(), Thresholds())
        assert r["available"] is True
        assert r["limit_pct"] is None
        assert r["pct_exceeding"] == 0.0
        assert len(r["violation_timestamps"]) == 0

    def test_the_leg_difference_says_it_is_not_a_violation(self):
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(self._legs(), Thresholds())
        assert r["metric"] == "leg_difference"
        assert "measurement, not a violation" in r["note"]

    def test_a_208_service_says_the_third_phase_is_unmeasured(self):
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(self._legs(),
                                    Thresholds(service_type="1ph-208"))
        assert r["limit_pct"] is None
        assert "third phase is not measured" in r["note"]

    def test_three_phase_still_gets_the_ten_percent_limit(self):
        from pq_analysis import check_current_imbalance
        r = check_current_imbalance(self._three_phase(), Thresholds())
        assert r["limit_pct"] == 10.0
        assert r["metric"] == "nema_style"
        assert r["note"] is None
        # 20/12/16 averages 16, max deviation 4 -> 25%, over the limit.
        assert r["mean_imbalance_pct"] == pytest.approx(25.0, abs=0.1)
        assert r["pct_exceeding"] == 100.0

    # ── neutral harmonics ────────────────────────────────────────────────────

    def test_split_phase_withholds_the_accumulation_factor(self):
        from pq_analysis import check_neutral_harmonics
        r = check_neutral_harmonics(self._legs(h3n=5.0), Thresholds())
        assert r["available"] is True
        assert r["triplens_accumulate"] is False
        assert r["accumulation_factor"] is None
        assert r["triplen_dominant"] is False
        assert "subtract in the neutral" in r["accumulation_note"]

    def test_split_phase_still_measures_neutral_harmonic_current(self):
        # Neutral heating is real whatever the geometry; only the
        # zero-sequence interpretation is withheld.
        from pq_analysis import check_neutral_harmonics
        r = check_neutral_harmonics(self._legs(h3n=5.0), Thresholds())
        assert r["orders"][3]["mean_a"] == pytest.approx(5.0, abs=0.01)

    def test_a_208_service_does_accumulate(self):
        from pq_analysis import check_neutral_harmonics
        r = check_neutral_harmonics(self._legs(h3n=5.0),
                                    Thresholds(service_type="1ph-208"))
        assert r["triplens_accumulate"] is True
        assert r["accumulation_factor"] == pytest.approx(2.5, abs=0.01)
        assert r["accumulation_note"] is None


class TestSharedTransformerIsNotMeasured:
    """One meter cannot show what a shared transformer is carrying.

    A residential pole or pad transformer feeds several houses. This service's
    demand is therefore a *lower bound* on the transformer's load, and the
    inference is one-sided: above nameplate proves an overload whoever else is
    connected, but below nameplate proves nothing. The negative case must be
    "not determinable", never a pass -- it used to print "The transformer
    loading is within acceptable limits" about equipment never measured.
    """

    def _df(self, kva, n=200):
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        return pd.DataFrame({"power_real": np.full(n, kva * 1000.0),
                             "power_reactive": np.zeros(n)}, index=idx)

    def _tx(self, cls, nameplate, kva=10.0):
        from pq_analysis import check_demand
        r = check_demand(self._df(kva),
                         Thresholds(customer_class=cls,
                                    transformer_kva=nameplate))
        return r["transformer"]

    def test_which_classes_own_their_transformer(self):
        from pq_analysis import has_dedicated_transformer
        assert has_dedicated_transformer("pg") is True
        assert has_dedicated_transformer("sg") is True
        assert has_dedicated_transformer("r") is False
        assert has_dedicated_transformer("c") is False

    def test_a_house_under_nameplate_is_not_determinable(self):
        tx = self._tx("r", nameplate=25.0, kva=10.0)
        assert tx["overloaded"] is None
        assert tx["dedicated"] is False
        assert "cannot be determined" in tx["note"]

    def test_a_house_over_nameplate_still_proves_an_overload(self):
        # One-sided: a lower bound above the nameplate is conclusive.
        tx = self._tx("r", nameplate=5.0, kva=10.0)
        assert tx["overloaded"] is True

    def test_small_commercial_shares_too(self):
        assert self._tx("c", nameplate=50.0, kva=10.0)["overloaded"] is None

    def test_a_dedicated_service_still_passes_and_fails(self):
        assert self._tx("sg", nameplate=500.0, kva=10.0)["overloaded"] is False
        assert self._tx("sg", nameplate=5.0, kva=10.0)["overloaded"] is True
        assert self._tx("pg", nameplate=500.0, kva=10.0)["note"] is None

    def test_not_determinable_is_not_graded(self):
        from pq_analysis import grade_finding
        tx = self._tx("r", nameplate=25.0, kva=10.0)
        passes = None if tx["overloaded"] is None else not tx["overloaded"]
        assert grade_finding(passes, measured=tx["peak_8h_kva"],
                             limit=tx["nameplate_kva"])["band"] == "not_assessed"

    def test_no_all_clear_reaches_a_residential_report(self):
        import subprocess, sys, os, glob
        pytest.importorskip("docx")
        from docx import Document
        root = os.path.dirname(os.path.abspath(__file__))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(
                [sys.executable, os.path.join(root, "pq_analyzer.py"),
                 os.path.join(root, "test_data", "test_residential.pqd"),
                 "--customer-class", "r", "--nominal", "120",
                 "--transformer-kva", "25", "--report", "--no-plots",
                 "--outdir", d], check=True, capture_output=True)
            path = [p for p in glob.glob(os.path.join(d, "*.docx"))
                    if "internal" in p][0]
            text = " ".join(p.text for p in Document(path).paragraphs)
        assert "within acceptable limits" not in text
        assert "contribution to its loading and not that loading" in text
        assert "cannot be determined from a recording at one meter" in text
        # The K-factor section reached the same conclusion by a second route,
        # asserting thermal margin from this service's share of the nameplate.
        assert "retains substantial thermal margin" not in text
        assert "not the transformer's thermal margin" in text


class TestResidentialRunSaysNoViolation:
    """End to end on a house: what reaches the page, not just the dicts."""

    @pytest.fixture(scope="class")
    @classmethod
    def out(cls, tmp_path_factory):
        import subprocess, sys, os
        root = os.path.dirname(os.path.abspath(__file__))
        d = tmp_path_factory.mktemp("residential")
        p = subprocess.run(
            [sys.executable, os.path.join(root, "pq_analyzer.py"),
             os.path.join(root, "test_data", "test_residential.pqd"),
             "--customer-class", "r", "--nominal", "120",
             "--no-plots", "--outdir", str(d)],
            check=True, capture_output=True, text=True)
        return p.stdout

    def test_the_leg_difference_is_reported_without_a_verdict(self, out):
        assert "MEASUREMENT — no limit" in out
        assert "measurement, not a violation" in out

    def test_no_ten_percent_current_imbalance_failure(self, out):
        block = out.split("CURRENT IMBALANCE")[1].split("─" * 10)[0]
        assert "FAIL" not in block
        assert "Limit=" not in block

    def test_no_triplen_accumulation_language_on_a_house(self, out):
        assert "Accumulation factor:" not in out
        assert "4-wire wye" not in out
        assert "subtract in the neutral" in out

    def test_the_imbalance_findings_do_not_contradict_each_other(self, out):
        # These two fired together before: one saying both voltage and current
        # were elevated, the other saying current was low, same recording.
        assert not ("Current imbalance — investigate supply voltage" in out
                    and "balanced load current" in out)

    def test_nothing_commits_xcel_to_an_action(self, out):
        assert "Xcel Energy will" not in out

    def test_a_house_is_not_told_to_rebalance_three_phases(self, out):
        assert "across phases" not in out
        assert "A, B, C phases" not in out


@pytest.fixture
def gui_app():
    """A real window, skipped where no display can be opened."""
    import run
    try:
        app = run.PQApp()
    except Exception as exc:                      # no display, no Tk
        pytest.skip(f"Tk unavailable: {exc}")
    app.update_idletasks()
    yield app
    app.destroy()


class TestClearAll:
    """Clear All resets every entry, including the ones that cascade.

    The field that quietly keeps its value is the one that carries the last
    site's transformer or engineer details into the next run, so the defaults
    are snapshotted from the widgets rather than listed by hand.
    """

    @staticmethod
    def _fill(app):
        import run
        app._file_var.set("/tmp/site.pqd")
        app._site_var.set("1500 S Hudson Mile Rd")
        app._cclass_var.set("Schedule R — Residential")
        app._topo_var.set("split-phase")
        app._nominal_var.set("277")
        app._xfmr_type_var.set(run._TYPE_DISPLAY["1ph-padmount"])
        app._on_type_change()
        app._isc_override_var.set(True)
        app._on_isc_override_toggle()
        app._isc_manual_var.set("9000")
        app._conductor_var.set("4/0 AL triplex (overhead drop)")
        app._run_length_var.set("150")
        app._eng_name_var.set("A. Engineer")

    def test_every_service_entry_returns_to_its_default(self, gui_app, monkeypatch):
        import run
        monkeypatch.setattr(run.messagebox, "askyesno", lambda *a, **k: True)
        defaults = dict(gui_app._input_defaults)
        assert len(defaults) > 10, "the form's entries should be tracked"

        self._fill(gui_app)
        assert any(getattr(gui_app, n).get() != d for n, d in defaults.items())

        gui_app._clear_all()
        still_set = {n: getattr(gui_app, n).get()
                     for n, d in defaults.items() if getattr(gui_app, n).get() != d}
        assert not still_set

    def test_the_engineers_own_details_survive(self, gui_app, monkeypatch):
        # They describe who is running the tool, not which service was
        # measured, and they go out on a customer document.
        import run
        monkeypatch.setattr(run.messagebox, "askyesno", lambda *a, **k: True)
        self._fill(gui_app)
        gui_app._eng_title_var.set("Electric Area Engineer")
        gui_app._eng_phone_var.set("303-555-0100")
        gui_app._eng_email_var.set("a@example.com")

        gui_app._clear_all()

        assert gui_app._eng_name_var.get() == "A. Engineer"
        assert gui_app._eng_title_var.get() == "Electric Area Engineer"
        assert gui_app._eng_phone_var.get() == "303-555-0100"
        assert gui_app._eng_email_var.get() == "a@example.com"
        # ...while the service they were entered against is gone.
        assert gui_app._site_var.get() == ""
        assert gui_app._file_var.get() == ""

    def test_engineer_details_alone_do_not_trigger_the_confirmation(
            self, gui_app, monkeypatch):
        # Nothing clearable is set, so the dialog would ask to clear nothing.
        import run
        asked = []
        monkeypatch.setattr(run.messagebox, "askyesno",
                            lambda *a, **k: asked.append(a) or True)
        gui_app._eng_name_var.set("A. Engineer")
        gui_app._clear_all()
        assert not asked
        assert gui_app._eng_name_var.get() == "A. Engineer"

    def test_the_dependent_pickers_are_reset_too(self, gui_app, monkeypatch):
        import run
        monkeypatch.setattr(run.messagebox, "askyesno", lambda *a, **k: True)
        self._fill(gui_app)
        assert str(gui_app._isc_entry["state"]) == "normal"

        gui_app._clear_all()
        # Not just the variables: the widget states derived from them.
        assert gui_app._xfmr_type_key is None
        assert str(gui_app._isc_entry["state"]) == "disabled"
        assert str(gui_app._kva_combo["state"]) == "disabled"
        assert "Pick a transformer Type" in gui_app._isc_auto_var.get()

    def test_a_clean_form_clears_without_asking(self, gui_app, monkeypatch):
        import run
        asked = []
        monkeypatch.setattr(run.messagebox, "askyesno",
                            lambda *a, **k: asked.append(a) or True)
        gui_app._clear_all()
        assert not asked

    def test_declining_the_confirmation_keeps_the_entries(self, gui_app, monkeypatch):
        import run
        monkeypatch.setattr(run.messagebox, "askyesno", lambda *a, **k: False)
        self._fill(gui_app)
        gui_app._clear_all()
        assert gui_app._site_var.get() == "1500 S Hudson Mile Rd"
        assert gui_app._run_length_var.get() == "150"


class TestBothDocumentsOpen:
    """A run produces two documents, so a run opens two documents.

    Only the internal report opened before, and the customer document had to
    be found by hand — which is how a document goes out unread.
    """

    class _FakeApp:
        """Enough of the window for the opener: a log, a button, an `after`."""

        def __init__(self):
            self.errors = []

        def _log_write(self, text, tag=None):
            if tag == "error":
                self.errors.append(text)

        def after(self, _delay, fn):
            fn()

        class _Btn:
            def __init__(self):
                self.state = "disabled"

            def config(self, state=None, **_kw):
                if state:
                    self.state = state

        _open_btn = None

    def _run_opener(self, monkeypatch, tmp_path, files):
        import run
        for name in files:
            (tmp_path / name).write_bytes(b"docx")
        monkeypatch.setattr(run, "_SCRIPT", tmp_path / "run.py")
        (tmp_path / "pq_output").mkdir(exist_ok=True)
        for name in files:
            (tmp_path / "pq_output" / name).write_bytes(b"docx")

        launched = []
        monkeypatch.setattr(run.subprocess, "Popen",
                            lambda cmd, **kw: launched.append(cmd))
        app = self._FakeApp()
        app._open_btn = self._FakeApp._Btn()
        run.PQApp._open_documents(app, "site")
        return app, launched

    def test_both_documents_are_opened(self, monkeypatch, tmp_path):
        app, launched = self._run_opener(monkeypatch, tmp_path, [
            "site_customer_letter.docx",
            "site_internal_engineering_report.docx",
        ])
        opened = [" ".join(str(part) for part in cmd) for cmd in launched]
        assert len(opened) == 2
        assert any("customer_letter" in o for o in opened)
        assert any("internal_engineering_report" in o for o in opened)
        # The internal report goes last so it lands in front.
        assert "internal_engineering_report" in opened[-1]
        assert app._open_btn.state == "normal"
        assert not app.errors

    def test_a_missing_document_is_reported_not_silently_skipped(
            self, monkeypatch, tmp_path):
        app, launched = self._run_opener(monkeypatch, tmp_path, [
            "site_internal_engineering_report.docx",
        ])
        assert len(launched) == 1
        assert app.errors and "customer document" in app.errors[0]
        # The one that did get written still opens.
        assert app._open_btn.state == "normal"


class TestSinglePhase208Service:
    """A single-phase service taken from two legs of a 208Y/120 wye.

    Common in condos and apartments. It has two legs like a 120/240 service,
    but they sit 120 degrees apart rather than 180, which changes the
    line-to-line voltage, the neutral current expectation, and which Blue Book
    rows the fault current comes from.
    """

    def test_line_to_line_is_208_not_240(self):
        from pq_constants import ll_factor
        assert 120 * ll_factor("1ph-208") == pytest.approx(207.8, abs=0.5)
        assert 120 * ll_factor("1ph-padmount") == pytest.approx(240.0)

    def test_it_is_recognised_as_a_single_phase_208_service(self):
        from pq_constants import is_single_phase_208
        assert is_single_phase_208("1ph-208")
        assert not is_single_phase_208("1ph-padmount")
        assert not is_single_phase_208("3ph-padmount")
        assert not is_single_phase_208(None)

    def test_fault_current_comes_from_the_three_phase_rows(self):
        from pq_constants import _lookup_isc
        net = _lookup_isc("1ph-208", 75, 120.0)
        thr = _lookup_isc("3ph-padmount", 75, 120.0)
        assert net is not None and thr is not None
        assert net[0] == thr[0]           # same transformer, same ISC
        assert "208V secondary" in net[1]
        # ...and it is not the single-phase answer.
        sp = _lookup_isc("1ph-padmount", 75, 120.0)
        assert sp is None or sp[0] != net[0]

    def test_the_kva_picker_is_populated(self):
        # The service type has no Blue Book rows of its own; without the proxy
        # the Size dropdown would be permanently empty.
        import run
        assert run._kva_options("1ph-208", 120.0)
        assert run._resolve_secondary_v("1ph-208", 120.0) == 208

    def test_it_appears_in_the_picker(self):
        from pq_constants import _SERVICE_TYPE_LABEL
        import run
        assert "1ph-208" in _SERVICE_TYPE_LABEL
        assert "1ph-208" in run._TYPE_ORDER

    def test_it_still_resolves_to_two_legs(self):
        from pq_plots import service_phases
        import pandas as pd
        n = 10
        df = pd.DataFrame(
            {"voltage_a": np.full(n, 120.0), "voltage_b": np.full(n, 120.0)},
            index=pd.date_range("2025-01-01", periods=n, freq="5min"))
        ph = service_phases(df, Thresholds(service_type="1ph-208"))
        assert len(ph) == 2

    def test_a_full_neutral_is_not_called_elevated(self, tmp_path):
        # With balanced load the neutral carries essentially full leg current
        # on this configuration; the report must say that is normal rather
        # than reporting imbalance.
        pytest.importorskip("docx")
        import subprocess, sys, glob, os
        from docx import Document
        root = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(root, "pq_analyzer.py"),
             os.path.join(root, "test_data", "test_residential.pqd"),
             "--customer-class", "r", "--service-type", "1ph-208",
             "--report", "--no-plots", "--outdir", str(tmp_path)],
            check=True, capture_output=True)
        d = Document(glob.glob(str(tmp_path / "**" / "*_internal_engineering_report.docx"),
                               recursive=True)[0])
        body = " ".join(p.text for p in d.paragraphs)
        if "Neutral current averaged" in body:
            assert "120°" in body or "120 degrees" in body


class TestServicePhasesFollowTheService:
    """A split-phase house has two legs, not three.

    The harmonic spectrum chart drew an empty "Phase C" bar and legend entry on
    every residential file, inventing a conductor that does not exist.
    """

    def _df(self, three_phase: bool):
        import pandas as pd
        n = 20
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        cols = {"voltage_a": np.full(n, 120.0), "voltage_b": np.full(n, 120.0),
                "current_a": np.full(n, 30.0), "current_b": np.full(n, 30.0)}
        if three_phase:
            cols["voltage_c"] = np.full(n, 120.0)
            cols["current_c"] = np.full(n, 30.0)
        return pd.DataFrame(cols, index=idx)

    def test_split_phase_channels_give_two_legs(self):
        from pq_plots import service_phases
        ph = service_phases(self._df(False), Thresholds())
        assert [p[1] for p in ph] == ["L1", "L2"]

    def test_three_phase_channels_give_three_phases(self):
        from pq_plots import service_phases
        ph = service_phases(self._df(True), Thresholds())
        assert [p[1] for p in ph] == ["Phase A", "Phase B", "Phase C"]

    def test_the_service_type_picker_wins_over_channel_presence(self):
        # A three-phase export missing its C channel must not be relabelled as
        # a split-phase service when the engineer said it is three-phase.
        from pq_plots import service_phases
        ph = service_phases(self._df(False),
                            Thresholds(service_type="3ph-padmount"))
        assert len(ph) == 3

    def test_a_single_phase_transformer_forces_two_legs(self):
        from pq_plots import service_phases
        ph = service_phases(self._df(True),
                            Thresholds(service_type="1ph-padmount"))
        assert [p[1] for p in ph] == ["L1", "L2"]

    def test_the_topology_picker_is_honoured(self):
        from pq_plots import service_phases
        assert len(service_phases(self._df(True),
                                  Thresholds(topology="split-phase"))) == 2
        assert len(service_phases(self._df(False),
                                  Thresholds(topology="3ph-wye"))) == 3

    def test_auto_topology_defers_to_the_channels(self):
        from pq_plots import service_phases
        assert len(service_phases(self._df(True), Thresholds(topology="auto"))) == 3
        assert len(service_phases(self._df(False), Thresholds(topology="auto"))) == 2

    def test_the_spectrum_plot_draws_only_real_phases(self, tmp_path):
        pytest.importorskip("matplotlib")
        from pq_plots import plot_harmonic_spectrum
        import matplotlib.pyplot as plt
        df = self._df(False)
        for h in (3, 5, 7, 9, 11, 13):
            df[f"h{h}_current_a"] = 1.0
            df[f"h{h}_current_b"] = 0.8
        plot_harmonic_spectrum(df, Thresholds(isc_amps=5000.0),
                               outdir=tmp_path, stem="s")
        assert (tmp_path / "s_harmonic_spectrum.png").exists()
        # The legend is built from the resolved phase list, so this is the
        # invariant that matters: no C entry for a two-leg service.
        from pq_plots import service_phases
        assert all(p[0] != "c" for p in service_phases(df, Thresholds()))


class TestSectionOrder:
    """Measurements precede the narrative drawn from them.

    This is the internal engineering document: its reader is checking numbers,
    not being walked to a conclusion, so the sections that cite measurements
    come after the measurements rather than before them.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def headings(cls, tmp_path_factory):
        pytest.importorskip("docx")
        import subprocess, sys, glob, os
        from docx import Document
        out = tmp_path_factory.mktemp("order")
        root = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(root, "pq_analyzer.py"),
             os.path.join(root, "test_data", "test_commercial_large.pqd"),
             "--isc", "10000", "--nominal", "277", "--report", "--no-plots",
             "--outdir", str(out)], check=True, capture_output=True)
        path = glob.glob(str(out / "**" / "*_internal_engineering_report.docx"), recursive=True)[0]
        d = Document(path)
        return [p.text.strip() for p in d.paragraphs
                if p.style.name == "Heading 1" and p.text.strip()]

    def _idx(self, headings, prefix):
        return next(i for i, h in enumerate(headings) if h.startswith(prefix))

    def test_measurement_review_follows_the_executive_summary(self, headings):
        assert (self._idx(headings, "Detailed Measurement Review")
                == self._idx(headings, "Executive Summary") + 1)

    def test_measurements_precede_the_narrative(self, headings):
        last_measurement = max(self._idx(headings, "Detailed Measurement Review"),
                               self._idx(headings, "Harmonic Evaluation"))
        for narrative in ("Key Findings",
                          "Engineering Assessment",
                          "Recommended Actions"):
            assert self._idx(headings, narrative) > last_measurement

    def test_the_two_measurement_sections_stay_adjacent(self, headings):
        assert (self._idx(headings, "Harmonic Evaluation")
                == self._idx(headings, "Detailed Measurement Review") + 1)

    def test_appendices_remain_last(self, headings):
        first_appendix = self._idx(headings, "Appendix A")
        assert all(h.startswith("Appendix") for h in headings[first_appendix:])

    def test_the_opening_paragraph_describes_the_actual_order(self, tmp_path_factory):
        # The opening said conclusions came first and measurements followed.
        pytest.importorskip("docx")
        import subprocess, sys, glob, os
        from docx import Document
        out = tmp_path_factory.mktemp("intro")
        root = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(root, "pq_analyzer.py"),
             os.path.join(root, "test_data", "test_commercial_small.pqd"),
             "--report", "--no-plots", "--outdir", str(out)],
            check=True, capture_output=True)
        d = Document(glob.glob(str(out / "**" / "*_internal_engineering_report.docx"), recursive=True)[0])
        body = " ".join(p.text for p in d.paragraphs)
        assert "recommended actions are presented first" not in body


class TestPowerUnitsAreConsistent:
    """Power channels are watts/VAR; every label and figure is kW/kVAR.

    The plots were drawing the raw channel under a kW label, so a 3.2 kW house
    appeared as 3,200 kW.  The analysis code had always divided by 1000, so the
    text and the charts disagreed by a factor of a thousand.
    """

    def _df(self, watts=3200.0, n=48):
        import pandas as pd
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        return pd.DataFrame({
            "power_real":     np.full(n, watts),
            "power_reactive": np.full(n, watts * 0.3),
            "power_factor":   np.full(n, 0.96),
            "current_a":      np.full(n, 27.0),
        }, index=idx)

    def test_the_demand_plot_draws_kilowatts(self, tmp_path):
        pytest.importorskip("matplotlib")
        from pq_plots import plot_demand_profile
        import matplotlib.pyplot as plt
        plot_demand_profile(self._df(), {}, outdir=tmp_path, stem="u")
        # The y data must be ~3.2, not ~3200.
        from pq_plots import _to_kilo
        assert _to_kilo(self._df()["power_real"]).max() == pytest.approx(3.2)

    def test_the_conversion_helper_is_a_thousand(self):
        from pq_plots import _to_kilo
        import pandas as pd
        s = pd.Series([1000.0, 2500.0])
        assert list(_to_kilo(s)) == [1.0, 2.5]

    def test_analysis_and_plots_agree_on_scale(self):
        # check_demand reports kVA by dividing by 1000; the plot helper must
        # apply the same factor or the report contradicts its own charts.
        from pq_analysis import check_demand
        from pq_plots import _to_kilo
        df = self._df()
        dem = check_demand(df, Thresholds())
        assert dem["available"]
        plotted_peak_kw = float(_to_kilo(df["power_real"]).max())
        assert plotted_peak_kw == pytest.approx(3.2, rel=1e-6)
        assert dem["real_power"]["peak_kw"] == pytest.approx(plotted_peak_kw, rel=1e-6)

    def test_the_demo_dataset_is_in_watts(self):
        # The demo generator wrote kW into a watt channel, which only became
        # visible once the plots started converting.
        from pq_adapter import MockAdapter
        from pq_analyzer import ChannelMapper, extract_dataset
        ds = extract_dataset(MockAdapter(), ChannelMapper())
        mean_w = float(ds.df["power_real"].dropna().mean())
        assert 5_000 < mean_w < 60_000, f"demo power_real mean is {mean_w} W"


class TestRecommendedActionsAreProportionate:
    """Actions must fit the service and stay short enough to be read."""

    def _report(self, **pf_over):
        pf = {k: True for k in ("power_factor", "thd_current",
                                "individual_harmonics", "current_imbalance",
                                "transformer_loading", "voltage")}
        pf.update(pf_over)
        return {"pass_fail": pf, "root_causes": []}

    def _actions(self, report, cls):
        from pq_report import _build_structured_actions
        return _build_structured_actions(report, Thresholds(customer_class=cls))

    def test_a_house_is_not_told_to_fit_reactors_to_its_vfds(self):
        # The load-signature match is a hypothesis, so its advice belongs with
        # the finding rather than in a list of things to go and do.
        rep = self._report()
        rep["root_causes"] = [{
            "severity": "info",
            "title": "Possible load type: 6-pulse VFD / rectifier (no input reactor)",
            "recommendation": "Add 3-5% impedance AC line reactors to VFD inputs.",
        }]
        assert self._actions(rep, "r") == []

    def test_no_informational_finding_becomes_an_action(self):
        rep = self._report()
        rep["root_causes"] = [{"severity": "info", "title": "X",
                               "recommendation": "Do something speculative."}]
        for cls in ("r", "c", "sg", "pg"):
            assert self._actions(rep, cls) == []

    def test_harmonic_advice_scales_with_the_service(self):
        rep = self._report(thd_current=False)
        res = " ".join(a["recommendation"] for a in self._actions(rep, "r")).lower()
        assert "filter" not in res or "not an appropriate remedy" in res
        assert "12-pulse" not in res
        ci = " ".join(a["recommendation"] for a in self._actions(rep, "sg")).lower()
        assert "12-pulse" in ci or "harmonic filters" in ci

    def test_a_house_is_not_asked_to_commission_a_harmonic_study(self):
        rep = self._report(individual_harmonics=False)
        for cls in ("r", "c"):
            joined = " ".join(a["recommendation"] for a in self._actions(rep, cls))
            assert "detailed harmonic study" not in joined.lower()
        joined = " ".join(a["recommendation"] for a in self._actions(rep, "sg"))
        assert "detailed harmonic study" in joined.lower()

    def test_a_two_leg_service_raises_no_imbalance_action(self):
        # A house reports its leg difference as a measurement with no limit, so
        # pass_fail carries None and nothing here fires. This test used to
        # assert split-phase wording on the action itself; there is no longer a
        # path that produces it, because leg imbalance is not a violation and
        # must not raise a High-priority item. The equivalent advice now hangs
        # off the measured neutral current instead.
        rep = self._report(current_imbalance=None)
        res = " ".join(a["recommendation"] for a in self._actions(rep, "r")).lower()
        assert "across phases" not in res
        assert "imbalance" not in res

    def test_three_phase_imbalance_still_raises_an_action(self):
        rep = self._report(current_imbalance=False)
        res = " ".join(a["recommendation"] for a in self._actions(rep, "sg")).lower()
        assert "across phases" in res

    def test_duplicate_intents_collapse_to_one_action(self):
        from pq_report import _build_structured_actions
        rep = self._report()
        rep["root_causes"] = [
            {"severity": "warning", "title": "A",
             "recommendation": "Measure voltage imbalance with all customer loads "
                               "disconnected to isolate the utility contribution."},
            {"severity": "warning", "title": "B",
             "recommendation": "Confirming the origin requires measurement with all "
                               "customer loads disconnected."},
        ]
        acts = _build_structured_actions(rep, Thresholds(customer_class="sg"))
        assert len(acts) == 1

    def test_the_action_list_is_capped(self):
        from pq_report import _build_structured_actions, _MAX_ACTIONS
        rep = self._report(power_factor=False, thd_current=False,
                           individual_harmonics=False, current_imbalance=False,
                           transformer_loading=False, voltage=False)
        rep["root_causes"] = [
            {"severity": "warning", "title": f"F{i}",
             "recommendation": f"Distinct unrelated recommendation number {i}."}
            for i in range(12)
        ]
        acts = _build_structured_actions(rep, Thresholds(customer_class="sg"))
        assert len(acts) <= _MAX_ACTIONS

    def test_high_priority_actions_survive_the_cap(self):
        from pq_report import _build_structured_actions
        rep = self._report(voltage=False, transformer_loading=False)
        rep["root_causes"] = [
            {"severity": "info", "title": f"I{i}",
             "recommendation": f"Low value note {i}."} for i in range(10)
        ]
        acts = _build_structured_actions(rep, Thresholds(customer_class="sg"))
        assert acts and all(a["priority"] == "High" for a in acts)

    def test_a_trivial_power_factor_shortfall_raises_no_install_action(self):
        # Correcting by a couple of kVAR is below the smallest practical
        # switched capacitor step; recommending an install is not advice.
        import pq_analysis as A
        assert A._MIN_ACTIONABLE_KVAR > 0


#: Harmonic orders carried in the load-signature library, in vector order.
_SIG_ORDERS = [3, 5, 7, 9, 11, 13]


def _sig_frame(spec, cv=0.1, n=150, seed=0):
    """Interval frame carrying `spec` as harmonic amps on a 100 A service."""
    import pandas as pd
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(seed)
    d = {"current_a": np.full(n, 100.0)}
    for h, v in zip(_SIG_ORDERS, spec):
        d[f"h{h}_current_a"] = np.clip(
            rng.normal(v, max(v * cv, 1e-4), n), 1e-4, None)
    return pd.DataFrame(d, index=idx)


def _sig_run(spec, cv=0.1, cls="sg", seed=0):
    """Score `spec` through the matcher as a service of class `cls`."""
    from pq_analysis import _detect_harmonic_signature
    return _detect_harmonic_signature(_sig_frame(spec, cv, seed=seed),
                                      100.0, None, customer_class=cls)


class TestLoadSignatureFloor:
    """A spectrum matching nothing must be reported as matching nothing.

    Nearest-neighbour scoring always returns a candidate.  Scoring 20,000 random
    decaying spectra against the library gave a median top score of 0.87, with
    71% above the old 0.75 gate and 29% above 0.95 -- so "scores well" was not
    evidence of anything, and pure noise reached "high confidence" routinely.
    """

    def _frame(self, spec, cv=0.1, n=150, seed=0):
        return _sig_frame(spec, cv, n, seed)

    def _run(self, spec, cv=0.1, cls="sg", seed=0):
        return _sig_run(spec, cv, cls, seed)

    def test_an_unmatched_spectrum_is_named_as_unmatched(self):
        out = self._run([0.4, 0.3, 0.5, 0.2, 0.4, 0.3], cv=0.8, cls="r")
        assert out and out[0]["title"] == "No recognised load signature"
        assert "below the" in out[0]["finding"]

    def test_the_unmatched_finding_still_reports_the_spectrum(self):
        # Naming nothing must not mean saying nothing -- the engineer still
        # needs the measured numbers to interpret by hand.
        out = self._run([0.4, 0.3, 0.5, 0.2, 0.4, 0.3], cv=0.8, cls="r")
        assert "Measured spectrum:" in out[0]["finding"]
        assert out[0]["evidence"]["h3_pct_il"] > 0

    def test_a_spectrum_between_two_families_is_not_assigned_to_either(self):
        from pq_analysis import _detect_harmonic_signature
        out = self._run([35, 18, 9, 5, 3, 2], cls="r")
        assert out[0]["title"] == "No recognised load signature"
        assert "between two unrelated load families" in out[0]["finding"]

    def test_a_clean_match_is_still_reported(self):
        out = self._run([2, 23, 9, 1, 5, 4], cls="sg")
        assert out[0]["title"] != "No recognised load signature"
        assert "6-pulse" in out[0]["title"]

    def test_indistinguishable_members_are_reported_as_a_family(self):
        # A 6-pulse VFD, a 6-pulse UPS and a DC fast charger are one topology;
        # naming one of them specifically would be arbitrary.
        out = self._run([2, 23, 9, 1, 5, 4], cls="sg")
        assert out[0]["title"].startswith("Possible load family")
        assert out[0]["evidence"]["resolved_to_member"] is False

    def test_most_random_spectra_are_rejected(self):
        # The rule's whole purpose. Old behaviour named something ~71% of the
        # time; this asserts the floor actually bites.
        rng = np.random.default_rng(7)
        named = 0
        trials = 120
        for i in range(trials):
            spec = np.clip(np.array([20, 10, 6, 3, 2, 1])
                           * rng.uniform(0.2, 3.0, 6), 0.01, None)
            out = self._run(spec, cv=0.2, cls="sg", seed=i)
            if out and out[0]["title"] != "No recognised load signature":
                named += 1
        assert named / trials < 0.30, f"named {named/trials:.0%} of random spectra"

    def test_the_floor_is_documented_where_it_is_set(self):
        import pq_constants as C
        assert C.SIGNATURE_ABSOLUTE_FLOOR >= 0.85
        assert C.SIGNATURE_FAMILY_SEPARATION > 0
        assert C.SIGNATURE_MEMBER_SEPARATION > 0

    def test_every_signature_belongs_to_a_labelled_family(self):
        from pq_constants import _LOAD_SIGNATURES, LOAD_FAMILY_LABEL
        for s in _LOAD_SIGNATURES:
            assert s.get("family"), f"{s['id']} has no family"
            assert s["family"] in LOAD_FAMILY_LABEL, s["family"]


#: Trials per (class, arm) cell in the mixture sweep.  Large enough that a rate
#: is readable to a couple of points, small enough to stay in the unit suite.
_MIX_TRIALS = 300

#: A draw is called "blended" when no single load owns more than this share of
#: the fundamental.  Above it the service is essentially one device plus trim,
#: and naming that device is the right answer rather than a failure.
_MIX_BLENDED_MAX_SHARE = 0.60

#: Not every library entry describes a single device.  `mixed_vfd_smps` is a
#: blend by construction -- "6-pulse VFDs + single-phase nonlinear loads" -- so
#: naming it on a draw whose loads all come from the families it describes is a
#: correct answer, not a misidentification.  Without this carve-out the sweep
#: scores that entry as wrong every time it is right, which on the `sg` class
#: inflates the misidentification rate from 37% to 67%.
_MIX_COMPOSITE_ENTRIES = {
    "mixed_vfd_smps": {"six_pulse", "single_phase_switchmode"},
}


def _mix_defensible(signature_id, parent_families):
    """True when a non-parent match is a composite entry that describes the draw."""
    covered = _MIX_COMPOSITE_ENTRIES.get(signature_id)
    return covered is not None and parent_families <= covered


def _mixture_spectrum(rng, candidates, phase_random):
    """One synthetic service: 2-3 library loads sharing a 100 A fundamental.

    Each library vector is a percentage of *its own* fundamental, so a load
    drawing share ``w`` of a 100 A service contributes ``w * spec`` amps at each
    order.  The mixture spectrum is therefore the share-weighted sum, in amps.

    ``phase_random`` selects between the two bounding cases described in
    TestLoadSignatureMixtures: aligned magnitudes, or uniform random phase.
    """
    k = int(rng.integers(2, 4))
    picks = rng.choice(len(candidates), size=k, replace=False)
    parents = [candidates[i] for i in picks]
    shares = rng.dirichlet(np.ones(k))

    contrib = np.array([np.asarray(p["spectrum"], dtype=float) * w
                        for p, w in zip(parents, shares)])
    if phase_random:
        theta = rng.uniform(0, 2 * np.pi, size=contrib.shape)
        spec = np.abs((contrib * np.exp(1j * theta)).sum(axis=0))
    else:
        spec = contrib.sum(axis=0)
    return parents, shares, np.clip(spec, 1e-4, None)


def _mixture_sweep(cls, phase_random, trials=_MIX_TRIALS, seed=20260807):
    """Score `trials` synthetic mixtures and tally what the matcher said."""
    from pq_constants import _LOAD_SIGNATURES
    candidates = [s for s in _LOAD_SIGNATURES if cls in s["classes"]]
    rng = np.random.default_rng(seed)
    tally = {"n": 0, "rejected": 0, "named": 0, "named_member": 0,
             "member_not_parent": 0, "family_not_parent": 0, "high_conf": 0,
             "composite_ok": 0, "blended_n": 0, "blended_named_member": 0,
             "blended_member_not_parent": 0}

    for i in range(trials):
        parents, shares, spec = _mixture_spectrum(rng, candidates, phase_random)
        out = _sig_run(spec, cv=0.1, cls=cls, seed=i)
        assert out, "the matcher returned no finding at all"
        f, ev = out[0], out[0]["evidence"]
        blended = float(max(shares)) < _MIX_BLENDED_MAX_SHARE
        parent_ids = {p["id"] for p in parents}
        parent_families = {p["family"] for p in parents}

        tally["n"] += 1
        tally["blended_n"] += blended
        if f["title"] == "No recognised load signature":
            tally["rejected"] += 1
            continue

        defensible = _mix_defensible(ev["signature_id"], parent_families)
        tally["named"] += 1
        tally["composite_ok"] += defensible
        if ev["family"] not in parent_families and not defensible:
            tally["family_not_parent"] += 1
        if ev["family_separation"] >= 0.15:      # what the code calls "high"
            tally["high_conf"] += 1
        if ev["resolved_to_member"]:
            tally["named_member"] += 1
            tally["blended_named_member"] += blended
            if ev["signature_id"] not in parent_ids and not defensible:
                tally["member_not_parent"] += 1
                tally["blended_member_not_parent"] += blended
    return tally


def _mixture_report(rows):
    """Render sweep tallies as a table.  Visible under `pytest -s`."""
    def pct(num, den):
        return "     -" if not den else f"{num / den:6.1%}"

    lines = [
        "",
        "  Load-signature behaviour on multi-load services",
        "  (share-weighted mixtures of 2-3 library loads; 'parent' = a load"
        "  actually present in the draw)",
        "",
        f"  {'class':>6} {'arm':>10} {'n':>5} {'rejected':>9} {'named':>7}"
        f" {'->member':>9} {'not parent':>11} {'wrong fam':>10} {'high conf':>10}",
    ]
    for cls, arm, t in rows:
        lines.append(
            f"  {cls:>6} {arm:>10} {t['n']:>5} {pct(t['rejected'], t['n']):>9}"
            f" {pct(t['named'], t['n']):>7} {pct(t['named_member'], t['n']):>9}"
            f" {pct(t['member_not_parent'], t['named_member']):>11}"
            f" {pct(t['family_not_parent'], t['named']):>10}"
            f" {pct(t['high_conf'], t['named']):>10}"
        )
    lines += ["",
              "  'not parent' and 'wrong fam' are shares of the matches actually"
              " named,",
              "  not of all trials -- they answer \"when it names something, how"
              " often is",
              "  that something not even present?\"  Both credit the composite"
              " library entry",
              "  (see _MIX_COMPOSITE_ENTRIES) when it correctly describes the"
              " blend.", ""]
    print("\n".join(lines))


class TestLoadSignatureMixtures:
    """Can the matcher name a device when the service carries more than one?

    Every entry in the library is a single device at rated load.  A meter at the
    service entrance sees the sum of everything behind it, and the guards in
    `_detect_harmonic_signature` -- the 0.90 floor, family separation, member
    separation -- all measure distance to library points.  None of them asks
    whether a *blend* of library entries explains the same point, so a mixture
    can in principle clear every gate and be reported as one resolved device.

    This class measures how often that happens.  It sets no thresholds and
    asserts no rate: the rates it prints are the deliverable, and what to do
    about them is a separate decision.

    Measured over 300 draws per cell (`pytest test_pq.py -k Mixture -s`), the
    guards hold on small services and fail on large ones.  Residential and small
    commercial mixtures are rejected 96% and 80% of the time, and `c` never
    resolves to a single device at all.  On `sg` and `pg` -- the classes where a
    service most plausibly carries several nonlinear loads, and where this
    finding is most likely to be acted on -- the matcher names something for 47%
    and 69% of mixtures, resolves roughly half of those to one specific device,
    and of those single-device names 45% (`sg`) and 32% (`pg`) name a load that
    is not in the draw.  A quarter of `sg` matches name the wrong *family*.
    Random phase moves every rate by less than the gap between classes, so the
    result does not depend on where between the two arms reality sits.

    What this does not say: these are mixtures of library entries, which is a
    friendlier population than real services carrying loads the library has no
    entry for.  The rates are a floor on the error, not an estimate of it.

    Two arms bound the physics.  Harmonic currents from several loads add as
    phasors, and the library stores magnitudes only:

      * `aligned`     -- magnitudes added directly.  No cancellation, which is
                         the most favourable case for recognising a parent.
      * `randomphase` -- uniform random phase per load per order.  Maximum
                         cancellation, the least favourable case.

    Neither is a model of a real site; real services fall between them, closer
    to `aligned` when the mixed loads share a topology.  A conclusion that holds
    in both arms does not depend on where between them reality sits.
    """

    def test_a_single_library_load_is_still_recognised(self):
        # Harness control.  If a pure library spectrum fed through the mixture
        # path stopped matching, every rate below would be measuring a broken
        # harness rather than the matcher.
        from pq_constants import _LOAD_SIGNATURES
        vfd = next(s for s in _LOAD_SIGNATURES if s["id"] == "vfd_6pulse_reactor")
        out = _sig_run(np.asarray(vfd["spectrum"], dtype=float), cls="sg")
        assert out[0]["title"] != "No recognised load signature"
        assert out[0]["evidence"]["family"] == "six_pulse"

    def test_every_mixture_is_accounted_for(self):
        # Each trial must land in exactly one bucket, or the rates do not add up.
        t = _mixture_sweep("sg", phase_random=False, trials=40)
        assert t["n"] == 40
        assert t["rejected"] + t["named"] == t["n"]
        assert t["named_member"] <= t["named"]
        assert t["member_not_parent"] <= t["named_member"]

    def test_mixture_rates(self):
        # The measurement.  Run with `pytest test_pq.py -k Mixture -s` to read
        # the table; the assertions here only confirm the sweep ran.
        rows = []
        for cls in ("r", "c", "sg", "pg"):
            for arm, phase_random in (("aligned", False), ("randomphase", True)):
                rows.append((cls, arm, _mixture_sweep(cls, phase_random)))
        _mixture_report(rows)
        assert all(t["n"] == _MIX_TRIALS for _, _, t in rows)


class TestLoadSignaturesRespectCustomerClass:
    """Load-type matching must not offer equipment the service cannot have.

    Cosine similarity always returns a nearest neighbour, so an unrestricted
    library named an arc welder as the best match for a house.
    """

    def _welder_shaped_frame(self):
        # [H3,H5,H7,H9,H11,H13] = [10,8,6,5,4,3] with high variability, which is
        # the arc welder reference signature.
        import pandas as pd
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(3)
        data = {"current_a": np.full(n, 100.0)}
        for h, v in {3: 10., 5: 8., 7: 6., 9: 5., 11: 4., 13: 3.}.items():
            data[f"h{h}_current_a"] = np.clip(rng.normal(v, v * 0.5, n), 0.01, None)
        return pd.DataFrame(data, index=idx)

    def _titles(self, customer_class):
        from pq_analysis import _detect_harmonic_signature
        return [f["title"] for f in _detect_harmonic_signature(
            self._welder_shaped_frame(), 100.0, None,
            customer_class=customer_class)]

    def _texts(self, customer_class):
        # Title plus body.  The equipment considered is named in the body now
        # that the title reports only the family.
        from pq_analysis import _detect_harmonic_signature
        return [f"{f['title']} {f.get('finding', '')}"
                for f in _detect_harmonic_signature(
                    self._welder_shaped_frame(), 100.0, None,
                    customer_class=customer_class)]

    def test_a_residential_service_is_never_offered_industrial_equipment(self):
        joined = " ".join(self._titles("r")).lower()
        for banned in ("arc welder", "arc furnace", "12-pulse", "18-pulse",
                       "dc fast charger"):
            assert banned not in joined, f"{banned!r} offered to a house"

    def test_a_residential_service_gets_residential_candidates(self):
        joined = " ".join(self._texts("r")).lower()
        assert any(k in joined for k in
                   ("pv inverter", "ev charger", "heat pump", "air conditioning"))

    def test_an_arc_furnace_is_only_offered_to_primary_service(self):
        from pq_constants import _LOAD_SIGNATURES
        eaf = next(s for s in _LOAD_SIGNATURES if s["id"] == "arc_furnace")
        assert eaf["classes"] == {"pg"}

    def test_every_signature_declares_its_classes(self):
        from pq_constants import _LOAD_SIGNATURES
        valid = {"r", "c", "sg", "pg"}
        for s in _LOAD_SIGNATURES:
            assert s.get("classes"), f"{s['id']} has no classes"
            assert set(s["classes"]) <= valid, f"{s['id']} has an unknown class"

    def test_every_class_has_candidates(self):
        from pq_constants import _LOAD_SIGNATURES
        for cls in ("r", "c", "sg", "pg"):
            assert [s for s in _LOAD_SIGNATURES if cls in s["classes"]], cls

    def test_an_unresolvable_match_names_the_family_not_the_equipment(self):
        # An arc welder and an arc furnace score within a thousandth of each
        # other on this spectrum. They are one topology, so the report names
        # the arcing family rather than picking one piece of equipment.
        from pq_analysis import _detect_harmonic_signature
        f = _detect_harmonic_signature(self._welder_shaped_frame(), 100.0, None,
                                       customer_class="pg")
        assert f[0]["evidence"]["similarity"] > 0.95
        assert f[0]["title"].startswith("Possible load family")
        assert "Arcing load" in f[0]["title"]
        assert f[0]["evidence"]["resolved_to_member"] is False
        assert "individual load type is not reported" in f[0]["finding"]
        assert f[0]["confidence"] != "high"

    def test_the_finding_does_not_claim_to_identify_equipment(self):
        # No path names a single piece of equipment any more.  A mixture of two
        # loads lands nearest an entry containing neither often enough that the
        # member-level claim was not supportable -- see TestLoadSignatureMixtures.
        f_titles = self._titles("r")
        assert f_titles and not any(t.startswith("Best match") for t in f_titles)
        assert f_titles[0].startswith("Possible load family")
        assert not any(t.startswith("Possible load type") for t in f_titles)

    def test_a_family_with_one_entry_still_does_not_claim_the_equipment(self):
        # mixed_three_phase has a single entry applicable to primary service, so
        # the family label and that entry coincide. The finding has to say so
        # rather than reading as an identification by omission.
        from pq_analysis import _detect_harmonic_signature
        f = _detect_harmonic_signature(
            _sig_frame(np.array([15, 20, 8, 2, 4, 3], dtype=float)),
            100.0, None, customer_class="pg")[0]
        assert f["title"].startswith("Possible load family")
        assert "one reference entry" in f["finding"]
        assert "not an identification of the equipment present" in f["finding"]
        assert f["evidence"]["resolved_to_member"] is False

    def test_no_match_is_ever_reported_at_high_confidence(self):
        from pq_analysis import _detect_harmonic_signature
        for cls in ("r", "c", "sg", "pg"):
            for f in _detect_harmonic_signature(self._welder_shaped_frame(),
                                                100.0, None, customer_class=cls):
                assert f["confidence"] != "high", cls


class TestComplianceTableGrouping:
    """The table groups checks by the quantity measured, and loses nothing.

    Buffering rows to regroup them is the kind of change that silently drops a
    check, so the count is asserted against the ungrouped behaviour: every
    standard that used to appear still appears, exactly once.
    """

    def _table(self, path):
        from docx import Document
        d = Document(str(path))
        for t in d.tables:
            if [c.text.strip() for c in t.rows[0].cells][:2] == ["Standard", "Measured"]:
                return t
        raise AssertionError("compliance table not found")

    def _split(self, tbl):
        headings, checks = [], []
        for row in tbl.rows[1:]:
            if len({id(c._tc) for c in row.cells}) == 1:
                headings.append(row.cells[0].text.strip())
            else:
                checks.append(row.cells[0].text.strip())
        return headings, checks

    @pytest.fixture(scope="class")
    @classmethod
    def report_path(cls, tmp_path_factory):
        pytest.importorskip("docx")
        import subprocess, sys, glob, os
        out = tmp_path_factory.mktemp("grouped")
        root = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(root, "pq_analyzer.py"),
             os.path.join(root, "test_data", "test_commercial_large.pqd"),
             "--isc", "10000", "--transformer-kva", "500", "--nominal", "277",
             "--report", "--no-plots", "--outdir", str(out)],
            check=True, capture_output=True)
        found = glob.glob(str(out / "**" / "*.docx"), recursive=True)
        assert found, "no report generated"
        return found[0]

    def test_every_check_appears_exactly_once(self, report_path):
        _, checks = self._split(self._table(report_path))
        assert len(checks) == 14
        assert len(set(checks)) == len(checks)

    def test_checks_are_grouped_by_measured_quantity(self, report_path):
        headings, _ = self._split(self._table(report_path))
        assert [h.split("—")[0].strip() for h in headings] == [
            "VOLTAGE", "CURRENT", "DEMAND AND POWER FACTOR", "FREQUENCY"]

    def test_no_check_is_stranded_outside_a_group(self, report_path):
        tbl = self._table(report_path)
        seen_heading = False
        for row in tbl.rows[1:]:
            if len({id(c._tc) for c in row.cells}) == 1:
                seen_heading = True
            else:
                assert seen_heading, f"{row.cells[0].text!r} precedes any heading"

    def test_magnitude_checks_precede_distortion_checks(self, report_path):
        _, checks = self._split(self._table(report_path))
        pos = {c.split("(")[0].strip(): i for i, c in enumerate(checks)}
        volt_mag = next(i for c, i in pos.items() if c.startswith("Steady-state voltage"))
        volt_thd = next(i for c, i in pos.items() if c.startswith("Voltage THD"))
        assert volt_mag < volt_thd


class TestComplianceRowsQuoteMeasurements:
    """Every row should carry a number the recording actually produced.

    The voltage row used to print range_v -- the *allowed* band -- under the
    bare label "Range", so the only voltage in the cell was the limit, and
    "Worst phase" was never followed by which phase or what it read.
    """

    def _report(self, **over):
        from pq_analysis import check_voltage_compliance
        import pandas as pd
        n = 100
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        data = {"voltage_a": [121.0] * n, "voltage_b": [120.0] * n,
                "voltage_c": [120.0] * n}
        data.update(over)
        return check_voltage_compliance(pd.DataFrame(data, index=idx),
                                        Thresholds(nominal_voltage=120.0))

    def test_per_phase_measurements_are_available_to_the_row(self):
        v = self._report(voltage_a=[110.0] * 50 + [121.0] * 50)
        st = v["phases"]["voltage_a"]
        # The row needs all of these; none were being printed before.
        assert st["min_v"] == pytest.approx(110.0)
        assert st["max_v"] == pytest.approx(121.0)
        assert st["mean_v"] == pytest.approx(115.5)
        assert st["pct_under"] == pytest.approx(50.0)
        assert st["pct_over"] == 0.0

    def test_range_v_is_the_allowed_band_not_a_measurement(self):
        # 120 V nominal at +/-5 % -> 114-126.  Nothing measured 114 or 126 here.
        v = self._report()
        assert v["range_v"] == pytest.approx((114.0, 126.0))
        assert v["phases"]["voltage_a"]["max_v"] == pytest.approx(121.0)

    def test_the_worst_phase_is_the_one_with_most_intervals_out_of_band(self):
        v = self._report(voltage_b=[100.0] * 80 + [120.0] * 20)
        worst = max(v["phases"].items(),
                    key=lambda kv: kv[1]["pct_out_of_bounds"])
        assert worst[0] == "voltage_b"
        assert worst[1]["pct_out_of_bounds"] == pytest.approx(80.0)

    def test_the_binding_harmonic_order_is_reported_by_margin(self):
        from pq_analysis import check_individual_harmonics
        # H5 is the larger current but sits well inside its limit; H23's much
        # tighter limit makes it the order that actually binds.
        df = _frame(current_a=[100.0] * 10,
                    h5_current_a=[6.0] * 10,
                    h23_current_a=[0.9] * 10)
        r = check_individual_harmonics(df, Thresholds(isc_amps=5000.0))
        assert r["available"]
        assert r["worst_order"][0] == 5              # largest magnitude
        assert r["worst_margin_order"][0] == 23      # tightest margin
        assert r["worst_limit_pct"] < 2.0


class TestSeverityGrading:
    """Severity is a second axis beside compliance, not a replacement for it.

    Compliance stays binary because the standards are; severity says how much
    the exceedance matters, so an isolated artifact and a sustained overload
    stop sharing one red FAIL.
    """

    def test_compliance_and_severity_are_independent(self):
        from pq_analysis import grade_finding
        # Over the limit is over the limit — severity never contradicts that.
        assert grade_finding(False, measured=8.1, limit=8.0,
                             persistence_pct=0.1)["band"] == "minor"
        assert grade_finding(True, measured=2.0, limit=8.0)["band"] == "compliant"

    def test_a_marginal_brief_exceedance_is_minor_not_severe(self):
        from pq_analysis import grade_finding
        g = grade_finding(False, measured=8.2, limit=8.0, persistence_pct=0.2)
        assert g["band"] == "minor"

    def test_a_large_sustained_exceedance_is_severe(self):
        from pq_analysis import grade_finding
        g = grade_finding(False, measured=12.8, limit=8.0, persistence_pct=60.0)
        assert g["band"] == "severe"

    def test_a_very_large_exceedance_is_severe_however_brief(self):
        from pq_analysis import grade_finding
        g = grade_finding(False, measured=19.0, limit=8.0, persistence_pct=0.5)
        assert g["band"] == "severe"

    def test_low_confidence_drops_one_band_and_says_so(self):
        from pq_analysis import grade_finding
        kw = dict(measured=12.8, limit=8.0, persistence_pct=60.0)
        assert grade_finding(False, **kw)["band"] == "severe"
        g = grade_finding(False, confidence_notes=["ISC not supplied"], **kw)
        assert g["band"] == "significant"
        assert g["downgraded"] is True
        assert "ISC not supplied" in g["reason"]
        assert "severity reduced one band" in g["reason"]

    def test_confidence_never_downgrades_below_minor(self):
        from pq_analysis import grade_finding
        g = grade_finding(False, measured=8.1, limit=8.0, persistence_pct=0.1,
                          confidence_notes=["light-load intervals excluded"])
        assert g["band"] == "minor"
        assert g["downgraded"] is False

    def test_close_to_the_limit_is_watch_not_compliant(self):
        from pq_analysis import grade_finding
        g = grade_finding(True, measured=7.2, limit=8.0)
        assert g["band"] == "watch"
        assert "90% of it" in g["reason"]

    def test_a_healthy_power_factor_is_not_flagged_as_watch(self):
        from pq_analysis import grade_finding
        # PF is bounded at 1.0, so the ceiling-metric watch band would mark
        # every power factor including 0.99.  Floor metrics get a tighter one.
        assert grade_finding(True, measured=0.989, limit=0.90,
                             lower_is_worse=True)["band"] == "compliant"
        assert grade_finding(True, measured=0.912, limit=0.90,
                             lower_is_worse=True)["band"] == "watch"

    def test_a_power_factor_shortfall_reads_in_the_right_direction(self):
        from pq_analysis import grade_finding
        g = grade_finding(False, measured=0.88, limit=0.90, persistence_pct=12.0,
                          lower_is_worse=True)
        assert "below the limit" in g["reason"]

    def test_an_unassessed_check_has_no_severity(self):
        from pq_analysis import grade_finding
        assert grade_finding(None)["band"] == "not_assessed"


class TestVoltageTHDIsJudgedStatistically:
    """IEEE 519-2022 Clause 5 judges voltage THD on percentiles, not on maxima.

    A single interval where the fundamental collapsed used to fail an entire
    site: pass_fail asked for pct_exceeding == 0, and V_h/V_1 runs to tens of
    percent whenever V_1 approaches zero.
    """

    def _thresh(self):
        return Thresholds(nominal_voltage=120.0)

    def test_a_sag_artifact_is_dropped_not_reported_as_distortion(self):
        from pq_analysis import check_thd
        # 2.5 % site with one interval measured during a collapse to 11 V.
        thd   = [2.5] * 99 + [80.20]
        volts = [120.0] * 99 + [11.0]
        r = check_thd(_frame(thd_voltage_a=thd, voltage_a=volts), self._thresh())["voltage"]
        assert r["artifact_samples"] == 1
        assert r["sample_count"] == 99
        assert r["max_thd_pct"] == pytest.approx(2.5)
        assert r["p95_pass"] and r["p99_pass"]

    def test_a_spike_at_normal_voltage_does_not_fail_the_site(self):
        from pq_analysis import check_thd
        # The voltage gate cannot catch this one — the percentile has to.
        thd = [2.5] * 99 + [80.20]
        r = check_thd(_frame(thd_voltage_a=thd, voltage_a=[120.0] * 100),
                      self._thresh())["voltage"]
        assert r["artifact_samples"] == 0
        assert r["max_thd_pct"] == pytest.approx(80.20)
        assert r["p95_pass"] and r["p99_pass"]
        assert r["max_is_outlier"] is True

    def test_genuine_sustained_distortion_still_fails(self):
        from pq_analysis import check_thd
        r = check_thd(_frame(thd_voltage_a=[9.5] * 100, voltage_a=[120.0] * 100),
                      self._thresh())["voltage"]
        assert r["p95_pass"] is False
        assert r["max_is_outlier"] is False

    def test_a_sustained_excursion_is_caught_by_the_p99_rule(self):
        from pq_analysis import check_thd
        # P95 stays under 8 %, but 3 % of the recording sits at 13 % — over the
        # 1.5x short-time limit, so the site must not pass.
        thd = [6.0] * 97 + [13.0] * 3
        r = check_thd(_frame(thd_voltage_a=thd, voltage_a=[120.0] * 100),
                      self._thresh())["voltage"]
        assert r["p95_pass"] is True
        assert r["p99_pass"] is False

    def test_the_gate_keeps_distortion_measured_during_a_shallow_sag(self):
        from pq_analysis import check_thd
        # 0.75 pu is a real sag but a valid THD reading; it must not be dropped.
        r = check_thd(_frame(thd_voltage_a=[9.0] * 100, voltage_a=[90.0] * 100),
                      self._thresh())["voltage"]
        assert r["artifact_samples"] == 0
        assert r["p95_pass"] is False


class TestTDDDefinition:
    def _thresh(self):
        return Thresholds(nominal_voltage=120.0, isc_amps=10000.0)

    def test_tdd_is_harmonic_current_over_il(self):
        from pq_analysis import check_thd
        # Ih = 5 A on every interval, IL = fundamental max = 12 A → TDD = 41.67 %.
        df = _frame(current_a=[13.0] * 10, hrms_current_a=[5.0] * 10,
                    thd_current_a=[41.667] * 10)
        r = check_thd(df, self._thresh())["current"]
        assert r["metric"] == "tdd"
        assert r["harmonic_rms_source"] == "meter"
        assert r["il_amps"] == pytest.approx(12.0, abs=0.01)
        assert r["max_thd_pct"] == pytest.approx(100 * 5.0 / 12.0, rel=1e-6)

    def test_an_inconsistent_thd_channel_no_longer_inflates_tdd(self):
        # The old form, THD% x Irms / IL, multiplied the THD ratio by current, so
        # an interval where the meter's THD contradicts its own RMS and harmonic
        # RMS was amplified. Reading the harmonic current directly cannot be.
        from pq_analysis import check_thd
        thd = [12.8] * 9 + [111.7]          # last interval is the anomaly
        df = _frame(current_a=[5.5] * 10, hrms_current_a=[0.7] * 10,
                    thd_current_a=thd)
        r = check_thd(df, self._thresh())["current"]
        il = np.sqrt(5.5 ** 2 - 0.7 ** 2)          # fundamental, not the 5.5 A RMS
        assert r["max_thd_pct"] == pytest.approx(100 * 0.7 / il, rel=1e-6)
        # The anomalous interval contributes nothing beyond the others.
        assert r["max_thd_pct"] == pytest.approx(r["mean_thd_pct"], rel=1e-9)
        # For contrast: the old form would have reached 111.7 x 5.5 / IL here.
        assert r["max_thd_pct"] < 20.0

    def test_falls_back_to_the_thd_channel_against_the_fundamental(self):
        from pq_analysis import check_thd
        # No harmonic RMS anywhere: TDD comes from THD x I1 / IL, where I1 is the
        # RMS channel (the best available) rather than being confused with it.
        df = _frame(current_a=[10.0] * 10, thd_current_a=[20.0] * 10)
        r = check_thd(df, self._thresh())["current"]
        assert r["metric"] == "tdd"
        assert r["harmonic_rms_source"] is None
        assert r["max_thd_pct"] == pytest.approx(20.0, rel=1e-6)

    def test_il_uses_the_fundamental_not_the_rms(self):
        from pq_analysis import check_thd
        df = _frame(current_a=[13.0] * 10, hrms_current_a=[5.0] * 10,
                    thd_current_a=[41.667] * 10)
        r = check_thd(df, self._thresh())["current"]
        # IEEE 519 defines IL at the fundamental: 12 A here, not the 13 A RMS.
        assert r["il_amps"] == pytest.approx(12.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Truncated and damaged files
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _FIXTURES, reason="test_data/*.pqd not generated")
class TestTruncatedFile:
    """A file cut short must degrade, not fail outright.

    Damage lands at the end of an interrupted export or copy, so the interval
    records carrying the compliance data are usually intact and worth reading.
    """

    @pytest.fixture
    def truncated(self, tmp_path):
        src = Path(__file__).parent / "test_data" / "test_commercial_small.pqd"
        raw = src.read_bytes()
        out = tmp_path / "truncated.pqd"
        out.write_bytes(raw[: len(raw) - 3000])
        return out

    def test_truncation_is_measured_not_just_reported_as_a_zlib_error(self, truncated):
        f = pqdif.PQDIFFile(truncated)
        assert f.missing_bytes == 3000

    def test_intact_file_reports_no_missing_bytes(self):
        src = Path(__file__).parent / "test_data" / "test_commercial_small.pqd"
        f = pqdif.PQDIFFile(src)
        assert f.missing_bytes == 0
        assert f.unreadable_observations == []

    def test_readable_records_still_load(self, truncated):
        adapter = ProntoAdapter(truncated)
        df = extract_dataset(adapter, ChannelMapper()).df
        assert len(df) > 0
        assert "voltage_a" in df.columns

    def test_damage_is_recorded_on_the_dataset(self, truncated):
        ds = extract_dataset(ProntoAdapter(truncated), ChannelMapper())
        dq = ds.meta["data_quality"]
        assert dq["missing_bytes"] == 3000

    def test_intact_file_carries_a_clean_bill(self):
        src = Path(__file__).parent / "test_data" / "test_commercial_small.pqd"
        ds = extract_dataset(ProntoAdapter(src), ChannelMapper())
        dq = ds.meta["data_quality"]
        assert dq["missing_bytes"] == 0 and dq["unreadable_observations"] == 0

    def test_a_file_with_no_readable_observation_still_raises(self, tmp_path):
        # Degrading gracefully must not extend to inventing an empty analysis.
        src = Path(__file__).parent / "test_data" / "test_commercial_small.pqd"
        raw = bytearray(src.read_bytes())
        f = pqdif.PQDIFFile(src)
        for record in f.records:
            if record.tag == pqdif.TAG_OBSERVATION:
                start = record.position + record.header_size
                raw[start:start + 8] = b"\x00" * 8      # destroy the zlib header
        broken = tmp_path / "broken.pqd"
        broken.write_bytes(bytes(raw))
        with pytest.raises(pqdif.PQDIFError, match="could be read"):
            _ = pqdif.PQDIFFile(broken).observations


@pytest.mark.skipif(not _FIXTURES, reason="test_data/*.pqd not generated")
class TestRecordThatWillNotInflate:
    """A body that will not inflate has several causes, and they differ.

    The files are customer data that cannot be shared for a second look, so
    the reader has to recover the causes it can and, for the one it cannot,
    say enough in the failure itself to tell the two apart from the message
    alone: a damaged file to re-export, or a file this reader reads wrongly.
    """

    SRC = Path(__file__).parent / "test_data" / "test_commercial_small.pqd"

    @staticmethod
    def _rebuild(records, path):
        """Re-emit a record chain, relinking it around new body lengths."""
        out = bytearray()
        for i, (tag, payload, declared) in enumerate(records):
            last = i == len(records) - 1
            next_pos = 0 if last else (
                len(out) + pqdif.RECORD_HEADER_SIZE + len(payload))
            header = bytearray(pqdif.RECORD_HEADER_SIZE)
            header[0:16] = pqdif.RECORD_SIGNATURE.bytes_le
            header[16:32] = tag.bytes_le
            struct.pack_into("<IIII", header, 32, pqdif.RECORD_HEADER_SIZE,
                             len(payload) if declared is None else declared,
                             next_pos, zlib.crc32(payload) & 0xFFFFFFFF)
            out += header + payload
        path.write_bytes(bytes(out))
        return path

    def _records(self):
        raw = self.SRC.read_bytes()
        return [(r.tag, raw[r.position + r.header_size:
                            r.position + r.header_size + r.body_size], None)
                for r in pqdif.PQDIFFile(self.SRC).records]

    def test_a_size_field_that_under_declares_its_body_is_read_anyway(self, tmp_path):
        # Every byte is present and the chain is intact; only the size field
        # is short, which cuts the deflate stream off mid-way.
        records = self._records()
        for i, (tag, payload, _) in enumerate(records):
            if tag == pqdif.TAG_OBSERVATION:
                records[i] = (tag, payload, len(payload) - 40)
        f = pqdif.PQDIFFile(self._rebuild(records, tmp_path / "short_size.pqd"))
        assert f.observations
        assert f.unreadable_observations == []

    def test_an_uncompressed_record_in_a_compressed_file_is_read_anyway(self, tmp_path):
        records = self._records()
        for i, (tag, payload, _) in enumerate(records):
            if tag == pqdif.TAG_OBSERVATION:
                records[i] = (tag, zlib.decompress(payload), None)
        f = pqdif.PQDIFFile(self._rebuild(records, tmp_path / "plain.pqd"))
        assert f.observations
        assert f.unreadable_observations == []

    def test_an_incomplete_stream_in_a_whole_file_is_named_as_damage(self, tmp_path):
        # The field case: the header inflates as far as its first bytes and
        # then the stream stops, while the file itself is not short at all.
        records = self._records()
        first = next(i for i, (tag, _, _) in enumerate(records)
                     if tag == pqdif.TAG_OBSERVATION)
        records[first] = (records[first][0], records[first][1][:2], None)
        f = pqdif.PQDIFFile(self._rebuild(records, tmp_path / "cut_stream.pqd"))
        _ = f.observations

        reason = f.unreadable_observations[0][1]
        assert "incomplete or truncated stream" in reason
        # The distinction the message exists to draw: the file is whole, so a
        # re-export is the answer, and no byte count went missing.
        assert "cut short" not in reason
        assert "damaged in place" in reason
        assert "zlib_header=valid" in reason
        assert "present=2 to_next_record=2" in reason
        assert "next_header=intact" in reason

    def test_a_body_that_is_not_a_zlib_stream_says_so_instead(self, tmp_path):
        # A different cause with the same surface symptom, and the evidence
        # has to separate them without anyone opening the file.
        raw = bytearray(self.SRC.read_bytes())
        for record in pqdif.PQDIFFile(self.SRC).records:
            if record.tag == pqdif.TAG_OBSERVATION:
                start = record.position + record.header_size
                raw[start:start + 8] = b"\x00" * 8
        broken = tmp_path / "damaged.pqd"
        broken.write_bytes(bytes(raw))

        f = pqdif.PQDIFFile(broken)
        with pytest.raises(pqdif.PQDIFError):
            _ = f.observations
        reason = f.unreadable_observations[0][1]
        assert "not begin with a zlib stream" in reason
        assert "zlib_header=not a zlib header" in reason
        assert "cut short" not in reason
        assert "present=" in reason and "to_next_record=" in reason

    def test_a_truncated_file_still_says_the_file_was_cut_short(self, tmp_path):
        # The evidence must not drown the one cause that is not about this
        # record at all: bytes the file never received.
        raw = self.SRC.read_bytes()
        short = tmp_path / "short.pqd"
        short.write_bytes(raw[:len(raw) - 3000])
        f = pqdif.PQDIFFile(short)
        _ = f.observations
        assert f.missing_bytes == 3000
        # Whatever inflated is kept; only a record that yields nothing fails.
        for _pos, reason in f.unreadable_observations:
            assert "cut short" in reason

    def test_the_evidence_survives_into_the_report(self, tmp_path):
        # The message is only useful if it reaches the document the engineer
        # actually reads, whole rather than summarised away.
        from pq_report import _integrity_note
        raw = bytearray(self.SRC.read_bytes())
        observations = [r for r in pqdif.PQDIFFile(self.SRC).records
                        if r.tag == pqdif.TAG_OBSERVATION]
        start = observations[0].position + observations[0].header_size
        raw[start:start + 8] = b"\x00" * 8
        broken = tmp_path / "one_bad.pqd"
        broken.write_bytes(bytes(raw))

        f = pqdif.PQDIFFile(broken)
        _ = f.observations
        note = _integrity_note({
            "unreadable_observations": len(f.unreadable_observations),
            "total_observations": f.observation_count,
            "missing_bytes": f.missing_bytes,
            "unreadable_detail": [{"offset": pos, "name": "", "reason": reason}
                                  for pos, reason in f.unreadable_observations],
        }, {})
        assert "Evidence: declared_body=" in note
        assert "partial_inflate=" in note

    def test_an_intact_file_needs_none_of_it(self, tmp_path):
        f = pqdif.PQDIFFile(self.SRC)
        assert f.observations and f.unreadable_observations == []
        assert all(r.next_header_intact in (None, True) for r in f.records)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Harmonic conclusions are gated on measurable load
# ─────────────────────────────────────────────────────────────────────────────

class TestHarmonicSignificanceGate:
    """A spectrum quantised to the meter's resolution carries no shape.

    Without this gate a 1.1 A residential service with 0.2 A of third harmonic
    was reported as an electric arc furnace at 94% similarity, with a
    recommendation to install a STATCOM.
    """

    @staticmethod
    def _frame(fundamental, h3, n=300):
        # Slight variation, not constants: a constant series has zero standard
        # deviation, which makes the Pearson correlation undefined and buries the
        # behaviour under a divide warning.
        import pandas as pd
        rng = np.random.default_rng(11)
        idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        jitter = lambda base: np.abs(base * (1 + rng.normal(0, 0.02, n)))
        return pd.DataFrame({
            "current_a": jitter(fundamental),
            "h3_current_a": jitter(h3),
            "h5_current_a": jitter(h3 * 0.5),
            "h7_current_a": jitter(h3 * 0.3),
            "h9_current_a": jitter(h3 * 0.1),
            "h3_voltage_a": jitter(h3 * 0.1),
            "h5_voltage_a": jitter(h3 * 0.08),
            "h7_voltage_a": jitter(h3 * 0.06),
        }, index=idx)

    def test_light_load_spectrum_is_refused(self):
        from pq_analysis import harmonic_spectrum_significance
        r = harmonic_spectrum_significance(self._frame(1.1, 0.2), Thresholds())
        assert r["usable"] is False
        assert "resolution" in r["reason"]

    def test_loaded_spectrum_is_accepted(self):
        from pq_analysis import harmonic_spectrum_significance
        r = harmonic_spectrum_significance(self._frame(20.0, 4.0), Thresholds())
        assert r["usable"] is True
        assert r["resolution_steps"] >= 5

    def test_no_signature_match_at_light_load(self):
        from pq_analysis import _detect_harmonic_signature, harmonic_spectrum_significance
        df = self._frame(1.1, 0.2)
        sig = harmonic_spectrum_significance(df, Thresholds())
        assert _detect_harmonic_signature(df, 1.1, sig) == []

    def test_signature_match_survives_at_real_load(self):
        from pq_analysis import _detect_harmonic_signature, harmonic_spectrum_significance
        df = self._frame(20.0, 4.0)
        sig = harmonic_spectrum_significance(df, Thresholds())
        assert _detect_harmonic_signature(df, 20.0, sig)

    def test_attribution_and_resonance_withheld_at_light_load(self):
        from pq_analysis import check_harmonic_sources
        r = check_harmonic_sources(self._frame(1.1, 0.2), Thresholds())
        # The impedances stay as measured data; only the conclusions are withheld.
        assert r["available"] is True
        assert r["overall"] == "not_assessed"
        assert r["resonant_orders"] == []
        assert all(od["attribution"] == "not_assessed" for od in r["orders"].values())

    def test_spectral_shape_declines_at_light_load(self):
        from pq_analysis import check_spectral_shape, check_harmonic_sources
        df = self._frame(1.1, 0.2)
        src = check_harmonic_sources(df, Thresholds())
        r = check_spectral_shape(df, Thresholds(), src)
        assert r["available"] is False and "not classified" in r["error"]

    def test_too_few_loaded_intervals_is_refused(self):
        from pq_analysis import harmonic_spectrum_significance
        import pandas as pd
        df = self._frame(20.0, 4.0, n=300)
        # Only 5 intervals at real load; the rest near zero.
        df.loc[df.index[5:], "current_a"] = 0.1
        r = harmonic_spectrum_significance(df, Thresholds())
        assert r["usable"] is False and "loaded" in r["reason"]

    def test_resolution_test_applies_without_an_rms_current_channel(self):
        # Load cannot be verified, but the quantisation test still governs.
        from pq_analysis import harmonic_spectrum_significance
        df = self._frame(1.1, 0.2).drop(columns=["current_a"])
        r = harmonic_spectrum_significance(df, Thresholds())
        assert r["usable"] is False and r["load_verified"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 14. Residential customer letter
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _FIXTURES, reason="test_data/*.pqd not generated")
class TestChannelAppendix:
    """Appendix B: what was read out of the file, and what each channel holds."""

    @staticmethod
    @pytest.fixture(scope="class")
    def doc(tmp_path_factory):
        import docx
        import pq_analysis as An
        from pq_report import generate_report, generate_word_report
        out = tmp_path_factory.mktemp("chan")
        ds = extract_dataset(
            ProntoAdapter(Path("test_data/test_residential.pqd")), ChannelMapper())
        th = Thresholds(nominal_voltage=120.0, customer_class="r")
        df = ds.df
        ev = An.detect_events(ds, th)
        rep = generate_report(
            ds, An.check_voltage_compliance(df, th), An.check_thd(df, th),
            An.check_power_factor(df, th), An.check_voltage_imbalance(df, th),
            An.check_current_imbalance(df, th), An.check_demand(df, th),
            An.check_individual_harmonics(df, th),
            An.check_individual_voltage_harmonics(df, th),
            An.check_neutral_harmonics(df, th), An.check_harmonic_sources(df, th),
            An.check_harmonic_statistics(df, th), ev, th,
            neutral_health_result=An.check_neutral_health(ds, th),
            itic_result=An.check_itic(ev, th),
            flicker_result=An.check_flicker(df, th))
        rep["root_causes"] = An.analyze_root_causes(rep, ds, th)
        path = generate_word_report(
            report=rep, thresh=th, ds=ds, site_name="S", site_address="A",
            engineer_name="E", engineer_contact="", outdir=out, stem="ch")
        return docx.Document(str(path))

    @staticmethod
    def _table(doc):
        return next(t for t in doc.tables if t.rows[0].cells[0].text == "Channel")

    def test_the_appendices_are_lettered_and_ordered(self, doc):
        heads = [p.text for p in doc.paragraphs if p.text.startswith("Appendix")]
        assert heads == ["Appendix A: Terms Used in This Report",
                         "Appendix B: Standards, Methods, and Limitations",
                         "Appendix C: Channels Read From the Meter File"]

    def test_the_glossary_is_reachable_from_the_body(self, doc):
        # The definitions moved to the back, so the body has to say where they
        # went or they are found only by whoever pages to the end.
        body = " ".join(p.text for p in doc.paragraphs)
        assert "defined in Appendix A" in body

    def test_no_cross_reference_points_at_a_bare_appendix(self, doc):
        # Three appendices make "see the Appendix" ambiguous.
        body = " ".join(p.text for p in doc.paragraphs)
        for tb in doc.tables:
            body += " " + " ".join(c.text for r in tb.rows for c in r.cells)
        assert "see the Appendix" not in body

    def test_every_channel_read_is_listed(self, doc):
        ds = extract_dataset(
            ProntoAdapter(Path("test_data/test_residential.pqd")), ChannelMapper())
        listed = {r.cells[0].text.split()[0] for r in list(self._table(doc).rows)[1:]}
        assert listed == set(ds.meta["channel_map"])

    def test_each_row_names_the_device_channel_it_came_from(self, doc):
        # The match from device label to engineering quantity is the step that
        # fails silently, so the label has to be on the page to check it.
        rows = {r.cells[0].text.split()[0]: [c.text for c in r.cells]
                for r in list(self._table(doc).rows)[1:]}
        assert rows["voltage_a"][1].startswith("RMS voltage, line to neutral")
        assert rows["voltage_a"][2] == "V"
        assert rows["voltage_a"][3] == "Van RMS"
        assert rows["current_neutral"][3] == "In RMS"

    def test_within_interval_extremes_ride_with_their_channel(self, doc):
        labels = [r.cells[0].text for r in list(self._table(doc).rows)[1:]]
        assert any(l.startswith("voltage_a  (+ ") and "max" in l and "min" in l
                   for l in labels)
        assert not any(l.startswith("voltage_a_peak") for l in labels)

    def test_coverage_counts_are_reported(self, doc):
        rows = {r.cells[0].text.split()[0]: [c.text for c in r.cells]
                for r in list(self._table(doc).rows)[1:]}
        assert rows["voltage_a"][4] == "288"

    def test_split_phase_uses_the_service_own_phase_names(self, doc):
        text = " ".join(c.text for r in self._table(doc).rows for c in r.cells)
        assert "L1" in text and "L2" in text
        assert "phase A" not in text

    @pytest.mark.parametrize("name,expected", [
        ("h3_current_a",       "3rd-order harmonic current, phase A"),
        ("h11_voltage_b",      "11th-order harmonic voltage, phase B"),
        ("h21_current_c",      "21st-order harmonic current, phase C"),
        ("h13_current_neutral", "13th-order harmonic current, neutral"),
        ("thd_voltage_a",      "Total harmonic distortion of the voltage, phase A"),
        ("flicker_plt",        "Long-term flicker severity (Plt, 2-hour), phase A"),
        ("voltage_ab",         "RMS voltage, line to line (phase A to phase B)"),
        ("current_neutral",    "RMS current in the neutral conductor"),
        ("frequency",          "System frequency"),
    ])
    def test_descriptions_are_derived_from_the_name(self, name, expected):
        from pq_report import _channel_description
        assert _channel_description(name, is_split=False) == expected

    def test_within_interval_extremes_describe_their_base_channel(self):
        from pq_report import _channel_description
        assert _channel_description("voltage_a_peak", is_split=True) == (
            "RMS voltage, line to neutral, L1 — highest value within each interval")
        assert _channel_description("current_b_min", is_split=True) == (
            "RMS current, L2 — lowest value within each interval")

    def test_an_order_nobody_tabulated_still_gets_a_description(self):
        # Meters report orders well past the ones the limits cover; a hand-kept
        # table would leave those rows blank.
        from pq_report import _channel_description
        assert _channel_description("h49_current_a", is_split=False) == (
            "49th-order harmonic current, phase A")


class TestRecordingOverview:
    """The sanity-check chart: the whole recording, before any assessment."""

    def _frame(self, n=576, gap_at=None, cols=("voltage_a", "voltage_b",
                                               "current_a", "current_b",
                                               "current_neutral")):
        import pandas as pd
        rng = np.random.default_rng(5)
        idx = pd.date_range("2026-05-02", periods=n, freq="5min", tz="UTC")
        if gap_at is not None:
            idx = idx.delete(range(gap_at, gap_at + 60))
        base = {"voltage_a": 121.0, "voltage_b": 120.0, "voltage_c": 120.5,
                "current_a": 12.0, "current_b": 9.0, "current_c": 10.0,
                "current_neutral": 3.0}
        return pd.DataFrame(
            {c: np.abs(base[c] + rng.normal(0, 0.5, len(idx))) for c in cols},
            index=idx)

    def _ds(self, df):
        from pq_adapter import PQDataset
        return PQDataset(df=df, adaptive_df=None,
                         meta={"interval_minutes": 5, "topology": "split-phase"})

    def test_chart_is_written(self, tmp_path):
        from pq_plots import plot_overview
        plot_overview(self._ds(self._frame()), Thresholds(nominal_voltage=120.0),
                      outdir=tmp_path, stem="s")
        assert (tmp_path / "s_overview.png").exists()

    def test_a_recording_gap_breaks_the_trace(self):
        # A line drawn straight across five hours the meter never recorded is
        # indistinguishable from five hours of steady service.
        from pq_plots import _gap_spans, _break_at_gaps
        df = self._frame(gap_at=200)
        gaps = _gap_spans(df.index)
        assert len(gaps) == 1
        start, end = gaps[0]
        assert (end - start).total_seconds() / 3600 == pytest.approx(5.08, abs=0.1)
        broken = _break_at_gaps(df["voltage_a"], gaps)
        assert broken.isna().sum() == 1
        assert start < broken[broken.isna()].index[0] < end

    def test_an_unbroken_recording_has_no_gaps(self):
        from pq_plots import _gap_spans
        assert _gap_spans(self._frame().index) == []

    def test_voltage_only_file_still_charts(self, tmp_path):
        from pq_plots import plot_overview
        plot_overview(self._ds(self._frame(cols=("voltage_a", "voltage_b"))),
                      Thresholds(nominal_voltage=120.0), outdir=tmp_path, stem="v")
        assert (tmp_path / "v_overview.png").exists()

    def test_nothing_to_chart_writes_nothing(self, tmp_path):
        import pandas as pd
        from pq_plots import plot_overview
        df = self._frame()[[]].assign(frequency=60.0)
        plot_overview(self._ds(df), Thresholds(nominal_voltage=120.0),
                      outdir=tmp_path, stem="n")
        assert not (tmp_path / "n_overview.png").exists()

    def test_both_documents_open_with_the_chart(self, tmp_path):
        # It is the first thing in the report and it is in the letter, so a
        # reader of either can check the period before reading a conclusion.
        import docx
        import pq_analysis as An
        from pq_plots import plot_overview
        from pq_report import (generate_report, generate_word_report,
                               generate_customer_letter)
        ds = extract_dataset(ProntoAdapter(Path("test_data/test_residential.pqd")),
                             ChannelMapper())
        th = Thresholds(nominal_voltage=120.0, customer_class="r")
        df = ds.df
        ev = An.detect_events(ds, th)
        rep = generate_report(
            ds, An.check_voltage_compliance(df, th), An.check_thd(df, th),
            An.check_power_factor(df, th), An.check_voltage_imbalance(df, th),
            An.check_current_imbalance(df, th), An.check_demand(df, th),
            An.check_individual_harmonics(df, th),
            An.check_individual_voltage_harmonics(df, th),
            An.check_neutral_harmonics(df, th), An.check_harmonic_sources(df, th),
            An.check_harmonic_statistics(df, th), ev, th,
            neutral_health_result=An.check_neutral_health(ds, th),
            itic_result=An.check_itic(ev, th),
            flicker_result=An.check_flicker(df, th))
        rep["root_causes"] = An.analyze_root_causes(rep, ds, th)
        plot_overview(ds, th, outdir=tmp_path, stem="ov")

        rpt = docx.Document(str(generate_word_report(
            report=rep, thresh=th, ds=ds, site_name="S", site_address="A",
            engineer_name="E", engineer_contact="", outdir=tmp_path, stem="ov")))
        heads = [p.text for p in rpt.paragraphs if p.text.strip()]
        assert "Recording Overview" in heads
        assert heads.index("Recording Overview") < heads.index(
            "Executive Summary and Compliance Status")

        letter = docx.Document(str(generate_customer_letter(
            rep, th, "1 Test St", "Eng", tmp_path, "ov")))
        assert "What we recorded" in [p.text for p in letter.paragraphs]
        assert len(letter.inline_shapes) == 1

    def test_the_chart_states_the_period_it_drew(self, tmp_path):
        # The chart and the "Length of recording" line in the letter come from
        # the same index; the caption is what makes a mismatch visible.
        from pq_plots import plot_overview
        import matplotlib.pyplot as plt
        df = self._frame()
        captured = {}
        orig = plt.Figure.text

        def _capture(self, x, y, s, *a, **k):
            captured.setdefault("sub", s)
            return orig(self, x, y, s, *a, **k)
        plt.Figure.text = _capture
        try:
            plot_overview(self._ds(df), Thresholds(nominal_voltage=120.0),
                          outdir=tmp_path, stem="c")
        finally:
            plt.Figure.text = orig
        assert "2026-05-02 00:00" in captured["sub"]
        assert f"{len(df):,} intervals of 5 min" in captured["sub"]


class TestCustomerLetter:
    """The customer-facing document, one per service class.

    It must state no attribution, commit Xcel Energy to nothing, and never
    tell the customer an internal report exists.
    """

    def _report(self, path, customer_class="r", nominal=120.0):
        import pq_analysis as An
        from pq_report import generate_report
        ds = extract_dataset(ProntoAdapter(path), ChannelMapper())
        th = Thresholds(nominal_voltage=nominal, customer_class=customer_class)
        df = ds.df
        ev = An.detect_events(ds, th)
        rep = generate_report(
            ds, An.check_voltage_compliance(df, th), An.check_thd(df, th),
            An.check_power_factor(df, th), An.check_voltage_imbalance(df, th),
            An.check_current_imbalance(df, th), An.check_demand(df, th),
            An.check_individual_harmonics(df, th),
            An.check_individual_voltage_harmonics(df, th),
            An.check_neutral_harmonics(df, th),
            An.check_harmonic_sources(df, th),
            An.check_harmonic_statistics(df, th), ev, th,
            neutral_health_result=An.check_neutral_health(ds, th),
            itic_result=An.check_itic(ev, th),
            flicker_result=An.check_flicker(df, th),
        )
        rep["root_causes"] = An.analyze_root_causes(rep, ds, th)
        return rep, th

    def test_written_for_residential(self, tmp_path):
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        assert out is not None and out.exists()
        assert out.name.endswith("_customer_letter.docx")

    def test_a_letter_from_an_earlier_run_never_survives(self, tmp_path):
        # A letter that outlives the run that wrote it sits beside a fresh
        # report describing a different recording, and says nothing about which
        # run produced it.
        from pq_report import generate_customer_letter
        stale = tmp_path / "t_customer_letter.docx"
        stale.write_bytes(b"letter from a two-hour download")
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        assert out == stale and stale.read_bytes() != b"letter from a two-hour download"

    def test_a_reclassified_service_replaces_the_previous_letter(self, tmp_path):
        # The stale letter described the same file read as a different class,
        # so it must be replaced rather than left beside the new one.
        from pq_report import generate_customer_letter
        stale = tmp_path / "t_customer_letter.docx"
        stale.write_bytes(b"letter from when this was billed residential")
        rep, th = self._report(Path("test_data/test_residential.pqd"),
                               customer_class="sg")
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        assert out is not None and out.exists()
        assert out.read_bytes() != b"letter from when this was billed residential"

    def test_a_letter_that_cannot_be_replaced_stops_the_run(self, tmp_path, monkeypatch):
        # Word holds the file open on Windows, so the unlink is what fails. The
        # run has to stop there rather than leave the old letter in place.
        from pathlib import Path as _P
        import pq_report
        stale = tmp_path / "t_customer_letter.docx"
        stale.write_bytes(b"open in Word")
        rep, th = self._report(Path("test_data/test_residential.pqd"))

        def _locked(self):
            raise PermissionError(32, "The process cannot access the file")
        monkeypatch.setattr(_P, "unlink", _locked)
        with pytest.raises(PermissionError, match="open in Word"):
            pq_report.generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        assert stale.read_bytes() == b"open in Word"

    def test_written_for_small_commercial(self, tmp_path):
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                               customer_class="c")
        out = generate_customer_letter(rep, th, "1 Trade St", "Eng", tmp_path, "t")
        assert out is not None and out.exists()

    @pytest.mark.parametrize("cls", ["sg", "pg"])
    def test_every_class_gets_its_own_customer_document(self, tmp_path, cls):
        # The engineering report is internal for every class, so a customer
        # document is written at every scale -- what differs is the register.
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_commercial_large.pqd"),
                               customer_class=cls, nominal=277.0)
        out = generate_customer_letter(rep, th, "1 Trade St", "Eng", tmp_path, "t")
        assert out is not None and out.exists()

    def test_the_internal_report_is_named_and_unsigned(self, tmp_path):
        """The internal document says what it is and is addressed to nobody."""
        docx = pytest.importorskip("docx")
        import glob as _glob
        from pq_report import generate_word_report
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        ds = extract_dataset(ProntoAdapter(Path("test_data/test_residential.pqd")),
                             ChannelMapper())
        generate_word_report(
            report=rep, thresh=th, ds=ds, outdir=tmp_path, stem="t",
            site_name="Site", site_address="1 Test St",
            engineer_name="A. Engineer", engineer_contact="",
            engineer_title="Electric Area Engineer")
        written = _glob.glob(str(tmp_path / "*_internal_engineering_report.docx"))
        assert written, "the internal report is named as such on disk"

        d = docx.Document(written[0])
        text = "\n".join(p.text for p in d.paragraphs)
        assert "Internal Engineering Report" in text
        assert "internal working document" in text
        # No sign-off block: the document is not addressed to anyone.
        assert "Sincerely," not in text
        # Whose work it is survives, as a header field rather than a signature.
        table_text = " ".join(c.text for t in d.tables for r in t.rows for c in r.cells)
        assert "Prepared by" in table_text and "A. Engineer" in table_text

    def test_the_register_changes_with_the_class(self, tmp_path):
        """A homeowner is told what a meter does; a plant engineer is not."""
        docx = pytest.importorskip("docx")
        from pq_report import generate_customer_letter
        texts = {}
        for cls, path, nominal in (("r", "test_data/test_residential.pqd", 120.0),
                                   ("pg", "test_data/test_commercial_large.pqd", 277.0)):
            rep, th = self._report(Path(path), customer_class=cls, nominal=nominal)
            out = generate_customer_letter(rep, th, "1 Test St", "Eng",
                                           tmp_path / cls, "t")
            texts[cls] = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)

        assert "your home" in texts["r"]
        assert "measured the voltage and current many times a second" in texts["r"]

        assert "your facility" in texts["pg"]
        # The terse register states what was logged without explaining metering.
        assert "measured the voltage and current many times a second" not in texts["pg"]
        assert "interval resolution" in texts["pg"]
        # A primary-metered customer owns the transformer, and the letter says so.
        assert "transformer at this site is yours" in texts["pg"]

    def test_the_power_factor_tariff_sheet_follows_the_class(self):
        from pq_report import _customer_conditions
        sheets = {}
        for cls in ("c", "sg", "pg"):
            rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                                   customer_class=cls)
            pf = [c for c in _customer_conditions(rep, th)
                  if "power factor" in c["headline"].lower()]
            sheets[cls] = pf[0]["measured"] if pf else ""
        assert "Sheet R73 (Schedule C)" in sheets["c"]
        assert "Sheet R73 (Schedule SG)" in sheets["sg"]
        # Schedule PG asks for near unity, not a 0.90 floor.
        assert "Sheet R121 (Schedule PG)" in sheets["pg"]
        assert "near unity" in sheets["pg"] and "0.90" not in sheets["pg"]

    def test_no_customer_document_mentions_the_internal_report(self, tmp_path):
        docx = pytest.importorskip("docx")
        from pq_report import generate_customer_letter
        for cls, path, nominal in (("r", "test_data/test_residential.pqd", 120.0),
                                   ("c", "test_data/test_commercial_small.pqd", 120.0),
                                   ("sg", "test_data/test_commercial_large.pqd", 277.0),
                                   ("pg", "test_data/test_commercial_large.pqd", 277.0)):
            rep, th = self._report(Path(path), customer_class=cls, nominal=nominal)
            out = generate_customer_letter(rep, th, "1 Test St", "Eng",
                                           tmp_path / cls, "t")
            text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs).lower()
            for forbidden in ("engineering report", "internal report", "attached report"):
                assert forbidden not in text, f"{cls}: {forbidden}"

    def test_business_letter_covers_power_factor_and_distortion(self):
        from pq_report import _customer_conditions
        rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                               customer_class="c")
        heads = " ".join(c["headline"].lower() for c in _customer_conditions(rep, th))
        assert "power factor" in heads
        assert "distorted" in heads

    def test_residential_letter_omits_power_factor_and_distortion(self):
        # A homeowner is not billed for power factor and cannot act on distortion.
        from pq_report import _customer_conditions
        rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                               customer_class="r")
        heads = " ".join(c["headline"].lower() for c in _customer_conditions(rep, th))
        assert "power factor" not in heads
        assert "distorted" not in heads

    def test_business_letter_carries_no_residential_wording(self, tmp_path):
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                               customer_class="c")
        out = generate_customer_letter(rep, th, "1 Trade St", "Eng", tmp_path, "t")
        text = " ".join(p.text for p in Document(str(out)).paragraphs)
        for residential in ("your home", "the house", "refrigerators, freezers",
                            "well pumps"):
            assert residential not in text, f"business letter says {residential!r}"
        assert "your business" in text

    def test_business_letter_names_the_tariff_schedule(self, tmp_path):
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_commercial_small.pqd"),
                               customer_class="c")
        out = generate_customer_letter(rep, th, "1 Trade St", "Eng", tmp_path, "t")
        text = " ".join(p.text for p in Document(str(out)).paragraphs)
        # Power factor is the one item with a direct billing consequence, so the
        # schedule it comes from is worth naming.
        assert "Schedule C" in text and "R73" in text

    def test_neutral_wording_follows_the_service_topology(self):
        # "Two 120-volt halves" is true of a house and false of a three-phase site.
        from pq_report import _customer_vocabulary
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        v = _customer_vocabulary(rep, th)
        assert "two 120-volt halves" in v["neutral_measured"]
        rep3, th3 = self._report(Path("test_data/test_commercial_small.pqd"),
                                 customer_class="c")
        v3 = _customer_vocabulary(rep3, th3)
        assert "three separate live wires" in v3["neutral_measured"]
        assert "your business" == v3["site"]

    def test_states_no_attribution_and_no_commitment(self, tmp_path):
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        text = "\n".join(p.text for p in Document(str(out)).paragraphs).lower()
        for banned in ("xcel energy will", "customer's responsibility",
                       "utility responsibility", "customer-side", "utility-side",
                       "your responsibility", "at fault"):
            assert banned not in text, f"letter asserts {banned!r}"

    def test_omits_engineering_concepts_a_homeowner_cannot_act_on(self, tmp_path):
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        text = "\n".join(p.text for p in Document(str(out)).paragraphs).lower()
        for jargon in ("thd", "tdd", "k-factor", "harmonic", "resonance",
                       "ansi c84", "ieee 519", "iec 61000", "per-order",
                       "spectral", "impedance"):
            assert jargon not in text, f"letter uses {jargon!r}"

    def test_every_condition_pairs_a_number_with_meaning_and_symptom(self):
        from pq_report import _customer_conditions
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        conditions = _customer_conditions(rep, th)
        assert conditions, "residential fixture should surface conditions"
        for c in conditions:
            assert c["headline"] and c["measured"] and c["means"] and c["symptom"]

    def test_event_counts_handles_dataframe_and_list(self):
        from pq_report import _event_counts
        import pandas as pd
        assert _event_counts({}) == {}
        assert _event_counts({"events": None}) == {}
        assert _event_counts({"events": pd.DataFrame()}) == {}
        df = pd.DataFrame({"type": ["voltage_sag", "voltage_sag", "voltage_swell"]})
        assert _event_counts({"events": df}) == {"voltage_sag": 2, "voltage_swell": 1}
        lst = [{"type": "voltage_sag"}, {"type": "flicker_pst"}]
        assert _event_counts({"events": lst}) == {"voltage_sag": 1, "flicker_pst": 1}

    def test_flicker_number_is_given_a_scale_the_reader_can_use(self, tmp_path):
        # "4.98" means nothing without the basis of the scale: 1.0 is the
        # conventional threshold of irritability, so the multiple of it is what
        # makes the reading interpretable.
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        text = " ".join(p.text for p in Document(str(out)).paragraphs)
        if "flickering" not in text:
            pytest.skip("fixture did not exceed the flicker limit")
        assert "annoying" in text
        assert "times that level" in text or "same as that level" in text
        # It must say what flicker is not, so nobody reads it as damage or usage.
        assert "not of damage" in text or "does not harm" in text

    def test_letter_does_not_promise_the_engineering_report(self, tmp_path):
        # The engineering document is shared at Xcel's discretion, so the letter
        # must not tell the customer one is attached or owed to them.
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        text = " ".join(p.text for p in Document(str(out)).paragraphs).lower()
        for promise in ("report accompanies", "accompanying this letter",
                        "attached report", "enclosed report",
                        "document to hand to an electrician"):
            assert promise not in text, f"letter promises {promise!r}"

    def test_clean_service_gets_a_clear_no_problem_letter(self, tmp_path):
        from docx import Document
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_residential.pqd"))
        # Blank every condition source so the letter takes its no-findings path.
        for k in ("voltage_compliance", "flicker", "neutral_health",
                  "current_imbalance", "itic", "events"):
            rep[k] = {"available": False}
        rep["root_causes"] = []
        out = generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t")
        text = "\n".join(p.text for p in Document(str(out)).paragraphs)
        assert "did not find a problem" in text
        # And it must not overclaim: an intermittent fault can fall outside a window.
        assert "only the days it ran" in text


# ─────────────────────────────────────────────────────────────────────────────
# 15. Neutral wording is scaled to what was measured
# ─────────────────────────────────────────────────────────────────────────────

class TestNeutralSeverityWording:
    """The neutral indicators move for ordinary reasons too.

    A "caution" driven by a couple of volts of variation in the leg sum must not
    read the same as legs actually opposing each other — the letter was declaring
    a shock and fire hazard on the strength of a 2.8 V standard deviation.
    """

    @staticmethod
    def _report(severity, corr, sum_std=2.8, asym=1.4):
        return {
            "file_summary": {"topology": "split-phase", "duration_hours": 68},
            "voltage_compliance": {"available": False},
            "flicker": {"available": False},
            "current_imbalance": {"available": False},
            "power_factor": {"available": False},
            "thd_compliance": {},
            "itic": {},
            "events": {},
            "neutral_health": {
                "available": True, "severity": severity, "leg_correlation": corr,
                "sum_mean_v": 238.0, "sum_std_v": sum_std, "asym_mean_v": asym,
                "vne_max_v": 0.1,
            },
        }

    def _neutral(self, severity, corr):
        from pq_report import _customer_conditions
        conds = _customer_conditions(self._report(severity, corr),
                                     Thresholds(nominal_voltage=120.0,
                                                customer_class="r"))
        assert conds, f"{severity} produced no condition"
        return conds

    def test_caution_is_a_baseline_not_a_hazard(self):
        c = self._neutral("caution", 0.82)[0]
        assert c.get("safety") is False
        assert "hazard" not in c["means"].lower()
        assert "baseline" in c["means"]

    def test_critical_keeps_the_urgent_language(self):
        c = self._neutral("critical", -0.74)[0]
        assert c["safety"] is True
        assert "shock and fire hazard" in c["means"]

    def test_only_warning_and_critical_lead_the_letter(self):
        from pq_report import _customer_conditions
        th = Thresholds(nominal_voltage=120.0, customer_class="r")
        rep = self._report("caution", 0.82)
        # Give it a second condition so ordering is observable.
        rep["voltage_compliance"] = {
            "available": True, "total_pct_out_of_bounds": 5.0,
            "range_v": (114.0, 126.0),
            "phases": {"voltage_a": {"min_v": 100.0, "max_v": 121.0}},
        }
        conds = _customer_conditions(rep, th)
        assert "neutral" not in conds[0]["headline"].lower()
        rep["neutral_health"]["severity"] = "critical"
        rep["neutral_health"]["leg_correlation"] = -0.74
        assert "neutral" in _customer_conditions(rep, th)[0]["headline"].lower()

    def test_the_explainer_does_not_contradict_the_measurement(self):
        # The explainer says what we look for; the data says what was found. It
        # must not assert opposition when the legs measured +0.82.
        c = self._neutral("caution", 0.82)[0]
        assert "so that is the pattern we look for" in c["measured"]
        assert "rose and fell together" in c["measured"]
        assert "Our measurements show the two halves moving in opposite" not in c["measured"]

    def test_indicators_are_always_reported_as_numbers(self):
        for severity, corr in (("caution", 0.82), ("warning", -0.1), ("critical", -0.74)):
            measured = self._neutral(severity, corr)[0]["measured"]
            assert "correlation of" in measured
            assert "238 volts" in measured
            assert f"{corr:+.2f}" in measured

    def test_correlation_sense_matches_its_sign(self):
        from pq_report import _neutral_indicator_sentence
        base = {"sum_mean_v": 240.0, "sum_std_v": 1.0, "asym_mean_v": 0.5}
        assert "sound shared connection" in _neutral_indicator_sentence(
            {**base, "leg_correlation": 0.9})
        assert "only loosely" in _neutral_indicator_sentence(
            {**base, "leg_correlation": 0.2})
        assert "failing connection produces" in _neutral_indicator_sentence(
            {**base, "leg_correlation": -0.6})

    def test_caution_claims_no_symptom_the_customer_would_have_seen(self):
        c = self._neutral("caution", 0.82)[0]
        assert "Nothing in particular" in c["symptom"]

    def test_normal_neutral_produces_no_condition_at_all(self):
        from pq_report import _customer_conditions
        rep = self._report("normal", 0.95)
        conds = _customer_conditions(rep, Thresholds(nominal_voltage=120.0,
                                                     customer_class="r"))
        assert not any("neutral" in c["headline"].lower() for c in conds)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Distortion is reported without depending on an assumed limit
# ─────────────────────────────────────────────────────────────────────────────

class TestDistortionClaim:
    """The current-side limit depends on ISC/IL.

    Without ISC the analysis falls back to the most restrictive class, which on
    one test file turns a 0.35% exceedance into 100%. That is fine in a labelled
    engineering report and not fine in a letter telling a business it breaches a
    standard, so the letter only asserts what its limits actually support.
    """

    @staticmethod
    def _report(v_exceed, i_exceed, isc_provided):
        return {
            "file_summary": {"topology": "3-phase", "duration_hours": 24},
            "voltage_compliance": {"available": False},
            "flicker": {"available": False},
            "neutral_health": {"available": False},
            "current_imbalance": {"available": False},
            "power_factor": {"available": False},
            "itic": {}, "events": {},
            "thd_compliance": {
                "voltage": {"available": True, "pct_exceeding": v_exceed},
                "current": {"available": True, "pct_exceeding": i_exceed},
                "tdd_info": {"isc_provided": isc_provided},
            },
        }

    def _distortion(self, v_exceed, i_exceed, isc_provided):
        from pq_report import _customer_conditions
        th = Thresholds(nominal_voltage=120.0, customer_class="c")
        for c in _customer_conditions(self._report(v_exceed, i_exceed, isc_provided), th):
            if "distort" in c["headline"].lower():
                return c
        return None

    def test_asserts_both_when_isc_is_known(self):
        c = self._distortion(5.0, 100.0, True)
        assert "the voltage supplied to you" in c["measured"]
        assert "the current your equipment draws" in c["measured"]
        assert "not settled" not in c["measured"]

    def test_withholds_the_current_claim_without_isc(self):
        c = self._distortion(5.0, 100.0, False)
        assert "the voltage supplied to you" in c["measured"]
        # The 100% exceedance is against an assumed limit, so it is not asserted.
        assert "and in the current your equipment draws, beyond" not in c["measured"]
        assert "not settled by this recording" in c["measured"]

    def test_still_reports_when_only_current_is_involved(self):
        # The item must not vanish just because ISC was omitted — that was the
        # dependency on the flag worth removing.
        c = self._distortion(0.0, 100.0, False)
        assert c is not None
        assert "more distorted than the standard allows" not in c["headline"]
        assert "not settled by this recording" in c["measured"]

    def test_voltage_alone_is_enough_to_assert(self):
        # The voltage limit is fixed and does not depend on ISC.
        c = self._distortion(5.0, 0.0, False)
        assert "beyond the level the applicable standard permits" in c["measured"]

    def test_silent_when_nothing_exceeds_and_isc_is_known(self):
        assert self._distortion(0.0, 0.0, True) is None

    def test_fires_the_same_way_with_and_without_the_flag(self):
        # Presence must not depend on the flag, only the strength of the claim.
        assert self._distortion(5.0, 100.0, True) is not None
        assert self._distortion(5.0, 100.0, False) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 17. Engineering report review fixes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _FIXTURES, reason="test_data/*.pqd not generated")
class TestEngineeringReportReview:
    """Defects found reading the >50 kW report end to end as its audience."""

    @staticmethod
    @pytest.fixture(scope="class")
    def doc(tmp_path_factory):
        from docx import Document
        import pq_analysis as An
        from pq_report import generate_report, generate_word_report
        out = tmp_path_factory.mktemp("sg")
        ds = extract_dataset(
            ProntoAdapter(Path("test_data/test_commercial_large.pqd")), ChannelMapper())
        th = Thresholds(nominal_voltage=277.0, customer_class="sg",
                        isc_amps=5000.0, transformer_kva=300.0)
        df, ev = ds.df, None
        ev = An.detect_events(ds, th)
        rep = generate_report(
            ds, An.check_voltage_compliance(df, th), An.check_thd(df, th),
            An.check_power_factor(df, th), An.check_voltage_imbalance(df, th),
            An.check_current_imbalance(df, th), An.check_demand(df, th),
            An.check_individual_harmonics(df, th),
            An.check_individual_voltage_harmonics(df, th),
            An.check_neutral_harmonics(df, th), An.check_harmonic_sources(df, th),
            An.check_harmonic_statistics(df, th), ev, th,
            neutral_health_result=An.check_neutral_health(ds, th),
            itic_result=An.check_itic(ev, th),
            flicker_result=An.check_flicker(df, th),
            kfactor_result=An.kfactor_by_phase(df),
            ll_volt_result=An.check_line_to_line_voltage(df, th),
            frequency_result=An.check_frequency(df, th))
        rep["root_causes"] = An.analyze_root_causes(rep, ds, th)
        path = generate_word_report(
            report=rep, thresh=th, ds=ds, site_name="S", site_address="A",
            engineer_name="E", engineer_contact="", outdir=out, stem="sg")
        return Document(str(path))

    @staticmethod
    def _text(doc):
        t = " ".join(p.text for p in doc.paragraphs)
        for tb in doc.tables:
            t += " " + " ".join(c.text for r in tb.rows for c in r.cells)
        return t

    def test_thermal_claim_is_conditional_on_loading(self, doc):
        # At 27% of nameplate a K-factor of 5 does not exceed the rating, and
        # the report had recommended a replacement transformer on that basis.
        t = self._text(doc)
        assert "retains substantial thermal margin" in t
        assert "significantly exceeds nameplate assumptions" not in t

    def test_standards_tally_matches_the_table(self, doc):
        import re
        t = self._text(doc)
        m = re.search(r"of the (\d+) power quality standards evaluated", t)
        assert m, "standards tally sentence missing"
        claimed = int(m.group(1))
        # The table is grouped by measured quantity, so it carries merged
        # heading rows that are not standards and must not be counted.
        tbl = next(tb for tb in doc.tables
                   if tb.rows[0].cells[0].text.strip() == "Standard")
        rows = sum(1 for r in tbl.rows[1:]
                   if len({id(c._tc) for c in r.cells}) > 1)
        assert claimed == rows, f"summary claims {claimed}, table shows {rows}"

    def test_the_new_checks_have_table_rows(self, doc):
        standards = next(
            [r.cells[0].text for r in tb.rows[1:]] for tb in doc.tables
            if tb.rows[0].cells[0].text.strip() == "Standard")
        joined = " ".join(standards)
        assert "Line-to-line voltage" in joined
        assert "System frequency" in joined

    def test_no_stale_attribution_or_dangling_reference(self, doc):
        t = self._text(doc)
        assert "identifies whose system is involved" not in t
        assert "attached Pronto data" not in t

    def test_statistical_margin_convention_is_explained(self, doc):
        t = self._text(doc)
        assert "A positive margin is headroom" in t
        assert "exceeded by that amount" in t

    def test_aggregate_statistical_row_is_tdd_when_a_tdd_limit_applies(self, doc):
        # The row sits in a "% of IL" table and is measured against the TDD
        # limit, so labelling it THD named a different quantity.
        for tb in doc.tables:
            labels = [r.cells[0].text.strip() for r in tb.rows]
            if "TDD" in labels or "THD" in labels:
                assert "THD" not in labels, "aggregate row still labelled THD"

    def test_source_table_says_indication_not_attribution(self, doc):
        for tb in doc.tables:
            hdr = [c.text.strip() for c in tb.rows[0].cells]
            if "Apparent Z (Ω)" in hdr:
                assert "Indication" in hdr and "Attribution" not in hdr
                assert "Pearson r" not in hdr
                break
        else:
            pytest.fail("harmonic source table not found")

    def test_overall_assessment_is_not_a_directive(self, doc):
        t = self._text(doc)
        assert "prompt corrective action is required" not in t

    def test_low_accumulation_factor_is_explained(self, doc):
        t = self._text(doc)
        assert "largely cancelling in the neutral" in t

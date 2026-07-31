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

import pqdif
from pq_constants import (
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
        return Thresholds(nominal_voltage=120.0)

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

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

    def test_both_appendices_are_lettered_and_ordered(self, doc):
        heads = [p.text for p in doc.paragraphs if p.text.startswith("Appendix")]
        assert heads == ["Appendix A: Standards, Methods, and Limitations",
                         "Appendix B: Channels Read From the Meter File"]

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
    """The second, customer-facing document.

    It must be residential-only, must state no attribution, and must commit
    Xcel Energy to nothing.
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

    def test_a_class_that_gets_no_letter_clears_the_previous_one(self, tmp_path):
        from pq_report import generate_customer_letter
        stale = tmp_path / "t_customer_letter.docx"
        stale.write_bytes(b"letter from when this was billed residential")
        rep, th = self._report(Path("test_data/test_residential.pqd"),
                               customer_class="sg")
        assert generate_customer_letter(rep, th, "1 Test St", "Eng", tmp_path, "t") is None
        assert not stale.exists()

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
    def test_not_written_above_50kw(self, tmp_path, cls):
        # At that scale the engineering report is the customer document.
        from pq_report import generate_customer_letter
        rep, th = self._report(Path("test_data/test_commercial_large.pqd"),
                               customer_class=cls, nominal=277.0)
        assert generate_customer_letter(rep, th, "1 Trade St", "Eng", tmp_path, "t") is None

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
        rows = next(len(tb.rows) - 1 for tb in doc.tables
                    if tb.rows[0].cells[0].text.strip() == "Standard")
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

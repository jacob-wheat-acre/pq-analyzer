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

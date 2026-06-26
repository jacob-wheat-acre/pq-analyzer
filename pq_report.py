from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pq_constants import (
    Thresholds,
    _H519_ORDERS,
    _SERVICE_TYPE_LABEL,
    _lookup_isc,
    _tdd_class,
    _tdd_limit,
)
from pq_adapter import PQDataset

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Word report dependencies (optional)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from docx import Document as _DocxDocument
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

# Xcel Energy brand colours
_XE_BLUE   = RGBColor(0x00, 0x4B, 0x87) if _DOCX_AVAILABLE else None
_XE_LBLUE  = RGBColor(0x00, 0x9D, 0xD9) if _DOCX_AVAILABLE else None
_XE_ORANGE = RGBColor(0xE8, 0x77, 0x22) if _DOCX_AVAILABLE else None
_PASS_CLR  = RGBColor(0x1F, 0x7A, 0x1F) if _DOCX_AVAILABLE else None
_FAIL_CLR  = RGBColor(0xCC, 0x00, 0x00) if _DOCX_AVAILABLE else None
_GRAY_CLR  = RGBColor(0xF2, 0xF2, 0xF2) if _DOCX_AVAILABLE else None


def _cell_shade(cell, hex_color: str) -> None:
    """Apply background shading to a table cell."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def _set_col_widths(table, widths_cm):
    for row in table.rows:
        for cell, w in zip(row.cells, widths_cm):
            cell.width = Cm(w)


def _bold(para, text: str, color=None, size_pt: int = 11):
    run = para.add_run(text)
    run.bold = True
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = color
    return run


def _normal(para, text: str, color=None, size_pt: int = 11):
    run = para.add_run(text)
    run.font.size = Pt(size_pt)
    if color:
        run.font.color.rgb = color
    return run


def _pf_sym(passes) -> str:
    if passes is True:  return "PASS"
    if passes is False: return "FAIL"
    return "N/A"


def _pf_color(passes):
    if passes is True:  return _PASS_CLR
    if passes is False: return _FAIL_CLR
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORT GENERATION & EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    ds: PQDataset,
    volt_result: dict,
    thd_result: dict,
    pf_result: dict,
    volt_imb_result: dict,
    curr_imb_result: dict,
    demand_result: dict,
    harm_result: dict,
    volt_harm_result: dict,
    neutral_harm_result: dict,
    source_harm_result: dict,
    stat_result: dict,
    event_result: dict,
    thresh: Thresholds,
    neutral_health_result: Optional[dict] = None,
) -> dict:
    """Compile all analysis results into a structured summary dictionary."""
    df = ds.df

    transformer_pass: Optional[bool] = None
    if "transformer" in demand_result:
        transformer_pass = not demand_result["transformer"]["overloaded"]

    report = {
        "file_summary": {
            "start_time":       df.index[0].strftime("%Y-%m-%d %H:%M"),
            "end_time":         df.index[-1].strftime("%Y-%m-%d %H:%M"),
            "duration_hours":   round(ds.duration_hours, 3),
            "sample_count":     len(df),
            "channels":         sorted(df.columns.tolist()),
            "interval_minutes": ds.meta.get("interval_minutes", 5),
            "topology":         ds.meta.get("topology", "unknown"),
            "has_maxmin":       ds.has_maxmin,
            "has_adaptive":     ds.has_adaptive,
            "catalog":          ds.catalog(),
        },
        "voltage_compliance":    volt_result,
        "thd_compliance":        thd_result,
        "power_factor":          pf_result,
        "voltage_imbalance":     volt_imb_result,
        "current_imbalance":     curr_imb_result,
        "demand":                demand_result,
        "individual_harmonics":         harm_result,
        "individual_voltage_harmonics": volt_harm_result,
        "neutral_harmonics":            neutral_harm_result,
        "harmonic_sources":             source_harm_result,
        "harmonic_statistics":          stat_result,
        "events":                       event_result,
        "neutral_health":               neutral_health_result or {"available": False, "reason": "not run"},
        "pass_fail": {
            "transformer_loading":    transformer_pass,
            "voltage":                volt_result["total_pct_out_of_bounds"] == 0
                                      if volt_result["available"] else None,
            "thd_voltage":            thd_result["voltage"]["pct_exceeding"] == 0
                                      if thd_result["voltage"]["available"] else None,
            "thd_current":            thd_result["current"]["pct_exceeding"] == 0
                                      if thd_result["current"]["available"] else None,
            "individual_harmonics":   harm_result.get("overall_pass", None)
                                      if harm_result.get("available") else None,
            "individual_voltage_harmonics": volt_harm_result.get("overall_pass", None)
                                      if volt_harm_result.get("available") else None,
            "power_factor":           pf_result["pct_below_limit"] == 0
                                      if pf_result["available"] else None,
            "voltage_imbalance":      volt_imb_result["pct_exceeding"] == 0
                                      if volt_imb_result["available"] else None,
            "current_imbalance":      curr_imb_result["pct_exceeding"] == 0
                                      if curr_imb_result["available"] else None,
            "harmonic_statistics":    stat_result.get("overall_pass")
                                      if stat_result.get("available") else None,
            "neutral_health":         (
                (neutral_health_result or {}).get("severity") in ("normal", "caution")
                if (neutral_health_result or {}).get("available") else None
            ),
        },
    }
    return report


def print_report(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  POWER QUALITY ANALYSIS SUMMARY")
    print(f"{'═'*60}")
    fs = report["file_summary"]
    print(f"  Period : {fs['start_time']}  →  {fs['end_time']}")
    print(f"  Duration: {fs['duration_hours']:.2f} h   |   Samples: {fs['sample_count']:,}")
    print(f"  Channels in use: {', '.join(fs['channels'])}")

    # ── Demand ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DEMAND")
    dem = report["demand"]
    if not dem["available"]:
        print(f"  {dem['error']}")
    else:
        if "apparent_power" in dem:
            ap = dem["apparent_power"]
            lf = f"{ap['load_factor']:.3f}" if ap["load_factor"] is not None else "n/a"
            print(f"  Apparent: peak={ap['peak_kva']:.1f} kVA  mean={ap['mean_kva']:.1f} kVA  "
                  f"8-hr peak={ap['peak_8h_kva']:.1f} kVA  load factor={lf}")
        if "real_power" in dem:
            rp = dem["real_power"]
            print(f"  Real:     peak={rp['peak_kw']:.1f} kW   mean={rp['mean_kw']:.1f} kW")
        if "reactive_power" in dem:
            qp = dem["reactive_power"]
            print(f"  Reactive: peak={qp['peak_kvar']:.1f} kVAR  mean={qp['mean_kvar']:.1f} kVAR")
        if "peak_current" in dem:
            pc = dem["peak_current"]
            ph_str = "  ".join(f"{ph.upper()}={a:.0f} A" for ph, a in pc["phases"].items())
            print(f"  Peak current (interval max): {pc['max_a']:.0f} A worst  [{ph_str}]")
        if "transformer" in dem:
            tx = dem["transformer"]
            sym = "FAIL — OVERLOADED" if tx["overloaded"] else "PASS"
            print(f"  Transformer: {tx['nameplate_kva']:.0f} kVA nameplate  "
                  f"8-hr peak={tx['peak_8h_kva']:.1f} kVA ({tx['pct_nameplate']:.1f}%)  [{sym}]")
        else:
            print("  (Pass --transformer-kva to check transformer loading)")

    # ── Voltage ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  VOLTAGE (ANSI C84.1)")
    vc = report["voltage_compliance"]
    if not vc["available"]:
        print(f"  {vc['error']}")
    else:
        _any_extremes = any(s.get("used_interval_extremes") for s in vc["phases"].values())
        for ph, s in vc["phases"].items():
            sym = "PASS" if s["pct_out_of_bounds"] == 0 else "FAIL"
            print(f"  {ph:12s}: {s['min_v']:6.1f} / {s['mean_v']:6.1f} / {s['max_v']:6.1f} V  "
                  f"  {s['pct_out_of_bounds']:5.2f}% OOB  [{sym}]")
        print(f"  Allowed range: {vc['range_v'][0]:.1f} – {vc['range_v'][1]:.1f} V  "
              f"(nominal {vc['nominal_v']} V ± {(vc['range_v'][1]/vc['nominal_v']-1)*100:.0f}%)")
        if _any_extremes:
            print("  (min/max from obs[24] interval extremes; captures within-window excursions)")

    # ── THD / TDD ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    tdd_info = report["thd_compliance"].get("tdd_info", {})
    if tdd_info:
        print(f"  THD / TDD (IEEE 519-2022)")
        print(f"  ISC={tdd_info['isc_amps']:.0f} A  IL={tdd_info['il_amps']:.0f} A  "
              f"ISC/IL={tdd_info['isc_il_ratio']:.1f}  "
              f"class {tdd_info['tdd_class']}  → TDD limit {tdd_info['tdd_limit_pct']:.1f}%")
        if tdd_info.get("isc_source"):
            print(f"  ISC source: {tdd_info['isc_source']}")
    else:
        print("  THD (IEEE 519)  [pass --isc for TDD class calculation]")

    for key, label in [("voltage", "Voltage THD"), ("current", "Current TDD" if tdd_info else "Current THD")]:
        td = report["thd_compliance"][key]
        if not td["available"]:
            print(f"  {label}: no data")
            continue
        sym = "PASS" if td["pct_exceeding"] == 0 else "FAIL"
        print(f"  {label}: max={td['max_thd_pct']:.2f}%  mean={td['mean_thd_pct']:.2f}%  "
              f"limit={td['limit_pct']:.1f}%  exceed={td['pct_exceeding']:.2f}%  [{sym}]")
        if key == "current" and "peak_max_tdd_pct" in td:
            pk_sym = "PASS" if td["peak_pct_exceeding"] == 0 else "FAIL"
            print(f"  {label} (peak within interval): max={td['peak_max_tdd_pct']:.2f}%  "
                  f"exceed={td['peak_pct_exceeding']:.2f}%  [{pk_sym}]")

    # ── Power factor ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  POWER FACTOR")
    pfr = report["power_factor"]
    if not pfr["available"]:
        print(f"  {pfr['error']}")
    else:
        sym = "PASS" if pfr["pct_below_limit"] == 0 else "FAIL"
        print(f"  Min={pfr['min_pf']:.4f}  Mean={pfr['mean_pf']:.4f}  "
              f"Limit={pfr['limit']:.2f}  Below limit={pfr['pct_below_limit']:.2f}%  [{sym}]")

    # ── Voltage imbalance ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  VOLTAGE IMBALANCE (NEMA MG1 / ANSI C84.1)")
    imb = report["voltage_imbalance"]
    if not imb["available"]:
        print(f"  {imb['error']}")
    else:
        sym = "PASS" if imb["pct_exceeding"] == 0 else "FAIL"
        print(f"  Max={imb['max_imbalance_pct']:.2f}%  Mean={imb['mean_imbalance_pct']:.2f}%  "
              f"Limit={imb['limit_pct']:.1f}%  Exceed={imb['pct_exceeding']:.2f}%  [{sym}]")

    # ── Current imbalance ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CURRENT IMBALANCE")
    ci = report["current_imbalance"]
    if not ci["available"]:
        print(f"  {ci['error']}")
    else:
        sym = "PASS" if ci["pct_exceeding"] == 0 else "FAIL"
        print(f"  Max={ci['max_imbalance_pct']:.2f}%  Mean={ci['mean_imbalance_pct']:.2f}%  "
              f"Limit={ci['limit_pct']:.1f}%  Exceed={ci['pct_exceeding']:.2f}%  [{sym}]")
        if "neutral_current" in ci:
            nc = ci["neutral_current"]
            print(f"  Neutral current: mean={nc['mean_amps']:.1f} A ({nc['mean_pct_of_phase']:.1f}% of phase avg)  "
                  f"max={nc['max_amps']:.1f} A ({nc['max_pct_of_phase']:.1f}%)")

    # ── Individual harmonics ──────────────────────────────────────────────────
    print(f"\n{sep}")
    ih = report["individual_harmonics"]
    if not ih.get("available"):
        note = ih.get("note", "Pass --isc to enable per-order IEEE 519-2022 check")
        print(f"  INDIVIDUAL HARMONICS (IEEE 519-2022)  [{note}]")
    else:
        sym = "PASS" if ih["overall_pass"] else "FAIL"
        print(f"  INDIVIDUAL HARMONICS (IEEE 519-2022)  [{sym}]")
        print(f"  IL={ih['il_amps']:.0f} A  ISC/IL={ih['isc_il_ratio']:.1f}")
        # Print header + one row per harmonic order showing worst phase
        order_rows = []
        for h in _H519_ORDERS:
            worst_max = 0.0; worst_ph = None; limit = 0.0
            for ph in ("a", "b", "c"):
                r = ih["phases"].get(ph, {}).get(h)
                if r and r["max_pct_il"] > worst_max:
                    worst_max = r["max_pct_il"]; worst_ph = ph; limit = r["limit_pct_il"]
            if worst_ph:
                sym_h = "PASS" if worst_max <= limit else "FAIL"
                order_rows.append((h, worst_max, limit, worst_ph, sym_h))
        if order_rows:
            print(f"  {'H':>3}  {'worst%IL':>9}  {'limit%IL':>9}  {'ph':>3}  status")
            for h, wmax, lim, wph, sym_h in order_rows:
                marker = " ←" if sym_h == "FAIL" else ""
                print(f"  H{h:<2}  {wmax:>9.2f}  {lim:>9.1f}  {wph:>3}  {sym_h}{marker}")

    # ── Neutral harmonics ─────────────────────────────────────────────────────
    nh = report.get("neutral_harmonics", {})
    if nh.get("available"):
        print(f"\n{sep}")
        print("  NEUTRAL HARMONICS (informational)")
        t_pct = nh.get("triplen_pct", 0.0)
        acc   = nh.get("accumulation_factor")
        acc_s = f"{acc:.1f}×" if acc is not None else "n/a"
        print(f"  Triplen content: {t_pct:.0f}%  |  Accumulation factor: {acc_s}")
        for h, od in sorted(nh["orders"].items()):
            tag = " [triplen]" if od["is_triplen"] else "          "
            print(f"  H{h:<3}{tag}  mean={od['mean_a']:.3f} A  max={od['max_a']:.3f} A")

    # ── Harmonic source attribution ───────────────────────────────────────────
    sh = report.get("harmonic_sources", {})
    if sh.get("available"):
        print(f"\n{sep}")
        print("  HARMONIC SOURCE ATTRIBUTION (indicative)")
        overall_labels = {
            "customer":          "Customer-side injection",
            "resonance_suspect": "Resonance suspected",
            "mixed":             "Mixed / indeterminate",
            "indeterminate":     "Indeterminate",
        }
        print(f"  Overall: {overall_labels.get(sh.get('overall'), sh.get('overall'))}")
        resonant = sh.get("resonant_orders", [])
        if resonant:
            print(f"  Resonance suspects: {', '.join('H'+str(h) for h in sorted(resonant))}")
        print(f"  {'Order':<6}  {'Z_h(Ω)':>8}  {'Z_ratio':>8}  {'Pearson_r':>10}  Attribution")
        for h, od in sorted(sh["orders"].items()):
            ratio_s = f"{od['z_ratio']:.2f}×" if od["z_ratio"] is not None else "   n/a"
            corr_s  = f"{od['corr']:.2f}"     if od["corr"]    is not None else "   n/a"
            print(f"  H{h:<5}  {od['z_ohm']:>8.4f}  {ratio_s:>8}  {corr_s:>10}  {od['attribution']}")

    # ── Neutral health ────────────────────────────────────────────────────────
    nh = report.get("neutral_health", {})
    if nh.get("available"):
        print(f"\n{sep}")
        print("  NEUTRAL HEALTH (split-phase)")
        sev = nh["severity"].upper()
        print(
            f"  Severity: {sev}  |  "
            f"L1+L2 sum={nh['sum_mean_v']:.1f} V (std {nh['sum_std_v']:.2f} V)  |  "
            f"Leg corr r={nh['leg_correlation']:.3f}  |  "
            f"Asym={nh['asym_mean_v']:.1f} V ({nh['asym_pct']:.1f}%)"
        )
        if nh.get("vne_available"):
            print(f"  Vne: max={nh['vne_max_v']:.2f} V  mean={nh['vne_mean_v']:.2f} V")
        if nh.get("coincident_events"):
            print(f"  Coincident opposing sag/swell events: {nh['coincident_events']}")
        for f_txt in nh.get("findings", []):
            print(f"  • {f_txt}")

    # ── Root cause analysis ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ROOT CAUSE ANALYSIS")
    rca = report.get("root_causes", [])
    if not rca:
        print("  No root cause findings generated.")
    else:
        _sev_rank = {"critical": 0, "warning": 1, "info": 2}
        for finding in sorted(rca, key=lambda f: _sev_rank.get(f["severity"], 9)):
            sev   = finding["severity"].upper()
            conf  = finding["confidence"].upper()
            resp  = finding["responsibility"].upper()
            title = finding["title"]
            print(f"\n  [{sev}] [{conf} confidence] [{resp}]  {title}")
            print(f"    Finding:  {finding['finding']}")
            print(f"    Cause:    {finding['cause']}")
            print(f"    Action:   {finding['recommendation']}")

    # ── Events ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EVENTS")
    ev = report["events"]
    src_label = " (adaptive cycle-level)" if ev.get("data_source") == "adaptive" else " (interval avg)"
    print(f"  Total events detected: {ev['event_count']}{src_label}")
    if ev["event_count"] and len(ev["events"]) > 0:
        summary = ev["events"]["type"].value_counts()
        for etype, cnt in summary.items():
            print(f"    {etype:20s}: {cnt}")

    print(f"\n{'═'*60}\n")


def export_results(
    ds: PQDataset,
    report: dict,
    outdir: Path,
    stem: str = "pq_analysis",
) -> None:
    """Write CSVs: interval data, adaptive events, violations, and events."""
    df = ds.df
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Full interval dataset (avg + peak/min columns)
    data_path = outdir / f"{stem}_data.csv"
    df.to_csv(data_path)
    log.info("Saved interval data → %s", data_path)

    # 2. Adaptive (cycle-level) data when present
    if ds.has_adaptive:
        assert ds.adaptive_df is not None
        adap_path = outdir / f"{stem}_adaptive.csv"
        ds.adaptive_df.to_csv(adap_path)
        log.info("Saved adaptive data → %s  (%d rows)", adap_path, len(ds.adaptive_df))

    # 3. Violations CSV — union of all violation timestamps
    viol_sets = []
    vc = report["voltage_compliance"]
    if "violation_timestamps" in vc and len(vc["violation_timestamps"]) > 0:
        viol_sets.append(pd.Series("voltage_oob", index=vc["violation_timestamps"]))

    for key in ("voltage", "current"):
        td = report["thd_compliance"][key]
        if td and td.get("violation_timestamps"):
            idx = pd.DatetimeIndex(td["violation_timestamps"])
            viol_sets.append(pd.Series(f"thd_{key}", index=idx))

    pf_viol = report["power_factor"].get("violation_timestamps")
    if pf_viol is not None and len(pf_viol) > 0:
        viol_sets.append(pd.Series("power_factor", index=pf_viol))

    imb_viol = report["voltage_imbalance"].get("violation_timestamps")
    if imb_viol is not None and len(imb_viol) > 0:
        viol_sets.append(pd.Series("imbalance", index=imb_viol))

    if viol_sets:
        all_viols = pd.concat(viol_sets).rename("violation_type").sort_index()
        all_viols = all_viols[~all_viols.index.duplicated(keep="first")]
        viol_df = df.loc[df.index.intersection(all_viols.index)].copy()
        viol_df.insert(0, "violation_type", all_viols.reindex(viol_df.index))
        viol_path = outdir / f"{stem}_violations.csv"
        viol_df.to_csv(viol_path)
        log.info("Saved violations → %s  (%d rows)", viol_path, len(viol_df))

    # 4. Events CSV
    ev_df = report["events"]["events"]
    if len(ev_df) > 0:
        ev_path = outdir / f"{stem}_events.csv"
        ev_df.to_csv(ev_path, index=False)
        log.info("Saved events → %s  (%d rows)", ev_path, len(ev_df))


# ─────────────────────────────────────────────────────────────────────────────
# 8c. WORD REPORT GENERATOR — private section helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section_heading(doc, title: str) -> None:
    p = doc.add_paragraph()
    _bold(p, title, color=_XE_BLUE, size_pt=11)


def _body(doc, text: str) -> None:
    doc.add_paragraph(text)


def _word_site_info_table(doc, site_name, stem, site_address, meter_id, feeder, substation,
                          fs, nominal_v, nominal_ll) -> None:
    rows_data = [
        ("Customer / Site", site_name or stem),
    ]
    if site_address:
        rows_data.append(("Address", site_address))
    if meter_id:
        rows_data.append(("Meter / Account #", meter_id))
    if feeder:
        rows_data.append(("Feeder / Circuit",  feeder))
    if substation:
        rows_data.append(("Substation",         substation))
    rows_data += [
        ("Recording period", f"{fs['start_time']}  →  {fs['end_time']}"),
        ("Duration",         f"{fs['duration_hours']:.2f} hours  |  {fs['sample_count']:,} intervals"),
        ("Service voltage",  f"{nominal_v:.0f} V L-N  /  {nominal_ll} V L-L"),
        ("Topology",         fs.get("topology", "unknown")),
        ("Data sources",     (
            "Interval avg"
            + (", interval max/min" if fs.get("has_maxmin") else "")
            + (", adaptive events" if fs.get("has_adaptive") else "")
        )),
    ]
    info_tbl = doc.add_table(rows=len(rows_data), cols=2)
    info_tbl.style = 'Table Grid'
    _set_col_widths(info_tbl, [5.0, 11.5])
    for i, (label, value) in enumerate(rows_data):
        cell_l, cell_r = info_tbl.rows[i].cells
        _cell_shade(cell_l, "E8F1FA")
        cell_l.paragraphs[0].add_run(label).bold = True
        cell_r.paragraphs[0].add_run(value)
    doc.add_paragraph()


def _word_compliance_table(doc, report, thresh, df) -> None:
    pf   = report["pass_fail"]
    volt = report["voltage_compliance"]
    thd  = report["thd_compliance"]
    pfr  = report["power_factor"]
    imb  = report["voltage_imbalance"]
    ci   = report["current_imbalance"]
    dem  = report["demand"]
    ih   = report["individual_harmonics"]
    ivh  = report.get("individual_voltage_harmonics", {})
    hs   = report.get("harmonic_statistics", {})
    tdd_info = thd.get("tdd_info", {})

    hdr_p = doc.add_paragraph()
    _bold(hdr_p, "Compliance Summary", color=_XE_BLUE, size_pt=12)

    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = 'Table Grid'
    _set_col_widths(tbl, [8.5, 5.5, 2.5])

    # Header row
    hdr_cells = tbl.rows[0].cells
    for cell, text in zip(hdr_cells, ["Standard", "Measured", "Result"]):
        _cell_shade(cell, "004B87")
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(10)

    def add_row(standard, measured, passes):
        row   = tbl.add_row()
        cells = row.cells
        cells[0].paragraphs[0].add_run(standard).font.size = Pt(10)
        cells[1].paragraphs[0].add_run(measured).font.size = Pt(10)
        sym = _pf_sym(passes)
        clr = _pf_color(passes)
        _cell_shade(cells[2], "E8F4E8" if passes is True else ("FFE8E8" if passes is False else "F2F2F2"))
        r = cells[2].paragraphs[0].add_run(sym)
        r.bold = True
        r.font.size = Pt(10)
        if clr:
            r.font.color.rgb = clr

    # Demand / transformer loading
    if "transformer" in dem:
        tx   = dem["transformer"]
        meas = f"{tx['peak_8h_kva']:.0f} kVA 8-hr peak  /  {tx['nameplate_kva']:.0f} kVA nameplate  ({tx['pct_nameplate']:.0f}%)"
        add_row("Steady-state demand ≤ transformer nameplate (8-hr peak)", meas, not tx["overloaded"])
    else:
        add_row("Steady-state demand ≤ transformer nameplate (8-hr peak)", "No nameplate provided", None)

    # Power factor
    if pfr["available"]:
        meas = f"Min {pfr['min_pf']:.3f}  /  Mean {pfr['mean_pf']:.3f}  (limit ≥ {pfr['limit']:.2f})"
        add_row("Power factor ≥ 0.90 lagging (Xcel tariff)", meas, pf["power_factor"])
    else:
        add_row("Power factor ≥ 0.90 lagging (Xcel tariff)", "No data", None)

    # Voltage compliance
    if volt["available"]:
        phases = volt["phases"]
        worst_oob = max(v["pct_out_of_bounds"] for v in phases.values())
        rng = volt["range_v"]
        meas = (f"Range {rng[0]:.1f}–{rng[1]:.1f} V  |  "
                f"Worst phase: {worst_oob:.2f}% intervals out of band")
        add_row("Steady-state voltage within ANSI C84.1-2016 Range A (±5%)", meas, pf["voltage"])
    else:
        add_row("Steady-state voltage within ANSI C84.1-2016 Range A (±5%)", "No data", None)

    # Voltage transients / ITIC
    add_row("Voltage transients within ITIC curve", "See Pronto waveform data", None)

    # Voltage THD
    v_thd = thd["voltage"]
    if v_thd["available"]:
        meas = f"Max {v_thd['max_thd_pct']:.2f}%  /  Mean {v_thd['mean_thd_pct']:.2f}%  (limit {v_thd['limit_pct']:.1f}%)"
        add_row("Voltage THD < 8.0% (IEEE 519-2022, secondary)", meas, pf["thd_voltage"])
    else:
        add_row("Voltage THD < 8.0% (IEEE 519-2022, secondary)", "No voltage THD channel", None)

    # Current TDD / THD
    c_thd = thd["current"]
    if c_thd["available"]:
        metric = "TDD" if tdd_info else "THD"
        lim    = c_thd["limit_pct"]
        cls    = f"  [ISC/IL={tdd_info['isc_il_ratio']:.0f}, class {tdd_info['tdd_class']}]" if tdd_info else ""
        meas   = f"Max {c_thd['max_thd_pct']:.2f}%  /  Mean {c_thd['mean_thd_pct']:.2f}%  (limit {lim:.1f}%{cls})"
        add_row(f"Current {metric} within IEEE 519-2022 Table 2", meas, pf["thd_current"])
    else:
        add_row("Current TDD within IEEE 519-2022 Table 2", "No current THD channel", None)

    # Individual current harmonics
    if ih.get("available"):
        wo = ih.get("worst_order")
        if ih["overall_pass"]:
            meas = (f"All orders within limits  (worst: H{wo[0]} at {ih['worst_pct_of_il']:.2f}% of IL)"
                    if wo else "All orders within limits")
        else:
            meas = (f"One or more orders exceed limit  (worst: H{wo[0]} at {ih['worst_pct_of_il']:.2f}% of IL)"
                    if wo else "One or more orders exceed limit")
        add_row("Individual harmonic orders within IEEE 519-2022 Table 2", meas, pf["individual_harmonics"])
    else:
        note = ih.get("note", "Pass --isc to enable per-order check")
        add_row("Individual harmonic orders within IEEE 519-2022 Table 2", note, None)

    # Individual voltage harmonics
    if ivh.get("available"):
        vwo = ivh.get("worst_order")
        if ivh["overall_pass"]:
            meas = (f"All orders within 5% limit  (worst: H{vwo[0]} at {ivh['worst_pct_nom']:.2f}% of nominal)"
                    if vwo else "All orders within 5% limit")
        else:
            meas = (f"One or more orders exceed 5% limit  (worst: H{vwo[0]} at {ivh['worst_pct_nom']:.2f}% of nominal)"
                    if vwo else "One or more orders exceed limit")
        add_row("Individual voltage harmonics within IEEE 519-2022 Table 1 (5% of nominal)", meas,
                pf.get("individual_voltage_harmonics"))
    else:
        add_row("Individual voltage harmonics within IEEE 519-2022 Table 1 (5% of nominal)",
                ivh.get("note", "Per-order voltage harmonics not available in this meter format"), None)

    # Statistical compliance (IEEE 519-2022 Clause 5)
    if hs.get("available"):
        period_note = f"{hs['period_days']:.1f}-day recording"
        if hs["overall_pass"]:
            hs_meas = f"P95 ≤ 1.0× and P99 ≤ 1.5× limits for all orders ({period_note})"
        else:
            hs_meas = f"One or more orders exceed P95 or P99 statistical limits ({period_note})"
        add_row(
            "Harmonic P95 / P99 within IEEE 519-2022 Clause 5 statistical limits",
            hs_meas, pf.get("harmonic_statistics"),
        )
    else:
        add_row(
            "Harmonic P95 / P99 within IEEE 519-2022 Clause 5 statistical limits",
            hs.get("note", "Pass --isc to enable statistical check"), None,
        )

    # Voltage imbalance
    if imb["available"]:
        meas = f"Max {imb['max_imbalance_pct']:.2f}%  /  Mean {imb['mean_imbalance_pct']:.2f}%  (limit {imb['limit_pct']:.1f}%)"
        add_row("Voltage imbalance < 3% (ANSI C84.1 / NEMA MG1)", meas, pf["voltage_imbalance"])
    else:
        add_row("Voltage imbalance < 3% (ANSI C84.1 / NEMA MG1)", "No data", None)

    # Current imbalance
    if thresh.customer_class in ("sg", "pg"):
        ci_label = "Current imbalance < 10%  (PSCo Tariff Sheet R121 ≤ 15% for C&I)"
    else:
        ci_label = "Current imbalance < 10% (NEMA MG1)"
    if ci["available"]:
        meas = f"Max {ci['max_imbalance_pct']:.2f}%  /  Mean {ci['mean_imbalance_pct']:.2f}%  (limit {ci['limit_pct']:.1f}%)"
        add_row(ci_label, meas, pf["current_imbalance"])
    else:
        add_row(ci_label, "No data", None)

    # Flicker
    if df is not None and "flicker_pst" in df.columns and "flicker_plt" in df.columns:
        pst_max = float(df["flicker_pst"].max())
        plt_max = float(df["flicker_plt"].max())
        pst_pass = pst_max <= 1.0
        plt_pass = plt_max <= 0.65
        flicker_pass = pst_pass and plt_pass
        add_row(
            "Flicker within IEC 61000-3-3 limits (Pst ≤ 1.0, Plt ≤ 0.65)",
            f"Pst max {pst_max:.2f} (limit 1.00)  /  Plt max {plt_max:.2f} (limit 0.65)",
            flicker_pass,
        )
    else:
        add_row("Flicker within IEC 61000-3-3 limits (Pst ≤ 1.0, Plt ≤ 0.65)", "Not measured in this recording", None)

    doc.add_paragraph()


def _word_demand(doc, report, thresh) -> None:
    dem = report["demand"]

    _section_heading(doc, "Demand")
    if dem["available"]:
        ap = dem.get("apparent_power", {})
        rp = dem.get("real_power", {})
        pc = dem.get("peak_current", {})
        _pk_str = (f" Peak current within any 5-minute interval was {pc['max_a']:.0f} A."
                   if pc else "")
        _body(doc,
            f"Peak apparent demand was {ap.get('peak_kva', 0):.1f} kVA "
            f"(mean {ap.get('mean_kva', 0):.1f} kVA, load factor {ap.get('load_factor', 0):.2f}). "
            f"Peak real power was {rp.get('peak_kw', 0):.1f} kW.{_pk_str}"
        )
        if "transformer" in dem:
            tx = dem["transformer"]
            if tx["overloaded"]:
                _body(doc,
                    f"The transformer is overloaded. The nameplate is {tx['nameplate_kva']:.0f} kVA; "
                    f"the 8-hour rolling peak demand was {tx['peak_8h_kva']:.1f} kVA "
                    f"({tx['pct_nameplate']:.0f}% of nameplate). "
                    "Transformers can be loaded above nameplate on an 8-hour basis but not continuously. "
                    "A transformer upgrade should be evaluated."
                )
            else:
                _body(doc,
                    f"The transformer loading is within acceptable limits. "
                    f"The 8-hour peak was {tx['peak_8h_kva']:.1f} kVA "
                    f"({tx['pct_nameplate']:.0f}% of the {tx['nameplate_kva']:.0f} kVA nameplate)."
                )
    doc.add_paragraph()


def _word_power_factor(doc, report, thresh) -> None:
    pfr = report["power_factor"]

    _section_heading(doc, "Power Factor")
    if pfr["available"]:
        direction = "lagging" if pfr["mean_pf"] > 0 else "leading"
        is_residential = thresh.customer_class == "r"
        if is_residential:
            _body(doc,
                f"Measured mean power factor was {pfr['mean_pf']:.3f} {direction} "
                f"(minimum {pfr['min_pf']:.3f}). "
                "Residential services (PSCo Schedule R) are not subject to the power factor "
                "tariff clause — values in the 0.85–0.95 range are normal for homes with HVAC, "
                "appliances, and lighting. No corrective action is required."
            )
        elif thresh.customer_class == "pg":
            # Schedule PG — C&I Primary: Sheet R121 requires "near unity"
            if pfr["pct_below_limit"] == 0:
                _body(doc,
                    f"Power factor was maintained near unity as required by PSCo Electric Tariff "
                    f"Sheet R121 (Schedule PG — C&I Primary service). "
                    f"Measured mean {pfr['mean_pf']:.3f} {direction}, minimum {pfr['min_pf']:.3f}."
                )
            else:
                _body(doc,
                    f"Power factor fell below {pfr['limit']:.2f} during "
                    f"{pfr['pct_below_limit']:.1f}% of the recording "
                    f"(mean {pfr['mean_pf']:.3f} {direction}, minimum {pfr['min_pf']:.3f}). "
                    "PSCo Electric Tariff Sheet R121 requires Primary service customers "
                    "(Schedule PG) to maintain power factor as near unity as practicable. "
                    "The customer should evaluate power factor correction equipment to comply "
                    "with tariff requirements."
                )
        else:
            # Schedule C or SG — Sheet R73 requires PF ≥ 0.90 lagging
            sched = "SG" if thresh.customer_class == "sg" else "C"
            tariff_cite = f"PSCo Electric Tariff Sheet R73 (Schedule {sched})"
            if pfr["pct_below_limit"] == 0:
                _body(doc,
                    f"Power factor was consistently above the 0.90 lagging requirement "
                    f"({tariff_cite}). "
                    f"Measured mean {pfr['mean_pf']:.3f} {direction}, minimum {pfr['min_pf']:.3f}."
                )
            else:
                _body(doc,
                    f"Power factor fell below the 0.90 lagging requirement during "
                    f"{pfr['pct_below_limit']:.1f}% of the recording "
                    f"(mean {pfr['mean_pf']:.3f} {direction}, minimum {pfr['min_pf']:.3f}). "
                    f"{tariff_cite} requires Commercial and C&I Secondary customers to maintain "
                    f"power factor of not less than 0.90 lagging. The Company reserves the right "
                    f"to discontinue service to customers not complying with this requirement."
                )
    doc.add_paragraph()


def _word_voltage(doc, report) -> None:
    volt = report["voltage_compliance"]

    _section_heading(doc, "Steady-State Voltage")
    if volt["available"]:
        rng = volt["range_v"]
        all_pass = all(v["pct_out_of_bounds"] == 0 for v in volt["phases"].values())
        _used_ext = any(v.get("used_interval_extremes") for v in volt["phases"].values())
        _ext_note = " Extremes reflect within-interval min/max from the meter's max-min record." if _used_ext else ""
        if all_pass:
            vals = {ph: v for ph, v in volt["phases"].items()}
            phase_str = "  ".join(
                f"{ph}: {v['min_v']:.1f}–{v['max_v']:.1f} V (mean {v['mean_v']:.1f} V)"
                for ph, v in vals.items()
            )
            _body(doc,
                f"Voltage was within ANSI C84.1 Range A ({rng[0]:.1f}–{rng[1]:.1f} V) "
                f"for the entire recording period. {phase_str}.{_ext_note}"
            )
        else:
            for ph, v in volt["phases"].items():
                if v["pct_out_of_bounds"] > 0:
                    _body(doc,
                        f"Phase {ph.upper()}: {v['pct_out_of_bounds']:.2f}% of intervals were outside the "
                        f"acceptable range ({rng[0]:.1f}–{rng[1]:.1f} V). "
                        f"Min {v['min_v']:.1f} V, mean {v['mean_v']:.1f} V, max {v['max_v']:.1f} V.{_ext_note}"
                    )
    doc.add_paragraph()


def _word_harmonics(doc, report, thresh, df, outdir) -> None:
    thd      = report["thd_compliance"]
    ih       = report["individual_harmonics"]
    ivh      = report.get("individual_voltage_harmonics", {})
    nh       = report.get("neutral_harmonics", {})
    sh       = report.get("harmonic_sources", {})
    hs       = report.get("harmonic_statistics", {})
    dem      = report["demand"]
    tdd_info = thd.get("tdd_info", {})
    c_thd    = thd["current"]
    is_split = "voltage_c" not in report.get("file_summary", {}).get("channels", [])

    _section_heading(doc, "Harmonics (IEEE 519-2022)")
    if tdd_info:
        _body(doc,
            f"The available short-circuit current at the point of delivery is {tdd_info['isc_amps']:,.0f} A "
            f"(source: {tdd_info.get('isc_source', 'provided')}). "
            f"The maximum demand load current (IL) over the recording was {tdd_info['il_amps']:.0f} A. "
            f"The resulting ISC/IL ratio is {tdd_info['isc_il_ratio']:.1f}, placing this service in the "
            f"IEEE 519-2022 {tdd_info['tdd_class']} class with a TDD limit of {tdd_info['tdd_limit_pct']:.1f}%."
        )
    if c_thd["available"]:
        metric = "TDD" if tdd_info else "THD"
        if c_thd["pct_exceeding"] == 0:
            _body(doc,
                f"Current {metric} was within the {c_thd['limit_pct']:.1f}% limit throughout the recording. "
                f"Maximum {metric} was {c_thd['max_thd_pct']:.2f}%, mean {c_thd['mean_thd_pct']:.2f}%."
            )
        else:
            _body(doc,
                f"Current {metric} exceeded the {c_thd['limit_pct']:.1f}% limit during "
                f"{c_thd['pct_exceeding']:.1f}% of the recording "
                f"(max {c_thd['max_thd_pct']:.2f}%, mean {c_thd['mean_thd_pct']:.2f}%). "
                "This is the customer's responsibility to correct. Common sources include VFDs, "
                "UPS systems, switched-mode power supplies, and arc furnaces. "
                "Mitigation options include passive harmonic filters, active front-end drives, "
                "or 12-pulse converter topologies."
            )
        if "peak_max_tdd_pct" in c_thd:
            pk_pass = c_thd["peak_pct_exceeding"] == 0
            pk_verdict = "remained within" if pk_pass else "also exceeded"
            _body(doc,
                f"On a within-interval peak basis (using the meter's 5-minute max record), "
                f"current {metric} {pk_verdict} the {c_thd['limit_pct']:.1f}% limit "
                f"(peak max {c_thd['peak_max_tdd_pct']:.2f}%, "
                f"peak exceedance {c_thd['peak_pct_exceeding']:.1f}%)."
            )

    if ih.get("available") and not ih["overall_pass"]:
        fail_orders = [
            f"H{h} (phase {ph})"
            for ph in ("a", "b", "c")
            for h, r in ih["phases"].get(ph, {}).items()
            if not r["pass"]
        ]
        _body(doc,
            f"The following individual harmonic orders exceeded their IEEE 519-2022 per-order limits: "
            + ", ".join(fail_orders) + ". "
            "Individual harmonic limits are more restrictive than TDD for higher-order harmonics. "
            "See the harmonic spectrum table in the attached Pronto data (Page 7)."
        )
    elif ih.get("available"):
        h_ord = ih.get("worst_order")
        if h_ord:
            worst_r = ih["phases"].get(h_ord[1], {}).get(h_ord[0], {})
            _body(doc,
                f"All individual harmonic orders are within IEEE 519-2022 limits. "
                f"The most constraining harmonic is H{h_ord[0]} (phase {h_ord[1].upper()}) "
                f"at {ih['worst_pct_of_il']:.2f}% of IL "
                f"against a limit of {worst_r.get('limit_pct_il', '—')}%."
            )

    # Individual harmonic table (if available)
    if ih.get("available"):
        doc.add_paragraph()
        ih_hdr = doc.add_paragraph()
        _bold(ih_hdr, "Individual Harmonic Current Summary (% of IL)", size_pt=10)
        harm_tbl = doc.add_table(rows=1, cols=5)
        harm_tbl.style = 'Table Grid'
        _set_col_widths(harm_tbl, [2.0, 3.5, 3.5, 3.5, 3.5])
        for cell, text in zip(harm_tbl.rows[0].cells,
                               ["Order", "Phase A (%IL)", "Phase B (%IL)", "Phase C (%IL)", "Limit (%IL)"]):
            _cell_shade(cell, "E8F1FA")
            cell.paragraphs[0].add_run(text).bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)
        for h in _H519_ORDERS:
            # Skip orders where no phase has data
            if not any(ih["phases"].get(ph, {}).get(h) for ph in ("a", "b", "c")):
                continue
            row_cells = harm_tbl.add_row().cells
            row_cells[0].paragraphs[0].add_run(f"H{h}").font.size = Pt(9)
            limit_shown = False
            any_fail = False
            for j, ph in enumerate(("a", "b", "c")):
                r = ih["phases"].get(ph, {}).get(h)
                if r:
                    txt = f"{r['max_pct_il']:.2f}"
                    run = row_cells[j+1].paragraphs[0].add_run(txt)
                    run.font.size = Pt(9)
                    if not r["pass"]:
                        run.bold = True
                        run.font.color.rgb = _FAIL_CLR
                        any_fail = True
                    if not limit_shown:
                        row_cells[4].paragraphs[0].add_run(f"{r['limit_pct_il']:.1f}").font.size = Pt(9)
                        limit_shown = True
            if any_fail:
                for cell in row_cells:
                    _cell_shade(cell, "FFF0F0")

    # ── Individual voltage harmonic table ─────────────────────────────────────
    if ivh.get("available"):
        doc.add_paragraph()
        ivh_hdr = doc.add_paragraph()
        _bold(ivh_hdr, "Individual Harmonic Voltage Summary (% of nominal)", size_pt=10)
        doc.add_paragraph(
            f"Limit: 5.0% of nominal ({thresh.nominal_voltage:.0f} V) per IEEE 519-2022 Table 1 "
            f"(bus voltage < 1 kV). Values are absolute Volts converted to % of nominal."
        )
        volt_harm_tbl = doc.add_table(rows=1, cols=5)
        volt_harm_tbl.style = 'Table Grid'
        _set_col_widths(volt_harm_tbl, [2.0, 3.5, 3.5, 3.5, 3.5])
        for cell, text in zip(volt_harm_tbl.rows[0].cells,
                               ["Order", "Phase A (%nom)", "Phase B (%nom)", "Phase C (%nom)", "Limit (%nom)"]):
            _cell_shade(cell, "E8F1FA")
            cell.paragraphs[0].add_run(text).bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)
        for h in (3, 5, 7, 11, 13):
            if not any(ivh["phases"].get(ph, {}).get(h) for ph in ("a", "b", "c")):
                continue
            row_cells = volt_harm_tbl.add_row().cells
            row_cells[0].paragraphs[0].add_run(f"H{h}").font.size = Pt(9)
            limit_shown = False
            any_fail = False
            for j, ph in enumerate(("a", "b", "c")):
                r = ivh["phases"].get(ph, {}).get(h)
                if r:
                    txt = f"{r['max_pct_nom']:.2f}"
                    run = row_cells[j+1].paragraphs[0].add_run(txt)
                    run.font.size = Pt(9)
                    if not r["pass"]:
                        run.bold = True
                        run.font.color.rgb = _FAIL_CLR
                        any_fail = True
                    if not limit_shown:
                        row_cells[4].paragraphs[0].add_run(f"{r['limit_pct']:.1f}").font.size = Pt(9)
                        limit_shown = True
            if any_fail:
                for cell in row_cells:
                    _cell_shade(cell, "FFF0F0")

    # ── Neutral harmonic content (informational) ──────────────────────────────
    if nh.get("available"):
        doc.add_paragraph()
        nh_hdr = doc.add_paragraph()
        _bold(nh_hdr, "Neutral Harmonic Content (Informational)", size_pt=10)

        acc = nh.get("accumulation_factor")
        t_pct = nh.get("triplen_pct", 0.0)
        acc_str = f"{acc:.1f}×" if acc is not None else "n/a (phase harmonics not available)"
        if is_split:
            doc.add_paragraph(
                f"In a single-phase 3-wire (split-phase) service, the neutral carries the difference "
                f"current between L1 and L2, not the sum of zero-sequence currents from three phases. "
                f"Neutral harmonic content here reflects load imbalance between legs rather than "
                f"triplen accumulation. Triplen content: {t_pct:.0f}% of total neutral harmonic current. "
                f"Accumulation factor (H3-neutral ÷ mean H3-phase): {acc_str}."
            )
        else:
            doc.add_paragraph(
                f"Triplens (H3, H9, H15) are zero-sequence harmonics that add arithmetically in a "
                f"4-wire wye neutral. Triplen content: {t_pct:.0f}% of total neutral harmonic current. "
                f"Accumulation factor (H3-neutral ÷ mean H3-phase): {acc_str}. "
                f"Factor > 3 indicates resonance amplification; factor ≈ 3 indicates full accumulation "
                f"from balanced single-phase loads on all three phases."
            )

        nh_tbl = doc.add_table(rows=1, cols=4)
        nh_tbl.style = 'Table Grid'
        _set_col_widths(nh_tbl, [2.0, 3.5, 3.5, 3.5])
        for cell, text in zip(nh_tbl.rows[0].cells,
                               ["Order", "Mean (A)", "Max (A)", "Type"]):
            _cell_shade(cell, "E8F1FA")
            cell.paragraphs[0].add_run(text).bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)

        for h, od in sorted(nh["orders"].items()):
            row_cells = nh_tbl.add_row().cells
            row_cells[0].paragraphs[0].add_run(f"H{h}").font.size = Pt(9)
            row_cells[1].paragraphs[0].add_run(f"{od['mean_a']:.3f}").font.size = Pt(9)
            row_cells[2].paragraphs[0].add_run(f"{od['max_a']:.3f}").font.size = Pt(9)
            label = "Triplen (zero-seq)" if od["is_triplen"] else "Non-triplen"
            run = row_cells[3].paragraphs[0].add_run(label)
            run.font.size = Pt(9)
            if od["is_triplen"]:
                run.bold = True

    # ── Harmonic source attribution ───────────────────────────────────────────
    if sh.get("available"):
        doc.add_paragraph()
        sh_hdr = doc.add_paragraph()
        _bold(sh_hdr, "Harmonic Source Attribution (Indicative)", size_pt=10)

        resonant  = sh.get("resonant_orders", [])
        overall   = sh.get("overall", "indeterminate")
        res_str   = f"H{', H'.join(str(h) for h in sorted(resonant))}" if resonant else "none detected"
        overall_labels = {
            "customer":         "Customer-side injection",
            "resonance_suspect": "Resonance suspected",
            "mixed":            "Mixed / indeterminate",
            "indeterminate":    "Indeterminate",
        }
        doc.add_paragraph(
            f"Overall attribution: {overall_labels.get(overall, overall)}. "
            f"Resonance suspects: {res_str}. "
            "Attribution is based on Pearson correlation between V_h and I_h interval series; "
            "exact source direction requires waveform phasor measurements."
        )

        sh_tbl = doc.add_table(rows=1, cols=5)
        sh_tbl.style = 'Table Grid'
        _set_col_widths(sh_tbl, [1.5, 2.5, 2.5, 2.5, 3.5])
        for cell, text in zip(sh_tbl.rows[0].cells,
                               ["Order", "Z_h (Ω)", "Z_ratio", "Pearson r", "Attribution"]):
            _cell_shade(cell, "E8F1FA")
            cell.paragraphs[0].add_run(text).bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)

        for h, od in sorted(sh["orders"].items()):
            row_cells = sh_tbl.add_row().cells
            row_cells[0].paragraphs[0].add_run(f"H{h}").font.size = Pt(9)
            row_cells[1].paragraphs[0].add_run(
                f"{od['z_ohm']:.4f}").font.size = Pt(9)
            ratio_str = f"{od['z_ratio']:.2f}×" if od["z_ratio"] is not None else "—"
            row_cells[2].paragraphs[0].add_run(ratio_str).font.size = Pt(9)
            corr_str  = f"{od['corr']:.2f}" if od["corr"] is not None else "—"
            row_cells[3].paragraphs[0].add_run(corr_str).font.size = Pt(9)
            attr      = od.get("attribution", "indeterminate")
            attr_labels = {
                "customer":          "Customer",
                "resonance_suspect": "Resonance suspect",
                "indeterminate":     "Indeterminate",
            }
            attr_run = row_cells[4].paragraphs[0].add_run(attr_labels.get(attr, attr))
            attr_run.font.size = Pt(9)
            if attr == "resonance_suspect":
                attr_run.bold = True
                attr_run.font.color.rgb = _FAIL_CLR
                for cell in row_cells:
                    _cell_shade(cell, "FFF0F0")

    # ── IEEE 519-2022 Clause 5 statistical compliance tables ──────────────────
    if hs.get("available"):
        doc.add_paragraph()
        _bold(doc.add_paragraph(),
              "Statistical Harmonic Compliance — IEEE 519-2022 Clause 5", size_pt=10)
        doc.add_paragraph(
            f"Percentiles computed over the {hs['period_days']:.1f}-day recording period "
            f"(ISC/IL = {hs['isc_il_ratio']:.0f}, class {hs['isc_class']}). "
            "Short Time (ST) values use 5-minute interval data as a proxy for "
            "IEC 61000-4-30 10-minute measurements. Very Short Time (VST) values are "
            "approximated from daily P99 of 5-minute data (conservative — true VST "
            "requires 3-second measurements)."
        )

        ph_cols = [ph for ph in ("a", "b", "c")
                   if any(ph in hs["weekly"].get(k, {}) for k in hs["weekly"])]
        ph_labels = {"a": "Phase A", "b": "Phase B", "c": "Phase C"}
        stat_orders = [("h3", "H3"), ("h5", "H5"), ("h7", "H7"),
                       ("h9", "H9"), ("h11", "H11"), ("h13", "H13"),
                       ("h17", "H17"), ("h19", "H19"), ("h23", "H23"), ("h25", "H25"),
                       ("thd", "THD")]

        def _stat_table(title: str, val_key: str, lim_key: str,
                        pass_key: str, lim_label: str) -> None:
            _bold(doc.add_paragraph(), title, size_pt=9)
            n_cols = 2 + len(ph_cols)
            tbl = doc.add_table(rows=1, cols=n_cols)
            tbl.style = 'Table Grid'
            col_w = [2.0] + [4.0] * len(ph_cols) + [3.0]
            _set_col_widths(tbl, col_w[:n_cols])
            hdrs = ["Order"] + [ph_labels[p] + " (%IL)" for p in ph_cols] + [lim_label]
            for cell, txt in zip(tbl.rows[0].cells, hdrs):
                _cell_shade(cell, "E8F1FA")
                r = cell.paragraphs[0].add_run(txt)
                r.bold = True
                r.font.size = Pt(9)

            for key, label in stat_orders:
                ph_data = hs["weekly"].get(key, {})
                if not any(ph in ph_data for ph in ph_cols):
                    continue
                row = tbl.add_row()
                row.cells[0].paragraphs[0].add_run(label).font.size = Pt(9)
                any_fail_row = False
                limit_shown = False
                for j, ph in enumerate(ph_cols):
                    d = ph_data.get(ph, {})
                    if not d:
                        continue
                    val = d.get(val_key, 0.0)
                    passes = d.get(pass_key)
                    margin = d.get(
                        "p95_margin" if pass_key == "p95_pass" else "p99_margin", 0.0
                    )
                    margin_str = (
                        f" (+{margin:.2f})" if margin is not None and margin >= 0
                        else (f" ({margin:.2f})" if margin is not None else "")
                    )
                    txt = f"{val:.2f}%{margin_str}"
                    run = row.cells[j + 1].paragraphs[0].add_run(txt)
                    run.font.size = Pt(9)
                    if passes is False:
                        run.bold = True
                        run.font.color.rgb = _FAIL_CLR
                        any_fail_row = True
                    if not limit_shown:
                        lim_val = d.get(lim_key, 0.0)
                        row.cells[n_cols - 1].paragraphs[0].add_run(
                            f"{lim_val:.1f}%"
                        ).font.size = Pt(9)
                        limit_shown = True
                if any_fail_row:
                    for cell in row.cells:
                        _cell_shade(cell, "FFF0F0")

        _stat_table(
            f"Weekly 95th Percentile vs 1.0× Limit (Short Time, {hs.get('period_note', '')})",
            "p95", "limit", "p95_pass", "Limit (1.0×)",
        )
        doc.add_paragraph()
        _stat_table(
            "Weekly 99th Percentile vs 1.5× Limit (Short Time)",
            "p99", "limit_1p5x", "p99_pass", "1.5× Limit",
        )
        doc.add_paragraph()

        # VST daily proxy table (separate — uses daily_vst data)
        _bold(doc.add_paragraph(),
              "Daily 99th Percentile vs 2.0× Limit (Very Short Time proxy)", size_pt=9)
        n_cols = 2 + len(ph_cols)
        vst_tbl = doc.add_table(rows=1, cols=n_cols)
        vst_tbl.style = 'Table Grid'
        col_w = [2.0] + [4.5] * len(ph_cols) + [3.0]
        _set_col_widths(vst_tbl, col_w[:n_cols])
        hdrs = ["Order"] + [ph_labels[p] + " worst-day P99" for p in ph_cols] + ["2.0× Limit"]
        for cell, txt in zip(vst_tbl.rows[0].cells, hdrs):
            _cell_shade(cell, "E8F1FA")
            r = cell.paragraphs[0].add_run(txt)
            r.bold = True
            r.font.size = Pt(9)
        for key, label in stat_orders:
            ph_data = hs["daily_vst"].get(key, {})
            if not any(ph in ph_data for ph in ph_cols):
                continue
            row = vst_tbl.add_row()
            row.cells[0].paragraphs[0].add_run(label).font.size = Pt(9)
            any_fail_row = False
            limit_shown = False
            for j, ph in enumerate(ph_cols):
                d = ph_data.get(ph, {})
                if not d:
                    continue
                val = d.get("p99", 0.0)
                passes = d.get("pass", True)
                margin = d.get("margin", 0.0)
                margin_str = f" (+{margin:.2f})" if margin >= 0 else f" ({margin:.2f})"
                day = d.get("worst_day", "")
                txt = f"{val:.2f}%{margin_str}\n({day})"
                run = row.cells[j + 1].paragraphs[0].add_run(txt)
                run.font.size = Pt(9)
                if not passes:
                    run.bold = True
                    run.font.color.rgb = _FAIL_CLR
                    any_fail_row = True
                if not limit_shown:
                    lim_val = d.get("limit_2x", 0.0)
                    row.cells[n_cols - 1].paragraphs[0].add_run(
                        f"{lim_val:.1f}%"
                    ).font.size = Pt(9)
                    limit_shown = True
            if any_fail_row:
                for cell in row.cells:
                    _cell_shade(cell, "FFF0F0")
        doc.add_paragraph(
            "Note: True VST evaluation requires 3-second measurements per IEC 61000-4-30. "
            "Values above are daily P99 of 5-minute interval data — a conservative approximation "
            "that may not capture sub-minute harmonic peaks."
        ).runs[0].font.size = Pt(8)

    # Harmonic spectrum chart
    spec_img = outdir / "harmonic_spectrum.png"
    if spec_img.exists():
        doc.add_paragraph()
        ih_chart_hdr = doc.add_paragraph()
        _bold(ih_chart_hdr, "Current Harmonic Spectrum (Median over Recording Period)", size_pt=10)
        doc.add_picture(str(spec_img), width=Cm(15))

    # K-factor section
    if df is not None and "kfactor_meter" in df.columns:
        doc.add_paragraph()
        _section_heading(doc, "Transformer K-Factor")
        kf_med  = float(df["kfactor_meter"].median())
        kf_max  = float(df["kfactor_meter"].max())
        kf_min  = float(df["kfactor_meter"].min())
        kf_rate = int(kf_med) + (1 if kf_med % 1 >= 0.5 else 0)
        # Round up to nearest standard K-rating (4, 7, 13, 20)
        for std_k in (4, 7, 13, 20, 30, 40, 50):
            if std_k >= kf_med:
                kf_rate = std_k
                break
        if kf_med <= 1.0:
            kf_interp = "K=1 rated (standard) transformer is adequate."
        elif kf_med <= 4.0:
            kf_interp = f"K-4 rated transformer recommended — light harmonic load."
        elif kf_med <= 7.0:
            kf_interp = f"K-7 rated transformer recommended — moderate harmonic load."
        elif kf_med <= 13.0:
            kf_interp = f"K-13 rated transformer recommended — heavy harmonic load."
        else:
            kf_interp = f"K-{kf_rate} rated transformer recommended — very high harmonic load."
        _body(doc,
            f"The meter-measured harmonic K-factor (IEEE C57.110) over the recording period: "
            f"median {kf_med:.1f}, minimum {kf_min:.1f}, maximum {kf_max:.1f}. "
            f"Standard distribution transformers are designed for K=1 (sinusoidal load). "
            f"Harmonic currents cause additional eddy-current and hysteresis losses in the "
            f"transformer core and windings, reducing rated capacity and accelerating insulation "
            f"aging. {kf_interp}"
        )
        if dem.get("transformer"):
            tx     = dem["transformer"]
            pct_tx = tx.get("pct_nameplate", 0)
            _body(doc,
                f"With the transformer currently loaded to {pct_tx:.0f}% of its {tx['nameplate_kva']:.0f} kVA "
                f"nameplate and a K-factor of {kf_med:.1f}, the effective thermal load on the "
                f"transformer significantly exceeds nameplate assumptions. "
                f"A K-{kf_rate} rated unit is recommended before any additional load is added."
            )

    doc.add_paragraph()


def _word_flicker(doc, report, df) -> None:
    if df is not None and "flicker_pst" in df.columns and "flicker_plt" in df.columns:
        _section_heading(doc, "Voltage Flicker (IEC 61000-3-3)")
        pst_med = float(df["flicker_pst"].median())
        pst_max = float(df["flicker_pst"].max())
        plt_med = float(df["flicker_plt"].median())
        plt_max = float(df["flicker_plt"].max())
        pst_pass = pst_max <= 1.0
        plt_pass = plt_max <= 0.65
        if pst_pass and plt_pass:
            _body(doc,
                f"Short-term flicker severity (Pst) remained below the IEC 61000-3-3 limit of 1.0 "
                f"throughout the recording (median {pst_med:.2f}, max {pst_max:.2f}). "
                f"Long-term flicker severity (Plt) remained below the 0.65 limit "
                f"(median {plt_med:.2f}, max {plt_max:.2f}). "
                "No objectionable lamp flicker from this service is expected."
            )
        else:
            exceedances = []
            if not pst_pass:
                exceedances.append(f"Pst max {pst_max:.2f} exceeds the 1.0 limit")
            if not plt_pass:
                exceedances.append(f"Plt max {plt_max:.2f} exceeds the 0.65 limit")
            _body(doc,
                f"Flicker severity exceeded IEC 61000-3-3 limits: {'; '.join(exceedances)}. "
                f"Pst median {pst_med:.2f} (max {pst_max:.2f}, limit 1.00); "
                f"Plt median {plt_med:.2f} (max {plt_max:.2f}, limit 0.65). "
                "Flicker complaints are plausible under these conditions. Common causes include "
                "arc furnaces, large motor starting, welders, or rapidly cycling loads. "
                "Source investigation (utility vs. customer) requires voltage measurements at "
                "the service entrance with all customer loads disconnected."
            )
        doc.add_paragraph()


def _word_neutral_health(doc, report, thresh) -> None:
    nh = report.get("neutral_health", {})
    if not nh.get("available"):
        return

    _section_heading(doc, "Neutral Integrity Assessment")

    sev = nh.get("severity", "normal")
    sev_colors = {
        "critical": _FAIL_CLR,
        "warning":  RGBColor(0xCC, 0x66, 0x00),
        "caution":  RGBColor(0xCC, 0x99, 0x00),
        "normal":   _PASS_CLR,
    }
    sev_labels = {
        "critical": "CRITICAL — Open or High-Resistance Neutral Suspected",
        "warning":  "WARNING — Neutral Integrity Concern",
        "caution":  "CAUTION — Neutral Anomaly Detected",
        "normal":   "NORMAL — Neutral Appears Healthy",
    }
    sev_p = doc.add_paragraph()
    _bold(sev_p, sev_labels.get(sev, sev.upper()),
          color=sev_colors.get(sev, _XE_BLUE), size_pt=11)

    indicators = [
        ("L1 + L2 Sum (mean / std)",
         f"{nh['sum_mean_v']:.1f} V / {nh['sum_std_v']:.2f} V",
         "Healthy: ~240 V, std < 1 V"),
        ("L1–L2 Correlation (Pearson r)",
         f"{nh['leg_correlation']:.3f}",
         "Healthy: r > 0.80; open neutral → r ≈ −1"),
        ("Voltage Asymmetry |L1 − L2|",
         f"{nh['asym_mean_v']:.1f} V mean, {nh['asym_max_v']:.1f} V max ({nh['asym_pct']:.1f}%)",
         "Healthy: < 2% of nominal"),
        ("Coincident Opposing Events",
         str(nh["coincident_events"]),
         "Healthy: 0"),
    ]
    if nh.get("vne_available"):
        indicators.append((
            "Neutral-to-Earth Voltage (Vne)",
            f"{nh['vne_mean_v']:.2f} V mean, {nh['vne_max_v']:.2f} V max",
            "Healthy: < 0.5 V; > 5 V is safety hazard",
        ))

    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Table Grid"
    _set_col_widths(tbl, [3.0, 3.0, 3.5])
    for cell, hdr_txt in zip(tbl.rows[0].cells, ["Indicator", "Measured", "Benchmark"]):
        _cell_shade(cell, "E8F1FA")
        cell.paragraphs[0].add_run(hdr_txt).bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
    for ind, val, bench in indicators:
        cells = tbl.add_row().cells
        cells[0].paragraphs[0].add_run(ind).font.size = Pt(9)
        cells[1].paragraphs[0].add_run(val).font.size = Pt(9)
        cells[2].paragraphs[0].add_run(bench).font.size = Pt(9)

    doc.add_paragraph()

    for finding in nh.get("findings", []):
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(finding).font.size = Pt(10)

    if sev in ("critical", "warning"):
        doc.add_paragraph()
        rec_p = doc.add_paragraph()
        rec_p.paragraph_format.left_indent = Cm(0.5)
        rec_run = rec_p.add_run("Recommendation:  ")
        rec_run.bold = True
        rec_run.font.size = Pt(10)
        if sev == "critical":
            rec_text = (
                "An open or high-resistance neutral is a safety emergency. This can cause "
                "severe overvoltage on the lightly-loaded leg (potentially exceeding 200 V on "
                "a 120 V circuit), damaging appliances and posing shock and fire hazards. "
                "Contact Xcel Energy immediately to inspect the service neutral from the "
                "transformer secondary to the meter socket. Schedule a same-day inspection."
            )
        else:
            rec_text = (
                "Investigate the service neutral for loose connections, corrosion, or "
                "undersized conductors. Check neutral bar connections in the main panel and "
                "the meter socket lug. Xcel Energy should inspect the transformer secondary "
                "neutral and service drop. Re-measure after any repairs to confirm resolution."
            )
        rec_p.add_run(rec_text).font.size = Pt(10)

    doc.add_paragraph()


def _word_imbalance(doc, report, thresh) -> None:
    imb = report["voltage_imbalance"]
    ci  = report["current_imbalance"]

    _section_heading(doc, "Voltage and Current Imbalance")
    if imb["available"]:
        if imb["pct_exceeding"] == 0:
            _body(doc,
                f"Voltage imbalance was within the 3% limit throughout the recording. "
                f"Maximum {imb['max_imbalance_pct']:.2f}%, mean {imb['mean_imbalance_pct']:.2f}%."
            )
        else:
            _body(doc,
                f"Voltage imbalance exceeded 3% during {imb['pct_exceeding']:.1f}% of the recording "
                f"(max {imb['max_imbalance_pct']:.2f}%, mean {imb['mean_imbalance_pct']:.2f}%). "
                "Xcel Energy will investigate and correct voltage imbalance caused by the distribution system. "
                "Measurements should be repeated with all customer loads disconnected to distinguish "
                "utility-side from load-side imbalance."
            )

    if ci["available"]:
        nc_text = ""
        if "neutral_current" in ci:
            nc = ci["neutral_current"]
            nc_text = (
                f" Neutral current averaged {nc['mean_amps']:.1f} A "
                f"({nc['mean_pct_of_phase']:.1f}% of phase average) with a peak of "
                f"{nc['max_amps']:.1f} A ({nc['max_pct_of_phase']:.1f}%)."
            )
            if nc["mean_pct_of_phase"] > 15:
                nc_text += (
                    " Elevated neutral current is consistent with load imbalance and/or significant "
                    "triplen harmonic currents (3rd, 9th, 15th) from nonlinear single-phase loads "
                    "such as computers, lighting controls, and variable-speed drives."
                )

        if ci["pct_exceeding"] == 0:
            _body(doc,
                f"Current imbalance was within the 10% limit throughout the recording. "
                f"Maximum {ci['max_imbalance_pct']:.2f}%, mean {ci['mean_imbalance_pct']:.2f}%.{nc_text}"
            )
        else:
            _body(doc,
                f"Current imbalance exceeded 10% during {ci['pct_exceeding']:.1f}% of the recording "
                f"(max {ci['max_imbalance_pct']:.2f}%, mean {ci['mean_imbalance_pct']:.2f}%). "
                f"Load imbalance is the customer's responsibility to correct.{nc_text}"
            )
    doc.add_paragraph()


def _word_events(doc, report) -> None:
    ev = report["events"]

    _section_heading(doc, "Voltage & Flicker Events")
    adap_note = (
        " Event detection used cycle-level adaptive records (~17 ms resolution),"
        " which capture within-interval sags/swells missed by 5-minute averages."
        if ev.get("data_source") == "adaptive"
        else " Event detection used 5-minute interval averages."
    )
    if ev["event_count"] == 0:
        _body(doc,
            "No significant voltage sag, swell, transient, or flicker events were detected"
            " during the recording period." + adap_note
        )
    else:
        edf = ev["events"]
        type_counts = edf["type"].value_counts().to_dict() if len(edf) > 0 else {}
        parts = [f"{cnt} {etype.replace('_', ' ')}" for etype, cnt in sorted(type_counts.items())]
        _body(doc,
            f"{ev['event_count']} event(s) detected: {', '.join(parts)}." + adap_note + " "
            "Voltage event causes may include faults on adjacent feeders, motor starting inrush, "
            "transformer energization, or switching operations. "
            "Flicker events (PST > 1.0 or PLT > 0.65) indicate arc-type or intermittent loads."
        )
    doc.add_paragraph()


def _word_rca(doc, report, thresh) -> None:
    pf  = report["pass_fail"]
    rca = report.get("root_causes", [])

    # ── Root cause analysis section ───────────────────────────────────────────
    if rca:
        _section_heading(doc, "Root Cause Analysis")
        _sev_rank = {"critical": 0, "warning": 1, "info": 2}
        _sev_label = {"critical": "Critical", "warning": "Warning", "info": "Observation"}
        _resp_label = {"utility": "Utility responsibility",
                       "customer": "Customer responsibility",
                       "shared": "Shared responsibility",
                       "unknown": "Responsibility TBD"}
        for finding in sorted(rca, key=lambda f: _sev_rank.get(f["severity"], 9)):
            sev  = finding["severity"]
            resp = finding.get("responsibility", "unknown")
            conf = finding.get("confidence", "")
            p = doc.add_paragraph()
            _bold(p, f"{_sev_label.get(sev, sev).upper()}: {finding['title']}",
                  color=(_FAIL_CLR if sev == "critical" else
                         RGBColor(0xCC, 0x66, 0x00) if sev == "warning" else _XE_BLUE),
                  size_pt=10)
            tag_txt = f"[{_resp_label.get(resp, resp)}  |  {conf.capitalize()} confidence]"
            tag = p.add_run(f"  {tag_txt}")
            tag.font.size = Pt(9)
            tag.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

            body_p = doc.add_paragraph()
            body_p.paragraph_format.left_indent = Cm(0.5)
            run_f = body_p.add_run("Finding:  ")
            run_f.bold = True
            run_f.font.size = Pt(10)
            body_p.add_run(finding["finding"]).font.size = Pt(10)

            body_p2 = doc.add_paragraph()
            body_p2.paragraph_format.left_indent = Cm(0.5)
            run_c = body_p2.add_run("Cause:  ")
            run_c.bold = True
            run_c.font.size = Pt(10)
            body_p2.add_run(finding["cause"]).font.size = Pt(10)

            doc.add_paragraph()

    # ── Recommended customer actions ──────────────────────────────────────────
    _section_heading(doc, "Recommended Actions")

    # Pull recommendations from root causes (customer/shared responsibility only)
    rca_actions = [
        f["recommendation"]
        for f in sorted(rca, key=lambda f: {"critical": 0, "warning": 1, "info": 2}.get(f["severity"], 9))
        if f.get("responsibility") in ("customer", "shared") and f.get("recommendation")
    ]
    # Add any compliance-driven actions not already covered by root cause rules
    fallback_actions = []
    if (thresh.customer_class != "r"
            and not any("power factor" in a.lower() for a in rca_actions)):
        if pf["power_factor"] is False:
            if thresh.customer_class == "pg":
                fallback_actions.append(
                    "Install power factor correction to maintain near unity power factor per "
                    "PSCo Electric Tariff Sheet R121 (Schedule PG — C&I Primary service)."
                )
            else:
                sched = "SG" if thresh.customer_class == "sg" else "C"
                fallback_actions.append(
                    f"Install power factor correction capacitors to bring power factor above "
                    f"0.90 lagging per PSCo Electric Tariff Sheet R73 (Schedule {sched})."
                )
    if not any("harmonic" in a.lower() or "vfd" in a.lower() or "rectifier" in a.lower()
               for a in rca_actions):
        if pf.get("thd_current") is False:
            fallback_actions.append(
                "Investigate and mitigate harmonic current sources (VFDs, rectifiers, UPS). "
                "Consider passive or active harmonic filters, or 12-pulse drive topologies."
            )
        if pf.get("individual_harmonics") is False:
            fallback_actions.append(
                "Specific harmonic orders exceed IEEE 519-2022 per-order limits. "
                "A detailed harmonic study with individual source identification is recommended."
            )
    if not any("imbalance" in a.lower() or "balance" in a.lower() for a in rca_actions):
        if pf["current_imbalance"] is False:
            fallback_actions.append(
                "Balance single-phase loads across phases to reduce current imbalance. "
                "Investigate whether triplen harmonics are contributing to elevated neutral current."
            )
    if pf.get("transformer_loading") is False:
        fallback_actions.append(
            "The serving transformer is overloaded. Contact your Xcel Energy Area Engineer to "
            "discuss a transformer upgrade."
        )
    # Utility-responsibility items
    utility_actions = [
        f["recommendation"]
        for f in rca
        if f.get("responsibility") == "utility" and f.get("recommendation")
    ]
    if pf["voltage"] is False and not utility_actions:
        fallback_actions.append(
            "Steady-state voltage is outside ANSI C84.1 Range A. "
            "Xcel Energy will investigate the distribution system for this condition."
        )

    all_actions = rca_actions + fallback_actions + utility_actions

    if not all_actions:
        _body(doc,
            "No corrective actions are required at this time. All measured parameters are within "
            "applicable standards. Continue to monitor power quality if issues recur."
        )
    else:
        for i, action in enumerate(all_actions, 1):
            p = doc.add_paragraph(style='List Number')
            p.add_run(action)

    doc.add_paragraph()


def _word_signoff(doc, engineer_name, engineer_title, engineer_phone, engineer_email,
                  engineer_contact) -> None:
    doc.add_paragraph("Sincerely,")
    doc.add_paragraph()
    p = doc.add_paragraph()
    _bold(p, engineer_name or "[Engineer Name]")
    title_line = engineer_title or "Electric Area Engineer"
    doc.add_paragraph(title_line)
    doc.add_paragraph("Xcel Energy — PSCo Area Engineering")
    # Phone / email — show individually if provided, fall back to combined contact string
    if engineer_phone or engineer_email:
        if engineer_phone:
            doc.add_paragraph(f"Phone: {engineer_phone}")
        if engineer_email:
            doc.add_paragraph(f"Email: {engineer_email}")
    elif engineer_contact:
        doc.add_paragraph(engineer_contact)


# ─────────────────────────────────────────────────────────────────────────────
# 8c. WORD REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_word_report(
    report: dict,
    thresh: Thresholds,
    site_name: str,
    site_address: str,
    engineer_name: str,
    engineer_contact: str,
    outdir: Path,
    stem: str,
    *,
    ds: Optional["PQDataset"] = None,
    meter_id: str = "",
    feeder: str = "",
    substation: str = "",
    engineer_title: str = "",
    engineer_phone: str = "",
    engineer_email: str = "",
) -> Optional[Path]:
    """Generate a Word (.docx) power quality response letter matching the PSC template."""
    if not _DOCX_AVAILABLE:
        log.warning("python-docx not installed — skipping Word report. pip install python-docx")
        return None

    df: Optional[pd.DataFrame] = ds.df if ds is not None else None
    doc = _DocxDocument()

    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    fs         = report["file_summary"]
    nominal_v  = thresh.nominal_voltage
    is_split   = "voltage_c" not in fs.get("channels", [])
    nominal_ll = round(nominal_v * 2) if is_split else round(nominal_v * 3**0.5)

    hdr = doc.add_paragraph()
    hdr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _bold(hdr, "Xcel Energy — Power Quality Analysis Report", color=_XE_BLUE, size_pt=14)
    doc.add_paragraph()

    _word_site_info_table(doc, site_name, stem, site_address, meter_id, feeder, substation,
                          fs, nominal_v, nominal_ll)

    opening = doc.add_paragraph()
    opening.add_run(
        "The power quality standards applicable to this service and the measurement results "
        "are summarized below. The following table shows compliance status against each "
        "standard. Sections where the standard is not met are discussed in detail."
    )
    doc.add_paragraph()

    _word_compliance_table(doc, report, thresh, df)
    _word_demand(doc, report, thresh)
    _word_power_factor(doc, report, thresh)
    _word_voltage(doc, report)
    _word_harmonics(doc, report, thresh, df, outdir)
    _word_flicker(doc, report, df)
    _word_imbalance(doc, report, thresh)
    _word_neutral_health(doc, report, thresh)
    _word_events(doc, report)
    _word_rca(doc, report, thresh)
    _word_signoff(doc, engineer_name, engineer_title, engineer_phone, engineer_email,
                  engineer_contact)

    # ── Save ──────────────────────────────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{stem}_report.docx"
    doc.save(out_path)
    log.info("Word report saved → %s", out_path)
    return out_path

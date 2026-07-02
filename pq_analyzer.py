"""
pq_analyzer.py — PQ Analyzer CLI entry point.

The analysis engine is split across focused modules:
  pq_constants.py  — Thresholds, IEEE tables, Blue Book ISC lookup
  pq_adapter.py    — Channel mapping, PQDIF adapters, PQDataset
  pq_analysis.py   — Compliance checks, harmonic detection, root cause analysis
  pq_report.py     — Report generation, Word export, CSV export
  pq_plots.py      — Matplotlib visualizations

This file contains only parse_args() and main().
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Re-export symbols that run.py imports from this module ───────────────────
from pq_constants import (
    __version__,
    Thresholds,
    _BLUE_BOOK_ISC,
    _SERVICE_TYPE_LABEL,
    _infer_secondary_v,
    _lookup_isc,
)
from pq_adapter import (
    PQDIFAdapter,
    ProntoAdapter,
    MockAdapter,
    ChannelMapper,
    PQDataset,
    extract_dataset,
    _PQDIF_AVAILABLE,
)
from pq_analysis import (
    check_voltage_compliance,
    check_thd,
    check_power_factor,
    check_voltage_imbalance,
    check_current_imbalance,
    check_demand,
    check_individual_harmonics,
    check_individual_voltage_harmonics,
    check_neutral_harmonics,
    check_harmonic_sources,
    check_harmonic_statistics,
    detect_events,
    check_neutral_health,
    analyze_root_causes,
)
from pq_report import (
    generate_report,
    print_report,
    export_results,
    generate_word_report,
)
from pq_plots import (
    plot_voltage,
    plot_thd,
    plot_summary,
    plot_harmonic_spectrum,
    plot_itic,
    plot_neutral_health,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pq_analyzer")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=f"PQDIF Power Quality Analyzer v{__version__} — PSCo electric service compliance tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
STANDARDS APPLIED
  Voltage       ANSI C84.1 Range A  (nominal ± volt-tol, default ±5%)
  Current THD   IEEE 519-2022 TDD when --isc is provided; raw interval THD
                fallback otherwise (light-load intervals < 10% of peak demand
                are automatically excluded to prevent divide-by-zero blowup)
  Power factor  PSCo Tariff R73 (Schedules C/SG, ≥ 0.90 lag) / R121 (PG)
  Flicker       IEC 61000-3-3  (Pst ≤ 1.0, Plt ≤ 0.65)
  Imbalance     NEMA MG1 / IEEE 112  (voltage ≤ 3%, current ≤ 10%)
  Neutral       Split-phase only: L1+L2 sum stability, cross-leg Pearson r,
                Vne, and coincident opposing sag/swell event detection

CUSTOMER CLASSES  (--customer-class)
  r    Residential       120 V split-phase      No PF tariff; open-neutral check active
  c    Small Commercial  120/208 V 3-phase      Tariff R73 Schedule C
  sg   C&I Secondary     277/480 V 3-phase      Tariff R73 Schedule SG  [default]
  pg   C&I Primary       13,200+ V 3-phase      Tariff R121 Schedule PG

IEEE 519-2022 TDD  (--isc, --transformer-kva, --service-type)
  TDD(t) = THD%(t) × I(t) / IL   where IL = peak demand current in recording.
  ISC/IL ratio selects the per-Table-2 TDD class limit (5 / 8 / 12 / 15 / 20%):

    ISC/IL < 20   →  5%     ISC/IL < 100  → 12%     ISC/IL ≥ 1000 → 20%
    ISC/IL < 50   →  8%     ISC/IL < 1000 → 15%

  Provide --isc directly, or auto-look up from the PSCo Blue Book:
    --transformer-kva 500 --service-type 3ph-padmount
  Service types: 1ph-padmount, 3ph-padmount, 1ph-overhead, 3ph-overhead,
                 network, spot-network

TOPOLOGY  (--topology)
  auto          Inferred from loaded channels: no Vcn → split-phase (default)
  split-phase   Force single-phase 3-wire; activates neutral integrity section
  3ph-wye       Force three-phase 4-wire

OUTPUT
  Plots (.png)  Voltage, THD, summary, harmonic spectrum, ITIC, neutral health
  CSV           Per-interval data export alongside the plots
  Word (.docx)  Full engineering response letter; requires --report flag

EXAMPLES
  Residential — 120 V split-phase, open-neutral check, Word report:
    python3 pq_analyzer.py site.pqd --nominal 120 --customer-class r --report \\
      --site-name "123 Main St" --engineer "J. Smith" --engineer-title "Area Engineer"

  Small commercial — Blue Book ISC auto-lookup, 150 kVA padmount:
    python3 pq_analyzer.py site.pqd --nominal 120 --customer-class c \\
      --transformer-kva 150 --service-type 3ph-padmount --report

  C&I Secondary — 480 V, manual ISC, transformer loading check:
    python3 pq_analyzer.py site.pqd --nominal 277 --customer-class sg \\
      --isc 10000 --transformer-kva 1000 --report

  C&I Primary — 13.2 kV, typical 5 kA fault current:
    python3 pq_analyzer.py site.pqd --nominal 13200 --customer-class pg \\
      --isc 5000 --report

  Debug channel mapping (use before analysis if channels are missing):
    python3 pq_analyzer.py site.pqd --list-channels

  Demo mode (synthetic data, no file required):
    python3 pq_analyzer.py --demo --nominal 277 --customer-class sg
""",
    )
    p.add_argument("--version", action="version", version=f"pq-analyzer {__version__}")
    p.add_argument("filepath", nargs="?", help="Path to .pqd PQDIF file")
    p.add_argument("--demo",          action="store_true", help="Run with synthetic demo data")
    p.add_argument("--list-channels", action="store_true", help="Print all channels and exit")
    p.add_argument("--nominal",   type=float, default=120.0,  help="Nominal voltage V (default 120)")
    p.add_argument("--volt-tol",  type=float, default=0.05,   help="Voltage tolerance ±fraction (default 0.05)")
    p.add_argument("--thd-limit", type=float, default=8.0,    help="Voltage THD %% limit (default 8.0)")
    p.add_argument("--pf-limit",  type=float, default=0.90,   help="Power factor lower limit (default 0.90)")
    p.add_argument("--imb-limit", type=float, default=3.0,    help="Voltage imbalance %% limit (default 3.0)")
    p.add_argument("--curr-imb-limit", type=float, default=10.0,
                   help="Current imbalance %% limit (default 10.0, per NEMA MG1)")
    p.add_argument("--isc",       type=float, default=None,
                   help="Available short-circuit current at service point (A); enables IEEE 519-2022 TDD class. "
                        "Auto-calculated from Blue Book when --transformer-kva and --service-type are provided.")
    p.add_argument("--transformer-kva", type=float, default=None,
                   help="Service transformer nameplate kVA; enables transformer loading check and ISC auto-lookup")
    p.add_argument("--topology", default="auto",
                   choices=["auto", "3ph-wye", "split-phase"],
                   help=("Service topology: '3ph-wye' (three-phase wye), "
                         "'split-phase' (120/240 V single-phase), "
                         "or 'auto' (default — inferred from loaded channels)."))
    p.add_argument("--service-type", default=None,
                   choices=list(_SERVICE_TYPE_LABEL.keys()),
                   metavar="TYPE",
                   help=("Transformer service type for Blue Book ISC lookup. "
                         "Choices: " + ", ".join(_SERVICE_TYPE_LABEL.keys()) + ". "
                         "Default: 3ph-padmount when --nominal≥200, else 1ph-padmount."))
    p.add_argument("--resample",  default=None,  help="Resample interval, e.g. '1s', '1min', '10min'")
    p.add_argument("--outdir",    default=str(Path(__file__).parent / "pq_output"),
                   help="Output directory (default: pq_output/ next to this script)")
    p.add_argument("--no-plots",  action="store_true", help="Skip plot generation")
    p.add_argument("--report",    action="store_true", help="Generate Word (.docx) report")
    p.add_argument("--site-name",      default=None, help="Site name for the report header")
    p.add_argument("--site-address",   default=None, help="Site address for the report header")
    p.add_argument("--meter-id",       default=None, help="Meter or account number for the report header")
    p.add_argument("--feeder",         default=None, help="Feeder / circuit name for the report header")
    p.add_argument("--substation",     default=None, help="Substation name for the report header")
    p.add_argument("--engineer",       default=None, help="Engineer name for the report sign-off")
    p.add_argument("--engineer-title", default=None, help="Engineer title (default: Electric Area Engineer)")
    p.add_argument("--engineer-phone", default=None, help="Engineer phone number for sign-off")
    p.add_argument("--engineer-email", default=None, help="Engineer email address for sign-off")
    p.add_argument("--engineer-contact", default=None, help="Legacy combined phone/email for sign-off")
    p.add_argument("--customer-class", default="sg",
                   choices=["r", "c", "sg", "pg"],
                   help="PSCo tariff schedule: r=Residential, c=Small Comm., sg=C&I Secondary, pg=C&I Primary")
    p.add_argument("--verbose",   action="store_true", help="Debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # ── ISC auto-lookup from Blue Book ────────────────────────────────────────
    isc_amps = args.isc
    isc_note: str | None = None
    if isc_amps is None and args.transformer_kva is not None:
        svc_type = args.service_type
        if svc_type is None:
            svc_type = "3ph-padmount" if args.nominal >= 200 else "1ph-padmount"
            log.info(
                "--service-type not specified; assuming %s based on --nominal %.0f V",
                svc_type, args.nominal,
            )
        result = _lookup_isc(svc_type, args.transformer_kva, args.nominal)
        if result:
            isc_amps, isc_note = result
            log.info("Blue Book ISC lookup: %s", isc_note)
        else:
            log.warning(
                "No Blue Book entry for service-type=%s kVA=%.0f nominal=%.0f V. "
                "Pass --isc manually for TDD calculation.",
                svc_type, args.transformer_kva, args.nominal,
            )

    isc_source: str | None = None
    if args.isc is not None:
        isc_source = f"Manual (--isc {args.isc:.0f} A)"
    elif isc_amps is not None:
        isc_source = isc_note

    thresh = Thresholds(
        nominal_voltage=args.nominal,
        volt_tolerance=args.volt_tol,
        thd_voltage_limit=args.thd_limit,
        power_factor_limit=args.pf_limit,
        imbalance_limit=args.imb_limit,
        current_imbalance_limit=args.curr_imb_limit,
        isc_amps=isc_amps,
        isc_source=isc_source,
        transformer_kva=args.transformer_kva,
        customer_class=args.customer_class,
    )

    # ── Choose adapter ────────────────────────────────────────────────────────
    if args.demo:
        log.info("Running in DEMO mode with synthetic data.")
        adapter = MockAdapter(duration_hours=2.0, nominal=args.nominal)
        stem = "demo"
    elif args.filepath:
        fp = Path(args.filepath)
        if not fp.exists():
            log.error("File not found: %s", fp)
            sys.exit(1)
        log.info("Opening %s  (%.1f MB)", fp, fp.stat().st_size / 1e6)
        if fp.suffix.lower() == ".pqd":
            adapter = ProntoAdapter(fp)
        elif _PQDIF_AVAILABLE:
            adapter = PQDIFAdapter(fp)
        else:
            log.error(
                "pqdifpy is not installed and this is not a .pqd file.\n"
                "  pip install pqdifpy   or use a .pqd Pronto file."
            )
            sys.exit(1)
        stem = fp.stem
    else:
        log.error("Provide a .pqd file or use --demo.")
        sys.exit(1)

    # ── List-channels debug mode ──────────────────────────────────────────────
    if args.list_channels:
        channels = adapter.list_channels()
        print(f"\nFound {len(channels)} channels:\n")
        for ch in channels:
            print(ch.debug_str())
        print(
            "\nHint: copy any label above into _NAME_PATTERNS in pq_adapter.py "
            "if it is not being matched automatically."
        )
        return

    # ── Extract unified dataset ───────────────────────────────────────────────
    mapper = ChannelMapper()
    ds = extract_dataset(adapter, mapper, resample=args.resample)

    if ds.df.empty:
        log.error("DataFrame is empty after extraction. Check channel matching.")
        sys.exit(1)

    # ── Run analysis ──────────────────────────────────────────────────────────
    log.info("Running compliance analysis …")
    df = ds.df  # shorthand for plot functions
    volt_result      = check_voltage_compliance(df, thresh)
    thd_result       = check_thd(df, thresh)
    pf_result        = check_power_factor(df, thresh)
    imb_result       = check_voltage_imbalance(df, thresh)
    curr_imb_result  = check_current_imbalance(df, thresh)
    demand_result    = check_demand(df, thresh)
    harm_result         = check_individual_harmonics(df, thresh)
    volt_harm_result    = check_individual_voltage_harmonics(df, thresh)
    neutral_harm_result = check_neutral_harmonics(df, thresh)
    source_harm_result  = check_harmonic_sources(df, thresh)
    stat_result         = check_harmonic_statistics(df, thresh)
    event_result        = detect_events(ds, thresh)
    neutral_health_result = check_neutral_health(ds, thresh)

    # ── Compile report ────────────────────────────────────────────────────────
    report = generate_report(
        ds, volt_result, thd_result, pf_result,
        imb_result, curr_imb_result, demand_result,
        harm_result, volt_harm_result, neutral_harm_result,
        source_harm_result, stat_result, event_result, thresh,
        neutral_health_result=neutral_health_result,
    )
    report["root_causes"] = analyze_root_causes(report, ds, thresh)

    print_report(report)

    # ── Export ────────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    export_results(ds, report, outdir, stem=stem)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        log.info("Generating plots …")
        plot_voltage(df, volt_result, thresh, outdir=outdir)
        plot_thd(df, thd_result, thresh, outdir=outdir)
        plot_summary(df, imb_result, outdir=outdir)
        plot_harmonic_spectrum(df, thresh, outdir=outdir)
        plot_itic(event_result["events"], thresh, outdir=outdir)
        plot_neutral_health(ds, neutral_health_result, thresh, outdir=outdir)
        log.info("All plots saved to %s/", outdir)

    # ── Word report ───────────────────────────────────────────────────────────
    if args.report:
        generate_word_report(
            report=report,
            thresh=thresh,
            ds=ds,
            site_name=args.site_name or stem,
            site_address=args.site_address or "",
            engineer_name=args.engineer or "",
            engineer_contact=args.engineer_contact or "",
            outdir=outdir,
            stem=stem,
            meter_id=args.meter_id or "",
            feeder=args.feeder or "",
            substation=args.substation or "",
            engineer_title=args.engineer_title or "",
            engineer_phone=args.engineer_phone or "",
            engineer_email=args.engineer_email or "",
        )

    log.info("Done.")


if __name__ == "__main__":
    main()

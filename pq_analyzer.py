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
    conductor_label,
    conductor_options,
    is_single_phase_208,
    isc_lookup_type,
    ll_factor,
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
    check_harmonic_direction,
    check_source_impedance,
    check_harmonic_sources,
    check_spectral_shape,
    check_harmonic_statistics,
    detect_events,
    check_neutral_health,
    check_itic,
    check_line_to_line_voltage,
    check_frequency,
    check_flicker,
    kfactor_by_phase,
    analyze_root_causes,
)
from pq_report import (
    generate_report,
    print_report,
    export_results,
    generate_customer_letter,
    generate_word_report,
)
from pq_plots import (
    plot_overview,
    plot_voltage,
    plot_thd,
    plot_summary,
    plot_harmonic_spectrum,
    plot_itic,
    plot_neutral_health,
    plot_demand_profile,
    plot_harmonic_trend,
    plot_imbalance,
    plot_flicker,
    plot_pf_load,
    plot_waveform_capture,
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
  Voltage       ANSI C84.1-2016 service voltage, on interval averages: Range A,
                Range B, or outside both. Table 1 has two groups —
                  120–600 V     Range A ±5%,        Range B −8.33%/+5.83%
                  2.4–34.5 kV   Range A 97.5–105%,  Range B 95–105.8%
                The over-600 V group is tighter below nominal, reserving that
                headroom for the drop through the customer's own transformation.
                Sags and swells are graded against ITIC, not against C84.1.
  Current THD   IEEE 519-2022 TDD when --isc is provided; raw interval THD
                fallback otherwise (light-load intervals < 10% of peak demand
                are automatically excluded to prevent divide-by-zero blowup)
  Power factor  PSCo Tariff R73 (≥ 0.90 lagging, General rules — all classes)
                and R121 (near unity, C&I rules — Schedules C, SG and PG)
  Flicker       IEC 61000-3-3  (Pst ≤ 1.0, Plt ≤ 0.65)
  Imbalance     NEMA MG1 / IEEE 112  (voltage ≤ 3%, current ≤ 10%)
  Neutral       Split-phase only: L1+L2 sum stability, cross-leg Pearson r,
                Vne, and coincident opposing sag/swell event detection

CUSTOMER CLASSES  (--customer-class)
  r    Residential       120 V split-phase      Open-neutral check active
  c    Small Commercial  120/208 V 3-phase      Demand < 50 kW, secondary
  sg   C&I Secondary     277/480 V 3-phase      Secondary voltage  [default]
  pg   C&I Primary       13,200+ V 3-phase      Primary voltage

  The tariff separates C from SG on the 50 kW demand in Schedule C, and SG
  from PG on secondary versus primary voltage. SG itself states no kW floor.

  Net metering is not one of these. Schedule NM applies "as a service element
  under all rate schedules", so a solar customer keeps the class above and sets
  --service-role.

WHICH WAY POWER FLOWS  (--service-role, --rated-ac-kw)
  Electrically there are three cases, and the schedule name does not decide it —
  what is physically behind the meter does:

    load        consumes only. Includes the solar schedules where the array is
                somewhere else: OS-NM (other property), RC/RCF (Renewable*Connect
                subscription), SRCS (Solar*Rewards Community share). These bill
                like solar and measure like any other load.
    mixed       load and generation in parallel: NM, PV, RE (recycled energy —
                waste heat, not solar), AVPP (aggregated batteries).
    generation  a plant with no load worth the name — a Solar*Rewards Community
                producer's array on the Company's production meter. Note SRCS
                names the subscribers who buy its output, not the array itself.

  CT polarity is why the middle case is separate. A load should import and a
  plant should export, so a wrong sign catches reversed clamps at either end;
  in the middle both signs are legitimate and the check cannot be made at all.

  At a generation site IL has no demand load to come from, so pass --rated-ac-kw
  with the plant nameplate. Without it IL is the largest export measured, which
  grades the plant against the week it happened to have rather than what it can
  do — a cloudy recording then inflates every percentage taken against it.

IEEE 519-2022 TDD  (--isc, --transformer-kva, --service-type, --il-amps)
  Applies where 519 governs — see WHICH STANDARD below.
  TDD(t) = 100 × Ih(t) / IL, with Ih the harmonic RMS current and IL the maximum
  demand current at the fundamental — not the peak RMS, which is larger by
  sqrt(1 + THD²).
  519-2022 defines IL as the twelve previous months' 15- or 30-min maximum
  demands averaged: a billing quantity, not a measurement. Pass --il-amps with
  it. Without it the recording's largest fundamental stands in and every
  percentage taken against it moves with how typical that week was.
  ISC/IL ratio selects the per-Table-2 TDD class limit (5 / 8 / 12 / 15 / 20%):

    ISC/IL < 20   →  5%     ISC/IL < 100  → 12%     ISC/IL ≥ 1000 → 20%
    ISC/IL < 50   →  8%     ISC/IL < 1000 → 15%

  Provide --isc directly, or auto-look up from the PSCo Blue Book:
    --transformer-kva 500 --service-type 3ph-padmount
  Service types: 1ph-overhead, 1ph-padmount, 1ph-208, 3ph-padmount,
                 3ph-overhead-wye, 3ph-open-delta, 3ph-closed-delta

  Three secondary configurations, two of which share a transformer:
    split phase 120/240   center-tapped single-phase can; legs 180° apart
    three phase 120/208   three-phase transformer, customer pulls all phases
    single phase 120/208  same transformer, customer pulls two legs only;
                          legs 120° apart, so L-L is 208 V and the neutral
                          carries the vector sum rather than the difference
  Use --service-type 1ph-208 for the third. Its Blue Book fault current is
  read from the three-phase rows, because it is the same transformer.

WHICH STANDARD  (--rated-ac-kw, --annual-avg-load-kw)
  519-2022 Clause 5.2 limits its own scope to a PCC "primarily with harmonic
  producing loads" and sends inverter-based installations elsewhere. Figure 1
  is the decision tree, and this tool follows it:

    DER or IBR present?             no  → IEEE 519 at the PCC
    rated generation < 10% of
      annual average load demand?   yes → IEEE 519 at the PCC
                                    no  → IEEE 1547 (Clause 7.3)

  Under 1547 the metric is TRD = sqrt(I_rms² − I₁²) / I_rated, the limits are
  fixed (4.0/2.0/1.5/0.6/0.3 by order, 5.0 aggregate) and there is no ISC/IL
  class. Both inputs to the test come from records, so without them the tool
  reports the standard as undetermined rather than guessing.

TOPOLOGY  (--topology)
  auto          Inferred from loaded channels: no Vcn → split-phase (default)
  split-phase   Force single-phase 3-wire; activates neutral integrity section
  3ph-wye       Force three-phase 4-wire

OUTPUT
  Plots (.png)  Voltage, THD, summary, harmonic spectrum, ITIC, neutral health
  CSV           Per-interval data export alongside the plots
  Word (.docx)  Two documents, both written with --report:
                <stem>_internal_engineering_report.docx  (internal, all classes)
                <stem>_customer_letter.docx              (customer, all classes)

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
    p.add_argument("--list-sessions", action="store_true",
                   help="Print the recording sessions in the file and exit")
    p.add_argument("--session", type=int, default=None, metavar="N",
                   help="Analyse session N (1-based) of a file holding several; "
                        "default is the longest")
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
    p.add_argument("--conductor", default=None,
                   choices=[k for k, _label in conductor_options()],
                   metavar="KEY",
                   help=("Service conductor between the transformer and the "
                         "meter, for the expected-impedance comparison. "
                         "Choices: " + ", ".join(k for k, _l in conductor_options())))
    p.add_argument("--run-length-ft", type=float, default=None,
                   help=("Length of that run in feet, transformer to meter. "
                         "Needed with --conductor; without both, the service "
                         "impedance is measured but not compared."))
    p.add_argument("--shared-secondary", default=None,
                   choices=[k for k, _label in conductor_options()],
                   metavar="KEY",
                   help=("Shared secondary main between the transformer and "
                         "this service's tap, where the service is not a "
                         "dedicated run. Same choices as --conductor."))
    p.add_argument("--shared-secondary-ft", type=float, default=None,
                   help=("Length of the shared secondary in feet, transformer "
                         "to the tap for this service."))
    p.add_argument("--primary-metered", action="store_true",
                   help=("Service is metered on the primary. The transformer "
                         "and secondary are the customer's and downstream of "
                         "the meter, so the expected impedance is the primary "
                         "line, entered with --primary-r1/--primary-x1."))
    p.add_argument("--primary-voltage", type=float, default=None, metavar="VLL",
                   help=("Nominal line-to-line voltage at a primary metering "
                         "point (e.g. 13200). Required with --primary-metered: "
                         "it sets the ANSI C84.1 band, the L-N nominal is "
                         "derived from it, and nothing in the meter file names "
                         "the primary voltage for it to be inferred from."))
    p.add_argument("--primary-r1", type=float, default=None,
                   help="Primary line positive-sequence resistance to the meter (ohms).")
    p.add_argument("--primary-x1", type=float, default=None,
                   help="Primary line positive-sequence reactance to the meter (ohms).")
    p.add_argument("--primary-r0", type=float, default=None,
                   help=("Primary line zero-sequence resistance (ohms). Optional; "
                         "used for triplen harmonics and earth-return unbalance."))
    p.add_argument("--primary-x0", type=float, default=None,
                   help="Primary line zero-sequence reactance (ohms). Optional.")
    p.add_argument("--resample",  default=None,  help="Resample interval, e.g. '1s', '1min', '10min'")
    p.add_argument("--outdir",    default=str(Path(__file__).parent / "pq_output"),
                   help="Output directory (default: pq_output/ next to this script)")
    p.add_argument("--no-plots",  action="store_true", help="Skip plot generation")
    p.add_argument("--report",    action="store_true", help="Generate Word (.docx) report")
    p.add_argument("--site-name",      default=None, help="Site name for the report header")
    p.add_argument("--site-address",   default=None, help="Site address for the report header")
    p.add_argument("--engineer",       default=None, help="Engineer name for the report sign-off")
    p.add_argument("--engineer-title", default=None, help="Engineer title (default: Electric Area Engineer)")
    p.add_argument("--engineer-email", default=None, help="Engineer email address for sign-off")
    p.add_argument("--customer-class", default="sg",
                   choices=["r", "c", "sg", "pg"],
                   help="PSCo tariff schedule: r=Residential, c=Small Comm., sg=C&I Secondary, pg=C&I Primary")
    p.add_argument("--service-role", default="load",
                   choices=["load", "mixed", "generation"],
                   help="Which way power flows at this meter: load (default), mixed "
                        "for a service with generation in parallel behind it "
                        "(Schedules NM, PV, RE, AVPP), or generation for a plant "
                        "with no load worth the name (a producer's array).")
    p.add_argument("--rated-ac-kw", type=float, default=None,
                   help="Combined site rated generation, kW AC. This is I_rated for "
                        "the IEEE 1547 limits and the numerator of the 519-2022 "
                        "Figure 1 test for which standard applies.")
    p.add_argument("--annual-avg-load-kw", type=float, default=None,
                   help="Annual average load demand, kW, from billing history: what "
                        "the site consumes, before its own generation offsets any of "
                        "it — not the net at the meter. Denominator of the 519-2022 "
                        "Figure 1 test; at or above 10%% generation the installation "
                        "goes to IEEE 1547.")
    p.add_argument("--il-amps", type=float, default=None,
                   help="Maximum demand load current from billing: the twelve "
                        "previous months' 15- or 30-min maximum demands, averaged, "
                        "per IEEE 519-2022. Without it the recording's largest "
                        "fundamental stands in and is labelled as doing so.")
    p.add_argument("--verbose",   action="store_true", help="Debug logging")
    args = p.parse_args()
    # Checked here rather than in main() so it fails as a usage error, with the
    # usage text, before any file is opened.
    if args.primary_metered and not args.primary_voltage:
        p.error("--primary-metered requires --primary-voltage VLL: the ANSI "
                "C84.1 band is built from it, and nothing in the meter file "
                "names the primary voltage for it to be inferred from.")
    return args


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
                "TDD will assume the most restrictive class; pass --isc manually "
                "for the true class limit.",
                svc_type, args.transformer_kva, args.nominal,
            )

    isc_source: str | None = None
    if args.isc is not None:
        isc_source = f"Manual (--isc {args.isc:.0f} A)"
    elif isc_amps is not None:
        isc_source = isc_note

    # A primary metering point is described by its L-L nominal; the per-phase
    # ANSI check runs on L-N, so derive it rather than making the caller pass
    # two numbers that have to agree.
    nominal_v = args.nominal
    if args.primary_metered and args.primary_voltage:
        nominal_v = args.primary_voltage / ll_factor(args.service_type, args.topology)
        log.info("Primary-metered: %.0f V L-L -> %.1f V L-N nominal.",
                 args.primary_voltage, nominal_v)

    thresh = Thresholds(
        nominal_voltage=nominal_v,
        primary_ll_voltage=args.primary_voltage,
        volt_tolerance=args.volt_tol,
        thd_voltage_limit=args.thd_limit,
        power_factor_limit=args.pf_limit,
        imbalance_limit=args.imb_limit,
        current_imbalance_limit=args.curr_imb_limit,
        isc_amps=isc_amps,
        isc_source=isc_source,
        transformer_kva=args.transformer_kva,
        customer_class=args.customer_class,
        service_role=args.service_role,
        rated_ac_kw=args.rated_ac_kw,
        annual_avg_load_kw=args.annual_avg_load_kw,
        il_amps_billing=args.il_amps,
        service_type=args.service_type,
        topology=args.topology,
        conductor_key=args.conductor,
        run_length_ft=args.run_length_ft,
        shared_secondary_key=args.shared_secondary,
        shared_secondary_ft=args.shared_secondary_ft,
        primary_metered=args.primary_metered,
        primary_r1_ohm=args.primary_r1,
        primary_x1_ohm=args.primary_x1,
        primary_r0_ohm=args.primary_r0,
        primary_x0_ohm=args.primary_x0,
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
            # --session is 1-based for the user, 0-based inside.
            adapter = ProntoAdapter(
                fp, session=None if args.session is None else args.session - 1)
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

    # ── List-sessions mode ────────────────────────────────────────────────────
    if args.list_sessions:
        sessions = getattr(adapter, "sessions", []) or []
        if not sessions:
            print("\nThis file holds one recording session.\n")
            return
        current = getattr(adapter, "session_index", 0)
        print(f"\n{len(sessions)} recording sessions:\n")
        for s in sessions:
            mark = "→" if s["index"] == current else " "
            print(f"  {mark} --session {s['index'] + 1}   "
                  f"{(s['start_time'] or '')[:16].replace('T', ' ')} → "
                  f"{(s['end_time'] or '')[:16].replace('T', ' ')}   "
                  f"{s['hours']:.1f} h, {s['intervals']} intervals")
        print("\n(→ marks the one analysed by default: the longest.)\n")
        return

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
    source_harm_result   = check_harmonic_sources(df, thresh)
    spectral_shape_result = check_spectral_shape(df, thresh, source_harm_result)
    direction_result      = check_harmonic_direction(ds, thresh)
    impedance_result      = check_source_impedance(df, thresh)
    stat_result         = check_harmonic_statistics(df, thresh)
    event_result        = detect_events(ds, thresh)
    neutral_health_result = check_neutral_health(ds, thresh)
    itic_result         = check_itic(event_result, thresh)
    ll_volt_result      = check_line_to_line_voltage(df, thresh)
    frequency_result    = check_frequency(df, thresh)
    flicker_result      = check_flicker(df, thresh)
    kfactor_result      = kfactor_by_phase(df)

    # ── Compile report ────────────────────────────────────────────────────────
    report = generate_report(
        ds, volt_result, thd_result, pf_result,
        imb_result, curr_imb_result, demand_result,
        harm_result, volt_harm_result, neutral_harm_result,
        source_harm_result, stat_result, event_result, thresh,
        neutral_health_result=neutral_health_result,
        spectral_shape_result=spectral_shape_result,
        direction_result=direction_result,
        impedance_result=impedance_result,
        itic_result=itic_result,
        ll_volt_result=ll_volt_result,
        frequency_result=frequency_result,
        flicker_result=flicker_result,
        kfactor_result=kfactor_result,
    )
    report["root_causes"] = analyze_root_causes(report, ds, thresh)

    print_report(report)

    # ── Export ────────────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    export_results(ds, report, outdir, stem=stem)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        log.info("Generating plots …")
        plot_overview(ds, thresh, outdir=outdir, stem=stem)
        plot_voltage(df, volt_result, thresh, outdir=outdir, stem=stem)
        plot_thd(df, thd_result, thresh, outdir=outdir, stem=stem)
        plot_summary(df, imb_result, outdir=outdir, stem=stem)
        plot_harmonic_spectrum(df, thresh, outdir=outdir, stem=stem)
        plot_itic(event_result["events"], thresh, outdir=outdir, stem=stem)
        plot_neutral_health(ds, neutral_health_result, thresh, outdir=outdir, stem=stem)
        plot_demand_profile(df, thd_result, outdir=outdir, stem=stem)
        plot_harmonic_trend(df, outdir=outdir, stem=stem)
        plot_imbalance(df, imb_result, curr_imb_result, outdir=outdir, stem=stem)
        plot_pf_load(df, pf_result, outdir=outdir, stem=stem)
        plot_flicker(df, flicker_result, outdir=outdir, stem=stem)
        plot_waveform_capture(ds, thresh, outdir=outdir, stem=stem)
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
            outdir=outdir,
            stem=stem,
            engineer_title=args.engineer_title or "",
            engineer_email=args.engineer_email or "",
        )
        # Second, separate document: the plain-language letter for the customer.
        # Residential only; other service classes get the engineering report alone.
        generate_customer_letter(
            report=report,
            thresh=thresh,
            site_address=args.site_address or args.site_name or stem,
            engineer_name=args.engineer or "",
            outdir=outdir,
            stem=stem,
            engineer_title=args.engineer_title or "",
            engineer_email=args.engineer_email or "",
        )

    log.info("Done.")


if __name__ == "__main__":
    main()

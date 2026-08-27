# PQ Analyzer — Dev Notes

Production tool used by Xcel Energy / PSCo field engineers. Reports go to
customers and, in the regulatory workflow, to state commissions. A wrong number
in a report is a real-world problem, not a failed test. Prefer refusing to state
a value over stating one that wasn't measured.

`README.md` is written for **users** (install, run, troubleshoot). This file is
for whoever is changing the code. Don't duplicate the README here.

## Module map

The engine is deliberately split; `pq_analyzer.py` holds only `parse_args()` and
`main()`. Sizes matter here — several of these files are far too large to read
whole, so grep to the section you need.

| File | Lines | Holds |
|---|---:|---|
| `pqdif.py` | 1,341 | IEEE 1159.3-2019 PQDIF reader. Pure format, no analysis. |
| `pq_adapter.py` | 3,235 | Channel mapping, adapters, `PQDataset`. Meter quirks live here. |
| `pq_constants.py` | 2,463 | `Thresholds`, IEEE 519 tables, ANSI C84.1 bands, Blue Book ISC lookup, `__version__` |
| `pq_analysis.py` | 6,430 | Compliance checks, harmonic detection, attribution, root cause |
| `pq_report.py` | 7,864 | Word report + customer letter + CSV export |
| `pq_plots.py` | 1,255 | Matplotlib (Agg backend, never interactive) |
| `run.py` | 3,722 | tkinter GUI launcher — the thing the shortcut opens |
| `test_pq.py` | 9,448 | 880 tests. See below. |

Dependency direction is one-way: `pqdif` → `pq_adapter` → `pq_analysis` →
`pq_report`/`pq_plots`, with `pq_constants` imported by everything. Keep it that
way; no analysis in the adapter, no thresholds outside `pq_constants`.

## Running things

```bash
pytest test_pq.py -v                  # full suite, 880 tests
pytest test_pq.py -k Harmonic -v      # one area
python pq_analyzer.py <file.pqd>      # CLI
python run.py                         # GUI
python check_install.py               # diagnoses a broken install
python check_record.py <file.pqd>     # short file vs. bodyless record
```

Run the suite before every commit. It is the only thing standing between a
refactor and a wrong number in a customer report.

## Test fixtures

`test_data/` holds nine `.pqd` fixtures covering the service populations that
behave differently: residential, small/large commercial, primary-metered,
imbalanced, net-metered solar, producer array, ProView per-phase, two-session.
`make_test_pqd.py` (1,719 lines) generates them.

`TestFixturesAreCompliant` and `TestBothDocumentsBuildForEveryFixture` run
across all of them — a change that only works on one service type fails there.
When you add a service population the analyzer treats differently, add a
fixture; don't special-case it in the analysis.

## Standards are the spec

Checks cite specific clauses, and the citation is load-bearing — engineers read
it and look it up. If you change a limit, change the citation with it, and don't
invent one.

- **IEEE 519-2022** — Table 1/2 per-order limits, TDD via `_h519_limit` /
  `_tdd_limit` / `_tdd_class`. TDD needs ISC; without it the tool falls back to
  a THD check and must say so.
- **ANSI C84.1-2016** — Range A/B on *interval averages*, both voltage groups.
  Within-interval extremes are graded as events against ITIC, not against C84.1.
- **IEEE 1547** — Clause 6.4.2 voltage and 6.5.2 frequency ride-through.
- **IEEE 1159** — event detection. **IEEE 1453-2022** — Pst/Plt.
- **NEMA MG1** — voltage unbalance. Leg difference is *not* graded against NEMA
  (`TestLegDifferenceIsNotGradedAgainstNEMA` enforces this).
- **PSCo Tariff / Blue Book** — power factor, current imbalance. Jurisdiction
  matters: `jurisdiction_gap` and `TestJurisdictionFailsClosed` mean a
  state-specific finding must be flagged as such, and an unknown jurisdiction
  fails closed rather than assuming Colorado.

A producer is judged against its **interconnection agreement**, a load against
the **tariff**. Don't let a billing clause leak into a producer's letter
(`TestTariffScopingIsNotMisstated`, `TestTheLetterToAProducer`).

## Invariants worth knowing before you edit

- **Say what wasn't read.** Established by "Say what the analysis did not read,
  and never pass on data nobody measured" (827a1a0). If a channel is absent, the
  report says the check couldn't run — it does not silently omit the section and
  it does not infer the value. `TestChannelCoverageIsVisible`,
  `TestUnavailableShapes`. The same rule governs the unmapped-channel table:
  it names every channel in a group rather than counting them, because one
  PQDIF triple can cover both a subtotal nobody needs and an aggregate the
  tool would otherwise read.
- **Severity is graded on the quantity the verdict came from** — not a
  correlated one. Needs both margin *and* persistence
  (`TestSeverityNeedsMarginAndPersistence`).
- **Power factor is graded on magnitude**, with the operating quadrant reported
  separately. Sign convention is tested (`TestPowerFactorSignConvention`) and is
  easy to get backwards on a net-metered service.
- **Primary nominal comes from the engineer**, never inferred from the measured
  L-L/L-N ratio — the ratio recovers topology, not nominal. See the comment on
  `Thresholds.primary_ll_voltage`.
- **Timestamps** — `TestTimestampEpoch` exists because PQDIF epochs are a
  recurring source of off-by-years bugs.
- `pq_plots.py` forces the Agg backend. Never introduce an interactive backend;
  the GUI runs analysis on a worker thread.

## Meter files

The reader is written from IEEE 1159.3-2019 by traversal, never by
reverse-engineered byte offsets (see the `pqdif.py` docstring). If a new vendor
file fails to load, fix the traversal or the channel mapping in `pq_adapter.py`
— do not add an offset hack. Verified against two Pronto firmware generations.

## House style

Commit subjects are declarative statements of the outcome, lowercase after the
first word, no `feat:`/`fix:` prefixes:

> Say what the analysis did not read, and never pass on data nobody measured
> Grade severity on the quantity the verdict was reached on
> Route harmonic limits through the standard that actually applies

Report prose is written *to a customer*: plain, specific, no hedging, no
apology. Match the surrounding tone before adding a new finding string.

## Distribution

Field engineers run this on Xcel-managed Windows PCs and update with `git pull`.
That means:

- `pq_output/` is gitignored and must never be touched by an update.
- Don't add a dependency casually — pip through the corporate proxy is a real
  obstacle for users. `pq_analyzer_requirements.txt` is the contract.
- `rapidfuzz` and `pqdifpy` are optional; keep the stdlib fallbacks working.
- The maintainer publishes; users never push.

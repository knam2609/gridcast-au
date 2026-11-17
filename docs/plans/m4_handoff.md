# M4 handoff: start here

Read this, then [`m4_plan.md`](m4_plan.md). For the M3 history, see [`m3_handoff.md`](m3_handoff.md) and [`m3_plan.md`](m3_plan.md).

> **Update, 2026-11-17 (seventeenth session, later: step 2b and the warm-up smoke run):**
> - **New:**
>   - `anomalies/inputs.py`: the database series and the QR band with run 2's common mask, checked against run 2's issue times and actuals;
>   - `anomalies/injection.py`: §2.9's harness, with shared starts;
>   - `anomalies/smoke.py` and `make anomalies-smoke`.
>   - Tests: the harness equals a full recompute where §2.9's unmodified context applies; eligibility; an integration test of the loader. The unit suite has 384 tests.
> - **Smoke run** (warm-up only, 18 s; details and numbers in [`m4_plan.md`](m4_plan.md), step 2 outcome).
>   - **No coding bug was found.** The top D1 and D4 scores are real data (New Year's Eve 2021 evening demand; negative prices at Sunday dawn).
>   - D6's weakness is a property of Isolation Forests: a value beyond the training range behaves like the training maximum.
> - **It previews likely pre-registered failures.** Power at the largest sizes: D1 55–58%, D3 76–86%, D4 28% ($15,000) and 52% (−$1,000), D6 1–2%. D2 and D5 reached 100%. D1 alerted at p ≤ 0.01 on 0.14% of intervals.
> - **Open question for the user (asked 2026-11-17):** keep the pre-registered recipe and publish any failures, or revise the weak detectors before the freeze? Ask again if not answered.
>
> **Next:** the user's answer, then step 3 (evidence, summaries and the SHAP core).

> **Update, 2026-11-17 (seventeenth session: step 2a, the detector core):**
> - **Question round, both recommendations chosen** (in [`m4_plan.md`](m4_plan.md) §1, "Clarified in step 2"):
>   - the injection starts are shared: 1,000 per detector, reused for every size and sign;
>   - D6's three z-features use plain profiles (all day types), as §2.3's generic definition.
> - The same §1 block records the readings fixed while coding: the window is [t − 26 weeks, t), events never cross the split boundary, the peak tie rule, and others.
> - Every §2 setting is in `configs/base.yaml` → `anomalies` (typed in `AnomaliesConfig`).
> - New in `src/gridcast/anomalies/`: `pvalues.py`, `series.py`, `profiles.py`, `detectors.py`, `isolation.py`, `events.py` and `run.py`. `run.detect(series, band, split, cfg)` returns scores, p-values, tiers and events.
> - Tests are synthetic, with no real data scored: `tests/anomaly_fixtures.py` and `tests/unit/test_anomaly_{pvalues,profiles,detectors,events,causality}.py`. A5's property tests and the leaky control pass, about 16 s. The full unit suite has 370 tests.
> - A full synthetic pass over 13k intervals takes about 0.9 s. The Fenwick sweep costs about 25 µs per interval, so a 525k-interval run should take minutes.
>
> **Next: step 2b.**
> - `anomalies/inputs.py`: load `MarketSeries` from `market_intervals`, `generation_intervals` and `calendar_days` (business day = not a weekend or VIC public holiday, by the local date of interval start).
> - The `QuantileBand` from the QR batch's `q025`/`q975`, with run 2's common mask. Check that the stored actuals equal market demand.
> - `anomalies/injection.py` (§2.9; shared starts).
> - The warm-up smoke run: market data truncated at 2022-04-30, residual at 2023-05-31. Bugs only.

> **Update, 2026-11-16 (sixteenth session: `make dq` fixed):**
> - `implausible_timestamps` no longer flags days after `ingestion.end_month` that a successful `nemweb_daily` run loaded. Grid, duration and future checks still apply to them.
> - A new warning, `daily_report_days`, lists those days, one row per NEM day (now 1 Oct 2026, 288 intervals), so they stay visible.
> - `make dq` exits 0 again. Tests: `tests/integration/test_dq_checks.py` (exempt when daily and successful; flagged when from MMSDM or from a failed daily run).
>
> **Update, 2026-11-15 (fifteenth session: step 1b done):**
> - **Question round, both recommendations chosen** (recorded in [`m4_plan.md`](m4_plan.md) §1, "Clarified in step 1b"):
>   - D6's curtailment share is 1 − cleared / **UIGF**, because `SS_*_AVAILABILITY` exists only from May 2024 (table v8);
>   - battery evidence is unavailable before the first non-zero bidirectional-unit interval (6 Aug 2024), and later values are labelled "bidirectional units only".
> - **Migration 0007** is applied to the main database (`generation_intervals`, `anomalies`, `model_runs` kind `anomaly` / target `market`).
> - **`gridcast ingest-generation`** (`make ingest-generation`) loaded 60 months plus 1 Oct 2026 for VIC1:
>   - 526,176 rows each for wind, solar and dispatchable, and 248,403 for battery;
>   - no downloads;
>   - September 2026 from the daily reports matched MMSDM exactly first.
>   - The log is `<data root>/logs/m4_ingest_generation.log`.
> - **Data-quality checks:** `generation_missing` and `generation_negative` pass. `generation_above_uigf` warns: 707 wind/solar intervals, most in Dec 2022, max 293.7 MW. Details are in [`docs/data_sources.md`](../data_sources.md) §8.
> - **`make dq` exited 1, but not because of this step.** `implausible_timestamps` flagged the 288 market intervals of 1 Oct 2026, which M2 step 6 loaded on purpose but which lie outside `ingestion.end_month`. It already failed on 2026-11-14. Fixed on 2026-11-16 (see the update above).
> - New code:
>   - `src/gridcast/ingestion/generation.py`;
>   - the reader's `optional` columns in `mms_csv.py`;
>   - `upsert_generation`;
>   - three SQL checks (`GENERATION_CHECKS`);
>   - unit tests and `tests/integration/test_generation.py`.
>   - Still no detector code.
>
> **Next: step 2 (detector core).** `src/gridcast/anomalies/` should contain:
> - window p-values on M3's Fenwick structure;
> - D1–D6 and the lines;
> - the event merge rule;
> - the injection harness;
> - the §2.8 property tests and the leaky control.
>
> The smoke run may use only the warm-up periods and synthetic data. D6 reads wind and solar from `generation_intervals` (`uigf_mw`), and dispatchable generation from the same table.

> **Update, 2026-11-15 (fourteenth session, later: step 1a done):**
> - The `APCFLAG` bit meanings were read on Global-Roam's mirror of the MMS Data Model, because AEMO's own pages returned 404. They match the plan's assumption, so the classes in §3.1 stand.
> - [`docs/m4_reference_events.md`](../m4_reference_events.md) is generated by `gridcast anomaly-reference-events` (`make anomaly-reference-events`).
>   - It lists 17 events. **A2 judges 7: P03–P09**, exactly the ones the plan expected.
>   - The holdout has only O08 (23 Mar 2026).
> - New code: `src/gridcast/anomalies/reference.py`, the `anomalies` config section (periods and reference rules), and unit tests. No detector code yet.
>
> **Next (done, see above): step 1b.**
> - Migration 0007: `generation_intervals`, the `anomalies` table, and `model_runs` accepting kind `anomaly` and target `market`.
> - `gridcast ingest-generation`, re-parsing the cached `DISPATCHREGIONSUM` zips and the 1 Oct 2026 daily reports.
> - Data-quality checks.
> - Column definitions for `docs/data_sources.md` (from the mirror's `DISPATCHREGIONSUM` page):
>   - `SS_WIND/SOLAR_CLEAREDMW`: semi-scheduled cleared MW;
>   - `SS_*_UIGF`: unconstrained intermittent generation forecast;
>   - `SS_*_AVAILABILITY`: aggregate energy availability;
>   - `DISPATCHABLEGENERATION`: "Dispatched Generation";
>   - `AVAILABLEGENERATION`: "Aggregate generation bid available in region";
>   - `BDU_CLEAREDMW_GEN/LOAD`: battery cleared MW, generation and load;
>   - `TOTALINTERMITTENTGENERATION`: an allowance for non-scheduled generation in AEMO's demand forecast, **not** an output (do not use it as wind or solar).

> **Update, 2026-11-15 (fourteenth session: M4 planned):**
> - M3 was merged into `main` on 2026-11-14 (PR #3 → `c5fd5d3`). Branch `m4-anomalies-explainability` is from `main` at `c5fd5d3`.
> - **Three question rounds; the user chose every recommendation** ([`m4_plan.md`](m4_plan.md) §1):
>   - scope: anomaly detection plus SHAP for LightGBM C (price residuals wait for M2b);
>   - causal scoring;
>   - a pre-registered three-part check: alert budget, AEMO-flagged reference events, synthetic injection;
>   - evidence from the cached `DISPATCHREGIONSUM` region totals (no new downloads);
>   - calibrated p-value tiers (notable 0.001, severe 0.0001, trailing 26 weeks) plus fixed price lines at $300 and $5,000;
>   - LightGBM's built-in TreeSHAP over a seeded sample;
>   - descriptive price associations for question 5;
>   - freeze, then score the holdout once;
>   - bars: factor 2 on alert rates, every class-P event caught, ≥ 90% power at each detector's largest injection;
>   - residual detectors on M3's raw QR band with their own p-value.
> - **Added by the plan, accepted by the user on review:** every detector setting is fixed in §2 now, with no tuning spike. The recipe is frozen before the development run, so the development verdicts are also blind.
> - **Disclosed in §3.3:** holdout facts seen during the question rounds (the price calm, 8 Jul 2026 above $5,000, the 6 and 10 Nov 2025 demand jumps, the flagged 23 Mar 2026).
>
> **Next (done, see above): step 1a.**
> 1. Start Docker (`make db-up`) and check that the SSD is mounted.
> 2. Check the `APCFLAG` and `MARKETSUSPENDEDFLAG` bit meanings in the MMS Data Model (visualisations.aemo.com.au).
> 3. Write `gridcast anomaly-reference-events` and commit the generated `docs/m4_reference_events.md` **before any detector code exists**.
> 4. Then 1b: migration 0007 and `gridcast ingest-generation`.

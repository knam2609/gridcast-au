# M4 handoff: start here

Read this, then [`m4_plan.md`](m4_plan.md). For the M3 history, see [`m3_handoff.md`](m3_handoff.md) and [`m3_plan.md`](m3_plan.md).

> **Update, 2026-11-20 (twenty-first session: the step-5 question round):** the user chose every recommendation. The decisions are recorded in [`m4_plan.md`](m4_plan.md) §1, "Clarified in step 5", before any global SHAP value or association rate existed:
> - **Global SHAP decomposes the whole forecast.** Run 2's fits mix target forms (6–24 h: 5 direct, 4 residual, 2 anchored, 1 latest). Each fit's anchor goes to its own features as x − x̄ (an exact linear Shapley term), so fits of any form can be pooled. The raw tree-only values are shown by form beside the headline.
> - **Question 5:** the top 5% is taken within each group (overall, summer, winter).
> - **Price associations:** the market development period (2022-05-01 → 2025-09-30). The bin edges come from development data and step 6 reuses them; solar gets a zero-output bin plus deciles of positive output.
> - Step 5's report settings go in a new `explain_report` config section, so neither anomaly freeze changes.
>
> **Next: 5a** (`gridcast explain-dev`), then **5b** (`gridcast report-m4-explain` → `reports/m4_explainability.md`).

> **Update, 2026-11-20 (twentieth session, later: steps 4f–4h, v2 built, frozen and scored on development):**
> - **4f `38e4687`:** the frozen modules are generalised behind `detectors.Recipe`; version 1 is the default everywhere. `gridcast anomalies-reproduce` (`anomalies/reproduce.py`, not frozen) recomputes a stored development run and compares every column, injection, month-model range, cut, event (evidence and summary included) and the run record. It passed on the untouched code first, and then on the generalised code.
> - **4g `04da607` (code) + `17ca965` (freeze):**
>   - `anomalies/distance.py` (frozen), the `anomalies_v2` config section, and `recipe_for` / `recipe_config_hash`;
>   - a `--version` option on `anomalies-smoke`, `evidence-smoke`, `anomalies-freeze-check`, `anomalies-dev`, `anomalies-reproduce` and `report-m4` (Make: `VERSION=2`);
>   - the causality suite runs both versions, plus `tests/unit/test_anomaly_v2.py`;
>   - the warm-up smoke runs under v2 found no bug.
>   - From a clean tree at `04da607`, run 104 reproduced exactly. **v1:** hash `2ce8b3fc523ae81c`, code carried from `2dc58fa` to `04da607`. **v2:** hash `44abbf4b24cf6ac8`, code `04da607`.
> - **4h:**
>   - The first v2 development run stored nothing: migration 0007's detector list rejected `robust_distance`. **Migration 0008** (`7b989af`) fixed that, and it is applied to the main database. What was seen of the failed attempt is disclosed in the plan.
>   - The rerun is **run 106, batch `0ff972b0-…`**. `report-m4 --version 2` wrote [`reports/m4_anomalies_v2.md`](../../reports/m4_anomalies_v2.md) (`d5f9c53`).
> - **v2 verdicts (post hoc): A1, A2, A4, A5 and E1 pass; A3 fails.**
>   - D6 reaches 100% at ±2,000 MW, and A1 is 12 of 12. P05 is caught.
>   - D4 reaches 84.6% at $15,000 and 66.2% at −$1,000, still below 90%. D1 and D3 are unchanged.
>   - v1's pre-registered 3 of 6 stands as published.
> - **Next: step 5** (`explain-dev`, `report-m4-explain`; independent of the version). Then **step 6**: the holdout once per version (each run must first reproduce its own development batch: run 104 for v1, run 106 for v2). M5/M6 serve v2. Run a question round for step 5 and for step 6 where the plan leaves choices open.

> **Update, 2026-11-20 (twentieth session: the v2 question round, step 4e):** before the question round, a read-only diagnosis of the stored development run found each failure's mechanism. It is [`docs/m4_v1_diagnosis.md`](../m4_v1_diagnosis.md), from `gridcast anomalies-diagnose-v1` (`anomalies/diagnosis.py`, not frozen). It recomputes v1's own state, checks it bit for bit against run 104, and scores no alternative:
> - **D4:** asinh(price / $10) makes a −$50 → +$50 move as large as a 95-fold spike, so the per-slot robust scale decides the ranking. 54.8% of alerts sit at the floor scale; power at $15,000 is 36.8% at the tightest-scale starts and 0.4% at the widest.
> - **D6:** the Isolation Forest saturates (0 of 41 monthly models let a one-feature extreme reach p ≤ 0.001), and its training-window calibration drifts (11 of 41 months above 0.2%, 19 with no alert).
> - **D1** follows the real spread of demand (78–84% power at stable slots, 0.8% at the widest), so it is not a defect. **D3** was not examined.
>
> **The user chose every recommendation** (recorded in [`m4_plan.md`](m4_plan.md) §1, "Decided after the development verdicts", and §2.10):
> - a disclosed **v2 for D4 and D6 only**; v1's verdicts stay published;
> - **D4 v2** uses x = asinh(price / $100), with everything else as v1. D5 keeps asinh(price / $10);
> - **D6 v2** is a **robust distance**: the same nine features (D4's z from v2) and monthly refit, distance from the training median under a Ledoit–Wolf covariance, and a rolling p-value against its own out-of-sample scores;
> - **the holdout scores v1 and v2, once each**, and M5/M6 serve v2 whatever the holdout shows;
> - v2's development verdicts are labelled **post hoc**.
>
> **How (readings fixed before coding):** the frozen modules are generalised behind a recipe version rather than copied. v1's code freeze is then carried by behaviour: `gridcast anomalies-reproduce` must reproduce run 104 exactly (every column, injection, forest range, cut, event, evidence field and summary) before v2 changes anything. The v1 holdout repeats that check. v1's `anomalies` config and hash do not change; v2's settings go in a new `anomalies_v2` section with its own freeze. v2's D6 is stored as `robust_distance`. The v2 freeze and development run come before any holdout score.
>
> **Next: step 4f** (generalise, then reproduce run 104), **4g** (v2 code + freeze), **4h** (v2's development run + `reports/m4_anomalies_v2.md`). Then step 5 (explanations), then step 6 (holdout, v1 and v2).

> **Update, 2026-11-19 (nineteenth session: step 4 question round):** the user chose every recommendation. The decisions are in [`m4_plan.md`](m4_plan.md) §1, "Clarified in step 4", recorded before any freeze code or development score:
> - **Freeze:** the hash covers the whole `anomalies` section plus the split dates. `FROZEN_PATHS` = M3's frozen paths, the scoring, merging, injection and evidence modules of `anomalies/`, and `explainability/` without its smoke module.
> - **Holdout band:** split by issue time (development QR + run 2's mask before 2025-10-01, holdout QR + `holdout_main`'s mask after). The Aug–Sep 2025 overlap is bit-identical (checked today), and the holdout run must refuse if it is not.
> - **Case-study ties:** lowest p, then more calibrated detectors, then a seeded draw; the named events come on top.
> - **A5:** the synthetic suite plus 3 seeded real-data cuts in the development run.
> - **Storage:** one `model_runs` row per run, parquet under `<data root>/anomalies/<batch>/`, events in `anomalies`.
>
> **Step 4a done `2dc58fa`:** `anomalies/recipe.py` (frozen: hash, splits, case rule, cuts, forest check), `anomalies/freeze.py`, `inputs.split_band` (with `check_qr_overlap` and `concat_bands`), `anomalies/dev.py`, and `gridcast anomalies-freeze-check` / `anomalies-dev` (Make targets of the same names). It adds 13 unit tests (429 in the suite) and 3 integration tests. Before the freeze, the warm-up smoke runs reproduced step 3's results (33 + 7 events, all complete, E1 passes).
>
> **Step 4b, the freeze:** `anomalies.frozen_config_hash` = `2ce8b3fc523ae81c`, `anomalies.frozen_commit` = `2dc58fa`, pinned by `tests/unit/test_anomaly_freeze.py`. No development score existed when it was recorded.
>
> **Development run done:** run 104, batch `40cbe0f5-…`, from `4990ebf`, about 3 min. **`report-m4`** (`74fc3a2`, `b0a5c5b`) recomputed the stored run with no mismatch. **A4, A5 and E1 (demand peaks) pass; A1, A2 and A3 fail** ([`reports/m4_anomalies.md`](../../reports/m4_anomalies.md); numbers in [`m4_plan.md`](m4_plan.md), step 4 outcome):
> - A1: D6 flags 0.259% of intervals at p ≤ 0.001 (bound 0.2%); the other 11 checks are inside.
> - A2: P05 (5 Aug 2024, $17,500 for 10 min) missed; D4's lowest p near it was 1.48 × 10⁻².
> - A3: only D2 (±) and D5 reach 90%; D1, D3, D4 and D6 fail, as the smoke run previewed.
>
> **Next: a question round.** Should a disclosed, re-hashed v2 follow for the failing detectors (plan §6), and with what changes? Or should v1 stand as published? Then step 5 (`explain-dev`, `report-m4-explain`), which does not depend on the anomaly verdicts. A v2 must not touch the holdout until its own development run is done.

> **Update, 2026-11-19 (eighteenth session: step 3, evidence, summaries and the SHAP core):** in the question round the user chose every recommendation. Recorded in [`m4_plan.md`](m4_plan.md) §1, "Clarified in step 3", before any evidence or SHAP code:
> - a demand event's SHAP explains its demand peak (the lowest-p D1–D3 alert);
> - summaries have three sentences;
> - E1's forecast check is an exact float32 match (M2 stored float32);
> - the smoke runs cover the warm-up events' evidence and SHAP on retrain 2022-11-01, and print only completeness, banned-phrase counts, E1 results and timings.
>
> - **Step 3a done:**
>   - `src/gridcast/explainability/` (families, stored fits, TreeSHAP with E1's gate, the seeded sample, pair explanations);
>   - `anomalies.explain` in the config;
>   - `make explain-smoke`;
>   - 13 new unit tests (397 in the suite).
> - **SHAP smoke run** (retrain 2022-11-01, 80,000 pairs, 37 s): E1 passes on every pair, with no float32 mismatch and a largest SHAP-sum error of 2.5 × 10⁻¹¹. The full global samples should take about 11–17 min (development) and 5–7 min (holdout), depending on machine load. Details are in [`m4_plan.md`](m4_plan.md), step 3 outcome.
>
> - **Step 3b done:**
>   - `anomalies/evidence.py` (47 labelled fields, three-sentence summaries, A4's checks);
>   - `anomalies/evidence_inputs.py` (loads only what the events need; SHAP behind E1);
>   - `profiles.SlotValues`;
>   - `anomalies.evidence` in the config;
>   - `make evidence-smoke`;
>   - folds clipped to each batch's period;
>   - 19 new unit tests (416 in the suite) and 2 integration tests.
> - **Evidence smoke run** (20 s): all 40 warm-up events are complete, no summary has a banned phrase, and E1 passes on all 14 demand-peak pairs. Three wording bugs were found in the examples and fixed. Details are in [`m4_plan.md`](m4_plan.md), step 3 outcome.
>
> **Next: step 4** (freeze, the development run, `report-m4`). Run a question round first. Open points:
> - **What the freeze covers.** The candidates are `FROZEN_PATHS` = M2's frozen paths, every `anomalies/` module that scores, merges or describes (the injection harness and the smoke modules too?), and `explainability/`. The hash should cover the whole `anomalies` config section, including `evidence` and `explain`.
> - **The holdout band for D2/D3.** `inputs.read_band` needs the QR batch and the M2 batch to have the same issue times. The holdout QR batch (`b226706b`) starts at 2025-08-01, but `holdout_main` starts at 2025-10-01. The one continuous holdout run therefore needs a combined band: the development QR batch with run 2's mask before 2025-10-01, and the holdout QR batch with `holdout_main`'s mask from then on. `evidence_inputs.point_sources` already combines its forecasts this way. The combined band must be fixed before the freeze.
> - **Storage of the development batch.** Per-interval scores and p-values go to parquet, events to `anomalies`, plus `model_runs` rows and the injection results. The layout is still to be decided.

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
> - **User decision (2026-11-19): keep the pre-registered recipe.** No setting changes. Any A1/A3 failure in development is published. A disclosed v2 with a new hash may follow the development verdicts (plan §6).
>
> **Next: step 3** (evidence, summaries and the SHAP core). Run a question round first where the plan leaves choices open.

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

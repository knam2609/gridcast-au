# Milestone 2 plan: weather-driven demand ML

**Goal:** find out whether machine learning with honestly available weather forecasts beats M1's best baseline at each lead time for 24-hour-ahead Victorian demand, and how it compares with AEMO's own forecasts.

**Status:** step 0 done ([`docs/m2_data_spike.md`](../m2_data_spike.md), 2026-11-02); step 1 done (2026-11-03, weather loaded and checked, see [`docs/data_sources.md`](../data_sources.md) §4–5); step 2 done (2026-11-04, feature builder, leakage tests, [`docs/feature_catalogue.md`](../feature_catalogue.md) and [`reports/m2_feature_audit.md`](../../reports/m2_feature_audit.md)); step 3 done (2026-11-06, full walk-forward run stored; outcome under §4 step 3). Success criteria were agreed before any ML results existed (see §3). The step-0 decisions below refine *how* C, D and the benchmark are built; they do not change S1–S5.

## 1. Decisions (agreed 2026-11-02)

| Topic | Decision |
|---|---|
| Target | Victorian demand (`TOTALDEMAND`) only. Price ML is **M2b** and will use the M2 demand forecast as a feature |
| Weather variants | **A** no weather · **C** archived weather forecasts (headline, deployable) · **D** actual weather, ERA5 reanalysis (perfect-foresight upper bound). After step 0: **C = GFS temperature only** (full window), **C⁺** = all GFS variables (from 2024-01-19, secondary); **D mirrors C** and **D⁺ mirrors C⁺** |
| Weather sites | ~5 population-weighted Victorian sites (Melbourne, Geelong, Ballarat, Bendigo, Latrobe Valley) |
| Weather source | Open-Meteo: Previous Runs API, model `gfs_seamless` (C, C⁺), Historical Weather API, model `era5` (D, D⁺). Free tier (non-commercial), CC BY 4.0. JMA was checked and rejected (less accurate, interpolated timing) |
| Forecast availability | `available_at = run_init + 6 h`, with `run_init = floor₆ₕ(t − 24K h)` (verified exactly in step 0) |
| GFS gap 2023-12-30 → 2024-01-20 | C drops training rows without weather; pairs without C weather are excluded from scoring for **every** model and baseline, and the count is reported |
| Models | Ridge regression (linear check) and LightGBM, both trained per variant |
| AEMO forecasts | **Benchmark only**, never a model input: pre-dispatch (30-min, every run) and 5-minute pre-dispatch (included after step 0: 1.25 GB) |
| Benchmark window | Oct 2024 → Sep 2026 (1.22 GB pre-dispatch + 1.25 GB 5-minute pre-dispatch, zipped) |
| Holdout | Oct 2025 → Sep 2026, scored **once**, after the model is frozen |

## 2. Design

### 2.1 Weather data

- **Sites and weights:** site coordinates in `configs/base.yaml`. Population weights come from ABS estimated resident population for each urban area (CC BY 4.0), with the source and year recorded in the config. The features are weights-applied averages, plus Melbourne's values on their own.
- **Variables (hourly):** `temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `cloud_cover`, `shortwave_radiation` (sunlight; important because `TOTALDEMAND` excludes rooftop PV), `wind_speed_10m` (kept for M2b price). Archived GFS forecasts of everything except temperature start 2024-01-19, so headline C/D use **temperature only**; C⁺/D⁺ use all variables.
- **Hourly → 5-minute:** values measured at an instant (temperature, humidity, cloud) are interpolated linearly between hours. `shortwave_radiation` is a preceding-hour mean, so the value labelled hour H applies to the twelve 5-minute intervals ending in (H−1h, H]. An interpolated value is available only when **both** hourly values it uses are available.
- **When an archived forecast became available:** step 0 showed that GFS `previous_day{k}` at hour *t* comes exactly from the 6-hourly run initialised at `run_init = floor₆ₕ(t − 24k h)`. With `publication_delay = 6 h` (Open-Meteo published a GFS run after 5.4–5.5 h), `available_at = run_init + 6 h`. A target uses the **freshest offset with `available_at ≤ o`**, falling back to the older offset if that value is missing; day1 is usable for 84.4% of (issue time, lead) pairs and day2 always. The feature builder only reads rows with `available_at ≤ o`.
- **Storage:** migration `0003_weather`. Tables `weather_sites` (lat/lon, weight, source), `weather_observations` (variant D: site, valid_time, variables, source) and `weather_forecasts` (variant C: site, valid_time, model, offset_days, run_init, available_at, variables). New data-quality checks cover gaps, ranges, the forecast offsets present, and staleness.

### 2.2 Forecast setup (unchanged from M1)

Forecasts are issued every 30 minutes for horizons h = 1…288 five-minute steps. The information set is everything ending at or before the issue time *o*. At the target time *t*, only calendar features and weather (forecast for C, actual for D) are used.

### 2.3 Features (variant A base set; C and D add weather)

| Group | Features | Available at *o* because |
|---|---|---|
| Lead time | lead time in minutes | design |
| Calendar at *t* | 5-min slot of day, day of week, month, day type, public holiday, DST-change day, time-of-day sin/cos | known in advance |
| Calendar at *t* − 24 h | yesterday's day type / holiday flag (yesterday's value may be unusual) | known in advance |
| Demand at the issue time | y(o); y(o) − y(o−288) (today vs yesterday); y(o) − y(o−12) (last-hour change); rolling mean/std over the last 1 h and 24 h | all end at or before *o* |
| Same time yesterday / last week | y(t−288), y(t−2016), y(t−288) − y(t−2016) | t−288 ≤ o for all h ≤ 288 |
| Weather at *t* (C/D) | weighted temperature; cooling/heating degrees (base 18 °C); temperature²; Melbourne temperature | C: `available_at ≤ o`; D: perfect-foresight oracle by construction |
| Extra weather at *t* (C⁺/D⁺ only) | weighted apparent temperature, humidity, cloud cover, shortwave radiation | as above; GFS archive starts 2024-01-19 |

- **Variants C and D use identical feature names**, and so do C⁺ and D⁺. Only the source differs, so each C-vs-D gap isolates weather-forecast error.
- **Deferred:** features comparing the weather to yesterday's. They need an operationally available analysis of yesterday's weather; ERA5 is published about 5 days late.

### 2.4 Models and training

- **Target:** the correction to "same time yesterday", r = y(t) − y(t−288); forecast = y(t−288) + r̂. One ablation on development data compares this with predicting y(t) directly. *Step 3 (2026-11-06):* the target form is selected at every retrain (see Tuning) among **residual** (above), **direct** (y(t)) and **anchored**: the correction to y(o) + [y(t−288) − y(o−288)], i.e. persistence plus yesterday's change over the same lead (`dem_now_vs_yesterday` + `dem_same_time_yesterday`, no new feature). Anchored was added because neither planned form lets a tree model reproduce the current demand precisely at short leads.
- **Lead bands (settled 2026-11-06):** each ML model is **two fits**, one for leads ≤ 2 h (h = 1…24) and one for leads > 2 h (h = 25…288), each trained on its own band's rows and selected separately; forecasts take each horizon from its band. S1 (h = 24…288) is therefore judged on the long-lead fit, except at exactly 2 h (h = 24), which the short-lead fit serves. *Why:* the step-3 smoke run showed one global model 5× worse than persistence at 5 minutes. On pre-scoring data (the first retrain's early-stopping slice, 6 Sep → 31 Oct 2022; `scripts/spikes/m2/target_forms.py`) a model fitted on leads ≤ 2 h only cut the MAE over leads ≤ 1 h from 198 to 97 MW, against 113 MW for "persistence plus yesterday's change"; the global model was smearing long-lead corrections into short leads, which are only 8% of its rows.
- **LightGBM:** a single model per lead band with lead time as a feature; the L2 objective and an MAE-oriented alternative (L1) are both in the selection grid. **Ridge:** standardised features, one-hot half-hour × day type, temperature spline terms, plus one-hot lead-time hour and its interactions with `dem_now_vs_yesterday` and `dem_change_1h` (a linear model cannot otherwise learn that their effect fades with lead time). Ridge fits a seeded subsample of 500,000 training rows (about 220 coefficients).
- **Training sample (settled 2026-11-06):** every 30-minute issue time × targets every 15 minutes (96 per issue time), **staggered**: the first horizon of each issue time rotates through 1, 2, 3, shifted by one every day, so every lead time and every 5-minute slot is trained on. About 1.7 million rows per year of history (~6.5 M at the last development retrain). The plan first said "every 2 h", which gives only ~0.42 M a year and contradicted the 1.7 M figure. **Scoring:** every 30-minute issue time and every 5-minute target, processed fold by fold to keep memory bounded.
- **Walk-forward:** expanding training window, retrained **quarterly**. Early stopping uses the last 8 weeks of each training window, so no data from the evaluation fold is used; the model is then **refit on the whole window** (including those 8 weeks) with the early-stopped number of trees (settled 2026-11-06). A training pair is used only if its target is known at the retrain time. The first ML issue time is **2022-11-01**, so every model has at least 12 months of training data.
- **Tuning: re-selected at every retrain (settled 2026-11-06, replacing the pre-scoring window agreed earlier the same day).** At each quarterly retrain and for each variant, and lead band, every setting in the grid (target form residual/direct/anchored × LightGBM objective L2/L1 × 31/127 leaves; ridge: target form × alpha 0.1–1000; 255 leaves was dropped before the full run because on pre-scoring data it was both the slowest and the least accurate setting in each band, `scripts/spikes/m2/grid_cost.py`) is fitted on the early-stopping rows and scored by MAE on that retrain's last 8 weeks. The lowest MAE wins (ties go to the planned residual target, first in grid order), and only the winner is refit on the whole window. Selection uses only data before the retrain, so no scored pair is ever seen, and the residual-vs-direct ablation is reported per retrain. *Why the earlier plan was dropped:* the pre-scoring tuning retrain (2022-07-01) has only 9 months of training data, all from October to May, so the early-stopping fit never sees winter. Its validation error first *rises* (31 leaves: 358 MW after 10 trees, 408 after 50) and only slowly recovers (327 after ~600), so early stopping halts after about 10 trees; the 127-leaf model never recovers (best 343 MW after 6 trees, daily naive 380). At the first walk-forward retrain (2022-11-01, early-stopping slice 5 Sep → 31 Oct 2022, still pre-scoring) the curves fall smoothly to 270 MW after 325 trees (31 leaves) and 276 MW after 198 (127 leaves), against 458 MW for daily naive (`scripts/spikes/m2/tuning_window.py`). The 8 h delay check reuses headline C's chosen settings at every retrain, so only the input timing differs. C⁺ and D⁺ are selected the same way at their own retrains. The holdout is never used.
- **Like-for-like comparison:** M1 baselines are rescored on exactly the same (issue time, horizon) pairs as the ML models (Nov 2022 → Sep 2025 development). Pairs without C weather (the GFS gap, 2023-12-30 → 2024-01-20) are excluded for every model and baseline, and the count is reported.
- **C⁺ and D⁺:** trained from 2024-01-19 only (D⁺ on the same window, so only the source differs). First issue time **2025-02-01**, the first quarterly retrain with ≥ 12 months of history (same rule as above), giving Feb → Sep 2025 for development. Compared with C on identical pairs; reported, not pass/fail.
- **Reproducibility:** saved model files, a config hash, git SHA and seeds are recorded in `model_runs` (`model_kind = 'ml'`).

### 2.5 Feature audit (deliverable)

- **`features/catalogue.py` → generated `docs/feature_catalogue.md`:** each feature's source, when it is known, maximum lag, which variants use it, and why it is safe.
- **Leakage property tests**, extended from M1. Perturbing demand ending after *o*, or weather-forecast rows with `available_at > o` (including the later hour an interpolated value uses), must leave every A, C and C⁺ feature unchanged. Variants D and D⁺ are explicitly exempt, because they are the labelled oracles.
- **Investigate before believing:** any feature that improves pooled MAE by more than 15% on its own (one feature added or removed at a time) is checked for leakage first.
- **Importance:** built-in gain and split importance only. SHAP explanations stay in M4.

### 2.6 AEMO benchmark

- **Data:** `PREDISP_ALL_DATA/PREDISPATCHREGIONSUM` (every run, 48 a day) and `P5MIN_ALL_DATA/P5MIN_REGIONSOLUTION` (every run, 288 a day; included after step 0) for Oct 2024 → Sep 2026. Stored by migration `0004_aemo_forecasts` (source, region, run time, `available_at` = `LASTCHANGED`, target end, resolution, demand).
- **Loader rules from step 0:** map columns by name; accept `PREDISPATCHREGIONSUM` v8–9 and `P5MIN_REGIONSOLUTION` v9–10 (columns were added in 2025-04 without a version bump) and fail on anything else; keep `INTERVENTION = 0`; fail on duplicate keys; cache downloads via `listing.json`.
- **Alignment:**
  1. For each of our issue times *o*, take the latest AEMO run with `available_at ≤ o`.
  2. Pre-dispatch is 30-minute and its `TOTALDEMAND` is the period mean (confirmed in step 0), so compare against the **30-minute mean** of our 5-minute forecasts and of the actual demand, grouped by lead time to the end of the period.
  3. 5-minute pre-dispatch: the freshest run at *o* is always `RUN_DATETIME = o`, whose intervals 1–11 are our h = 1…11. Compare at the 5-minute level.
  4. Pre-dispatch looks only as far as the end of the next trading day: runs from 04:00 to 12:00 NEM time reach 16–24 h, and 64.6% of runs reach 24 h. Report coverage for each lead time and compare only where both forecasts exist.
- **Timing:** the development part (Oct 2024 → Sep 2025) is compared in step 5; the holdout part only in step 6.

### 2.7 New evaluation segments

- **Very hot day:** the weighted daily maximum temperature (ERA5) is at least 35 °C.
- **Very cloudy day:** daytime mean cloud cover is at least 80%.

Both are defined from what actually happened, like M1's market-state segments.

## 3. Success criteria (pre-registered)

These apply to **LightGBM, variant C, demand, in the development period from Nov 2022 to Sep 2025**, scored on exactly the same pairs as the rescored baselines.

| # | Criterion | Pass if |
|---|---|---|
| S1 | Beats the best baseline at every lead time from 2 h to 24 h | MAE_C(h) < min over baselines of MAE(h), for every h ∈ [24, 288] |
| S2 | Beats the best baseline across all lead times | pooled MAE_C < mean over h of (min over baselines of MAE(h)) |
| S3 | Weather adds value | pooled MAE_C < pooled MAE_A |
| S4 | Consistent with perfect foresight and leakage-free | pooled MAE_D ≤ pooled MAE_C, and every leakage property test passes |
| S5 | Holdout used once | holdout scored by a single `gridcast evaluate-holdout` run against the frozen model config; the command refuses to run again for the same config hash |

- **Variant definitions after step 0:** "C" is the headline GFS-temperature model and "D" is its ERA5 mirror (same features), so S4 compares like with like. Pairs without C weather are excluded for every model (§2.4).
- **No pass/fail:** the AEMO comparison, the ridge model, C⁺/D⁺, and lead times under 2 h. All are reported.
- **Failures are published:** a failed criterion is reported as a failure in the M2 report and the README. If S4 fails (C beats D), results are withheld until the cause is found.

## 4. Steps (vertical slices; one branch, one commit per step)

Branch `m2-weather-demand-ml` from `m1-pipeline-baseline` (or from `main`, once M1 is merged).

**Step 0. Data-availability check.** ✅ Done 2026-11-02. Output: [`docs/m2_data_spike.md`](../m2_data_spike.md); scripts in `scripts/spikes/m2/`. Outcome: the second fallback rule applied (only temperature archived before 2024), temperature gaps 1.1–1.2% (< 2%), and the decisions recorded in §1.
- Using Melbourne as the test site, confirm which variables `gfs_seamless` has in the Previous Runs archive, and from which date.
- Use the Single Runs API on a sample to confirm when each offset's run started, then fix `publication_delay`.
- Check `PREDISPATCHREGIONSUM` "ALL" files: schema, run identifiers, and how `LASTCHANGED` relates to the run time.
- Check how pre-dispatch defines `TOTALDEMAND` (period mean vs period end) against actual 5-minute data.
- Check the size of the 5-minute pre-dispatch archive.
- **Fallback rules, agreed in advance:**
  - If all C variables are archived from ≤ 2021-11 → C uses the full window.
  - If only temperature is archived before 2024 → headline C uses forecast temperature (and derived features) over the full window. A secondary model, C⁺, uses all forecast variables but trains from 2024-01 and is evaluated only where it has enough history. S1–S5 are judged on headline C.
  - If even temperature has gaps of more than 2% → stop and revisit with the user.

**Step 1. Weather ingestion.** ✅ Done 2026-11-03.
- Open-Meteo client: rate-limited, cached, retried, with a project User-Agent.
- Migration `0003_weather`, site weights, 5-minute alignment, and weather data-quality checks.
- Update `docs/data_sources.md`.
- Outcome: `gridcast ingest-weather` loaded all 60 months for both datasets. `weather_gaps` is 0 at every site; the only missing values are the step-0 known ranges. Decisions made during the step: weights are fixed ABS ERP at 30 June 2024 (Latrobe Valley = Traralgon–Morwell + Moe–Newborough); the Melbourne point is the spike point; 5-minute alignment is computed on the fly (`gridcast/weather/align.py`), never stored. A new finding: the GFS archive holds 153 cloud cover values of exactly −1 % or 101 %, a rounding artefact. Values within 1 point of a bounded range are clipped on load and counted (`weather_clipped_values`, warn).

**Step 2. Feature builder and audit.** ✅ Done 2026-11-04.
- Catalogue, A/C/D feature matrices, leakage property tests, generated `docs/feature_catalogue.md`.
- Outcome: `src/gridcast/features/` gained `catalogue.py` (single source of feature names, definitions and leakage reasoning), `weather.py` (5-minute alignment and the freshest-published-offset rule), `builder.py` (`FeatureBuilder.build(variant, origins, horizons)`, variants `A`, `C`, `D`, `Cplus`, `Dplus`), `inputs.py` (database loading) and `audit.py`. New commands: `gridcast feature-catalogue` (`make catalogue`) and `gridcast feature-audit` (`make feature-audit`). Features are computed on the fly, never stored; a full development-period audit (14.7 M pairs × 6 builds) takes about 5 seconds.
- Decisions made with the user: (1) two generated documents, kept separate: `docs/feature_catalogue.md` from code (a unit test fails if it is stale) and `reports/m2_feature_audit.md` from the database; (2) the **8 h publication-delay sensitivity check is adopted**: `features.delay_sensitivity_hours`, reported in step 4, never the headline (a delay shorter than the configured 6 h is refused as a leak); (3) the step-1 cloud-cover clipping rule is kept.
- Decisions made while building: all weather features of a pair come from **one** forecast offset (the freshest whose every required value is published by *o*); the Melbourne temperature feature is `wx_temp_single_site` (`features.single_site`); demand-window statistics use ddof = 0 and are missing if any interval in the window is missing (none are).
- Leakage tests: future demand perturbed → all five variants unchanged; forecast rows with `available_at > o` perturbed or deleted → C and C⁺ unchanged; controls show both tests fail when the rule is broken. Two deliberate mutations (forecasts read 1 h before publication; demand cutoff one interval late) were each caught.
- Audit findings (from the generated report): 1.94% of development pairs have no C weather (the GFS seam, targets 2023-12-29 23:05 → 2024-01-20 11:55 UTC) and are excluded for every model; day 1 is used for 82.7% of pairs and 84.4% of those with weather, matching step 0's 84.4%; C-vs-ERA5 temperature differs by about 1.1 °C at every lead time, because C's forecast is 24–54 h old at every lead; the timing check is smallest at zero shift.

**Step 3. Models in the walk-forward harness.**
- *Decisions made with the user (2026-11-06):* (1) training sample = every 30-minute issue time × staggered targets every 15 min (§2.4); (2) tuning and the target ablation per variant, first agreed on the pre-scoring window, then (after the spike below) **re-selected at every retrain** on its early-stopping slice (§2.4); (3) after early stopping, refit on the whole window; (4) after the first smoke run, **two lead bands** (≤ 2 h, > 2 h) per model and the **anchored** target form (§2.4); (5) LightGBM needs the OpenMP runtime on macOS (`brew install libomp`), noted in the README. Defaults taken: A, C and D train on the **same rows** (pairs without C weather are dropped for all three, so C vs D compares like with like); the "very cloudy day" segment uses local daytime 09:00–17:00; model files live under `data_root/models/`.
- LightGBM and ridge for A, C and D, retrained quarterly, with inner-validation tuning; C⁺ and D⁺ from 2025-02-01.
- Baselines rescored on identical pairs. Model files and provenance saved.
- New segments added.
- *Outcome (2026-11-06):* `make backtest-ml` ran from the clean step-3 commit `15623d6` (config hash `9e6042bef320ee47`) in 7.7 h and stored three batches. `main` = `b424894e-7a93-4c56-b5e6-76a9bf592c66` (A/C/D × LightGBM/ridge plus the three baselines, 9 runs). `plus` = `a57961eb-5191-444c-b48f-94e9c5f00b88` (C⁺/D⁺ next to C, 9 runs). `delay` = `b2aa39b2-720f-4a34-9062-6b2c9ad7a6db` (C with the 8 h delay, 4 runs). 216 band fits were selected and refit: 144 for A/C/D (3 variants × 2 families × 12 retrains × 2 bands), 24 for C⁺/D⁺ (3 retrains) and 48 for the delay check. **Selection:** LightGBM's short-lead fit chose the *direct* target at every retrain; the long-lead fit chose direct, residual or anchored. The L1 objective and 31 leaves won most fits. The planned residual target is therefore not the usual winner; the full per-retrain table belongs in the step-4 report. **Two observations that step 4 must show** (S1–S4 verdicts come only from the generated report):
  1. At 5 minutes LightGBM is worse than persistence; from 10 minutes on it is better (leads under 2 h are not pass/fail).
  2. The MAE curve jumps at the band boundary, between 2 h (short-lead fit) and 2 h 5 min (long-lead fit), so the long-lead fit is weakest at its shortest leads.

**Step 4. Development-period evaluation and success criteria.**
- `gridcast report-m2` generates `reports/m2_demand.md`: S1–S4 pass/fail, MAE by lead time against the best-baseline curve, the C–D gap, importance, and segment and failure-case tables (hot days, cloudy days, holidays, DST days).
- *Decisions made with the user (2026-11-06):*
  1. **The step-3 run is reported as it is.** Its two weaknesses (5-minute MAE above persistence; the jump at the 2 h band boundary) are shown in the report. Any fix becomes a separate, disclosed development run, compared like for like with this one before the holdout freeze.
  2. **"Investigate before believing" (§2.5):** weather (C vs A) cuts pooled MAE by about 20%, above the 15% threshold. The report gives the existing evidence: the leakage tests (run by `report-m2` itself), the 8 h delay check, the gain by lead time and D ≤ C. It also runs a drop/add test, `gridcast weather-ablation` (`make weather-ablation`). At two retrains (2025-02-01 and 2025-08-01, summer and winter early-stopping slices), LightGBM C's chosen setting is refitted on the early-stopping rows with each weather feature dropped or added alone. Any feature that cuts the pooled early-stopping MAE by more than 15% is flagged. The all-features fit must reproduce the stored early-stopping MAE exactly.
  3. **Failure cases:** the 10 local days with the highest LightGBM C MAE at leads 2–24 h, with day type, ERA5 maximum temperature and C's temperature-forecast error, plus the known events of 2024-02-13 (storms) and 2024-12-09 (unexplained demand spike).
- *Defaults taken:* S1–S4 are computed in code exactly as written in §3, and the report runs the leakage property tests itself, so S4's second half is measured rather than asserted. Importance is LightGBM's built-in gain and split share, averaged over the 12 retrains per lead band. Errors are also reported by year (risk table, §6). M1's `gridcast report` now ignores ML batches; it had been picking the latest demand batch, which was an ML batch.

**Step 5. AEMO benchmark (development part).** Migration `0004_aemo_forecasts`, ingestion of pre-dispatch and 5-minute pre-dispatch for Oct 2024 → Sep 2026 (holdout months downloaded but not scored), alignment and comparison.

**Step 6. Freeze and holdout.**
- Freeze the model config and evaluate the holdout once (S5), including the holdout part of the AEMO benchmark.
- Update the README sections *Models*, *Results*, *Failure cases* and *Operational implications*.

## 5. New dependencies and compute

- **Dependencies:** `lightgbm`, `scikit-learn` (ridge, splines).
- **Compute:** 12 quarterly retrains × 3 variants × 2 lead bands, each scoring 12 LightGBM and 15 ridge settings, plus 3 retrains each for C⁺ and D⁺ and 12 fixed-setting retrains for the delay check. Measured in the step-3 smoke run (first retrain, 1.8 M training rows, 12-core Apple M-series laptop): about 45 minutes with the 255-leaf settings still in the grid; ridge fits take about 1 s, LightGBM fits 3–50 s. Expect roughly 8–10 hours for the full walk-forward (an overnight run); the full run took 7.7 hours and wrote about 0.2 GB of model files. Scoring about 15 M pairs per variant is done month by month.
- **Data:** Open-Meteo needs only a few dozen API calls, well within the free tier's 10,000 a day. Pre-dispatch and 5-minute pre-dispatch are about 2.5 GB zipped in total, cached under the data root and gitignored.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Archived forecast coverage before 2024 | Resolved in step 0: headline C is temperature-only; C⁺ shows what the other variables add from 2024 |
| Weather-model upgrades change forecast error over time | Report errors by year; note it as a limitation |
| Publication timing leaks future information | Run time verified exactly in step 0; 6 h delay vs 5.4–5.5 h observed for one run (≈0.5 h margin); `available_at` filtering, property tests. 8 h delay sensitivity check (adopted in step 2, reported at step 4) |
| Cloud-driven midday swings (rooftop PV is not in `TOTALDEMAND`) | Headline C has no cloud or radiation forecasts (none archived before 2024); C⁺/D⁺ and the "very cloudy day" segment show the cost. Rooftop-PV-adjusted "underlying demand" stays a future idea |
| Pre-dispatch is 30-minute and covers only part of the next day | 30-minute aggregation; per-lead-time coverage reporting |
| Overfitting through tuning | Small grid, inner validation inside the development period, holdout scored once |

## 7. Out of scope for M2

Price ML (M2b), prediction intervals (M3), anomaly detection and SHAP (M4), API (M5), dashboard (M6), monitoring (M7), regions other than Victoria, live/near-real-time ingestion.

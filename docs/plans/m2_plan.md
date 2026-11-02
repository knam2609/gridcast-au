# Milestone 2 plan: weather-driven demand ML

**Goal:** find out whether machine learning with honestly available weather forecasts beats M1's best baseline at each lead time for 24-hour-ahead Victorian demand, and how it compares with AEMO's own forecasts.

**Status:** step 0 done ([`docs/m2_data_spike.md`](../m2_data_spike.md), 2026-11-02); step 1 done (2026-11-03, weather loaded and checked, see [`docs/data_sources.md`](../data_sources.md) §4–5); step 2 next. Success criteria were agreed before any ML results existed (see §3). The step-0 decisions below refine *how* C, D and the benchmark are built; they do not change S1–S5.

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

- **Target:** the correction to "same time yesterday", r = y(t) − y(t−288); forecast = y(t−288) + r̂. One ablation on development data compares this with predicting y(t) directly.
- **LightGBM:** a single global model with lead time as a feature, L2 objective (MAE-oriented alternatives are tried in tuning). **Ridge:** standardised features, one-hot half-hour × day type, temperature spline terms.
- **Training sample:** issue times every 2 h × targets every 15 min, about 1.7 million rows per year of history. **Scoring:** every 30-minute issue time and every 5-minute target, processed fold by fold to keep memory bounded.
- **Walk-forward:** expanding training window, retrained **quarterly**. Early stopping uses the last 8 weeks of each training window, so no data from the evaluation fold is used. The first ML issue time is **2022-11-01**, so every model has at least 12 months of training data.
- **Tuning:** a small LightGBM grid evaluated on an inner, time-ordered validation slice **inside the development period**. The holdout is never used.
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

**Step 2. Feature builder and audit.** Catalogue, A/C/D feature matrices, leakage property tests, generated `docs/feature_catalogue.md`.

**Step 3. Models in the walk-forward harness.**
- LightGBM and ridge for A, C and D, retrained quarterly, with inner-validation tuning; C⁺ and D⁺ from 2025-02-01.
- Baselines rescored on identical pairs. Model files and provenance saved.
- New segments added.

**Step 4. Development-period evaluation and success criteria.**
- `gridcast report-m2` generates `reports/m2_demand.md`: S1–S4 pass/fail, MAE by lead time against the best-baseline curve, the C–D gap, importance, and segment and failure-case tables (hot days, cloudy days, holidays, DST days).

**Step 5. AEMO benchmark (development part).** Migration `0004_aemo_forecasts`, ingestion of pre-dispatch and 5-minute pre-dispatch for Oct 2024 → Sep 2026 (holdout months downloaded but not scored), alignment and comparison.

**Step 6. Freeze and holdout.**
- Freeze the model config and evaluate the holdout once (S5), including the holdout part of the AEMO benchmark.
- Update the README sections *Models*, *Results*, *Failure cases* and *Operational implications*.

## 5. New dependencies and compute

- **Dependencies:** `lightgbm`, `scikit-learn` (ridge, splines).
- **Compute:** about 12 quarterly retrains × 3 variants × 2 models, plus about 3 retrains each for C⁺ and D⁺ in development. LightGBM fits take roughly 1–2 minutes each on ~1.7 M rows, and scoring about 20 M pairs per variant is done fold by fold. Expect around an hour for the full walk-forward on a laptop.
- **Data:** Open-Meteo needs only a few dozen API calls, well within the free tier's 10,000 a day. Pre-dispatch and 5-minute pre-dispatch are about 2.5 GB zipped in total, cached under the data root and gitignored.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Archived forecast coverage before 2024 | Resolved in step 0: headline C is temperature-only; C⁺ shows what the other variables add from 2024 |
| Weather-model upgrades change forecast error over time | Report errors by year; note it as a limitation |
| Publication timing leaks future information | Run time verified exactly in step 0; 6 h delay vs 5.4–5.5 h observed for one run (≈0.5 h margin); `available_at` filtering, property tests. Optional: an 8 h delay sensitivity check at step 4 |
| Cloud-driven midday swings (rooftop PV is not in `TOTALDEMAND`) | Headline C has no cloud or radiation forecasts (none archived before 2024); C⁺/D⁺ and the "very cloudy day" segment show the cost. Rooftop-PV-adjusted "underlying demand" stays a future idea |
| Pre-dispatch is 30-minute and covers only part of the next day | 30-minute aggregation; per-lead-time coverage reporting |
| Overfitting through tuning | Small grid, inner validation inside the development period, holdout scored once |

## 7. Out of scope for M2

Price ML (M2b), prediction intervals (M3), anomaly detection and SHAP (M4), API (M5), dashboard (M6), monitoring (M7), regions other than Victoria, live/near-real-time ingestion.

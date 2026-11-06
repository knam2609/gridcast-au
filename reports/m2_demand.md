# Milestone 2: weather-aware demand forecasts (development period)

> **Generated** by `gridcast report-m2` on 2026-11-06 13:12 UTC from the stored walk-forward run. Do not edit by hand: re-run the pipeline instead. The holdout period is not scored here.

## Provenance

| Item | Value |
| :--- | :--- |
| Region | VIC1 |
| Batches | main `b424894e-7a93-4c56-b5e6-76a9bf592c66`, plus `a57961eb-5191-444c-b48f-94e9c5f00b88`, delay `b2aa39b2-720f-4a34-9062-6b2c9ad7a6db` |
| Git commit(s) of the runs | `15623d61ca6f3eed8792e44d2433930994ec3b89` |
| ML config hash | `9e6042bef320ee47` (matches the current config) |
| Development issue times (NEM time) | 2022-11-01 00:00 → 2025-09-30 23:30, 51,120 (every 30 min) |
| Retrains | 12, every 3 months, 2022-11-01 → 2025-08-01 |
| Lead bands (steps of 5 min) | 1–24, 25–288 |
| Scored (issue time, lead) pairs, every model | 14,436,480 of 14,722,560 (286,080 excluded, 1.94%) |
| Training rows at the last retrain (shared by A, C, D) | 6,323,488 |
| Leakage property tests (run by this command) | 6 passed, 0 failed |
| Report generated from commit | `1f367d7778f9025f4c59621bdbe4f948a412d84e` |

## Success criteria (pre-registered)

Set in `docs/plans/m2_plan.md` §3 before any ML result existed. They apply to LightGBM, variant C (GFS temperature forecasts as available at the issue time), scored on exactly the same pairs as the rescored baselines. *Best baseline* means the lowest MAE among persistence, daily naive, weekly naive at each lead time. S5 (the holdout, scored once) belongs to step 6.

| # | Pass if | Measured | Result |
| :--- | :--- | :--- | :--- |
| S1 | MAE_C(h) < min over baselines of MAE(h), for every h from 2 h to 24 h | 265 of 265 lead times; closest at 24 h: 296.2 vs 476.7 MW | **PASS** |
| S2 | pooled MAE_C < mean over h of (min over baselines of MAE(h)) | 267.5 vs 463.0 MW | **PASS** |
| S3 | pooled MAE_C < pooled MAE_A | 267.5 vs 332.6 MW | **PASS** |
| S4 | pooled MAE_D ≤ pooled MAE_C, and every leakage property test passes | 263.0 vs 267.5 MW; leakage tests 6 passed, 0 failed | **PASS** |

**4 of 4 criteria pass.**

## How to read the results

- Each issue time (every 30 min) gets a 24-hour forecast in 5-minute steps; errors are grouped by **lead time** and pooled over every issue time.
- **A** has no weather, **C** uses GFS temperature forecasts published before the issue time, **D** uses ERA5 reanalysis for the target time (perfect foresight, an upper bound for C's inputs). C and D have identical features, so C − D is the cost of weather-forecast error.
- Every ML model is two fits: one for leads up to 2 h, one beyond. Its settings (target form, objective, size or penalty) are re-selected at every retrain (every 3 months) on the last 8 weeks before it.
- **Bold** marks the lowest error in each column. Error = forecast − actual.

## MAE by lead time

![MAE by lead time](figures/m2_demand_mae_by_lead.png)

### MAE (MW), development period

| Model | 5 min | 30 min | 1 h | 2 h | 2 h 5 min | 4 h | 12 h | 24 h | All horizons |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM A | 56.0 | 87.1 | 119.2 | 181.7 | 284.5 | 315.6 | 357.1 | 375.1 | 332.6 |
| LightGBM C | 58.4 | 86.7 | 115.9 | 170.7 | **247.4** | 264.0 | 283.4 | 296.2 | 267.5 |
| LightGBM D | 58.2 | **86.2** | **115.0** | **169.3** | 249.8 | **262.3** | **279.0** | **285.8** | **263.0** |
| Ridge A | 99.4 | 107.6 | 151.2 | 233.9 | 299.6 | 346.9 | 384.3 | 390.6 | 357.1 |
| Ridge C | 101.8 | 108.6 | 149.3 | 224.8 | 281.6 | 315.7 | 341.6 | 346.2 | 319.4 |
| Ridge D | 102.6 | 109.4 | 149.4 | 224.2 | 285.5 | 318.5 | 344.6 | 347.5 | 321.8 |
| Persistence | **46.9** | 175.1 | 321.0 | 591.3 | 610.9 | 990.2 | 946.8 | 476.7 | 935.4 |
| Daily naive | 476.3 | 476.7 | 476.6 | 476.6 | 476.3 | 476.6 | 476.6 | 476.7 | 476.7 |
| Weekly naive | 542.7 | 542.7 | 542.7 | 542.7 | 542.7 | 542.7 | 542.6 | 542.8 | 542.5 |

### RMSE (MW), development period

| Model | 5 min | 30 min | 1 h | 2 h | 2 h 5 min | 4 h | 12 h | 24 h | All horizons |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM A | 82.0 | 124.4 | 173.1 | 268.1 | 436.5 | 482.4 | 538.4 | 566.6 | 511.6 |
| LightGBM C | 83.7 | 122.6 | 166.4 | 248.5 | **358.0** | 385.9 | 410.7 | 441.7 | 396.2 |
| LightGBM D | 83.4 | **122.1** | **165.3** | **247.4** | 363.7 | **383.8** | **402.9** | **415.7** | **387.1** |
| Ridge A | 126.6 | 145.5 | 210.1 | 328.1 | 409.8 | 486.2 | 547.0 | 559.4 | 516.3 |
| Ridge C | 129.9 | 145.8 | 205.9 | 314.0 | 378.9 | 430.1 | 471.8 | 480.4 | 447.8 |
| Ridge D | 131.0 | 146.5 | 205.9 | 313.0 | 381.8 | 431.6 | 472.1 | 476.9 | 447.8 |
| Persistence | **60.6** | 219.8 | 402.8 | 734.4 | 757.7 | 1,195.9 | 1,178.0 | 705.1 | 1,193.9 |
| Daily naive | 705.0 | 704.9 | 704.9 | 704.9 | 705.0 | 704.9 | 704.9 | 705.1 | 705.0 |
| Weekly naive | 803.1 | 803.1 | 803.1 | 803.1 | 803.1 | 803.1 | 803.0 | 803.3 | 802.8 |

### Bias (MW, forecast − actual), development period

| Model | 5 min | 30 min | 1 h | 2 h | 2 h 5 min | 4 h | 12 h | 24 h | All horizons |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM A | -1.0 | 0.5 | 0.3 | 5.2 | 14.4 | 16.5 | 16.9 | 19.1 | 15.2 |
| LightGBM C | -4.1 | -3.6 | -5.2 | -2.0 | -4.8 | -4.9 | -5.8 | -7.1 | -6.0 |
| LightGBM D | -4.9 | -4.4 | -6.0 | -2.6 | -1.0 | -1.1 | -0.5 | -0.2 | -1.2 |
| Ridge A | 8.5 | 8.4 | 8.2 | 8.1 | 40.1 | 41.8 | 42.8 | 42.1 | 37.7 |
| Ridge C | 6.7 | 6.7 | 6.5 | 6.4 | 28.0 | 29.1 | 29.7 | 28.0 | 25.8 |
| Ridge D | 7.6 | 7.6 | 7.4 | 7.3 | 29.2 | 30.3 | 31.1 | 31.1 | 27.2 |
| Persistence | -0.2 | 0.0 | 0.1 | 0.1 | -0.1 | 0.2 | 0.2 | 0.4 | -2.0 |
| Daily naive | 0.5 | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 | 0.7 | 0.4 | 0.6 |
| Weekly naive | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 4.2 | 4.1 | 4.2 |

## Short lead times and the band boundary

![Short lead times](figures/m2_demand_short_leads.png)

- At 5 min, LightGBM C's MAE is 58.4 MW against 46.9 MW for persistence; LightGBM C is better from 10 min on. Lead times under 2 h are reported, not pass/fail.
- Between 2 h (the short-lead fit's last step) and 2 h 5 min (the long-lead fit's first), LightGBM C's MAE changes by +76.8 MW (A: +102.8 MW), against a median step of 0.4 MW between neighbouring lead times elsewhere. The long-lead fit is weakest at its shortest leads.
- Both are known weaknesses of this run. They are reported as they are; a fix would be a separate, disclosed development run (decision of 2026-11-06).

## Weather value by lead time

Mean of the per-lead MAE within each lead-time range (MW). The delay column comes from the delay batch, on its own common pairs.

| Lead time | LightGBM A | LightGBM C | LightGBM D | C vs A (%) | C − D (MW) | 8 h delay − C (MW) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min–1 h | 88.3 | 87.7 | 87.3 | -0.6 | +0.5 | +0.0 |
| 1 h 5 min–2 h | 152.5 | 145.0 | 143.8 | -4.9 | +1.3 | +0.0 |
| 2 h 5 min–6 h | 314.3 | 263.1 | 261.7 | -16.3 | +1.4 | +1.5 |
| 6 h 5 min–12 h | 351.0 | 280.7 | 276.1 | -20.0 | +4.6 | +1.9 |
| 12 h 5 min–24 h | 364.8 | 287.6 | 281.5 | -21.2 | +6.1 | +2.2 |

## Weather value: the “investigate before believing” check

The plan (§2.5) says any feature that improves pooled MAE by more than 15% on its own is checked for leakage before it is believed. Over the development period, weather (C vs A) cuts the pooled MAE by **19.6%**, above that threshold. Weather is a group of features, so each one is also tested alone below.

**Drop/add test.** At each listed retrain, LightGBM C's chosen setting was refitted on that retrain's early-stopping rows (the last 8 weeks before it are monitored) with each weather feature added alone to the weather-free set, or dropped from the full set. MAE is pooled over both lead bands. *Keep gain* is what the feature adds on top of the other weather features. A feature is flagged above 15%. The all-features fit reproduced the stored run's early-stopping MAE exactly at every retrain and band. Run from commit `dd16e80689a9e71177450594a98bf1656263559d`.

| Retrain | Feature | MAE no weather | MAE + feature | Add gain (%) | Keep gain (%) | All weather gain (%) | Flagged |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 2025-02-01 | `wx_temp` | 475.1 | 352.0 | 25.9 | 0.2 | 26.7 | **yes** |
| 2025-02-01 | `wx_cooling_degrees` | 475.1 | 346.4 | 27.1 | 1.2 | 26.7 | **yes** |
| 2025-02-01 | `wx_heating_degrees` | 475.1 | 469.2 | 1.2 | 0.7 | 26.7 | no |
| 2025-02-01 | `wx_temp_sq` | 475.1 | 354.7 | 25.4 | 1.1 | 26.7 | **yes** |
| 2025-02-01 | `wx_temp_single_site` | 475.1 | 349.6 | 26.4 | 0.6 | 26.7 | **yes** |
| 2025-08-01 | `wx_temp` | 306.0 | 260.9 | 14.7 | 0.3 | 15.0 | no |
| 2025-08-01 | `wx_cooling_degrees` | 306.0 | 309.5 | -1.1 | -0.1 | 15.0 | no |
| 2025-08-01 | `wx_heating_degrees` | 306.0 | 260.4 | 14.9 | -0.1 | 15.0 | no |
| 2025-08-01 | `wx_temp_sq` | 306.0 | 259.8 | 15.1 | -0.2 | 15.0 | **yes** |
| 2025-08-01 | `wx_temp_single_site` | 306.0 | 262.7 | 14.1 | -0.2 | 15.0 | no |

**Flagged:** `wx_cooling_degrees`, `wx_temp`, `wx_temp_single_site`, `wx_temp_sq`. The evidence below is the leakage check the plan asks for.

- Monitored issue times per retrain: 2025-02-01: 2024-12-07 → 2025-01-31; 2025-08-01: 2025-06-06 → 2025-07-31.
- No weather feature matters much given the others: the largest keep gain is 1.2%. The features are near-substitutes, so a flag measures what temperature as a whole adds, not one suspicious column.
- The degree features behave as their definitions imply: cooling degrees (heat above the base temperature) help alone at 2025-02-01 (27.1%) but not at 2025-08-01 (-1.1%); heating degrees the reverse (14.9% vs 1.2%). Information leaked from future demand would not depend on the season like this.

**Leakage evidence:**

- **Leakage property tests:** 6 passed, 0 failed (future demand and unpublished forecast rows perturbed or deleted; C's features must not change).
- **Publication timing:** assuming GFS runs are published 8 h after their run time instead of 6 h changes the pooled MAE by +1.8 MW, against a C-vs-A gain of 65.1 MW. The gain does not hinge on the last hours before publication, where a timing leak would sit.
- **Where the gain comes from:** C vs A is -0.6% at 5 min–1 h and -21.2% at 12 h 5 min–24 h, so weather helps most at long leads, where recent demand says least.
- **Perfect foresight bounds it:** D's pooled MAE is 4.5 MW below C's, and C is no better than D in any lead-time range.

## C⁺ and D⁺: all forecast weather variables (reported, not pass/fail)

C⁺ adds apparent temperature, humidity, cloud cover and solar radiation forecasts, archived only from 2024-01-19, so it trains from then and is scored from 2025-02-01 (11,616 issue times, 3,345,408 pairs), next to headline C on the same pairs.

| Model | 5 min | 30 min | 1 h | 2 h | 2 h 5 min | 4 h | 12 h | 24 h | All horizons |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM C⁺ | 66.5 | 95.4 | 127.1 | 181.8 | 246.0 | 250.7 | 257.1 | 264.9 | 245.5 |
| LightGBM D⁺ | 67.3 | 94.6 | **124.8** | **172.9** | **226.4** | **229.8** | **234.6** | **239.4** | **224.2** |
| LightGBM C | 59.4 | **91.9** | 125.8 | 186.2 | 259.0 | 278.7 | 299.2 | 306.7 | 281.7 |
| Ridge C⁺ | 123.7 | 121.9 | 162.4 | 231.9 | 266.9 | 293.7 | 304.5 | 305.9 | 287.3 |
| Ridge D⁺ | 127.6 | 123.4 | 160.3 | 224.1 | 264.6 | 280.6 | 290.7 | 291.6 | 276.4 |
| Ridge C | 106.6 | 115.2 | 165.3 | 249.9 | 284.8 | 326.1 | 351.9 | 351.8 | 329.4 |
| Persistence | **50.2** | 201.1 | 373.7 | 693.7 | 716.8 | 1,159.3 | 1,032.8 | 501.3 | 1,061.7 |
| Daily naive | 504.4 | 503.9 | 503.9 | 503.9 | 504.4 | 503.9 | 503.6 | 501.3 | 503.5 |
| Weekly naive | 610.1 | 610.3 | 610.3 | 610.3 | 610.1 | 610.2 | 610.2 | 607.5 | 609.1 |

## Publication-delay sensitivity (reported, never the headline)

C rebuilt as if every GFS run were published 8 h after its run time (headline: 6 h), with headline C's chosen settings at every retrain, so only the input timing differs.

| Model | 5 min | 30 min | 1 h | 2 h | 2 h 5 min | 4 h | 12 h | 24 h | All horizons |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM C | **58.4** | **86.7** | **115.9** | **170.7** | **247.4** | **264.0** | **283.4** | **296.2** | **267.5** |
| Ridge C | 101.8 | 108.6 | 149.3 | 224.8 | 281.6 | 315.7 | 341.6 | 346.2 | 319.4 |
| LightGBM C (8 h delay) | **58.4** | **86.7** | **115.9** | **170.7** | 248.5 | 265.5 | 285.3 | 297.8 | 269.4 |
| Ridge C (8 h delay) | 101.8 | 108.6 | 149.3 | 224.8 | 281.5 | 315.8 | 341.9 | 346.2 | 319.7 |

## Errors by year

Pooled MAE (MW) by NEM year of the issue time. Weather-model upgrades can change C's forecast error over time; C − D tracks that cost.

| Year | Pairs | LightGBM A | LightGBM C | LightGBM D | Ridge C | Daily naive | C − D |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 (Nov–Dec) | 843,264 | 303.5 | 243.9 | 235.7 | 287.0 | 419.2 | +8.2 |
| 2023 | 5,016,528 | 301.0 | 247.8 | 243.0 | 302.4 | 442.9 | +4.8 |
| 2024 | 4,802,736 | 344.8 | 274.9 | 272.0 | 321.1 | 491.0 | +2.8 |
| 2025 (Jan–Sep) | 3,773,952 | 365.6 | 289.8 | 284.2 | 347.3 | 516.1 | +5.6 |

## Model selection at each retrain

LightGBM C, per retrain and lead band: the winning setting and the best early-stopping MAE each target form reached (residual: correction to the same time yesterday; direct: demand itself; anchored: correction to the latest demand plus yesterday's change). This is the planned residual-vs-direct ablation.

| Retrain | Lead band | Chosen | Trees | Best residual (MW) | Best direct (MW) | Best anchored (MW) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| 2022-11-01 | 5 min–2 h | direct, L1, 31 leaves | 1587 | 121.3 | **116.3** | 120.4 |
| 2022-11-01 | 2 h 5 min–24 h | direct, L2, 31 leaves | 407 | 274.4 | **265.9** | 272.2 |
| 2023-02-01 | 5 min–2 h | direct, L1, 127 leaves | 492 | 119.8 | **108.9** | 117.3 |
| 2023-02-01 | 2 h 5 min–24 h | direct, L1, 31 leaves | 545 | 306.3 | **300.2** | 307.7 |
| 2023-05-01 | 5 min–2 h | direct, L2, 31 leaves | 1415 | 106.9 | **100.0** | 105.9 |
| 2023-05-01 | 2 h 5 min–24 h | direct, L2, 31 leaves | 197 | 238.8 | **234.2** | 235.7 |
| 2023-08-01 | 5 min–2 h | direct, L1, 31 leaves | 2254 | 110.8 | **102.4** | 109.5 |
| 2023-08-01 | 2 h 5 min–24 h | direct, L1, 31 leaves | 464 | 217.1 | **212.8** | 219.7 |
| 2023-11-01 | 5 min–2 h | direct, L1, 31 leaves | 1925 | 122.7 | **119.2** | 124.2 |
| 2023-11-01 | 2 h 5 min–24 h | residual, L1, 31 leaves | 406 | **273.9** | 285.4 | 278.1 |
| 2024-02-01 | 5 min–2 h | direct, L1, 127 leaves | 504 | 137.9 | **123.1** | 136.5 |
| 2024-02-01 | 2 h 5 min–24 h | residual, L1, 127 leaves | 313 | **338.7** | 342.3 | 345.5 |
| 2024-05-01 | 5 min–2 h | direct, L1, 127 leaves | 1090 | 101.6 | **98.1** | 100.9 |
| 2024-05-01 | 2 h 5 min–24 h | anchored, L1, 31 leaves | 960 | 224.6 | 222.4 | **221.5** |
| 2024-08-01 | 5 min–2 h | direct, L1, 31 leaves | 1525 | 125.7 | **124.2** | 125.0 |
| 2024-08-01 | 2 h 5 min–24 h | residual, L1, 31 leaves | 482 | **256.3** | 274.5 | 262.2 |
| 2024-11-01 | 5 min–2 h | direct, L1, 31 leaves | 2315 | 135.1 | **127.9** | 133.5 |
| 2024-11-01 | 2 h 5 min–24 h | anchored, L1, 31 leaves | 588 | 303.2 | 303.7 | **302.9** |
| 2025-02-01 | 5 min–2 h | direct, L1, 31 leaves | 2141 | 156.0 | **134.4** | 146.2 |
| 2025-02-01 | 2 h 5 min–24 h | direct, L1, 31 leaves | 1090 | 368.7 | **367.9** | 389.6 |
| 2025-05-01 | 5 min–2 h | direct, L1, 31 leaves | 2268 | 116.4 | **111.3** | 114.6 |
| 2025-05-01 | 2 h 5 min–24 h | direct, L1, 127 leaves | 272 | 283.1 | **278.7** | 280.3 |
| 2025-08-01 | 5 min–2 h | direct, L1, 127 leaves | 1299 | 128.2 | **121.6** | 127.9 |
| 2025-08-01 | 2 h 5 min–24 h | anchored, L2, 31 leaves | 723 | 273.9 | 274.9 | **273.0** |

Winning target form across every grid-selected fit (A, C, D, C⁺, D⁺):

| Family | Lead band | anchored | direct | residual |
| :--- | :--- | ---: | ---: | ---: |
| LightGBM | 5 min–2 h | 0 | 42 | 0 |
| LightGBM | 2 h 5 min–24 h | 7 | 23 | 12 |
| Ridge | 5 min–2 h | 38 | 1 | 3 |
| Ridge | 2 h 5 min–24 h | 18 | 10 | 14 |

## Feature importance (LightGBM C)

Built-in importance of the saved models, as a share of each model's total, averaged over the retrains. Gain depends on the target form, which changes between retrains, so read it as a ranking, not an effect size.

### Lead band 5 min–2 h

| Feature | Gain share, mean (%) | Range over retrains (%) | Split share, mean (%) |
| :--- | ---: | ---: | ---: |
| `dem_now` | 49.5 | 44.4–79.1 | 7.2 |
| `dem_same_time_yesterday` | 7.4 | 4.5–9.7 | 5.2 |
| `dem_now_vs_yesterday` | 6.2 | 2.2–7.7 | 8.0 |
| `dem_change_1h` | 5.2 | 3.7–5.5 | 6.7 |
| `lead_minutes` | 5.2 | 1.4–6.0 | 5.6 |
| `cal_tod_cos` | 3.7 | 1.3–4.9 | 3.7 |
| `dem_same_time_last_week` | 3.2 | 1.8–3.8 | 4.5 |
| `cal_tod_sin` | 3.1 | 1.6–3.7 | 3.9 |
| `cal_slot` | 2.8 | 0.8–3.2 | 5.7 |
| `dem_mean_24h` | 2.3 | 0.5–2.8 | 8.1 |
| `dem_mean_1h` | 2.2 | 1.3–3.6 | 5.6 |
| `cal_month` | 1.4 | 0.2–1.7 | 4.7 |

### Lead band 2 h 5 min–24 h

| Feature | Gain share, mean (%) | Range over retrains (%) | Split share, mean (%) |
| :--- | ---: | ---: | ---: |
| `dem_same_time_yesterday` | 26.5 | 5.4–56.8 | 7.8 |
| `dem_now_vs_yesterday` | 10.8 | 0.5–42.7 | 3.7 |
| `dem_yesterday_vs_last_week` | 7.9 | 0.7–17.5 | 7.3 |
| `dem_same_time_last_week` | 7.6 | 1.5–15.0 | 6.8 |
| `cal_day_of_week` | 4.9 | 2.2–9.1 | 7.1 |
| `cal_slot` | 4.9 | 2.3–6.4 | 7.8 |
| `wx_temp_single_site` | 4.8 | 3.5–6.9 | 5.8 |
| `cal_month` | 4.1 | 2.0–6.7 | 9.1 |
| `cal_non_working` | 3.6 | 2.6–5.1 | 2.4 |
| `wx_temp` | 3.2 | 1.8–4.8 | 4.5 |
| `wx_heating_degrees` | 3.2 | 1.5–5.1 | 4.1 |
| `dem_mean_24h` | 3.1 | 0.7–6.8 | 8.5 |

## MAE (MW) by segment, all lead times pooled

Weather and market-state segments are defined by what actually happened (*ex-post*); they show where models fail, not a way to choose one.

| Segment | Pairs | LightGBM A | LightGBM C | LightGBM D | Ridge C | Persistence | Daily naive |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All target intervals | 14,436,480 | 332.6 | 267.5 | **263.0** | 319.4 | 935.4 | 476.7 |
| Working days (not weekend or public holiday) | 9,877,584 | 333.2 | 270.0 | **263.5** | 325.4 | 929.7 | 463.1 |
| Evening peak (local 17:00-21:00) | 2,404,920 | 374.4 | 261.7 | **251.5** | 322.6 | 1,204.0 | 494.4 |
| Saturday/Sunday | 4,123,866 | 319.1 | **252.0** | 253.0 | 300.2 | 946.8 | 498.0 |
| Victorian public holidays | 545,622 | 409.6 | 327.3 | **322.7** | 340.6 | 931.2 | 527.2 |
| Local days when clocks change (23 h / 25 h days) | 69,696 | 213.7 | **202.9** | 210.3 | 222.5 | 848.9 | 305.6 |
| No price spike, negative price or market intervention | 11,238,577 | 286.8 | 227.0 | **222.5** | 274.4 | 917.4 | 423.6 |
| Realised price > $300/MWh (ex-post) | 141,936 | 519.2 | 384.8 | **357.9** | 495.1 | 1,477.3 | 539.1 |
| Realised price < $0/MWh (ex-post) | 3,043,247 | 493.4 | 412.2 | **408.5** | 477.9 | 977.9 | 669.6 |
| Administered pricing or market suspension (ex-post) | 14,976 | 239.1 | 195.5 | **162.0** | 204.6 | 648.5 | 472.2 |
| Local days whose weighted ERA5 maximum temperature is at least 35 °C (ex-post) | 317,952 | 899.0 | 470.0 | **438.7** | 480.1 | 1,342.3 | 906.5 |
| Local days whose weighted ERA5 cloud cover averages at least 80% over 09:00-17:00 local (ex-post) | 5,166,486 | 331.4 | **272.5** | 274.1 | 322.6 | 866.2 | 500.0 |

## Failure cases

The 10 local days with the highest LightGBM C MAE at lead times 2 h–24 h (S1's range), among days with at least 90% of the usual number of scored pairs (1,042 of 1,047 days). Bias is forecast − actual. *C temp forecast error* is the mean absolute difference between C's forecast temperature and ERA5 on that day's pairs.

| Rank | Local date | Day | MAE LightGBM C | MAE LightGBM A | MAE Daily naive | Bias C (MW) | ERA5 max temp (°C) | C temp forecast error (°C) |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2024-12-17 | Tuesday | 1,254.4 | 1,139.5 | 2,197.2 | +695 | 23.7 | 0.8 |
| 2 | 2024-12-16 | Monday | 1,142.5 | 1,872.6 | 2,119.2 | -982 | 38.6 | 1.9 |
| 3 | 2024-12-27 | Friday | 1,042.0 | 1,230.2 | 1,313.0 | +1,018 | 22.0 | 0.8 |
| 4 | 2025-01-28 | Tuesday | 999.4 | 849.6 | 1,134.9 | +728 | 22.6 | 0.7 |
| 5 | 2025-02-06 | Thursday | 826.4 | 1,194.6 | 1,349.8 | -824 | 34.7 | 1.6 |
| 6 | 2025-01-27 | Australia Day | 791.0 | 959.4 | 1,381.3 | -467 | 39.4 | 1.8 |
| 7 | 2025-02-04 | Tuesday | 746.4 | 1,103.6 | 893.2 | -700 | 35.8 | 1.0 |
| 8 | 2024-06-13 | Thursday | 736.9 | 630.8 | 641.4 | -737 | 10.8 | 1.8 |
| 9 | 2025-07-25 | Friday | 736.5 | 778.5 | 856.0 | -637 | 11.1 | 0.9 |
| 10 | 2023-03-06 | Monday | 735.4 | 654.4 | 527.3 | +734 | 24.5 | 1.5 |

**Known events** (rank among all complete days; listed whatever their rank):

| Event | Rank | Local date | Day | MAE LightGBM C | MAE LightGBM A | MAE Daily naive | Bias C (MW) | ERA5 max temp (°C) | C temp forecast error (°C) |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Severe storms: transmission towers down, widespread outages | 42 | 2024-02-13 | Tuesday | 538.0 | 898.6 | 1,079.5 | +351 | 36.5 | 1.6 |
| Short afternoon demand spike, cause unverified (m2_handoff.md) | 289 | 2024-12-09 | Monday | 329.0 | 271.5 | 789.9 | -181 | 21.1 | 1.3 |

## Notes and limitations

- S1–S4 judge LightGBM C only. Ridge, C⁺/D⁺, the delay check and lead times under 2 h are reported, not pass/fail.
- All of this is the development period, inside which the settings were selected (each retrain sees only data before it). The holdout is scored once, in step 6.
- Headline C uses forecast temperature only: other forecast variables are archived only from 2024-01-19 (see C⁺). Pairs without C weather (the GFS archive gap) are excluded for every model.
- The drop/add test runs at a few retrains on their early-stopping slices, not as a full walk-forward per feature.
- `TOTALDEMAND` excludes rooftop PV, so cloudy and sunny days shift midday demand in ways temperature alone cannot explain.

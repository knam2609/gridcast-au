# Feature catalogue (M2)

> **Generated** by `gridcast feature-catalogue` from `src/gridcast/features/catalogue.py`. Do not edit by hand. A unit test fails if this file is out of date. Measured coverage and missing rates are in the generated [`reports/m2_feature_audit.md`](../reports/m2_feature_audit.md).

## Notation

- *o*: issue time (every 30 minutes). *h*: horizon, 1…288 five-minute steps. *t* = *o* + 5 min × *h*: the target interval's end.
- *c* = *o* − latency: the data cutoff. Intervals ending at or before *c* are known at *o*. The configured latency is 0 intervals, so *c* = *o*.
- y(·): Victorian total demand (MW) of the interval ending at that time.
- Times are UTC on a complete 5-minute grid. Local (Australia/Melbourne) time is used only for calendar features.

## Variants

C and D have identical feature names, and so do C⁺ and D⁺: only the weather source differs, so a C-vs-D gap measures the cost of weather-forecast error.

| Variant | Weather | Feature groups | Features | First issue (NEM date) | Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A | none | — | 21 | 2022-11-01 | No weather |
| C | GFS forecast (`gfs_seamless`) | Weather at the target | 26 | 2022-11-01 | Archived GFS temperature forecasts (headline) |
| D | ERA5 (`era5`) | Weather at the target | 26 | 2022-11-01 | ERA5 temperature: perfect-foresight mirror of C |
| C⁺ | GFS forecast (`gfs_seamless`) | Weather at the target, Extra weather at the target | 30 | 2025-02-01 | All archived GFS variables (secondary) |
| D⁺ | ERA5 (`era5`) | Weather at the target, Extra weather at the target | 30 | 2025-02-01 | ERA5 mirror of C⁺ |

Every variant also has the base groups: lead time, calendar (target and 24 h earlier) and demand (issue time and seasonal lags).

## Features

### Lead time

- **Source:** Forecast design.
- **Known:** By construction.
- **Why it is safe:** Depends only on h.
- **Variants:** A, C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `lead_minutes` | 5 × h, for h = 1…288 | none |

### Calendar at the target

- **Source:** `calendar_days` (Victorian public holidays, `holidays` package) and the local clock (Australia/Melbourne) at the target interval's start.
- **Known:** In advance, for any date in `calendar_days`.
- **Why it is safe:** Calendar attributes are fixed long before t. Building the context fails if a target date is missing from `calendar_days`, so a stale calendar cannot silently mark a holiday as a working day.
- **Variants:** A, C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `cal_slot` | Local 5-minute slot of the day (0–287) | calendar |
| `cal_day_of_week` | Local day of the week (0 = Monday) | calendar |
| `cal_month` | Local month (1–12) | calendar |
| `cal_non_working` | 1 on weekends and public holidays | calendar |
| `cal_public_holiday` | 1 on Victorian public holidays | calendar |
| `cal_dst_change_day` | 1 on local days of 23 or 25 hours (clock change) | calendar |
| `cal_tod_sin` | sin(2π × local minute of day / 1440) | calendar |
| `cal_tod_cos` | cos(2π × local minute of day / 1440) | calendar |

### Calendar 24 hours before the target

- **Source:** As above, for the interval ending at t − 24 h (the interval that supplies `dem_same_time_yesterday`).
- **Known:** In advance.
- **Why it is safe:** Calendar attributes only.
- **Variants:** A, C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `cal_yesterday_non_working` | `cal_non_working` of the interval at t − 24 h | calendar |
| `cal_yesterday_public_holiday` | `cal_public_holiday` of the interval at t − 24 h | calendar |

### Demand at the issue time

- **Source:** `market_intervals.total_demand_mw` (AEMO `DISPATCHREGIONSUM.TOTALDEMAND`).
- **Known:** Intervals ending at or before the data cutoff c = o − latency.
- **Why it is safe:** Every window ends at c. Values are computed from the intervals that end in the window and nothing else, so later intervals cannot change them. Any missing interval in a window makes the feature missing.
- **Variants:** A, C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `dem_now` | y(c) | c |
| `dem_now_vs_yesterday` | y(c) − y(c − 24 h) | c − 24 h |
| `dem_change_1h` | y(c) − y(c − 1 h) | c − 1 h |
| `dem_mean_1h` | Mean of the 12 intervals ending in (c − 1 h, c] | c − 55 min |
| `dem_std_1h` | Standard deviation (ddof = 0) of the same 12 | c − 55 min |
| `dem_mean_24h` | Mean of the 288 intervals ending in (c − 24 h, c] | c − 23 h 55 min |
| `dem_std_24h` | Standard deviation (ddof = 0) of the same 288 | c − 23 h 55 min |

### Demand at the same time yesterday and last week

- **Source:** `market_intervals.total_demand_mw` (AEMO `DISPATCHREGIONSUM.TOTALDEMAND`).
- **Known:** t − k × period ≤ c, with the smallest such k ≥ 1 (with latency 0 and h ≤ 288, k = 1: exactly 24 h and 7 days before t).
- **Why it is safe:** The source position is checked against c for every row (`assert_known`); with h ≤ 288 and latency 0, t − 24 h ≤ o always holds. `dem_same_time_yesterday` is also the daily-naive forecast, the base of the residual target.
- **Variants:** A, C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `dem_same_time_yesterday` | y(t − 24 h) | o − 23 h 55 min (h = 1) |
| `dem_same_time_last_week` | y(t − 7 d) | o − 6 d 23 h 55 min (h = 1) |
| `dem_yesterday_vs_last_week` | y(t − 24 h) − y(t − 7 d) | as above |

### Weather at the target

- **Source:** C: GFS forecasts (`weather_forecasts`, Open-Meteo Previous Runs). D: ERA5 reanalysis (`weather_observations`). Weighted over five sites by population.
- **Known:** C: the freshest forecast run with `available_at` ≤ o. D: never at forecast time (ERA5 is published about 5 days late).
- **Why it is safe:** C reads a 5-minute value only if both hours it interpolates between have `available_at` ≤ o (see *Forecast weather selection*). D is a perfect-foresight oracle by construction and is exempt from the weather leakage test.
- **Variants:** C, D, C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `wx_temp` | Population-weighted 2 m temperature (°C) | valid at t |
| `wx_cooling_degrees` | max(`wx_temp` − 18 °C, 0) | valid at t |
| `wx_heating_degrees` | max(18 °C − `wx_temp`, 0) | valid at t |
| `wx_temp_sq` | `wx_temp`² | valid at t |
| `wx_temp_single_site` | 2 m temperature at Melbourne (°C) | valid at t |

### Extra weather at the target

- **Source:** As for *Weather at the target*. The GFS archive of these variables starts 2024-01-19 12:00 UTC (day 1) and 2024-01-20 12:00 UTC (day 2).
- **Known:** As above.
- **Why it is safe:** As above.
- **Variants:** C⁺, D⁺.

| Feature | Definition | Oldest data read |
| :--- | :--- | :--- |
| `wx_apparent_temp` | Weighted apparent temperature (°C) | valid at t |
| `wx_humidity` | Weighted relative humidity at 2 m (%) | valid at t |
| `wx_cloud_cover` | Weighted total cloud cover (%) | valid at t |
| `wx_shortwave` | Weighted shortwave radiation (W/m²), mean over the hour ending ceil₁ₕ(t) | hour ending ceil₁ₕ(t) |

## Forecast weather selection (C and C⁺)

1. Each archived GFS value (`gfs_seamless`, offsets day 1, day 2) carries `run_init` = floor_6h(valid time − 24 h × offset) and `available_at` = `run_init` + 6 h (verified in step 0, `docs/m2_data_spike.md` §3).
2. Hourly values are weighted over the sites, then interpolated to 5 minutes. A 5-minute value exists only if both hours it uses exist, and it is available from the **later** hour's `available_at`.
3. For each pair (*o*, *t*), the builder takes the **freshest offset** whose every required value at *t* exists and has `available_at` ≤ *o*. All weather features of a pair come from that one offset. If no offset qualifies, the pair has no C weather and is excluded for every model and baseline (the GFS gap rule, m2_plan.md §2.4).
4. A sensitivity run can recompute `available_at` with a longer delay (`publication_delay` override). The headline uses the configured delay.

## Target

The models predict the correction to the same time yesterday, r = y(*t*) − `dem_same_time_yesterday`, and forecast `dem_same_time_yesterday` + r̂. Step 3 also tries predicting y(*t*) directly as an ablation on development data.

## Leakage audit

Property tests (`tests/unit/test_feature_leakage.py`) rebuild the features after perturbing data that is unknown at *o* and require bit-for-bit identical results:

- **Future demand:** every value after *c* is replaced by noise. All five variants, including D and D⁺, must be unchanged.
- **Unpublished forecasts:** every hourly forecast row with `available_at` > *o* is replaced by noise, or deleted. C and C⁺ must be unchanged. This covers the later hour an interpolated value uses.
- **Controls:** perturbing forecast rows that *are* available changes C, and perturbing future ERA5 changes D. These show that the tests can fail.
- **Exempt:** D and D⁺ under weather perturbation. They are the labelled perfect-foresight oracles.

Rule for steps 3–4: any single feature that improves pooled MAE by more than 15% on its own (added or removed one at a time) is investigated for leakage before the result is believed.

## Deferred

- Weather compared with yesterday's weather. It needs an analysis of yesterday's weather that is available at *o*, and ERA5 is published about 5 days late.

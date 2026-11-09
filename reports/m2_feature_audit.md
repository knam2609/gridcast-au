# M2 feature audit: coverage on the real data

> **Generated** by `gridcast feature-audit` on 2026-11-03 13:40 UTC from the data in PostgreSQL, using the same feature builder as the models. Do not edit by hand. Feature definitions are in [`docs/feature_catalogue.md`](../docs/feature_catalogue.md).

## Scope

| Item | Value |
| :--- | :--- |
| Region | VIC1 |
| Demand data (interval end) | 2021-09-30 14:05 UTC → 2026-09-30 14:00 UTC |
| Development issue times (NEM time) | 2022-11-01 00:00 → before 2025-10-01 00:00, every 30 min, horizons 1–288 |
| C⁺ / D⁺ issue times (NEM time) | from 2025-02-01 00:00 |
| GFS publication delay | 6 h (headline); 8 h (sensitivity) |
| Git commit | `f8cd7fd00e0132e8a573c2b3745d36eeee87c24f` |

## Pairs with every feature

A pair (issue time, horizon) can be scored only if every feature of the variant and the actual demand are present. Pairs that some variant cannot score are excluded for every model and baseline when they are compared (m2_plan.md §2.4).

| Variant | Issue times | Pairs | With every feature | Excluded | Share excluded |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A | 51,120 | 14,722,560 | 14,722,560 | 0 | 0.000% |
| C | 51,120 | 14,722,560 | 14,436,480 | 286,080 | 1.943% |
| D | 51,120 | 14,722,560 | 14,722,560 | 0 | 0.000% |
| C⁺ | 11,616 | 3,345,408 | 3,345,408 | 0 | 0.000% |
| D⁺ | 11,616 | 3,345,408 | 3,345,408 | 0 | 0.000% |
| C (8 h delay) | 51,120 | 14,722,560 | 14,435,328 | 287,232 | 1.951% |

### Months with excluded pairs

| Variant | Issue month (NEM) | Excluded pairs | Share of month | First target | Last target |
| :--- | :--- | ---: | ---: | :--- | :--- |
| C | 2023-12 | 29,232 | 6.82% | 2023-12-29 23:05 UTC | 2024-01-01 13:30 UTC |
| C | 2024-01 | 256,848 | 59.94% | 2023-12-31 14:05 UTC | 2024-01-20 11:55 UTC |
| C (8 h delay) | 2023-12 | 29,232 | 6.82% | 2023-12-29 23:05 UTC | 2024-01-01 13:30 UTC |
| C (8 h delay) | 2024-01 | 258,000 | 60.20% | 2023-12-31 14:05 UTC | 2024-01-20 11:55 UTC |

## Which forecast run C uses

For each pair, C takes the freshest forecast offset whose values are published by the issue time (`docs/feature_catalogue.md`, *Forecast weather selection*). Shares of pairs by lead time:

| Run | Lead time | Day 1 | Day 2 | None (excluded) | Day 1 among pairs with weather |
| :--- | :--- | ---: | ---: | ---: | ---: |
| C | 5 min – 6 h | 98.1% | 0.0% | 1.93% | 100.0% |
| C | 6 h 5 min – 12 h | 98.1% | 0.0% | 1.93% | 100.0% |
| C | 12 h 5 min – 18 h | 97.4% | 0.7% | 1.93% | 99.3% |
| C | 18 h 5 min – 24 h | 37.5% | 60.6% | 1.99% | 38.2% |
| C | **all leads** | 82.7% | 15.3% | 1.94% | 84.4% |
| C (8 h delay) | 5 min – 6 h | 98.1% | 0.0% | 1.93% | 100.0% |
| C (8 h delay) | 6 h 5 min – 12 h | 98.1% | 0.0% | 1.93% | 100.0% |
| C (8 h delay) | 12 h 5 min – 18 h | 87.9% | 10.2% | 1.94% | 89.6% |
| C (8 h delay) | 18 h 5 min – 24 h | 14.3% | 83.7% | 2.01% | 14.6% |
| C (8 h delay) | **all leads** | 74.6% | 23.5% | 1.95% | 76.1% |

## Forecast temperature against ERA5

Weighted temperature of C minus that of D, for the same pairs (both present). This is the weather-forecast error that separates C from its perfect-foresight mirror, plus the GFS-vs-ERA5 grid difference.

The archive keeps only day-1 and day-2 forecasts (a day-0 run is never published before its valid time), so at **every** lead time C uses a forecast made 24–30 h (day 1) or 48–54 h (day 2) before the valid time. The difference therefore changes little with our lead time and grows only where day 2 takes over.

| Lead time | Pairs | Mean absolute difference (°C) | Mean difference (°C) |
| :--- | ---: | ---: | ---: |
| 5 min – 6 h | 3,609,660 | 1.07 | 0.41 |
| 6 h 5 min – 12 h | 3,609,660 | 1.07 | 0.41 |
| 12 h 5 min – 18 h | 3,609,636 | 1.07 | 0.41 |
| 18 h 5 min – 24 h | 3,607,524 | 1.10 | 0.39 |
| **all leads** | 14,436,480 | 1.08 | 0.41 |

**Timing check.** The same difference with ERA5 read at the target time shifted by whole hours. A time-alignment error would make a non-zero shift fit better.

| ERA5 shift | Pairs | Mean absolute difference (°C) |
| :--- | ---: | ---: |
| -1 h | 14,436,480 | 1.307 |
| +0 h | 14,436,480 | 1.076 |
| +1 h | 14,436,480 | 1.286 |

Smallest difference at **+0 h** (aligned).

## Feature ranges and missing values

Base and weather features from C over the development issue times, the extra weather features from C⁺ over its window, and the ERA5 weather features from D and D⁺.

| Feature | From | Missing | Min | Mean | Max |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `lead_minutes` | C | 0.000% | 5.00 | 722.50 | 1,440.00 |
| `cal_slot` | C | 0.000% | 0.00 | 143.50 | 287.00 |
| `cal_day_of_week` | C | 0.000% | 0.00 | 3.00 | 6.00 |
| `cal_month` | C | 0.000% | 1.00 | 6.42 | 12.00 |
| `cal_non_working` | C | 0.000% | 0.00 | 0.32 | 1.00 |
| `cal_public_holiday` | C | 0.000% | 0.00 | 0.04 | 1.00 |
| `cal_dst_change_day` | C | 0.000% | 0.00 | 0.00 | 1.00 |
| `cal_tod_sin` | C | 0.000% | -1.00 | 0.00 | 1.00 |
| `cal_tod_cos` | C | 0.000% | -1.00 | 0.00 | 1.00 |
| `cal_yesterday_non_working` | C | 0.000% | 0.00 | 0.32 | 1.00 |
| `cal_yesterday_public_holiday` | C | 0.000% | 0.00 | 0.04 | 1.00 |
| `dem_now` | C | 0.000% | 1,433.27 | 4,760.80 | 9,887.28 |
| `dem_now_vs_yesterday` | C | 0.000% | -5,436.81 | 0.59 | 5,349.10 |
| `dem_change_1h` | C | 0.000% | -1,982.43 | 0.02 | 2,468.47 |
| `dem_mean_1h` | C | 0.000% | 1,451.24 | 4,763.01 | 9,596.76 |
| `dem_std_1h` | C | 0.000% | 7.09 | 106.76 | 1,065.45 |
| `dem_mean_24h` | C | 0.000% | 2,985.01 | 4,762.75 | 6,843.37 |
| `dem_std_24h` | C | 0.000% | 225.02 | 796.72 | 2,134.46 |
| `dem_same_time_yesterday` | C | 0.000% | 1,400.13 | 4,762.75 | 9,887.28 |
| `dem_same_time_last_week` | C | 0.000% | 1,400.13 | 4,762.00 | 9,887.28 |
| `dem_yesterday_vs_last_week` | C | 0.000% | -6,322.33 | 0.76 | 5,649.28 |
| `wx_temp` | C | 1.943% | 3.34 | 15.52 | 42.16 |
| `wx_cooling_degrees` | C | 1.943% | 0.00 | 1.36 | 24.16 |
| `wx_heating_degrees` | C | 1.943% | 0.00 | 3.84 | 14.66 |
| `wx_temp_sq` | C | 1.943% | 11.13 | 274.38 | 1,777.65 |
| `wx_temp_single_site` | C | 1.943% | 3.50 | 15.71 | 42.30 |
| `wx_apparent_temp` | C⁺ | 0.000% | -0.46 | 12.43 | 40.26 |
| `wx_humidity` | C⁺ | 0.000% | 12.01 | 63.41 | 97.41 |
| `wx_cloud_cover` | C⁺ | 0.000% | 0.00 | 57.25 | 100.00 |
| `wx_shortwave` | C⁺ | 0.000% | 0.00 | 149.29 | 1,028.88 |
| `wx_temp (ERA5)` | D | 0.000% | -0.23 | 15.22 | 39.40 |
| `wx_cooling_degrees (ERA5)` | D | 0.000% | 0.00 | 1.26 | 21.40 |
| `wx_heating_degrees (ERA5)` | D | 0.000% | 0.00 | 4.05 | 18.23 |
| `wx_temp_sq (ERA5)` | D | 0.000% | 0.00 | 265.25 | 1,552.34 |
| `wx_temp_single_site (ERA5)` | D | 0.000% | -0.70 | 15.31 | 39.80 |
| `wx_apparent_temp (ERA5)` | D⁺ | 0.000% | -1.02 | 12.20 | 38.35 |
| `wx_humidity (ERA5)` | D⁺ | 0.000% | 17.59 | 72.19 | 99.54 |
| `wx_cloud_cover (ERA5)` | D⁺ | 0.000% | 0.00 | 54.67 | 100.00 |
| `wx_shortwave (ERA5)` | D⁺ | 0.000% | 0.00 | 146.30 | 1,019.68 |

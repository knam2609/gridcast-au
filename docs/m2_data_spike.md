# M2 step 0: data-availability check

Run on 2026-11-02, before any M2 model code or results existed. Every number below is copied from the output of the scripts in [`scripts/spikes/m2/`](../scripts/spikes/m2/) (see §9 to re-run them). Open-Meteo and NEMWeb archives can be revised, so a later re-run may differ slightly.

All times are UTC unless marked *NEM time* (AEST, fixed UTC+10). The study window is Oct 2021 → Sep 2026.

## Summary

| Question (from the handoff) | Answer | Consequence |
|---|---|---|
| 1. Which GFS variables are archived in Previous Runs, and from when? | `temperature_2m` from **2021-03-24**. The other five planned variables only from **2024-01-19 12:00**. Temperature has one gap, 2023-12-30 → 2024-01-19/20 (1.1–1.2% of the window) | Agreed fallback 2 applies: headline C is **temperature only**, and C⁺ uses all variables from 2024-01-19 |
| 2. Which run supplies `previous_day1/2`, and how late is it published? | GFS `previous_dayK` at hour *t* comes **exactly** from the run initialised at **floor₆ₕ(t − 24K h)** (100% of tested hours). Open-Meteo published a GFS run 5.4–5.5 h after initialisation | `available_at = run_init + 6 h`, computed per row. day1 is usable for 84.4% of (issue time, lead) pairs (74.0% under the plan's simpler bound) |
| 3. What do pre-dispatch "ALL" files contain? | 48 runs per day. `available_at = LASTCHANGED` (2–4 min after the nominal run time). The horizon runs to the end of the next trading day. **`TOTALDEMAND` is the 30-minute mean**, not the period-end value | Compare on 30-minute means, as planned. 64.6% of runs reach 24 h ahead |
| 4. Is 5-minute pre-dispatch a manageable size? | 1.25 GB for 24 months, about the same as 30-minute pre-dispatch (1.22 GB) | Included (agreed rule) |
| (found on the way) | Both AEMO tables changed columns mid-window **without** a version bump, then changed version **without** a column change | The loader maps columns by name and accepts both versions |

The decisions taken from these findings are in §7.

## 1. Open-Meteo Previous Runs: coverage (Melbourne, −37.83, 144.98)

The query covered 2021-01-01 → 2026-10-31 with offsets 0–2 for each model. "From" is the first non-missing `previous_day1` hour.

| Model (`models=`) | Grid point | `temperature_2m` from | Apparent temp., humidity, cloud, radiation, wind from | Notes |
|---|---|---|---|---|
| `gfs_seamless` | (−37.78, 144.96) | **2021-03-24** | **2024-01-19 12:00** | Same series as `gfs_global` here |
| `jma_seamless` | (−38.00, 145.00) | ≤ 2021-01-01 (query start) | ≤ 2021-01-01, except **no shortwave radiation at all** | Humidity and cloud day1 are missing 2026-04-13 18:00 → 04-26 23:00 (318 h) |
| `ecmwf_ifs025` | (−37.75, 145.00) | 2024-02-04 | 2024-02-04 to 2024-03-06, varying by variable | |
| `icon_seamless`, `gem_seamless` | (−37.75, 145.00), (−37.80, 145.05) | 2024-01-19 | 2024-01-19 | |
| `bom_access_global` | (−37.79, 144.93) | 2024-01-19 | 2024-01-19 | Ends 2025-07-06, so only 52.5% of hours are covered after the start |

**GFS temperature gap.** Inside the study window there is exactly one gap: `previous_day1` is missing from 2023-12-30 00:00 to 2024-01-19 11:00 (492 h, 1.12%), and `previous_day2` from 2023-12-30 00:00 to 2024-01-20 11:00 (516 h, 1.18%). Latrobe Valley (−38.23, 146.40) has the identical gap, so the outage affects the whole archive, not one site. The gap ends at the hour the multi-variable archive begins, which suggests it is the seam between an older temperature-only archive and the current one. It is under the agreed 2% stop threshold.

**Are these real past forecasts?** Yes, by two checks. In each year, GFS `previous_day1` equals `previous_day0` for only 7.4–8.3% of hours. The mean absolute difference from day0 grows from day1 (0.55–0.61 °C) to day2 (0.73–0.81 °C). Both figures are stable from 2021 to 2026, including across the 2024 seam, so the older temperature archive behaves like the current one. JMA passes the same checks.

## 2. Forecast skill against ERA5 (Melbourne)

ERA5 (Historical Weather API, `models=era5`, grid point −37.75, 145.00) is complete for the window, with 0 missing values. ERA5-Land has only temperature and humidity among the planned variables, so variant D uses ERA5.

**Day1 temperature, MAE against ERA5 (°C).** 2021 covers Oct–Dec only, and 2026 covers Jan–Sep.

| Model | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Bias |
|---|---|---|---|---|---|---|---|
| GFS | 1.12 | 1.18 | 1.12 | 1.26 | 1.20 | 1.20 | +0.49 |
| JMA | 1.91 | 2.01 | 1.99 | 2.19 | 2.20 | 2.04 | +0.35 |
| ECMWF IFS 0.25° | – | – | – | 0.90 | 0.89 | 0.80 | −0.17 |
| ICON | – | – | – | 0.99 | 1.07 | 1.08 | +0.54 |

**Other day1 variables, MAE against ERA5 (range of yearly values).**

| Variable | GFS (2024+) | JMA (2021+) | ECMWF (2024+) | ICON (2024+) |
|---|---|---|---|---|
| Relative humidity (%) | 10.60–11.75, bias −10.00 | 7.82–8.42 | 4.44–4.85 | 5.68–5.87 |
| Cloud cover (%) | 24.54–27.79 | 25.53–27.11, bias −10.28 | 19.05–28.16 | 21.67–22.94 |
| Shortwave radiation (W/m²) | 35.81–40.39 | not archived | 22.27–25.95 | 27.65–29.72 |

Caveats:

- ERA5 is a reanalysis built with ECMWF's IFS, so the ECMWF scores are flattered by sharing its parent model.
- This is one site, compared against a 0.25° reanalysis, not station observations.
- The point is to rank sources for the pre-2024 period. The scores are not estimates of true forecast error.

## 3. Run timing and publication delay

**GFS.** For 168 valid hours (2026-09-28 → 10-04), each archived `previous_day1` and `previous_day2` value was compared with 48 individual 6-hourly runs from the Single Runs API. Temperature and cloud cover had to match together. The run initialised at **floor₆ₕ(t − 24K h)** reproduced the archived value exactly in **100%** of hours, for both K = 1 and K = 2. The run age is therefore 24–29 h for day1 and 48–53 h for day2.

The model ids need care:

- In the Single Runs API, `gfs_seamless` resolves to HRRR, a US-only model.
- `gfs_global` resolves to `ncep_gfs013`, the same grid point that Previous Runs `gfs_seamless` returns for Melbourne.
- The Single Runs archive keeps only recent months; runs from early 2026 and before were not available.

**Publication delay.** Open-Meteo's metadata for `ncep_gfs013` reported its latest run (initialised 2026-11-01 18:00) as available 5.39–5.49 h after initialisation, across three reads. That is a single run: the API exposes no history of past runs' publication times. The agreed 6 h delay therefore leaves roughly 0.5 h of margin over this observation. NOAA's own GFS feed is typically earlier, but that was not measured here.

**JMA is not timing-safe.** JMA GSM's native step is 6 h. Archived values match the floor₆ₕ rule run exactly at 00/06/12/18 UTC, but deviate between those hours. The mean deviation by hour is up to 0.34 °C for day1 and 1.11 °C for day2. For 69.6% of day1 hours, no single run matches at all. The intermediate hours are therefore interpolated, possibly across runs, so a JMA value cannot be tied to one run's publication time. JMA's own publication delay in the metadata was 9.71 h.

**How often day1 can be used** under `available_at = run_init + 6 h`. A 5-minute target interpolates between two hourly values, so the later hour, ceil₁ₕ(t), must be available. Shares are computed over every issue time in a day × every lead from 5 min to 24 h.

| Lead time (h) | day1 usable, exact rule | day1 usable, plan's simple bound (`t − 24 h + 6 h`) | day2 usable |
|---|---|---|---|
| (0, 12] | 100.0% | 100.0% | 100% |
| (12, 18] | 99.3% | 95.8% | 100% |
| (18, 20] | 70.8% | 0.0% | 100% |
| (20, 22] | 37.5% | 0.0% | 100% |
| (22, 23] | 12.5% | 0.0% | 100% |
| (23, 24] | 0.0% | 0.0% | 100% |
| **all leads** | **84.4%** | **74.0%** | 100% |

## 4. AEMO pre-dispatch: `PREDISPATCHREGIONSUM`, all runs

**Where and how big.** The files are `MMSDM_YYYY_MM/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01#YYYYMM010000.zip`. They are 44.9–54.3 MB per month, **1.22 GB** for Oct 2024 → Sep 2026. January 2025 unzips to 178 MB. The same table in `DATA/` is about 1 MB per month, far too small to hold every run, so it is not used.

**Structure.** In January 2025, VIC1 has 82,584 rows across 1,488 runs over 31 trading days, i.e. **48 runs per day**.

- `RUNNO` is always 1 and `INTERVENTION` is always 0. There are no duplicate `(PREDISPATCHSEQNO, RUNNO, INTERVENTION, DATETIME)` keys.
- `PREDISPATCHSEQNO` = `YYYYMMDDPP`: the trading day (starting 04:00 NEM time) plus run number PP. The nominal run time is 04:00 + (PP − 1) × 30 min.
- `LASTCHANGED` is identical within a run, and falls **76–241 s (median 134 s)** after the nominal run time. It is used as `available_at`.
- `DATETIME` is the period end. Each run's first period (`PERIODID` 01) ends 30 minutes after the nominal run time.
- **Horizon:** runs from 04:00 to 12:00 NEM time end at the next 04:00, giving 16.0–24.0 h. Runs from 12:30 to 03:30 extend to the end of the following trading day, giving 24.5–39.4 h. **64.6% of runs reach 24 h.**

**`TOTALDEMAND` is the 30-minute mean.** The table shows MAE against actual 5-minute demand, by lead from `LASTCHANGED` to period end, in MW.

| Lead (h) | Periods | vs 30-min mean | vs period-end value | Bias vs mean | Bias vs period end |
|---|---|---|---|---|---|
| (0, 0.5] | 1,488 | **66.8** | 116.8 | +6.5 | +7.5 |
| (0.5, 1] | 1,488 | **103.8** | 142.9 | +9.1 | +10.2 |
| (2, 6] | 11,904 | **177.9** | 197.8 | −0.2 | +0.9 |
| (12, 24] | 31,496 | **210.4** | 224.5 | +8.2 | +7.4 |
| (0, 0.5], steep ramps only (\|30-min change\| ≥ 300 MW) | 204 | **120.7** | 285.1 | −36.6 | −112.6 |

The 30-minute mean wins at every lead, and the gap is widest on ramps. There, a period-end interpretation would show a large, systematic bias.

## 5. AEMO 5-minute pre-dispatch: `P5MIN_REGIONSOLUTION`, all runs

**Where and how big.** The files are in `P5MIN_ALL_DATA/PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#ALL#FILE01#YYYYMM010000.zip`. They are 46.1–55.7 MB per month, **1.25 GB** for the 24 months, about the same as 30-minute pre-dispatch. January 2025 unzips to 249 MB.

**Structure.** In January 2025, VIC1 has 107,136 rows across 8,928 runs, i.e. **288 runs per day**, all with `INTERVENTION` 0.

- Each run has 12 intervals, ending at `RUN_DATETIME`, +5 min, …, +55 min.
- `LASTCHANGED` falls **160–276 s before** `RUN_DATETIME` (median 270 s).

**Interval 0 equals "actual" demand exactly (MAE 0.0 MW).** `DISPATCHREGIONSUM.TOTALDEMAND`, the demand used as the target since M1, is AEMO's dispatch demand target, fixed before the interval starts. It is AEMO's operational demand figure, not metered consumption. This is worth one sentence in the README.

**Alignment with our issue times.** At issue time *o*, the run with `RUN_DATETIME = o + 5 min` was published after *o* (24–140 s after) in every case in January 2025. The freshest available run is therefore always `RUN_DATETIME = o`. Its interval 0 ends at *o*, which is not a target. Its intervals 1–11 are our horizons h = 1…11 (5–55 min ahead), with MAE rising from 39.6 to 134.4 MW. P5MIN does not cover h = 12 (60 min).

## 6. Schema changes inside the benchmark window

These come from each month's MMS header row, read from the first 64 KB of each zip.

| Month | Table | Version | Change |
|---|---|---|---|
| 2024-10 | `PREDISPATCHREGIONSUM` / `P5MIN_REGIONSOLUTION` | 8 / 9 | baseline: 125 / 117 columns |
| **2025-04** | both | 8 / 9, **unchanged** | **+2 columns** appended: `BDU_INITIAL_ENERGY_STORAGE`, `DECGEN_INITIAL_ENERGY_STORAGE` |
| **2025-06** | both | **9 / 10** | **no column change** |

Every column M2 needs is present in all months. Pre-dispatch needs `PREDISPATCHSEQNO`, `RUNNO`, `REGIONID`, `PERIODID`, `INTERVENTION`, `TOTALDEMAND`, `LASTCHANGED` and `DATETIME`. P5MIN needs `RUN_DATETIME`, `INTERVAL_DATETIME`, `REGIONID`, `INTERVENTION`, `TOTALDEMAND` and `LASTCHANGED`.

## 7. Decisions (agreed 2026-11-02)

The agreed fallback rule for "only temperature archived before 2024" applies. The user then decided the four questions it left open:

| Topic | Decision | Why |
|---|---|---|
| Headline C weather | **GFS `temperature_2m` only**, plus features derived from it, over the full window. **C⁺** uses all GFS variables from 2024-01-19 and is reported as secondary. JMA is not used | GFS temperature is much more accurate than JMA (§2), and JMA values cannot be tied to one run (§3) |
| GFS gap, 2023-12-30 → 2024-01-20 | C's training rows without weather are dropped. Pairs where C has no weather are **excluded from scoring for every model and baseline**, and the count is reported | Keeps every comparison on identical pairs, without mixing in a second model |
| `available_at` | **Exact run time + 6 h:** `available_at = floor₆ₕ(t − 24K h) + 6 h`. The builder takes the freshest K with `available_at ≤ o` | The run time is known exactly (§3). The simple bound would discard forecasts that were genuinely available |
| Variant D | **D mirrors headline C** (ERA5 temperature and the same derived features) and is used for S4. **D⁺** mirrors C⁺ (all ERA5 variables, same training window as C⁺) and is reported only | Only the weather source differs between C and D, so the gap isolates forecast error |
| 5-minute pre-dispatch | **Included** in the AEMO benchmark (h = 1…11) | It meets the agreed "manageable size" rule: 1.25 GB, about the same as 30-minute pre-dispatch |

## 8. What this means for later steps

**Step 1 (weather ingestion).**

- Store `run_init` as well as `available_at` on every forecast row, and validate the floor₆ₕ rule on load.
- A data-quality check should fail on any GFS temperature gap other than the known seam (2023-12-30 00:00 → 2024-01-20 11:00).
- C⁺ variables start at 2024-01-19 12:00 (day1) and 2024-01-20 12:00 (day2).

**Step 2 (features).**

- An interpolated 5-minute value is available only when **both** hourly values it uses are available. For shortwave radiation (C⁺), use the value for the hour ending ceil₁ₕ(t).
- Within a target, prefer the freshest available offset. If its value is missing, fall back to the next older available offset. If none is left, the pair has no C weather and is excluded (the gap rule).
- The leakage property tests should perturb rows with `available_at > o`, including the later bracketing hour.

**Step 3 (models).** C⁺ and D⁺ start at **2025-02-01**. That is the first quarterly retrain with at least 12 months of C⁺ history, the same rule as the main models. This gives 8 development months, Feb → Sep 2025.

**Step 5 (AEMO benchmark).**

- Map columns by name. Accept `PREDISPATCHREGIONSUM` v8–9 and `P5MIN_REGIONSOLUTION` v9–10, and fail on any other version or on missing required columns.
- Keep only rows with `INTERVENTION = 0`, and fail on duplicate keys. Set `available_at = LASTCHANGED`.
- Compare pre-dispatch on 30-minute means. Compare P5MIN intervals 1–11 at the 5-minute level.
- Download the archive with the same `listing.json` cache as M1 (no globbing on the exFAT drive).

**Suggestion (adopted in step 2, 2026-11-04).** The 6 h delay has about 0.5 h of margin over the one observed run. A cheap robustness check would be to rebuild C with an 8 h delay at step 4 and report the change in pooled MAE, without changing the headline. The builder supports the override (`features.delay_sensitivity_hours`), and `reports/m2_feature_audit.md` already shows how the offsets used change with 8 h.

## 9. Re-running

The scripts are standalone and cache downloads under the directory you pass them. `aemo_forecasts.py` also needs the database (`make db-up`).

```bash
C=$GRIDCAST_DATA_ROOT/spikes/m2        # or any scratch directory
uv run python scripts/spikes/m2/weather_coverage.py   $C   # §1 coverage (≈45 Open-Meteo calls)
uv run python scripts/spikes/m2/weather_quality.py    $C   # §1 gaps and genuineness, §2 skill vs ERA5
uv run python scripts/spikes/m2/weather_run_timing.py $C   # §3 run matching, publication delay, offset usage
uv run python scripts/spikes/m2/aemo_archive.py            # §4–6 sizes and schema history (HEAD/Range only)
uv run python scripts/spikes/m2/aemo_forecasts.py     $C   # §4–5 one month (≈100 MB download)
```

`weather_run_timing.py` depends on the Single Runs archive, which keeps only recent months. Once Sep–Oct 2026 drops out of it, the run-matching window in the script has to be moved forward.

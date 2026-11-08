# Data sources

Each source used by GridCast-AU is documented below: where it comes from, its licence, frequency, schema, timestamp conventions, missing-data characteristics and known limitations. Sources marked *planned* are documented ahead of implementation.

| Source | Used for | Status |
|---|---|---|
| AEMO MMSDM monthly archive (NEMWeb): `DISPATCHREGIONSUM` | 5-min regional demand | Milestone 1 |
| AEMO MMSDM monthly archive (NEMWeb): `DISPATCHPRICE` | 5-min regional spot price (RRP) | Milestone 1 |
| `holidays` Python package (subdivision VIC) | Public-holiday calendar | Milestone 1 |
| AEMC schedule of reliability settings | Market price cap/floor (QA bounds only) | Milestone 1 |
| Open-Meteo Previous Runs API (GFS) | Archived weather forecasts (variant C) | Milestone 2 |
| Open-Meteo Historical Weather API (ERA5) | Reanalysis "actual" weather (variant D, oracle) | Milestone 2 |
| ABS Regional population 2024-25 | Population weights for weather sites | Milestone 2 |
| AEMO MMSDM `DISPATCH_UNIT_SCADA` / `ROOFTOP_PV_ACTUAL` | Generation mix | *planned* |
| AEMO MMSDM `PREDISPATCHREGIONSUM` / `P5MIN_REGIONSOLUTION` (all runs) | Benchmark vs AEMO's own forecasts | *planned (M2 step 5)* |

---

## 1. AEMO MMSDM monthly archive (NEMWeb)

**URL pattern**

```
https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{YYYY}/MMSDM_{YYYY}_{MM}/MMSDM_Historical_Data_SQLLoader/DATA/<file>.zip
```

The file naming convention changed over time. Both forms are handled by discovering files from the directory listing with an anchored regular expression (`src/gridcast/ingestion/nemweb.py`):

| Period (observed) | Example |
|---|---|
| older | `PUBLIC_DVD_DISPATCHREGIONSUM_202110010000.zip` |
| newer | `PUBLIC_ARCHIVE#DISPATCHREGIONSUM#FILE01#202501010000.zip` (served URL-encoded as `%23`) |

Large tables may be split into `FILE01`, `FILE02`, …; all parts are downloaded. Similarly named tables (`PREDISPATCHREGIONSUM`, `PREDISPATCHPRICE`, …) are explicitly excluded by the regex anchor.

**Licence / terms.** AEMO's [Copyright Permissions](https://aemo.com.au/privacy-and-legal-notices/copyright-permissions) notice gives general permission to use AEMO material for any purpose, provided AEMO is accurately and appropriately attributed as author. Older NEMWeb readme files describe the data as provided "for information only" and "not intended for commercial use". GridCast-AU is a non-commercial portfolio project and attributes AEMO as the data author. Anyone reusing it commercially should confirm terms with AEMO.

> Data source: Australian Energy Market Operator (AEMO), MMS Data Model monthly archive, NEMWeb. © AEMO.

**Access method.** These are published static files on AEMO's public file server, which is NEMWeb's intended use; there is no HTML scraping of web pages. The downloader identifies itself with a project User-Agent, waits between requests (`request_delay_seconds`), retries with exponential backoff, and caches each file locally with a SHA-256 manifest so nothing is downloaded twice. The aggregated price-and-demand CSVs on `aemo.com.au` are **not** used: that host rejects automated requests (HTTP 403), and the MMSDM tables are the authoritative source anyway.

**Frequency and latency.** Each archive covers one calendar month. A month's archive is typically published 2–6 weeks after it ends (e.g. `MMSDM_2026_09` appeared on 12 Oct 2026). The monthly archive is therefore suitable for historical modelling but **not** for real-time forecasting. Live operation (a later milestone) will need the NEMWeb `Reports/Current/DispatchIS_Reports` feed.

**File format (MMS "CSV" report format).** Each row begins with a record type:

| Record | Meaning |
|---|---|
| `C` | Comment: first line is a file header (`C,SETP.WORLD,DVD_DISPATCHPRICE,AEMO,PUBLIC,<date>,…`), last line is `C,"END OF REPORT",<n>` |
| `I` | Column header: `I,<package>,<table>,<version>,<col1>,<col2>,…` |
| `D` | Data row with the same layout as the preceding `I` row |

The table version changes over time, e.g. `DISPATCHPRICE` was version 4 in 2021 (54 columns) and version 5 in 2025 (66 columns). The parser therefore maps columns **by name from the `I` row**, never by position.

### 1a. `DISPATCHREGIONSUM` (demand)

| Column | Type | Used as | Notes |
|---|---|---|---|
| `SETTLEMENTDATE` | datetime | `interval_end` | End of the 5-min dispatch interval, NEM time |
| `RUNNO` | int | dedup tiebreak | Always `1` in the archive samples inspected |
| `REGIONID` | text | `region_id` | `NSW1`, `QLD1`, `SA1`, `TAS1`, `VIC1` |
| `INTERVENTION` | 0/1 | filter | Keep `0`, the pricing run (see limitations) |
| `TOTALDEMAND` | MW | **target: `total_demand_mw`** | Regional demand met by scheduled/semi-scheduled generation and net imports (see limitations) |
| `NETINTERCHANGE` | MW | `net_interchange_mw` | Positive = net export from the region |
| `AVAILABLEGENERATION` | MW | `available_generation_mw` | Bid availability of scheduled/semi-scheduled units |
| `LASTCHANGED` | datetime | dedup tiebreak | Record write time |

The table has ~114 columns (mostly FCAS quantities); only the above are kept.

### 1b. `DISPATCHPRICE` (price)

| Column | Type | Used as | Notes |
|---|---|---|---|
| `SETTLEMENTDATE` | datetime | `interval_end` | As above |
| `RUNNO`, `REGIONID`, `INTERVENTION`, `LASTCHANGED` | | as above | |
| `RRP` | $/MWh | **target: `rrp_aud_mwh`** | Regional reference price. Can be negative (floor −$1,000/MWh) |
| `APCFLAG` | 5-bit integer | `apc_flag_bits` | **Bit field, not a boolean** (see below) |
| `MARKETSUSPENDEDFLAG` | NUMBER(3,0) | `market_suspended_flag` | Non-zero = market suspended |
| `PRICE_STATUS` | text | QA only | `FIRM` once prices are final |

`APCFLAG` records which post-processing steps changed the original NEMDE price (ROP), per AEMO's MMS Data Model Report (table `DISPATCHPRICE`):

| Bit value | Meaning | Decoded column |
|---|---|---|
| 1 | Administered Price Cap/Floor binding | `administered_price_binding` |
| 2 | VoLL override applied | (raw bits only) |
| 4 | Market Price Cap/Floor binding | `market_cap_floor_binding` |
| 8 | Price manually overwritten | `price_manually_overwritten` |
| 16 | Price scaled by the inter-regional loss factor | (raw bits only) |

The first full ingest treated `APCFLAG` as 0/1 and **rejected 15 months** that contained values such as 4, 8, 20 and 24. Migration `0002` fixed this: it stores the raw code and decodes the documented bits into generated columns.

### Timestamp conventions (critical)

- `SETTLEMENTDATE` is the **end** of the interval: the interval labelled `2021/10/01 00:05:00` covers 00:00–00:05.
- All AEMO market times are **NEM time = AEST = UTC+10 all year, with no daylight saving**, even though Victoria observes AEDT in summer.
- GridCast-AU parses the naive timestamp, attaches a fixed `+10:00` offset, and stores `timestamptz` (UTC) values for both `interval_start` and `interval_end`. Local (Australia/Melbourne) time is derived **only** for calendar features, from `interval_start`.
- A monthly file holds intervals ending `YYYY-MM-01 00:05` through `YYYY-(MM+1)-01 00:00` (NEM time): 288 × days-in-month rows per region per table. The October 2021 file has exactly 8,928 = 31 × 288 rows per region.

### Missing-data characteristics

Measured on the full VIC1 load (Oct 2021 → Sep 2026, 60 monthly archives):

- **525,888 rows**, exactly 1,826 days × 288. There are zero missing intervals, zero duplicate keys in the source files, and zero intervals with demand but no price (or the reverse). Every price is `FIRM`.
- Table versions changed during the period: `DISPATCHREGIONSUM` v5 → v8 → v9 and `DISPATCHPRICE` v4 → v5. Columns are mapped by name, so this needed no code changes.
- **Intervention runs** (`INTERVENTION = 1`, dropped in favour of the pricing run) appear in 6 months: Feb 2022 (108 rows), Jun 2022 (5,344), Jul 2022 (196), Nov 2022 (1,242), Feb 2023 (60) and Nov 2024 (24).
- **Administered pricing** (bit 1) applied to 311 intervals, all between 13 and 15 June 2022 (UTC). **Market suspension** applied to 2,903 intervals: 2,591 in June 2022, 296 in April 2023 and 16 in September 2024.
- Other `APCFLAG` bits: market price cap/floor binding 184 intervals, inter-regional loss factor scaling 1,175, manual overwrite 18, VoLL override 10.
- Ranges: demand 1,225–10,784 MW; price −$1,000 (the floor) to $19,069.69/MWh. No price exceeds its financial year's cap. 20.4% of intervals have a negative price and 3.0% exceed $300/MWh.

The data-quality checks that produce these figures are in `sql/checks/`, and their results are written to `dq_results`.

### Known limitations

- **Demand definition.** `TOTALDEMAND` is the demand that dispatch must meet from scheduled and semi-scheduled generation plus net interconnector imports. It **excludes rooftop PV** and most small non-scheduled generation, so midday "demand" falls as rooftop solar grows. Some of what looks like behavioural change is therefore structural. This is not consumer consumption.
- **Interventions.** When AEMO intervenes (directions or RERT), dispatch runs twice. The `INTERVENTION = 0` run sets prices; the physical dispatch follows the intervention run. GridCast uses the `INTERVENTION = 0` row for both demand and price so the two targets stay consistent.
- **Administered pricing and suspension.** During the June 2022 energy crisis, administered price caps applied and AEMO then suspended the spot market. `RRP` in those intervals reflects administered or suspension pricing, not a competitive clearing price. The exact intervals are taken from the `APCFLAG` bit 1 and `MARKETSUSPENDEDFLAG` columns rather than hard-coded dates; they are flagged and evaluated as a separate segment.
- **Price revisions.** Dispatch prices can be revised after publication (e.g. manifestly incorrect inputs). The monthly archive contains final values, so backtests use final prices, which may differ slightly from what a live system would have seen.
- **Publication lag.** See *Frequency and latency* above.

---

## 2. Public holidays: `holidays` Python package

- **Source:** [`holidays`](https://github.com/vacanza/holidays) (MIT licence), `country_holidays("AU", subdiv="VIC")`.
- **Frequency:** static calendar; versioned with the package (`uv.lock` pins it).
- **Schema:** `date → holiday name`. Includes Victoria-specific days: Melbourne Cup Day (first Tuesday in November) and the Friday before the AFL Grand Final.
- **Timestamp convention:** local calendar dates in Australia/Melbourne.
- **Limitations:** Melbourne Cup Day is a public holiday in metropolitan Melbourne, but some regional councils substitute another day. The library also cannot capture school holidays. One-off holidays depend on the library version: the pinned version does include the 22 Sep 2022 National Day of Mourning. A cross-check against data.gov.au's *Australian Public Holidays Dates Machine Readable Dataset* (CC BY) is planned.

## 3. Market price cap / floor (data-quality bounds)

- **Source:** AEMC *Schedule of reliability settings*, published each February for the following financial year.
- **Use:** `configs/market_limits.yaml` gives the per-financial-year cap and the −$1,000/MWh floor. They are used **only** to flag implausible `RRP` values; nothing is clipped or dropped.
- **Provenance:** each value is tagged `aemc` (taken from an AEMC schedule) or `secondary` (third-party compilation, to be confirmed). AEMC's site rejects automated fetches, and one secondary source contradicted AEMC for 2025-26, so the `secondary` values should be checked by hand before being used for anything beyond QA.

## 4. Weather: Open-Meteo

Two Open-Meteo APIs supply hourly weather for five Victorian sites (§5). Code: `src/gridcast/ingestion/open_meteo.py` (client), `src/gridcast/ingestion/weather.py` (ETL), `src/gridcast/weather/` (run timing, 5-minute alignment). Settings: `weather:` in `configs/base.yaml`. The step-0 investigation behind these choices is in [`m2_data_spike.md`](m2_data_spike.md).

| | Variant C: archived forecasts | Variant D: reanalysis |
|---|---|---|
| API | Previous Runs, `https://previous-runs-api.open-meteo.com/v1/forecast` | Historical Weather, `https://archive-api.open-meteo.com/v1/archive` |
| Model (`models=`) | `gfs_seamless` (NOAA GFS; for Victoria the 0.11° `ncep_gfs013` grid) | `era5` (ECMWF ERA5, 0.25°) |
| Columns requested | `<variable>_previous_day1`, `<variable>_previous_day2` | `<variable>` |
| Table | `weather_forecasts` (one row per model, site, offset, valid hour) | `weather_observations` (one row per model, site, valid hour) |
| Usable at forecast time? | Yes, from `available_at` (below) | **No.** ERA5 is published about 5 days late, so D is a perfect-foresight oracle by construction |

**Variables.** `temperature_2m` (°C), `apparent_temperature` (°C), `relative_humidity_2m` (%), `cloud_cover` (%), `shortwave_radiation` (W/m²) and `wind_speed_10m` (m/s, requested with `wind_speed_unit=ms`). Every response's `hourly_units` is checked against the configured unit, and a mismatch fails the month.

**Licence and attribution.** Open-Meteo data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the free API is for non-commercial use. ERA5 is produced by ECMWF/Copernicus and GFS by NOAA.

> Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), from NOAA GFS forecasts and ECMWF ERA5 reanalysis.

**Access method.** The client sends the project User-Agent, waits `request_delay_seconds` between requests, and retries HTTP 429, 5xx and transport errors with exponential backoff (5, 10, 20, 40, 60 s; six attempts). A 4xx error or an error body fails immediately with Open-Meteo's `reason`. Each request is one site and one NEM month, and the response is cached as JSON under `raw/open_meteo/<api>/<model>/<site>/<start>_<end>.json`. A `.meta.json` file records the request parameters and SHA-256. A cached file is reused only if both still match, so changing a variable, model or coordinate downloads again. Paths are built, never globbed (exFAT data root). Responses whose last date is within `settle_days` (10) of today are not cached, because the archives fill in recent days late. `gridcast ingest-weather --refresh` re-downloads everything.

**Rate limits.** The free tier allows 600 calls a minute, 5,000 an hour and 10,000 a day. A request covering more than two weeks or more than ten variables counts as several calls, fractionally: one site-month of 12 forecast columns counts as about 2.7 calls. The full window is 60 months × 5 sites × 2 APIs = 600 requests.

### Timestamp conventions (critical)

- Requests use `timezone=GMT`. `time` labels are converted to UTC `valid_time` values on whole hours.
- **Instant variables** (temperature, apparent temperature, humidity, cloud cover, wind) are values at `valid_time`. **`shortwave_radiation` is the mean over the hour ending at `valid_time`.**
- **Run timing.** For valid hour *t* and offset *K* days, the value comes from the GFS run initialised at `run_init = floor_6h(t − 24·K h)`. This was verified exactly against individual runs in step 0. Day offsets are fixed 24-hour steps in UTC, never calendar days. Each row stores `run_init` and `available_at = run_init + 6 h`. Open-Meteo published one observed GFS run 5.4–5.5 h after initialisation, so the 6 h delay leaves about 0.5 h of margin. Nothing in the archive records the real publication time of past runs. The rule is enforced three times: by the pandera schema on load, by the `weather_run_timing` data-quality check, and partly by table constraints (`run_init <= valid_time − K × 24 h`, `available_at > run_init`).
- `previous_day0` is never stored. Its run starts after *t* − 6 h, so it can never be available before the target time.
- **Monthly chunks.** A NEM month's 5-minute intervals end in (start, end] (NEM time). Hourly values are needed from `floor_1h` of the first interval to `ceil_1h` of the last, i.e. the UTC hours `[start, end]`. For October 2021 that is 2021-09-30 14:00 → 2021-10-31 14:00 UTC. Adjacent months share one hour, and the idempotent upsert absorbs the overlap.

### From hourly values to 5-minute features

`src/gridcast/weather/align.py` holds pure functions that the step-2 feature builder calls. Nothing at 5-minute resolution is stored.

- **Site weighting:** fixed population weights (§5). A weighted value exists only if every site has a value. Weights are never renormalised over the sites that happen to be present.
- **Interpolation:** for an interval ending at *t*, instant variables are interpolated linearly between `floor_1h(t)` and `ceil_1h(t)`; on the hour, the hour's own value is used. `shortwave_radiation` takes the value labelled `ceil_1h(t)`.
- **Availability:** a 5-minute forecast value needs both bracketing hours, so its `available_at` is the later of the two hours' `available_at`. If either hour is missing, the value is missing and never available.

### Missing-data characteristics

Known before ingestion (step 0, Melbourne and Latrobe Valley):

- GFS `temperature_2m` is archived from 2021-03-24, before the study window. It has one gap: `previous_day1` is missing for valid times 2023-12-30 00:00 → 2024-01-19 11:00 and `previous_day2` for 2023-12-30 00:00 → 2024-01-20 11:00.
- The other five GFS variables start at 2024-01-19 12:00 (day1) and 2024-01-20 12:00 (day2). Both edges correspond to the run initialised 2024-01-18 12:00.
- These ranges are declared in `weather.forecasts.expected_missing`. The `weather_known_gaps` check reports them as warnings; any other missing value fails `weather_gaps`.

Measured on the full load (all five sites, NEM months Oct 2021 → Sep 2026 = valid hours 2021-09-30 14:00 → 2026-09-30 14:00 UTC; `gridcast dq` on 2026-11-02, with breakdowns from the same check queries):

- **Rows:** `weather_observations` 219,125 = 5 sites × 43,825 hours, with no gaps. `weather_forecasts` 433,210 = 5 sites × 2 offsets × 43,825 hours, minus 5,040 rows with no values at all (the seam below).
- **`weather_gaps`: 0.** There are no unexpected missing values at any site.
- **`weather_known_gaps`: 1,014,740 missing values, all expected.** Each range is identical at all five sites:

  | Reason | Variables | Offset | Missing valid hours | Hours per site |
  |---|---|---|---|---|
  | `gfs_archive_seam` | temperature | day1 | 2023-12-30 00:00 → 2024-01-19 11:00 | 492 |
  | `gfs_archive_seam` | temperature | day2 | 2023-12-30 00:00 → 2024-01-20 11:00 | 516 |
  | `before_multivariable_archive` | the other five | day1 | window start → 2024-01-19 11:00 | 20,182 |
  | `before_multivariable_archive` | the other five | day2 | window start → 2024-01-20 11:00 | 20,206 |

  The seam therefore affects the whole archive, as step 0 found for two sites.
- **Rounding artefacts:** the GFS archive holds **153 cloud cover values of exactly 101 % (116) or −1 % (37)**. They fall between 2024-01-23 and 2025-11-25, at all five sites (23–34 per site) and both offsets, and **only at 00, 06, 12 and 18 UTC**. Nothing similar appears in ERA5 or any other variable. They are artefacts of how the archive stores values, not weather, so the loader clips values at most `clip_tolerance` (1 point) outside a bounded variable's range (`clip_to`, 0–100 % for cloud cover and humidity) and records each count in `ingestion_runs.files[].clipped_values`. The `weather_clipped_values` check reports them as a warning (153). Anything further out of range is still rejected. The first full load, before this rule existed, rejected 23 months on these values, so the loader is doing its job.
- **`weather_out_of_range`, `weather_run_timing`, `weather_stale`, `weather_site_weights`: 0 violations.** Observed temperature ranges: GFS −1.4 to 46.0 °C, ERA5 −2.6 to 43.0 °C.
- **Transient failures:** one response in the first load was an empty HTTP 200 body, and the archive APIs returned HTTP 503 "The service is overloaded" for at least 70 minutes on 2026-11-02 (observed 14:37–15:48 UTC). Empty or non-JSON bodies are now retried like 5xx errors. Each month's latest attempt succeeded (`ingestion_failures`: 0).
- **Grid cells actually used** (from `ingestion_runs.files`; elevation in m):

  | Site | GFS cell | ERA5 cell | Elevation |
  |---|---|---|---|
  | Melbourne | −37.78, 144.96 | −37.75, 145.00 | 15 |
  | Geelong | −38.13, 144.38 | −38.25, 144.25 | 23 |
  | Ballarat | −37.55, 143.91 | −37.50, 143.75 | 446 |
  | Bendigo | −36.73, 144.26 | −36.75, 144.25 | 223 |
  | Latrobe Valley | −38.25, 146.37 | −38.25, 146.50 | 72 |

### Known limitations

- **Grid cells, not stations.** Each site is the model grid cell containing its coordinates (Open-Meteo's default `cell_selection=land`), with temperature adjusted to the site's elevation. The returned grid point and elevation are recorded per request in `ingestion_runs.files`. GFS (0.11°) and ERA5 (0.25°) use different cells, so part of any C–D difference is resolution, not forecast error.
- **ERA5 is not ground truth.** It is a reanalysis built with ECMWF's model, so variant D is "perfect foresight of ERA5", not of station observations.
- **Publication delay is an assumption.** It rests on one observed run (§3 of the spike). Step 4 can rebuild C with an 8 h delay as a sensitivity check.
- **Archives can change.** Open-Meteo may revise or extend its archives. The cache pins what was downloaded, and its SHA-256 is recorded in `ingestion_runs.files`.
- **GFS upgrades.** NOAA upgrades GFS from time to time, so forecast error may shift within the window. M2 reports errors by year.

## 5. Population weights: ABS Regional population

- **Source:** Australian Bureau of Statistics, *Regional population, 2024-25* (released 31 March 2026), data cube `32180DS0004`, Table 2 "Estimated resident population, Significant Urban Areas, Australia". Estimated resident population (ERP) at **30 June 2024**, a revised estimate (2025 values are preliminary).
- **Licence:** [CC BY 4.0](https://www.abs.gov.au/website-privacy-copyright-and-disclaimer), © Commonwealth of Australia.
- **Use:** one fixed weight per weather site, its Significant Urban Area (SUA) population divided by the region's total over all sites. Weights are computed in code from the populations in `configs/base.yaml` and stored in `weather_sites`.

| Site | Coordinates | SUA(s) | ERP 30 Jun 2024 |
|---|---|---|---|
| Melbourne | −37.83, 144.98 (near the CBD; the step-0 spike point) | Melbourne | 5,225,364 |
| Geelong | −38.15, 144.36 | Geelong | 308,021 |
| Ballarat | −37.56, 143.85 | Ballarat | 118,841 |
| Bendigo | −36.76, 144.28 | Bendigo | 105,610 |
| Latrobe Valley | −38.23, 146.40 (Morwell) | Traralgon – Morwell + Moe – Newborough | 43,841 + 22,359 |

> Source: Australian Bureau of Statistics, Regional population 2024-25, CC BY 4.0.

**Limitations.** Melbourne carries about 90% of the weight, so the weighted temperature is close to Melbourne's own. The weights are fixed, although Melbourne grew from 4.87 M (2021) to 5.33 M (2025, preliminary); the shares barely move. A population is not an electricity load: industrial loads (for example the Portland smelter, outside these sites) are not represented.

## 6. AEMO's own demand forecasts: pre-dispatch "all runs" archives (M2 benchmark)

- **Use:** a **benchmark only**, never a model input (`docs/plans/m2_plan.md` §2.6). Loaded by `gridcast ingest-aemo` (`make ingest-aemo`) into `aemo_demand_forecasts` (migration `0004_aemo_forecasts`), for VIC1, archive months Sep 2024 → Sep 2026. September 2024 holds the runs for issue times before 04:00 on 1 October 2024. Holdout months are loaded but compared only in step 6.
- **Files:** in each MMSDM month, `PREDISP_ALL_DATA/PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01#…zip` and `P5MIN_ALL_DATA/PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#ALL#FILE01#…zip`, each about 50 MB a month (2.4 GB over the 25 months). The `#ALL#` archives hold every run; the `DATA/` tables of the same name do not. They are cached under `raw/mmsdm/<TABLE>_ALL/` with the same `listing.json` and SHA-256 rules as §1.
- **Licence:** as §1 (AEMO copyright permissions).

| | 30-minute pre-dispatch | 5-minute pre-dispatch |
|---|---|---|
| MMS section | `PREDISPATCH.REGION_SOLUTION`, v8–9 accepted | `P5MIN.REGIONSOLUTION`, v9–10 accepted |
| Runs | 48 a day | 288 a day |
| `run_time` (stored, UTC) | 04:00 NEM time + (PP − 1) × 30 min on the trading day of `PREDISPATCHSEQNO` = `YYYYMMDDPP` | `RUN_DATETIME` |
| Targets | 30-minute periods ending 30 min … 16–39.5 h after the run time | 5-minute intervals ending 0 … 55 min after it |
| `TOTALDEMAND` | mean over the 30-minute period (step 0) | that interval; interval 0 is the dispatch demand |
| `available_at` = `LASTCHANGED` | 55–819 s **after** the run time | 119–279 s **before** the run time |

Timing ranges are over the 25 loaded months (`aemo_run_timing` passes, and every run has one `LASTCHANGED`).

**Loader rules.** Columns are mapped by name; other table versions fail. Only `INTERVENTION = 0` rows are kept, and a duplicate (region, run, target) fails the month. A run whose first pre-dispatch period does not end 30 minutes after its run time fails, and so does a 5-minute interval outside 0–55 minutes.

**Data-quality checks** (`gridcast dq`):

- `aemo_missing_runs` (warn): runs missing from the 30- or 5-minute cadence. Over Sep 2024 → Sep 2026 these are 3 pre-dispatch and 17 five-minute runs: about an hour on 5 September 2024 (13:30–14:25 NEM time), 20 minutes on 19 November 2024 and one five-minute run on 4 March 2025. Every run sits in its own trading month's archive. In November 2024 the loader also dropped 143 pre-dispatch and 144 five-minute rows of an intervention run (`INTERVENTION = 1`).
- `aemo_run_timing` (warn): runs published outside the expected window (pre-dispatch up to 15 min after its run time; 5-minute pre-dispatch up to 10 min before it), or published more than once.
- `aemo_p5min_alignment` (fail): a timestamp-alignment guard. Interval 0 of a 5-minute run should equal the dispatch demand of the interval ending at its run time. Aligned, at most 7.25% of a month's values differ (mean absolute difference ≤ 0.32 MW; the cause of the small differences is unverified). Shifted by one interval either way, at least 99.97% differ by 38–124 MW on average. The check fails a month where more than half differ.
- `aemo_out_of_range` (fail): forecast demand outside the region's plausible bounds.

# Data sources

Each source used by GridCast-AU is documented below: where it comes from, its licence, frequency, schema, timestamp conventions, missing-data characteristics and known limitations. Sources marked *planned* are documented ahead of implementation.

| Source | Used for | Status |
|---|---|---|
| AEMO MMSDM monthly archive (NEMWeb): `DISPATCHREGIONSUM` | 5-min regional demand | Milestone 1 |
| AEMO MMSDM monthly archive (NEMWeb): `DISPATCHPRICE` | 5-min regional spot price (RRP) | Milestone 1 |
| `holidays` Python package (subdivision VIC) | Public-holiday calendar | Milestone 1 |
| AEMC schedule of reliability settings | Market price cap/floor (QA bounds only) | Milestone 1 |
| Open-Meteo historical weather API | Temperature, humidity, etc. | *planned (M2)* |
| AEMO MMSDM `DISPATCH_UNIT_SCADA` / `ROOFTOP_PV_ACTUAL` | Generation mix | *planned* |
| AEMO MMSDM `PREDISPATCHREGIONSUM_D` / `PREDISPATCHPRICE_D` | Benchmark vs AEMO's own forecasts | *optional* |

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

## 4. Weather: Open-Meteo *(planned, milestone 2)*

- **Source:** [Open-Meteo](https://open-meteo.com/) Historical Weather API (ERA5/ERA5-Land reanalysis plus national models), hourly. Data licence **CC BY 4.0**; the free tier is for non-commercial use.
- **Why not BOM directly:** the Bureau of Meteorology's sub-daily historical observations are not freely available in bulk, and BOM's website terms forbid automated scraping.
- **Leakage warning:** reanalysis weather *at the target time* would not be known when the forecast is issued. Using it as a feature makes an **oracle** (perfect-weather-forecast) model. Milestone 2 must either use archived weather *forecasts* (e.g. Open-Meteo's historical-forecast or previous-runs APIs, after checking their coverage) or label any observed-weather model explicitly as an oracle upper bound.

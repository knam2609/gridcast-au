# M3 step 0: AEMO's probabilistic demand forecasts (data spike)

Run on 2026-11-10. Script: [`scripts/spikes/m3/aemo_poe.py`](../scripts/spikes/m3/aemo_poe.py). Command, for each month: `uv run python scripts/spikes/m3/aemo_poe.py <data root>/raw/spikes/m3 --month YYYY-MM`. The months were 2024-10, 2025-01, 2025-07 and 2026-09: the start of the benchmark window, a summer month, a winter month and the last holdout month. All figures are for VIC1 and are copied from the script's output.

**Question** (`m3_plan.md` §2.6 and step 0): can AEMO's 10/50/90% probability-of-exceedance (POE) demand forecasts serve as a fair outside benchmark for our 80% intervals?

**Answer: yes, with one disclosed caveat.** Every 30-minute run is archived, and the run timing matches M2's pre-dispatch alignment. But `DEMAND50` sits a consistent ~40 MW above AEMO's own pre-dispatch `TOTALDEMAND`, and neither the documentation nor the data says why. The user decided (2026-11-10, §6) to use the band **as published** as the headline, with a band **re-centred** on pre-dispatch `TOTALDEMAND` reported beside it.

## 1. Source and structure

- **Table:** `PDPASA_REGIONSOLUTION`, version 7 in all four months. It lives in the MMSDM monthly `DATA/` folder (`PUBLIC_ARCHIVE#PDPASA_REGIONSOLUTION#FILE01#YYYYMM010000.zip`), not in an `#ALL#` archive.
- **Size:** 65–67 MB zipped for a full month (about 256 MB of CSV); September 2026 is 29 MB (one run type, see below). Oct 2024 → Sep 2026 is therefore roughly 1.1 GB to download. Only VIC1 rows of one run type are kept: 82,584 rows a month.
- **Columns used:** `RUN_DATETIME`, `INTERVAL_DATETIME`, `REGIONID`, `RUNTYPE`, `DEMAND10`, `DEMAND50`, `DEMAND90`, `LASTCHANGED`. The key is (`RUN_DATETIME`, `RUNTYPE`, `INTERVAL_DATETIME`, `REGIONID`), with no duplicates.
- **Run types:**
  - `LOR` and `OUTAGE_LRC` until mid-2025; `OUTAGE_LRC` is incomplete in July 2025 (80,850 rows against 82,584) and absent in September 2026. The data model notes that it was discontinued from 31 July 2025 ([GR-MMS mirror of the MMS Data Model](https://mmsdocs.global-roam.com/MMS/PDPASA/PDPASA_REGIONSOLUTION/)).
  - **Use `LOR` only.** Its demand columns are the ones published throughout.
- **Every run is archived:** 48 `LOR` runs a day, 30 minutes apart, with no gaps in any of the four months.

## 2. Run timing

- **Run identity:** `RUN_DATETIME` is "the first half hour ended interval of the run" (data model). The first `INTERVAL_DATETIME` of every run equals its `RUN_DATETIME`.
- **Publication:** `LASTCHANGED` is identical within each run. Medians of `LASTCHANGED − RUN_DATETIME` were −1,386, −1,442, −1,509 and −1,560 s. The run is therefore published **4–7 minutes after** `RUN_DATETIME − 30 min`. The extremes over the four months were 2 and 17 minutes.
- **Matching pre-dispatch run:** `DEMAND50` was compared with pre-dispatch `TOTALDEMAND` for the same period end, assuming different nominal run times. Taking the nominal run time as **`RUN_DATETIME − 30 min`** is the only assumption that matches 100% of periods in every month, and it gives the smallest median difference. This is the same nominal run time that M2's matched-information alignment uses for pre-dispatch.
- **Alignment for step 5:**
  - *Matched information (headline):* at our issue time *o*, use the run with `RUN_DATETIME = o + 30 min`. Its information ends at *o*, like ours, and it is published a few minutes later.
  - *Strict:* use the latest run with `LASTCHANGED ≤ o`, which is the run with `RUN_DATETIME = o`.
- **Reach:** like pre-dispatch, runs end at the end of the next trading day. 64.6% of runs reach 24 h after publication in every month (the shortest 15.9 h, the longest 39.4 h), so coverage must be reported by lead, as in M2.

## 3. The POE band

- **Order:** `DEMAND10 ≥ DEMAND50 ≥ DEMAND90` in 100% of rows.
- **Exactly symmetric:** `DEMAND10 − DEMAND50` equals `DEMAND50 − DEMAND90` in every row of every month, to within 1 MW (rounding). Empirical errors are not symmetric (§5), so the band is not built from an empirical error distribution in both directions.
- **Width:** the half-width averages 4.09%, 5.82%, 3.68% and 4.07% of `DEMAND50` (Oct 2024, Jan 2025, Jul 2025, Sep 2026). It grows with lead: in January 2025 it is 2.72% of `DEMAND50` for periods within 30 minutes and 6.64% beyond 24 h.

## 4. What `DEMAND50` measures

- **Definition:** the data model describes it only as the "50% Probability of Exceedance demand forecast". AEMO's own Data Model Report pages returned 404 to scripted requests on 2026-11-10.
- **Against the matched pre-dispatch run:** `DEMAND50 − TOTALDEMAND` averages **+46.7, +41.9, +56.4 and +36.1 MW** (standard deviations 27.5, 21.8, 43.9 and 24.7 MW). It is negative in 0.4–4.2% of rows and below −10 MW in at most 0.17% (July 2025). Its mean by lead stays within about 15 MW in each month. In January 2025 it varies by hour, from 32 MW around midnight to 54 MW at 14:00 NEM time.
- **No column of the row explains it:**
  - For every numeric column of the January 2025 rows, the mean gap between the column and the offset is no smaller than about 42 MW, the size of the offset itself.
  - `AGGREGATESCHEDULEDLOAD` and the wholesale demand response columns are zero.
  - `DEMAND_AND_NONSCHEDGEN − DEMAND50` equals `TOTALINTERMITTENTGENERATION` exactly in all four months (documented as the allowance for non-scheduled generation). That is a separate component.
- **Bias against actual demand** (30-minute mean of 5-minute `TOTALDEMAND`, at leads of 12–24 h):

| Month | `DEMAND50` bias (MW) | Pre-dispatch `TOTALDEMAND` bias (MW) |
|---|---:|---:|
| Oct 2024 | −9.4 | −58.4 |
| Jan 2025 | +52.8 | +9.4 |
| Jul 2025 | +26.1 | −28.5 |
| Sep 2026 | −9.4 | −44.8 |

  In three of the four months pre-dispatch under-forecasts beyond 1 h, and the offset partly cancels that. **The data cannot tell a definition difference** (DEMAND50 = TOTALDEMAND + a component) **from a different forecast**: both readings predict these numbers.

## 5. How often actual demand falls in the band

Pooled over all leads (`LOR` runs, actual 30-minute mean demand):

| Month | Inside [DEMAND90, DEMAND10] | Below DEMAND90 | Above DEMAND10 | `DEMAND50` MAE (MW) | `DEMAND50` bias (MW) |
|---|---:|---:|---:|---:|---:|
| Oct 2024 | 70.9% | 13.3% | 15.9% | 135.1 | −3.6 |
| Jan 2025 | 71.3% | 17.0% | 11.7% | 192.6 | +46.4 |
| Jul 2025 | 77.6% | 12.9% | 9.5% | 155.6 | +10.8 |
| Sep 2026 | 66.1% | 16.7% | 17.2% | 189.3 | −2.0 |

- If the band were a calibrated 80% interval, 80% of outcomes would fall inside it and 10% on each side. In the sampled months it covers 66–78% pooled. By lead it covers 61.1–77.9% at every lead up to 24 h in every month, and 79.3–91.6% beyond 24 h.
- The band does behave like an uncertainty band at the level of each interval. It widens with lead, and its coverage is in the range of a central interval. A band built from one extreme weather scenario would be expected to cover far more. So comparing coverage and interval scores is meaningful. The band's symmetry and under-coverage are facts about AEMO's product, reported as such.
- These are four sampled months, not the benchmark. The step-5 comparison covers every month of Oct 2024 → Sep 2025 (development part) and, in step 6, Oct 2025 → Sep 2026.

## 6. Decision (user, 2026-11-10)

The plan's fallback rules did not cover an unexplained offset, so the user was asked.
- **First answer:** report both versions, with the band re-centred on the matched pre-dispatch `TOTALDEMAND` as the headline. The reason given was that re-centring gives AEMO the benefit of the doubt.
- **Revised:** the per-month bias table (§4) then showed that re-centring would usually *worsen* AEMO's band, so the question was put again.
- **Final decision:**
  1. **Headline: AEMO's band as published** (`DEMAND90`, `DEMAND10`), with `DEMAND50` as its median.
  2. **Beside it, as a sensitivity check:** the same half-width placed around the matched pre-dispatch run's `TOTALDEMAND` (published before the PDPASA run, so no later information is used).
  3. **Disclosed:** the unexplained offset of about +40 MW, the band's symmetry, and the fact that the data cannot tell a definition difference from a different forecast.
  4. Use `LOR` runs only. Use M2's two alignments (matched information as the headline, strict beside it). Report coverage by lead because of the reach limit.
- The comparison is reported only, never pass/fail (`m3_plan.md` §3). Our intervals are recalibrated for 30-minute means for it (`m3_plan.md` §2.6).

# Milestone 3 plan: prediction intervals for demand

**Goal:** give the M2 demand forecast honest uncertainty: prediction intervals whose coverage matches their stated level overall, at every lead time and during extreme events, that are sharper than intervals built around the baselines, and that are compared with AEMO's own probabilistic demand forecasts where that comparison is valid.

**Status:** planned 2026-11-10 after a question round with the user. Step 0 done (2026-11-10, [`docs/m3_data_spike.md`](../m3_data_spike.md)): AEMO's POE forecasts are usable, with a disclosed caveat. Step 1's pre-scoring spike done (2026-11-10, [`docs/m3_prescoring_spike.md`](../m3_prescoring_spike.md)): the candidate grid and the cold threshold are fixed below, before any scored interval existed. Steps 2–3 done (2026-11-11: QR and CQR development batches). Step 4 done (2026-11-12, [`reports/m3_intervals.md`](../../reports/m3_intervals.md)): **U1–U4 pass** on the development period. Step 5 done (2026-11-13): the interval recipe is frozen (`bc5337e`), and the AEMO comparison is in [`reports/m3_aemo.md`](../../reports/m3_aemo.md). Step 6 done (2026-11-13): the holdout was scored once ([`reports/m3_holdout.md`](../../reports/m3_holdout.md)); **U5 passes and U1–U4 hold on the holdout**. M3 is complete. Branch `m3-prediction-intervals`, from `main` at `a1b26bb` (M1 and M2 merged). The success criteria in §3 were agreed before any interval result existed.

## 1. Decisions (agreed 2026-11-10)

| Topic | Decision |
|---|---|
| Target | Victorian demand only, around the frozen M2 forecasts. Price intervals wait for price ML (M2b) |
| Methods | (1) **Conformal around the stored point forecasts**: rolling calibration on past residuals, no retraining. (2) **LightGBM quantile regression (QR)**: a new walk-forward run for variant C. (3) **Conformalised QR (CQR)**: the QR bounds corrected by the same rolling conformal machinery. All three are compared on identical (issue time, lead) pairs |
| Headline | **CQR on LightGBM C**, pre-registered. U1–U3 judge it; the other methods are reported |
| Interval baselines | Conformal intervals around persistence, daily naive and weekly naive, built with the same procedure as method (1) |
| Levels | **80% headline** (AEMO's POE90–POE10 band); 50% and 95% also reported. QR quantiles: 0.025, 0.1, 0.25, 0.75, 0.9, 0.975. The median used in the weighted interval score is the point forecast |
| Conformal settings | **Re-selected at every quarterly retrain** from past data only, as in M2: every candidate setting is scored on the last 8 weeks before the retrain and the best one is used for the next quarter |
| QR tuning | **Changed 2026-11-11 (user decision after the step-2 smoke run): per-quantile selection.** At each retrain and lead band, every quantile scores M2's 8 settings (4 target forms × leaf counts 31 and 127; objective `quantile`) by its own pinball loss on the early-stopping slice and refits the lowest. *Originally:* reuse the point model's target form and leaf count, a choice made to save compute. The smoke run on the first retrain (pre-scoring data only) put that design at about 1.5 h instead of the 4–6 h estimated, which removed the reason for it. Its raw-QR results (80% coverage 67%; 22% of pairs crossed before sorting) were seen when deciding; this is disclosed. **Second smoke run, with the selection (same retrain and fold):** 1,206 s, projecting 9–11 h for the full run. On the fold it was slightly worse than reusing the point settings (IS80 1370.5 against 1339.4, WIS 155.9 against 154.1, 80% coverage 68.4% against 67.2%, 27.5% of pairs crossed against 22.4%). The user kept per-quantile selection for an overnight run, since one retrain is thin evidence |
| Extreme events | **Both kinds, labelled separately:** groups known at the issue time (honest conditional coverage) and groups defined by the outcome (shown with their built-in downward bias explained). **Reported only, not pass/fail** |
| Holdout | Develop and select on the development period only, freeze the interval recipe, then score the holdout year (Oct 2025 → Sep 2026) **once**, with an S5-style lock. **Disclosed:** the holdout is new for intervals, but its point errors (including the winter under-forecast) were already seen in M2 |
| AEMO probabilistic forecasts | **Included after step 0** (`PDPASA_REGIONSOLUTION`, `LOR` runs). **Headline: the POE90–POE10 band as published**; beside it, the same half-width re-centred on the matched pre-dispatch `TOTALDEMAND`. `DEMAND50`'s unexplained offset of about +40 MW is disclosed (user decision 2026-11-10, spike §6) |

## 2. Design

### 2.1 Periods and pairs

- **Interval development period:** issue times **1 Aug 2023 → 30 Sep 2025**. The ML forecasts start on 2022-11-01. The longest candidate calibration window (26 weeks), plus the 8 weeks scored at the first selection, needs about 34 weeks of residuals before the first scored issue time. 2023-08-01 is the first quarterly retrain after that warm-up.
- **Pre-scoring data:** issue times 1 Nov 2022 → 31 Jul 2023. Spikes that settle the candidate grid (step 1) and QR details (step 2) may use only these, as M2's spikes did.
- **Holdout:** issue times 1 Oct 2025 → 30 Sep 2026, scored once (step 6).
- **Identical pairs:** every method and interval baseline is scored on the same pairs, the M2 common mask (pairs without C weather excluded). Calibration windows use the same mask, so no method gets extra calibration data.
- **Inputs:** run 2's stored development forecasts (batch `main` = `6d783932-5834-4e18-a9e8-0e3567db46e6`) and, in step 6 only, the stored holdout forecasts (`holdout_main` = `fcb6abb6-49a3-4923-9d79-64e05a44d75f`).

### 2.2 Rolling conformal intervals (methods 1 and 3)

- **Score.**
  - Method (1) uses the signed residual e = y − ŷ: the upper bound is ŷ plus a high quantile of e, the lower bound ŷ plus a low quantile.
  - Method (3) (CQR) uses one score per side (the asymmetric form of Romano et al. 2019): q̂_lo − y for the lower bound and y − q̂_hi for the upper. *Clarified in step 1:* the symmetric score max(q̂_lo − y, y − q̂_hi) would correct both sides by the same amount, which contradicts the next point.
  - The lower and upper bounds come from separate quantiles, so the intervals can be asymmetric. They correct for a bias such as the winter under-forecast, rather than only widening.
- **Information rule.**
  - At issue time *o*, the calibration set for lead *h* holds only pairs (o′, h) whose target is known at *o* (o′ + h ≤ o), within the trailing window.
  - Intervals at *o* change only when data ending at or before *o* change. Property tests check this (§2.5).
- **Candidate grid (fixed in step 1 on pre-scoring data, recorded before scoring; [`docs/m3_prescoring_spike.md`](../m3_prescoring_spike.md), user decisions 2026-11-10):**
  - window length ∈ {4, 8, 13, 26} weeks;
  - grouping ∈ {per lead; per lead × a time-of-day block of the target}. **Blocks: 8 × 3 h**, starting at local hours 0, 3, …, 21 of the target's start;
  - adaptive conformal inference (ACI, Gibbs & Candès 2021): **step γ ∈ {0 (off), 0.001, 0.005}**. ACI updates each lead's miscoverage target with only the coverage outcomes known by *o*, so it reacts to drift with a delay of *h*.
  - That is 4 × 2 × 3 = 24 candidate streams per point model.
- **Implementation rules (step 1, [`src/gridcast/intervals/conformal.py`](../../src/gridcast/intervals/conformal.py)):**
  - *Window:* the latest W issue times whose lead-*h* target is known at *o*, so every lead has a window of the same length; only scored pairs (the common mask) enter it.
  - *Conformal quantile:* the k-th smallest of the window's n scores, k = ⌈(n + 1)(1 − β)⌉, with β the side's miscoverage target (α/2 without ACI). If k falls outside [1, n], the window's most extreme score is used and the bound is counted as clipped.
  - *ACI:* per lead, level and side: β ← β + γ(α/2 − miss), applied at the issue time when the outcome becomes known, with β kept in [0, ½].
  - *Crossed intervals* (lower above upper, possible with ACI or CQR) are output as a zero-width interval at the midpoint and counted.
  - *Bookkeeping:* every outcome is used (for ACI, selection and diagnostics) only from the issue time at which it is known. The leakage tests (§2.5) cover all three uses.
- **Selection.**
  - All candidate streams are computed continuously from the warm-up onwards, each using past data only.
  - At each quarterly retrain, for each level and each M2 lead band (5–30 min, 35 min–2 h, 2 h 5 min–6 h, 6 h 5 min–24 h), the stream with the lowest mean interval score over the last 8 weeks is used for the next quarter. Ties go to the first candidate in grid order.
  - The same rule applies to the interval baselines.

### 2.3 Quantile LightGBM (method 2)

- **Setup.**
  - Same features (variant C), lead bands, training sample, retrain schedule and walk-forward harness as M2.
  - The objective is `quantile` with α ∈ {0.025, 0.1, 0.25, 0.75, 0.9, 0.975}.
  - **Per-quantile selection (§1, changed 2026-11-11):** at each retrain and band, each quantile early-stops every candidate (target form ∈ {residual, direct, anchored, latest} × `num_leaves` ∈ {31, 127}, in that order) by pinball loss on the last 8 weeks of the window. The lowest wins (ties to the first in grid order) and only it is refit on the whole window, as M2's point models were. *Originally:* the target form and `num_leaves` copied from run 2's stored selection records.
  - A quantile of y − anchor plus an anchor known at *o* is a quantile of y, so every target form applies.
- **Crossing:** quantiles are sorted per pair (rearrangement), and the crossing rate before sorting is reported.
- **Cost:** measured on the first retrain in a smoke run before the full run (§5). If the full run looks longer than about 10 h, ask the user again before starting it.
- **Implementation (step 2, [`src/gridcast/intervals/quantile.py`](../../src/gridcast/intervals/quantile.py), [`qr_batch.py`](../../src/gridcast/intervals/qr_batch.py)):**
  - *Where the code lives:* under `gridcast.intervals`, not next to `models/ml.py` as first written, so that the frozen M2 paths (`evaluation.holdout.FROZEN_PATHS`) stay byte-identical. The quantile model subclasses M2's `LightGBMModel` (objective `quantile`, metric `quantile` = mean pinball loss).
  - *Training rows:* the loop reuses M2's harness functions unchanged. It builds M2's A, C and D samples and trains C on the rows all three share, exactly as run 2 did. M2's `run_walk_forward` cannot train one variant on a group's shared rows, so the loop itself is new.
  - *Point reproduction at every (retrain, band):* the loop also refits run 2's chosen point setting with its stored tree count (still useful after the switch to per-quantile selection: it pins the training rows). Its cutoff and row counts must match run 2's record before any quantile is fitted, and each fold's point forecasts must equal run 2's stored ones exactly; otherwise the run stops and nothing is stored. This proves the quantiles train on the point model's rows.
  - *Issue times:* forecasts for every development issue time from 2022-11-01 (the pre-scoring ones are CQR's calibration warm-up in step 3). Metrics only from `intervals.first_issue`, on run 2's common mask, scored against run 2's stored actuals, as in step 1.
  - *Provenance:* the full run refuses a working tree with uncommitted changes. It records every quantile's candidates and selection (pinball loss, tree count), the crossing share per (retrain, band) and the reproduction result.

### 2.4 Scores and diagnostics

- **Per level:**
  - empirical coverage;
  - mean width;
  - **interval score** IS_α = (u − l) + (2/α)(l − y)⁺ + (2/α)(y − u)⁺ (Gneiting & Raftery 2007), with α = 0.2 for 80%.
- **Across levels:**
  - weighted interval score (WIS: three intervals plus the median);
  - pinball loss per quantile;
  - a reliability diagram (the observed share of y ≤ q_τ against τ).
- **Conditional coverage:**
  - by lead;
  - by local hour of the target;
  - by month (the winter bias found in M2);
  - by year and by retrain quarter.
- **Uncertainty of the coverage numbers:** errors are correlated in time, so coverage and score differences get **block-bootstrap intervals that resample whole days**, as in M2's AEMO comparison. Step 4 adds a 7-day-block sensitivity for the pooled headline rows (user decision 2026-11-12).
- **Extreme-event groups.**
  - *Known at issue time:* forecast daily maximum ≥ 35 °C (weighted GFS temperature); forecast daily maximum **≤ 12 °C** (cold, fixed in step 1 from pre-scoring data: 11 such days against 8 hot days in Nov 2022 → Jul 2023); public holidays; DST-change days.
  - *Defined by outcome (biased by construction, in a separate table):*
    - the top 1% of actual demand intervals;
    - intervals with price > $300/MWh (`evaluation.price_spike_threshold`);
    - the 2024-02-13 storm;
    - the 2024-12-09 demand jump noted in M2.

### 2.5 Leakage tests and the M2 freeze

- **Property tests:**
  - Perturbing actual demand that ends after *o* must leave every interval issued at *o* unchanged, for all three methods and the baselines.
  - Perturbing point or QR forecasts issued after *o* must also leave them unchanged.
  - Control tests show the checks fail when the information rule is broken.
- **The M2 freeze stays intact.**
  - M3 settings go in a new top-level `intervals:` config section, so the M2 config hash (`b88eae0a04e176af`) and its unit test are unaffected.
  - New code lives in new modules under `src/gridcast/intervals/`, including the quantile model (step 2 placed it there rather than next to `models/ml.py`, so the frozen paths stay unchanged).
  - If a frozen module must change, the change is additive and disclosed, and a test shows the M2 point path is unchanged.

### 2.6 AEMO probabilistic demand (included after step 0)

- **Data (confirmed in step 0, [`docs/m3_data_spike.md`](../m3_data_spike.md)).**
  - Pre-dispatch PASA, `PDPASA_REGIONSOLUTION` v7, from the MMSDM monthly `DATA/` folder: every 30-minute run is archived, with `DEMAND10/50/90` for each 30-minute period.
  - Use `LOR` runs only (`OUTAGE_LRC` was discontinued from 31 July 2025).
  - About 1.1 GB to download for Oct 2024 → Sep 2026. Only VIC1 `LOR` rows are kept, about 83,000 a month.
- **What the band is.**
  - It is exactly symmetric around `DEMAND50` and widens with lead.
  - In the four sampled months it covered 66–78% of actual demand (pooled), so it behaves like an under-covering central band, not a weather scenario.
  - Its coverage and interval score are reported as facts about AEMO's band, without calling it miscalibrated.
- **Centre (user decision, spike §6).**
  - `DEMAND50` sits about +40 MW above the matched pre-dispatch `TOTALDEMAND`, for reasons neither the documentation nor the data reveal.
  - **Headline:** the band as published.
  - **Beside it:** the same half-width re-centred on the matched pre-dispatch `TOTALDEMAND` (published before the PDPASA run).
  - The offset is disclosed.
- **Resolution.** AEMO's values are 30-minute. A 30-minute mean of 5-minute quantiles is not a quantile of the 30-minute mean, so the comparison uses our intervals recalibrated for 30-minute means:
  - conformal on the 30-minute mean residuals;
  - CQR on the period-averaged QR bounds, conformalised on 30-minute scores.
- **Alignment.**
  - *Headline (matched information):* use the run with `RUN_DATETIME = o + 30 min`. A run is identified by its first period end, so its nominal time is *o*, and it is published 4–7 minutes later.
  - *Reported beside it (strict):* the latest run with `LASTCHANGED ≤ o`, that is, the run with `RUN_DATETIME = o`.
  - Runs reach only to the end of the next trading day (64.6% reach 24 h), so coverage is reported by lead.

## 3. Success criteria (pre-registered)

These apply to the **headline CQR (LightGBM C) at 80%, demand, on the interval development period (issue times 1 Aug 2023 → 30 Sep 2025)**. The pairs are the same ones scored for the interval baselines (conformal around persistence, daily naive and weekly naive; same procedure).

| # | Criterion | Pass if |
|---|---|---|
| U1 | Calibrated overall | pooled coverage ∈ [78%, 82%] |
| U2 | Calibrated at every lead time | coverage(h) ∈ [75%, 85%] for every h ∈ [1, 288] |
| U3 | Sharper than the baseline intervals | IS₈₀(h) < min over baseline intervals of IS₈₀(h) for every h ∈ [24, 288], **and** pooled IS₈₀ < mean over h of that minimum |
| U4 | Leakage-free | every interval leakage property test passes (run by the report itself) |
| U5 | Holdout used once | the holdout is scored by a single `gridcast evaluate-holdout-intervals` run against the frozen interval recipe; the command refuses to run again for the same recipe hash |

- **Reported, not pass/fail:**
  - the 50% and 95% levels;
  - conformal-on-point and raw QR;
  - WIS and pinball loss;
  - conditional coverage (lead, hour, month, year);
  - both kinds of extreme-event groups;
  - the QR crossing rate;
  - the AEMO comparison.
- **Holdout replication:** U1–U4 are recomputed on the holdout and reported as *holds* or *does not hold*. They do not change the development verdicts; U5 is the only holdout criterion.
- **Failures are published** in the M3 report and the README. If U4 fails, results are withheld until the cause is found.
- **Disclosures written now:**
  - The candidate grid (including ACI) was designed after M2's development and holdout point errors had been seen, including the winter bias. Development interval numbers do not benefit from in-sample tuning (settings are re-selected from past data only), but the design itself was not blind.

## 4. Steps (vertical slices; one branch, one commit per logical step)

**Step 0. AEMO probabilistic-demand spike.** Output: `docs/m3_data_spike.md`; scripts in `scripts/spikes/m3/`.
- Read the MMS Data Model pages (visualisations.aemo.com.au) for `PDPASA_REGIONSOLUTION` and `STPASA_REGIONSOLUTION`: the POE columns, the run identifiers, and the definition of each demand field.
- Find where every run is archived on NEMWeb (MMSDM `DATA/` holds only some tables' latest runs; check for an `#ALL#` archive), its size for Oct 2024 → Sep 2026, and `LASTCHANGED` against the run time.
- On a sample, compare `DEMAND50` with pre-dispatch `TOTALDEMAND` for the same run and with actual demand, to check the demand definition. Check how often actual demand falls outside the POE90–POE10 band, to see whether it behaves like a quantile or like a scenario.
- **Fallback rules, agreed in advance:**
  - Every run archived and `DEMAND50` on the `TOTALDEMAND` definition → include it as a benchmark (§2.6, step 5).
  - The definition differs by a component available at the run time → include it only with that documented adjustment.
  - Otherwise → drop the AEMO comparison and record why in the spike document and the README.
- *Outcome (2026-11-10):* [`docs/m3_data_spike.md`](../m3_data_spike.md), four sampled months (Oct 2024, Jan 2025, Jul 2025, Sep 2026).
  - Every run is archived, and the timing matches M2's matched-information alignment.
  - The band is symmetric and covers 66–78% of actual demand.
  - `DEMAND50` is about +40 MW above the matched pre-dispatch `TOTALDEMAND`. No column or document explains it, and the data cannot tell a definition difference from a different forecast.
  - The fallback rules did not cover this case, so the user decided: **include the comparison; headline = the band as published; the band re-centred on pre-dispatch beside it; disclose the offset.**
  - The user first chose the re-centred band as the headline, then revised that after the per-month bias table showed re-centring would usually make AEMO's band worse (spike §4, §6).

**Step 1. Interval core and conformal intervals around the stored forecasts.**
- Package `src/gridcast/intervals/`:
  - scores (coverage, width, interval score, WIS, pinball, crossing);
  - rolling signed-score conformal with grouping and ACI;
  - per-retrain stream selection.
- Leakage property tests and their controls (§2.5).
- **Pre-scoring spike** (`scripts/spikes/m3/`, issue times before 2023-08-01 only) to fix:
  - the ACI step values;
  - the time-of-day blocks;
  - the cold threshold.
  Record them in this plan before any scored interval exists.
- *Spike outcome (2026-11-10, [`docs/m3_prescoring_spike.md`](../m3_prescoring_spike.md)); the user chose the recommended option each time:*
  - **Blocks:** 8 × 3 h. Every layout beat per-lead calibration for LightGBM C (IS80 −8% to −15%). The gain flattens at 3 h; 2 h blocks were 0.5% better at 80% but have a third fewer calibration points per block.
  - **ACI:** γ ∈ {0, 0.001, 0.005}. No γ > 0 improved LightGBM C on pre-scoring data, and γ ≥ 0.01 under-covered long leads. Small steps did help persistence and weekly naive, and the per-retrain selection can still choose γ = 0.
  - **Cold:** forecast daily maximum ≤ 12 °C.
- Migration `0005_intervals`, if needed (for example, allow `model_kind = 'interval'`).
- `gridcast intervals-conformal` stores interval batches (parquet bounds per level under the data root, plus `model_runs` rows) for LightGBM C and the three baselines.
- *Outcome (2026-11-10):* the core and the spike are in commit `819e862`. Migration 0005 allows `model_kind = 'interval'`.
  - `gridcast intervals-conformal` ran from that commit in 308 s. It stored batch `72eb5209-9bef-4870-a375-131a86ee7e0f` (interval config hash `d34273400899af69`): 24 candidate streams per point model and 108 selections each (9 retrains × 3 levels × 4 bands), with no empty intervals.
  - Storage is 1.4 GB, above the §5 estimate of about 1 GB.
  - Results are only printed and stored at this step; the step-4 report evaluates them.

**Step 2. Quantile LightGBM walk-forward (development).**
- Quantile model with per-quantile selection at every retrain (§2.3; first written as reusing run 2's per-retrain selections, changed 2026-11-11, §1).
- Smoke run on the first retrain (2022-11-01) to measure the cost, then the full development run from a clean commit (an overnight run, logged under `<data root>/logs/`).
- `gridcast backtest-qr` stores the QR batch with fits, tree counts and crossing rates.
- *Outcome (2026-11-11):* `gridcast backtest-qr` ran from a worktree at `6fdbac5` in 28,059 s (7.8 h; folds 19.5–64.2 min, the 2025-08-01 retrain the longest). It stored QR batch **`4bab11d7-4301-4c74-9f32-ddd9644f3921`** (run 85; QR config hash `8767a8948966637c`), 505 MB of quantiles under `<data root>/intervals/` and 265 MB of models under `<data root>/models/`. Log: `<data root>/logs/m3/backtest-qr-20261111-0123.log`.
  - **Point reproduction passed** in all 12 folds: 48 refits, 51,120 issue times, 14,436,480 pairs equal to run 2's stored forecasts.
  - **Selections:** of the 288 (retrain, band, quantile) choices, 201 chose `latest`, 43 `direct`, 26 `residual` and 18 `anchored`; 228 chose 31 leaves. 110 matched the point model's form and leaf count.
  - **Crossing before sorting:** 23.4% of all pairs (22.4% of scored pairs); 18.1–30.0% per fold, 1.3–4.8% in the 1–6 lead band.
  - **Raw QR, pooled** (reported only; CQR in step 3 is what U1–U3 judge), on the 10,662,528 scored pairs step 1 used:

    | run | cov50 % | cov80 % | cov95 % | IS80 | WIS |
    |---|---|---|---|---|---|
    | qr_lgbm_C | 35.8 | 64.7 | 87.6 | 1382.2 | 159.6 |

    Raw QR under-covers at every level and misses on both sides, more often above (80%: 18.8% of actuals above the upper bound, 16.5% below).

**Step 3. CQR.**
- Apply the step-1 machinery to the QR scores. Store the CQR batch.
- *Implementation (step 3a, [`src/gridcast/intervals/cqr_batch.py`](../../src/gridcast/intervals/cqr_batch.py), `gridcast intervals-cqr`, `make intervals-cqr`):*
  - *Inputs:* the QR batch named by `intervals.qr_batch`, and run 2's batch for the actuals, the common mask and lgbm_C's point forecast (the WIS median).
    - The run refuses a QR batch that is incomplete, failed its point reproduction, was trained against another source batch, or has a QR config hash other than the current recipe's.
    - It also refuses one whose issue times or actuals differ from run 2's, or that lacks a quantile on any pair of the common mask.
    - The CLI refuses a working tree with uncommitted changes, since U1–U3 judge this batch.
  - *Method:* step 1's `quantile_problem` (one score per side, q̂_lo − y and y − q̂_hi) on the rearranged quantiles of each level. It runs the same 24 candidate streams with the same per-retrain selection.
    - Calibration starts at the first development issue time (2022-11-01), as step 1's did. The pairs are run 2's common mask, so CQR gets no more calibration data than the conformal intervals.
  - *Parallel by level:* levels share no state, so each runs in its own process. A unit test checks that the merged parts equal a single run over all levels.
  - *Config:* `intervals.qr_batch` is an input pointer, excluded from the interval config hash like `output_dir`. The hash therefore stays `d34273400899af69`, the value step 1's and step 2's batches recorded, and a unit test pins it.
  - *Stored metrics:* step 1's metrics, plus, per quantile level of the corrected bounds, the pinball loss and the share of actuals at or below each bound (for the reliability diagram). Also stored: the share of pairs whose levels are not nested (per-level corrections can break the nesting).
- *Outcome (2026-11-11, step 3b):* `gridcast intervals-cqr` ran from `8168a39` in 160 s and stored CQR batch **`484194d7-2e62-4176-80b0-b8f117354929`** (run 86; interval config hash `d34273400899af69`; QR batch `4bab11d7…`). It wrote 383 MB under `<data root>/intervals/`; log `<data root>/logs/m3/intervals-cqr-20261111.log`.
  - **Pooled development results** (reported only here; step 4's report computes U1–U4), on the 10,662,528 pairs scored in steps 1 and 2. The first two rows are from those steps' outputs:

    | run | cov50 % | cov80 % | cov95 % | IS80 | WIS |
    |---|---|---|---|---|---|
    | conformal_lgbm_C (step 1) | 50.3 | 79.1 | 94.4 | 1267.2 | 150.6 |
    | qr_lgbm_C, raw (step 2) | 35.8 | 64.7 | 87.6 | 1382.2 | 159.6 |
    | **cqr_lgbm_C** | 49.7 | 79.4 | 93.9 | 1248.1 | 150.5 |

  - **Selections** (108 = 9 retrains × 3 levels × 4 bands): 101 chose the time-of-day grouping, against all 108 for conformal-around-C. An ACI stream (γ > 0) won 25, 22 and 17 of the 36 selections at 50%, 80% and 95%.
  - **Crossed intervals** (output at the midpoint): 2,714 at 50%, 11 at 80%, 1,482 at 95%, out of 14.7 million output pairs per level.
  - **Not nested:** 14.3% of scored pairs have a wider level's bound inside a narrower level's, against 1.4% for conformal-around-C (computed from step 1's stored bounds for this comparison). The raw QR quantiles were rearranged and are always nested, so the cause is the per-level correction: each level has its own scores, ACI state and selection.
    - Most of these pairs are between 80% and 95%: the 95% lower bound sits above the 80% one on 5.5% of pairs, and the upper bound below it on 4.1%.
    - Each level's coverage and interval score are unaffected. The WIS and the reliability diagram combine levels, so step 4 should disclose this. Reporting a rearranged version beside them would be a new, reported-only choice for step 4's question round.
  - Clipped bounds: at most 12.0% of one candidate's 95% upper bounds (`w4_lead_tod_g0.005`), against 12.9% for the same candidate in step 1.

**Step 4. Development report.**
- `gridcast report-m3` writes `reports/m3_intervals.md` and its figures. It computes U1–U4 in code and runs the leakage tests itself.
- Contents: coverage and width by lead and level; the reliability diagram; the interval score against the baseline envelope; WIS; conditional coverage; both extreme-event tables with bootstrap intervals; the per-retrain selection summary.
- *Decisions (question round, 2026-11-12, before the report was built). The user chose the recommended option each time:*
  - **Non-nesting (step 3):** disclose it (share of pairs, which levels, the cause). Also add a reported-only row, **CQR, rearranged**: per pair, the six CQR bounds sorted as the quantiles τ = 0.025 … 0.975, as step 2 did for raw QR. It appears in the pooled, WIS, pinball and reliability tables, and step 6 scores it too, so development and holdout match. It does not replace the headline, and U1–U3 judge the CQR bounds as stored. *Why:* it shows whether coherent levels cost accuracy, and an API (M5) would need nested intervals. *Disclosed:* it was added after step 3's results were seen.
  - **Figures (four):** 80% coverage by lead for every method, with U2's 75–85% band; IS80 by lead, CQR against the baseline envelope (U3); the reliability diagram; 80% coverage by target month for CQR, conformal-around-C and raw QR.
  - **Bootstrap:** M2's settings throughout: 2,000 draws resampling whole NEM issue days, 95% percentile intervals, seed `ml.seed`. A **sensitivity with 7-day blocks** (consecutive weeks from the first scored issue day) is added for the pooled headline rows: CQR's coverage at each level, and its IS80 and WIS differences against conformal-around-C and the best baseline. *Why:* misses cluster over weather spells and 24 h leads overlap the next issue day, so day blocks may give intervals that are too narrow.
- *Details settled without a question (stated in the report):*
  - U3's pooled part compares with the mean over **all** leads h = 1 … 288 of the per-lead baseline minimum, as M2's S2 did.
  - **Hot and cold groups, known at issue time:** for each pair, the population-weighted GFS temperature of every 5-minute interval of the target's local day, using the freshest forecast published by the issue time (the rule C's features use). The day's maximum is taken over the intervals that have such a value. A pair whose target day has none is in neither group.
  - Hour, month and year groups use the target's local time; the retrain quarter uses the issue time, since selections apply to issue times.
  - The conditional and extreme-event tables show CQR, conformal-around-C, raw QR and the **best baseline**, the conformal baseline with the lowest pooled IS80.
- *Implementation (step 4a, [`src/gridcast/intervals/evaluation.py`](../../src/gridcast/intervals/evaluation.py), [`report.py`](../../src/gridcast/intervals/report.py), `gridcast report-m3`, `make report-m3`):*
  - *Inputs:* the latest CQR run built on `intervals.qr_batch`, that QR run, and the latest conformal runs, which must come from one batch. Every run must carry the current interval config hash and `intervals.source_batch`, or the report stops.
  - *Recomputation:* every per-pair statistic is recomputed from the stored bounds on run 2's common mask, then summed by NEM issue day and lead (the bootstrap's material), by target hour and month, and by event group. **The pooled values must equal each batch's stored metrics** (relative tolerance 10⁻⁹), or the report stops. The rearranged CQR row has no stored metrics; it is computed here from the stored CQR bounds.
  - *U4:* the command runs `tests/unit/test_interval_leakage.py` and the QR leakage test (`test_quantiles_ignore_demand_after_their_issue_time`).
  - *Presentation rules (no effect on any criterion):* event-group coverage gets a bootstrap interval only if its pairs come from at least 10 issue days. Target months with fewer than 7 days are tabled but left out of the month figure.
  - *Tests:* `tests/unit/test_m3_evaluation.py` (rearrangement, aggregation against brute force, bootstrap, U1–U4 boundaries, and the hot/cold label's information rule with a control). `tests/integration/test_m3_report.py` builds every batch on the test database with the real commands (QR faked with stored metrics). It then writes the report, and checks that the report stops when a stored metric differs or the recipe hash changed.

- *Outcome (2026-11-12, step 4b):* `make report-m3` ran from `02681d6` (step 4a is `f806044` plus `02681d6`, a generated note on months with missing target days). It takes about 1 min, including the 7 leakage tests (68 s measured from `f806044`). It wrote [`reports/m3_intervals.md`](../../reports/m3_intervals.md) and four figures. Every pooled number it recomputed equalled the stored metrics. **U1–U4 pass:**
  - **U1:** pooled 80% coverage 79.40% (95% CI 78.48–80.28% with day blocks, 78.22–80.59% with 7-day blocks).
  - **U2:** 288 of 288 leads within 75–85%; lowest 78.97% (18 h), highest 80.36% (5 h 55 min).
  - **U3:** 265 of 265 leads from 2 h; closest at 2 h 5 min (926.8 vs 1,574.6 MW, persistence); pooled 1,248.1 vs 2,256.1 MW.
  - **U4:** 7 leakage tests passed, 0 failed.
  - Reported only:
    - CQR beats conformal-around-C on IS80 by 19.0 MW (95% CI 6.8–32.2 MW with day blocks, 1.2–36.6 MW with 7-day blocks). It does not beat it on WIS: −0.1 MW (−0.9 to +0.7).
    - By target month, 80% coverage ranges from 67.6% (Dec 2023) to 87.8% (Apr 2025).
    - On days with a forecast maximum ≥ 35 °C, coverage is 71.5% (67.1–75.9%), missing above on 23.9% of pairs.
    - Rearranging CQR's bounds changes its WIS by −0.30 MW (−0.35 to −0.26).

**Step 5. Freeze, then the AEMO comparison (development part).**
- Freeze the interval recipe: a hash of the `intervals:` config, plus the code commit, recorded in config and guarded by a unit test. This happens before the AEMO comparison, so the benchmark cannot influence the recipe. What is frozen is the recipe, including the per-retrain selection rule, not one set of fitted models.
- Load AEMO's POE data (`PDPASA_REGIONSOLUTION`, `LOR` runs, VIC1; migration `0006` if needed; about 1.1 GB cached under the data root) and compare on Oct 2024 → Sep 2025 (§2.6): the band as published as the headline, the re-centred band beside it, and both alignments. `reports/m3_aemo.md`.
- *Decisions (question round, 2026-11-13, before any step-5 code). The user chose the recommended option each time:*
  - **Order:** first build the 30-minute recalibration (§2.6) and store its development batch, from our own data only. Then freeze both resolutions. Only then download `PDPASA_REGIONSOLUTION`. Nothing that faces AEMO can change once the full AEMO data are on disk. *Disclosed:* step 0 saw four sampled months of AEMO's band.
  - **The rearranged row is frozen:** the per-pair sort moves into a frozen module, so step 6 scores it with identical code. It stays reported only.
  - **Our intervals against AEMO:** CQR (the headline), conformal-around-C and the best conformal baseline, all recalibrated at 30 minutes. Scored at 80% only, since AEMO publishes only that band.
  - **Layout: M2's AEMO report.**
    - A per-lead table (30 min, 1, 2, 4, 6, 12, 18 and 24 h): coverage, share below and above, mean width and IS80. The headline column is the band as published, with matched alignment.
    - A sensitivity table: the re-centred band and the strict alignment.
    - Pooled IS80 and coverage differences, with the day-block bootstrap and the 7-day sensitivity.
    - A table by issue quarter, and figures of coverage and IS80 over all 48 half-hour leads.
    - Scored on the pairs where both alignments exist, as in M2.
- *Details settled without a question:*
  - **The 30-minute recipe is the interval recipe applied to half-hour periods.**
    - *Target:* the mean of the six 5-minute actuals of the period ending o + 30k min (k = 1 … 48), as in M2's AEMO comparison.
    - *Inputs:* the point forecast is the mean of the six 5-minute forecasts. CQR's quantiles are the means of the six rearranged 5-minute QR quantiles.
    - *Scored pairs:* a period is scored and used for calibration only if all six of its 5-minute pairs are in the common mask.
    - *Settings:* the same candidate grid, the same time-of-day blocks (of the period's start), the same ACI steps and the same per-retrain selection.
    - *Lead bands:* mapped to whole periods: 30 min, 1–2 h, 2.5–6 h and 6.5–24 h (periods 1, 2–4, 5–12, 13–48).
    - *Levels:* all three are built, since the recipe works per level; the AEMO comparison uses 80%.
  - **Frozen code:** M2's frozen paths, plus every `intervals` module that produces bounds (`batch`, `conformal`, `cqr_batch`, `halfhour`, `inputs`, `qr_batch`, `quantile`, `rolling`, `scores`). Not frozen: the report modules (`evaluation`, `report`), the freeze check, and the AEMO code written after the freeze.
  - **Check before the freeze commit:** the frozen code recomputes the step-1 and step-3 development bounds, and they must equal the stored batches exactly. `batch.py` and `inputs.py` were refactored after step 1, so this check is a real test. The QR code has not changed since the QR batch was built; step 6 reproduces its Aug–Sep 2025 forecasts.
- *Outcome (2026-11-13, steps 5a and 5b):*
  - **5a, `bc5337e`:** [`src/gridcast/intervals/halfhour.py`](../../src/gridcast/intervals/halfhour.py) (`gridcast intervals-halfhour`) and [`freeze.py`](../../src/gridcast/intervals/freeze.py) (`gridcast intervals-freeze-check`).
    - The conformal core takes the lead step length as a parameter (default 5 minutes, so 5-minute results are unchanged); `rearrange_bounds` moved to `scores.py`.
    - Tests: half-hour leakage properties and their controls in the U4 file (10 tests, against 7 in step 4), the half-hour run against the slow reference, the batch end to end, and the reproduction check catching a changed bound.
  - **Half-hour batch** `27aa78d6-8ab8-4509-aa40-d8b08cda68ea` (runs 87–91), built from `bc5337e` in 153 s; log `<data root>/logs/m3/intervals-halfhour-20261113.log`. It scores 1,777,080 periods from 2023-08-01. Pooled development results, printed by the command (reported only):

    | run | cov50 % | cov80 % | cov95 % | IS80 | WIS |
    |---|---|---|---|---|---|
    | conformal30_lgbm_C | 50.2 | 78.8 | 94.4 | 1249.9 | 148.4 |
    | conformal30_persistence | 49.4 | 79.7 | 94.0 | 2707.0 | 383.8 |
    | conformal30_daily_naive | 49.4 | 79.6 | 94.1 | 2395.2 | 280.1 |
    | conformal30_weekly_naive | 50.8 | 78.2 | 93.0 | 2743.0 | 321.1 |
    | cqr30_lgbm_C | 49.7 | 79.4 | 93.8 | 1230.5 | 148.3 |

  - **Reproduction check**, run from a worktree at `bc5337e` (458 s; log `<data root>/logs/m3/intervals-freeze-check-20261113.log`): the step-1 conformal batch (4 runs) and the step-3 CQR batch, 38,016 issue times each, **reproduced exactly**, bounds and selections.
  - **5b, the freeze:** `intervals.frozen_config_hash` = `d34273400899af69`, `intervals.frozen_qr_config_hash` = `8767a8948966637c`, `intervals.frozen_commit` = `bc5337e` (`configs/base.yaml`, excluded from the hash). Unit tests pin both hashes and check that no frozen path has changed since `bc5337e`. Only after this commit is `PDPASA_REGIONSOLUTION` downloaded.
- *Outcome (2026-11-13, steps 5c and 5d):*
  - **5c, `526f048`:** migration 0006 (`aemo_demand_poe`), [`src/gridcast/ingestion/aemo_poe.py`](../../src/gridcast/ingestion/aemo_poe.py) (`gridcast ingest-aemo-poe`) and [`src/gridcast/intervals/aemo_report.py`](../../src/gridcast/intervals/aemo_report.py) (`gridcast report-m3-aemo`).
    - Both were written after the freeze and before the data were loaded, and neither is frozen.
    - The report refuses to run without the freeze record, or on runs from another recipe or commit. It stops unless every pooled 80% statistic recomputed from the stored bounds equals the stored metric.
  - **Load:** Sep 2024 → Sep 2026 (25 months, table version 7 throughout, `LOR` runs only) in under 2 minutes; 1.1 GB cached under `<data root>/raw/mmsdm/PDPASA_REGIONSOLUTION/`. Log: `<data root>/logs/m3/ingest-aemo-poe-20261113.log`.
  - **5d:** `make report-m3-aemo` ran from `526f048` and wrote [`reports/m3_aemo.md`](../../reports/m3_aemo.md) and two figures.
    - The window has 17,519 of 17,520 runs and 785,067 compared pairs on 365 days.
    - Reported only:
      - **Coverage:** AEMO's band (matched) contains 70.9% of actual demand, CQR's 80% interval 80.2%.
      - **Width:** CQR's interval is wider (961.0 against 433.3 MW).
      - **IS80:** CQR's is higher (1,321.5 against 925.7 MW, +395.8 MW, 95% CI +348.8 to +444.4 with day blocks). CQR scores better only at the first 4 of 48 half-hour leads, up to 2 h.
    - This matches M2's finding that AEMO's pre-dispatch centre is far more accurate beyond 1 h. The re-centred band and the strict alignment do not change the picture.
    - `DEMAND50` averages +46.4 MW above the matched pre-dispatch `TOTALDEMAND`; the band is symmetric on every row.

**Step 6. Holdout, scored once.**
- `gridcast evaluate-holdout-intervals` refuses to start unless:
  - the recipe hash matches the freeze;
  - the working tree is clean;
  - the frozen code is unchanged;
  - the data reach the last holdout target;
  - no interval holdout batch exists for the recipe (the U5 lock).
- It runs QR from the 2025-08-01 retrain with step 2's per-quantile selection. The point refits for the reproduction check use the holdout point model's stored selections and tree counts.
  - Before scoring anything, its Aug–Sep 2025 QR forecasts and fits must reproduce step 2's stored ones exactly; otherwise it stops.
  - It then builds conformal intervals and CQR on the holdout, with calibration windows that reach back into development data.
  - It runs from a git worktree, as M2's did.
- `gridcast report-m3-holdout` writes `reports/m3_holdout.md`: the replication verdicts, everything in step 4, and the holdout part of the AEMO comparison.
- Rewrite the README's *Uncertainty* section and its status, results, limitations and reproduction sections from the generated reports.
- *Decisions (question round, 2026-11-13, before any step-6 code and before any holdout interval or quantile existed). The user chose the recommended option each time:*
  - **Calibration history: one continuous run.**
    - The frozen recipe runs once over issue times 2022-11-01 → 2026-09-30, as if it had been deployed continuously. Calibration windows, ACI state and selections carry across the boundary.
    - Its point inputs are run 2's stored forecasts before 2025-10-01 joined to `holdout_main`'s from 2025-10-01. Both batches hold the same models, so the common mask is defined the same way on both sides.
    - Its QR inputs are step 2's quantiles before 2025-10-01 joined to the holdout QR quantiles from 2025-10-01.
    - M2's holdout run reproduced run 2's 2025-08-01 fold exactly, and step 6's QR run must do the same. So the joined inputs equal what one continuous walk-forward would have produced.
    - Only issue times from 2025-10-01 are scored, against `holdout_main`'s stored actuals and common mask.
  - **Reproduction before anything is scored: everything.** Any mismatch stops the run.
    - *QR:* the 2025-08-01 retrain's fit records (rows, every candidate's pinball loss and tree count, the selection) and its rearranged Aug–Sep 2025 quantiles must equal step 2's stored ones.
    - *Point refits:* at every holdout retrain they must equal run 2's forecasts (Aug–Sep 2025) and `holdout_main`'s (from Oct 2025).
    - *Intervals:* the development part of the continuous run must equal every stored development batch exactly, bounds and selections: step 1's conformal batch (4 runs), step 3's CQR and step 5a's half-hour batch (5 runs). This also checks the new holdout wrapper around the frozen core.
  - **Two stages** (the plan above had one command):
    - **6a, `gridcast backtest-qr-holdout`.** QR from the 2025-08-01 retrain through the holdout (5 retrains, step 2's per-quantile selection). The first retrain runs alone and its reproduction check runs before the other four are fitted.
      - It stores the rearranged quantiles and the per-pair crossing flags, with no metric computed or logged.
      - It refuses unless the freeze holds (both recipe hashes, a clean tree, frozen code unchanged since `bc5337e`). It also refuses if a QR holdout batch already exists for the QR recipe.
    - **6b, `gridcast evaluate-holdout-intervals`, the U5 lock.** Everything that scores happens here. *Why two stages:* a bug in the scoring stage then costs minutes, not a rerun of the 4–6 h QR fits. The QR quantiles are forecasts, not scores.
  - **Lock scope: everything under one lock.** The locked run stores, together and only after checking the lock again:
    - 5-minute conformal intervals around C and the three baselines;
    - CQR;
    - raw QR's holdout metrics;
    - the half-hour batch (5 runs) that the AEMO holdout part reads.
    - The *CQR, rearranged* row is computed by the report from the stored CQR bounds, with the frozen `rearrange_bounds`, as in step 4.
  - **`reports/m3_holdout.md` contents, fixed now (M2's holdout report as the model):**
    - provenance;
    - U5;
    - U1–U4 recomputed on the holdout (*holds* or *does not hold*);
    - a development-vs-holdout table;
    - everything step 4 reports, on the holdout;
    - the AEMO holdout part, with step 5's layout on Oct 2025 → Sep 2026 and the pooled differences shown for development and holdout;
    - **a pre-registered winter section.** M2's holdout found a winter under-forecast (bias −104 MW, worst in Jun 2026), the known stress test. For target months Jun–Aug 2026 it shows 80% coverage and the share below and above by lead band, for CQR, conformal-around-C, raw QR and the best baseline, with day-block intervals. It also shows the ACI step γ selected per band at the 2026-05-01 retrain.
    - Anything added after scoring is labelled as such, as M2 did.
- *Details settled without a question:*
  - *Best baseline:* the same rule as step 4, the conformal baseline with the lowest pooled IS80, applied to the holdout. That gives the baselines their strongest showing.
  - *Order:* 6a's code is committed and its run started first. The scoring and report code (6b, 6c) is written and committed while it runs, before any holdout interval is scored. The QR quantiles on disk are not scored outside 6b.
- *Implementation (2026-11-13, before any holdout interval was scored):*
  - **6a, `3cc2528`:** [`src/gridcast/intervals/qr_holdout.py`](../../src/gridcast/intervals/qr_holdout.py), `gridcast backtest-qr-holdout`.
    - It fits the first retrain alone and checks it against step 2's batch before the other four retrains: the fit records (rows, every candidate's pinball loss and tree count, the selection) and the rearranged quantiles of Aug–Sep 2025. Then it fits the other four.
    - It stores the quantiles and per-pair crossing flags with a `model_runs` row (`intervals_holdout_qr`, `scored: false`) and no `forecast_metrics`.
    - Tests: `tests/integration/test_qr_holdout.py`, which builds M2's development and holdout runs and step 2's batch on synthetic data. Its control changes one stored development quantile, and the run must stop before the second retrain.
  - **6b, `d19778d`:** [`src/gridcast/intervals/holdout.py`](../../src/gridcast/intervals/holdout.py), `gridcast evaluate-holdout-intervals`, the U5 lock.
    - The workers mirror `conformal_for`, `cqr_for` and `period_run` on the joined inputs and call the frozen functions unchanged.
    - The reproduction compares every development result's bounds and its selections at the development retrains. Holdout metrics are stored with split `holdout`, the bounds under `<batch>/5min/` and `<batch>/30min/`.
    - Tests: `tests/integration/test_holdout_intervals.py`, with development batches built by the real commands. Its controls change one stored development bound or one overlapping QR quantile, and nothing may be stored.
  - **6c, `89b8165`:** [`src/gridcast/intervals/holdout_report.py`](../../src/gridcast/intervals/holdout_report.py), `gridcast report-m3-holdout`.
    - `report.py` and `aemo_report.py` now take the evaluation set, split, pairs and holdout runs as parameters. Regenerating both development reports from the changed code gave identical numbers and byte-identical figures. Only the leakage-test count changed (7 → 11, from step 5a's half-hour tests), and the generated date and commit lines.
    - The *retrain whose quarter holds June* is derived: the latest selection retrain on or before 1 June 2026, which is 2026-05-01.
  - Both CLIs refuse a dirty tree. On the real database, `evaluate-holdout-intervals --check-only` refused only because step 6a's batch did not exist yet.
- *Outcome (2026-11-13):*
  - **6a run:** `backtest-qr-holdout` ran from a worktree at `3cc2528` in 6.7 h (log `<data root>/logs/m3/backtest-qr-holdout-20261113-1201.log`).
    - The 2025-08-01 retrain reproduced step 2 exactly: 4 fit records and 5,059,584 quantile values of 2,928 development issue times.
    - The point refits equalled run 2's and `holdout_main`'s forecasts at all 5 retrains (5,889,024 pairs).
    - It stored QR holdout batch **`b226706b-0bed-476c-b4fc-6c283786a8cc`** (run 92) unscored: 209 MB of quantiles and 219 MB of models. The worktree was removed.
    - Facts about the forecasts, recorded before scoring:
      - At the 2026-08-01 retrain, four quantile fits early-stopped at or within 7 rounds of the 5,000-round cap (4,993, 4,993, 4,998 and 5,000 trees) (`ml.lightgbm.max_rounds`).
      - The crossing share before sorting was 38–48% in the two longest bands at the last two retrains, against 17–32% at the first three.
  - **6b, the one look:** `evaluate-holdout-intervals` ran from `074e900` in 14.9 min (log `<data root>/logs/m3/evaluate-holdout-intervals-20261113.log`).
    - All 10 development interval runs reproduced exactly: 38,016 issue times and 108 selections each. Step 6a's quantiles equalled step 2's on the overlap.
    - It stored holdout batch **`2744fd09-4e6d-41c5-9579-a13c8fe9a5b0`** (runs 93–103).
  - **6c:** `make report-m3-holdout` ran from `074e900` (about 1 min) and wrote [`reports/m3_holdout.md`](../../reports/m3_holdout.md) and six figures (`7607853`). **U5 passes, and U1–U4 hold on the holdout:**
    - U1: 79.69% (95% CI 78.17–81.08%).
    - U2: 288 of 288 leads, 79.24–80.51%.
    - U3: 265 of 265 leads; pooled 1,365.2 vs 2,335.6 MW.
    - U4: 11 tests passed.
  - Reported only:
    - CQR beats conformal-around-C on IS80 by 25.1 MW (CI 5.3–45.1), but not on WIS (+0.1, −1.2 to +1.4).
    - Winter (Jun–Aug 2026): 79.1% (76.1–81.9%), misses 11.2% below and 9.7% above.
    - Hot days (forecast ≥ 35 °C): 73.1%, missing above on 22.1%.
    - Against AEMO's band, which covers 69.2%: CQR's IS80 is +327.1 MW (+270.0 to +386.8). CQR is better only up to 2 h.
  - **README** rewritten from the three M3 reports (`aca5b66`).

## 5. New dependencies and compute

- **Dependencies:** none expected. LightGBM's `quantile` objective, numpy and pandas are enough.
- **Conformal:** minutes to tens of minutes per batch. Every candidate stream for 288 leads, three levels and four point models (C plus three baselines) is vectorised over leads.
- **QR, development:** with per-quantile selection (§1), 6 quantiles × 4 bands × (8 early-stopping fits + 1 refit) × 12 retrains = 2,592 fits, plus 48 point refits for the reproduction check.
  - *Planned estimate (reusing the point settings, 576 fits):* 4–6 h for development and about 2 h for the holdout's 5 retrains.
  - *Measured (step-2 smoke runs, first retrain):* 203 s reusing the point settings (projected 1.5 h for the full run); 1,206 s with per-quantile selection, projected **9.0 h** scaled by training rows and **11.0 h** scaled by run 2's own fit-time growth.
  - *Measured (step-2 full run):* 7.8 h. The longest fold, the 2025-08-01 retrain, took 64 min. The holdout's 5 retrains train on more rows, so they will take longer than the 2 h planned; at 45–75 min a retrain, about 4–6 h (a projection).
- **Storage:** about 1 GB of interval bounds and about 0.5 GB of QR model files under the data root, gitignored. *Measured:* 1.4 GB for step 1; 505 MB of quantiles and 265 MB of models for step 2.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Errors drift (the winter under-forecast), so past residuals stop describing future ones | Rolling windows and ACI candidates, asymmetric bounds, coverage by month; the holdout winter is a known stress test, disclosed |
| Correlated errors make the coverage numbers look more precise than they are | Block bootstrap by day; tolerances (±2 pp pooled, ±5 pp per lead) chosen with this in mind |
| The ACI feedback for lead *h* arrives *h* later | Updates use only outcomes known by *o*; covered by the leakage tests |
| Tail quantiles would prefer other settings than the point model's | *Originally accepted to save compute.* Resolved on 2026-11-11: each quantile now selects its own setting at every retrain (§1), since the smoke run showed the compute was cheap. Raw QR calibration is still reported, and CQR corrects coverage |
| Per-quantile settings make crossing more likely (different target forms per quantile) | Rearrangement; the crossing share before sorting is recorded per (retrain, band) and per lead |
| QR quantiles cross | Rearrangement; crossing rate reported |
| QR costs more than estimated | Smoke run first; ask again above about 10 h. The development run took 7.8 h (step 2) |
| AEMO's POE band is symmetric, under-covers, and its centre is offset ~+40 MW from `TOTALDEMAND` for unknown reasons (step 0) | Reported as facts about AEMO's band, not as a calibration contest; the band as published is the headline with a re-centred sensitivity beside it; the offset is disclosed |
| M3 code accidentally changes the frozen M2 behaviour | New modules and a new config section; a test that the M2 point path is unchanged |

## 7. Out of scope for M3

- Price ML and price intervals (M2b).
- Anomaly detection and SHAP (M4).
- The API (M5), including writing intervals to the `forecasts` table.
- The dashboard (M6) and monitoring (M7).
- Regions other than Victoria.
- Live ingestion.
- A pre-registered evaluation of C⁺ (the M2 follow-up candidate).

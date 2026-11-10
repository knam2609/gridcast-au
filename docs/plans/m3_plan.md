# Milestone 3 plan: prediction intervals for demand

**Goal:** give the M2 demand forecast honest uncertainty: prediction intervals whose coverage matches their stated level overall, at every lead time and during extreme events, that are sharper than intervals built around the baselines, and that are compared with AEMO's own probabilistic demand forecasts where that comparison is valid.

**Status:** planned 2026-11-10 after a question round with the user. Step 0 done (2026-11-10, [`docs/m3_data_spike.md`](../m3_data_spike.md)): AEMO's POE forecasts are usable, with a disclosed caveat. Branch `m3-prediction-intervals`, from `main` at `a1b26bb` (M1 and M2 merged). The success criteria in §3 were agreed before any interval result existed.

## 1. Decisions (agreed 2026-11-10)

| Topic | Decision |
|---|---|
| Target | Victorian demand only, around the frozen M2 forecasts. Price intervals wait for price ML (M2b) |
| Methods | (1) **Conformal around the stored point forecasts**: rolling calibration on past residuals, no retraining. (2) **LightGBM quantile regression (QR)**: a new walk-forward run for variant C. (3) **Conformalised QR (CQR)**: the QR bounds corrected by the same rolling conformal machinery. All three are compared on identical (issue time, lead) pairs |
| Headline | **CQR on LightGBM C**, pre-registered. U1–U3 judge it; the other methods are reported |
| Interval baselines | Conformal intervals around persistence, daily naive and weekly naive, built with the same procedure as method (1) |
| Levels | **80% headline** (AEMO's POE90–POE10 band); 50% and 95% also reported. QR quantiles: 0.025, 0.1, 0.25, 0.75, 0.9, 0.975. The median used in the weighted interval score is the point forecast |
| Conformal settings | **Re-selected at every quarterly retrain** from past data only, as in M2: every candidate setting is scored on the last 8 weeks before the retrain and the best one is used for the next quarter |
| QR tuning | **Reuse the point model's choices:** at each retrain and lead band, the target form and leaf count that M2's point model chose; only the objective changes to quantile. Early stopping by pinball loss |
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
  - Method (1) uses the signed residual e = y − ŷ.
  - Method (3) (CQR) uses s = max(q̂_lo − y, y − q̂_hi).
  - The lower and upper bounds come from separate quantiles of the score, so the intervals can be asymmetric. They correct for a bias such as the winter under-forecast, rather than only widening.
- **Information rule.**
  - At issue time *o*, the calibration set for lead *h* holds only pairs (o′, h) whose target is known at *o* (o′ + h ≤ o), within the trailing window.
  - Intervals at *o* change only when data ending at or before *o* change. Property tests check this (§2.5).
- **Candidate grid (fixed in step 1 on pre-scoring data, recorded before scoring):**
  - window length ∈ {4, 8, 13, 26} weeks;
  - grouping ∈ {per lead; per lead × a time-of-day block of the target};
  - adaptive conformal inference (ACI, Gibbs & Candès 2021): step γ ∈ {0 (off), and up to two positive values}. ACI updates each lead's miscoverage target with only the coverage outcomes known by *o*, so it reacts to drift with a delay of *h*.
- **Selection.**
  - All candidate streams are computed continuously from the warm-up onwards, each using past data only.
  - At each quarterly retrain, for each level and each M2 lead band (5–30 min, 35 min–2 h, 2 h 5 min–6 h, 6 h 5 min–24 h), the stream with the lowest mean interval score over the last 8 weeks is used for the next quarter. Ties go to the first candidate in grid order.
  - The same rule applies to the interval baselines.

### 2.3 Quantile LightGBM (method 2)

- **Setup.**
  - Same features (variant C), lead bands, training sample, retrain schedule and walk-forward harness as M2.
  - At each retrain and band, the target form and `num_leaves` are copied from run 2's stored selection records. The objective becomes `quantile` with α ∈ {0.025, 0.1, 0.25, 0.75, 0.9, 0.975}.
  - Each quantile early-stops by pinball loss on the last 8 weeks of the window and is then refit on the whole window, as M2's point models were.
  - A quantile of y − anchor plus an anchor known at *o* is a quantile of y, so every target form carries over.
- **Crossing:** quantiles are sorted per pair (rearrangement), and the crossing rate before sorting is reported.
- **Cost:** measured on the first retrain in a smoke run before the full run (§5). If the full run looks longer than about 10 h, ask the user again before starting it.

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
- **Uncertainty of the coverage numbers:** errors are correlated in time, so coverage and score differences get **block-bootstrap intervals that resample whole days**, as in M2's AEMO comparison.
- **Extreme-event groups.**
  - *Known at issue time:* forecast daily maximum ≥ 35 °C (weighted GFS temperature); a cold threshold, fixed in step 1 from pre-scoring data; public holidays; DST-change days.
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
  - New code lives in new modules (`src/gridcast/intervals/`, plus a quantile model next to `models/ml.py`).
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
- Migration `0005_intervals`, if needed (for example, allow `model_kind = 'interval'`).
- `gridcast intervals-conformal` stores interval batches (parquet bounds per level under the data root, plus `model_runs` rows) for LightGBM C and the three baselines.

**Step 2. Quantile LightGBM walk-forward (development).**
- Quantile model; reuse run 2's per-retrain selections (§2.3).
- Smoke run on the first retrain (2022-11-01) to measure the cost, then the full development run from a clean commit (an overnight run, logged under `<data root>/logs/`).
- `gridcast backtest-qr` stores the QR batch with fits, tree counts and crossing rates.

**Step 3. CQR.**
- Apply the step-1 machinery to the QR scores. Store the CQR batch.

**Step 4. Development report.**
- `gridcast report-m3` writes `reports/m3_intervals.md` and its figures. It computes U1–U4 in code and runs the leakage tests itself.
- Contents: coverage and width by lead and level; the reliability diagram; the interval score against the baseline envelope; WIS; conditional coverage; both extreme-event tables with bootstrap intervals; the per-retrain selection summary.

**Step 5. Freeze, then the AEMO comparison (development part).**
- Freeze the interval recipe: a hash of the `intervals:` config, plus the code commit, recorded in config and guarded by a unit test. This happens before the AEMO comparison, so the benchmark cannot influence the recipe. What is frozen is the recipe, including the per-retrain selection rule, not one set of fitted models.
- Load AEMO's POE data (`PDPASA_REGIONSOLUTION`, `LOR` runs, VIC1; migration `0006` if needed; about 1.1 GB cached under the data root) and compare on Oct 2024 → Sep 2025 (§2.6): the band as published as the headline, the re-centred band beside it, and both alignments. `reports/m3_aemo.md`.

**Step 6. Holdout, scored once.**
- `gridcast evaluate-holdout-intervals` refuses to start unless:
  - the recipe hash matches the freeze;
  - the working tree is clean;
  - the frozen code is unchanged;
  - the data reach the last holdout target;
  - no interval holdout batch exists for the recipe (the U5 lock).
- It runs QR from the 2025-08-01 retrain, using the holdout point model's stored selections for the holdout retrains.
  - Before scoring anything, its Aug–Sep 2025 QR forecasts and fits must reproduce step 2's stored ones exactly; otherwise it stops.
  - It then builds conformal intervals and CQR on the holdout, with calibration windows that reach back into development data.
  - It runs from a git worktree, as M2's did.
- `gridcast report-m3-holdout` writes `reports/m3_holdout.md`: the replication verdicts, everything in step 4, and the holdout part of the AEMO comparison.
- Rewrite the README's *Uncertainty* section and its status, results, limitations and reproduction sections from the generated reports.

## 5. New dependencies and compute

- **Dependencies:** none expected. LightGBM's `quantile` objective, numpy and pandas are enough.
- **Conformal:** minutes to tens of minutes per batch. Every candidate stream for 288 leads, three levels and four point models (C plus three baselines) is vectorised over leads.
- **QR, development:** 6 quantiles × 4 bands × 2 fits (early-stopping fit, refit) × 12 retrains = 576 fits.
  - Run 2's LightGBM fits took about 10–20 s each on average, and the quantile objective is usually slower.
  - Rough estimate: **4–6 h** for development and about **2 h** for the holdout's 5 retrains. The step-2 smoke run measures this before the full run.
- **Storage:** about 1 GB of interval bounds and about 0.5 GB of QR model files under the data root, gitignored.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Errors drift (the winter under-forecast), so past residuals stop describing future ones | Rolling windows and ACI candidates, asymmetric bounds, coverage by month; the holdout winter is a known stress test, disclosed |
| Correlated errors make the coverage numbers look more precise than they are | Block bootstrap by day; tolerances (±2 pp pooled, ±5 pp per lead) chosen with this in mind |
| The ACI feedback for lead *h* arrives *h* later | Updates use only outcomes known by *o*; covered by the leakage tests |
| Tail quantiles would prefer other settings than the point model's | Accepted by the user to save compute; raw QR calibration is reported, and CQR corrects coverage |
| QR quantiles cross | Rearrangement; crossing rate reported |
| QR costs more than estimated | Smoke run first; ask again above about 10 h |
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

# Milestone 4 plan: market anomalies and demand-model explanations

**Goal:** answer the brief's questions 4 and 5 for Victoria: *when is the market behaving abnormally?* and *which factors are associated with unusually high demand or price?*
- **Anomaly detection:** detectors whose false-alarm rate is stated and checked, that catch the operator-flagged price events and injected disturbances of known size. Each alert is merged into an event that carries observable evidence and makes no causal claim.
- **Explanations:** SHAP explanations of the frozen M2 demand model (LightGBM C), and a descriptive, association-only analysis of high and negative prices.

**Status:** planned 2026-11-15 after three question rounds with the user (every recommendation chosen). After the development verdicts, a disclosed v2 of D4 and D6 was decided on 2026-11-20 (§1, §2.10). Branch `m4-anomalies-explainability`, from `main` at `c5fd5d3` (M1–M3 merged). The success criteria in §3 and every detector setting in §2 were fixed before any detector code or score existed.

## 1. Decisions (agreed 2026-11-15)

| Topic | Decision |
|---|---|
| Scope | **Anomalies + demand SHAP.** The detectors cover unusual demand, unexpected demand-forecast residuals, price spikes, unusual prices, volatility changes, and a multivariate Isolation Forest. SHAP explains LightGBM C and feeds each demand event's evidence. Price-residual detection waits for the price model (M2b) |
| Timing | **Causal.** Every score at interval *t* uses only data known at *t*'s end, so the same code can run live behind the M5 API and M7 monitoring. Alerts are merged into events by a rule that also works live |
| Evaluation | **Pre-registered three-part check** (§3): (1) each detector's alert rate on unseen data stays near its stated budget; (2) recall on AEMO-flagged reference events, fixed by flag type before any detector runs (the flags are never inputs); (3) synthetic injection for detection power by size |
| Evidence data | **Cached region totals.** Victorian semi-scheduled wind and solar output and availability (Oct 2021 on; availability means UIGF, see the step-1b clarification below), dispatchable generation, and battery output (from May 2024), re-parsed from the cached `DISPATCHREGIONSUM` files. No new downloads. Also used: demand, price, net interchange, available generation, weather forecasts and ERA5, AEMO's own demand forecasts (loaded Sep 2024 → Sep 2026), and the operator flags |
| Thresholds | **Calibrated tiers + price lines.** Each detector turns its score into a p-value against its own trailing 26 weeks: *notable* p ≤ 0.001 (about 2 flagged 5-minute intervals a week per detector), *severe* p ≤ 0.0001 (about 1 a month). Fixed market lines also flag price ≥ $300/MWh (notable) and ≥ $5,000/MWh (severe), whatever the recent past |
| SHAP | **LightGBM's built-in TreeSHAP** (`pred_contrib`) on the stored M2 models, no new dependency, over a seeded sample. Global summaries by feature and feature family; local explanations for every demand event |
| Price drivers (question 5) | **Descriptive associations:** the rate of prices above $300/MWh and below $0 by observed conditions in the same interval, with day-block bootstrap CIs, labelled as association only |
| Holdout | **Freeze, then run once.** Every setting is fixed in this plan; the detector recipe is frozen (hash + commit) before the development run; the holdout year (Oct 2025 → Sep 2026) is scored by one locked run |
| Criterion bars | Alert rate within a **factor of 2** of nominal; **all class-P reference events caught** (class O reported); synthetic power **≥ 90% at each detector's largest pre-registered size**, with the full power curves reported |
| Residual detectors | **The M3 quantile band plus the detector's own p-value:** the distance of the actual outside the raw quantile-LightGBM C 2.5–97.5% band, scaled by the band's width, as a conformal p-value. This is CQR at any level; the frozen M3 recipe is untouched |

**Added by this plan, then accepted by the user on review (2026-11-15):**
- **No tuning spike.** Every detector setting below is fixed now, before any code exists, so development results are also blind to tuning. A smoke run may use only the warm-up periods (before 2022-05-01; before 2023-06-01 for the residual detectors) and synthetic data, and only to find bugs. Any later change to a setting in §2 is recorded here with its reason and disclosed in the reports.
- **The recipe is frozen before the development run**, not only before the holdout as in M2/M3. With no tuning step, nothing is lost by freezing early, and the development verdicts become a clean pre-registered test.

**Clarified in step 1b, before any detector code (2026-11-15; the user chose both recommendations):**
- **Curtailment is measured against UIGF.** AEMO's `SS_*_AVAILABILITY` columns exist only from table version 8 (May 2024); October 2021 → April 2024 has only `SS_*_UIGF`, AEMO's unconstrained intermittent generation forecast. D6's curtailment share is therefore 1 − cleared / UIGF throughout (§2.3), so the feature keeps one definition across development and holdout. Where both columns exist, availability was never above UIGF in the 60 months loaded. Availability is still loaded and shown in the evidence from May 2024.
- **Battery evidence starts at the first non-zero interval.** The bidirectional-unit (battery) columns appear on 22 May 2024 but are 0 MW at every Victorian interval until 6 Aug 2024. Batteries outside that unit category sit inside the dispatchable totals, so a 0 MW there does not mean the batteries were idle. The evidence marks battery output as unavailable, with that reason, before the first non-zero interval (computed from the data), and labels later values "bidirectional units only".

**Clarified in step 2, before any detector score on real data (2026-11-17; the user chose both recommendations):**
- **Injection starts are shared.** One seeded draw of 1,000 eligible starts per detector is reused for every size and sign, so the power curves differ only by size (§2.9).
- **D6's z-features use plain profiles.** Net interchange, dispatchable generation and available generation minus demand are z-scored against the generic profile (all day types, as D4). Only D1 splits by day type, as §2.3 says.
- **Readings fixed while coding (no setting changed):**
  - the p-value window holds the intervals ending in [t − 26 weeks, t), the 52,416 that §2.2 counts;
  - a profile is missing until its 28 local days of history exist, which affects only the warm-up;
  - a D6 month is scored only if its training window holds at least 256 complete rows (one tree's sample); every month after the warm-up has a full window;
  - curtailment shares are not clipped, so cleared output above UIGF (§1, step 1b) gives a small negative share;
  - events are merged within a split, so none crosses the development/holdout boundary, and the holdout run's development part can reproduce the development batch exactly;
  - an event's peak is the alert with the lowest p, then the largest score, then the earliest. Scores of different detectors are not on one scale, so a cross-detector tie is settled by this fixed but arbitrary rule. A lines-only event's peak is its highest price;
  - the reported-only flat-price injection holds the price at its value at the injection start for 3 h.

**Clarified in step 3, before any evidence or SHAP value on real data (2026-11-19; the user chose every recommendation):**
- **A demand event's SHAP breakdown explains its demand peak.** That is the D1–D3 alert with the lowest p, with §2.4's tie rule. It equals the event peak whenever a demand detector holds that peak. In a mixed event, the overall peak may be a price alert at an interval where demand was ordinary. The evidence names the interval explained, and §2.5 item 8's forecasts are taken at it.
- **Summaries have three sentences:**
  1. the lead fact for the peak's category: the value against its profile, or the actual against the forecast band;
  2. the other market variable over the same 5 minutes (price for demand events, demand otherwise);
  3. the operator flags in the event's span.
  - Supply and weather facts stay in the evidence fields. The fewer facts a summary places side by side, the fewer causal readings it invites.
- **E1's forecast check is exact at storage precision.** M2 stored its forecasts as float32, and one float32 step at 5,000 MW is about 0.0005 MW, so "within 10⁻⁶ MW" cannot be tested against them. The recomputed forecast, cast to float32 as M2 stored it, must equal the stored value bit for bit, the standard of M3's point reproduction. The second check, base value + Σ SHAP = raw prediction within 10⁻⁶, stays as written, in float64.
- **Smoke runs are for bugs and cost only:**
  - evidence and summaries are built for step 2b's warm-up smoke events, and the run prints only field completeness, banned-phrase counts and 5 example summaries;
  - SHAP runs on retrain 2022-11-01, whose issue times fall in D2/D3's warm-up. The run prints only the E1 results and timings, with no SHAP summary.
- **Readings fixed before coding (no setting changed):**
  - LightGBM C's stored forecasts start at issue time 2022-11-01. Events before then, and pairs outside M2's common mask (e.g. the GFS gap of Dec 2023 → Jan 2024), mark §2.5 items 3 and 8 unavailable, with that reason;
  - the *latest* forecast of interval *t* is the one from the issue time at lead 1–6 (as D2), and the *day-ahead* forecast is from the issue time at lead 283–288 (as D3);
  - AEMO's *matched* forecasts follow M2's benchmark (`evaluation/aemo_benchmark.py`). They are the 5-minute pre-dispatch and 30-minute pre-dispatch runs whose nominal run time is our issue time, at both issue times. Pre-dispatch is compared with the mean actual demand over its 30-minute period. A run that does not reach *t* is marked unavailable;
  - holdout SHAP explains holdout issue times only. The `holdout_main` retrain of 2025-08-01 contributes its October 2025 issue times; its Aug–Sep 2025 issue times are development, already explained by run 2's identical model.

**Clarified in step 4, before the freeze and before any development score (2026-11-19; the user chose every recommendation):**
- **What the freeze covers.**
  - The anomaly config hash covers the whole `anomalies` section: the reference-event rules, every detector setting, the injection, the evidence and the SHAP settings. It leaves out output directories and the freeze record itself. It also covers the split dates (`backtest.holdout_start`, `backtest.holdout_end`).
  - `FROZEN_PATHS` = M3's frozen paths (which hold M2's), every `anomalies` module that scores, merges, injects or describes, and `explainability/` except its smoke module.
  - The smoke, report and freeze modules and the CLI are not frozen. A unit test classifies every module of both packages, so new recipe code cannot slip past the freeze.
- **The holdout band for D2/D3 is split by issue time.** Issue times before `backtest.holdout_start` use the development QR batch with run 2's common mask. Later issue times use the holdout QR batch with `holdout_main`'s mask. The evidence's forecasts already follow this rule. The two QR batches overlap on Aug–Sep 2025 (2,928 issue times); there they were bit-identical when checked on 2026-11-19 (843,264 finite pairs per quantile). The holdout run refuses to start if they ever differ.
- **Case-study ties (§3.2).** p has a floor of 1/(1 + n), which any score above its whole 26-week window reaches, so ties at the floor are likely.
  - The rule: the 5 development events with the lowest p from a calibrated detector. Ties go to the event with more calibrated detectors firing, then to a seeded draw (`anomalies.seed`).
  - The two named events (13 Feb 2024 and 9 Dec 2024) are added on top. A rule pick that is also a named event is shown once and not replaced.
  - The holdout uses the same rule, without named events.
- **A5 also checks real data.** The report runs the synthetic suite, as pre-registered. In addition, the development run reruns detection on the real series truncated at 3 seeded cut times inside the development scored period. Every score, p-value, window size, tier and event opening at or before each cut must equal the full pass exactly. Both parts count toward A5.
- **Storage (the default, stated in the question round):** one `model_runs` row per anomaly run (kind `anomaly`, target `market`). Per-interval scores, p-values, window sizes and tiers, the injection results, the forest training ranges and the cut checks go to parquet under `<data root>/anomalies/<batch>/`. Events, with their evidence and summaries, go to the `anomalies` table.
- **Readings fixed while coding, before the freeze (no setting changed):**
  - a named day's case study is the calibrated event ranked first by the case rule among those whose span (first to last alert) touches that NEM day. If none does, the report says that no calibrated event touched the day;
  - the seeded draws use their own streams of `anomalies.seed`: (seed, 101) gives one uniform key per event in time order for the case ties, and (seed, 102) gives the cuts. The injections keep streams 1–7;
  - the cuts are uniform over the interval indices of the development scored period (from 2022-05-01), never its last interval. A truncated run keeps the forecasts issued at or before the cut;
  - the development run reads data only up to NEM midnight of 2025-10-01. It refuses to run if the freeze does not hold, the tree is dirty, or a successful development run is already stored for the hash. Development is scored once;
  - if E1 fails at any demand peak, the run stops before anything is stored (§2.6's withholding).
- **How `report-m4` reads the criteria, fixed while the development run was in progress and before any of its output was read (2026-11-19):**
  - A1's *scored intervals* are those in the detector's scored period that have a p-value: a valid score, and for D6 a month model;
  - A2 counts an alert at any interval whose end lies in [start − 30 min, end + 30 min] of the reference event, where start is the start of its first flagged interval and end the end of its last;
  - A3's power is the detected share of the injections drawn, with their number shown (fewer than 1,000 only if fewer starts were eligible). The flat-price injection is reported only;
  - A4 covers every development event of the stored run;
  - A5 passes only if the synthetic suite (`tests/unit/test_anomaly_causality.py`, controls included) passes, every real cut shows no difference, and no forest's last training interval ends after its month's start (recomputed from the stored ranges);
  - the report recomputes the stored counts: every tier from its p-value and the price lines from the stored prices, and each detector's alert count and the event count against the run's record. It stops on any mismatch.

**Decided after the development verdicts, before any v2 code (2026-11-20; the user chose every recommendation):**
- **A disclosed v2 for D4 and D6 only.** v1's verdicts stay as published: they are the pre-registered result. First, a read-only diagnosis of the stored run found the mechanism of each failure. It is [`docs/m4_v1_diagnosis.md`](../m4_v1_diagnosis.md), generated by `gridcast anomalies-diagnose-v1`. It recomputes v1's own state and scores no alternative setting. It found a design defect in two detectors:
  - **D4:** asinh(price / $10) makes a routine move from −$50 to +$50 (4.62 units) about as large as the 95-fold spike from $185 to $17,500 (4.55). The per-slot robust scale therefore decides the ranking. 54.8% of D4's alerts sit at the floor scale, against 5.9% of scored intervals, and the middle half of its alerts lie between −$38 and $68/MWh. Power at $15,000 falls from 36.8% at the tightest-scale quartile of starts to 0.4% at the widest. P05's |z| was 8.4; the smallest |z| that alerted in development was 14.2.
  - **D6:** the Isolation Forest saturates. In none of the 41 monthly models did the median training row, with one z-feature set to 5, 20 or 80, reach p ≤ 0.001. Its calibration against the static training window also drifts from month to month: 11 of 41 months are above 0.2%, and 19 have no alert.
  - **D1 and D3 stay as v1.** D1's power follows the real day-to-day spread of demand: 78–84% in the tightest-scale quartile of starts, 0.8% in the widest. D2, which conditions on the weather forecast, catches far smaller steps. Changing D1 or D3 would only loosen them to reach the bar.
  - The question round quoted a few figures from an exploratory version of the same script (e.g. "11 of 42 months"). The generated diagnosis is the record; it counts the scored months that have a model.
- **D4 v2:** x = asinh(price / **$100**). Everything else stays as §2.3: the profile over all day types, the floor of 0.25 (now in the new x's units), two-sided scoring. One setting addresses both mechanisms: routine moves across $0 shrink, and the log compression starts at $100 instead of $10. **D5 keeps asinh(price / $10)**, so its scores stay v1's.
- **D6 v2: a robust distance** (§2.10). It keeps the nine features and the monthly refit. The score is the distance from the training median under a shrunk covariance, which grows without bound for a one-feature extreme. Its p-value is against the trailing 26 weeks of its own out-of-sample scores, as for D1–D5.
- **The holdout scores v1 and v2, once each.** v1 gets its pre-registered replication, and v2 its first blind test. Fixed now, before any holdout score: **M5 and M6 serve v2, whatever the holdout shows**, so the holdout chooses nothing.
- **v2's development verdicts are post hoc.** v2 was designed after v1's development results, so its development A1–A5 and E1 are labelled *post hoc (not blind)*. P05 (A2) in particular was known to be missed. The holdout is v2's only blind test, and the holdout has no class-P event, so A2 stays a development-only, post hoc check for v2.
- **How v2 is built and frozen (readings fixed before coding, no setting involved):**
  - **v1 must stay runnable for its own holdout.** The detector names and rules are threaded through the frozen modules, so the frozen modules are generalised behind a *recipe version* (version 1 = the frozen behaviour) rather than copied. That edits files under v1's `FROZEN_PATHS`. v1's code freeze (`2dc58fa`) is therefore carried by behaviour, as M3 carried its folds. Before v2 is frozen, `gridcast anomalies-reproduce` recomputes v1 with the generalised code. It must reproduce run 104 exactly: every stored per-interval column, the injection results, the forest ranges, the cut checks, and every event with its evidence and summary. The v1 holdout run repeats that check before it scores anything. v1's record then names the generalisation commit, with the reproduction as its evidence.
  - **v1's config is untouched.** The `anomalies` section and its hash (`2ce8b3fc523ae81c`) do not change. v2's settings, and only those that differ, live in a new section `anomalies_v2` with its own freeze record. v2's hash covers `anomalies`, `anomalies_v2` and the split dates.
  - **v2's frozen paths** are v1's (as generalised) plus any new recipe module. The module-classification test covers both.
  - **Stored names.** v2's D6 is stored as `robust_distance`, so a v2 event never reads as an Isolation Forest event. v2's D4 keeps `price_level` (the same detector with a new transform). Each version has its own `model_runs` rows (the recipe version is in their config), batches and events.
  - **Order:** v2's freeze and its development run come before any holdout score of either version. Step 5 (explanations) does not depend on the version.

**Clarified in step 5, before any global SHAP value or price-association rate was computed (2026-11-20; the user chose every recommendation):**
- **The global summaries decompose the whole forecast.** Run 2's 48 LightGBM C fits do not share one target form. In the 6–24 h band, 5 fits are *direct*, 4 *residual*, 2 *anchored* and 1 *latest*. In the shorter bands most are *latest*. Raw TreeSHAP explains only each fit's correction to its own anchor (§2.6). A family's credit would then mean different things from fit to fit, and a change of form between retrains would look like instability.
  - Each anchor is a sum of model features with weight 1: `dem_now` (*latest*), `dem_same_time_yesterday` (*residual*), and both `dem_same_time_yesterday` and `dem_now_vs_yesterday` (*anchored*).
  - The anchor's share therefore goes to those features: x − x̄, with x̄ the mean over the fit's own global sample. That is the exact Shapley value of a linear term with the sample as background. The fit's constant becomes the base value + the sample mean of the anchor.
  - Every fit then decomposes the forecast itself: forecast = constant + Σ contributions, exactly. Fits of different forms can be pooled.
  - Disclosed: the sum mixes path-dependent TreeSHAP (the trees) with an interventional attribution of the linear anchor (sample background), so it is the Shapley value of neither game exactly. Local accuracy holds.
  - The raw, tree-only values stay stored. The report shows them in a table by (band, target form), beside the headline.
- **Question 5's top 5% is taken within each group.** The overall row uses the top 5% of all targets. The summer (Dec–Feb) and winter (Jun–Aug) rows use the top 5% of that season's targets, compared with all of that season's targets. The forecast rows do the same with the stored forecast.
- **The price associations cover the market development period,** NEM days 2022-05-01 → 2025-09-30. That is the span of the market detectors and the price lines, so every development number in the M4 reports covers the same span.
- **Bin edges come from development data and are fixed now.** Step 6 reuses them unchanged, so the holdout's rates sit on the same bins, and distribution shifts show up as bin occupancy.
  - Solar output is exactly 0 MW in 47.6% of development intervals, so it gets a separate *zero output* bin plus deciles of positive output. Plain deciles would merge about five bins into one.
- **Readings fixed before coding (no setting in §2 changed):**
  - **`gridcast explain-dev`:**
    - draws §2.6's global sample (20,000 pairs per fit, seeded) for run 2's 48 fits, over issue times 2022-11-01 → 2025-09-30;
    - runs E1's gate on every pair;
    - stores, per fit, the pairs, their feature rows, actuals, stored forecasts, anchors, base values and raw contributions, plus each booster's gain importance, under `<data root>/explain/dev/`;
    - writes a manifest with each fit's gate, the commit and the timings.
    - If E1 fails for any pair, it stores nothing (explanations withheld). It refuses a dirty tree, or a tree where either anomaly freeze does not hold (`explainability/` is under both).
    - It is deterministic and reported only, so a rerun is allowed with `--replace`. It is not a once-only score.
  - **Pooling:** the summaries pool a band's fits with one weight per pair. A family's contribution is the sum of its features' contributions; its mean |SHAP| is the mean of the absolute family sum.
  - **Stability:** per band, the Spearman correlation of consecutive retrains' rankings of the 26 features by mean |SHAP| (whole forecast), 11 per band. Each pair's two target forms are shown beside it.
  - **Weather dependence:** per band, the weather family's contribution against `wx_temp` (the population-weighted GFS forecast temperature at the target), in 1 °C bins: mean and 10–90% range. Bins with fewer than 100 pairs are not drawn.
  - **Gain importance:** M2's split gain per feature, as a share of each fit's total, averaged over the band's fits. It is labelled tree-only, i.e. relative to the anchor.
  - **Question 5:** the 6–24 h band (leads 73–288), the pooled global sample of its 12 fits. A target is in the top 5% if its value is above its group's 95th percentile. Seasons go by the target's local month. The table shows mean contributions by family in the top 5% and in all targets, their difference, and the pair counts. No interval is attached: it describes the model's arithmetic on a sample.
  - **Price associations:**
    - intervals are NEM days by interval start (§2.1);
    - the rates are P(price > $300/MWh) and P(price < $0/MWh);
    - the observables, all from the same interval: demand, ERA5 population-weighted temperature (the value the evidence uses), wind and solar cleared output, net interchange, local hour and local month;
    - deciles are right-closed bins between the development 10% … 90% quantiles. Temperature bins are fixed 5 °C bins at multiples of 5. The two-way table is demand tercile × wind tercile;
    - an interval that lacks a variable is left out of that variable's table only, and the counts are shown;
    - the CIs are percentile 95% intervals from 1,000 day-block bootstrap resamples (NEM days), seeded.
  - **Config:** step 5's report settings live in a new section, `explain_report`. Both anomaly hashes cover all of `anomalies`, so a setting added there would break the freezes. The global sample's own settings (`anomalies.explain`) are already frozen and do not change.
  - **Freeze classification:** the new `explainability` modules (the global run and the report) and the association module only orchestrate, read and summarise. They are classified as not frozen, as the smoke modules are.

## 2. Design

### 2.1 Periods

- **Data:** 5-minute market intervals from 2021-10-01 (5-minute settlement) to 2026-09-30, plus 1 Oct 2026 from the daily reports. Days are NEM-time days by interval start.
- **Warm-up, never scored:** profiles need 4 weeks and p-values need 26 weeks of scores.
  - Market detectors (D1, D4, D5, D6) and the price lines: warm-up 2021-10-01 → 2022-04-30.
  - Residual detectors (D2, D3): the stored QR quantiles start at issue time 2022-11-01, so their warm-up is 2022-11-01 → 2023-05-31.
- **Development (scored once, after the freeze):** 2022-05-01 → 2025-09-30 for the market detectors and price lines; 2023-06-01 → 2025-09-30 for D2 and D3.
  - D2/D3 score only targets inside M2's common mask (pairs with C weather). The GFS gap of Dec 2023 → Jan 2024 leaves a hole that is reported, not filled.
- **Holdout (scored once, locked):** 2025-10-01 → 2026-09-30 for every detector. One continuous run from 2021-10-01, so trailing windows cross the boundary. Its development part must reproduce the stored development batch exactly before anything is scored.

### 2.2 Conformal p-values and tiers

- Every calibrated detector produces one score per 5-minute interval *t*. A larger score means more unusual.
- **p-value:** p(t) = (1 + #{s in the window : s ≥ s(t)}) / (1 + n), where the window holds the detector's valid scores for intervals ending in (t − 26 weeks, t). A full window has 52,416 intervals, so p can go as low as about 1.9 × 10⁻⁵. Ties count against the alert (conservative).
- **Implementation:** reuse M3's Fenwick rolling order statistics (`intervals/rolling.py`): rank all scores once, then count the window entries ranked at or above s(t). M3 argued, and its leakage tests showed, that ranking up front does not leak; M4's causality tests check this again (§2.8).
- **Tiers:**
  - notable: p ≤ 0.001;
  - severe: p ≤ 0.0001.
  - Price lines: price ≥ $300/MWh is notable and ≥ $5,000/MWh is severe, whatever the window says. The lines are not calibrated, and criterion A1 does not apply to them.

### 2.3 Detectors (settings fixed now)

Local time is Australia/Melbourne, as in M2's calendar features. *Profile* means the median and MAD of the same local half-hour over the previous 28 days, ending before *t*'s own day. Robust scale = max(1.4826 × MAD, floor).

| # | Name | Score at interval *t* | Fixed settings |
|---|---|---|---|
| D1 | `demand_level`: unusual demand | \|z\|, where z = (demand − profile median) / robust scale | Profile split by day type: business day vs weekend or VIC public holiday (`calendar_days`). Floor 50 MW |
| D2 | `demand_residual_short`: unexpected vs the latest forecast | Let o be the latest issue time before *t* (lead 1–6, i.e. 5–30 min). s = max(q₀.₀₂₅ − y, y − q₀.₉₇₅) / max(q₀.₉₇₅ − q₀.₀₂₅, 50 MW), using M3's raw quantile LightGBM C at (o, t), rearranged (sorted) | Leads pooled. Development quantiles: QR batch `4bab11d7-…`; holdout: `b226706b-…` |
| D3 | `demand_residual_dayahead`: unexpected vs the day-ahead forecast | The same score at lead 283–288 (23.5–24 h) | As D2 |
| D4 | `price_level`: unusual price for the time of day | \|z\| of x = asinh(price / $10) against x's profile | All day types. Floor 0.25 (asinh units). Two-sided, so extreme negative prices count too |
| D5 | `price_volatility`: rapid volatility change | v(t) = root-mean-square of the 5-minute changes in x over the 12 intervals ending at *t* (1 h). b(t) = median of v over the 7 days ending 1 h before *t*. Score \|ln((v + 0.01) / (b + 0.01))\| | Two-sided: a sudden calm (e.g. prices held at an administered cap) also scores |
| D6 | `isolation_forest`: multivariate | −`score_samples` of scikit-learn's `IsolationForest` (200 trees, 256 samples per tree, fixed seed), refitted at each NEM-month start on the previous 26 weeks. Its p-value is against the training window's scores | Nine causal, signed features: D1's z; D4's z; D5's log ratio; the 5-minute demand change scaled by the robust scale of all 5-minute changes in the previous 28 days (floor 10 MW); net interchange, dispatchable generation, and available generation minus demand, each as a z against its profile (floor 50 MW); wind and solar curtailment shares, 1 − cleared / UIGF when UIGF ≥ 100 MW, otherwise 0 (UIGF, not availability: clarified in step 1b, §1). No forecasts, flags, ERA5 or battery data |
| L1 | `price_line_300` | price ≥ $300/MWh | Notable |
| L2 | `price_line_5000` | price ≥ $5,000/MWh | Severe |

*Why the Isolation Forest p-value uses training scores:* each tree sees 256 of about 52,000 rows, so a training row's score is close to an out-of-sample score. A1 checks whether that holds in practice.

### 2.4 Events

- **Merge rule:** notable-or-severe alerts from any detector (lines included) separated by at most 30 min (6 intervals) form one event. Live, an event opens at its first alert and closes after 30 quiet minutes.
- **Stored per event:** start, end, the peak interval (lowest p, ties by largest score), every detector that fired with its lowest p, and the tier (severe if any severe alert or L2). A category label comes from the detectors that fired: *unusual demand*, *unexpected demand (short-term)*, *unexpected demand (day-ahead)*, *high price* (lines only), *unusual price*, *volatility change*, *multivariate*.
- **Reported separately:** events from calibrated detectors and events from the price lines only. Prices above $300 occur in 3.6% of development intervals, so line-only events would otherwise swamp the catalogue.

### 2.5 Evidence and summaries

Each event's evidence is facts with times and sources. Every field is labelled *known at the time* or *retrospective* (published later, e.g. ERA5 a few days later). Unavailable fields say why (e.g. "battery data starts May 2024").

1. **The event:** start and end in local and NEM time, duration, detectors with tiers and lowest p.
2. **Values at the peak:** demand and price; their profile median and 10th–90th percentile range; their percentile in the trailing 26 weeks.
3. **Forecasts:** LightGBM C's latest and day-ahead forecasts for the peak interval, the QR 2.5–97.5% band and the residual. AEMO's matched forecasts (P5MIN and pre-dispatch) and their errors, where loaded (Sep 2024 on).
4. **Changes:** demand and price over the previous 5 min, 30 min and 1 h.
5. **Supply:** wind and solar output, their UIGF (and availability, from May 2024), curtailment against UIGF, and their 1-hour change; dispatchable generation; battery (bidirectional units only; unavailable before the first non-zero interval, §1); net interchange (import or export); available generation and its margin over demand.
6. **Weather:** the GFS temperature forecast the model used; ERA5 temperature (retrospective).
7. **Operator flags in the event's span:** APC bits and market suspension, as evidence only.
8. **Demand events (D1–D3 fired):** the SHAP breakdown of LightGBM C's forecast for the peak interval at both leads (§2.6), i.e. how the model arrived at what it expected.

- **Summaries:** template sentences of facts only, e.g. "Demand at 15:40 was 5,518 MW, 2,839 MW above the latest forecast, issued at 15:30, and outside its 95% band. Over the same 5 minutes, the price moved from −$21 to $95/MWh. AEMO flagged nothing."
- **Banned phrases (A4):** "because", "cause", "caused", "causes", "due to", "driven by", "drove", "led to", "leads to", "resulted in", "as a result", "triggered", "explains", "explained by", "the reason". The test is case-insensitive and on word boundaries.

### 2.6 SHAP for LightGBM C

- **Models:** the stored M2 boosters for LightGBM C:
  - development: run 2, batch `6d783932-…`, 12 retrains × 4 lead bands;
  - holdout: `holdout_main`, batch `fcb6abb6-…`.
  - No model is refitted.
- **Gate (E1):** feature rows are rebuilt with M2's frozen builder. For every explained pair:
  - the booster's prediction, mapped through the band's target form, must equal the stored forecast within 10⁻⁶ MW;
  - base value + Σ SHAP must equal the raw prediction within 10⁻⁶.
  - If either check fails, explanations are withheld.
- **Decomposition:** forecast = anchor + base value + Σ contributions. The anchor is the target form's fixed term (0 for *direct*, y(o) for *latest*, y(t − 24 h) for *residual*, and y(o) + y(t − 24 h) − y(o − 24 h) for *anchored*). Each (retrain, band) states its form, and summaries say what the contributions are relative to (e.g. "relative to the latest demand").
- **Families**, from the feature catalogue:
  - lead time;
  - calendar at the target;
  - calendar 24 h before;
  - demand at the issue time;
  - demand at the same time yesterday and last week;
  - weather at the target.
- **Global sample:** per (retrain, band), 20,000 seeded pairs, uniform over the quarter's scored issue times and the band's leads. Stored under `<data root>/explain/`.
- **Reported:**
  - mean |SHAP| per feature and family, per band;
  - stability across retrains (Spearman correlation of consecutive retrains' feature rankings);
  - dependence of the weather contribution on forecast temperature;
  - M2's gain importance beside it.
- **Question 5, demand:** in the 6–24 h band, mean SHAP by family for targets in the top 5% of actual demand, against all targets. Shown overall, for summer (Dec–Feb) and for winter (Jun–Aug), and also for the top 5% of forecast demand.
- **Language:** contributions describe the model's arithmetic, not the grid's physics. Correlated features (several temperature terms, demand lags) share credit, which is why families are the main unit.

### 2.7 Price associations (question 5, price)

- **Rates:** P(price > $300/MWh) and P(price < $0) by bins of same-interval observables:
  - demand deciles;
  - ERA5 population-weighted temperature in 5 °C bins;
  - wind output deciles;
  - solar output deciles;
  - net-interchange deciles;
  - local hour;
  - month;
  - plus one two-way table, demand tercile × wind tercile.
- **Uncertainty:** day-block bootstrap 95% CIs (1,000 resamples, fixed seed).
- **Periods:** development in step 5; the holdout part only after the locked run (step 6).
- **Wording:** "more frequent when", never "because".

### 2.8 Causality (A5) and the freeze

- **Property tests, run by the report itself as M3's U4 was:**
  1. Truncation: for random cut times *c*, every score, p-value, tier and event opening for intervals ≤ *c*, computed from data truncated at *c*, equals the full run's.
  2. Perturbation: randomising every input after *c* changes nothing at or before *c*.
  3. Isolation Forest: the model that scores month *m* was fitted only on data before *m*'s start.
  4. A deliberately leaky control (a centred profile window) must fail tests 1–2, which shows the tests have power.
- **Freeze (step 4):** an anomaly config hash over the `anomalies` config section, and `FROZEN_PATHS` covering the detector, p-value, event and evidence modules plus M2's frozen feature and model code, recorded in `configs/base.yaml` with the commit. A unit test pins both, as for M2 and M3.

### 2.9 Synthetic injection (A3)

- **Placement:** 1,000 seeded start times per (detector, size, sign), uniform over the detector's scored intervals in the period. The same 1,000 starts serve every size and sign of a detector (clarified in step 2, §1). Starts are placed only where the unmodified detector raised no alert within the injection span ± 1 h, so a detection must come from the injection.
- **One at a time:** each injection is scored alone. Trailing windows, profiles and Isolation Forest models are the unmodified ones; an injection only changes data from its start onwards.
- **Detected:** at least one notable alert from that detector inside the injection span.
- **Sizes (largest in bold; both signs where ±):**

  | Detector | Injection | Sizes |
  |---|---|---|
  | D1 | add to demand for 1 h | ±250, ±500, ±1,000, **±2,000 MW** |
  | D2 | add to demand for the 6 targets of one issue time (leads 1–6, 30 min), so the stored forecast is the one a live system would have used | ±100, ±250, ±500, **±1,000 MW** |
  | D3 | add to demand for 2 h | ±250, ±500, ±1,000, **±2,000 MW** |
  | D4 | set the price for 30 min to | $300, $1,000, $5,000, **$15,000**; −$100, −$300, **−$1,000** |
  | D5 | for 1 h, alternate the price between its actual value + A and − A | A = $50, $200, $1,000, **$5,000** |
  | D6 | D1's demand shapes | ±250, ±500, ±1,000, **±2,000 MW** |

- **Reported only:** a 3-hour flat-price injection (D5's calm side); the smallest size reaching 50% and 90% power; power by year and month.

### 2.10 v2: D4 and D6 revised after the development verdicts (decided 2026-11-20)

Only D4 and D6 change (§1, "Decided after the development verdicts"). D1, D2, D3, D5, the price lines, the event rule, the evidence (except D6's sentence), the injections and the criteria are v1's. v2's development run must reproduce v1's scores and p-values of D1, D2, D3 and D5 exactly.

| # | v2 | Score at interval *t* | Settings |
|---|---|---|---|
| D4 | `price_level` | \|z\| of x₄ = asinh(price / $100) against x₄'s profile | All day types, floor 0.25 (x₄ units), two-sided: as v1 |
| D6 | `robust_distance`: multivariate distance | √((f − m)ᵀ S⁻¹ (f − m)) for the interval's nine features f, under its NEM month's model | Refitted at each NEM-month start on the previous 26 weeks' complete rows, if at least 256 (as v1). m = coordinate-wise median of the training rows. S = Ledoit–Wolf shrinkage covariance of the training rows about m (scikit-learn `LedoitWolf(assume_centered=True)` on f − m) |

- **D6's features** are v1's nine (§2.3), with D4's z as v2 defines it (x₄). D6's features are defined as the detectors' own, and v1's D4 z carries the diagnosed defect.
- **D6's p-value** is a rolling conformal p-value, as for D1–D5 (§2.2). It is taken against D6's valid scores of the intervals ending in [t − 26 weeks, t). Each score comes from its own month's model, so every score in the window is out of sample.
  - The first monthly models are fitted on short windows (the first, November 2021, on 3 days), as in v1. Their scores enter the windows of the first development weeks. Disclosed, not changed.
- **Injection (A3):** D1's demand shapes, as v1 (§2.9). D1's z, the scaled demand change and the margin z are recomputed from the injected demand and scored by each row's month model. p-values come from the rolling windows.
- **A5:** no monthly model sees an interval ending after its month's start. The synthetic causality suite gains D6 v2 cases, and the real-data cuts run on v2.
- **Evidence:** D6's summary sentence reads "the multivariate detector (robust distance) gave the 5 minutes to … a p-value of … against its trailing 26 weeks". Every other field is v1's.
- **Criteria:** A1–A5 and E1 as §3, on development (*post hoc*, §1) and on the holdout (*holds* / *does not hold*, §3.2), with A6 per recipe hash.
- **How `report-m4` reads v2, fixed before v2's development run (2026-11-20):**
  - every reading in §1 ("How `report-m4` reads the criteria") applies unchanged;
  - A3's D6 checks are `robust_distance` at ±2,000 MW;
  - A5 runs the causality suite's v2 cases (ids `v2`, controls included). Its month-model leak check covers the distance models;
  - the report is `reports/m4_anomalies_v2.md`. Each v2 verdict is labelled *post hoc*, with v1's development result beside it;
  - the case studies follow the same rule (§3.2, step-4 ties).

## 3. Success criteria (pre-registered)

These apply on the **development period** (§2.1) unless stated. *Calibrated detectors* = D1–D6.

| # | Criterion | Pass if |
|---|---|---|
| A1 | Alert budget | For every calibrated detector, the share of scored intervals with p ≤ 0.01 lies in [0.5%, 2%], **and** the share with p ≤ 0.001 lies in [0.05%, 0.2%] (12 checks) |
| A2 | Reference events | Every scorable class-P reference event (§3.1) has at least one notable alert from a calibrated detector (the price lines excluded) between 30 min before its start and 30 min after its end |
| A3 | Detection power | Every calibrated detector catches ≥ 90% of injections at its largest size, separately for each sign (§2.9; 11 checks: D1 ±, D2 ±, D3 ±, D4 $15,000 and −$1,000, D5, D6 ±) |
| A4 | Evidence without causal claims | Every development event has every §2.5 field filled or explicitly marked unavailable with a reason, and no summary contains a banned phrase |
| A5 | Causal | Every §2.8 property test passes, and the leaky control fails as it should (run by the report itself) |
| A6 | Holdout used once | The holdout is scored by a single `gridcast evaluate-holdout-anomalies` run against the frozen recipe; the command refuses to run again for the same recipe hash |
| E1 | SHAP reproduces the forecasts | §2.6's gate passes for every explained pair (development and holdout models). Explanations are withheld otherwise |

### 3.1 Reference events (A2)

- **Source:** AEMO's own flags in `DISPATCHPRICE` (`APCFLAG` bits, `MARKETSUSPENDEDFLAG`). They are never detector inputs.
- **Class P (price-observable by definition):** intervals with APC bit 1 (APC/APF binding), 2 (VoLL override) or 4 (MPC/MPF binding). Flagged intervals less than 24 h apart form one event. An event is scorable if it lies in the scored period.
- **Class O (may leave no visible trace):** market suspension, and APC bits 8 (manual overwrite) and 16 ("IRLF scaled") with none of P's bits. Recall is reported, with no claim that anything observable happened.
- **Step 1 confirms the bit meanings** from the MMS Data Model before any detector code runs. If they differ from the ingestion comment used here, the classes are re-derived by the same principle (price at a cap, floor or administered limit = P) and the change is recorded before step 2.
- **Expected from the flags, before the bit check:** about 7 scorable class-P events in development: 13–15 Jun 2022, 13 Feb 2024, 5 Aug 2024, 28 Oct 2024, 15 Nov 2024, 17 Nov 2024 and 12 Jun 2025. The events of 31 Jan and 21 Feb 2022 fall in the warm-up. The holdout has no class-P event, so A2 is judged on development only; the holdout's one class-O day (23 Mar 2026) is reported.

### 3.2 Reporting rules

- **Holdout replication:** A1, A3, A4, A5 and E1 are recomputed on the holdout and reported as *holds* or *does not hold*. They do not change the development verdicts. A6 is the only holdout criterion.
- **Reported, not pass/fail:**
  - the severe-tier rates;
  - alert rates by month and year;
  - the price lines' event counts;
  - overlap between detectors;
  - the class-O recall;
  - the power curves;
  - the event catalogue (counts, durations, categories);
  - SHAP summaries;
  - the price associations.
- **Case studies, chosen by a fixed rule:** the 5 development events with the lowest p from a calibrated detector, plus two named in advance:
  - 13 Feb 2024 (price at the cap);
  - 9 Dec 2024 (demand +2.9 GW in 40 min then −3.1 GW in 5 min; cause unverified, noted in M2).
  - The holdout uses the same rule, without named events.
- **Failures are published** in the M4 reports and the README. If A5 or E1 fails, results are withheld until the cause is found.

### 3.3 Disclosures written now

- The detectors were designed after M2's and M3's development and holdout results had been seen, including the winter under-forecast.
- The question rounds for this plan saw:
  - **Development facts:** the flagged days and their prices; the largest 5-minute demand changes, including 9 Dec 2024.
  - **Holdout facts:**
    - the price calm: 0.2% of intervals above $300, against 3.6% in development; 25.9% negative, against 19.0%;
    - one day above $5,000 (8 Jul 2026);
    - the demand jumps of 1,758 MW in 5 min on 6 Nov 2025 and 1,130 MW on 10 Nov 2025;
    - the flagged day of 23 Mar 2026 (prices $45–95).
- No detector score existed when the settings in §2 were fixed.

## 4. Steps (vertical slices; one branch, one commit per logical step)

**Step 0. Plan.** This document, `m4_handoff.md`, and the M3 handoff's note that M3 is merged.

**Step 1. Reference events and supply data.**
- **1a:**
  - Read the MMS Data Model pages for `DISPATCHPRICE` (`APCFLAG`, `MARKETSUSPENDEDFLAG`) and `DISPATCHREGIONSUM` (the `SS_*`, `BDU_*`, `DISPATCHABLEGENERATION` and `UIGF` definitions).
  - `gridcast anomaly-reference-events` writes the generated `docs/m4_reference_events.md` from the stored flags. Commit it before any detector code exists.
- *Outcome (2026-11-15, step 1a):*
  - AEMO's own Data Model pages returned HTTP 404, so the bit meanings were read on Global-Roam's mirror (<https://mmsdocs.global-roam.com/MMS/DISPATCH/DISPATCHPRICE/>).
    - 1 = APC or APF binding; 2 = VoLL override; 4 = MPC or MPF binding; 8 = price manually overwritten.
    - 16 = price scaling via the **inter**-regional loss factor (the ingestion comment's "IRLF scaled").
    - These match the plan's assumption, so §3.1's classes stand unchanged.
  - [`docs/m4_reference_events.md`](../m4_reference_events.md), generated by `gridcast anomaly-reference-events` with the rules in `configs/base.yaml` (`anomalies.reference_events`), lists 17 events: 9 class P and 8 class O.
  - **A2 judges 7 class-P events, P03–P09**, the seven §3.1 expected: 13–15 Jun 2022 (335 intervals, −$300 to $14,571), 13 Feb 2024, 5 Aug 2024, 28 Oct 2024, 15 Nov 2024, 17 Nov 2024 and 12 Jun 2025.
  - P01 and P02 (Jan and Feb 2022) fall in the warm-up. The only holdout event is O08 (23 Mar 2026, bit 16).
  - The step-1a commit contains no detector code.
- **1b:**
  - Migration 0007:
    - `generation_intervals`: region, interval end, type ∈ {wind, solar, battery, dispatchable}, cleared MW, available MW (battery: generation and load columns);
    - the `anomalies` table: event id, run id, region, start, end, peak, tier, detectors, lowest p, category, evidence jsonb, summary;
    - `model_runs` accepts `model_kind = 'anomaly'` and `target = 'market'`.
  - `gridcast ingest-generation` re-parses the cached `DISPATCHREGIONSUM` months (and the daily reports for 1 Oct 2026), with lineage in `ingestion_runs`.
  - Data-quality checks: missing intervals, cleared ≤ available + tolerance, no negative output.
- *Outcome (2026-11-15, step 1b):*
  - **Question round** (both recommendations chosen; see §1, "Clarified in step 1b"): curtailment against UIGF, because `SS_*_AVAILABILITY` exists only from May 2024; battery evidence unavailable before the first non-zero bidirectional-unit interval.
  - **Migration 0007:**
    - `generation_intervals` keyed by (region, interval end, kind) holds `cleared_mw`, `uigf_mw` and `available_mw` (wind and solar), and `load_mw` (battery). A check constraint ties the columns to the kind.
    - `anomalies` stores times as interval ends (first, last, peak), `detectors` and `categories` as arrays checked against the §2.3 and §2.4 names, a generated `lines_only` flag (§2.4's separate reporting), `split` (dev or holdout), and the events cascade with their run.
    - `model_runs` accepts kind `anomaly` and target `market`.
  - **`gridcast ingest-generation`** loaded all 60 cached months with no download, then 1 Oct 2026 from the daily reports. September 2026 matched MMSDM exactly in every column first. The loader requires each table version's own columns (v5 → Apr 2024; v8/v9 add availability and battery). The MMS reader gained optional columns for this.
  - **Data-quality checks** (`generation_missing`, `generation_negative` pass; `generation_above_uigf` warns: 655 wind and 33 solar intervals above UIGF, kept as published). Figures and limits: `docs/data_sources.md` §8.
  - Pre-existing, not from this step: `implausible_timestamps` failed on the 288 market intervals of 1 Oct 2026, which M2 step 6 loaded on purpose but which lie outside `ingestion.end_month` (it already failed on 2026-11-14). The generation checks keep to the same window.
  - **Fixed after step 1b (2026-11-16):** `implausible_timestamps` no longer flags days after the window that a successful `nemweb_daily` run loaded; the new `daily_report_days` warning lists them (1 Oct 2026, 288 intervals). `make dq` exits 0. Details: `docs/data_sources.md` §7.

**Step 2. Detector core.**
- Package `src/gridcast/anomalies/` (window p-values on M3's Fenwick structure, D1–D6, the lines, the event merge rule, the injection harness), with the §2.8 property tests and control.
- **Smoke run:** warm-up periods and synthetic data only, for bugs.
- *Outcome (2026-11-17, step 2a: the core on synthetic data):*
  - **Question round,** both recommendations chosen (§1, "Clarified in step 2"): shared injection starts; plain profiles for D6's z-features.
  - Every §2 setting is now in `configs/base.yaml` (`anomalies`), typed in `AnomaliesConfig`, so step 4 can hash it.
  - New modules under `src/gridcast/anomalies/`:
    - `pvalues.py`: `RollingCounts` subclasses M3's `RollingOrderStats` with a count query. M3's frozen `rolling.py` is unchanged. The sweep also answers extra (interval, detector, value) queries against the unmodified windows, for the injections;
    - `series.py`: inputs on one complete 5-minute grid, and each interval's local day, half-hour slot and NEM month;
    - `profiles.py`, `detectors.py` (D1–D6), `isolation.py` (monthly forests), `events.py`, and `run.py` (scored periods, tiers, lines, events per split).
  - **Tests,** all synthetic:
    - brute-force checks of the window counts, profiles (including Melbourne's DST fall-back half-hour, which holds 12 values), volatility and band look-up;
    - the event rule;
    - **A5's property tests** in `tests/unit/test_anomaly_causality.py`: truncation and randomising the future leave every score, p-value, window size, tier and event opening at or before the cut unchanged, and every forest was fitted before its month. A centred, leaky profile fails both property tests, and perturbing data known at the cut changes the outputs.
  - Still to come in step 2b: loading the real inputs, the injection harness, and the warm-up smoke run.
- *Outcome (2026-11-17, step 2b: inputs, injection, warm-up smoke run):*
  - **New modules:**
    - `anomalies/inputs.py` loads the series from `market_intervals`, `generation_intervals` and `calendar_days`. It reads the QR band with run 2's common mask, after checking that the issue times and actuals agree, and checks the stored actuals against market demand;
    - `anomalies/injection.py` is §2.9's harness, with shared starts. Injected values are scored against the unmodified profiles, baselines, bands, forests and windows: one extra window sweep for D1–D5, the month's forest for D6;
    - `anomalies/smoke.py` and `gridcast anomalies-smoke` (`make anomalies-smoke`).
  - **Tests:**
    - the harness equals a full recompute wherever §2.9 says the context stays unmodified;
    - eligibility (scored, valid, no alert within ±1 h, D2 aligned) and seeded shared starts;
    - an integration test of the loader.
  - **Smoke run (warm-up data only, bugs only; 18 s).**
    - The market pass read data ending 2022-04-30 and treated 6 Mar → 30 Apr 2022 as pseudo-scored.
    - The residual pass read demand to 2023-05-31 and computed D2/D3 only; the market detectors and lines were never computed on that data.
    - Stored QR actuals equal market demand on all 2,923,920 pairs checked.
    - **No coding bug was found.** Three checks were run:
      - D1's largest |z| are New Year's Eve 2021 evening demand;
      - D4's largest are negative prices at Sunday dawn;
      - D6's forest scores a row at 100× one feature's 99th percentile near the training median, because split thresholds come from the training data. That is a known property of Isolation Forests, not a code fault.
    - **But the run previews likely pre-registered failures** (immature windows, one season):
      - power at the largest sizes, 200 starts: D1 ±2,000 MW 55–58%; D3 ±2,000 MW 76–86%; D4 $15,000 28% and −$1,000 52%; D6 ±2,000 MW 1–2%. D2 (±1,000 MW) and D5 ($5,000) reached 100%;
      - D1's p ≤ 0.01 rate was 0.14%, against A1's band of 0.5–2%;
      - in asinh(price/$10) units, a −$43 dawn price is as far from a typical $74 as a $15,000 spike, so frequent negative prices fill D4's top 0.1%.
    - **User decision (2026-11-19): keep the recipe as pre-registered.** Nothing in §2 changes. Development runs blind after the freeze, and any A1/A3 failure is published. If failures occur, a v2 may follow the development verdicts with a recorded reason, a new hash and a disclosed rerun (§6), as M2's second development run did. The smoke run stays bugs-only.

**Step 3. Evidence, summaries and the SHAP core.**
- `src/gridcast/explainability/`: TreeSHAP via `pred_contrib`, the E1 gate and families.
- `anomalies/evidence.py`: the §2.5 fields and templates, plus the banned-phrase test.
- **SHAP smoke run** on one retrain, to measure cost.
- *Outcome (2026-11-19, step 3a: the SHAP core):*
  - **Question round,** every recommendation chosen (§1, "Clarified in step 3").
  - **Config:** `anomalies.explain` pins both M2 batches (run 2 and `holdout_main`), the sample size, the SHAP-sum tolerance and the output directory. It sits inside the `anomalies` section, so step 4's hash covers it.
  - **New package `src/gridcast/explainability/`:**
    - `families.py`: the six families are the feature catalogue's groups of variant C;
    - `stored.py`: the fits from M2's retrain records, each with its fold and band. It refuses a batch that is not a complete run of the frozen M2 recipe, or a model file whose columns or tree count differ from the record. A loaded booster sits inside M2's own `LightGBMModel`, so every forecast follows M2's code path;
    - `treeshap.py`: `pred_contrib`, the decomposition anchor + base value + Σ contributions, and E1's gate (`GateError` withholds the explanations);
    - `sample.py`: the seeded per-fit draw, and feature rows rebuilt one lead at a time with M2's builder and its float32 conversion;
    - `pairs.py`: explains any set of stored pairs, which step 3b reuses for event peaks;
    - `smoke.py` and `gridcast explain-smoke` (`make explain-smoke`).
  - **Tests** (`tests/unit/test_explainability.py`, 13): a synthetic M2 walk-forward stores one booster per target form. The explainer reloads them and reproduces the stored float32 forecasts bit for bit, and each forecast decomposes within 10⁻⁶. A one-ulp change in a stored value fails the gate, and so do a SHAP sum that no longer adds up and stored values that are not float32. The unit suite has 397 tests.
  - **SHAP smoke run** (retrain 2022-11-01, 4 fits × 20,000 pairs, 37 s; E1 results and timings only):
    - **E1 passes on all 80,000 pairs.** No recomputed forecast differs from the stored float32 value, and the largest SHAP-sum error is 2.5 × 10⁻¹¹.
    - The largest float64-vs-stored gap, 4.8 × 10⁻⁴ MW, is float32 rounding: 45 of the fold's lead 1–6 forecasts lie above 8,192 MW, where one float32 step is 9.8 × 10⁻⁴ MW.
    - **Cost:** predict + SHAP took 1.4–2.7 s per 31-leaf fit and 24.4 s for the 127-leaf fit. Scaled by tree count, the full global samples need about 11 min (development, 48 fits) and 5 min (holdout, 20 fits), well under §5's hour. A rerun after the fold fix of step 3b, sharing the CPU with the integration suite, projected 17 and 7 min. Feature building is negligible.
- *Outcome (2026-11-19, step 3b: evidence and summaries):*
  - **Config:** `anomalies.evidence` holds the change horizons (5, 30 and 60 min), the supply-change horizon (60 min) and §2.5's banned phrases, so step 4's hash covers them.
  - **`anomalies/evidence.py`** (pure):
    - 47 required fields (`REQUIRED_FIELDS`) in eight groups: event, values, forecasts, changes, supply, weather, flags, shap. Each is a fact (value, unit, source, *known at the time* or *retrospective*) or `{"unavailable": reason}`, with no NaN, so it is jsonb-ready;
    - the peak detector and the demand peak follow `merge_events`' tie rule exactly;
    - profiles come from `profiles.SlotValues`, the same windows as the detectors, so the evidence's demand median is D1's. The percentile in the trailing 26 weeks is the share of intervals at or below the peak's value;
    - AEMO's matched runs: a fact is *retrospective* if it was published after the peak or its 30-minute period ends after it;
    - battery output is unavailable until the first non-zero bidirectional-unit interval at or before the peak; availability is unavailable before table v8;
    - `summarise` writes the three sentences from the evidence alone; `missing_fields` and `banned_in` are A4's checks.
  - **`anomalies/evidence_inputs.py`** loads only what the events need:
    - flags, battery output and availability;
    - stored forecasts, the QR band and M2's common mask, read one file at a time. A holdout event also reads run 2, because its day-ahead pair can be issued in late September 2025;
    - GFS from variant C's feature rows and ERA5, from a builder truncated at the series' end;
    - AEMO runs;
    - SHAP at demand peaks behind E1's gate.
  - **Fold fix in `explainability/stored.py`:** folds are clipped to the batch's period. Otherwise `holdout_main`'s first retrain also covers Aug–Sep 2025, which belong to run 2.
  - **Tests:**
    - 18 unit tests in `tests/unit/test_anomaly_evidence.py`, plus one for the fold clipping: completeness, jsonb readiness, three sentences and no banned phrase for every synthetic event; the demand peak of a mixed event; the profile median against D1's; AEMO labels; battery and availability start dates; flags; each detector's sentence; the stored lookup's reasons;
    - 2 integration tests of the loader in `tests/integration/test_anomaly_inputs.py`.
    - The unit suite has 416 tests; the full integration suite (82 tests, 11 min) passes.
  - **Evidence smoke run** (`make evidence-smoke`, 20 s, warm-up events of step 2b, no writes):
    - market pass: 33 events, all 47 fields complete, no banned phrase; no SHAP, as LightGBM C's forecasts start 2022-11-01;
    - residual pass: 7 events, complete, no banned phrase. **E1 passes on all 14 demand-peak pairs** (largest SHAP-sum error 1.1 × 10⁻¹¹).
    - The example summaries showed three wording bugs, all fixed with tests: "−$0" for small negative prices; "ranged from $19 to $19" for a price that moved by less than a cent (now "stayed at $19.18/MWh (to the cent)"); and "1 of the event's 1 intervals were".

**Step 4. Freeze, the development run, and the report.**
- Freeze record and unit test (§2.8).
- `gridcast anomalies-dev` from a clean commit stores the batch: per-interval scores and p-values (parquet under `<data root>/anomalies/`), events in `anomalies`, `model_runs` rows, and the injection results.
- `gridcast report-m4` → `reports/m4_anomalies.md` and figures: A1–A5 on development, power curves, the reference-event table, the catalogue and the case studies.
- *Outcome (2026-11-19, step 4):*
  - **Question round,** every recommendation chosen (§1, "Clarified in step 4"), committed `a7de6b2` before any freeze code.
  - **4a `2dc58fa`:** `anomalies/recipe.py`, `anomalies/freeze.py`, `inputs.split_band`, `anomalies/dev.py`, and `gridcast anomalies-freeze-check` / `anomalies-dev`. Before the freeze, the warm-up smoke runs reproduced step 3's results.
  - **4b `4990ebf`, the freeze:** hash `2ce8b3fc523ae81c`, code `2dc58fa`, pinned by a unit test. No development score existed yet.
  - **The development run** (run 104, batch `40cbe0f5-d232-4bb9-8b92-0402ab249086`) ran once from `4990ebf` with a clean tree, in about 3 min: load 1 s, detect 46 s, inject 36 s, cuts 87 s, evidence 5 s. It stored 420,768 intervals, 44,000 injections, 47 forests and 1,102 events.
  - **4c `74fc3a2` + `b0a5c5b`:** `gridcast report-m4`. The plan's readings of each criterion were written while the run was in progress, before any of its output was read. The report recomputed every tier, line, event and count from the stored run with no mismatch.
  - **Verdicts ([`reports/m4_anomalies.md`](../../reports/m4_anomalies.md)): A4, A5 and E1 (demand peaks) pass; A1, A2 and A3 fail.**
    - A1: 11 of 12 checks inside. D6's share at p ≤ 0.001 is 0.259%, above the 0.2% bound; its 1.79% at p ≤ 0.01 is inside.
    - A2: 6 of 7 class-P events caught. P05 (5 Aug 2024, 08:00–08:15 local, $17,500 for 2 intervals) was missed; the lowest p of any calibrated detector within 30 min of it was D4's 1.48 × 10⁻².
    - A3: 3 of 11. D2 reached 100% at ±1,000 MW and D5 91.4% at $5,000. Below 90%: D1 ±2,000 MW 33.2% / 33.4%; D3 ±2,000 MW 79.6% / 80.4%; D4 $15,000 10.7% and −$1,000 32.8%; D6 ±2,000 MW 1.9% each. These are the four detectors that step 2b's smoke run flagged.
    - A4: all 1,102 events complete, none with a banned phrase. A5: the synthetic suite passed 7 of 7, the 3 real cuts showed 0 differences, and no forest leaked. E1: 422 demand-peak pairs, 0 mismatches.
  - Reported only: 475 calibrated events (104 severe) and 627 price-line-only events.
  - **Decided afterwards (2026-11-20):** a disclosed v2 for D4 and D6 (§1, "Decided after the development verdicts"; §2.10), after the read-only diagnosis in [`docs/m4_v1_diagnosis.md`](../m4_v1_diagnosis.md). The holdout scores v1 and v2 once each, and M5/M6 serve v2.

**Step 4, v2 (disclosed revision).**
- **4e:** the question round and the diagnosis (`anomalies/diagnosis.py`, `gridcast anomalies-diagnose-v1`).
- **4f:** generalise the frozen modules behind a recipe version. `gridcast anomalies-reproduce` must reproduce run 104 exactly from the generalised code before anything else changes.
  - *Outcome (2026-11-20):* `anomalies/reproduce.py` and `gridcast anomalies-reproduce` were written first and run on the untouched code. Run 104 reproduced exactly: 420,768 intervals, 44,000 injections, 47 forest ranges, 3 cuts, 1,102 events with evidence and summaries, and the run record. A pass there shows the comparison itself is sound; unit tests show it catches a one-ulp change, a NaN that became a value, and a changed summary or evidence field.
  - Then `detectors.Recipe` (version, D4's and D5's price scales, D6's detector) went through `compute_scores`, `run_detectors`, `alert_tiers`, `detect`, the injections, the cut checks and the case ties. Version 1 is the default everywhere. Run 104 again reproduced exactly from the generalised code (a check on a dirty tree; the recorded check runs at v2's freeze commit, 4g).
  - Until that record, `test_frozen_anomaly_code_is_unchanged` fails, because files under v1's `FROZEN_PATHS` changed since `2dc58fa`. That is the expected state between the code and its freeze record.
- **4g:** v2's code (D4's transform, the robust distance with rolling calibration, its injection, evidence sentence and causality tests), the `anomalies_v2` config, and v2's freeze.
  - *Outcome, code (2026-11-20):*
    - new and frozen: `anomalies/distance.py`;
    - `Recipe` gains the robust distance (a rolling detector in v2), plus `recipe.recipe_for` and `recipe_config_hash`;
    - the `anomalies_v2` config section;
    - a version option on `anomalies-smoke`, `evidence-smoke`, `anomalies-freeze-check`, `anomalies-dev` and `anomalies-reproduce` (Make: `VERSION=2`).
  - Tests:
    - the causality suite runs for both versions (7 tests each, controls included);
    - `tests/unit/test_anomaly_v2.py`: on synthetic data, v2 reproduces v1's D1, D2, D3 and D5 exactly; D4 is \|z\| of asinh(price / $100); D6's p-value is against its own trailing scores;
    - the distance's form, its batch invariance, and its growth beyond the training range;
    - the injection harness equals a full recompute for v2's D4 and D6.
  - **Warm-up smoke runs under v2** (bugs only, before the freeze): no error. D1, D2, D3 and D5 match v1's step-2b smoke run. The preview at the largest sizes: D4 $15,000 100% and −$1,000 89%, and D6 ±2,000 MW 100% (v1's smoke run: 28%, 52% and 1–2%).
    - D6's alert rate in the pseudo-scored warm-up weeks is low: 0.11% at p ≤ 0.01. v1's D1 showed 0.14% there and passed A1 in development, so this predicts nothing on its own.
    - The evidence smoke run: 24 of 24 events complete, 0 banned phrases, E1 passes on 14 pairs.
  - **The freeze (2026-11-20).** The code is `04da607`. From a clean tree there, `gridcast anomalies-reproduce --version 1` reproduced run 104 exactly: 420,768 intervals, 44,000 injections, 47 forest ranges, 3 cuts, 1,102 events and the run record, in 181 s.
    - v1's record now reads hash `2ce8b3fc523ae81c` (unchanged), code `04da607`, carried from `2dc58fa` by that reproduction.
    - v2's record is hash `44abbf4b24cf6ac8`, code `04da607`.
    - Both are pinned by `tests/unit/test_anomaly_freeze.py`. No v2 development score existed when they were recorded.
- **4h:** v2's one development run and `report-m4` for v2 (`reports/m4_anomalies_v2.md`), with v1 beside it and every v2 verdict labelled post hoc.
  - *The first attempt stored nothing (2026-11-20).* `anomalies-dev --version 2` ran from `c0436a8`, a clean tree with both freezes holding. It computed every output, then failed on insert: the `anomalies` table's detector list (migration 0007) named v1's detectors only and rejected `robust_distance`. The run row and events were in one transaction, which rolled back, and the files were removed, so no v2 run was stored.
    - Fix: **migration 0008** adds `robust_distance` to that list and changes nothing else. `sql/` is not under the frozen paths, so both freezes still hold. An integration test now stores v2 events.
    - **Disclosed:** what was seen of that attempt's output. While `report-m4`'s v2 code was being written, its progress log was read once; it showed that the first two real cuts had 0 differences. The error message then showed the fields of 11 events (dates, detectors, lowest p, summaries; 10 in May 2022, 1 in Sep 2025). Nothing else was read, and no setting or reading changed.
    - The rerun uses the same frozen code and data, and the run is deterministic.
  - *Outcome (2026-11-20):*
    - **The development run** (run 106, batch `0ff972b0-9d46-4ca0-a84f-86d85c450be1`) ran from `7b989af` with a clean tree (frozen code `04da607`), in about 2.5 min. It stored 994 events: 388 calibrated (76 severe) and 606 price-line-only.
    - **`report-m4 --version 2`** wrote [`reports/m4_anomalies_v2.md`](../../reports/m4_anomalies_v2.md). It recomputed the stored run with no mismatch. v2's D1, D2, D3 and D5 (20 stored columns) equal run 104's exactly.
    - **Verdicts, post hoc: A1, A2, A4, A5 and E1 pass; A3 fails** (5 of 6, against v1's pre-registered 3 of 6, which stands):
      - A1: 12 of 12 inside. D4 1.07% / 0.153% and D6 1.41% / 0.141% (p ≤ 0.01 / p ≤ 0.001).
      - A2: 7 of 7. P05 was caught by D4 at p 7.44 × 10⁻⁴; it was known to be missed before v2 was designed.
      - A3: 5 of 11. D6 reached 100% at ±2,000 MW (v1: 1.9%). D4 reached 84.6% at $15,000 and 66.2% at −$1,000 (v1: 10.7% and 32.8%), still below 90%. D1 and D3 are as v1.
      - A4: 994 of 994 complete, none with a banned phrase. A5: the suite's v2 cases 7 of 7, the 3 cuts 0 differences, 0 month-model leaks. E1: 424 demand-peak pairs, 0 mismatches.
    - **Next:** step 5 (explanations; independent of the version), then step 6: the holdout, scored once for v1 and once for v2.

**Step 5. Explanations report.**
- `gridcast explain-dev` (the global SHAP sample and the E1 gate).
- `gridcast report-m4-explain` → `reports/m4_explainability.md`: global SHAP, stability, question 5 for demand, and price associations on development data.

**Step 6. Holdout, once per recipe version (v1 and v2; §1).**
- `gridcast evaluate-holdout-anomalies` (the A6 lock, per recipe hash): one continuous run per version that must first reproduce that version's development batch exactly (for v1, the reproduction that also carries its code freeze).
- `gridcast report-m4-holdout` → `reports/m4_holdout.md`:
  - A1, A3–A5 and E1 as holds or does not hold;
  - the class-O day;
  - holdout SHAP;
  - holdout price associations;
  - case studies by the same rule.
- README rewrite (every number from the reports); plan and handoff outcomes.

## 5. New dependencies and compute

- **Dependencies:** none. scikit-learn (`IsolationForest`), LightGBM (`pred_contrib`), numpy, pandas and matplotlib are already installed.
- **Detectors:** minutes per run. About 525,000 intervals and 6 calibrated detectors, plus 53 monthly Isolation Forest fits on about 52,000 rows each.
- **Injection:** 44 (detector, size, sign) settings × 1,000 = 44,000 injections per period, each re-scoring at most 24 intervals: minutes.
- **SHAP:** about 17 retrains × 4 bands × 20,000 rows, plus the event peaks. Cost depends on tree counts; step 3 measures it (expected under an hour).
- **Storage:** under 1 GB under the data root (scores, SHAP values), gitignored. Events and evidence live in PostgreSQL.
- **Needs:** Docker running (`make db-up`) and the BIWIN SSD mounted.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Market regime shifts (the 2022 crisis vs a calm holdout) make trailing windows lag, so alert rates drift | 26-week windows adapt within half a year; rates are reported by month; A1's factor-2 band; the holdout check is reported, not hidden |
| Autocorrelated 5-minute scores make one event many alerts | Events are the unit of the catalogue; A1 is a rate, not an event count |
| An extreme period raises the next 26 weeks' thresholds (masking) | Disclosed; rates and power by month show it |
| A $300 line floods 2022 with events | Line-only events reported separately (§2.4) |
| Fixed settings turn out badly chosen (e.g. MAD floors) | No tuning by design; a failure is published. A fix needs a recorded reason, a new hash and a disclosed rerun, as in M2's second development run |
| Few reference events, and none in the holdout | A2 is a face-validity check, stated as such; A3 carries the detection-power claim on both periods |
| SHAP credit is split among correlated features | Families as the main unit; "association, not cause" in every SHAP section |
| Evidence reads as a causal story | Facts-only templates, the A4 banned-phrase test, and *known at the time* vs *retrospective* labels |
| The flag bits mean something other than assumed | Step 1a checks the MMS Data Model before any detector code runs (§3.1) |
| M4 code changes the frozen M2/M3 behaviour | New packages and a new config section; the M2 and M3 freeze tests still run |

## 7. Out of scope for M4

- Price ML (M2b), price-forecast residual detection, and SHAP for price.
- Data-feed anomalies (stale or missing feeds) belong to monitoring (M7).
- Serving events and explanations through the API (M5). The `anomalies` table is designed for `GET /anomalies`.
- The dashboard (M6).
- New downloads (interconnector flows, unit-level output by fuel). AEMO forecasts before Sep 2024, for earlier events' evidence, are a possible follow-up.
- Regions other than Victoria; live ingestion.
- Any causal analysis.

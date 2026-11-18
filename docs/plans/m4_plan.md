# Milestone 4 plan: market anomalies and demand-model explanations

**Goal:** answer the brief's questions 4 and 5 for Victoria: *when is the market behaving abnormally?* and *which factors are associated with unusually high demand or price?*
- **Anomaly detection:** detectors whose false-alarm rate is stated and checked, that catch the operator-flagged price events and injected disturbances of known size. Each alert is merged into an event that carries observable evidence and makes no causal claim.
- **Explanations:** SHAP explanations of the frozen M2 demand model (LightGBM C), and a descriptive, association-only analysis of high and negative prices.

**Status:** planned 2026-11-15 after three question rounds with the user (every recommendation chosen). Branch `m4-anomalies-explainability`, from `main` at `c5fd5d3` (M1–M3 merged). The success criteria in §3 and every detector setting in §2 were fixed before any detector code or score existed.

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
    - **Cost:** predict + SHAP took 1.4–2.7 s per 31-leaf fit and 24.4 s for the 127-leaf fit. Scaled by tree count, the full global samples need about 11 min (development, 48 fits) and 5 min (holdout, 20 fits), well under §5's hour. Feature building is negligible.

**Step 4. Freeze, the development run, and the report.**
- Freeze record and unit test (§2.8).
- `gridcast anomalies-dev` from a clean commit stores the batch: per-interval scores and p-values (parquet under `<data root>/anomalies/`), events in `anomalies`, `model_runs` rows, and the injection results.
- `gridcast report-m4` → `reports/m4_anomalies.md` and figures: A1–A5 on development, power curves, the reference-event table, the catalogue and the case studies.

**Step 5. Explanations report.**
- `gridcast explain-dev` (the global SHAP sample and the E1 gate).
- `gridcast report-m4-explain` → `reports/m4_explainability.md`: global SHAP, stability, question 5 for demand, and price associations on development data.

**Step 6. Holdout, once.**
- `gridcast evaluate-holdout-anomalies` (the A6 lock): one continuous run that must first reproduce the development batch exactly.
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

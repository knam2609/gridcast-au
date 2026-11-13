# M3 step 1: pre-scoring spike (conformal candidate grid and cold threshold)

Run on 2026-11-10, after the interval core and its leakage tests existed and **before any scored interval existed**. Script: [`scripts/spikes/m3/prescoring_grid.py`](../scripts/spikes/m3/prescoring_grid.py). Command: `uv run python scripts/spikes/m3/prescoring_grid.py docs/m3_prescoring_tables.md` (about 6 minutes; section 4 needs the database). Every number below is copied from the generated tables in [`m3_prescoring_tables.md`](m3_prescoring_tables.md).

**Data.** The script reads M2 run 2's stored development forecasts (batch `main` = `6d783932-…`) and keeps **only issue times before 2023-08-01**, the pre-scoring data of [`plans/m3_plan.md`](plans/m3_plan.md) §2.1. Every candidate stream runs from the first ML issue time (2022-11-01), on past data only. Candidates are compared on **issue days 2023-02-03 → 2023-07-31**, when every window of up to 13 weeks is full. 26-week windows cannot be evaluated on pre-scoring data; they stay in the grid as the plan set them.

**Proxy.** CQR (the headline) needs quantile regression (step 2), which does not exist yet. The comparisons below therefore use conformal intervals around LightGBM C's point forecasts, plus the three baselines for the ACI question. The fixed settings apply to every method.

**Questions** (plan §2.2, §2.4, step 1): which time-of-day blocks, which ACI steps, which cold threshold.

## 1. Residual spread by local hour

LightGBM C's residual spread depends strongly on the target's local hour. The width of the central 80% of residuals at leads 6 h 5 min → 24 h is about 390–440 MW overnight and 1,300–1,390 MW at 11:00–14:00, the rooftop-solar hours. A single calibration set per lead therefore gives intervals that are too wide at night and too narrow at midday.

## 2. Time-of-day blocks

Seven block layouts, per lead × block with γ = 0, were compared with per-lead calibration at the same window length (tables, §2).

- **Every layout helps** at all three levels and window lengths. The 80% interval score falls by 7.8–8.0% for 2 blocks (07–22, 22–07) and by 14–15% for blocks of 3 h or less.
- **The gain flattens at 3 h.** Averaged over the 4-, 8- and 13-week windows, the change in IS80 is −14.2% for 8 × 3 h, −14.7% for 12 × 2 h and −14.5% for 24 × 1 h. The change in IS95 is −20.3%, −20.2% and −18.4%.
- **Smaller blocks have fewer calibration points.** A 4-week window holds 168 issue times per 3 h block against 112 per 2 h block, which matters for the 95% bound and for ACI.
- Coverage stays near nominal for every layout (80% level: 79.9–82.3%).

**Decision (user, 2026-11-10): 8 × 3 h**, blocks starting at local hours 0, 3, …, 21 (recommended option).

## 3. ACI steps

Per-lead streams, γ ∈ {0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05}, windows of 4, 8 and 13 weeks (tables, §3).

- **LightGBM C: no γ > 0 improved the interval score.** With a 13-week window, IS80 is 1,119.8 MW at γ = 0, 1,123.9 at 0.001, 1,170.1 at 0.005 and 1,229.5 at 0.01. ACI does move pooled coverage towards nominal (82.3% → 80.3% at 0.001).
- **Large steps under-cover long leads.** At leads 6 h 5 min → 24 h, 80% coverage is 78.4% at γ = 0.01, 74.9% at 0.02 and 65.1% at 0.05 (13-week window). For lead *h* the coverage outcome arrives *h* later, so large steps overshoot. At γ = 0.05, 7.6–9.5% of the 50% and 80% bounds sit at the window's extreme score, where ACI can no longer correct.
- **The baselines differ.** Persistence improves at γ = 0.001 (13 weeks: IS80 3,509.6 → 3,470.2 MW, coverage 76.1% → 79.7%), and weekly naive improves at γ = 0.005–0.01. Daily naive, like C, is best at γ = 0.
- **By month** (C, 80%), June–July 2023 coverage at γ = 0 is 78.1–79.7%, slightly below nominal. γ = 0.005 does not change that materially.

**Decision (user, 2026-11-10): γ ∈ {0, 0.001, 0.005}** (recommended option). The per-retrain selection can still choose γ = 0, and both positive steps keep clipping below 2% and C's coverage near nominal on pre-scoring data. Rejected: {0, 0.002, 0.01} (0.01 already under-covers C's long leads), {0} (no adaptive option for drift), and {0, 0.005}.

## 4. Cold threshold

The group is defined by the GFS day-ahead forecast of the population-weighted temperature, taking the maximum over each local day (tables, §4). There are 272 complete days, 2 Nov 2022 → 1 Aug 2023. The only winter in pre-scoring data is June–July 2023 (61 days; minimum forecast daily maximum 9.4 °C, median 14.7 °C).

- Forecast daily maximum ≤ 11 °C: 1 day; ≤ 12 °C: 11 days; ≤ 13 °C: 22 days; ≤ 14 °C: 37 days.
- Hot days (forecast daily maximum ≥ 35 °C, the existing hot group): 8 (ERA5: 5).

**Decision (user, 2026-11-10): forecast daily maximum ≤ 12 °C** (recommended option): close in rarity to the hot group, about the coldest 4% of pre-scoring days. In step 4, the group is evaluated on the forecast available at the issue time.

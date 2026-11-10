# M3 handoff: start here

Read this, then [`m3_plan.md`](m3_plan.md). For the M2 history, see [`m2_handoff.md`](m2_handoff.md) and [`m2_plan.md`](m2_plan.md).

> **Update, 2026-11-10 (ninth session, step 1 done):** two commits.
> - **`819e862` (step 1a):** the interval core in `src/gridcast/intervals/` (scores; exact rolling order statistics; conformal with time-of-day grouping, per-side ACI and per-retrain selection; the CQR form, ready for step 3), with its tests.
>   - The tests: an exact match against a slow reference, and leakage property tests whose controls catch an off-by-one information rule.
>   - Also in this commit: migration 0005 (`model_kind = 'interval'`) and `gridcast intervals-conformal`.
>   - **Pre-scoring spike** ([`docs/m3_prescoring_spike.md`](../m3_prescoring_spike.md)). The user chose the recommended option each time: 8 × 3 h time-of-day blocks, ACI γ ∈ {0, 0.001, 0.005}, cold days at forecast daily maximum ≤ 12 °C. These were recorded in the plan and `configs/base.yaml` before any scored interval existed.
> - **Step 1b:** `gridcast intervals-conformal` ran from `819e862` in 308 s and stored interval batch **`72eb5209-9bef-4870-a375-131a86ee7e0f`**: interval config hash `d34273400899af69`, 1.4 GB under `<data root>/intervals/`, log `<data root>/logs/m3/intervals-conformal-20261110.log`.
>   - Pooled development results printed by the command (reported only; U1–U3 judge CQR, from step 3):
>
>     | run | cov50 % | cov80 % | cov95 % | IS80 | WIS |
>     |---|---|---|---|---|---|
>     | conformal_lgbm_C | 50.3 | 79.1 | 94.4 | 1267.2 | 150.6 |
>     | conformal_persistence | 49.4 | 80.0 | 94.0 | 2717.0 | 384.9 |
>     | conformal_daily_naive | 49.8 | 79.5 | 94.2 | 2407.3 | 281.7 |
>     | conformal_weekly_naive | 50.7 | 78.3 | 92.8 | 2752.1 | 322.5 |
>
>   - Every LightGBM C selection chose a per lead × time-of-day stream. At 80%, an ACI stream (γ > 0) won 23 of the 36 (retrain, band) selections, although ACI did not help on pre-scoring data. No interval was empty.
>
> **Next:** step 2, the quantile LightGBM walk-forward (development). Reuse run 2's per-retrain selections (plan §2.3), smoke-run the first retrain (2022-11-01) to measure the cost, and ask the user again before the full run if it looks longer than about 10 h.
>
> **Update, 2026-11-10 (eighth session, step 0 done):** the AEMO POE spike is in [`docs/m3_data_spike.md`](../m3_data_spike.md) (script `scripts/spikes/m3/aemo_poe.py`; four sampled months cached under `<data root>/raw/spikes/m3/nemweb/`).
> - `PDPASA_REGIONSOLUTION` (`LOR` runs) archives every 30-minute run with `DEMAND10/50/90`.
> - Its POE90–POE10 band is exactly symmetric and covered 66–78% of actual demand.
> - `DEMAND50` sits an unexplained ~+40 MW above the matched pre-dispatch `TOTALDEMAND`.
> - **User decision:** include the comparison, with the band as published as the headline and a band re-centred on pre-dispatch beside it; disclose the offset. The plan's §1, §2.6 and step 0 are updated.
>
> **Next:** step 1. Build the interval core (`src/gridcast/intervals/`: scores, rolling conformal, ACI, per-retrain selection, leakage tests). Then run the pre-scoring spike (issue times before 2023-08-01 only) that fixes the ACI step values, the time-of-day blocks and the cold threshold, and record them in the plan before any scored interval exists. Then `gridcast intervals-conformal`.
>
> **Update, 2026-11-10 (eighth session, M3 planned):** three question rounds with the user settled the M3 design, recorded in `m3_plan.md` §1 before any interval code or result existed:
> - **Methods:** conformal intervals around the stored forecasts, quantile LightGBM C, and CQR (the pre-registered headline).
> - **Scope:** demand only, at 80% (with 50% and 95% also reported).
> - **Settings:** conformal settings re-selected at every retrain from past data; QR reuses the point model's per-retrain choices.
> - **Extreme events:** grouped both by what was known at issue time and by outcome, reported only.
> - **Evaluation:** develop on issue times 1 Aug 2023 → 30 Sep 2025, freeze, then score the holdout once (U5). The holdout's point errors were already seen; this is disclosed.
> - **AEMO:** a step-0 spike decides whether AEMO's POE10/50/90 demand can be a benchmark.
>
> Success criteria U1–U5 are in `m3_plan.md` §3. Branch `m3-prediction-intervals`, from `main` at `a1b26bb`.
>
> **Next:** step 0 (the AEMO POE spike), then step 1 (interval core, conformal intervals, the pre-scoring spike that fixes the candidate grid).

## Local environment

Unchanged from M2 (see `m2_handoff.md`, *Local environment*):
- **Data:** under `GRIDCAST_DATA_ROOT=/Volumes/BIWIN/gridcast-au/data` on the external SSD. Mount it first.
- **Database:** PostgreSQL in the Docker volume `gridcast-au_pgdata` (`make db-up`).

Inputs M3 reads:

| Input | Where |
|---|---|
| Development point forecasts (run 2) | batch `main` = `6d783932-5834-4e18-a9e8-0e3567db46e6`, parquet under `<data root>/backtests/` |
| Run 2's per-retrain selections (for QR) | the `model_runs.config` records of that batch |
| Conformal intervals around C and the baselines (step 1) | batch `72eb5209-9bef-4870-a375-131a86ee7e0f`, parquet under `<data root>/intervals/` |
| Holdout point forecasts (step 6 only) | batch `holdout_main` = `fcb6abb6-49a3-4923-9d79-64e05a44d75f` |
| Frozen M2 recipe (must stay reproducible) | `backtest.frozen_ml_config_hash` = `b88eae0a04e176af`, code at `96a6e50` |

Sanity check before starting:

```bash
ls /Volumes/BIWIN/gridcast-au/data         # SSD mounted?
make db-up && uv run alembic current        # expect 0005 (head)
make lint typecheck test test-integration
```

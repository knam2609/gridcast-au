# M3 handoff: start here

Read this, then [`m3_plan.md`](m3_plan.md). For the M2 history, see [`m2_handoff.md`](m2_handoff.md) and [`m2_plan.md`](m2_plan.md).

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
| Holdout point forecasts (step 6 only) | batch `holdout_main` = `fcb6abb6-49a3-4923-9d79-64e05a44d75f` |
| Frozen M2 recipe (must stay reproducible) | `backtest.frozen_ml_config_hash` = `b88eae0a04e176af`, code at `96a6e50` |

Sanity check before starting:

```bash
ls /Volumes/BIWIN/gridcast-au/data         # SSD mounted?
make db-up && uv run alembic current        # expect 0004 (head)
make lint typecheck test test-integration
```

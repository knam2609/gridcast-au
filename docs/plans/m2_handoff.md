# M2 handoff: start here

Written 2026-11-02 at the end of the M1 session. Read this, then [`m2_plan.md`](m2_plan.md).

> **Update, 2026-11-07 (fifth session, after the second run):** the second development run finished (15.4 h, plus 32 min for the drop/add test). The regenerated `reports/m2_demand.md` applies the pre-registered rule: **the second run's config (four bands, *latest* target form) is adopted for the holdout**. Batch ids and observations are in `m2_plan.md` step 4. **Next:** the user reviews the choices made overnight. Then step 5 (AEMO benchmark: migration `0004_aemo_forecasts`, pre-dispatch downloads ~2.5 GB under the data root), starting with a question round. Tests: 197 unit + 32 integration.
>
> **Update, 2026-11-07 (fifth session, overnight):** the user agreed to one disclosed second development run. Its design was pre-registered in `m2_plan.md` (step 4, *Second development run*) before any result: four lead bands plus the *latest* target form, chosen on pre-scoring data (`scripts/spikes/m2/band_layout.py`), and the adoption rule that `report-m2` applies in code (section *Development runs*). `make backtest-ml && make weather-ablation` was started from the clean commit that adds it; the log is `<data root>/logs/run2-*.log`, about 11–14 h expected. **Next:** check that the run stored three new batches and the ablation file, then `make report-m2` from a clean commit. Read the *Development runs* verdict, record the outcome under step 4, and show the user before step 5 / the freeze.
>
> **Update, 2026-11-07 (fifth session, step 4):** step 4 is done. `make weather-ablation`, then `make report-m2`, writes `reports/m2_demand.md` with S1–S4 computed in code (all four pass) and the leakage tests run by the command itself. The drop/add test's result lives in `<data root>/ablations/weather_<main batch>.json`. **Next:** (1) ask the user whether to make one disclosed second development run that fixes the 5-minute and 2 h band-boundary weaknesses before the freeze (decision of 2026-11-06: report first, decide after); (2) step 5, the AEMO benchmark (migration `0004_aemo_forecasts`, pre-dispatch downloads ~2.5 GB under the data root). Tests at the step-4 commits: 195 unit + 32 integration.
>
> **Update, 2026-11-06 (fifth session, after the full run):** step 3 is done. The full walk-forward took 7.7 h from the clean commit `15623d6` and stored batches `main` `b424894e-…`, `plus` `a57961eb-…` and `delay` `b2aa39b2-…` (full ids, selection summary and two observations are under step 3 in `m2_plan.md`). Model files are under `<data root>/models/<batch>/` (~0.2 GB). **Next:** step 4 (`gridcast report-m2`), starting with a question round. The two observations to settle are LightGBM losing to persistence at 5 min and the MAE jump at the 2 h band boundary.
>
> **Update, 2026-11-06 (fifth session):** step-3 code is done and committed: `gridcast backtest-ml` (`make backtest-ml`) in `src/gridcast/forecasting/ml_backtest.py`, with the harness in `forecasting/walkforward.py` and LightGBM/ridge in `models/ml.py`. Decisions (all in `m2_plan.md` §2.4 and §4 step 3): staggered 30 min × 15 min training sample; **settings re-selected at every retrain** on its last 8 weeks, then the winner is refit on the whole window (the pre-scoring tuning window was dropped, `scripts/spikes/m2/tuning_window.py`); **two lead bands** (≤ 2 h, > 2 h) per model and a third target form, **anchored** (`scripts/spikes/m2/target_forms.py`); 255 leaves dropped from the grid (`scripts/spikes/m2/grid_cost.py`). Results are stored as three batches per run (`config.evaluation_set` = `main`, `plus`, `delay`), with model files under `<data root>/models/<batch>/`. **Next:** check that the full walk-forward run (~8–10 h, started from the step-3 commit, log under `<data root>/logs/`) finished and stored its three batches, record its outcome under step 3 in the plan, then step 4 (`gridcast report-m2`). Test counts at the step-3 commit: 173 unit + 31 integration. LightGBM needs `brew install libomp` on macOS.
>
> **Update, 2026-11-04 (fourth session):** step 2 (feature builder and audit) is done. `FeatureBuilder` in `src/gridcast/features/builder.py` builds variants `A`, `C`, `D`, `Cplus`, `Dplus` for any (origins, horizons); `load_feature_builder(engine, cfg, region, delay_hours=None)` in `features/inputs.py` loads everything from the database. Generated documents: `docs/feature_catalogue.md` (`make catalogue`, from code; a unit test fails if stale) and `reports/m2_feature_audit.md` (`make feature-audit`, from the database). Decisions are in `m2_plan.md` §4 step 2, including the adopted 8 h delay sensitivity check. **Next task: step 3 (models in the walk-forward harness).** Settle the training-sample inconsistency noted under step 3 in `m2_plan.md` first. Test counts at the step-2 commit: 147 unit + 30 integration.
>
> **Open data observation (not a step-2 problem, cause unverified):** on 2024-12-09 VIC1 demand rises from 2,679 MW (15:00 local) to 5,518 MW (15:40) and falls to 2,418 MW at 15:45; price moves with it (−$21 → +$95 → −$146/MWh). Feature `dem_std_1h` peaks there. Worth a look in the step-4 failure cases and in M4 (anomaly detection). The other large 1-hour swings include the 2024-02-13 storm.
>
> **Update, 2026-11-03 (third session):** step 1 (weather ingestion) is done: migration `0003_weather`, `gridcast ingest-weather` / `make weather`, six new weather data-quality checks plus `weather_clipped_values`, and alignment helpers in `src/gridcast/weather/`. All 60 months of GFS forecasts and ERA5 for five sites are loaded, and `gridcast dq` passes, with warnings only for known gaps and clipped artefacts. Details: `docs/data_sources.md` §4–5. **Next task: step 2 (feature builder and audit).** It should use `load_weather_forecasts` / `load_weather_observations` → `site_weighted` → `to_five_minute`, then pick the freshest offset with `available_at <= o`.
>
> **Update, 2026-11-02 (second session):** step 0 is done. Findings and decisions are in [`docs/m2_data_spike.md`](../m2_data_spike.md) and folded into `m2_plan.md` §1–§6. **Next task: step 1 (weather ingestion).** The step-0 section below is kept for the record.

## Where things stand

- **M1 is complete and pushed:** branch `m1-pipeline-baseline` (11 commits) is on GitHub. **No PR has been opened, and it isn't merged into `main`.** Open the PR via https://github.com/knam2609/gridcast-au/pull/new/m1-pipeline-baseline when ready.
- **M2 is planned, nothing implemented:** `m2_plan.md` holds the agreed decisions, the design, the success criteria S1–S5 (set before any results), and steps 0–6.
- **This branch:** `m2-weather-demand-ml`, created from `m1-pipeline-baseline`. It contains the plan, this note, the data-root setting, and the step-0 spike (doc + `scripts/spikes/m2/`).

## Local environment (this machine)

| Item | State |
|---|---|
| Python | uv 0.13 (Homebrew) + Python 3.12 managed by uv. Homebrew's default `python3` is 3.15; don't use it for the project |
| Database | Docker container `gridcast-db` (PostgreSQL 17) on **localhost:5433**, **stopped** at handoff (`make db-up` to start; the data persists in the `pgdata` Docker volume, ~117 MB). Credentials are in `.env` (gitignored, randomly generated password). Alembic is at revision `0003` (database ~250 MB after the weather load) |
| Data loaded | VIC1, Oct 2021 → Sep 2026: 525,888 intervals, `calendar_days` 2021–2027, latest DQ run all pass/warn |
| Latest backtests | demand batch `5836b22c-39cc-4eec-a46c-ee22dfbb7cc6`, price batch `07acd3a0-8b48-43b1-8d23-908ba23ac0c6` (clean commit `945bd40`) |
| Data files | **On the external BIWIN SSD** (exFAT, USB): `GRIDCAST_DATA_ROOT=/Volumes/BIWIN/gridcast-au/data` in `.env`, holding `raw/` (~433 MB AEMO cache, plus `raw/open_meteo/`: 600 cached weather responses, ~0.3 GB on disk), `interim/` and `backtests/` (~0.9 GB, current batches only; superseded batches deleted). **Mount the SSD before running the pipeline**, or it stops with a clear error. Local `./data` is empty. `model_runs.artifact_path` is stored relative to the data root |
| Test database | `gridcast_test` on the same server, created automatically by integration tests |

Sanity check before starting:

```bash
ls /Volumes/BIWIN/gridcast-au/data         # SSD mounted?
make db-up && uv run alembic current        # expect 0003 (head)
make lint typecheck test test-integration   # 147 unit + 30 integration, all green at the step-2 commit
```

## M2 step 0 (data-availability check) — done, see `docs/m2_data_spike.md`

Output: `docs/m2_data_spike.md`. Answer each question, then apply the fallback rules agreed in `m2_plan.md` §4:

1. **Open-Meteo Previous Runs:** which hourly variables does `gfs_seamless` (or the right GFS id) have, and from what date, for Melbourne (-37.83, 144.98)? Only GFS `temperature_2m` from Mar 2021 is confirmed so far.
2. **Run timing:** use the Single Runs API on a sample to confirm which run supplies `previous_day1` / `previous_day2`. Then fix `publication_delay` (the plan assumes 6 h, giving the rule: lead ≤ 18 h → day1, else day2).
3. **Pre-dispatch "ALL" archive:** `MMSDM_YYYY_MM/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/`, e.g. `PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01#202501010000.zip` (~49 MB). Check the schema, run identifiers (`PREDISPATCHSEQNO`, `RUNNO`), whether `LASTCHANGED` matches run publication, and whether `TOTALDEMAND` is the period mean or the value at period end (compare with 5-min actuals).
4. **5-minute pre-dispatch:** check the size of `P5MIN_ALL_DATA`, and include it only if it's manageable.

## Things learned in M1 that matter for M2

- `aemo.com.au` and `aemc.gov.au` return **HTTP 403** to automated fetches. NEMWeb (`nemweb.com.au`) works and is the source to use. Use the project User-Agent `gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)`, and never put personal details in request headers.
- Never assume an MMS column is boolean or fixed-position: `APCFLAG` turned out to be a bit field (migration 0002), and table versions change. Map by name, validate strictly, and fail loudly.
- pandas 3 stores datetimes in **microseconds**. Never mix `asi8` with `Timedelta.value` (ns); use `(times - epoch) // INTERVAL`.
- Pandas deprecation warnings fail the test suite on purpose.
- Every number in the README or reports must come from generated reports (`gridcast report`), never be typed by hand.
- The data root is on an exFAT drive. Avoid globbing data directories (macOS can create `._*` AppleDouble files on exFAT); the NEMWeb cache already uses `listing.json` instead of globs. New M2 caches (weather, pre-dispatch) should follow the same pattern.
- Results that matter for M2 (development period, Nov 2021 → Sep 2025, from `reports/m1_baseline.md`):
  - **Demand:** persistence wins up to 1 h 35 min, then daily naive is best (pooled MAE 467.8 MW, MAPE 10.4%).
  - **Price:** the 28-day median profile is best beyond ~2.5 h (pooled MAE $62.9/MWh).
  - **Rescoring needed:** M2 ML scoring starts 2022-11-01, so the baselines must be rescored on exactly the ML pairs before S1–S3 are judged.

## Working agreements with the user

- Refine plans by asking focused questions with a recommended option; the user usually takes the recommendation but wants to be asked.
- Build in vertical slices on a feature branch, one commit per logical step. **Push only when asked.**
- Success criteria are set before results exist, and failures are reported plainly.

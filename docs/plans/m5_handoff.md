# M5 handoff: start here

Read this, then [`m5_plan.md`](m5_plan.md). For the M4 history, see [`m4_handoff.md`](m4_handoff.md) and [`m4_plan.md`](m4_plan.md).

> **Update, 2026-11-26 (twenty-sixth session): step 4 done** (local, not pushed). The API runs as a Compose service.
> - **Question round** (`5fa5a33`). The user chose every recommendation (plan §4, step 4):
>   - without the SSD, the API starts without its model files;
>   - the image records its commit and dirty flag, and `/health` serves them as `build`;
>   - the smoke check is a committed command, `make api-smoke`.
> - **Code `97b8436`:**
>   - `Dockerfile` (python:3.12-slim, uv 0.13.6, `uv sync --frozen --no-dev`, editable at `/app`, non-root, one uvicorn worker) and an allowlist `.dockerignore`, so `.env`, `.venv` and data never enter the build context.
>   - The Compose `api` service has profile `api`, so `docker compose up` alone does not build it. It listens on 127.0.0.1 only and has a health check on `/health`. `docker-compose.models.yml` mounts `<data root>/models` read-only at `/data/models` with `create_host_path: false`; `make api-up` adds it only when the directory exists.
>   - `make api-up` (build with the commit and the dirty flag, start, wait for healthy), `make api-down` (Postgres keeps running), `make api-smoke`. `make db-down` now also stops the API.
>   - `/health` gains `build`; `docs/openapi.json` is regenerated.
>   - `gridcast api-smoke` (`src/gridcast/api/smoke.py`) and `serving.smoke` in the config. It saves its result to `<data root>/logs/api_smoke_<UTC time>.json`.
> - **Found while building:**
>   - **Port 8000 is taken on this Mac** by another app. The host port is now `GRIDCAST_API_PORT` (default 8000), and `.env` sets 8001 here (a line added to the user's `.env`; `.env.example` documents it).
>   - **Logging was never configured under uvicorn,** so `GRIDCAST_LOG_LEVEL`/`GRIDCAST_LOG_FORMAT` had no effect (on the host too). The factory is now `gridcast.api.app:serve`. The container logs JSON.
> - **Measured** (the container on Docker Desktop, 4 CPUs, 3.8 GB; informal, not P3 or P5; `<data root>/logs/api_smoke_20261125_142901.json`):
>   - The clean image of `97b8436` is 458 MB (planned about 1 GB). Same Python, numpy, LightGBM and pandas versions as the host, on Linux aarch64.
>   - `make api-up` took 14 s with a cached build. At startup the features loaded in 4.21 s and the 68 boosters in 1.15 s.
>   - **Smoke: PASS.** At development 2024-03-15 02:00 UTC and holdout 2026-01-20 05:30 UTC, recompute matched 288 of 288 stored leads exactly, and explain passed E1's gate on 288 leads (0 forecast mismatches; max sum error 7.16e-12 and 1.09e-11).
>   - Client-side times (3 requests each, the first cold): recompute 74/43/44 ms (development) and 44/42/42 ms (holdout). Explain took 341/337/351 ms (development) and 217/195/195 ms (holdout), at about 1 MB per response.
>   - Memory after the smoke check: API 816 MiB, Postgres 284 MiB (`docker stats`).
>   - Without the SSD (simulated, `make api-up MODELS_DIR=/nonexistent/models`): `degraded`, model files named by relative path, recompute 503 with the reason, stored forecast 200, `api-smoke` exit 1. The clean image was then restarted with the models.
> - **Tests:** 590 unit tests (20 new: build record, Compose and Dockerfile guards, smoke logic, smoke on the synthetic walk-forward, the `serve` factory), 107 integration and 14 host tests pass.
> - **For step 5's question round:**
>   - **Development explain ran at about 340 ms** in the container, against P5's 500 ms limit (p95 over 200 seeded requests). The informal host median in step 3 was 174 ms. P5 is judged as pre-registered; the margin is narrower than the host suggested.
>   - **A docs-only commit makes the running image "stale"** (its `git_sha` is no longer HEAD), and `api-smoke` then fails until `make api-up` rebuilds it (about 15 s). `api-verify` should either require a rebuild or compare only the files the image copies.
>   - Should `api-verify` refuse a dirty image (`build.dirty`)?
> - **Next: step 5a** (`gridcast api-verify` and `gridcast report-m5`, committed before any run), then 5b (the run and `reports/m5_api.md`).

> **Update, 2026-11-26 (twenty-fifth session): step 3 done** (local, not pushed). Recompute and explain are in `src/gridcast/api/runtime.py` and two new endpoints.
> - **Question round** (`48143d4`). The user chose every recommendation (plan §4, step 3):
>   - gaps are reproduced;
>   - feature values are served beside their contributions;
>   - every booster loads at startup;
>   - real-model tests carry a `host` marker.
> - **Modules:**
>   - `runtime.py`: `ModelRuntime` loads the feature context and fits (Postgres), then every booster (data root), and retries what failed at most every 10 s. `PathModelCache` keys boosters by file, so the `2025-08-01` fits that both batches hold name the right batch's file. It composes the frozen `explainability` functions and changes none of them.
>   - `GET /v1/forecasts/demand/recompute` and `GET /v1/forecasts/demand/explain` (`lead` optional) in `app.py`. Their response models are in `models.py`. `docs/openapi.json` is regenerated.
>   - `/health` gains `feature_context` and `model_files`. If either fails, the status is `degraded` (the stored endpoints still work).
> - **Errors:**
>   - 503 until both parts are loaded, with the reason and no host path;
>   - 404 for explain where nothing is stored;
>   - a JSON 500 for a server fault;
>   - a 500 carrying the gate when E1 fails (explanations withheld).
> - **Bugs found and fixed before commit:**
>   - The `lead` bound referred to a local variable inside a postponed annotation, so FastAPI could not resolve it, and every explain request with `lead` was a 500. The bound is now checked in code (422).
>   - A stored value that is not float32 would have been a plain-text 500. It is now JSON.
> - **Measured** (host, in-process, informal; not P3 or P5):
>   - startup 4.9 s (features 3.9 s, 68 boosters 0.78 s); peak RSS 1.12 GB;
>   - warm medians: recompute 40 ms, explain of all 288 leads 174 ms;
>   - the explain body is about 1.05 MB for 288 leads, against my question-round estimate of about 0.3 MB; one lead is about 4.7 KB.
> - **Tests:**
>   - 570 unit tests, 16 of them new on a synthetic walk-forward whose fits mix target forms. They cover exact recompute with gaps, a one-ulp change, a misplaced gap, the decomposition and feature values, a failed gate (500), a missing model file (503, then recovery), an unmounted data root, the cache keyed by file, and batch checks.
>   - 107 integration tests. The fixture's batches are not the fits', so the integration health check now expects `degraded`, with the reason. The hypothesis search covers the two new endpoints: no 500.
>   - **`make test-host`: 14 tests in about 8 s, all passing** (main database and SSD, read-only). Recompute matched all 288 leads at 8 issue times: inside a fold of each split, both sides of a retrain boundary in each split, and both sides of the split boundary, each naming its own batch's `2025-08-01` files. It also matched where nothing was stored (2024-01-15 02:00 UTC) and at the first partly stored issue time. Explain passed E1 on 288 leads in each split.
> - **Next: step 4** (`Dockerfile`, the Compose `api` service mounting `<data root>/models` read-only, `make api-up` / `api-down`; then `/health` and one recompute per split against the container, a smoke check, not P3). The container needs about 1.1 GB for the API beside Postgres in the 4 GB Docker VM.

> **Update, 2026-11-23 (twenty-fourth session, later): step 2 done** (local, not pushed). The stored endpoints are in `src/gridcast/api/`.
> - **Modules:**
>   - `app.py`: the `create_app` factory; `make api` runs it with uvicorn `--factory`.
>   - `inputs.py`: timestamps, grids, the keyset cursor; errors map to 422/404/503.
>   - `catalogue.py`: a read-only engine (`default_transaction_read_only`); the latest successful publication per (product, split); provenance; pooled metrics; the v2 anomaly runs, checked against `anomalies_v2.frozen_config_hash`.
>   - `queries.py`, `models.py` (typed responses), `meta.py` (labels, the five disclosures, licences), `schema.py`.
> - **Contract:**
>   - `docs/openapi.json` is written by `gridcast api-schema` / `make api-schema`; a unit test fails on drift.
>   - Every error is `{"detail": "<message>"}`, including FastAPI's own validation errors.
>   - CORS is any origin, GET only.
>   - `serving.anomaly_runs` (106, 108) was added to the config, outside every hash.
> - **Exactness:** bounds come back from psycopg as their shortest float32 decimal. `queries.real` turns each into the exact float64 of the stored float32, as plan §1 says (points already were). A unit test checks the JSON round trip on 1,000 values.
> - **Tests:**
>   - unit: inputs, cursor, float exactness, schema drift, disclosures, the invalid-input table, database down → 503, GET only;
>   - integration on the published fixtures: every demand forecast and its bounds, evolution, price, actuals, anomaly paging (v2 only, keyset order), each event against its row, read-only engine;
>   - a hypothesis search over query parameters, 300 examples, with no 500;
>   - 549 unit and 107 integration tests pass.
> - **A bug found and fixed before commit:** the catalogue lock deadlocked (`Lock`, re-entered through the engine property). It is now an `RLock`.
> - **Smoke run against the main database (in-process TestClient, informal, not P5):** every endpoint answered in 1–16 ms, and `/health` was `ok`.
> - **Next: step 3** (recompute and explain, §2.5: feature context and fits loaded at startup, M2's code path, M4's `explain_pairs` with E1's gate, `/health` gains model-file and feature-context checks).

> **Update, 2026-11-23 (twenty-fourth session): step 1 done** (local, not pushed). Numbers come from the `publications` rows and the publish log (`<data root>/logs/publish_20261123_031938.log`).
> - **1a `fe217fc`:**
>   - migration 0009 (`forecast_intervals` with nesting and finiteness CHECKs; `publications`);
>   - `src/gridcast/serving/` (`sources.py` registry and refusals, `pgcopy.py` binary COPY, `publish.py`);
>   - `gridcast publish [--check-only] [--product] [--split]`, `make publish` / `publish-check`;
>   - a `serving` config section, outside every frozen hash (M2/M3/M4 hashes unchanged). It also fixes P2/P3/P5's sample sizes and limits before any measurement.
>   - 513 unit and 94 integration tests pass.
> - **1b:**
>   - The Docker VM disk had 51 GB free of 63 GB (memory 4 GB). Migration 0009 is applied to the main database.
>   - `make publish` took **5.3 min** (estimated 15–30). All 8 (product, split) publications succeeded.
>   - **P1 passes:** 63,378,240 rows were read back and equal their parquet bit for bit. The four existing 288-row samples were equal before writing.
>   - **P4 passes:** 0 of 15,708,288 interval pairs are non-nested.
>   - New interval runs: **109** (development) and **110** (holdout), `cqr_lgbm_C_rearranged`. Their M3 metrics from Postgres equal those from the parquet and are stored. Pooled 80% coverage is 79.27% and 79.65% (reports: 79.3% and 79.6%).
>   - The database grew from 1.6 to 10 GB; the VM disk now uses 18 of 63 GB, and the host has 35 GB free.
>   - `make publish-check` passes in 0.9 min with no writes.
> - **Step 2 question round (same session):** the user chose every recommendation (plan §4, step 2):
>   - a keyset cursor for `/v1/anomalies`;
>   - `/health` checks that the latest publication attempt of each of the 8 succeeded;
>   - `/v1/meta` carries one sentence per §3.3 disclosure, with its report.
> - **Next: step 2** (stored endpoints, `src/gridcast/api/`, adds `fastapi` and `uvicorn`).

> **Update, 2026-11-22 (twenty-third session): M5 planned.** Branch `m5-api` from `main` at `e338672`. Nothing is pushed.
> - **Three question rounds.** The user chose every recommendation except the storage layout:
>   - stored outputs plus an exact recompute;
>   - Postgres serving tables;
>   - issue times 2023-08-01 → 2026-09-30;
>   - price baselines, labelled;
>   - CQR rearranged;
>   - all three extras (per-forecast SHAP, recompute endpoint, forecast evolution);
>   - parity, latency and contract criteria;
>   - local Compose, read-only.
> - **Layout:** after I corrected my size estimate (63.4M rows, ~8 GB, not ~2 GB), the user chose one row per lead for every product over the recommended `real[]` arrays per issue time. I had also wrongly said `forecasts` was empty: it holds a 288-row sample for each of runs 1–80, and the two checked match their parquet exactly. Both corrections are in the plan, §1.
> - **Criteria P1–P6** (plan §3) were fixed before any API code or measurement. They are judged on the Compose service.
> - **Next: step 1a** (migration 0009, `src/gridcast/serving/`, `gridcast publish`), then **1b** (check the Docker VM disk, migrate, publish: ~15–30 min, +8–9 GB).

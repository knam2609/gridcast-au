# M5 handoff: start here

Read this, then [`m5_plan.md`](m5_plan.md). For the M4 history, see [`m4_handoff.md`](m4_handoff.md) and [`m4_plan.md`](m4_plan.md).

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

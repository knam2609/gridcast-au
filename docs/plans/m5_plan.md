# Milestone 5 plan: the API

**Goal:** serve what milestones 1–4 produced, as a read-only HTTP API:
- the demand forecasts and their prediction intervals;
- the price baselines;
- the actuals;
- the anomaly events (recipe v2).

Every value the API returns must be one that a milestone report scored, or an exact recomputation of one. A forecast can also be rebuilt from the stored model and explained on demand, and both must reproduce the stored forecast exactly. That shows there is no train/serve skew.

**Status:** planned 2026-11-22 after three question rounds with the user, on branch `m5-api` from `main` at `e338672` (M1–M4 merged). The user chose every recommendation except the storage layout (§1). The success criteria in §3 were fixed before any API code or measurement existed.

## 1. Decisions (agreed 2026-11-22)

| Topic | Decision |
|---|---|
| Scope | **Stored outputs + exact recompute.** The API serves the scored outputs by issue time. It can also rebuild a LightGBM C forecast from the stored model and the database's features, and the rebuild must equal the stored forecast exactly. M5 adds no data: no ingestion after 1 Oct 2026 and no new forecasts. Live data belongs to M7 |
| Storage | **Postgres serving tables.** A `gridcast publish` command loads the chosen runs into Postgres and checks every value against its parquet source. The API reads Postgres, plus the stored model files for recompute and explanations |
| Layout | **One row per (run, issue time, target), for every product** (the user's choice; the recommendation was one row per issue time holding `real[]` arrays). See the correction below |
| Period | **Issue times 2023-08-01 → 2026-09-30 (NEM).** That is interval development (from 2023-08-01) plus the holdout year, so every served demand forecast has its intervals. Each response names its split (development / holdout) |
| Price | **The M1 baselines, labelled as baselines:** persistence and the 28-day median profile, with their scored errors in the metadata. No price intervals. A price model stays a separate future milestone |
| Intervals | **CQR, rearranged:** per pair, the six CQR bounds sorted as quantiles, so the levels nest. This is M3's reported-only row, scored in both periods, not the pre-registered headline (§3.3) |
| Extras | **All three:** per-forecast SHAP (gated like E1), the recompute endpoint, and forecast evolution (every forecast issued for one target) |
| Criteria | **Parity + latency + contract** (§3) |
| Deployment | **Local Docker Compose, read-only, no auth.** An `api` service beside `db`, with the model files mounted read-only and CORS open for the M6 dashboard. Public hosting is decided with the dashboard in M6 |

**Corrected during planning (2026-11-22), before the layout question was re-asked:**
- **The `forecasts` table is not empty.** Earlier in the question round I said it was. It holds one sample issue time (288 rows) for each of runs 1–80, the "small backtest samples" its schema comment describes. Two of them (run 61, and price run 14) were compared with the parquet and match exactly. They are kept (§2.3).
- **The size estimate was too low.** In the storage question I quoted "~16M rows (~2 GB)", which counted the demand points only. Measured at 127 bytes per row, one row per lead for every product is about 63.4M rows and 8 GB. The layout question was asked again with that figure, and the user chose one row per lead.

**Added by this plan (for the user's review):**
- **Framework:** FastAPI on uvicorn, with pydantic v2 response models. SQLAlchemy Core with a pooled synchronous engine, and every connection read-only (`default_transaction_read_only`). Two new dependencies: `fastapi` and `uvicorn`.
- **Times:** every timestamp in and out is ISO 8601 in UTC. A timestamp without an offset is rejected (422). An issue time must lie on the 30-minute grid and a target on the 5-minute grid, inside the published range; otherwise 422 (off-grid) or 404 (not published).
- **Versioned paths** (`/v1/...`). `region` is a query parameter defaulting to `VIC1`; only `VIC1` is published.
- **Anomalies:** every v2 event in the `anomalies` table, i.e. development run 106 (from 2022-05-01, before the forecast period) and holdout run 108. v1's events stay in the database and are not served (M4: "M5 and M6 serve v2").
- **Pairs without a stored forecast are not published.** 286,080 development pairs (run 2's, mostly missing weather) have no forecast and no interval. The API returns all 288 leads of an issue time, with `null` where nothing was stored, and `publications` records the count.
- **The parity checks are exact,** as in M2–M4. Float32 values are stored exactly (as `real`, or `double precision` in `forecasts`) and written to JSON as their exact float64 value, so a client casting back to float32 recovers the stored bits.

## 2. Design

### 2.1 What is published

| Product | Source runs (batch) | Issue times (UTC) | Rows |
|---|---|---|---|
| Demand point, LightGBM C | Development: run 39, M2's run 2, `main` (`6d783932-…`). Holdout: run 61, `holdout_main` (`fcb6abb6-…`) | 2023-07-31 14:00 → 2025-09-30 13:30 (38,016), then 2025-09-30 14:00 → 2026-09-30 13:30 (17,520) | 15,708,288 |
| Demand intervals, CQR rearranged, 50/80/95% | Development: run 86 (`484194d7-…`). Holdout: run 97 (`2744fd09-…`, `5min/`). Sorted per pair by the frozen `intervals.scores.rearrange_bounds` | As the points | 15,708,288 |
| Price baselines: persistence, 28-day median profile | Runs 11 and 14, M1 batch `07acd3a0-…` (the batch `reports/m1_baseline.md` reports) | 2023-07-31 14:00 → 2026-09-29 14:00 (55,489; the M1 run's last issue time) | 2 × 15,980,832 |
| Anomaly events, v2 | Runs 106 (994 events) and 108 (133 events), already in `anomalies` | Events from 2022-05-01 → 2026-09-30 | not copied |
| Actuals | `market_intervals` (demand, price, flags) | 2021-10 → 2026-10-01 | not copied |

Row counts were computed from the parquet during planning; `publish` recounts them. The interval runs' rearranged bounds become two new `model_runs` rows (`cqr_lgbm_C_rearranged`, kind `interval`, one per split). Their config names the source run and the transform. Their `forecast_metrics` are computed by `publish` from the rows read back (§2.3).

### 2.2 Schema (migration 0009)

- **`forecast_intervals`:** `run_id`, `region_id`, `issue_time`, `target_end`, `horizon`, `lo95`, `lo80`, `lo50`, `hi50`, `hi80`, `hi95` (`real`). The primary key is (`run_id`, `issue_time`, `target_end`). A CHECK enforces lo95 ≤ lo80 ≤ lo50 ≤ hi50 ≤ hi80 ≤ hi95, so the database itself rejects a non-nested row.
- **`publications`:** one row per (product, split) publication:
  - the published `run_id` and its source run;
  - the source file, its SHA-256 and the issue-time range;
  - issue times, rows and skipped (missing) pairs;
  - git SHA, status, and the time it was published;
  - the P1/P4 check results (jsonb).

  `/v1/meta` and `/health` read it.
- **`forecasts`** is unchanged. Points go in `point`; `lower`, `upper` and `interval_level` stay `NULL` for these runs.
- **No new index.** A forecast is one primary-key range (288 rows). Evolution for target *T* scans the primary key over issue times [*T* − 24 h, *T*) and filters on `target_end`: at most 48 × 288 = 13,824 index entries. P5 checks that this is fast enough.

### 2.3 `gridcast publish`

For each product and split:
1. Read the parquet fold by fold and reshape it to one row per lead, dropping missing pairs (counted).
2. Compare any rows already in the table for that run (the 288-row samples) with the source; they must be equal.
3. COPY the rest into a staging table, then `INSERT … ON CONFLICT DO NOTHING`.
4. Read every row back and compare it with the parquet (**P1**). For intervals, also check nesting in SQL (**P4**).
5. Record the publication.

It is idempotent, so a rerun inserts nothing new and checks again. `--check-only` writes nothing. Before writing, it refuses if a source run is not `success`, or if its batch id or model name differs from §2.1.

**Interval metrics:** `publish` reads the rearranged bounds back from Postgres, with the actuals, and recomputes coverage, width and interval score per level and lead under M3's own scoring mask. It uses M3's evaluation code. The hit counts must equal what the same code gives from the parquet (**P1**), and are then stored in `forecast_metrics` for the two new runs. `/v1/meta` serves them, so no metric is typed by hand.

**Cost (estimated):** 15–30 min to load ~63.4M rows, and +8–9 GB in the Docker volume (41 GB free on the internal disk). Step 1b checks the Docker VM's disk first.

### 2.4 Endpoints

| Method and path | Parameters | Returns |
|---|---|---|
| `GET /health` | none | Database reachable, publications complete, model files readable, feature context loaded, data window |
| `GET /v1/meta` | none | Each product: its runs, batches, git SHAs, config hashes, issue-time range by split, and scored metrics by split (from `forecast_metrics`). Disclosures (§3.3), data licences and attribution (AEMO; Open-Meteo CC BY 4.0) |
| `GET /v1/forecasts/demand` | `issue_time` | 288 leads: `target_end`, `lead_minutes`, `point_mw`, and the six bounds. Also the split and the source run ids |
| `GET /v1/forecasts/demand/evolution` | `target_end` | Every published issue time in [*T* − 24 h, *T*) (≤ 48), each with its lead, point and bounds for that target |
| `GET /v1/forecasts/demand/recompute` | `issue_time` | The forecast rebuilt from the stored model: 288 values, `matches_stored`, the number of leads that differ, the largest absolute difference, and the fits used |
| `GET /v1/forecasts/demand/explain` | `issue_time`, optional `lead` | Per lead: forecast = anchor + base value + Σ contributions. Contributions are given by feature and by family, as in M4's event evidence. Withheld (error) if E1's gate fails |
| `GET /v1/forecasts/price` | `issue_time` | 288 leads of both baselines (`$/MWh`), labelled as baselines, with the split |
| `GET /v1/actuals` | `start`, `end` (≤ 7 days) | 5-minute demand, price and the market flags |
| `GET /v1/anomalies` | `start`, `end`, optional `tier`, `category`, `detector`, `limit` (≤ 500), `after` (cursor) | v2 events: times, tier, detectors, categories, lowest p, summary and split. No evidence |
| `GET /v1/anomalies/{event_id}` | none | One event with its full evidence |

- **Errors:** 422 for an invalid input (missing offset, off-grid time, end before start, range too long, unknown filter value); 404 for valid input with nothing published; 503 when a dependency is down (database, or model files for recompute/explain). User input never produces a 500. A failed E1 gate is a server fault, so it returns 500 with the gate's result.
- **CORS:** any origin, GET only.
- **Read-only:** the API has no write path.

### 2.5 Recompute and explain

- **At startup** (FastAPI lifespan): the feature context and weather inputs are loaded once (3.9 s when measured during planning). So are the stored fits of both splits: run 2's (`explain.dev_batch`) and `holdout_main`'s (`explain.holdout_batch`). Boosters load on first use and are cached (132 MB of model files). *Changed in step 3's question round: every booster loads at startup (step 3 below).*
- **Recompute** goes through M2's own code path (`LightGBMModel`, anchor + `booster.predict`), casts to float32 as M2 stored it, and compares with the published row.
- **Explain** reuses M4's `explainability.pairs.explain_pairs`: TreeSHAP and E1's gate (the forecast is float32-exact; base value + Σ contributions equals the raw prediction within `explain.sum_tolerance`). It groups contributions with M4's feature families.
- Building one issue time's 288 feature rows took 2.6 ms during planning.
- If the SSD is not mounted, `/health` says so, recompute and explain return 503, and the stored endpoints keep working.

### 2.6 Container

- **Image:** `python:3.12-slim` (linux/arm64 on this Mac), the locked dependencies installed with `uv sync --frozen --no-dev`, and `libgomp1` for LightGBM.
- **Compose service `api`:** waits for `db` to be healthy and publishes port 8000. It mounts `<data root>/models` read-only and has a health check on `/health`. *Changed in step 4: the host port is `GRIDCAST_API_PORT` (default 8000), published on 127.0.0.1 only.*
- **Make:** `make api` (local uvicorn, for development), `make api-up` / `make api-down` (Compose).

## 3. Success criteria (pre-registered)

Every criterion is judged on the **Compose service** (Linux, arm64), because that is what is served. The forecasts were computed on macOS, so P3 there is a real cross-platform train/serve check.

| Id | Criterion | Pass if |
|---|---|---|
| **P1** | Publication parity | Every published value equals its parquet source exactly, for every row of every product and split (read back in full, not sampled). Row counts equal issue times × 288 minus the counted missing pairs. The interval hit counts recomputed from Postgres equal M3's evaluation on the parquet, for every level, lead and split |
| **P2** | API parity | Seeded sample: 20 issue times per retrain fold (9 development + 5 holdout folds = 280). Every value served by `/forecasts/demand` and `/forecasts/price`, cast back to float32, equals the parquet. So does every value of `/forecasts/demand/evolution` for 280 seeded targets, and `/actuals` for 50 seeded days against `market_intervals`. Every v2 event served by `/anomalies` (paged through completely) and `/anomalies/{id}` equals its database row in every field |
| **P3** | Recompute parity | For the same 280 issue times: all 288 leads recomputed equal the stored float32 exactly (80,640 pairs, 100%). `/explain` returns an explanation, with no gate failure, for every one of them |
| **P4** | Nesting | lo95 ≤ lo80 ≤ lo50 ≤ hi50 ≤ hi80 ≤ hi95 in 100% of published interval pairs (SQL over every row, as well as the CHECK) |
| **P5** | Latency | Client-side p95 over a fixed seeded request mix, run sequentially from the host against the Compose service after 20 warm-up requests per endpoint: **≤ 100 ms** for each stored endpoint (forecast, evolution, price, actuals for one day, one month of anomalies with `limit` 100, one event) over 500 requests each; **≤ 500 ms** for recompute and explain (all leads) over 200 requests each |
| **P6** | Contract | The OpenAPI schema is committed (`docs/openapi.json`), and a test fails if the app's schema drifts from it. Every endpoint has a typed response model. A table of invalid inputs returns the documented 4xx with a message. A hypothesis test over query parameters finds no 500 |

### 3.1 Reporting rules

- `gridcast api-verify` measures P2, P3 and P5 against the running service and stores the results under `<data root>/serving/`. `gridcast report-m5` writes `reports/m5_api.md` from the stored results and the `publications` records. Both are committed before the first verification run.
- **Every verification run is kept.** The report shows the first run's verdicts. If a bug is fixed and the run repeated, the report shows each later run beside the change that preceded it. This is a software check, not a statistical holdout, so reruns are allowed, but they are disclosed.
- Reported only: the share of pairs whose point forecast lies outside the 50% interval (CQR is not centred on the point); the p50 and p99 latencies; the API container's memory use.
- The README's numbers come only from `reports/m5_api.md` and the earlier reports.

### 3.2 What the criteria do not cover

- Accuracy. Nothing is re-scored; the served values are the scored ones.
- Load. There is no concurrency test (the user's choice).
- Security beyond read-only access. There is no auth; the API is local.

### 3.3 Disclosures written now

- **The served intervals are not M3's pre-registered headline.** They are its reported-only *CQR, rearranged* row: the same bounds, sorted per pair. Rearrangement was added after M3's step-3 results were seen (m3_plan.md). Its holdout coverage at 80% was 79.6%, against 79.7% for the headline (`reports/m3_holdout.md`).
- **The development forecasts are run 2's.** Two development runs looked at the same period, so development numbers are slightly optimistic (README). The holdout forecasts are not affected.
- **Anomaly v2's development verdicts are post hoc.** The holdout year is its only blind test (`reports/m4_holdout.md`).
- **Price forecasts are baselines,** not a model.
- **This is an archive, not a live service.** Nothing after 1 Oct 2026 is served.
- **Seen during planning:** two sample rows in `forecasts` matched their parquet exactly; one-issue-time feature build time (2.6 ms) and context load time (3.9 s); the source row counts and missing pairs in §2.1. No API latency or parity result existed.

## 4. Steps (vertical slices; one branch, one commit per logical step)

0. **Plan** (this file, `m5_handoff.md`), and a top note in `m4_handoff.md` that M4 is merged.
1. **Publication.**
   - **1a (code):**
     - migration 0009 (`forecast_intervals`, `publications`);
     - `src/gridcast/serving/`: the source registry (§2.1), reshaping, comparison, COPY, the P1/P4 checks and the interval metrics;
     - `gridcast publish [--check-only]` (Make: `publish`, `publish-check`) and a `serving` config section (sources, period, sample sizes, latency limits; outside every frozen hash);
     - unit tests (reshape, missing pairs, a one-ulp difference is caught, rearrangement through the frozen function) and integration tests (tiny fixture parquet on the test database).
   - **1b (run):** check the Docker VM's disk, `make migrate`, `make publish`. Record P1 and P4 in the handoff.
   - **Outcome (2026-11-23):** 1a `fe217fc`. 1b: the VM disk had 51 GB free of 63 GB; migration 0009 applied; `make publish` took 5.3 min (estimated 15–30), and all 8 publications succeeded.
     - **P1 passes:** all 63,378,240 rows were read back and equal their parquet bit for bit. Before writing, the four 288-row samples (runs 39, 61, 11, 14) were found equal.
     - **P4 passes:** 0 of 15,708,288 interval pairs are not nested. The sort repaired 1,522,159 development and 535,453 holdout pairs (M3's non-nesting shares, 14.3% and 10.6%).
     - The interval metrics from Postgres equal those from the parquet. They are stored for runs 109 (development) and 110 (holdout), `cqr_lgbm_C_rearranged`. Pooled 80% coverage is 79.27% and 79.65%, matching the reports' 79.3% and 79.6%.
     - The database grew from 1.6 to 10 GB (`forecasts` 7.1 GB, `forecast_intervals` 2.0 GB). `make publish-check` re-checks everything in 0.9 min, with no writes.
2. **Stored endpoints:** `src/gridcast/api/` (app, settings, queries, response models), `/health`, `/v1/meta`, forecasts, evolution, price, actuals and anomalies; `docs/openapi.json` and its drift test; the invalid-input table and the hypothesis test; `make api`.
   - **Decided in step 2's question round (2026-11-23; the user chose every recommendation):**
     - **`/v1/anomalies` pages with a keyset cursor.** Events are ordered by (first interval end, `event_id`). The response carries an opaque `next` cursor encoding the last event's pair, and `after=<cursor>` continues from it.
     - **`/health`'s "publications complete":** for each of the 8 (product, split), the most recent `publications` row has status `success`. A later failed or running attempt makes `/health` report degraded.
     - **`/v1/meta` disclosures:** one sentence for each disclosure of §3.3, each with the report it comes from. The text is kept in code, and a test checks that all five are present.
   - **Outcome (2026-11-23):** done; see the handoff. Clarified while building:
     - bounds are served as the exact float64 of the stored float32 (psycopg returns `real` as its shortest decimal);
     - `/v1/actuals` returns the intervals lying inside [start, end];
     - `/v1/anomalies` returns an empty page, not 404, when no event matches;
     - the v2 anomaly runs are named in `serving.anomaly_runs` and checked at startup.
3. **Model endpoints:** recompute and explain (§2.5), with integration tests on a few issue times from each split on the host.
   - **Measured before the question round (2026-11-25, host, informal; not P3 or P5):**
     - The feature context and weather load in 4.3 s, and the process then holds about 0.86 GB. All 64 distinct boosters load in 0.7 s, which brings it to about 1.1 GB.
     - One issue time per split (dev 2024-03-15 02:00 UTC, holdout 2026-01-20 05:30 UTC) passed E1 on all 288 leads. Recompute and SHAP of every lead took about 0.9 s cold and 140–225 ms warm.
     - 2024-01-15 02:00 UTC has no stored forecast (every feature row is incomplete). January 2024 has 636 of its 1,488 issue times stored.
     - The 68 fits hold 64 distinct keys: `2025-08-01_<band>` is in both batches (run 2's last retrain and `holdout_main`'s first). The four file pairs are byte-identical, so M4's shared `ModelCache` (keyed by retrain and band) gave the same results. The API keys its own cache by model file, so it names the right batch's file. `explainability/` is frozen, so it is reused, not changed.
   - **Decided in step 3's question round (2026-11-25; the user chose every recommendation):**
     - **Gaps are reproduced.** M2 forecast only complete feature rows. `/recompute` answers with all 288 leads, `null` where the rebuilt row is incomplete, and `matches_stored` also requires those nulls to fall exactly where nothing was stored. `/explain` covers only stored leads; it answers 404 if the issue time, or the requested lead, has none.
     - **Feature values are served** beside their contributions (the float32 the model saw), and the anchor names its terms (e.g. the residual form: `dem_same_time_yesterday`).
     - **Boosters load at startup**, with the feature context (a change to §2.5, which loaded them on first use). `/health`'s model check means that every booster parsed and matched its record (trees, feature names). If the SSD is missing at startup, the next model request retries.
     - **Real-model tests** carry a new opt-in pytest marker, `host` (`make test-host`), skipped by default. They use the main database and the SSD: a few issue times per split, a gap, a fold boundary, and the `2025-08-01` fits that both batches hold. They are informal; P3 is judged in the container in step 5. Fixture tests on small synthetic walk-forward models run in the normal suites.
   - **Decided while planning the step (not asked):** model requests are serialized by one lock. They are CPU-bound, and LightGBM already uses `ml.lightgbm.num_threads` per call. Recompute serves the float32 M2 would have stored, as its exact float64, beside the stored value, and gives the largest float64 difference for information, as E1's gate does.
   - **Outcome (2026-11-26):** done; see the handoff. Clarified while building:
     - **Explain composes the frozen functions** that `explain_pairs` is made of (`locate`, `feature_rows`, `explain`, `gate`) instead of calling it, because `explain_pairs` does not return the feature rows the response now serves. `explainability/` is unchanged.
     - **The API's fits** come from the same frozen functions as `stored_fits` (`batch_config`, `fits_from_records`, `split_period`), without its check that the data root exists. So an unmounted drive fails only the boosters, not the feature context. A host test checks that both give equal fits.
     - **Errors:** recompute and explain answer 503 until both parts are loaded, with the reason and no host path. A server fault (a published issue time without a fit, or a stored value that is not float32) is a JSON 500. A failed gate is a 500 whose body carries the gate.
     - **The published run must be the fits' batch** (checked per request against the catalogue), and `serving.sources.demand_point` must name `anomalies.explain`'s batches (checked at load).
     - **`/health`** gains `feature_context` and `model_files`. When either fails the status is `degraded`, not 503, because the stored endpoints still work.
     - **The explain body is about 1.05 MB** for all 288 leads (one lead: about 4.7 KB). In the question round I estimated about 0.3 MB; the per-feature objects and the full-precision values make it larger.
4. **Container:** `Dockerfile`, the Compose service, `make api-up` / `api-down`. Then `/health` and one recompute per split against the container (a smoke check, not P3).
   - **Checked before the question round (2026-11-26):** a throwaway container on Docker Desktop read `<data root>/models` from the SSD read-only. The two published batches' model files take 304 MB (development) and 143 MB (holdout). Postgres used 0.7 GB of the VM's 3.8 GB.
   - **Decided in step 4's question round (2026-11-26; the user chose every recommendation):**
     - **Without the SSD, the API starts without models.** `make api-up` adds the read-only models mount (a Compose override file) only when `<data root>/models` exists. Otherwise it starts the API without the mount, so the stored endpoints work, `/health` reports `model_files`, and recompute and explain answer 503, as §2.5 says. The mount never creates its source (`create_host_path: false`), so Docker cannot create the path on the internal disk.
     - **The image records its code, and `/health` serves it.** `make api-up` passes the commit and a dirty flag as build arguments. The flag is set by any change, untracked files included, to the files the image copies. `/health` gains `build` (`git_sha`, `dirty`, and whether these come from the image or a checkout), and `docs/openapi.json` is regenerated. Step 5's `api-verify` can then refuse a dirty or stale image.
     - **The smoke check is a committed command:** `gridcast api-smoke` / `make api-smoke` (httpx). `/health` must be `ok`. Then recompute must report `matches_stored` and explain must pass its gate, at the two issue times measured before step 3's question round (development 2024-03-15 02:00 UTC, holdout 2026-01-20 05:30 UTC, also in the host tests). It reports client-side timings, and the Make target adds the containers' memory (`docker stats`). It is informal, not P3 or P5.
   - **Decided while planning the step (not asked):**
     - one uvicorn worker, because the model state takes about 1.1 GB per process and model requests are serialized anyway;
     - the port is published on 127.0.0.1 only (there is no auth);
     - the container runs as a non-root user, with the code installed editable at `/app`, so `PROJECT_ROOT` resolves as it does on the host;
     - uv is pinned to the host's version;
     - no `restart` policy and no memory limit (memory is measured instead).
   - **Outcome (2026-11-26):** code `97b8436`; see the handoff. Changed or found while building:
     - **The host port is configurable** (`GRIDCAST_API_PORT`, default 8000; the container always listens on 8000). Another app holds port 8000 on this Mac, so `.env` sets 8001 here. The same clash would have hit `make api`.
     - **Logging was never configured under uvicorn,** so `GRIDCAST_LOG_LEVEL` and `GRIDCAST_LOG_FORMAT` were ignored by `make api` too. uvicorn's factory is now `gridcast.api.app:serve`, which configures logging as the CLI does, then calls `create_app`. The container logs JSON.
     - The image is 458 MB (the plan estimated about 1 GB). It has the host's Python (3.12.16), numpy (2.5.4), LightGBM (4.7.0) and pandas (3.1.0), on Linux aarch64 (glibc 2.41).
     - **The smoke check passes** on the clean image of `97b8436`: `/health` is ok. At both issue times, recompute matches all 288 stored leads exactly, and explain passes E1's gate on all 288. These are the first values recomputed on Linux for forecasts made on macOS (informal; P3 is step 5's).
     - **Without the SSD** (simulated with `MODELS_DIR=/nonexistent/models`), `make api-up` warned and started the API without the mount. `/health` was `degraded` and named the missing file relative to the data root. Recompute answered 503 with the reason, the stored forecast answered 200, and `api-smoke` exited 1.
5. **Verification:**
   - **5a:** `gridcast api-verify` and `gridcast report-m5`, committed before any run.
   - **5b:** the run, then `reports/m5_api.md`.
   - **Decided in step 5's question round (2026-11-27; the user chose every recommendation):**
     - **The image must be the checkout, clean.** `api-verify` refuses, stores nothing, and exits 2 unless `/health`'s build comes from the image, its commit is the checkout's HEAD, the image is clean, and the checkout's tracked files are clean (the client that measures is the checkout's code). So one commit names the whole run, and `make api-up` rebuilds after any commit (about 15 s).
     - **P5 is run as pre-registered.** Nothing is tuned before the first run, and its verdicts lead the report. Two informal figures were seen before it:
       - step 4's smoke check: development explain about 340 ms, holdout about 200 ms, at one issue time each;
       - the question round's cost model (each fold's leaves weighted by the leads each band serves). It puts the heaviest served fold, 2024-08-01, at about 1.2× the smoke's development fold (2024-02-01): roughly 400–430 ms, an estimate, not a measurement.

       If P5 fails, a speed-up that changes no value is a disclosed change, and the rerun is shown beside the first run (§3.1).
     - **P5's timed requests are interleaved:** all 3,400 in one seeded random order, after the warm-ups.
   - **Decided while planning the step (not asked):**
     - **Samples.** Each draw has its own stream from `serving.verify.seed`.
       - P2/P3 issue times come from each retrain fold of the published fits, clipped to its split's published issue times (9 development + 5 holdout folds). From each fold, 20 issue times are drawn without replacement, among those with at least one stored lead, since explain is defined only there. P3's 80,640 pairs include the leads without a stored forecast, whose recomputation must also be null.
       - Price is checked at the same 280 issue times. One after the price baselines' last issue time (2026-09-29 14:00 UTC) must answer 404.
       - Evolution: 280 targets drawn uniformly on the 5-minute grid over (first published issue time, last + 24 h]. The served issue times must be exactly the published ones in [*T* − 24 h, *T*).
       - Actuals: 50 NEM days drawn among the days stored in full. Each request runs from that day's NEM midnight to the next (288 intervals) and is compared field by field with `market_intervals`, read by `api-verify`'s own SQL.
       - Anomalies: every v2 event, paged through with the default limit (100), and every event by id, against its database row (every field, evidence included).
     - **"Equal"** means that a served number, cast to float32, has the parquet's bits, and is itself exactly that float32 (§1). A served null must fall exactly where the parquet has NaN. Each response's split, run ids, leads and target times are checked too.
     - **P3 checks the bodies, not only their flags:**
       - recomputed and stored values against the parquet;
       - explain's leads without a forecast, and each explained lead's forecast;
       - per lead, anchor + base + Σ contributions against the float64 forecast within `explain.sum_tolerance`, as well as the gate's `passed`.
     - **P5:**
       - The time runs client-side, from sending the request to the last byte of its body (httpx), one request at a time on one keep-alive connection.
       - p95 is numpy's default (linear) percentile. A timed response that is not 200 fails P5 for its endpoint.
       - Its parameters are drawn separately from P2's. Forecast and price issue times are uniform over every published one; recompute and explain over those with a stored lead. Evolution targets use P2's domain. Actuals take one NEM day, anomalies one NEM calendar month (limit 100), and the event endpoint one v2 event, uniformly.
     - **P6 is also judged against the container,** since §3 judges every criterion there; the suites' tests stay. `api-verify` checks:
       - that the served `/openapi.json` equals `docs/openapi.json`, and every path's 200 response has a typed schema;
       - the invalid-input table, which moves from the unit tests to `gridcast.api.contract` (both use it): 422 with its message;
       - a 404 table for the main data: before the published range, region NSW1, a v1 event, a day without actuals, explain where nothing is stored;
       - the integration suite's hypothesis search over query parameters (derandomised, 300 examples): no 500.
     - **Every check runs to the end.** The results record counts and up to 20 mismatches per check. The order is: the build check and `/health`, then P6, P2, P3 and P5.
     - **Results** go to `<data root>/serving/verify_<UTC time>.json`, never overwritten. The file records:
       - the build, the Docker VM (`docker info`: CPUs, memory), the samples and every latency;
       - reported only: the containers' memory after the run (`docker stats`), and the share of pairs whose point lies outside the 50% interval (SQL over every published pair, by split).
     - **`gridcast report-m5`** reads every stored verification run and the publications (P1, P4). The first run's verdicts lead. Each later run is shown with the commits between it and the run before. Tables only, no figures.
   - **5a outcome (2026-11-27):** committed before any verification run; see the handoff. Clarified while building:
     - **The build must come from the image** (`build.source` is `image`), so `make api` on the host cannot be verified. `/health` is read again at the end, and the run fails if its build changed during the run. A refused run exits 2 and stores nothing.
     - **The 404 table**, built from the published data:
       - demand before and after the published range, region NSW1, a target before every issue time, price after its last issue time;
       - recompute and explain before the range, explain where nothing is stored, and a lead without a stored forecast (one of each per split where one exists);
       - a day before the first actual, and a v1 event.
     - **The search accepts only 200, 404 and 422** against the container, so a 503 also counts as a failure there.
     - **Explain's check** also requires each lead's family contributions to sum to its feature contributions within the tolerance.
     - The source matrices come from publish's reader, so the truth's interval bounds are the frozen rearrangement of the parquet bounds, as P1's are.
   - **5b outcome (2026-11-27):** **all six criteria pass on the first verification run** (`reports/m5_api.md`). The run took 3.7 min against the clean image of `9e70b42` (`<data root>/serving/verify_20261126_140655.json`).
     - **P2:** 0 mismatches over 564,480 demand values, 161,280 price values, 13,404 evolution items, 14,400 actual intervals, and 1,127 v2 events (paged, and by id).
     - **P3:** 80,640 of 80,640 pairs were recomputed equal on Linux, and 280 of 280 issue times were explained with the gate passed (80,382 leads).
     - **P5:** the stored endpoints' p95 is 2.8–23.5 ms, recompute's 52.8 ms, and **explain's 427.0 ms against the 500 ms limit**. Reported only: explain's p99 is 537.8 ms and its maximum 605.9 ms.
     - **P6:** the served schema equals the committed one, with 10 typed paths. 19 invalid inputs answer 422 and 11 unpublished requests answer 404. The search's 300 examples found no 500.
     - **Reported only:** the point lies outside the 50% interval in 19.2% (development) and 29.3% (holdout) of pairs, mostly below it. The API held 795 MiB after the run.
     - **After the run,** three wording fixes in `report-m5` (`7b0176f`) changed no number or verdict. A zero count of late price issue times now reads as such, a failed smoke check names its failing check, and the reported-only lines are a list.
6. **README** (an API section from the report), and the outcome in this plan and the handoff.

## 5. New dependencies and compute

- **Runtime:** `fastapi` and `uvicorn`. httpx (for tests and `api-verify`) is already a dependency.
- **Compute:** publish takes about 15–30 min. The API starts in a few seconds. Verification takes minutes.
- **Disk:** about +8–9 GB in the Docker volume and a Docker image of about 1 GB.

## 6. Risks

- **The Docker VM's disk.** The volume grows by about 8–9 GB. Docker Desktop's disk image has a limit, checked in step 1b; raising it is a Docker Desktop setting (the user's action).
- **The Docker VM's memory** (4 GB now) holds both Postgres and the API (feature context and boosters). If it isn't enough, the fix is a Docker Desktop setting (the user's action).
- **Exactness across platforms.** The stored forecasts came from macOS arm64; the container runs Linux arm64 with different numpy and LightGBM builds. If P3 fails in the container but passes on the host, that is a real train/serve skew. It gets reported as such, and any fix is disclosed.
- **Load time.** Writing ~63.4M rows with the primary keys in place may take longer than estimated. publish reports its timings; staging and batch size can change without touching any value.
- **The SSD must be mounted** for publish and for recompute/explain. The stored endpoints don't need it.

## 7. Out of scope for M5

- Live or near-real-time ingestion, and forecasts issued after the holdout (M7).
- A price model, price intervals, and C⁺.
- Authentication, rate limiting and public hosting (decided in M6).
- The dashboard (M6) and monitoring (M7).
- Regions other than Victoria.
- Serving anomaly v1.

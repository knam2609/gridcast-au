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

- **At startup** (FastAPI lifespan): the feature context and weather inputs are loaded once (3.9 s when measured during planning). So are the stored fits of both splits: run 2's (`explain.dev_batch`) and `holdout_main`'s (`explain.holdout_batch`). Boosters load on first use and are cached (132 MB of model files).
- **Recompute** goes through M2's own code path (`LightGBMModel`, anchor + `booster.predict`), casts to float32 as M2 stored it, and compares with the published row.
- **Explain** reuses M4's `explainability.pairs.explain_pairs`: TreeSHAP and E1's gate (the forecast is float32-exact; base value + Σ contributions equals the raw prediction within `explain.sum_tolerance`). It groups contributions with M4's feature families.
- Building one issue time's 288 feature rows took 2.6 ms during planning.
- If the SSD is not mounted, `/health` says so, recompute and explain return 503, and the stored endpoints keep working.

### 2.6 Container

- **Image:** `python:3.12-slim` (linux/arm64 on this Mac), the locked dependencies installed with `uv sync --frozen --no-dev`, and `libgomp1` for LightGBM.
- **Compose service `api`:** waits for `db` to be healthy and publishes port 8000. It mounts `<data root>/models` read-only and has a health check on `/health`.
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
2. **Stored endpoints:** `src/gridcast/api/` (app, settings, queries, response models), `/health`, `/v1/meta`, forecasts, evolution, price, actuals and anomalies; `docs/openapi.json` and its drift test; the invalid-input table and the hypothesis test; `make api`.
3. **Model endpoints:** recompute and explain (§2.5), with integration tests on a few issue times from each split on the host.
4. **Container:** `Dockerfile`, the Compose service, `make api-up` / `api-down`. Then `/health` and one recompute per split against the container (a smoke check, not P3).
5. **Verification:**
   - **5a:** `gridcast api-verify` and `gridcast report-m5`, committed before any run.
   - **5b:** the run, then `reports/m5_api.md`.
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

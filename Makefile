.DEFAULT_GOAL := help
UV ?= uv
START ?= 2021-10
END ?= 2026-09
REGIONS ?= VIC1
# M4 anomaly recipe version (1 or 2)
VERSION ?= 1
DAYS ?= 2026-10-01
CHECK_MONTH ?= 2026-09

.PHONY: help setup lint format typecheck test test-integration db-up db-down migrate ingest ingest-daily ingest-aemo ingest-aemo-poe ingest-generation weather calendar dq backtest backtest-ml evaluate-holdout weather-ablation report report-m2 report-aemo report-holdout intervals-conformal backtest-qr intervals-cqr intervals-halfhour intervals-freeze-check backtest-qr-holdout evaluate-holdout-intervals report-m3 report-m3-aemo report-m3-holdout anomaly-reference-events anomalies-smoke explain-smoke evidence-smoke anomalies-freeze-check anomalies-dev report-m4 anomalies-diagnose-v1 anomalies-reproduce explain-dev explain-holdout report-m4-explain catalogue feature-audit pipeline

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Install Python deps (uv) and create .env from the example if missing
	$(UV) sync
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example; set a password.")

lint: ## Ruff lint + format check
	$(UV) run ruff check src tests sql/migrations
	$(UV) run ruff format --check src tests sql/migrations

format: ## Auto-format and fix lint
	$(UV) run ruff format src tests sql/migrations
	$(UV) run ruff check --fix src tests sql/migrations

typecheck: ## mypy on src/
	$(UV) run mypy

test: ## Unit tests (no database required)
	$(UV) run pytest

test-integration: ## Integration tests against the docker-compose database
	$(UV) run pytest -m integration

db-up: ## Start PostgreSQL and wait until healthy
	docker compose up -d --wait db

db-down: ## Stop PostgreSQL (data volume is kept)
	docker compose down

migrate: ## Apply database migrations
	$(UV) run alembic upgrade head

ingest: ## Download, validate and load AEMO MMSDM data (START/END/REGIONS overridable)
	$(UV) run gridcast ingest --start $(START) --end $(END) --regions $(REGIONS)

ingest-daily: ## Load DAYS from NEMWeb's daily dispatch reports after an exact match with MMSDM over CHECK_MONTH (M2 step 6)
	$(UV) run gridcast ingest-daily $(foreach d,$(DAYS),--day $(d)) --check-month $(CHECK_MONTH) --regions $(REGIONS)

ingest-aemo: ## Download and load AEMO pre-dispatch + 5-min pre-dispatch forecasts (M2 benchmark; ~2.6 GB cached)
	$(UV) run gridcast ingest-aemo --regions $(REGIONS)

ingest-aemo-poe: ## Download and load AEMO's PDPASA POE demand forecasts (M3 benchmark; LOR runs; ~1.6 GB cached)
	$(UV) run gridcast ingest-aemo-poe --regions $(REGIONS)

ingest-generation: ## M4 step 1b: load wind, solar, dispatchable and battery totals from the cached DISPATCHREGIONSUM months, then DAYS from the daily reports after an exact match over CHECK_MONTH
	$(UV) run gridcast ingest-generation --start $(START) --end $(END) $(foreach d,$(DAYS),--day $(d)) --check-month $(CHECK_MONTH) --regions $(REGIONS)

weather: ## Download, validate and load Open-Meteo weather (GFS forecasts + ERA5; START/END overridable)
	$(UV) run gridcast ingest-weather --start $(START) --end $(END)

calendar: ## Build calendar_days (VIC holidays, weekends, DST-change days)
	$(UV) run gridcast calendar --regions $(REGIONS)

dq: ## Run data-quality checks and store results
	$(UV) run gridcast dq --regions $(REGIONS)

backtest: ## Walk-forward evaluation of all baselines (demand + price)
	$(UV) run gridcast backtest --target demand --region $(firstword $(REGIONS))
	$(UV) run gridcast backtest --target price --region $(firstword $(REGIONS))

backtest-ml: ## M2 walk-forward: LightGBM + ridge (A/C/D/C⁺/D⁺), per-retrain selection; several hours
	$(UV) run gridcast backtest-ml --region $(firstword $(REGIONS))

evaluate-holdout: ## M2 holdout, scored ONCE (S5): refuses unless frozen config/code, clean tree, not yet scored; took 10 h
	$(UV) run gridcast evaluate-holdout --region $(firstword $(REGIONS))

weather-ablation: ## M2 weather-feature drop/add test on the stored ML run (~30-40 min)
	$(UV) run gridcast weather-ablation --region $(firstword $(REGIONS))

report: ## Generate reports/m1_baseline.md from stored metrics
	$(UV) run gridcast report --region $(firstword $(REGIONS))

report-m2: ## Generate reports/m2_demand.md from the stored ML run (S1-S4, runs the leakage tests)
	$(UV) run gridcast report-m2 --region $(firstword $(REGIONS))

report-aemo: ## Generate reports/m2_aemo.md: frozen model vs AEMO pre-dispatch (development part)
	$(UV) run gridcast report-aemo --region $(firstword $(REGIONS))

intervals-conformal: ## M3 step 1: conformal intervals around run 2's dev forecasts (C + baselines); minutes
	$(UV) run gridcast intervals-conformal --region $(firstword $(REGIONS))

backtest-qr: ## M3 step 2: quantile LightGBM C walk-forward (dev, run 2's settings); hours, clean tree only
	$(UV) run gridcast backtest-qr --region $(firstword $(REGIONS))

intervals-cqr: ## M3 step 3: CQR on the stored QR batch (dev, intervals.qr_batch); minutes, clean tree only
	$(UV) run gridcast intervals-cqr --region $(firstword $(REGIONS))

intervals-halfhour: ## M3 step 5: conformal and CQR intervals for 30-minute means (dev, for the AEMO comparison); clean tree only
	$(UV) run gridcast intervals-halfhour --region $(firstword $(REGIONS))

intervals-freeze-check: ## M3 step 5: check the interval recipe against its freeze; recomputes the dev bounds (~8 min)
	$(UV) run gridcast intervals-freeze-check --region $(firstword $(REGIONS))

backtest-qr-holdout: ## M3 step 6a: quantile LightGBM C over the holdout, UNSCORED (reproduces step 2's first retrain first); hours, clean tree only
	$(UV) run gridcast backtest-qr-holdout --region $(firstword $(REGIONS))

evaluate-holdout-intervals: ## M3 step 6b: holdout intervals scored ONCE (U5): refuses unless frozen recipe/code, clean tree, not yet scored
	$(UV) run gridcast evaluate-holdout-intervals --region $(firstword $(REGIONS))

report-m3: ## Generate reports/m3_intervals.md from the stored interval batches (U1-U4, runs the leakage tests)
	$(UV) run gridcast report-m3 --region $(firstword $(REGIONS))

report-m3-aemo: ## Generate reports/m3_aemo.md: frozen half-hour intervals vs AEMO's POE band (development part)
	$(UV) run gridcast report-m3-aemo --region $(firstword $(REGIONS))

report-m3-holdout: ## Generate reports/m3_holdout.md from the stored holdout interval run (U5, replication, winter, AEMO holdout part)
	$(UV) run gridcast report-m3-holdout --region $(firstword $(REGIONS))

report-holdout: ## Generate reports/m2_holdout.md from the stored holdout run (S5, replication, AEMO holdout part)
	$(UV) run gridcast report-holdout --region $(firstword $(REGIONS))

anomaly-reference-events: ## M4 step 1a: write docs/m4_reference_events.md (AEMO-flagged events for A2) from the stored flags
	$(UV) run gridcast anomaly-reference-events --region $(firstword $(REGIONS))

anomalies-smoke: ## M4 step 2: detectors, events and injections on the warm-up periods only (bugs, no writes; minutes)
	$(UV) run gridcast anomalies-smoke --region $(firstword $(REGIONS)) --version $(VERSION)

evidence-smoke: ## M4 step 3b: evidence and summaries of the warm-up smoke events (completeness, banned phrases, E1, 5 examples; no writes)
	$(UV) run gridcast evidence-smoke --region $(firstword $(REGIONS)) --version $(VERSION)

explain-smoke: ## M4 step 3: SHAP of run 2's retrain 2022-11-01 (E1 results and timings only, no writes)
	$(UV) run gridcast explain-smoke --region $(firstword $(REGIONS))

anomalies-freeze-check: ## M4 step 4: the anomaly recipe against its freeze record (exit 1 on any difference)
	$(UV) run gridcast anomalies-freeze-check --version $(VERSION)

anomalies-dev: ## M4 step 4: the one development run of the frozen anomaly recipe (clean tree; stores the batch)
	$(UV) run gridcast anomalies-dev --region $(firstword $(REGIONS)) --version $(VERSION)

report-m4: ## M4 step 4: reports/m4_anomalies.md from the stored development run (runs the causality suite)
	$(UV) run gridcast report-m4 --region $(firstword $(REGIONS)) --version $(VERSION)

explain-dev: ## M4 step 5: global SHAP sample of run 2's LightGBM C fits with E1 on every pair (REPLACE=1 to redo)
	$(UV) run gridcast explain-dev --region $(firstword $(REGIONS)) $(if $(REPLACE),--replace,)

explain-holdout: ## M4 step 6: global SHAP sample of holdout_main's fits on holdout issue times, E1 on every pair (REPLACE=1 to redo)
	$(UV) run gridcast explain-holdout --region $(firstword $(REGIONS)) $(if $(REPLACE),--replace,)

report-m4-explain: ## M4 step 5: reports/m4_explainability.md (global SHAP, question 5, price associations; development)
	$(UV) run gridcast report-m4-explain --region $(firstword $(REGIONS))

anomalies-diagnose-v1: ## M4 after step 4: docs/m4_v1_diagnosis.md, why v1's D1, D4 and D6 failed (reported only)
	$(UV) run gridcast anomalies-diagnose-v1 --region $(firstword $(REGIONS))

anomalies-reproduce: ## M4 step 4f: recompute the stored development run with the current code; exit 1 on any difference
	$(UV) run gridcast anomalies-reproduce --region $(firstword $(REGIONS)) --version $(VERSION)

catalogue: ## Regenerate docs/feature_catalogue.md from the feature catalogue in code
	$(UV) run gridcast feature-catalogue

feature-audit: ## Generate reports/m2_feature_audit.md (feature coverage on the stored data)
	$(UV) run gridcast feature-audit --region $(firstword $(REGIONS))

pipeline: db-up migrate ingest ingest-generation weather calendar dq backtest report ## Full pipeline end to end

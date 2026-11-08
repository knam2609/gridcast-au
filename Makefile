.DEFAULT_GOAL := help
UV ?= uv
START ?= 2021-10
END ?= 2026-09
REGIONS ?= VIC1
DAYS ?= 2026-10-01
CHECK_MONTH ?= 2026-09

.PHONY: help setup lint format typecheck test test-integration db-up db-down migrate ingest ingest-daily ingest-aemo weather calendar dq backtest backtest-ml evaluate-holdout weather-ablation report report-m2 report-aemo report-holdout catalogue feature-audit pipeline

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

evaluate-holdout: ## M2 holdout, scored ONCE (S5): refuses unless frozen config/code, clean tree, not yet scored; ~12-14 h
	$(UV) run gridcast evaluate-holdout --region $(firstword $(REGIONS))

weather-ablation: ## M2 weather-feature drop/add test on the stored ML run (~30-40 min)
	$(UV) run gridcast weather-ablation --region $(firstword $(REGIONS))

report: ## Generate reports/m1_baseline.md from stored metrics
	$(UV) run gridcast report --region $(firstword $(REGIONS))

report-m2: ## Generate reports/m2_demand.md from the stored ML run (S1-S4, runs the leakage tests)
	$(UV) run gridcast report-m2 --region $(firstword $(REGIONS))

report-aemo: ## Generate reports/m2_aemo.md: frozen model vs AEMO pre-dispatch (development part)
	$(UV) run gridcast report-aemo --region $(firstword $(REGIONS))

report-holdout: ## Generate reports/m2_holdout.md from the stored holdout run (S5, replication, AEMO holdout part)
	$(UV) run gridcast report-holdout --region $(firstword $(REGIONS))

catalogue: ## Regenerate docs/feature_catalogue.md from the feature catalogue in code
	$(UV) run gridcast feature-catalogue

feature-audit: ## Generate reports/m2_feature_audit.md (feature coverage on the stored data)
	$(UV) run gridcast feature-audit --region $(firstword $(REGIONS))

pipeline: db-up migrate ingest weather calendar dq backtest report ## Full pipeline end to end

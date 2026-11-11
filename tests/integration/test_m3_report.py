"""M3 step 4 end to end on the test database: run 2 → conformal, QR, CQR batches → the report."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, load_config
from gridcast.database.load import ensure_regions, start_ingestion_run, upsert_market_intervals
from gridcast.evaluation.m2_report import LeakageOutcome
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.features.inputs import load_context
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import ingest_weather_range
from gridcast.intervals.batch import (
    EVALUATION_SET,
    bounds_metrics,
    insert_run,
    interval_config_hash,
    per_lead_rows,
    run_intervals_conformal,
)
from gridcast.intervals.cqr_batch import run_intervals_cqr
from gridcast.intervals.inputs import read_stored_batch
from gridcast.intervals.qr_batch import RUN_NAME as QR_RUN_NAME
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.quantile import interval_quantiles, quantile_key, quantile_levels
from gridcast.intervals.report import build_report
from gridcast.intervals.scores import pinball
from tests.integration.factories import clean_month
from tests.integration.test_intervals_batch import END, FIRST, _cfg
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

pytestmark = pytest.mark.integration

POINTS = {"lgbm_C": (20, 80), "persistence": (0, 300), "daily_naive": (0, 250),
          "weekly_naive": (0, 280)}  # fmt: skip
MONTHS = ((2021, 11), (2021, 12), (2022, 1))
PASSED = LeakageOutcome(3, 0, 0)


@pytest.fixture
def world(db: Engine, tmp_path: Path) -> Engine:
    """Clean market data, calendar days and fake weather for Nov 2021 → Jan 2022."""
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        for year, month in MONTHS:
            run = start_ingestion_run(conn, "mmsdm", "market_intervals", dt.date(year, month, 1))
            upsert_market_intervals(conn, clean_month(year, month), run)
            conn.execute(
                text("UPDATE ingestion_runs SET status='success' WHERE run_id=:r"), {"r": run}
            )
        upsert_calendar_days(
            conn,
            build_calendar_days(
                "VIC", "Australia/Melbourne", dt.date(2021, 10, 1), dt.date(2022, 2, 28)
            ),
        )
    client = OpenMeteoClient(
        om_config(), tmp_path / "om", transport=FakeOpenMeteo().transport(), today=lambda: TODAY
    )
    results = ingest_weather_range(
        db, cfg, dt.date(2021, 10, 1), dt.date(2022, 1, 1), client=client
    )
    assert {r.status for r in results} == {"success"}
    return db


def _write_source(db: Engine, cfg: AppConfig) -> str:
    """A fake run 2: the market data's demand as actuals, and four noisy point forecasts."""
    ctx = load_context(db, cfg, "VIC1")
    times = pd.date_range(nem_midnight_utc(str(FIRST)), nem_midnight_utc(str(END)), freq="30min")
    times = times[:-1]
    h = cfg.forecast.horizon_steps
    tpos = ctx.times.get_indexer(times)[:, None] + np.arange(1, h + 1)[None, :]
    actual = ctx.y[tpos]
    rng = np.random.default_rng(0)
    values = {"actual": actual}
    for name, (bias, sd) in POINTS.items():
        values[name] = actual + rng.normal(bias, sd, actual.shape)
    values["lgbm_C"][40:44] = np.nan  # pairs without C weather leave the common mask
    batch = str(uuid.uuid4())
    directory = cfg.data_path(cfg.paths.backtest_dir)
    directory.mkdir(parents=True)
    for name, matrix in values.items():
        frame = pd.DataFrame(
            matrix.astype(np.float32), columns=[f"h{k:03d}" for k in range(1, h + 1)]
        )
        frame.insert(0, "origin_time", times)
        frame.to_parquet(directory / f"demand__{name}__{batch}.parquet", index=False)
    with db.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id,
                                        config, status)
                VALUES (CAST(:b AS uuid), 'lgbm_C', 'ml', 'demand', 'VIC1',
                        CAST(:c AS jsonb), 'success')
                """
            ),
            {"b": batch, "c": json.dumps({"evaluation_set": "main", "complete_window": True})},
        )
    return batch


def _store_qr(db: Engine, cfg: AppConfig) -> str:
    """A fake QR run with its stored metrics: too-narrow quantiles around lgbm_C."""
    source = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch)
    point = source.forecasts["lgbm_C"]
    spread = np.random.default_rng(1).uniform(0.6, 1.0, point.shape)
    batch = uuid.uuid4()
    directory = cfg.data_path(cfg.intervals.output_dir) / str(batch)
    directory.mkdir(parents=True)

    def write(name: str, values: np.ndarray) -> None:
        frame = pd.DataFrame(values, columns=[f"h{k:03d}" for k in range(1, values.shape[1] + 1)])
        frame.insert(0, "origin_time", source.origin_times)
        frame.to_parquet(directory / name, index=False)

    write("actual.parquet", source.actual.astype(np.float32))
    q = {}
    for tau in quantile_levels(cfg.intervals.levels):
        q[quantile_key(tau)] = (point + 60 * (tau - 0.5) * 2 * spread).astype(np.float32)
        write(f"{QR_RUN_NAME}__{quantile_key(tau)}.parquet", q[quantile_key(tau)])

    rows = np.asarray(source.origin_times >= nem_midnight_utc(str(cfg.intervals.first_issue)))
    actual, mask = source.actual[rows], source.common_mask()[rows]
    lower, upper = {}, {}
    for level in cfg.intervals.levels:
        lo, hi = interval_quantiles(level)
        lower[level], upper[level] = q[lo][rows], q[hi][rows]
    metrics = bounds_metrics(lower, upper, actual, mask, point[rows])
    extra: list[tuple[str, int, float, int]] = []
    with np.errstate(invalid="ignore"):
        for key, values in q.items():
            tau = int(key[1:]) / 1000
            v = values[rows].astype(np.float64)
            extra += per_lead_rows(f"pinball_{key}", pinball(v, actual, tau), mask)
            extra += per_lead_rows(f"share_le_{key}", (actual <= v).astype(float), mask)
    extra += per_lead_rows("crossing", np.zeros(actual.shape), mask)
    metrics = pd.concat([metrics, pd.DataFrame(extra, columns=metrics.columns)])
    config = {
        "evaluation_set": EVALUATION_SET,
        "complete_window": True,
        "point_reproduction": {"passed": True},
        "source_batch": cfg.intervals.source_batch,
        "qr_config_hash": qr_config_hash(cfg),
        "interval_config_hash": interval_config_hash(cfg),
        "crossing_share_all_pairs": 0.0,
        "fits": [{"band": "h001-006", "crossing_share": 0.0}],
    }
    times = source.origin_times[rows]
    with db.begin() as conn:
        insert_run(
            conn, batch, QR_RUN_NAME, "VIC1", (times[0], times[-1]),
            nem_midnight_utc(str(cfg.backtest.holdout_start)), config, "test",
            directory.relative_to(cfg.data_root()), metrics,
        )  # fmt: skip
    return str(batch)


@pytest.fixture
def stored(world: Engine, tmp_path: Path) -> AppConfig:
    """Every batch the report reads, built by the real commands where they exist."""
    (tmp_path / "data").mkdir()
    cfg = _cfg(tmp_path / "data", "")
    cfg = _cfg(tmp_path / "data", _write_source(world, cfg))
    cfg.intervals = cfg.intervals.model_copy(update={"point_models": list(POINTS)})
    cfg.paths.reports_dir = tmp_path / "reports"  # never the repository's reports/
    run_intervals_conformal(world, cfg, "VIC1", workers=2)
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": _store_qr(world, cfg)})
    run_intervals_cqr(world, cfg, "VIC1", workers=2, sha="test")
    return cfg


def test_report_scores_every_stored_batch(world: Engine, stored: AppConfig) -> None:
    report = build_report(world, stored, "VIC1", PASSED)
    for key in ("U1", "U2", "U3", "U4"):
        assert f"| {key} |" in report
    assert "pre-registered development criteria pass" in report
    assert "(all equal)" in report  # every pooled number matched the stored metrics
    assert "CQR, rearranged (reported only)" in report
    assert "7-day blocks" in report and "3 passed, 0 failed" in report
    figures = stored.resolve(stored.paths.reports_dir) / "figures"
    for name in ("coverage_by_lead", "is80_vs_baselines", "reliability", "coverage_by_month"):
        assert (figures / f"m3_{name}.png").stat().st_size > 0
    # The rearranged CQR bounds are nested by construction.
    nesting = report.split("## Nesting of the levels")[1].split("\n## ")[0]
    row = next(r for r in nesting.splitlines() if r.startswith("| CQR, rearranged"))
    assert row.split("|")[2].strip() == "0.00"


def test_report_stops_when_a_stored_metric_differs(world: Engine, stored: AppConfig) -> None:
    with world.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE forecast_metrics SET value = value + 0.01
                WHERE metric = 'coverage_80' AND horizon = 0 AND run_id = (
                    SELECT run_id FROM model_runs WHERE model_name = 'conformal_persistence')
                """
            )
        )
    with pytest.raises(ValueError, match="recomputed coverage_80"):
        build_report(world, stored, "VIC1", PASSED)


def test_report_refuses_runs_of_another_recipe(world: Engine, stored: AppConfig) -> None:
    stored.intervals = stored.intervals.model_copy(update={"selection_weeks": 2})
    with pytest.raises(ValueError, match="interval config hash"):
        build_report(world, stored, "VIC1", PASSED)

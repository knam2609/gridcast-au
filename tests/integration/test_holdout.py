"""`evaluate-holdout` end to end on synthetic data: refusals, exact reproduction, S5 lock."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, GridConfig, LightGBMConfig, RidgeConfig, load_config
from gridcast.database.load import (
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_aemo_forecasts,
    upsert_market_intervals,
)
from gridcast.database.queries import load_market_frame
from gridcast.evaluation.aemo_benchmark import MATCHED, compare, load_benchmark
from gridcast.evaluation.holdout import (
    GitState,
    HoldoutRefused,
    ReproductionFailed,
    preflight,
    run_holdout,
    store_holdout,
)
from gridcast.evaluation.holdout_report import build_report, load_holdout_set
from gridcast.evaluation.m2_report import LeakageOutcome
from gridcast.evaluation.m2_results import artifact_path, load_evaluation_set
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.forecasting.ml_backtest import ml_config_hash, run_ml_backtest, store_sets
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import ingest_weather_range
from tests.integration.factories import clean_month
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

pytestmark = pytest.mark.integration

OCT, NOV = dt.date(2021, 10, 1), dt.date(2021, 11, 1)
GIT = GitState("abc1234", False, ())
OFFSET = 100.0  # synthetic AEMO forecasts: the actual value + 100 MW


@pytest.fixture
def two_months(db: Engine, tmp_path: Path) -> Engine:
    """October and November 2021: market data, weather (both datasets), calendar."""
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        for month in (OCT, NOV):
            run = start_ingestion_run(conn, "mmsdm", "market_intervals", month)
            upsert_market_intervals(conn, clean_month(2021, month.month), run)
            finish_ingestion_run(conn, run, status="success")
        upsert_calendar_days(
            conn,
            build_calendar_days(
                "VIC", "Australia/Melbourne", dt.date(2021, 9, 1), dt.date(2021, 12, 31)
            ),
        )
    client = OpenMeteoClient(
        om_config(), tmp_path / "om", transport=FakeOpenMeteo().transport(), today=lambda: TODAY
    )
    results = ingest_weather_range(db, cfg, OCT, NOV, client=client)
    assert {r.status for r in results} == {"success"}
    return db


def _cfg(data_root: Path) -> AppConfig:
    """Monthly retrains on the 20th; development 20 → 29 Oct; holdout 29 Oct → 25 Nov."""
    cfg = load_config()
    cfg.paths.data_root = data_root
    cfg.backtest.holdout_start = dt.date(2021, 10, 29)
    cfg.backtest.holdout_end = dt.date(2021, 11, 25)
    cfg.ml = cfg.ml.model_copy(
        update={
            "first_issue": dt.date(2021, 10, 20),
            "plus_first_issue": dt.date(2021, 10, 20),
            "plus_train_start": dt.date(2021, 10, 9),
            "retrain_every_months": 1,
            "early_stopping_weeks": 1,
            "lightgbm": LightGBMConfig(
                learning_rate=0.3,
                min_data_in_leaf=50,
                max_rounds=20,
                early_stopping_rounds=5,
                num_threads=2,
            ),
            "ridge": RidgeConfig(max_rows=20_000, spline_knots=4),
            "grid": GridConfig(
                target_forms=["residual", "direct"],
                lightgbm_objective=["l2"],
                lightgbm_num_leaves=[7],
                ridge_alpha=[1.0],
            ),
        }
    )
    return cfg


def _development_run(engine: Engine, cfg: AppConfig, plus: bool = True) -> AppConfig:
    """Store a development run and freeze its config, as M2 did before the holdout."""
    sets = run_ml_backtest(
        engine, cfg, "VIC1", plus=plus, delay=plus, model_root=cfg.data_path(cfg.ml.model_dir)
    )
    store_sets(engine, cfg, "VIC1", sets)
    cfg.backtest.frozen_ml_config_hash = ml_config_hash(cfg)
    cfg.backtest.frozen_ml_commit = "frozen0"
    return cfg


def test_holdout_reproduces_scores_once_and_locks(two_months: Engine, tmp_path: Path) -> None:
    cfg = _development_run(two_months, _cfg(tmp_path))
    refs = preflight(two_months, cfg, "VIC1", GIT)
    assert set(refs) == {"main", "plus", "delay"}

    run = run_holdout(two_months, cfg, "VIC1", GIT, cfg.data_path(cfg.ml.model_dir), refs)
    assert [r.evaluation_set for r in run.reproductions] == ["main", "plus", "delay"]
    assert all(r.passed and r.retrain == "2021-10-20" for r in run.reproductions)
    main_rep = run.reproductions[0]
    assert main_rep.issue_times == 9 * 48  # 20 → 28 Oct
    assert main_rep.fits == 6 * len(cfg.ml.lead_bands)
    assert [r.retrain for r in run.reproductions[1:]] == ["2021-10-20", "2021-10-20"]

    batches = store_holdout(two_months, cfg, "VIC1", run, GIT)
    with two_months.connect() as conn:
        runs = (
            conn.execute(
                text(
                    """
                    SELECT batch_id::text AS batch, model_name, git_sha, eval_start, eval_end,
                           config->>'evaluation_set' AS es, config
                    FROM model_runs WHERE config->>'period' = 'holdout' ORDER BY run_id
                    """
                )
            )
            .mappings()
            .all()
        )
        splits = (
            conn.execute(
                text(
                    """
                    SELECT DISTINCT m.split FROM forecast_metrics m JOIN model_runs r USING (run_id)
                    WHERE r.config->>'period' = 'holdout'
                    """
                )
            )
            .scalars()
            .all()
        )
    assert {r["es"] for r in runs} == {"holdout_main", "holdout_plus", "holdout_delay"}
    assert {r["git_sha"] for r in runs} == {"abc1234"}
    assert splits == ["holdout"]
    main = [r for r in runs if r["batch"] == str(batches["main"])]
    assert {r["model_name"] for r in main} == {
        "lgbm_A", "ridge_A", "lgbm_C", "ridge_C", "lgbm_D", "ridge_D",
        "persistence", "daily_naive", "weekly_naive",
    }  # fmt: skip
    # Holdout issue times only: 29 Oct 00:00 → 24 Nov 23:30 NEM time.
    assert {pd.Timestamp(r["eval_start"]) for r in runs} == {pd.Timestamp("2021-10-28 14:00Z")}
    assert {pd.Timestamp(r["eval_end"]) for r in runs} == {pd.Timestamp("2021-11-24 13:30Z")}
    config = main[0]["config"]
    assert config["retrains"] == ["2021-10-20", "2021-11-20"]
    assert config["n_origins"] == 27 * 48
    assert config["reproduction"]["passed"]
    assert config["config_hash"] == cfg.backtest.frozen_ml_config_hash
    # The holdout batches never reach the development loaders.
    assert str(load_evaluation_set(two_months, "VIC1", "main").batch_id) != str(batches["main"])

    with pytest.raises(HoldoutRefused, match="already scored"):
        preflight(two_months, cfg, "VIC1", GIT)

    # The report: S5, replication verdicts, development vs holdout, AEMO holdout part.
    with two_months.begin() as conn:
        run = start_ingestion_run(conn, "mmsdm", "aemo_predispatch", OCT)
        upsert_aemo_forecasts(conn, _aemo_rows(two_months), run)
    cfg.paths.reports_dir = tmp_path / "reports"  # never the repository's reports/
    cfg.aemo = cfg.aemo.model_copy(update={"first_issue": dt.date(2021, 10, 22)})
    report = build_report(two_months, cfg, "VIC1", LeakageOutcome(12, 0, 0))
    assert "| S5 |" in report and "**PASS**" in report
    assert report.count("exact") >= 3  # one reproduction row per set
    assert "Replication of S1–S4 on the holdout" in report
    assert report.count("**holds**") + report.count("**does not hold**") == 4
    assert "AEMO benchmark, holdout part" in report
    assert "none: every target interval comes from MMSDM" in report
    assert (tmp_path / "reports" / "figures" / "m2_holdout_mae_by_lead.png").is_file()
    assert (tmp_path / "reports" / "figures" / "m2_holdout_aemo_mae_by_lead.png").is_file()
    # Synthetic AEMO forecasts are the actual + 100 MW at every lead.
    b = load_benchmark(
        two_months,
        cfg,
        "VIC1",
        "holdout",
        batch=load_holdout_set(two_months, "VIC1", "main", str(cfg.backtest.frozen_ml_config_hash)),
    )
    c = compare(b, cfg.ml.seed)
    assert c.pooled30[MATCHED] == pytest.approx(OFFSET)
    assert b.issue_times.min() == pd.Timestamp("2021-10-28 14:00", tz="UTC")


def _aemo_rows(engine: Engine) -> pd.DataFrame:
    """Pre-dispatch every 30 min and 5-min pre-dispatch every 5 min, actual + OFFSET."""
    demand = load_market_frame(engine, "VIC1").set_index("interval_end")["total_demand_mw"]
    period = demand.rolling("30min").mean()
    start, end = pd.Timestamp("2021-10-20", tz="UTC"), pd.Timestamp("2021-11-26", tz="UTC")
    rows = []
    for run in pd.date_range(start, end, freq="30min"):
        avail = run + pd.Timedelta(seconds=134)
        for k in range(1, 49):
            t = run + pd.Timedelta(minutes=30 * k)
            if t in period.index:
                rows.append(("predispatch", run, t, 30, avail, period[t] + OFFSET))
    for run in pd.date_range(start, end, freq="5min"):
        avail = run - pd.Timedelta(seconds=270)
        for i in range(12):
            t = run + pd.Timedelta(minutes=5 * i)
            if t in demand.index:
                rows.append(("p5min", run, t, 5, avail, demand[t] + (OFFSET if i else 0.0)))
    cols = [
        "source",
        "run_time",
        "target_end",
        "resolution_minutes",
        "available_at",
        "total_demand_mw",
    ]
    return pd.DataFrame(rows, columns=cols).assign(region_id="VIC1")


def test_reproduction_failure_scores_and_stores_nothing(two_months: Engine, tmp_path: Path) -> None:
    cfg = _development_run(two_months, _cfg(tmp_path), plus=False)
    ref = load_evaluation_set(two_months, "VIC1", "main")
    path = artifact_path(ref, "lgbm_C", cfg.data_root())
    stored = pd.read_parquet(path)
    stored.loc[3, "h010"] += 0.5  # one forecast of a development issue time
    stored.to_parquet(path, index=False)
    with pytest.raises(ReproductionFailed, match=r"lgbm_C: 1 of .* forecasts differ"):
        run_holdout(
            two_months,
            cfg,
            "VIC1",
            GIT,
            cfg.data_path(cfg.ml.model_dir),
            {"main": ref},
            plus=False,
            delay=False,
        )
    with two_months.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM model_runs WHERE config->>'period' = 'holdout'")
        ).scalar_one()
    assert n == 0


def test_refusals(two_months: Engine, tmp_path: Path) -> None:
    cfg = _development_run(two_months, _cfg(tmp_path), plus=False)
    # No plus/delay development batches stored: refused, along with every other problem.
    bad_git = GitState(None, True, ("src/gridcast/models/ml.py",))
    with pytest.raises(HoldoutRefused) as err:
        preflight(two_months, cfg, "VIC1", bad_git)
    message = str(err.value)
    assert "not a git checkout" in message
    assert "src/gridcast/models/ml.py" in message
    assert "'plus'" in message

    dirty = dataclasses.replace(GIT, dirty=True)
    with pytest.raises(HoldoutRefused, match="uncommitted changes"):
        preflight(two_months, cfg, "VIC1", dirty)

    drifted = cfg.model_copy(deep=True)
    drifted.ml.seed += 1
    with pytest.raises(HoldoutRefused, match="differs from the frozen"):
        preflight(two_months, drifted, "VIC1", GIT)

    # The data must reach the last holdout target (data end 30 Nov 24:00 NEM time).
    late = cfg.model_copy(deep=True)
    late.backtest.holdout_end = dt.date(2021, 12, 2)
    ref = load_evaluation_set(two_months, "VIC1", "main")
    with pytest.raises(HoldoutRefused, match="lack data for their full horizon"):
        run_holdout(
            two_months,
            late,
            "VIC1",
            GIT,
            late.data_path(late.ml.model_dir),
            {"main": ref},
            plus=False,
            delay=False,
        )

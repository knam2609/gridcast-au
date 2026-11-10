"""M3 step 1 end to end on the test database: a stored point batch → conformal intervals."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, IntervalGridConfig, load_config
from gridcast.database.load import ensure_regions
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import interval_metrics, run_intervals_conformal
from gridcast.intervals.conformal import ConformalResult
from gridcast.intervals.inputs import read_stored_batch

pytestmark = pytest.mark.integration

FIRST = dt.date(2021, 11, 1)
SELECT = dt.date(2021, 12, 1)
END = dt.date(2022, 1, 10)
MODELS = ("lgbm_C", "persistence")


def _cfg(data_root: Path, source: str) -> AppConfig:
    """Monthly retrains from 1 Nov 2021; selection from 1 Dec; one-week windows and slice."""
    cfg = load_config()
    cfg.paths.data_root = data_root
    cfg.backtest.holdout_start = END
    cfg.ml = cfg.ml.model_copy(
        update={"first_issue": FIRST, "plus_first_issue": SELECT, "retrain_every_months": 1}
    )
    cfg.intervals = cfg.intervals.model_copy(
        update={
            "first_issue": SELECT,
            "source_batch": source,
            "point_models": list(MODELS),
            "selection_weeks": 1,
            "grid": IntervalGridConfig(
                window_weeks=[1, 2], groupings=["lead", "lead_tod"], aci_gammas=[0.0, 0.01]
            ),
        }
    )
    return cfg


def _write_source(db: Engine, cfg: AppConfig) -> str:
    """A fake M2 'main' batch: actuals and two models, issue times FIRST → END."""
    batch = str(uuid.uuid4())
    rng = np.random.default_rng(0)
    times = pd.date_range(nem_midnight_utc(str(FIRST)), nem_midnight_utc(str(END)), freq="30min")
    times = times[:-1]
    n, h = len(times), cfg.forecast.horizon_steps
    actual = 5000 + rng.normal(0, 100, (n, h))
    actual[5, 7] = np.nan
    values = {
        "actual": actual,
        "lgbm_C": actual + rng.normal(20, 80, (n, h)),
        "persistence": actual + rng.normal(0, 300, (n, h)),
    }
    values["lgbm_C"][40:44] = np.nan  # pairs without C weather leave the common mask
    directory = cfg.data_path(cfg.paths.backtest_dir)
    directory.mkdir(parents=True)
    for name, matrix in values.items():
        frame = pd.DataFrame(
            matrix.astype(np.float32), columns=[f"h{k:03d}" for k in range(1, h + 1)]
        )
        frame.insert(0, "origin_time", times)
        frame.to_parquet(directory / f"demand__{name}__{batch}.parquet", index=False)
    with db.begin() as conn:
        ensure_regions(conn, cfg)
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


def test_conformal_batch_is_stored_and_rescored(db: Engine, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "")
    source = _write_source(db, cfg)
    cfg = _cfg(tmp_path, source)
    batch, metrics = run_intervals_conformal(db, cfg, "VIC1", workers=2)

    with db.connect() as conn:
        runs = conn.execute(
            text(
                "SELECT run_id, model_name, model_kind, artifact_path, config, eval_start "
                "FROM model_runs WHERE batch_id = :b ORDER BY model_name"
            ),
            {"b": batch},
        ).all()
        stored = pd.read_sql(
            text(
                "SELECT r.model_name, m.metric, m.horizon, m.value, m.n FROM forecast_metrics m "
                "JOIN model_runs r USING (run_id) WHERE r.batch_id = :b AND m.split = 'dev'"
            ),
            conn,
            params={"b": batch},
        )
    assert [r.model_name for r in runs] == ["conformal_lgbm_C", "conformal_persistence"]
    assert {r.model_kind for r in runs} == {"interval"}
    assert runs[0].eval_start == nem_midnight_utc(str(SELECT))
    config = runs[0].config
    assert config["source_batch"] == source
    assert config["retrains"] == [
        str(nem_midnight_utc("2021-12-01")),
        str(nem_midnight_utc("2022-01-01")),
    ]
    # 2 retrains × 3 levels × 4 bands, each naming one of the 8 candidates
    assert len(config["selections"]) == 24
    assert {s["chosen"] for s in config["selections"]} <= set(config["candidates"])

    # The parquet bounds rescore to exactly the stored metrics.
    directory = cfg.data_root() / runs[0].artifact_path
    actual = pd.read_parquet(directory / "actual.parquet")
    times = pd.DatetimeIndex(actual.pop("origin_time"))
    source_batch = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), source)
    rows = np.asarray(source_batch.origin_times >= times[0])
    mask = source_batch.common_mask()[rows]
    lower, upper = {}, {}
    for level in cfg.intervals.levels:
        lower[level] = (
            pd.read_parquet(directory / f"conformal_lgbm_C__lo{level}.parquet")
            .iloc[:, 1:]
            .to_numpy()
        )
        upper[level] = (
            pd.read_parquet(directory / f"conformal_lgbm_C__hi{level}.parquet")
            .iloc[:, 1:]
            .to_numpy()
        )
    res = ConformalResult(times, lower, upper, [], [], pd.DataFrame(), pd.DataFrame())
    again = interval_metrics(
        res, actual.to_numpy(np.float64), mask, source_batch.forecasts["lgbm_C"][rows]
    )
    mine = stored[stored["model_name"] == "conformal_lgbm_C"].set_index(["metric", "horizon"])
    again = again.set_index(["metric", "horizon"])
    pd.testing.assert_series_equal(
        mine["value"].sort_index(), again["value"].sort_index(), check_names=False
    )
    pooled = mine.xs(0, level="horizon")["value"]
    assert 0.7 < pooled["coverage_80"] < 0.9
    assert pooled["interval_score_95"] > pooled["width_95"]
    assert (directory / "conformal_persistence__daily.parquet").exists()
    assert set(metrics) == set(MODELS)


def test_refuses_an_unknown_source_batch(db: Engine, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, str(uuid.uuid4()))
    with pytest.raises(LookupError, match="no lgbm_C run"):
        run_intervals_conformal(db, cfg, "VIC1")

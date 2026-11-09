"""M2 walk-forward end to end on the test database: fit, score, store, provenance."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, GridConfig, LightGBMConfig, RidgeConfig, load_config
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.forecasting.ml_backtest import run_ml_backtest, store_sets

pytestmark = pytest.mark.integration


@pytest.fixture
def ready(loaded: Engine) -> Engine:
    days = build_calendar_days(
        "VIC", "Australia/Melbourne", dt.date(2021, 9, 1), dt.date(2021, 11, 30)
    )
    with loaded.begin() as conn:
        upsert_calendar_days(conn, days)
    return loaded


def _cfg(data_root: Path) -> AppConfig:
    """October 2021 only: one retrain on 20 Oct, a one-week early-stopping slice."""
    cfg = load_config()
    cfg.paths.data_root = data_root
    cfg.backtest.holdout_start = dt.date(2021, 10, 29)
    cfg.ml = cfg.ml.model_copy(
        update={
            "first_issue": dt.date(2021, 10, 20),
            "plus_first_issue": dt.date(2021, 10, 25),
            "plus_train_start": dt.date(2021, 10, 9),
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


def test_walk_forward_is_stored_with_provenance(ready: Engine, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    sets = run_ml_backtest(
        ready, cfg, "VIC1", plus=False, delay=True, model_root=cfg.data_path(cfg.ml.model_dir)
    )
    assert [s.key for s in sets] == ["main", "delay"]
    batches = store_sets(ready, cfg, "VIC1", sets)

    with ready.connect() as conn:
        runs = (
            conn.execute(
                text(
                    """
                SELECT batch_id::text AS batch, model_name, model_kind,
                       config->>'evaluation_set' AS evaluation_set, config
                FROM model_runs ORDER BY run_id
                """
                )
            )
            .mappings()
            .all()
        )
        pooled = (
            conn.execute(
                text(
                    """
                SELECT r.batch_id::text AS batch, m.n
                FROM forecast_metrics m JOIN model_runs r USING (run_id)
                WHERE m.segment = 'all' AND m.horizon = 0 AND m.metric = 'mae'
                """
                )
            )
            .mappings()
            .all()
        )
        segments = (
            conn.execute(text("SELECT DISTINCT segment FROM forecast_metrics")).scalars().all()
        )

    main = [r for r in runs if r["batch"] == str(batches["main"])]
    delay = [r for r in runs if r["batch"] == str(batches["delay"])]
    assert {r["model_name"] for r in main} == {
        "lgbm_A", "ridge_A", "lgbm_C", "ridge_C", "lgbm_D", "ridge_D",
        "persistence", "daily_naive", "weekly_naive",
    }  # fmt: skip
    assert sorted(r["model_kind"] for r in main).count("ml") == 6
    assert {r["evaluation_set"] for r in main} == {"main"}
    assert {r["model_name"] for r in delay} == {
        "lgbm_C", "ridge_C", "lgbm_C_delay8h", "ridge_C_delay8h",
    }  # fmt: skip

    # Every model of a set is scored on the same pairs.
    for batch in batches.values():
        assert len({r["n"] for r in pooled if r["batch"] == str(batch)}) == 1
    assert {"very_hot_day", "very_cloudy_day"} <= set(segments)

    # Provenance: config hash, per-retrain selection, model files under the data root.
    lgbm_c = next(r for r in main if r["model_name"] == "lgbm_C")["config"]
    assert len(lgbm_c["config_hash"]) == 16
    retrain = lgbm_c["model_params"]["retrains"][0]
    assert retrain["retrain"] == "2021-10-20"
    bands = [r["band"] for r in lgbm_c["model_params"]["retrains"]]
    assert bands == [f"h{lo:03d}-{hi:03d}" for lo, hi in cfg.ml.lead_bands]  # one fit per band
    assert len(retrain["candidates"]) == 2  # residual and direct
    assert (tmp_path / retrain["path"]).is_file()
    # The delay check reuses headline C's chosen settings.
    delayed = next(r for r in delay if r["model_name"] == "lgbm_C_delay8h")["config"]
    assert delayed["model_params"]["retrains"][0]["chosen"] == retrain["chosen"]
    assert delayed["delay_hours"] == cfg.features.delay_sensitivity_hours

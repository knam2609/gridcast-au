"""M2 step 4 end to end on the test database: stored ML run → drop/add test → report."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import Engine

from gridcast.evaluation.m2_results import load_evaluation_set
from gridcast.evaluation.report import load_latest
from gridcast.features import catalogue
from gridcast.forecasting.ablation import (
    fits_frame,
    flag_table,
    read_ablation,
    reproduction_ok,
    run_weather_ablation,
    write_ablation,
)
from gridcast.forecasting.ml_backtest import run_ml_backtest, store_sets
from tests.integration.test_ml_backtest import _cfg, ready  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration


def test_ablation_reproduces_the_stored_run(ready: Engine, tmp_path: Path) -> None:  # noqa: F811
    cfg = _cfg(tmp_path)
    cfg.evaluation = cfg.evaluation.model_copy(
        update={"weather_ablation_retrains": [dt.date(2021, 10, 20)]}
    )
    sets = run_ml_backtest(
        ready, cfg, "VIC1", plus=False, delay=False, model_root=cfg.data_path(cfg.ml.model_dir)
    )
    batches = store_sets(ready, cfg, "VIC1", sets)

    # M1's report ignores the ML batches (they also hold baseline rows).
    with pytest.raises(LookupError):
        load_latest(ready, "VIC1", "demand")
    assert load_evaluation_set(ready, "VIC1", "main").batch_id == str(batches["main"])
    with pytest.raises(LookupError, match="delay"):
        load_evaluation_set(ready, "VIC1", "delay")

    result = run_weather_ablation(ready, cfg, "VIC1")
    weather = [c for c in catalogue.feature_names("C") if c not in catalogue.feature_names("A")]
    assert result["weather_features"] == weather
    assert len(result["fits"]) == len(cfg.ml.lead_bands) * (2 + 2 * len(weather))
    # The all-features fit is the stored run's own early-stopping fit, bit for bit.
    assert reproduction_ok(result)
    assert len(result["reproduced"]) == len(cfg.ml.lead_bands)

    path = write_ablation(cfg, result)
    assert path.is_file()
    again = read_ablation(cfg, str(batches["main"]))
    assert again is not None and again["fits"] == result["fits"]
    flags = flag_table(fits_frame(again), again["flag_share"])
    assert list(flags["feature"]) == weather

    # The test refuses a retrain the stored run does not have.
    cfg.evaluation = cfg.evaluation.model_copy(
        update={"weather_ablation_retrains": [dt.date(2021, 10, 1)]}
    )
    with pytest.raises(ValueError, match="not retrains"):
        run_weather_ablation(ready, cfg, "VIC1")

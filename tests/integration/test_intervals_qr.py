"""M3 step 2 end to end on the test database: a stored M2 run → quantile LightGBM C batch.

The QR run reads the stored run's lgbm_C settings, reproduces its point forecasts exactly,
and stores the rearranged quantiles, their metrics and the fit records.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.forecasting.ml_backtest import run_ml_backtest, store_sets
from gridcast.intervals.qr_batch import (
    RUN_NAME,
    run_backtest_qr,
    score_development,
    store_qr,
)
from tests.integration.test_ml_backtest import _cfg as ml_cfg

pytestmark = pytest.mark.integration

SCORED_FROM = dt.date(2021, 10, 25)


@pytest.fixture
def ready(loaded: Engine) -> Engine:
    days = build_calendar_days(
        "VIC", "Australia/Melbourne", dt.date(2021, 9, 1), dt.date(2021, 11, 30)
    )
    with loaded.begin() as conn:
        upsert_calendar_days(conn, days)
    return loaded


def test_qr_batch_reproduces_the_point_run_and_is_stored(ready: Engine, tmp_path: Path) -> None:
    cfg = ml_cfg(tmp_path)
    sets = run_ml_backtest(ready, cfg, "VIC1", plus=False, delay=False)
    source = str(store_sets(ready, cfg, "VIC1", sets)["main"])
    cfg.intervals = cfg.intervals.model_copy(
        update={"source_batch": source, "first_issue": SCORED_FROM}
    )

    smoke = run_backtest_qr(ready, cfg, "VIC1", until=pd.Timestamp("2021-10-23 14:00", tz="UTC"))
    assert not smoke.complete
    with pytest.raises(ValueError, match="never stored"):
        store_qr(ready, cfg, "VIC1", smoke, pd.DataFrame(), {})

    run = run_backtest_qr(ready, cfg, "VIC1", model_root=cfg.data_path(cfg.ml.model_dir))
    assert run.complete and run.run_config["point_reproduction"]["passed"]
    metrics, summary = score_development(run, cfg)
    run_id = store_qr(ready, cfg, "VIC1", run, metrics, summary, sha="test")

    with ready.connect() as conn:
        row = (
            conn.execute(
                text("SELECT model_name, model_kind, config FROM model_runs WHERE run_id = :r"),
                {"r": run_id},
            )
            .mappings()
            .one()
        )
        pooled = dict(
            conn.execute(
                text("SELECT metric, n FROM forecast_metrics WHERE run_id = :r AND horizon = 0"),
                {"r": run_id},
            ).all()
        )
    assert (row["model_name"], row["model_kind"]) == (RUN_NAME, "interval")
    config = row["config"]
    assert config["method"] == "qr" and config["source_batch"] == source
    assert config["quantiles"] == [0.025, 0.1, 0.25, 0.75, 0.9, 0.975]
    assert config["first_issue"].startswith("2021-10-24 14:00")  # NEM midnight of 25 Oct
    assert len(config["fits"]) == len(cfg.ml.lead_bands)  # one retrain
    fit = config["fits"][0]
    assert len(fit["quantiles"]) == 6 and (tmp_path / fit["quantiles"][0]["path"]).is_file()
    # Each quantile searched the test grid (residual, direct × 7 leaves) and kept its best.
    assert config["selection"] == "per_quantile_grid" and len(config["grid"]) == 2
    for q in fit["quantiles"]:
        assert [c["setting"] for c in q["candidates"]] == config["grid"]
        assert q["es_pinball"] == min(c["pinball"] for c in q["candidates"])
    # Every metric is on the same pairs: run 2's common mask from the scored issue times.
    assert {"coverage_80", "interval_score_95", "wis", "pinball_q025", "share_le_q975"} <= set(
        pooled
    )
    assert "crossing" in pooled
    assert set(pooled.values()) == {summary["scored_pairs"]}

    out = tmp_path / cfg.intervals.output_dir / str(run.batch_id)
    q10 = pd.read_parquet(out / f"{RUN_NAME}__q100.parquet")
    q90 = pd.read_parquet(out / f"{RUN_NAME}__q900.parquet")
    assert len(q10) == len(run.times) and (out / "actual.parquet").is_file()
    lo, hi = q10.drop(columns="origin_time").to_numpy(), q90.drop(columns="origin_time").to_numpy()
    ok = np.isfinite(lo) & np.isfinite(hi)
    assert ok.any() and (lo[ok] <= hi[ok]).all()  # rearranged: never crossed


def test_qr_run_refuses_a_different_ml_recipe(ready: Engine, tmp_path: Path) -> None:
    cfg = ml_cfg(tmp_path)
    sets = run_ml_backtest(ready, cfg, "VIC1", plus=False, delay=False)
    source = str(store_sets(ready, cfg, "VIC1", sets)["main"])
    cfg.intervals = cfg.intervals.model_copy(
        update={"source_batch": source, "first_issue": SCORED_FROM}
    )
    cfg.ml = cfg.ml.model_copy(update={"seed": cfg.ml.seed + 1})
    with pytest.raises(ValueError, match="same recipe"):
        run_backtest_qr(ready, cfg, "VIC1")

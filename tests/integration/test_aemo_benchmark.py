"""M2 step 5 end to end: stored ML run + synthetic AEMO forecasts -> benchmark report."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.database.load import start_ingestion_run, upsert_aemo_forecasts
from gridcast.database.queries import load_market_frame
from gridcast.evaluation.aemo_benchmark import (
    MATCHED,
    P5,
    STRICT,
    build_report,
    load_benchmark,
    pooled_mae,
)
from gridcast.forecasting.ml_backtest import ml_config_hash, run_ml_backtest, store_sets
from tests.integration.test_ml_backtest import _cfg, ready  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration

OFFSET = 100.0  # every synthetic AEMO forecast is the actual value + 100 MW


def _aemo_rows(engine: Engine) -> pd.DataFrame:
    """Pre-dispatch every 30 min (48 periods) and 5-min runs every 5 min, actual + OFFSET."""
    demand = load_market_frame(engine, "VIC1").set_index("interval_end")["total_demand_mw"]
    period = demand.rolling("30min").mean()  # mean of the six intervals ending at t
    start, end = pd.Timestamp("2021-10-20 00:00", tz="UTC"), pd.Timestamp("2021-10-30", tz="UTC")
    rows = []
    for run in pd.date_range(start, end, freq="30min"):
        for k in range(1, 49):
            t = run + pd.Timedelta(minutes=30 * k)
            if t in period.index:
                rows.append(("predispatch", run, t, 30, run + pd.Timedelta(seconds=134),
                             period[t] + OFFSET))  # fmt: skip
    for run in pd.date_range(start, end, freq="5min"):
        for i in range(12):
            t = run + pd.Timedelta(minutes=5 * i)
            if t in demand.index:
                rows.append(("p5min", run, t, 5, run - pd.Timedelta(seconds=270),
                             demand[t] + (OFFSET if i else 0.0)))  # fmt: skip
    cols = ["source", "run_time", "target_end", "resolution_minutes", "available_at",
            "total_demand_mw"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols).assign(region_id="VIC1")


def _setup(engine: Engine, tmp_path: Path) -> AppConfig:
    cfg = _cfg(tmp_path)
    cfg.paths.reports_dir = tmp_path / "reports"  # never the repository's reports/
    cfg.aemo = cfg.aemo.model_copy(update={"first_issue": dt.date(2021, 10, 22)})
    sets = run_ml_backtest(
        engine, cfg, "VIC1", plus=False, delay=False, model_root=cfg.data_path(cfg.ml.model_dir)
    )
    store_sets(engine, cfg, "VIC1", sets)
    with engine.begin() as conn:
        run = start_ingestion_run(conn, "mmsdm", "aemo_predispatch", dt.date(2021, 10, 1))
        upsert_aemo_forecasts(conn, _aemo_rows(engine), run)
    return cfg


def test_benchmark_against_synthetic_aemo(ready: Engine, tmp_path: Path) -> None:  # noqa: F811
    cfg = _setup(ready, tmp_path)

    cfg.backtest.frozen_ml_config_hash = "not-the-stored-hash"
    with pytest.raises(LookupError, match="frozen"):
        load_benchmark(ready, cfg, "VIC1")
    cfg.backtest.frozen_ml_config_hash = ml_config_hash(cfg)

    b = load_benchmark(ready, cfg, "VIC1")
    first = pd.Timestamp("2021-10-21 14:00", tz="UTC")  # 22 Oct 00:00 NEM time
    assert b.issue_times.min() == first
    assert b.issue_times.max() < pd.Timestamp("2021-10-28 14:00", tz="UTC")  # holdout start
    assert set(b.strict_lag.unique()) == {pd.Timedelta(minutes=30)}
    assert b.p5_late == 0 and b.p5_missing == 0

    mask = b.common30
    assert mask.any()
    mae = pooled_mae(b.aemo30, b.actual30, mask)
    assert mae[MATCHED] == pytest.approx(OFFSET) and mae[STRICT] == pytest.approx(OFFSET)
    assert pooled_mae({P5: b.p5}, b.actual5, b.common5)[P5] == pytest.approx(OFFSET)

    text = build_report(ready, cfg, "VIC1")
    assert "# Milestone 2: AEMO benchmark (development part)" in text
    assert f"`{ml_config_hash(cfg)}` = frozen config" in text
    assert (tmp_path / "reports" / "figures" / "m2_aemo_mae_by_lead.png").is_file()
    assert np.isfinite(b.p5).any()

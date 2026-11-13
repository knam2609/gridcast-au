"""M3 step 5: the AEMO comparison report end to end on the test database."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.database.load import start_ingestion_run, upsert_aemo_forecasts, upsert_aemo_poe
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.aemo_report import FIGURES, build_report
from gridcast.intervals.batch import interval_config_hash
from gridcast.intervals.halfhour import run_intervals_halfhour
from gridcast.intervals.inputs import read_wide
from gridcast.intervals.qr_batch import qr_config_hash
from tests.integration.test_m3_report import stored, world

pytestmark = pytest.mark.integration
__all__ = ["stored", "world"]  # fixtures

AEMO_FIRST = dt.date(2021, 12, 15)


def _targets(cfg: AppConfig, batch_dir: str) -> pd.Series:
    """Actual 30-minute means by period end, from the half-hour batch."""
    times, y = read_wide(cfg.data_root() / batch_dir / "actual.parquet")
    naive = times.tz_localize(None).to_numpy()
    ends = naive[:, None] + (np.arange(1, y.shape[1] + 1) * 30).astype("timedelta64[m]")
    s = pd.Series(y.ravel(), index=pd.DatetimeIndex(ends.ravel(), tz="UTC"))
    return s[~s.index.duplicated()].dropna()


def _store_aemo(db: Engine, cfg: AppConfig, batch_dir: str) -> None:
    """PDPASA bands and pre-dispatch runs around the actuals, every 30 min, 50 periods (so
    the strict run, 30 min older, still reaches 24 h after the issue time)."""
    actual = _targets(cfg, batch_dir)
    rng = np.random.default_rng(3)
    first = nem_midnight_utc(str(AEMO_FIRST)) - pd.Timedelta(days=1)
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    runs = pd.date_range(first, end, freq="30min", inclusive="left")
    k = np.arange(1, 51)
    run_time = np.repeat(runs, len(k))
    target = run_time + pd.to_timedelta(np.tile(k, len(runs)) * 30, unit="min")
    y = actual.reindex(target).fillna(5000.0).to_numpy()
    mid = y + 40 + rng.normal(0, 120, len(y))
    half = 120 + 2 * np.tile(k, len(runs))
    poe = pd.DataFrame(
        {
            "region_id": "VIC1",
            "run_time": run_time,
            "target_end": target,
            "available_at": run_time + pd.Timedelta(minutes=5),
            "demand10_mw": mid + half,
            "demand50_mw": mid,
            "demand90_mw": mid - half,
        }
    )
    pre = pd.DataFrame(
        {
            "source": "predispatch",
            "region_id": "VIC1",
            "run_time": run_time,
            "target_end": target,
            "resolution_minutes": 30,
            "available_at": run_time + pd.Timedelta(minutes=2),
            "total_demand_mw": mid - 40,
        }
    )
    with db.begin() as conn:
        run = start_ingestion_run(conn, "mmsdm", "aemo_pdpasa", AEMO_FIRST.replace(day=1))
        upsert_aemo_poe(conn, poe, run)
        upsert_aemo_forecasts(conn, pre, run)


@pytest.fixture
def frozen(world: Engine, stored: AppConfig) -> AppConfig:
    cfg = stored
    cfg.intervals = cfg.intervals.model_copy(
        update={
            "frozen_config_hash": interval_config_hash(cfg),
            "frozen_qr_config_hash": qr_config_hash(cfg),
            "frozen_commit": "test",
        }
    )
    cfg.aemo = cfg.aemo.model_copy(update={"first_issue": AEMO_FIRST})
    run = run_intervals_halfhour(world, cfg, "VIC1", workers=2, sha="test")
    with world.connect() as conn:
        batch_dir = conn.execute(
            text("SELECT artifact_path FROM model_runs WHERE run_id = :r"),
            {"r": next(iter(run.run_ids.values()))},
        ).scalar_one()
    _store_aemo(world, cfg, batch_dir)
    return cfg


def test_report_compares_every_interval_with_aemo(world: Engine, frozen: AppConfig) -> None:
    report = build_report(world, frozen, "VIC1")
    assert "(all equal)" in report  # every pooled statistic matched the stored metrics
    for name in ("AEMO band (matched)", "AEMO band re-centred (strict)", "CQR, LightGBM C"):
        assert name in report
    assert "## Pooled differences, with uncertainty" in report
    assert "7-day blocks" in report and "| 2021 Q4 |" in report
    # The fake band is symmetric by construction, and DEMAND50 sits 40 MW above pre-dispatch.
    assert "100.0% of PDPASA `LOR` rows" in report
    assert "+40.0 MW above the matched pre-dispatch" in report
    figures = frozen.resolve(frozen.paths.reports_dir) / "figures"
    for name in FIGURES:
        assert (figures / name).stat().st_size > 0


def test_refuses_without_a_freeze_record(world: Engine, frozen: AppConfig) -> None:
    frozen.intervals = frozen.intervals.model_copy(update={"frozen_commit": None})
    with pytest.raises(LookupError, match="only after the freeze"):
        build_report(world, frozen, "VIC1")


def test_refuses_runs_from_another_commit(world: Engine, frozen: AppConfig) -> None:
    frozen.intervals = frozen.intervals.model_copy(update={"frozen_commit": "abc1234"})
    with pytest.raises(ValueError, match="not the frozen commit"):
        build_report(world, frozen, "VIC1")


def test_stops_when_a_stored_metric_differs(world: Engine, frozen: AppConfig) -> None:
    with world.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE forecast_metrics SET value = value + 0.01
                WHERE metric = 'interval_score_80' AND horizon = 0 AND run_id = (
                    SELECT run_id FROM model_runs WHERE model_name = 'cqr30_lgbm_C')
                """
            )
        )
    with pytest.raises(ValueError, match="recomputed interval_score_80"):
        build_report(world, frozen, "VIC1")

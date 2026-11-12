"""M3 step 5 end to end on the test database: run 2 + a QR batch → the half-hour batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.intervals.batch import bounds_metrics, interval_config_hash
from gridcast.intervals.conformal import ConformalResult
from gridcast.intervals.cqr_batch import cqr_metrics
from gridcast.intervals.halfhour import (
    CQR_NAME,
    PERIOD_MINUTES,
    conformal_name,
    period_mask,
    period_means,
    run_intervals_halfhour,
)
from gridcast.intervals.inputs import read_stored_batch
from gridcast.intervals.qr_batch import qr_config_hash
from tests.integration.test_intervals_batch import MODELS
from tests.integration.test_intervals_cqr import _ready

pytestmark = pytest.mark.integration


def test_halfhour_batch_is_stored_and_rescored(db: Engine, tmp_path: Path) -> None:
    cfg = _ready(db, tmp_path)
    run = run_intervals_halfhour(db, cfg, "VIC1", workers=2, sha="test")
    names = [conformal_name(m) for m in MODELS] + [CQR_NAME]
    assert sorted(run.run_ids) == sorted(names)

    with db.connect() as conn:
        rows = {
            r["model_name"]: r
            for r in conn.execute(
                text(
                    "SELECT run_id, model_name, artifact_path, config, git_sha FROM model_runs "
                    "WHERE batch_id = CAST(:b AS uuid)"
                ),
                {"b": str(run.batch_id)},
            ).mappings()
        }
    assert sorted(rows) == sorted(names)
    source = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch)
    for name, row in rows.items():
        config = row["config"]
        assert config["resolution_minutes"] == PERIOD_MINUTES and row["git_sha"] == "test"
        assert config["interval_config_hash"] == interval_config_hash(cfg)
        assert config["qr_config_hash"] == qr_config_hash(cfg)
        assert config["period_bands"] == [[1, 1], [2, 4], [5, 12], [13, 48]]
        # 2 retrains × 3 levels × 4 bands, labelled in periods
        assert len(config["selections"]) == 24
        assert {s["band"] for s in config["selections"]} == {
            "h001-001",
            "h002-004",
            "h005-012",
            "h013-048",
        }
        point = "lgbm_C" if name == CQR_NAME else name.removeprefix("conformal30_")
        assert config["point_model"] == point

        # The stored bounds rescore to exactly the stored metrics, on whole periods.
        directory = cfg.data_root() / row["artifact_path"]
        actual = pd.read_parquet(directory / "actual.parquet")
        times = pd.DatetimeIndex(actual.pop("origin_time"))
        keep = np.asarray(source.origin_times >= times[0])
        y30 = period_means(source.actual[keep])
        np.testing.assert_array_equal(actual.to_numpy(np.float64), y30)
        mask = period_mask(source.common_mask()[keep])

        def read(
            side: str, level: int, directory: Path = directory, name: str = name
        ) -> np.ndarray:
            path = directory / f"{name}__{side}{level}.parquet"
            return pd.read_parquet(path).iloc[:, 1:].to_numpy()

        lower = {lv: read("lo", lv) for lv in cfg.intervals.levels}
        upper = {lv: read("hi", lv) for lv in cfg.intervals.levels}
        assert lower[80].shape == (len(times), 48)
        point30 = period_means(source.forecasts[point][keep])
        if name == CQR_NAME:
            res = ConformalResult(times, lower, upper, [], [], pd.DataFrame(), pd.DataFrame())
            again = cqr_metrics(res, actual.to_numpy(np.float64), mask, point30)
        else:
            again = bounds_metrics(lower, upper, actual.to_numpy(np.float64), mask, point30)
        with db.connect() as conn:
            stored = pd.read_sql(
                text(
                    "SELECT metric, horizon, value FROM forecast_metrics "
                    "WHERE run_id = :r AND split = 'dev'"
                ),
                conn,
                params={"r": row["run_id"]},
            )
        mine = stored.set_index(["metric", "horizon"])["value"].sort_index()
        theirs = again.set_index(["metric", "horizon"])["value"].sort_index()
        pd.testing.assert_series_equal(mine, theirs, check_names=False)
        coverage = mine.loc[("coverage_80", 0)]
        assert 0.7 < coverage < 0.9, (name, coverage)

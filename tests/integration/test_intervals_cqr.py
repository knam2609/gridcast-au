"""M3 step 3 end to end on the test database: run 2 + a stored QR batch → the CQR batch."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import EVALUATION_SET
from gridcast.intervals.conformal import ConformalResult
from gridcast.intervals.cqr_batch import RUN_NAME, cqr_metrics, run_intervals_cqr
from gridcast.intervals.inputs import read_stored_batch
from gridcast.intervals.qr_batch import RUN_NAME as QR_RUN_NAME
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.quantile import quantile_key, quantile_levels
from tests.integration.test_intervals_batch import SELECT, _cfg, _write_source

pytestmark = pytest.mark.integration


def _write_qr(
    db: Engine,
    cfg: AppConfig,
    hole: tuple[int, int] | None = None,
    recipe: str | None = None,
) -> str:
    """A fake QR batch around run 2's lgbm_C: too-narrow quantiles, NaN where C is missing.

    ``hole`` blanks one pair of the 0.1 quantile; ``recipe`` overrides the QR config hash.
    """
    source = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch)
    point = source.forecasts["lgbm_C"]
    rng = np.random.default_rng(1)
    batch = str(uuid.uuid4())
    directory = cfg.data_path(cfg.intervals.output_dir) / batch
    directory.mkdir(parents=True)

    def write(name: str, values: np.ndarray) -> None:
        frame = pd.DataFrame(
            values.astype(np.float32),
            columns=[f"h{k:03d}" for k in range(1, values.shape[1] + 1)],
        )
        frame.insert(0, "origin_time", source.origin_times)
        frame.to_parquet(directory / name, index=False)

    write("actual.parquet", source.actual)
    spread = rng.uniform(0.6, 1.0, point.shape)
    for tau in quantile_levels(cfg.intervals.levels):
        values = point + 60 * (tau - 0.5) * 2 * spread  # ±60 MW at the extremes
        if hole is not None and tau == 0.1:
            values = values.copy()
            values[hole] = np.nan
        write(f"{QR_RUN_NAME}__{quantile_key(tau)}.parquet", values)
    config = {
        "evaluation_set": EVALUATION_SET,
        "complete_window": True,
        "point_reproduction": {"passed": True},
        "source_batch": cfg.intervals.source_batch,
        "qr_config_hash": recipe or qr_config_hash(cfg),
    }
    with db.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id,
                                        config, artifact_path, status)
                VALUES (CAST(:b AS uuid), :name, 'interval', 'demand', 'VIC1',
                        CAST(:c AS jsonb), :path, 'success')
                """
            ),
            {
                "b": batch,
                "name": QR_RUN_NAME,
                "c": json.dumps(config),
                "path": str(directory.relative_to(cfg.data_root())),
            },
        )
    return batch


def _ready(db: Engine, tmp_path: Path, **qr: object) -> AppConfig:
    cfg = _cfg(tmp_path, "")
    cfg = _cfg(tmp_path, _write_source(db, cfg))
    qr_batch = _write_qr(db, cfg, **qr)  # type: ignore[arg-type]
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": qr_batch})
    return cfg


def test_cqr_batch_is_stored_and_rescored(db: Engine, tmp_path: Path) -> None:
    cfg = _ready(db, tmp_path)
    run = run_intervals_cqr(db, cfg, "VIC1", workers=2, sha="test")

    with db.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT model_name, model_kind, artifact_path, config, eval_start, git_sha "
                    "FROM model_runs WHERE run_id = :r"
                ),
                {"r": run.run_id},
            )
            .mappings()
            .one()
        )
        stored = pd.read_sql(
            text(
                "SELECT metric, horizon, value, n FROM forecast_metrics "
                "WHERE run_id = :r AND split = 'dev'"
            ),
            conn,
            params={"r": run.run_id},
        )
    assert (row["model_name"], row["model_kind"], row["git_sha"]) == (RUN_NAME, "interval", "test")
    assert row["eval_start"] == nem_midnight_utc(str(SELECT))
    config = row["config"]
    assert config["method"] == "cqr" and config["qr_batch"] == cfg.intervals.qr_batch
    assert config["qr_config_hash"] == qr_config_hash(cfg)
    assert config["calibration_first_issue"] < config["first_issue"]
    # 2 retrains × 3 levels × 4 bands, each naming one of the 8 candidates
    assert len(config["selections"]) == 24
    assert {s["chosen"] for s in config["selections"]} <= set(config["candidates"])
    assert set(config["crossed"]) == {"50", "80", "95"}

    # The parquet bounds rescore to exactly the stored metrics, on run 2's common mask.
    directory = cfg.data_root() / row["artifact_path"]
    actual = pd.read_parquet(directory / "actual.parquet")
    times = pd.DatetimeIndex(actual.pop("origin_time"))
    source = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch)
    rows = np.asarray(source.origin_times >= times[0])
    mask = source.common_mask()[rows]

    def read(name: str) -> np.ndarray:
        return pd.read_parquet(directory / f"{RUN_NAME}__{name}.parquet").iloc[:, 1:].to_numpy()

    bounds = {
        side: {lv: read(f"{side}{lv}") for lv in cfg.intervals.levels} for side in ("lo", "hi")
    }
    res = ConformalResult(times, bounds["lo"], bounds["hi"], [], [], pd.DataFrame(), pd.DataFrame())
    again = cqr_metrics(
        res, actual.to_numpy(np.float64), mask, source.forecasts["lgbm_C"][rows]
    ).set_index(["metric", "horizon"])
    mine = stored.set_index(["metric", "horizon"])
    pd.testing.assert_series_equal(
        mine["value"].sort_index(), again["value"].sort_index(), check_names=False
    )
    pooled = mine.xs(0, level="horizon")
    assert set(pooled["n"]) == {int(mask.sum())}
    # The fake quantiles are far too narrow (±60 MW against ~100 MW noise); CQR widens them.
    assert 0.7 < pooled.loc["coverage_80", "value"] < 0.9
    assert {"pinball_q025", "share_le_q975", "non_nested", "wis"} <= set(pooled.index)
    assert (directory / f"{RUN_NAME}__daily.parquet").exists()


def test_refuses_a_different_qr_recipe(db: Engine, tmp_path: Path) -> None:
    cfg = _ready(db, tmp_path, recipe="0000000000000000")
    with pytest.raises(ValueError, match="same recipe"):
        run_intervals_cqr(db, cfg, "VIC1")


def test_refuses_a_quantile_missing_on_a_scored_pair(db: Engine, tmp_path: Path) -> None:
    cfg = _ready(db, tmp_path, hole=(100, 3))
    with pytest.raises(ValueError, match="missing on 1 pairs"):
        run_intervals_cqr(db, cfg, "VIC1")


def test_refuses_without_a_qr_batch(db: Engine, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "")
    cfg = _cfg(tmp_path, _write_source(db, cfg))
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": None})
    with pytest.raises(ValueError, match="qr_batch is not set"):
        run_intervals_cqr(db, cfg, "VIC1")

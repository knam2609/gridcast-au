"""M3 step 6a end to end: M2's development and holdout runs → step 2's QR batch → the QR
holdout run, which reproduces the first retrain exactly before fitting the rest and stores
its forecasts unscored."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.holdout import GitState, run_holdout, store_holdout
from gridcast.evaluation.m2_results import load_evaluation_set
from gridcast.forecasting.ml_backtest import ml_config_hash, run_ml_backtest, store_sets
from gridcast.intervals.batch import interval_config_hash
from gridcast.intervals.freeze import FreezeState
from gridcast.intervals.inputs import read_wide
from gridcast.intervals.qr_batch import RUN_NAME, qr_config_hash, run_backtest_qr, score_development
from gridcast.intervals.qr_batch import store_qr as store_dev_qr
from gridcast.intervals.qr_holdout import (
    CROSSED,
    EVALUATION_SET,
    QrHoldoutRefused,
    QrHoldoutReproductionFailed,
    preflight,
    run_qr_holdout,
    store_qr_holdout,
)
from tests.integration.test_holdout import _cfg as holdout_cfg
from tests.integration.test_holdout import two_months

pytestmark = pytest.mark.integration
__all__ = ["two_months"]  # fixture

GIT = GitState("abc1234", False, ())
FIRST_RETRAIN = dt.date(2021, 10, 20)


def _world(engine: Engine, data_root: Path) -> AppConfig:
    """M2 development run (frozen), its holdout run, step 2's QR batch, the interval freeze."""
    cfg = holdout_cfg(data_root)
    models = cfg.data_path(cfg.ml.model_dir)
    sets = run_ml_backtest(engine, cfg, "VIC1", plus=False, delay=False, model_root=models)
    source = str(store_sets(engine, cfg, "VIC1", sets)["main"])
    cfg.backtest.frozen_ml_config_hash = ml_config_hash(cfg)
    cfg.backtest.frozen_ml_commit = "frozen0"
    refs = {"main": load_evaluation_set(engine, "VIC1", "main")}
    m2 = run_holdout(engine, cfg, "VIC1", GIT, models, refs, plus=False, delay=False)
    store_holdout(engine, cfg, "VIC1", m2, GIT)

    cfg.intervals = cfg.intervals.model_copy(
        update={"source_batch": source, "first_issue": FIRST_RETRAIN}
    )
    qr = run_backtest_qr(engine, cfg, "VIC1", model_root=models)
    metrics, summary = score_development(qr, cfg)
    store_dev_qr(engine, cfg, "VIC1", qr, metrics, summary, sha="test")
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": str(qr.batch_id)})
    cfg.intervals = cfg.intervals.model_copy(
        update={
            "frozen_config_hash": interval_config_hash(cfg),
            "frozen_qr_config_hash": qr_config_hash(cfg),
            "frozen_commit": "frozen1",
        }
    )
    return cfg


def _freeze(cfg: AppConfig) -> FreezeState:
    return FreezeState(
        config_hash=interval_config_hash(cfg),
        qr_config_hash=qr_config_hash(cfg),
        frozen_config_hash=cfg.intervals.frozen_config_hash,
        frozen_qr_config_hash=cfg.intervals.frozen_qr_config_hash,
        frozen_commit=cfg.intervals.frozen_commit,
        sha="abc1234",
        dirty=False,
        frozen_changed=(),
    )


def _qr_holdout_runs(engine: Engine) -> list[int]:
    with engine.connect() as conn:
        return list(
            conn.execute(
                text("SELECT run_id FROM model_runs WHERE config->>'evaluation_set' = :es"),
                {"es": EVALUATION_SET},
            ).scalars()
        )


def test_qr_holdout_reproduces_then_stores_unscored(two_months: Engine, tmp_path: Path) -> None:
    cfg = _world(two_months, tmp_path)
    freeze = _freeze(cfg)
    with pytest.raises(QrHoldoutRefused, match="uncommitted changes"):
        preflight(two_months, cfg, "VIC1", dataclasses.replace(freeze, dirty=True))
    with pytest.raises(QrHoldoutRefused, match="frozen code changed"):
        preflight(
            two_months, cfg, "VIC1", dataclasses.replace(freeze, frozen_changed=("src/x.py",))
        )
    with pytest.raises(QrHoldoutRefused, match="QR config hash"):
        preflight(two_months, cfg, "VIC1", dataclasses.replace(freeze, qr_config_hash="other"))

    inputs = preflight(two_months, cfg, "VIC1", freeze)
    run = run_qr_holdout(two_months, cfg, "VIC1", inputs, cfg.data_path(cfg.ml.model_dir))
    rep = run.reproduction
    assert rep.passed and rep.retrain == "2021-10-20"
    assert rep.issue_times == 9 * 48  # 20 → 28 Oct, the development part of the first fold
    assert rep.fits == len(cfg.ml.lead_bands)
    assert rep.pairs > 0
    assert run.result.folds == ["2021-10-20", "2021-11-20"]
    assert len(run.times) == 36 * 48  # 20 Oct → 24 Nov
    assert run.run_config["holdout_issue_times"] == 27 * 48
    assert len(run.run_config["fits"]) == 2 * len(cfg.ml.lead_bands)

    run_id = store_qr_holdout(two_months, cfg, "VIC1", run, "abc1234")
    with two_months.connect() as conn:
        n_metrics = conn.execute(
            text("SELECT count(*) FROM forecast_metrics WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
        row = (
            conn.execute(
                text("SELECT model_name, git_sha, config FROM model_runs WHERE run_id = :r"),
                {"r": run_id},
            )
            .mappings()
            .one()
        )
    assert n_metrics == 0  # nothing scored
    assert (row["model_name"], row["git_sha"]) == (RUN_NAME, "abc1234")
    config = row["config"]
    assert config["scored"] is False and config["qr_reproduction"]["passed"]
    assert config["development_qr_batch"] == cfg.intervals.qr_batch
    assert config["qr_config_hash"] == cfg.intervals.frozen_qr_config_hash

    out = tmp_path / cfg.intervals.output_dir / str(run.batch_id)
    times, q100 = read_wide(out / f"{RUN_NAME}__q100.parquet")
    _, q900 = read_wide(out / f"{RUN_NAME}__q900.parquet")
    _, crossed = read_wide(out / f"{RUN_NAME}__{CROSSED}.parquet")
    assert times.equals(run.times) and (out / "actual.parquet").is_file()
    ok = np.isfinite(q100) & np.isfinite(q900)
    assert ok.any() and (q100[ok] <= q900[ok]).all()  # rearranged
    assert set(np.unique(crossed)) <= {0.0, 1.0}
    # The development part equals step 2's stored quantiles exactly.
    dev_dir = tmp_path / cfg.intervals.output_dir / str(cfg.intervals.qr_batch)
    dev_times, dev_q100 = read_wide(dev_dir / f"{RUN_NAME}__q100.parquet")
    overlap = np.asarray(times < pd.Timestamp("2021-10-28 14:00Z"))
    first = int(np.searchsorted(dev_times, times[0]))
    assert np.array_equal(
        q100[overlap], dev_q100[first : first + int(overlap.sum())], equal_nan=True
    )

    with pytest.raises(QrHoldoutRefused, match="already exists"):
        preflight(two_months, cfg, "VIC1", freeze)


def test_a_changed_development_quantile_stops_the_run(two_months: Engine, tmp_path: Path) -> None:
    cfg = _world(two_months, tmp_path)
    dev_dir = tmp_path / cfg.intervals.output_dir / str(cfg.intervals.qr_batch)
    path = dev_dir / f"{RUN_NAME}__q100.parquet"
    frame = pd.read_parquet(path)
    first = pd.Timestamp(dt.datetime(2021, 10, 19, 14, 0), tz="UTC")  # NEM midnight of 20 Oct
    row = int(np.flatnonzero(frame["origin_time"] >= first)[3])
    frame.loc[row, "h010"] = np.float32(frame.loc[row, "h010"] + 1.0)
    frame.to_parquet(path, index=False)

    inputs = preflight(two_months, cfg, "VIC1", _freeze(cfg))
    with pytest.raises(QrHoldoutReproductionFailed, match="q100: 1 quantile values differ"):
        run_qr_holdout(two_months, cfg, "VIC1", inputs, cfg.data_path(cfg.ml.model_dir))
    assert _qr_holdout_runs(two_months) == []

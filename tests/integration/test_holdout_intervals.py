"""M3 step 6b end to end on the test database: stored development batches (built by the real
commands), a fake ``holdout_main`` and step-6a QR batch → the locked holdout scoring run.

The continuous run must reproduce every stored development interval exactly before it scores
anything; the controls show a changed development bound or QR overlap stops it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, load_config
from gridcast.database.load import (
    ensure_regions,
    start_ingestion_run,
    upsert_aemo_forecasts,
    upsert_aemo_poe,
    upsert_market_intervals,
)
from gridcast.evaluation.m2_report import LeakageOutcome
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.features.inputs import load_context
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import ingest_weather_range
from gridcast.intervals.batch import interval_config_hash, run_intervals_conformal
from gridcast.intervals.cqr_batch import run_intervals_cqr
from gridcast.intervals.freeze import FreezeState
from gridcast.intervals.halfhour import run_intervals_halfhour
from gridcast.intervals.holdout import (
    EVALUATION_SET,
    HoldoutIntervalsRefused,
    HoldoutIntervalsReproductionFailed,
    preflight,
    run_holdout_intervals,
    store_holdout_intervals,
)
from gridcast.intervals.holdout_report import FIGURES as HOLDOUT_FIGURES
from gridcast.intervals.holdout_report import build_report as build_holdout_report
from gridcast.intervals.inputs import read_wide
from gridcast.intervals.qr_batch import RUN_NAME as QR_RUN_NAME
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.qr_holdout import CROSSED
from gridcast.intervals.qr_holdout import EVALUATION_SET as QR_HOLDOUT_SET
from gridcast.intervals.quantile import quantile_key, quantile_levels
from gridcast.intervals.scores import summarise
from tests.integration.factories import clean_month
from tests.integration.test_intervals_batch import FIRST, _cfg
from tests.integration.test_m3_report import POINTS, _store_qr
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

pytestmark = pytest.mark.integration

HOLD = dt.date(2022, 1, 10)  # holdout start (the development end of the step-1 fixtures)
HEND = dt.date(2022, 2, 20)  # holdout end, exclusive
MONTHS = ((2021, 11), (2021, 12), (2022, 1), (2022, 2))
SHA = "abc1234"
FROZEN = "frozen1"


@pytest.fixture
def hworld(db: Engine, tmp_path: Path) -> Engine:
    """Clean market data, calendar days and fake weather for Nov 2021 → Feb 2022."""
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        for year, month in MONTHS:
            run = start_ingestion_run(conn, "mmsdm", "market_intervals", dt.date(year, month, 1))
            upsert_market_intervals(conn, clean_month(year, month), run)
            conn.execute(
                text("UPDATE ingestion_runs SET status='success' WHERE run_id=:r"), {"r": run}
            )
        upsert_calendar_days(
            conn,
            build_calendar_days(
                "VIC", "Australia/Melbourne", dt.date(2021, 10, 1), dt.date(2022, 3, 31)
            ),
        )
    client = OpenMeteoClient(
        om_config(), tmp_path / "om", transport=FakeOpenMeteo().transport(), today=lambda: TODAY
    )
    results = ingest_weather_range(
        db, cfg, dt.date(2021, 10, 1), dt.date(2022, 2, 1), client=client
    )
    assert {r.status for r in results} == {"success"}
    return db


def _write_batch(
    db: Engine, cfg: AppConfig, times: pd.DatetimeIndex, values: dict[str, np.ndarray],
    config: dict[str, object],
) -> str:  # fmt: skip
    batch = str(uuid.uuid4())
    directory = cfg.data_path(cfg.paths.backtest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    h = cfg.forecast.horizon_steps
    for name, matrix in values.items():
        frame = pd.DataFrame(
            matrix.astype(np.float32), columns=[f"h{k:03d}" for k in range(1, h + 1)]
        )
        frame.insert(0, "origin_time", times)
        frame.to_parquet(directory / f"demand__{name}__{batch}.parquet", index=False)
    with db.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id,
                                        config, status)
                VALUES (CAST(:b AS uuid), 'lgbm_C', 'ml', 'demand', 'VIC1',
                        CAST(:c AS jsonb), 'success')
                """
            ),
            {"b": batch, "c": json.dumps(config)},
        )
    return batch


def _write_point_batches(db: Engine, cfg: AppConfig) -> tuple[str, str]:
    """A fake run 2 (FIRST → HOLD) and holdout_main (HOLD → HEND) from one series."""
    ctx = load_context(db, cfg, "VIC1")
    times = pd.date_range(nem_midnight_utc(str(FIRST)), nem_midnight_utc(str(HEND)), freq="30min")
    times = times[:-1]
    h = cfg.forecast.horizon_steps
    tpos = ctx.times.get_indexer(times)[:, None] + np.arange(1, h + 1)[None, :]
    actual = ctx.y[tpos]
    rng = np.random.default_rng(0)
    values = {"actual": actual}
    for name, (bias, sd) in POINTS.items():
        values[name] = actual + rng.normal(bias, sd, actual.shape)
    values["lgbm_C"][40:44] = np.nan  # pairs without C weather leave the common mask
    values["lgbm_C"][-300:-296, 5:9] = np.nan  # and some in the holdout
    dev = np.asarray(times < nem_midnight_utc(str(HOLD)))
    source = _write_batch(
        db, cfg, times[dev], {k: v[dev] for k, v in values.items()},
        {"evaluation_set": "main", "complete_window": True},
    )  # fmt: skip
    holdout = _write_batch(
        db, cfg, times[~dev], {k: v[~dev] for k, v in values.items()},
        {"evaluation_set": "holdout_main", "config_hash": ml_config_hash(cfg),
         "model_params": {"retrains": []}},
    )  # fmt: skip
    return source, holdout


def _store_qr_holdout(db: Engine, cfg: AppConfig, holdout: str) -> str:
    """A fake step-6a batch: step 2's quantiles from the last development retrain, then new."""
    dev_dir = cfg.data_path(cfg.intervals.output_dir) / str(cfg.intervals.qr_batch)
    first = nem_midnight_utc("2022-01-01")
    keys = [quantile_key(t) for t in quantile_levels(cfg.intervals.levels)]
    t_hold, point = read_wide(
        cfg.data_path(cfg.paths.backtest_dir) / f"demand__lgbm_C__{holdout}.parquet"
    )
    _, actual_hold = read_wide(
        cfg.data_path(cfg.paths.backtest_dir) / f"demand__actual__{holdout}.parquet"
    )
    spread = np.random.default_rng(2).uniform(0.6, 1.0, point.shape)
    batch = uuid.uuid4()
    directory = cfg.data_path(cfg.intervals.output_dir) / str(batch)
    directory.mkdir(parents=True)

    def write(name: str, times: pd.DatetimeIndex, values: np.ndarray) -> None:
        frame = pd.DataFrame(values, columns=[f"h{k:03d}" for k in range(1, values.shape[1] + 1)])
        frame.insert(0, "origin_time", times)
        frame.to_parquet(directory / name, index=False)

    times = None
    for key in keys:
        t_dev, q_dev = read_wide(dev_dir / f"{QR_RUN_NAME}__{key}.parquet")
        keep = np.asarray(t_dev >= first)
        tau = int(key[1:]) / 1000
        q_hold = point + 60 * (tau - 0.5) * 2 * spread
        times = pd.DatetimeIndex(t_dev[keep].append(t_hold))
        write(
            f"{QR_RUN_NAME}__{key}.parquet",
            times,
            np.concatenate([q_dev[keep], q_hold]).astype(np.float32),
        )
    assert times is not None
    _, a_dev = read_wide(dev_dir / "actual.parquet")
    t_dev, _ = read_wide(dev_dir / "actual.parquet")
    keep = np.asarray(t_dev >= first)
    write("actual.parquet", times, np.concatenate([a_dev[keep], actual_hold]).astype(np.float32))
    write(f"{QR_RUN_NAME}__{CROSSED}.parquet", times, np.zeros((len(times), point.shape[1]), bool))
    config = {
        "evaluation_set": QR_HOLDOUT_SET,
        "complete_window": True,
        "scored": False,
        "qr_reproduction": {
            "passed": True,
            "retrain": "2022-01-01",
            "reference_batch": cfg.intervals.qr_batch,
            "issue_times": 9 * 48,
            "pairs": 9 * 48 * point.shape[1] * len(keys),
            "fits": len(cfg.ml.lead_bands),
        },
        "point_reproduction": {"passed": True, "pairs": int(np.isfinite(point).sum())},
        "retrains": ["2022-01-01", "2022-02-01"],
        "forecast_first_issue": str(first),
        "seconds": 3600.0,
        "crossing_share_all_pairs": 0.0,
        "fits": [
            {"retrain": r, "band": f"h{lo:03d}-{hi:03d}", "crossing_share": 0.0}
            for r in ("2022-01-01", "2022-02-01")
            for lo, hi in cfg.ml.lead_bands
        ],
        "development_qr_batch": cfg.intervals.qr_batch,
        "source_batch": cfg.intervals.source_batch,
        "holdout_point_batch": holdout,
        "qr_config_hash": qr_config_hash(cfg),
        "frozen_commit": FROZEN,
    }
    with db.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id,
                                        config, git_sha, artifact_path, status)
                VALUES (:b, :m, 'interval', 'demand', 'VIC1', CAST(:c AS jsonb), :s, :a,
                        'success')
                """
            ),
            {
                "b": batch,
                "m": QR_RUN_NAME,
                "c": json.dumps(config),
                "s": SHA,
                "a": str(directory.relative_to(cfg.data_root())),
            },
        )
    return str(batch)


@pytest.fixture
def hstored(hworld: Engine, tmp_path: Path) -> AppConfig:
    """Every input of the holdout run: development batches by the real commands, fakes else."""
    (tmp_path / "data").mkdir()
    cfg = _cfg(tmp_path / "data", "")
    cfg.backtest.holdout_end = HEND
    source, holdout = _write_point_batches(hworld, cfg)
    cfg = _cfg(tmp_path / "data", source)
    cfg.backtest.holdout_end = HEND
    cfg.backtest.frozen_ml_config_hash = ml_config_hash(cfg)
    cfg.backtest.frozen_ml_commit = "frozen0"
    cfg.intervals = cfg.intervals.model_copy(update={"point_models": list(POINTS)})
    cfg.paths.reports_dir = tmp_path / "reports"  # never the repository's reports/
    run_intervals_conformal(hworld, cfg, "VIC1", workers=2)
    cfg.intervals = cfg.intervals.model_copy(update={"qr_batch": _store_qr(hworld, cfg)})
    run_intervals_cqr(hworld, cfg, "VIC1", workers=2, sha=f"{FROZEN}abc")
    run_intervals_halfhour(hworld, cfg, "VIC1", workers=2, sha=f"{FROZEN}abc")
    cfg.intervals = cfg.intervals.model_copy(
        update={
            "frozen_config_hash": interval_config_hash(cfg),
            "frozen_qr_config_hash": qr_config_hash(cfg),
            "frozen_commit": FROZEN,
        }
    )
    _store_qr_holdout(hworld, cfg, holdout)
    return cfg


def freeze_of(cfg: AppConfig) -> FreezeState:
    return FreezeState(
        config_hash=interval_config_hash(cfg),
        qr_config_hash=qr_config_hash(cfg),
        frozen_config_hash=cfg.intervals.frozen_config_hash,
        frozen_qr_config_hash=cfg.intervals.frozen_qr_config_hash,
        frozen_commit=cfg.intervals.frozen_commit,
        sha=SHA,
        dirty=False,
        frozen_changed=(),
    )


def _holdout_runs(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text("SELECT count(*) FROM model_runs WHERE config->>'evaluation_set' = :es"),
                {"es": EVALUATION_SET},
            ).scalar_one()
        )


def test_holdout_intervals_reproduce_score_once_and_lock(
    hworld: Engine, hstored: AppConfig
) -> None:
    cfg = hstored
    freeze = freeze_of(cfg)
    with pytest.raises(HoldoutIntervalsRefused, match="uncommitted changes"):
        preflight(hworld, cfg, "VIC1", dataclasses.replace(freeze, dirty=True))

    inputs = preflight(hworld, cfg, "VIC1", freeze)
    run = run_holdout_intervals(hworld, cfg, "VIC1", inputs, workers=2)
    assert len(run.reproductions) == 10  # 4 conformal + CQR, at 5 and 30 minutes
    assert all(r.passed and r.issue_times > 0 and r.selections > 0 for r in run.reproductions)
    assert run.qr_overlap_issue_times == 9 * 48  # 1 → 9 Jan
    assert run.times[0] == nem_midnight_utc(str(HOLD))
    assert len(run.times) == (HEND - HOLD).days * 48

    batch, run_ids = store_holdout_intervals(hworld, cfg, "VIC1", run, SHA)
    assert set(run_ids) == {
        "conformal_lgbm_C", "conformal_persistence", "conformal_daily_naive",
        "conformal_weekly_naive", "cqr_lgbm_C", "qr_lgbm_C",
        "conformal30_lgbm_C", "conformal30_persistence", "conformal30_daily_naive",
        "conformal30_weekly_naive", "cqr30_lgbm_C",
    }  # fmt: skip
    with hworld.connect() as conn:
        splits = conn.execute(
            text(
                "SELECT DISTINCT m.split FROM forecast_metrics m JOIN model_runs r USING (run_id) "
                "WHERE r.batch_id = :b"
            ),
            {"b": batch},
        ).scalars()
        assert list(splits) == ["holdout"]
        row = (
            conn.execute(
                text("SELECT config, eval_start, artifact_path FROM model_runs WHERE run_id = :r"),
                {"r": run_ids["cqr_lgbm_C"]},
            )
            .mappings()
            .one()
        )
        stored_cov = conn.execute(
            text(
                "SELECT value FROM forecast_metrics WHERE run_id = :r AND metric = 'coverage_80' "
                "AND horizon = 0"
            ),
            {"r": run_ids["cqr_lgbm_C"]},
        ).scalar_one()
    config = row["config"]
    assert config["reproduction"]["passed"] and config["qr_holdout_batch"]
    # Selections at 1 Dec, 1 Jan and 1 Feb; the holdout uses the last two.
    assert len(config["retrains"]) == 3 and len(config["holdout_retrains"]) == 2
    assert pd.Timestamp(row["eval_start"]) == nem_midnight_utc(str(HOLD))

    # The stored bounds rescore to the stored metric on holdout_main's common mask.
    directory = cfg.data_root() / row["artifact_path"]
    times, lo = read_wide(directory / "cqr_lgbm_C__lo80.parquet")
    _, hi = read_wide(directory / "cqr_lgbm_C__hi80.parquet")
    _, actual = read_wide(directory / "actual.parquet")
    assert times.equals(run.times)
    point = read_wide(
        cfg.data_path(cfg.paths.backtest_dir)
        / f"demand__lgbm_C__{inputs.holdout_point.batch_id}.parquet"
    )[1]
    mask = np.isfinite(actual) & np.isfinite(point)
    pooled, _n = summarise(lo, hi, actual, mask, 80)
    assert np.isclose(pooled["coverage"], stored_cov, rtol=1e-9)

    with pytest.raises(HoldoutIntervalsRefused, match="already scored"):
        preflight(hworld, cfg, "VIC1", freeze)


def test_a_changed_development_bound_stops_the_run(hworld: Engine, hstored: AppConfig) -> None:
    cfg = hstored
    inputs = preflight(hworld, cfg, "VIC1", freeze_of(cfg))
    run = inputs.dev_runs["cqr_lgbm_C"]
    path = cfg.data_root() / run.directory / "cqr_lgbm_C__hi80.parquet"
    frame = pd.read_parquet(path)
    values = frame["h003"].to_numpy(dtype="float64", copy=True)
    values[int(np.flatnonzero(np.isfinite(values))[-1])] += 1.0
    frame["h003"] = values.astype("float32")
    frame.to_parquet(path, index=False)
    with pytest.raises(HoldoutIntervalsReproductionFailed, match=r"cqr_lgbm_C \(bounds"):
        run_holdout_intervals(hworld, cfg, "VIC1", inputs, workers=2)
    assert _holdout_runs(hworld) == 0


def test_a_changed_qr_overlap_stops_the_run(hworld: Engine, hstored: AppConfig) -> None:
    cfg = hstored
    inputs = preflight(hworld, cfg, "VIC1", freeze_of(cfg))
    path = inputs.qr_holdout.directory / f"{QR_RUN_NAME}__q100.parquet"
    frame = pd.read_parquet(path)
    frame.loc[3, "h010"] = np.float32(frame.loc[3, "h010"] + 1.0)
    frame.to_parquet(path, index=False)
    with pytest.raises(HoldoutIntervalsReproductionFailed, match="q100 quantiles"):
        run_holdout_intervals(hworld, cfg, "VIC1", inputs, workers=2)
    assert _holdout_runs(hworld) == 0


# --------------------------------------------------------------------------- the report (6c)

AEMO_FIRST = dt.date(2021, 12, 15)


def _actual_30(cfg: AppConfig, directories: list[Path]) -> pd.Series:
    """Actual 30-minute means by period end, from half-hour batches' actuals."""
    parts = []
    for d in directories:
        times, y = read_wide(d / "actual.parquet")
        naive = times.tz_localize(None).to_numpy()
        ends = naive[:, None] + (np.arange(1, y.shape[1] + 1) * 30).astype("timedelta64[m]")
        parts.append(pd.Series(y.ravel(), index=pd.DatetimeIndex(ends.ravel(), tz="UTC")))
    s = pd.concat(parts)
    return s[~s.index.duplicated()].dropna()


def _store_aemo(db: Engine, cfg: AppConfig, actual: pd.Series) -> None:
    """PDPASA bands and pre-dispatch runs every 30 min from a day before AEMO_FIRST to HEND."""
    rng = np.random.default_rng(3)
    runs = pd.date_range(
        nem_midnight_utc(str(AEMO_FIRST)) - pd.Timedelta(days=1),
        nem_midnight_utc(str(HEND)),
        freq="30min",
        inclusive="left",
    )
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


def test_holdout_report(hworld: Engine, hstored: AppConfig) -> None:
    cfg = hstored
    inputs = preflight(hworld, cfg, "VIC1", freeze_of(cfg))
    run = run_holdout_intervals(hworld, cfg, "VIC1", inputs, workers=2)
    batch, _ = store_holdout_intervals(hworld, cfg, "VIC1", run, SHA)
    dev_30 = cfg.data_root() / inputs.dev_halfhour["cqr30_lgbm_C"].directory
    hold_30 = cfg.data_path(cfg.intervals.output_dir) / str(batch) / "30min"
    _store_aemo(hworld, cfg, _actual_30(cfg, [dev_30, hold_30]))
    cfg.aemo = cfg.aemo.model_copy(update={"first_issue": AEMO_FIRST})

    winter = ((2022, 1), (2022, 2))
    report = build_holdout_report(hworld, cfg, "VIC1", LeakageOutcome(3, 0, 0), winter)
    assert "| U5 |" in report and "**PASS**" in report
    assert "U5 passes" in report
    assert report.count("**holds**") + report.count("**does not hold**") == 4
    assert report.count("| exact |") == 10  # every development run reproduced
    assert "(all equal)" in report  # every pooled number matched the stored holdout metric
    assert "## Winter, Jan 2022, Feb 2022 (pre-registered)" in report
    assert "Settings selected at the 2022-01-01 retrain" in report
    assert "## Development vs holdout" in report and " → " in report
    assert "## AEMO's POE band, holdout part" in report
    assert "| **Holdout** |" in report and "| Development |" in report
    figures = cfg.resolve(cfg.paths.reports_dir) / "figures"
    for name in HOLDOUT_FIGURES.values():
        assert (figures / name).stat().st_size > 0
    # The development report's figures are not overwritten.
    assert not (figures / "m3_coverage_by_lead.png").exists()

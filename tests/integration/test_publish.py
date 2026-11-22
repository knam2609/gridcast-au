"""M5 step 1a end to end on the test database: tiny parquet sources → ``gridcast publish``."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from gridcast.config import AppConfig, ServingSourceRun, load_config
from gridcast.database.load import ensure_regions
from gridcast.serving.publish import PublishRefused, publish
from gridcast.serving.sources import INTERVAL_COLUMNS

pytestmark = pytest.mark.integration

H = 288
SHA = "0123456789abcdef"
LEVELS = (50, 80, 95)
# Development: two NEM days across a month boundary (two chunks); holdout: one NEM day.
DEV_TIMES = pd.date_range("2023-08-30 14:00", "2023-09-01 13:30", freq="30min", tz="UTC")
HOLD_TIMES = pd.date_range("2025-09-30 14:00", "2025-10-01 13:30", freq="30min", tz="UTC")
# Price: one file across both splits (the last development day and the first holdout day).
PRICE_TIMES = pd.date_range("2025-09-29 14:00", "2025-10-01 13:30", freq="30min", tz="UTC")


def _write_wide(path: Path, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    frame = pd.DataFrame(values.astype(np.float32), columns=[f"h{h:03d}" for h in range(1, H + 1)])
    frame.insert(0, "origin_time", times.as_unit("us"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@dataclass
class Fixture:
    cfg: AppConfig
    runs: dict[str, int]  # name -> run id
    point: dict[str, np.ndarray]  # split -> float32 (n, H)
    actual: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]  # M3's common mask of the scoring batch
    bounds: dict[str, dict[str, np.ndarray]]  # split -> raw stored bounds (float32)


def _insert_run(
    db: Engine,
    batch: str,
    name: str,
    kind: str,
    target: str,
    artifact: str,
    config: dict[str, object] | None = None,
    status: str = "success",
) -> int:
    with db.begin() as conn:
        return int(
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs (batch_id, model_name, model_kind, target, region_id,
                                            config, artifact_path, status)
                    VALUES (:b, :n, :k, :t, 'VIC1', CAST(:c AS jsonb), :a, :s)
                    RETURNING run_id
                    """
                ),
                {
                    "b": batch,
                    "n": name,
                    "k": kind,
                    "t": target,
                    "c": json.dumps(config or {}),
                    "a": artifact,
                    "s": status,
                },
            ).scalar_one()
        )


def _setup(db: Engine, tmp_path: Path) -> Fixture:
    cfg = load_config()
    cfg.paths.data_root = tmp_path
    with db.begin() as conn:
        ensure_regions(conn, cfg)
    rng = np.random.default_rng(5)
    runs: dict[str, int] = {}
    point, actual, mask, bounds = {}, {}, {}, {}
    sources: dict[str, dict[str, ServingSourceRun]] = {p: {} for p in cfg.serving.sources}
    for split, times in (("dev", DEV_TIMES), ("holdout", HOLD_TIMES)):
        n = len(times)
        batch = str(uuid.uuid4())
        y = 5000 + 400 * rng.standard_normal((n, H))
        c = y + 150 * rng.standard_normal((n, H))
        persistence = y + 300 * rng.standard_normal((n, H))
        if split == "dev":  # missing pairs, as run 2's missing weather; and a scoring-only gap
            c[3, :10] = np.nan
            c[40, 100:130] = np.nan
            persistence[7, :] = np.nan
        for model, values in (("actual", y), ("lgbm_C", c), ("persistence", persistence)):
            _write_wide(tmp_path / "backtests" / f"demand__{model}__{batch}.parquet", times, values)
        artifact = f"backtests/demand__lgbm_C__{batch}.parquet"
        runs[f"point_{split}"] = _insert_run(db, batch, "lgbm_C", "ml", "demand", artifact)
        sources["demand_point"][split] = ServingSourceRun(
            run_id=runs[f"point_{split}"], batch_id=batch, model_name="lgbm_C"
        )
        c32 = c.astype(np.float32)
        point[split] = c32
        actual[split] = y.astype(np.float32).astype(np.float64)
        mask[split] = np.isfinite(c32) & np.isfinite(persistence) & np.isfinite(y)

        # CQR bounds that sometimes cross between levels; NaN where C is missing.
        ib = str(uuid.uuid4())
        directory = Path("intervals") / ib / ("5min" if split == "holdout" else "")
        raw = {}
        for lv, w in zip(LEVELS, (120.0, 250.0, 400.0), strict=True):
            raw[f"lo{lv}"] = c32 - w + 80 * rng.standard_normal((n, H))
            raw[f"hi{lv}"] = c32 + w + 80 * rng.standard_normal((n, H))
        for column, values in raw.items():
            _write_wide(tmp_path / directory / f"cqr_lgbm_C__{column}.parquet", times, values)
        bounds[split] = {k: v.astype(np.float32) for k, v in raw.items()}
        key = "source_batch" if split == "dev" else "holdout_point_batch"
        config = {key: batch, "point_model": "lgbm_C", "interval_config_hash": "d342"}
        runs[f"interval_{split}"] = _insert_run(
            db, ib, "cqr_lgbm_C", "interval", "demand", str(directory).rstrip("/"), config
        )
        sources["demand_interval"][split] = ServingSourceRun(
            run_id=runs[f"interval_{split}"], batch_id=ib, model_name="cqr_lgbm_C"
        )

    price_batch = str(uuid.uuid4())
    for model, product in (
        ("persistence", "price_persistence"),
        ("median_profile", "price_median_profile"),
    ):
        values = 80 + 50 * rng.standard_normal((len(PRICE_TIMES), H))
        artifact = f"backtests/price__{model}__{price_batch}.parquet"
        _write_wide(tmp_path / artifact, PRICE_TIMES, values)
        runs[product] = _insert_run(db, price_batch, model, "baseline", "price", artifact)
        for split in ("dev", "holdout"):
            sources[product][split] = ServingSourceRun(
                run_id=runs[product], batch_id=price_batch, model_name=model
            )
    cfg.serving.sources = sources
    return Fixture(cfg, runs, point, actual, mask, bounds)


def _count(db: Engine, sql: str, **params: object) -> int:
    with db.connect() as conn:
        return int(conn.execute(text(sql), params).scalar_one())


def _sample(db: Engine, fx: Fixture) -> None:
    """A 288-row sample of the last development issue time, as M2's store wrote it."""
    t = DEV_TIMES[-1]
    rows = [
        {
            "r": fx.runs["point_dev"],
            "t": t.to_pydatetime(),
            "te": (t + pd.Timedelta(minutes=5 * h)).to_pydatetime(),
            "h": h,
            "p": float(fx.point["dev"][-1, h - 1]),
        }
        for h in range(1, H + 1)
        if np.isfinite(fx.point["dev"][-1, h - 1])
    ]
    with db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO forecasts (run_id, region_id, issue_time, target_end, horizon, point)"
                " VALUES (:r, 'VIC1', :t, :te, :h, :p)"
            ),
            rows,
        )


def test_publish_end_to_end(db: Engine, tmp_path: Path) -> None:
    fx = _setup(db, tmp_path)
    _sample(db, fx)
    results = publish(db, fx.cfg, check_only=False, sha=SHA)
    assert [r.status for r in results] == ["success"] * 8
    by = {r.label: r for r in results}

    dev_missing = int(np.isnan(fx.point["dev"]).sum())
    assert dev_missing == 40
    assert by["demand_point/dev"].missing_pairs == dev_missing
    assert by["demand_point/dev"].before.rows_read == H  # the sample, found equal
    assert by["demand_point/dev"].rows_inserted == len(DEV_TIMES) * H - dev_missing - H
    assert by["demand_interval/dev"].missing_pairs == dev_missing
    assert by["demand_point/holdout"].missing_pairs == 0
    assert by["price_persistence/dev"].issue_times == 48
    assert by["price_persistence/holdout"].issue_times == 48
    assert by["price_persistence/dev"].first_issue == PRICE_TIMES[0]
    assert all(r.p1 for r in results)

    assert _count(
        db, "SELECT count(*) FROM forecasts WHERE run_id = :r", r=fx.runs["point_dev"]
    ) == (len(DEV_TIMES) * H - dev_missing)
    assert (
        _count(
            db, "SELECT count(*) FROM forecasts WHERE run_id = :r", r=fx.runs["price_persistence"]
        )
        == len(PRICE_TIMES) * H
    )
    assert _count(db, "SELECT count(*) FROM publications WHERE status = 'success'") == 8

    # The published interval runs: rearranged, nested, with M3's metrics stored.
    with db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT run_id, status, config FROM model_runs "
                "WHERE model_name = 'cqr_lgbm_C_rearranged' ORDER BY run_id"
            )
        ).all()
    assert [r.status for r in rows] == ["success", "success"]
    published = {r.config["split"]: int(r.run_id) for r in rows}
    assert rows[0].config["source_run_id"] == fx.runs["interval_dev"]
    assert by["demand_interval/dev"].sql["p4"] == {
        "pairs": len(DEV_TIMES) * H - dev_missing,
        "non_nested": 0,
    }
    assert by["demand_interval/dev"].extra["source"]["non_nested_before"] > 0
    for split in ("dev", "holdout"):
        raw = fx.bounds[split]
        stacked = np.sort(np.stack([raw[c].astype(np.float64) for c in INTERVAL_COLUMNS]), axis=0)
        lo80, hi80 = stacked[1], stacked[4]  # rearranged: lo95 lo80 lo50 hi50 hi80 hi95
        y, m = fx.actual[split], fx.mask[split]
        expected = float(((lo80 <= y) & (y <= hi80))[m].mean())
        with db.connect() as conn:
            value, n = conn.execute(
                text(
                    "SELECT value, n FROM forecast_metrics WHERE run_id = :r AND split = :s "
                    "AND horizon = 0 AND metric = 'coverage_80'"
                ),
                {"r": published[split], "s": split},
            ).one()
        assert n == int(m.sum())
        assert value == pytest.approx(expected, rel=1e-12)
        with db.connect() as conn:
            stored = conn.execute(
                text(
                    "SELECT issue_time, horizon, lo80, hi80 FROM forecast_intervals "
                    "WHERE run_id = :r ORDER BY issue_time, horizon LIMIT 1"
                ),
                {"r": published[split]},
            ).one()
        assert np.float32(stored.lo80) == np.float32(lo80[0, 0])
        assert np.float32(stored.hi80) == np.float32(hi80[0, 0])

    # A rerun inserts nothing and checks again; check-only writes nothing.
    again = publish(db, fx.cfg, check_only=False, sha=SHA)
    assert [r.status for r in again] == ["success"] * 8
    assert sum(r.rows_inserted for r in again) == 0
    assert {r.metrics["stored"] for r in again if r.metrics} == {"equal"}
    checked = publish(db, fx.cfg, check_only=True, sha=f"{SHA}-dirty")
    assert [r.status for r in checked] == ["complete"] * 8
    assert _count(db, "SELECT count(*) FROM publications") == 16


def test_tampering_is_caught_and_publish_refuses(db: Engine, tmp_path: Path) -> None:
    fx = _setup(db, tmp_path)
    one = {"products": ("demand_point",), "splits": ("holdout",)}
    publish(db, fx.cfg, check_only=False, sha=SHA, **one)
    run = fx.runs["point_holdout"]
    with db.begin() as conn:
        t, h, p = conn.execute(
            text(
                "SELECT issue_time, horizon, point FROM forecasts WHERE run_id = :r "
                "ORDER BY issue_time, horizon OFFSET 100 LIMIT 1"
            ),
            {"r": run},
        ).one()
        conn.execute(
            text(
                "UPDATE forecasts SET point = :p WHERE run_id = :r AND issue_time = :t "
                "AND horizon = :h"
            ),
            {"p": float(np.nextafter(p, np.inf)), "r": run, "t": t, "h": h},
        )
    (res,) = publish(db, fx.cfg, check_only=True, sha=SHA, **one)
    assert res.status == "mismatch" and res.after is not None
    assert res.after.value_mismatches == 1
    pubs = _count(db, "SELECT count(*) FROM publications")
    with pytest.raises(PublishRefused, match="differ"):
        publish(db, fx.cfg, check_only=False, sha=SHA, **one)
    assert _count(db, "SELECT count(*) FROM publications") == pubs  # nothing written

    # Restore the value, delete a row: check-only reports it missing, publish restores it.
    with db.begin() as conn:
        conn.execute(
            text(
                "UPDATE forecasts SET point = :p WHERE run_id = :r AND issue_time = :t "
                "AND horizon = :h"
            ),
            {"p": float(p), "r": run, "t": t, "h": h},
        )
        conn.execute(
            text("DELETE FROM forecasts WHERE run_id = :r AND issue_time = :t AND horizon = :h"),
            {"r": run, "t": t, "h": h},
        )
    (res,) = publish(db, fx.cfg, check_only=True, sha=SHA, **one)
    assert res.status == "incomplete" and res.after is not None
    assert res.after.missing_rows == 1 and res.after.exact
    (res,) = publish(db, fx.cfg, check_only=False, sha=SHA, **one)
    assert res.status == "success" and res.rows_inserted == 1


def test_refusals_write_nothing(db: Engine, tmp_path: Path) -> None:
    fx = _setup(db, tmp_path)
    with pytest.raises(PublishRefused, match="clean git tree"):
        publish(db, fx.cfg, check_only=False, sha=f"{SHA}-dirty")
    good = fx.cfg.serving.sources["demand_point"]["dev"]
    fx.cfg.serving.sources["demand_point"]["dev"] = good.model_copy(update={"batch_id": "x"})
    with pytest.raises(PublishRefused, match="batch_id is"):
        publish(db, fx.cfg, check_only=False, sha=SHA)
    fx.cfg.serving.sources["demand_point"]["dev"] = good
    with db.begin() as conn:
        conn.execute(
            text("UPDATE model_runs SET status = 'failed' WHERE run_id = :r"),
            {"r": fx.runs["price_persistence"]},
        )
        conn.execute(
            text(
                "UPDATE model_runs SET config = jsonb_set(config, '{source_batch}', '\"other\"') "
                "WHERE run_id = :r"
            ),
            {"r": fx.runs["interval_dev"]},
        )
    with pytest.raises(PublishRefused) as exc:
        publish(db, fx.cfg, check_only=True, sha=SHA)
    assert "status is 'failed'" in str(exc.value)
    assert "scored on point batch 'other'" in str(exc.value)
    assert _count(db, "SELECT count(*) FROM publications") == 0
    assert _count(db, "SELECT count(*) FROM forecasts") == 0


def test_interval_table_rejects_crossed_and_nan_bounds(db: Engine, tmp_path: Path) -> None:
    fx = _setup(db, tmp_path)
    run = fx.runs["interval_dev"]
    insert = text(
        "INSERT INTO forecast_intervals (run_id, region_id, issue_time, target_end, horizon, "
        "lo95, lo80, lo50, hi50, hi80, hi95) VALUES (:r, 'VIC1', :t, :te, 1, "
        ":lo95, :lo80, :lo50, :hi50, :hi80, :hi95)"
    )
    t = DEV_TIMES[0].to_pydatetime()
    te = (DEV_TIMES[0] + pd.Timedelta(minutes=5)).to_pydatetime()
    ok = {"lo95": 1.0, "lo80": 2.0, "lo50": 3.0, "hi50": 4.0, "hi80": 5.0, "hi95": 6.0}
    for bad in ({"lo80": 3.5}, {"hi95": float("nan")}, {"lo50": float("nan")}):
        with pytest.raises(IntegrityError), db.begin() as conn:
            conn.execute(insert, {"r": run, "t": t, "te": te, **ok, **bad})
    with db.begin() as conn:
        conn.execute(insert, {"r": run, "t": t, "te": te, **ok})

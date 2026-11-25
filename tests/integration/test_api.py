"""M5 step 2 on the test database: the stored endpoints against what was published."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from gridcast.api.app import create_app
from gridcast.api.catalogue import read_only_engine
from gridcast.config import ServingSourceRun
from gridcast.database.load import start_ingestion_run, upsert_market_intervals
from gridcast.serving.publish import publish
from gridcast.serving.sources import INTERVAL_COLUMNS
from tests.integration.factories import clean_month
from tests.integration.test_publish import (
    DEV_TIMES,
    HOLD_TIMES,
    PRICE_TIMES,
    SHA,
    Fixture,
    H,
    _insert_run,
    _setup,
)

pytestmark = pytest.mark.integration

OCT = pd.Timestamp("2021-10-01", tz="UTC")


@dataclass
class Env:
    fx: Fixture
    client: TestClient
    db: Engine
    events: dict[int, str]  # served event id -> split
    v1_event: int
    data_root: Path


def _events(db: Engine, fx: Fixture) -> tuple[dict[int, str], int]:
    """v2 runs for both splits with a few events each, and a v1 run that is not served."""
    frozen = fx.cfg.anomalies_v2.frozen_config_hash
    runs = {}
    for split, version in (("dev", 2), ("holdout", 2), ("v1", 1)):
        config = {
            "recipe_version": version,
            "anomaly_config_hash": frozen if version == 2 else "v1hash",
            "split": "holdout" if split == "v1" else split,
        }
        runs[split] = _insert_run(
            db, str(uuid.uuid4()), "anomalies", "anomaly", "market", "/x", config
        )
    fx.cfg.serving.anomaly_runs = {
        s: ServingSourceRun(run_id=runs[s], batch_id=_batch(db, runs[s]), model_name="anomalies")
        for s in ("dev", "holdout")
    }
    rows = []
    base = {"dev": pd.Timestamp("2023-09-01 02:00", tz="UTC"), "holdout": HOLD_TIMES[0]}
    for split, n in (("dev", 5), ("holdout", 3), ("v1", 1)):
        start = base["holdout" if split == "v1" else split]
        for k in range(n):
            t = start + pd.Timedelta(minutes=35 * k)
            rows.append(
                {
                    "run": runs[split],
                    "split": "holdout" if split == "v1" else split,
                    "t": t.to_pydatetime(),
                    "peak": (t + pd.Timedelta(minutes=5)).to_pydatetime(),
                    "last": (t + pd.Timedelta(minutes=10)).to_pydatetime(),
                    "tier": "severe" if k % 3 == 0 else "notable",
                    "det": ["price_level", "price_line_300"] if k % 2 else ["robust_distance"],
                    "cat": ["high price"] if k % 2 else ["multivariate"],
                    "p": None if k == 4 else 1e-4 * (k + 1),
                    "ev": json.dumps({"k": k, "values": [1.5, -0.25], "note": "ünïcode"}),
                    "s": f"Event {k} ({split})",
                }
            )
    with db.begin() as conn:
        ids = [
            int(
                conn.execute(
                    text(
                        """
                        INSERT INTO anomalies (run_id, region_id, split, first_interval_end,
                            last_interval_end, peak_interval_end, tier, detectors, min_p,
                            categories, evidence, summary)
                        VALUES (:run, 'VIC1', :split, :t, :last, :peak, :tier, :det, :p, :cat,
                                CAST(:ev AS jsonb), :s)
                        RETURNING event_id
                        """
                    ),
                    r,
                ).scalar_one()
            )
            for r in rows
        ]
    served = {i: r["split"] for i, r in zip(ids, rows, strict=True) if r["run"] != runs["v1"]}
    v1 = next(i for i, r in zip(ids, rows, strict=True) if r["run"] == runs["v1"])
    return served, v1


def _batch(db: Engine, run_id: int) -> str:
    with db.connect() as conn:
        return str(
            conn.execute(
                text("SELECT batch_id::text FROM model_runs WHERE run_id = :r"), {"r": run_id}
            ).scalar_one()
        )


@pytest.fixture
def env(db: Engine, tmp_path: Path) -> Iterator[Env]:
    fx = _setup(db, tmp_path)
    results = publish(db, fx.cfg, check_only=False, sha=SHA)
    assert {r.status for r in results} == {"success"}
    with db.begin() as conn:
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", OCT.date())
        upsert_market_intervals(conn, clean_month(), run)
    events, v1 = _events(db, fx)
    engine = read_only_engine(db.url.render_as_string(hide_password=False))
    with TestClient(create_app(fx.cfg, engine), raise_server_exceptions=False) as client:
        yield Env(fx, client, db, events, v1, tmp_path)
    engine.dispose()


def _rearranged(fx: Fixture, split: str) -> np.ndarray:
    raw = fx.bounds[split]
    return np.sort(np.stack([raw[c].astype(np.float64) for c in INTERVAL_COLUMNS]), axis=0)


def _f32(value: float | None) -> np.float32 | None:
    return None if value is None else np.float32(value)


def test_health_and_meta(env: Env) -> None:
    h = env.client.get("/health")
    # The stored endpoints work; recompute and explain cannot (the fixture's batches are not
    # the stored fits'), so the service is degraded and says why.
    assert h.status_code == 200 and h.json()["status"] == "degraded", h.json()
    checks = h.json()["checks"]
    assert all(checks[k]["ok"] for k in ("database", "publications", "anomaly_runs"))
    assert not checks["feature_context"]["ok"] and not checks["model_files"]["ok"]
    assert "serving publishes lgbm_C of batch" in checks["feature_context"]["detail"]
    r = env.client.get("/v1/forecasts/demand/recompute", params={"issue_time": "2023-08-31T00:00Z"})
    assert r.status_code == 503 and "whose fits are loaded" in r.json()["detail"]
    meta = env.client.get("/v1/meta").json()
    products = {p["product"]: p for p in meta["products"]}
    assert set(products) == {
        "demand_point",
        "demand_interval",
        "price_persistence",
        "price_median_profile",
    }
    dev_interval = next(s for s in products["demand_interval"]["splits"] if s["split"] == "dev")
    assert dev_interval["source_run"]["run_id"] == env.fx.runs["interval_dev"]
    assert dev_interval["missing_pairs"] == 40
    assert "coverage_80" in dev_interval["metrics"]["values"]
    assert dev_interval["metrics"]["n"] == int(env.fx.mask["dev"].sum())
    assert [a["events"] for a in meta["anomalies"]] == [5, 3]
    assert len(meta["disclosures"]) == 5 and meta["leads"] == H

    # A later failed attempt makes the stored endpoints' health "unavailable" (503).
    with env.db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO publications (product, split, region_id, run_id, source_run_id, "
                "source_files, status) VALUES ('price_persistence', 'dev', 'VIC1', :r, :r, "
                "'[]', 'failed')"
            ),
            {"r": env.fx.runs["price_persistence"]},
        )
    h = env.client.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "unavailable"
    assert "price_persistence/dev: failed" in h.json()["checks"]["publications"]["detail"]


def test_demand_forecasts_equal_the_sources(env: Env) -> None:
    for split, times in (("dev", DEV_TIMES), ("holdout", HOLD_TIMES)):
        point, bounds = env.fx.point[split], _rearranged(env.fx, split)
        for i, t in enumerate(times):
            r = env.client.get("/v1/forecasts/demand", params={"issue_time": t.isoformat()})
            assert r.status_code == 200, r.json()
            body = r.json()
            assert body["split"] == split and len(body["leads"]) == H
            assert body["point_run_id"] == env.fx.runs[f"point_{split}"]
            for h, lead in enumerate(body["leads"]):
                assert pd.Timestamp(lead["target_end"]) == t + pd.Timedelta(minutes=5 * (h + 1))
                if np.isnan(point[i, h]):
                    assert lead["point_mw"] is None and lead["lo80_mw"] is None
                    continue
                assert lead["point_mw"] == float(point[i, h])  # exact float64 of the float32
                for k, c in enumerate(INTERVAL_COLUMNS):
                    assert lead[f"{c}_mw"] == float(np.float32(bounds[k, i, h]))


def test_evolution_collects_every_issue_time(env: Env) -> None:
    dev = env.fx.point["dev"]
    for target in (DEV_TIMES[60] + pd.Timedelta(minutes=35), DEV_TIMES[3] + pd.Timedelta(hours=2)):
        r = env.client.get(
            "/v1/forecasts/demand/evolution", params={"target_end": target.isoformat()}
        )
        assert r.status_code == 200, r.json()
        items = r.json()["items"]
        lo = target - pd.Timedelta(hours=24)
        expected = [t for t in DEV_TIMES if lo <= t < target]
        assert [pd.Timestamp(x["issue_time"]) for x in items] == expected
        for x in items:
            i = DEV_TIMES.get_loc(pd.Timestamp(x["issue_time"]))
            h = x["lead"] - 1
            assert (
                x["lead"] * 5
                == x["lead_minutes"]
                == (target - DEV_TIMES[i]) / pd.Timedelta(minutes=1)
            )
            value = dev[i, h]
            assert x["point_mw"] == (None if np.isnan(value) else float(value))
    r = env.client.get(
        "/v1/forecasts/demand/evolution", params={"target_end": "2024-06-01T00:00:00Z"}
    )
    assert r.status_code == 404


def test_price_baselines_equal_the_sources(env: Env) -> None:
    files = {
        m: pd.read_parquet(next((env.data_root / "backtests").glob(f"price__{m}__*.parquet")))
        for m in ("persistence", "median_profile")
    }
    for i in (0, 47, 48, len(PRICE_TIMES) - 1):
        t = PRICE_TIMES[i]
        body = env.client.get("/v1/forecasts/price", params={"issue_time": t.isoformat()}).json()
        assert body["baselines"] is True and body["split"] == ("dev" if i < 48 else "holdout")
        for model, frame in files.items():
            row = frame.iloc[i, 1:].to_numpy(np.float32)
            served = np.array([x[f"{model}_aud_mwh"] for x in body["leads"]], np.float64)
            assert np.array_equal(served, row.astype(np.float64))


def test_actuals_equal_market_intervals(env: Env) -> None:
    start, end = "2021-10-05T00:00:00+11:00", "2021-10-06T00:00:00+11:00"
    body = env.client.get("/v1/actuals", params={"start": start, "end": end}).json()
    assert len(body["intervals"]) == 288
    with env.db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT interval_end, total_demand_mw, rrp_aud_mwh, apc_flag_bits FROM "
                "market_intervals WHERE interval_end > :a AND interval_end <= :b "
                "ORDER BY interval_end"
            ),
            {"a": pd.Timestamp(start).to_pydatetime(), "b": pd.Timestamp(end).to_pydatetime()},
        ).all()
    for served, row in zip(body["intervals"], rows, strict=True):
        assert pd.Timestamp(served["interval_end"]) == pd.Timestamp(row.interval_end)
        assert served["demand_mw"] == row.total_demand_mw
        assert served["price_aud_mwh"] == row.rrp_aud_mwh
        assert served["apc_flag_bits"] == row.apc_flag_bits
    r = env.client.get(
        "/v1/actuals", params={"start": "2030-01-01T00:00:00Z", "end": "2030-01-02T00:00:00Z"}
    )
    assert r.status_code == 404


def test_anomalies_page_through_v2_only(env: Env) -> None:
    window = {"start": "2023-01-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    seen, after, pages = [], None, 0
    while True:
        params = {**window, "limit": "2", **({"after": after} if after else {})}
        body = env.client.get("/v1/anomalies", params=params).json()
        seen += body["events"]
        pages += 1
        after = body["next"]
        if after is None:
            break
    assert pages == 4 and sorted(e["event_id"] for e in seen) == sorted(env.events)
    keys = [(e["first_interval_end"], e["event_id"]) for e in seen]
    assert keys == sorted(keys)
    assert env.v1_event not in {e["event_id"] for e in seen}

    severe = env.client.get("/v1/anomalies", params={**window, "tier": "severe"}).json()
    assert {e["tier"] for e in severe["events"]} == {"severe"}
    by_det = env.client.get("/v1/anomalies", params={**window, "detector": "price_line_300"}).json()
    assert all("price_line_300" in e["detectors"] for e in by_det["events"])
    assert by_det["events"]

    with env.db.connect() as conn:
        for event_id, split in env.events.items():
            body = env.client.get(f"/v1/anomalies/{event_id}").json()
            row = conn.execute(
                text("SELECT * FROM anomalies WHERE event_id = :e"), {"e": event_id}
            ).one()
            assert body["split"] == split == row.split
            assert body["evidence"] == row.evidence
            assert body["summary"] == row.summary and body["min_p"] == row.min_p
            assert body["detectors"] == list(row.detectors)
            assert pd.Timestamp(body["peak_interval_end"]) == pd.Timestamp(row.peak_interval_end)
    assert env.client.get(f"/v1/anomalies/{env.v1_event}").status_code == 404


@pytest.mark.parametrize(
    ("path", "params", "status"),
    [
        ("/v1/forecasts/demand", {"issue_time": "2024-06-01T00:00:00Z"}, 404),
        ("/v1/forecasts/demand", {"issue_time": "2023-08-30T14:00:00Z", "region": "NSW1"}, 404),
        ("/v1/forecasts/price", {"issue_time": "2020-01-01T00:00:00Z"}, 404),
        ("/v1/anomalies/123456789", {}, 404),
        ("/v1/forecasts/demand", {"issue_time": "2023-08-30T14:00:00"}, 422),
    ],
)
def test_valid_but_unpublished_inputs(
    env: Env, path: str, params: dict[str, str], status: int
) -> None:
    r = env.client.get(path, params=params)
    assert r.status_code == status and isinstance(r.json()["detail"], str)


TEXT = st.one_of(
    st.text(max_size=40),
    st.sampled_from(
        [
            "2023-08-30T14:00:00Z",
            "2025-09-30T14:00:00+00:00",
            "9999-12-31T23:30:00-23:59",
            "0001-01-01T00:00:00+14:00",
            "2023-08-30T14:00:00\x00",
            "2023-08-31T00:00:00 10:00",
        ]
    ),
)


@given(
    path=st.sampled_from(
        [
            ("/v1/forecasts/demand", "issue_time"),
            ("/v1/forecasts/demand/evolution", "target_end"),
            ("/v1/forecasts/price", "issue_time"),
            ("/v1/actuals", "start"),
            ("/v1/actuals", "end"),
            ("/v1/anomalies", "start"),
            ("/v1/anomalies", "category"),
            ("/v1/anomalies", "after"),
            ("/v1/anomalies", "limit"),
            ("/v1/forecasts/demand", "region"),
            ("/v1/forecasts/demand/recompute", "issue_time"),
            ("/v1/forecasts/demand/explain", "issue_time"),
            ("/v1/forecasts/demand/explain", "lead"),
        ]
    ),
    value=TEXT,
)
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_no_query_produces_a_500(env: Env, path: tuple[str, str], value: str) -> None:
    route, name = path
    params = {
        "/v1/actuals": {"start": "2021-10-05T00:00:00Z", "end": "2021-10-06T00:00:00Z"},
        "/v1/anomalies": {"start": "2023-01-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
    }.get(route, {})
    if "/recompute" in route or "/explain" in route:  # the fixture has no stored fits: 503
        params = {"issue_time": "2023-08-31T00:00:00Z"}
    r = env.client.get(route, params={**params, name: value})
    allowed = (404, 422, 503) if "/recompute" in route or "/explain" in route else (200, 404, 422)
    assert r.status_code in allowed, (route, name, value, r.text)
    assert r.status_code == 200 or isinstance(r.json()["detail"], str)


def test_the_api_engine_cannot_write(env: Env) -> None:
    engine = read_only_engine(env.db.url.render_as_string(hide_password=False))
    with pytest.raises(DBAPIError, match="read-only"), engine.begin() as conn:
        conn.execute(text("DELETE FROM anomalies"))
    engine.dispose()

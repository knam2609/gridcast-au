"""M5 step 3 on the host: recompute and explain on the real stored models (read-only).

Runs only with ``make test-host`` (marker ``host``): it reads the main database and the
model files under the data root (the SSD), and writes nothing. It is informal: P3 is judged
on the Compose service in step 5 (m5_plan.md §3). The issue times were chosen before these
tests ran (step 3 question round, 2026-11-25): one inside a fold of each split, both sides of
a retrain boundary in each split, both sides of the split boundary (the ``2025-08-01`` fits
are in both batches), an issue time with nothing stored and one partly stored.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from gridcast.api.app import create_app
from gridcast.api.catalogue import read_only_engine
from gridcast.api.runtime import split_fits
from gridcast.config import get_settings, load_config
from gridcast.explainability.stored import stored_fits

pytestmark = pytest.mark.host

DEMAND = "/v1/forecasts/demand"
DEV_BATCH = "6d783932-5834-4e18-a9e8-0e3567db46e6"
HOLDOUT_BATCH = "fcb6abb6-49a3-4923-9d79-64e05a44d75f"
# (issue time, split, retrain of its fits)
FULL = [
    ("2024-03-15T02:00:00Z", "dev", "2024-02-01"),
    ("2024-04-30T13:30:00Z", "dev", "2024-02-01"),  # the last issue time of a fold
    ("2024-04-30T14:00:00Z", "dev", "2024-05-01"),  # the first of the next
    ("2025-09-30T13:30:00Z", "dev", "2025-08-01"),  # run 2's 2025-08-01 files
    ("2025-09-30T14:00:00Z", "holdout", "2025-08-01"),  # holdout_main's 2025-08-01 files
    ("2026-01-20T05:30:00Z", "holdout", "2025-11-01"),
    ("2026-01-31T13:30:00Z", "holdout", "2025-11-01"),
    ("2026-01-31T14:00:00Z", "holdout", "2026-02-01"),
]
NOTHING_STORED = "2024-01-15T02:00:00Z"  # January 2024: missing weather


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    cfg = load_config()
    if not cfg.resolve(cfg.paths.data_root).is_dir():
        pytest.skip("the data root is not mounted")
    with TestClient(create_app(cfg), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def db() -> Iterator[Engine]:
    engine = read_only_engine(get_settings().database_url)
    yield engine
    engine.dispose()


def _get(client: TestClient, path: str, **params: Any) -> dict[str, Any]:
    r = client.get(f"{DEMAND}/{path}", params=params)
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def test_health_has_the_feature_context_and_every_model_file(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok", h
    assert "68 fits (dev 48, holdout 20)" in h["checks"]["model_files"]["detail"]


def test_the_api_fits_are_m4s(db: Engine) -> None:
    cfg = load_config()
    for split in ("dev", "holdout"):
        assert split_fits(db, cfg, split) == stored_fits(db, cfg, split)


@pytest.mark.parametrize(("issue_time", "split", "retrain"), FULL)
def test_recompute_matches_and_names_the_right_batch(
    client: TestClient, issue_time: str, split: str, retrain: str
) -> None:
    b = _get(client, "recompute", issue_time=issue_time)
    assert b["split"] == split
    assert b["matches_stored"], {k: v for k, v in b.items() if k != "leads"}
    assert b["leads_stored"] == b["leads_recomputed"] == 288
    batch = DEV_BATCH if split == "dev" else HOLDOUT_BATCH
    assert {f["retrain"] for f in b["fits"]} == {retrain}
    assert all(f["batch_id"] == batch and batch in f["model_file"] for f in b["fits"])


@pytest.mark.parametrize("issue_time", [FULL[0][0], FULL[4][0]])
def test_explain_passes_e1_and_adds_up(client: TestClient, issue_time: str) -> None:
    b = _get(client, "explain", issue_time=issue_time)
    assert b["gate"]["passed"] and b["gate"]["n"] == 288 and len(b["leads"]) == 288
    for x in b["leads"]:
        total = x["anchor"]["mw"] + x["base_value_mw"]
        total += sum(f["contribution_mw"] for f in x["features"])
        assert abs(total - x["forecast_float64_mw"]) < 1e-6
        assert len(x["features"]) == 26 and len(x["families"]) == 6


def test_an_issue_time_with_nothing_stored(client: TestClient) -> None:
    b = _get(client, "recompute", issue_time=NOTHING_STORED)
    assert b["matches_stored"] and b["leads_stored"] == b["leads_recomputed"] == 0
    r = client.get(f"{DEMAND}/explain", params={"issue_time": NOTHING_STORED})
    assert r.status_code == 404


def test_a_partly_stored_issue_time(client: TestClient, db: Engine) -> None:
    with db.connect() as conn:
        row = conn.execute(
            text(
                "SELECT issue_time, count(*) AS n FROM forecasts WHERE run_id = 39 "
                "AND issue_time >= '2023-07-31 14:00+00' "
                "GROUP BY issue_time HAVING count(*) < 288 ORDER BY issue_time LIMIT 1"
            )
        ).one()
    t = row.issue_time.isoformat()
    b = _get(client, "recompute", issue_time=t)
    assert b["matches_stored"] and b["leads_stored"] == b["leads_recomputed"] == row.n
    e = _get(client, "explain", issue_time=t)
    assert e["gate"]["passed"] and len(e["leads"]) == row.n
    assert len(e["leads_without_forecast"]) == 288 - row.n

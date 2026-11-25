"""M5 step 2: the API's inputs, contract and float exactness, without a database."""

from __future__ import annotations

import contextlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.api import models as m
from gridcast.api.app import create_app
from gridcast.api.catalogue import read_only_engine
from gridcast.api.inputs import (
    InputError,
    check_grid,
    check_range,
    decode_cursor,
    encode_cursor,
    parse_instant,
)
from gridcast.api.meta import DISCLOSURES, LICENCES
from gridcast.api.queries import real
from gridcast.api.schema import SCHEMA_PATH, render_schema
from gridcast.config import PROJECT_ROOT

UTC = pd.Timestamp("2025-09-30 14:00", tz="UTC")


@pytest.mark.parametrize(
    "text",
    [
        "2025-09-30T14:00:00Z",
        "2025-09-30T14:00:00+00:00",
        "2025-10-01T00:00:00+10:00",
        "2025-10-01T01:00:00+11:00",
        "2025-09-30T23:30:00+09:30",
        "2025-10-01T00:00:00 10:00",  # a "+" decoded to a space in a query string
        "2025-09-30T14:00Z",
    ],
)
def test_instants_with_an_offset_parse_to_utc(text: str) -> None:
    assert parse_instant("t", text) == UTC


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("2025-09-30T14:00:00", "offset is required"),
        ("2025-09-30", "offset is required"),
        ("yesterday", "not an ISO 8601"),
        ("", "not an ISO 8601"),
        ("2025-13-01T00:00:00Z", "not an ISO 8601"),
        ("2025-09-30 14:00:00 10:00", "not an ISO 8601"),  # no 'T': the space is ambiguous
    ],
)
def test_bad_instants_are_input_errors(text: str, message: str) -> None:
    with pytest.raises(InputError, match=message):
        parse_instant("t", text)


@given(st.text(max_size=40))
@settings(max_examples=300)
def test_any_text_parses_or_is_an_input_error(text: str) -> None:
    with contextlib.suppress(InputError):
        parse_instant("t", text)


def test_grids() -> None:
    assert check_grid("t", UTC, 30) == UTC
    check_grid("t", UTC + pd.Timedelta(minutes=5), 5)
    for bad in (pd.Timedelta(minutes=5), pd.Timedelta(seconds=1), pd.Timedelta(microseconds=1)):
        with pytest.raises(InputError, match="30-minute grid"):
            check_grid("t", UTC + bad, 30)
    with pytest.raises(InputError, match="after start"):
        check_range(UTC, UTC, None)
    with pytest.raises(InputError, match="at most"):
        check_range(UTC, UTC + pd.Timedelta(days=7, minutes=5), pd.Timedelta(days=7))
    check_range(UTC, UTC + pd.Timedelta(days=7), pd.Timedelta(days=7))


def test_cursor_round_trip_and_garbage() -> None:
    cursor = encode_cursor(UTC, 2305)
    assert decode_cursor(cursor) == (UTC, 2305)
    assert cursor.isascii() and "=" not in cursor
    for bad in ("", "abc", "djI6MTo1", "!!!", "djE6eDox", "djE6MTow", "ü"):
        with pytest.raises(InputError, match="cursor"):
            decode_cursor(bad)


@given(st.text(max_size=60))
@settings(max_examples=300)
def test_any_text_decodes_or_is_an_input_error(text: str) -> None:
    with contextlib.suppress(InputError):
        decode_cursor(text)


def test_float32_values_survive_json_exactly() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(5000, 800, 1000).astype(np.float32)
    lead = m.DemandLead(
        lead=1,
        lead_minutes=5,
        target_end=UTC.to_pydatetime(),
        point_mw=None,
        lo95_mw=None,
        lo80_mw=None,
        lo50_mw=None,
        hi50_mw=None,
        hi80_mw=None,
        hi95_mw=None,
    )
    for v in values:
        # A bound arrives from psycopg as its shortest float32 decimal.
        shortest = float(np.format_float_positional(v, unique=True))
        served = lead.model_copy(update={"lo80_mw": real(shortest), "point_mw": float(v)})
        back = json.loads(served.model_dump_json())
        assert back["point_mw"] == float(v)  # the exact float64 of the float32
        assert back["lo80_mw"] == float(v)
        assert np.float32(back["lo80_mw"]).view(np.uint32) == v.view(np.uint32)


def test_openapi_matches_the_committed_contract() -> None:
    assert render_schema(create_app()) == SCHEMA_PATH.read_text(encoding="utf-8"), (
        "the API contract changed: review it, then run `make api-schema`"
    )


def test_every_disclosure_and_licence_is_present() -> None:
    assert [d.key for d in DISCLOSURES] == [
        "intervals_rearranged",
        "dev_forecasts_looked_at_twice",
        "anomaly_v2_post_hoc",
        "price_baselines",
        "archive_not_live",
    ]
    for d in DISCLOSURES:
        assert d.sources and all((PROJECT_ROOT / s).is_file() for s in d.sources)
        assert "%" not in d.text, "no metric values in disclosures (they live in metrics)"
    assert {lic.licence for lic in LICENCES} >= {"CC BY 4.0"}


# --------------------------------------------------------------------------- requests


@pytest.fixture
def client() -> TestClient:
    """The app without its lifespan; the engine points at a closed port."""
    engine = read_only_engine("postgresql+psycopg://nobody:x@127.0.0.1:9/none")
    return TestClient(create_app(engine=engine), raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("path", "params", "message"),
    [
        ("/v1/forecasts/demand", {}, "issue_time: Field required"),
        ("/v1/forecasts/demand", {"issue_time": "2025-10-01T00:00:00"}, "offset"),
        ("/v1/forecasts/demand", {"issue_time": "2025-10-01T00:05:00Z"}, "30-minute grid"),
        ("/v1/forecasts/demand/evolution", {"target_end": "2025-10-01T00:01:00Z"}, "5-minute"),
        ("/v1/forecasts/price", {"issue_time": "noon"}, "ISO 8601"),
        (
            "/v1/actuals",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-10-09T00:00:00Z"},
            "at most",
        ),
        (
            "/v1/actuals",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-10-01T00:00:00Z"},
            "after start",
        ),
        (
            "/v1/anomalies",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-11-01T00:00:00Z", "tier": "huge"},
            "tier",
        ),
        (
            "/v1/anomalies",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-11-01T00:00:00Z", "limit": "0"},
            "limit",
        ),
        (
            "/v1/anomalies",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-11-01T00:00:00Z", "limit": "501"},
            "limit",
        ),
        (
            "/v1/anomalies",
            {"start": "2025-10-01T00:00:00Z", "end": "2025-11-01T00:00:00Z", "after": "zz"},
            "cursor",
        ),
        ("/v1/forecasts/demand/recompute", {}, "issue_time: Field required"),
        ("/v1/forecasts/demand/recompute", {"issue_time": "2025-10-01T00:10:00Z"}, "grid"),
        (
            "/v1/forecasts/demand/explain",
            {"issue_time": "2025-10-01T00:00:00Z", "lead": "289"},
            "at most 288",
        ),
        (
            "/v1/forecasts/demand/explain",
            {"issue_time": "2025-10-01T00:00:00Z", "lead": "0"},
            "lead",
        ),
        (
            "/v1/forecasts/demand/explain",
            {"issue_time": "2025-10-01T00:00:00Z", "lead": "x"},
            "lead",
        ),
        ("/v1/anomalies/0", {}, "event_id"),
        ("/v1/anomalies/abc", {}, "event_id"),
        ("/v1/anomalies/99999999999999999999", {}, "event_id"),
    ],
)
def test_invalid_inputs_are_422_with_a_message(
    client: TestClient, path: str, params: dict[str, str], message: str
) -> None:
    r = client.get(path, params=params)
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"detail"} and isinstance(body["detail"], str)
    assert message in body["detail"]


def test_database_down_is_503_never_500(client: TestClient) -> None:
    for path in ("", "/recompute", "/explain"):
        r = client.get(f"/v1/forecasts/demand{path}", params={"issue_time": "2025-10-01T00:00:00Z"})
        assert r.status_code == 503 and r.json() == {"detail": "database unavailable"}
    h = client.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "unavailable"
    assert not h.json()["checks"]["database"]["ok"]
    context = h.json()["checks"]["feature_context"]
    assert not context["ok"] and context["detail"] == "database unavailable"
    assert client.get("/v1/meta").status_code == 503


def test_only_get_is_allowed(client: TestClient) -> None:
    assert client.post("/v1/meta").status_code == 405
    pre = client.options(
        "/v1/meta",
        headers={"Origin": "https://example.org", "Access-Control-Request-Method": "GET"},
    )
    assert pre.headers.get("access-control-allow-origin") == "*"

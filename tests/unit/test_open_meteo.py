import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from gridcast.ingestion.open_meteo import OpenMeteoClient, OpenMeteoError, OpenMeteoRequest
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

REQ = OpenMeteoRequest(
    api="previous_runs",
    model="gfs_seamless",
    site_id="melbourne",
    latitude=-37.83,
    longitude=144.98,
    hourly=("temperature_2m_previous_day1", "temperature_2m_previous_day2"),
    start_date=dt.date(2021, 9, 30),
    end_date=dt.date(2021, 10, 31),
)


def _client(fake: FakeOpenMeteo, tmp_path: Path, **cfg: object) -> OpenMeteoClient:
    return OpenMeteoClient(
        om_config(**cfg), tmp_path, transport=fake.transport(), today=lambda: TODAY
    )


def test_request_parameters_and_user_agent(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    with _client(fake, tmp_path) as client:
        resp = client.fetch(REQ)
    (request,) = fake.requests
    assert request.headers["User-Agent"] == "gridcast-test"
    assert str(request.url).startswith("https://previous-runs.example/v1/forecast?")
    assert dict(request.url.params) == {
        "latitude": "-37.8300",
        "longitude": "144.9800",
        "models": "gfs_seamless",
        "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
        "start_date": "2021-09-30",
        "end_date": "2021-10-31",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    assert not resp.from_cache
    assert resp.record()["grid_latitude"] == pytest.approx(-37.82)


def test_archive_api_url(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    req = OpenMeteoRequest(
        "archive",
        "era5",
        "melbourne",
        -37.83,
        144.98,
        ("temperature_2m",),
        REQ.start_date,
        REQ.end_date,
    )
    with _client(fake, tmp_path) as client:
        client.fetch(req)
    assert str(fake.requests[0].url).startswith("https://archive.example/v1/archive?")


def test_settled_response_is_cached_and_reused(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    with _client(fake, tmp_path) as client:
        first = client.fetch(REQ)
        second = client.fetch(REQ)
        path = client.cache_path(REQ)
    assert len(fake.requests) == 1
    assert second.from_cache and second.payload == first.payload
    assert second.sha256 == first.sha256
    assert (
        path == tmp_path / "open_meteo/previous_runs/gfs_seamless/melbourne/20210930_20211031.json"
    )
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text())
    assert meta["params"] == REQ.params()


def test_cache_ignored_when_parameters_change_or_refresh(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    with _client(fake, tmp_path) as client:
        client.fetch(REQ)
        meta_path = client.cache_path(REQ).with_name("20210930_20211031.json.meta.json")
        meta = json.loads(meta_path.read_text())
        meta["params"]["hourly"] = "temperature_2m_previous_day1"
        meta_path.write_text(json.dumps(meta))
        assert not client.fetch(REQ).from_cache  # stale parameters → re-download
        assert client.fetch(REQ).from_cache
        assert not client.fetch(REQ, refresh=True).from_cache
    assert len(fake.requests) == 3


def test_corrupt_cache_is_redownloaded(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    with _client(fake, tmp_path) as client:
        client.fetch(REQ)
        client.cache_path(REQ).write_text("{}")
        assert not client.fetch(REQ).from_cache
    assert len(fake.requests) == 2


def test_recent_range_is_not_cached(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    recent = OpenMeteoRequest(
        "archive",
        "era5",
        "melbourne",
        -37.83,
        144.98,
        ("temperature_2m",),
        TODAY - dt.timedelta(days=20),
        TODAY - dt.timedelta(days=5),
    )
    with _client(fake, tmp_path) as client:
        resp = client.fetch(recent)
        assert resp.cached_path is None
        assert not client.cache_path(recent).exists()
        client.fetch(recent)
    assert len(fake.requests) == 2


def test_overloaded_service_is_retried(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    overloaded = {"error": True, "reason": "The service is overloaded"}
    fake.responses = [httpx.Response(503, json=overloaded), httpx.Response(429, json={})]
    with _client(fake, tmp_path) as client:
        resp = client.fetch(REQ)
    assert len(fake.requests) == 3
    assert "hourly" in resp.payload


def test_retries_give_up_after_max_attempts(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    fake.responses = [httpx.Response(503, json={}) for _ in range(3)]
    with _client(fake, tmp_path) as client, pytest.raises(httpx.HTTPStatusError):
        client.fetch(REQ)
    assert len(fake.requests) == 3


def test_bad_request_raises_with_reason_without_retry(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    fake.responses = [httpx.Response(400, json={"error": True, "reason": "Cannot find model"})]
    with _client(fake, tmp_path) as client, pytest.raises(OpenMeteoError, match="Cannot find"):
        client.fetch(REQ)
    assert len(fake.requests) == 1


def test_error_payload_with_200_raises(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    fake.responses = [httpx.Response(200, json={"error": True, "reason": "boom"})]
    with _client(fake, tmp_path) as client, pytest.raises(OpenMeteoError, match="boom"):
        client.fetch(REQ)


def test_request_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="before start_date"):
        OpenMeteoRequest(
            "archive",
            "era5",
            "x",
            0.0,
            0.0,
            ("temperature_2m",),
            dt.date(2021, 2, 1),
            dt.date(2021, 1, 1),
        )


def test_non_json_body_is_retried(tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    fake.responses = [httpx.Response(200, content=b"")]
    with _client(fake, tmp_path) as client:
        resp = client.fetch(REQ)
    assert len(fake.requests) == 2
    assert "hourly" in resp.payload

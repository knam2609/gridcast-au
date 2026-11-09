"""Synthetic Open-Meteo responses, generated from the request's own parameters."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import numpy as np
import pandas as pd

from gridcast.config import OpenMeteoConfig

UNITS = {
    "temperature_2m": "°C",
    "apparent_temperature": "°C",
    "relative_humidity_2m": "%",
    "cloud_cover": "%",
    "shortwave_radiation": "W/m²",
    "wind_speed_10m": "m/s",
}


def _base_variable(column: str) -> tuple[str, int]:
    """``temperature_2m_previous_day2`` → (``temperature_2m``, 2); plain names → offset 0."""
    name, sep, k = column.rpartition("_previous_day")
    return (name, int(k)) if sep else (column, 0)


def fake_values(variable: str, offset: int, hours: pd.DatetimeIndex) -> np.ndarray:
    """Smooth, plausible diurnal series; forecasts differ slightly by offset."""
    h = hours.hour.to_numpy() + hours.day_of_year.to_numpy() * 24.0
    phase = np.sin(2 * np.pi * h / 24.0)
    match variable:
        case "temperature_2m":
            values = 15.0 + 5.0 * phase + 0.1 * offset
        case "apparent_temperature":
            values = 13.0 + 6.0 * phase
        case "relative_humidity_2m":
            values = 60.0 - 20.0 * phase
        case "cloud_cover":
            values = 50.0 + 40.0 * phase
        case "shortwave_radiation":
            values = np.maximum(0.0, 800.0 * phase)
        case "wind_speed_10m":
            values = 4.0 + phase
        case _:
            raise KeyError(variable)
    return np.round(values, 1)


def fake_payload(params: Mapping[str, str]) -> dict[str, Any]:
    columns = params["hourly"].split(",")
    start = pd.Timestamp(params["start_date"], tz="UTC")
    end = pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    hours = pd.date_range(start, end, freq="1h")
    hourly: dict[str, Any] = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours]}
    units = {"time": "iso8601"}
    for col in columns:
        var, offset = _base_variable(col)
        hourly[col] = fake_values(var, offset, hours).tolist()
        units[col] = UNITS[var]
    return {
        "latitude": round(float(params["latitude"]), 2) + 0.01,
        "longitude": round(float(params["longitude"]), 2) - 0.01,
        "generationtime_ms": 1.0,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": 20.0,
        "hourly_units": units,
        "hourly": hourly,
    }


Mutator = Callable[[httpx.Request, dict[str, Any]], dict[str, Any]]


class FakeOpenMeteo:
    """``httpx.MockTransport`` handler that records requests and serves fake payloads."""

    def __init__(self, mutate: Mutator | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.mutate = mutate
        self.responses: list[httpx.Response] = []  # queued responses served first

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.responses:
            return self.responses.pop(0)
        payload = fake_payload(dict(request.url.params))
        if self.mutate is not None:
            payload = self.mutate(request, payload)
        return httpx.Response(200, json=payload)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def om_config(**overrides: object) -> OpenMeteoConfig:
    base: dict[str, object] = {
        "previous_runs_url": "https://previous-runs.example/v1/forecast",
        "archive_url": "https://archive.example/v1/archive",
        "user_agent": "gridcast-test",
        "request_delay_seconds": 0.0,
        "timeout_seconds": 5,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "settle_days": 10,
    }
    return OpenMeteoConfig.model_validate(base | overrides)


TODAY = dt.date(2026, 11, 3)

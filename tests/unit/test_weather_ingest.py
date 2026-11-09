import datetime as dt
from typing import Any, cast

import pandas as pd
import pandera.errors as pe
import pytest
from sqlalchemy import Engine

from gridcast.config import ExpectedMissingConfig, WeatherConfig, load_config
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import (
    WeatherIngestor,
    _keep,
    clip_artefacts,
    hourly_frame,
    month_hours,
    parse_forecasts,
    parse_observations,
)
from gridcast.validation.schemas import weather_forecasts_schema, weather_observations_schema
from tests.open_meteo_fixtures import UNITS, fake_payload

SIX = pd.Timedelta(hours=6)
OCT = dt.date(2021, 10, 1)


def _params(hourly: list[str]) -> dict[str, str]:
    return {
        "latitude": "-37.83",
        "longitude": "144.98",
        "hourly": ",".join(hourly),
        "start_date": "2021-09-30",
        "end_date": "2021-10-31",
    }


def _forecast_payload() -> dict[str, Any]:
    cols = [f"{v}_previous_day{k}" for k in (1, 2) for v in UNITS]
    return fake_payload(_params(cols))


def test_month_hours_cover_nem_month() -> None:
    first, last = month_hours(OCT)
    # 2021-10-01 00:00 NEM time = 2021-09-30 14:00 UTC; the last interval ends at next 00:00.
    assert first == pd.Timestamp("2021-09-30 14:00", tz="UTC")
    assert last == pd.Timestamp("2021-10-31 14:00", tz="UTC")


def test_parse_forecasts_long_by_offset_with_run_timing() -> None:
    cfg = load_config().weather
    frame = parse_forecasts(
        _forecast_payload(), "melbourne", "gfs_seamless", cfg.variables, [1, 2], SIX, SIX
    )
    assert set(frame["offset_days"]) == {1, 2}
    assert list(frame.columns[:3]) == ["model", "site_id", "offset_days"]
    row = frame[
        (frame["offset_days"] == 2)
        & (frame["valid_time"] == pd.Timestamp("2021-10-05 05:00", tz="UTC"))
    ].iloc[0]
    assert row["run_init"] == pd.Timestamp("2021-10-03 00:00", tz="UTC")
    assert row["available_at"] == pd.Timestamp("2021-10-03 06:00", tz="UTC")
    # Offset 2 values come from the *_previous_day2 columns (the fake adds 0.1 per offset).
    day1 = frame[(frame["offset_days"] == 1) & (frame["valid_time"] == row["valid_time"])]
    assert row["temperature_2m"] - day1["temperature_2m"].iloc[0] == pytest.approx(0.1)
    weather_forecasts_schema(["melbourne"], [1, 2], SIX, SIX).validate(frame, lazy=True)


def test_parse_observations() -> None:
    cfg = load_config().weather
    frame = parse_observations(fake_payload(_params(list(UNITS))), "geelong", "era5", cfg.variables)
    assert len(frame) == 32 * 24
    assert frame["valid_time"].iloc[0] == pd.Timestamp("2021-09-30 00:00", tz="UTC")
    weather_observations_schema(["geelong"]).validate(frame, lazy=True)


def test_hourly_frame_rejects_unit_column_and_offset_problems() -> None:
    payload = fake_payload(_params(["temperature_2m"]))
    with pytest.raises(ValueError, match="unexpected units"):
        hourly_frame(payload, {"temperature_2m": "°F"})
    with pytest.raises(ValueError, match="lacks hourly columns"):
        hourly_frame(payload, {"cloud_cover": "%"})
    with pytest.raises(ValueError, match="GMT/UTC"):
        hourly_frame(payload | {"utc_offset_seconds": 36000}, {"temperature_2m": "°C"})


def test_keep_trims_to_month_and_drops_empty_rows() -> None:
    cfg = load_config().weather
    frame = parse_observations(fake_payload(_params(list(UNITS))), "geelong", "era5", cfg.variables)
    frame.loc[frame.index[20], list(UNITS)] = float("nan")  # 2021-09-30 20:00: all missing
    kept = _keep(frame, *month_hours(OCT))
    assert kept["valid_time"].min() == pd.Timestamp("2021-09-30 14:00", tz="UTC")
    assert kept["valid_time"].max() == pd.Timestamp("2021-10-31 14:00", tz="UTC")
    assert len(kept) == 31 * 24 + 1 - 1


def test_forecast_schema_rejects_wrong_run_timing() -> None:
    cfg = load_config().weather
    frame = parse_forecasts(
        _forecast_payload(), "melbourne", "gfs_seamless", cfg.variables, [1, 2], SIX, SIX
    )
    schema = weather_forecasts_schema(["melbourne"], [1, 2], SIX, SIX)
    early = frame.copy()
    early.loc[0, "available_at"] -= pd.Timedelta(hours=1)
    with pytest.raises(pe.SchemaErrors, match="publication delay"):
        schema.validate(early, lazy=True)
    shifted = frame.copy()
    shifted.loc[0, "run_init"] += SIX
    with pytest.raises(pe.SchemaErrors, match="run cadence"):
        schema.validate(shifted, lazy=True)


def test_schema_rejects_impossible_values() -> None:
    cfg = load_config().weather
    frame = parse_observations(fake_payload(_params(list(UNITS))), "geelong", "era5", cfg.variables)
    frame.loc[0, "relative_humidity_2m"] = 120.0
    with pytest.raises(pe.SchemaErrors):
        weather_observations_schema(["geelong"]).validate(frame, lazy=True)


def test_requests_per_dataset() -> None:
    cfg = load_config()
    ingestor = WeatherIngestor(cast(Engine, None), cast(OpenMeteoClient, None), cfg)
    fc = ingestor.request("forecasts", "bendigo", OCT)
    assert (fc.api, fc.model, fc.start_date, fc.end_date) == (
        "previous_runs",
        "gfs_seamless",
        dt.date(2021, 9, 30),
        dt.date(2021, 10, 31),
    )
    assert fc.hourly[0] == "temperature_2m_previous_day1"
    assert len(fc.hourly) == 6 * 2
    obs = ingestor.request("observations", "bendigo", OCT)
    assert (obs.api, obs.model, obs.hourly[-1]) == ("archive", "era5", "wind_speed_10m")
    assert (obs.latitude, obs.longitude) == (-36.76, 144.28)


def test_weather_config_weights_and_validation() -> None:
    cfg = load_config()
    weights = cfg.weather.site_weights("VIC1")
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights, key=weights.__getitem__) == "melbourne"
    assert cfg.weather.sites["latrobe_valley"].population == 43841 + 22359
    raw = cfg.weather.model_dump()
    raw["variables"].pop("wind_speed_10m")
    with pytest.raises(ValueError, match="exactly"):
        WeatherConfig.model_validate(raw)
    with pytest.raises(ValueError, match="UTC offset"):
        ExpectedMissingConfig(
            reason="x",
            variables=["temperature_2m"],
            offset_days=1,
            first=None,
            last=dt.datetime(2024, 1, 1),
        )


def test_clip_artefacts_only_within_tolerance() -> None:
    cfg = load_config().weather
    frame = parse_observations(fake_payload(_params(list(UNITS))), "geelong", "era5", cfg.variables)
    frame.loc[0:3, "cloud_cover"] = [101.0, -1.0, 100.0, 102.5]
    frame.loc[4, "relative_humidity_2m"] = 100.5
    frame.loc[5, "temperature_2m"] = -1.0  # not a bounded variable: untouched
    clipped, counts = clip_artefacts(frame, cfg.variables, tolerance=1.0)
    assert clipped.loc[0:3, "cloud_cover"].tolist() == [100.0, 0.0, 100.0, 102.5]
    assert clipped.loc[4, "relative_humidity_2m"] == 100.0
    assert clipped.loc[5, "temperature_2m"] == -1.0
    assert counts == {"cloud_cover": 2, "relative_humidity_2m": 1}
    assert frame.loc[0, "cloud_cover"] == 101.0  # input not modified
    with pytest.raises(pe.SchemaErrors):  # 102.5 % is beyond tolerance and still rejected
        weather_observations_schema(["geelong"]).validate(clipped, lazy=True)

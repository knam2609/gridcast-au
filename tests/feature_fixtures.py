"""Synthetic hourly weather (forecasts with run timing, plus observations) for feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.config import WEATHER_VARIABLES
from gridcast.features.weather import WeatherInputs
from gridcast.weather.runs import available_at_for, run_init_for

WEIGHTS = {"melbourne": 0.7, "geelong": 0.2, "bendigo": 0.1}
SINGLE = "melbourne"
KINDS = {
    "temperature_2m": "instant",
    "apparent_temperature": "instant",
    "relative_humidity_2m": "instant",
    "cloud_cover": "instant",
    "shortwave_radiation": "preceding_hour_mean",
    "wind_speed_10m": "instant",
}
RUN_EVERY = pd.Timedelta(hours=6)
DELAY = pd.Timedelta(hours=6)
HOUR = pd.Timedelta(hours=1)


def _values(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {var: rng.normal(20.0, 5.0, n) for var in WEATHER_VARIABLES}


def hourly_weather(
    times: pd.DatetimeIndex,
    seed: int = 0,
    missing_values: float = 0.02,
    missing_rows: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(forecasts, observations)`` covering ``times``, with random gaps.

    Forecasts follow the archive's run timing exactly (``run_init = floor_6h(t − 24K h)``,
    ``available_at = run_init + 6 h``), so the selection rule is exercised realistically.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(times[0].floor(HOUR), times[-1].ceil(HOUR), freq=HOUR)
    fc_parts, obs_parts = [], []
    for site in WEIGHTS:
        valid = pd.Series(hours)
        obs_parts.append(
            pd.DataFrame({"site_id": site, "valid_time": hours, **_values(len(hours), rng)})
        )
        for k in (1, 2):
            run_init = run_init_for(valid, k, RUN_EVERY)
            fc_parts.append(
                pd.DataFrame(
                    {
                        "site_id": site,
                        "offset_days": np.int16(k),
                        "valid_time": hours,
                        "run_init": run_init.to_numpy(),
                        "available_at": available_at_for(run_init, DELAY).to_numpy(),
                        **_values(len(hours), rng),
                    }
                )
            )
    forecasts = pd.concat(fc_parts, ignore_index=True)
    for col in ("valid_time", "run_init", "available_at"):
        forecasts[col] = pd.to_datetime(forecasts[col], utc=True)
    for var in WEATHER_VARIABLES:
        forecasts.loc[rng.random(len(forecasts)) < missing_values, var] = np.nan
    forecasts = forecasts[rng.random(len(forecasts)) >= missing_rows].reset_index(drop=True)
    observations = pd.concat(obs_parts, ignore_index=True)
    return forecasts, observations


def weather_inputs(
    times: pd.DatetimeIndex,
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    publication_delay: pd.Timedelta | None = None,
) -> WeatherInputs:
    return WeatherInputs.from_hourly(
        times,
        WEIGHTS,
        KINDS,
        SINGLE,
        forecasts=forecasts,
        observations=observations,
        publication_delay=publication_delay,
    )

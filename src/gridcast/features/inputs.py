"""Load everything the feature builder needs from PostgreSQL."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.database.queries import (
    load_market_frame,
    load_weather_forecasts,
    load_weather_observations,
)
from gridcast.features.builder import FeatureBuilder
from gridcast.features.calendar import load_calendar_days
from gridcast.features.weather import WeatherInputs
from gridcast.forecasting.origins import ForecastSetup, SeriesContext, build_context

HOUR = pd.Timedelta(hours=1)


def publication_delay(cfg: AppConfig, hours: float | None) -> pd.Timedelta | None:
    """Validated delay override. ``None`` keeps the stored ``available_at``.

    A delay shorter than the configured one would read forecasts before they were
    published (leakage), so it is refused; longer delays are sensitivity checks.
    """
    if hours is None:
        return None
    if hours < cfg.weather.forecasts.publication_delay_hours:
        raise ValueError(
            f"publication delay {hours} h is shorter than the configured "
            f"{cfg.weather.forecasts.publication_delay_hours} h (that would leak)"
        )
    return pd.Timedelta(hours=hours)


def load_context(
    engine: Engine, cfg: AppConfig, region_id: str, target_column: str = "total_demand_mw"
) -> SeriesContext:
    region = cfg.region(region_id)
    market = load_market_frame(engine, region_id)
    with engine.connect() as conn:
        calendar_days = load_calendar_days(conn, region.holiday_subdiv)
    return build_context(market, target_column, region.local_tz, calendar_days)


def load_weather_inputs(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    times: pd.DatetimeIndex,
    delay_hours: float | None = None,
) -> WeatherInputs:
    """Forecast and observed weather for a region, aligned to ``times``."""
    wcfg = cfg.weather
    first, last = times[0].floor(HOUR), times[-1].ceil(HOUR)
    forecasts = load_weather_forecasts(engine, wcfg.forecasts.model, first, last)
    observations = load_weather_observations(engine, wcfg.observations.model, first, last)
    weights = wcfg.site_weights(region_id)
    if forecasts.empty or observations.empty:
        raise LookupError("no weather loaded; run `gridcast ingest-weather` first")
    return WeatherInputs.from_hourly(
        times,
        weights,
        {name: v.kind for name, v in wcfg.variables.items()},
        cfg.features.single_site,
        forecasts=forecasts[forecasts["site_id"].isin(list(weights))],
        observations=observations[observations["site_id"].isin(list(weights))],
        offsets=wcfg.forecasts.offsets_days,
        publication_delay=publication_delay(cfg, delay_hours),
    )


def load_feature_builder(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    delay_hours: float | None = None,
    ctx: SeriesContext | None = None,
) -> FeatureBuilder:
    ctx = ctx if ctx is not None else load_context(engine, cfg, region_id)
    weather = load_weather_inputs(engine, cfg, region_id, ctx.times, delay_hours)
    return FeatureBuilder(ctx, ForecastSetup.from_config(cfg), weather, cfg.features.degree_base_c)

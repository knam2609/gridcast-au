"""Feature matrices for (issue time, horizon) pairs: variants A, C, D, C⁺ and D⁺.

Rows are pairs (*o*, *h*) in origin-major order. Every feature is defined in
:mod:`gridcast.features.catalogue`, which also fixes the column names and order. Demand
features read only ``ctx.y`` at positions ``<= o - latency`` (checked with
``assert_known``); forecast weather reads only values with ``available_at <= o``
(:func:`gridcast.features.weather.select_forecast`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.features import catalogue
from gridcast.features.weather import WeatherInputs, select_forecast, select_observed
from gridcast.forecasting.origins import ForecastSetup, SeriesContext
from gridcast.models.baselines import DAY_STEPS, WEEK_STEPS, assert_known

HOUR_STEPS = 12


@dataclass(slots=True)
class FeatureMatrix:
    variant: str
    origins: np.ndarray  # int64 origin position per row
    horizons: np.ndarray  # int16 horizon (steps) per row
    X: pd.DataFrame  # features, catalogue order
    actual: np.ndarray  # y(t), float64 (NaN where missing)
    weather_offset: np.ndarray | None  # forecast offset used (0 = none); C and C⁺ only

    @property
    def targets(self) -> np.ndarray:
        return self.origins + self.horizons

    def complete(self) -> np.ndarray:
        """Rows where every feature is present (the rows a model can use)."""
        return np.asarray(self.X.notna().all(axis=1))


def seasonal_source(target_pos: np.ndarray, h: np.ndarray, latency: int, period: int) -> np.ndarray:
    """``t − k × period`` with the smallest ``k ≥ 1`` such that it is known at the cutoff."""
    k = np.ceil((h + latency) / period).astype(np.int64)
    return target_pos - k * period


class FeatureBuilder:
    """Builds feature matrices from one :class:`SeriesContext` and its weather."""

    def __init__(
        self,
        ctx: SeriesContext,
        setup: ForecastSetup,
        weather: WeatherInputs | None = None,
        degree_base_c: float = 18.0,
    ) -> None:
        if weather is not None and len(weather.times_ns) != len(ctx):
            raise ValueError("weather is aligned to a different grid than the context")
        self.ctx = ctx
        self.setup = setup
        self.weather = weather
        self.degree_base_c = degree_base_c
        frame = ctx.frame
        self._calendar = {
            "cal_slot": ctx.slot,
            "cal_day_of_week": frame["day_of_week"].to_numpy(dtype=np.int8),
            "cal_month": frame["month"].to_numpy(dtype=np.int8),
            "cal_non_working": ctx.day_type,
            "cal_public_holiday": frame["is_public_holiday"].to_numpy(dtype=np.int8),
            "cal_dst_change_day": frame["is_dst_change_day"].to_numpy(dtype=np.int8),
            "cal_tod_sin": frame["tod_sin"].to_numpy(dtype=np.float64),
            "cal_tod_cos": frame["tod_cos"].to_numpy(dtype=np.float64),
        }

    def build(
        self, variant_id: str, origins: np.ndarray, horizons: np.ndarray | None = None
    ) -> FeatureMatrix:
        """Features for every (origin, horizon) pair; ``horizons`` defaults to 1…H."""
        spec = catalogue.variant(variant_id)
        setup = self.setup
        h_set = setup.horizons if horizons is None else np.asarray(horizons, dtype=np.int64)
        if h_set.size == 0 or h_set.min() < 1 or h_set.max() > setup.horizon_steps:
            raise ValueError(f"horizons must lie in 1..{setup.horizon_steps}")
        origins = np.asarray(origins, dtype=np.int64)
        o = np.repeat(origins, len(h_set))
        h = np.tile(h_set, len(origins))
        t = o + h
        if t.size and (o.min() < 0 or t.max() >= len(self.ctx)):
            raise IndexError("an origin or target lies outside the context grid")

        cols: dict[str, np.ndarray] = {"lead_minutes": (5 * h).astype(np.int16)}
        cols |= {name: values[t] for name, values in self._calendar.items()}
        prev = self._known_positions(t - DAY_STEPS, None)  # calendar only: no cutoff check
        cols["cal_yesterday_non_working"] = self.ctx.day_type[prev]
        cols["cal_yesterday_public_holiday"] = self._calendar["cal_public_holiday"][prev]
        cols |= self._demand_now(origins, len(h_set))
        cols |= self._demand_seasonal(o, h, t)

        weather_offset = None
        if spec.weather != "none":
            if self.weather is None:
                raise ValueError(f"variant {variant_id} needs weather inputs")
            required = catalogue.required_sources(variant_id)
            if spec.weather == "forecast":
                src, weather_offset = select_forecast(self.weather, required, o, t)
            else:
                src = select_observed(self.weather, required, t)
            cols |= self._weather(src)

        names = catalogue.feature_names(variant_id)
        if set(names) - set(cols):
            raise KeyError(f"builder does not compute {sorted(set(names) - set(cols))}")
        X = pd.DataFrame({name: cols[name] for name in names})
        return FeatureMatrix(
            variant=variant_id,
            origins=o,
            horizons=h.astype(np.int16),
            X=X,
            actual=self.ctx.y[t],
            weather_offset=weather_offset,
        )

    # ------------------------------------------------------------------ groups

    def _known_positions(self, positions: np.ndarray, origins: np.ndarray | None) -> np.ndarray:
        if positions.size and positions.min() < 0:
            raise IndexError(
                "an origin has too little history (needs one week before its first target)"
            )
        if origins is not None:
            assert_known(positions, origins, self.setup)
        return positions

    def _demand_now(self, origins: np.ndarray, n_h: int) -> dict[str, np.ndarray]:
        """Per-origin demand summaries, repeated for each horizon of the origin."""
        y = self.ctx.y
        c = origins - self.setup.latency_steps
        day = self._known_positions(
            c[:, None] - np.arange(DAY_STEPS - 1, -1, -1)[None, :], origins
        )  # (n_origins, 288): c - 287 … c
        window = y[day]
        last_hour = window[:, -HOUR_STEPS:]
        y_c = window[:, -1]
        per_origin = {
            "dem_now": y_c,
            "dem_now_vs_yesterday": y_c - y[self._known_positions(c - DAY_STEPS, origins)],
            "dem_change_1h": y_c - window[:, -1 - HOUR_STEPS],
            "dem_mean_1h": last_hour.mean(axis=1),
            "dem_std_1h": last_hour.std(axis=1),
            "dem_mean_24h": window.mean(axis=1),
            "dem_std_24h": window.std(axis=1),
        }
        return {name: np.repeat(v, n_h) for name, v in per_origin.items()}

    def _demand_seasonal(
        self, o: np.ndarray, h: np.ndarray, t: np.ndarray
    ) -> dict[str, np.ndarray]:
        lat = self.setup.latency_steps
        day = self._known_positions(seasonal_source(t, h, lat, DAY_STEPS), o)
        week = self._known_positions(seasonal_source(t, h, lat, WEEK_STEPS), o)
        yesterday, last_week = self.ctx.y[day], self.ctx.y[week]
        return {
            "dem_same_time_yesterday": yesterday,
            "dem_same_time_last_week": last_week,
            "dem_yesterday_vs_last_week": yesterday - last_week,
        }

    def _weather(self, src: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        temp = src["temp_weighted"]
        out = {
            "wx_temp": temp,
            "wx_cooling_degrees": np.maximum(temp - self.degree_base_c, 0.0),
            "wx_heating_degrees": np.maximum(self.degree_base_c - temp, 0.0),
            "wx_temp_sq": temp**2,
            "wx_temp_single_site": src["temp_single"],
        }
        if "apparent_temp_weighted" in src:
            out |= {
                "wx_apparent_temp": src["apparent_temp_weighted"],
                "wx_humidity": src["humidity_weighted"],
                "wx_cloud_cover": src["cloud_weighted"],
                "wx_shortwave": src["radiation_weighted"],
            }
        return out

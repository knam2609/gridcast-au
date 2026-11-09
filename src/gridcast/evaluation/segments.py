"""Evaluation segments, defined on the *target* interval.

Calendar segments (peak, weekend, holiday, DST) are known in advance. Market-state
segments (price spike, negative price, intervention) and the M2 weather segments (very hot
or very cloudy local days, from ERA5) condition on what actually happened and are
therefore *ex-post* diagnostics: they describe where a model fails, they cannot be used to
choose a model at forecast time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.config import EvaluationConfig

SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "all": "All target intervals",
    "working_day": "Working days (not weekend or public holiday)",
    "peak": "Evening peak (local {peak})",
    "weekend": "Saturday/Sunday",
    "public_holiday": "Victorian public holidays",
    "dst_change_day": "Local days when clocks change (23 h / 25 h days)",
    "normal": "No price spike, negative price or market intervention",
    "price_spike": "Realised price > ${spike:,.0f}/MWh (ex-post)",
    "negative_price": "Realised price < $0/MWh (ex-post)",
    "market_intervention": "Administered pricing or market suspension (ex-post)",
}


def describe_segments(cfg: EvaluationConfig) -> dict[str, str]:
    lo, hi = cfg.peak_hours_local
    return {
        k: v.format(peak=f"{lo:02d}:00-{hi:02d}:00", spike=cfg.price_spike_threshold)
        for k, v in SEGMENT_DESCRIPTIONS.items()
    }


def target_segments(frame: pd.DataFrame, cfg: EvaluationConfig) -> dict[str, np.ndarray]:
    """Boolean mask per segment, aligned with the rows of a :class:`SeriesContext` frame."""
    price = frame["rrp_aud_mwh"].to_numpy(dtype=np.float64, na_value=np.nan)
    apc_bits = frame["apc_flag_bits"].fillna(0).to_numpy(dtype=np.int64)
    suspended = frame["market_suspended_flag"].fillna(0).to_numpy(dtype=np.int64) != 0
    lo, hi = cfg.peak_hours_local
    hour = frame["local_hour"].to_numpy()
    weekend = frame["is_weekend"].to_numpy(dtype=bool)
    holiday = frame["is_public_holiday"].to_numpy(dtype=bool)

    spike = price > cfg.price_spike_threshold
    negative = price < 0
    intervention = ((apc_bits & 1) != 0) | suspended
    segments = {
        "all": np.ones(len(frame), dtype=bool),
        "working_day": ~(weekend | holiday),
        "peak": (hour >= lo) & (hour < hi),
        "weekend": weekend,
        "public_holiday": holiday,
        "dst_change_day": frame["is_dst_change_day"].to_numpy(dtype=bool),
        "normal": ~(spike | negative | intervention) & ~np.isnan(price),
        "price_spike": spike,
        "negative_price": negative,
        "market_intervention": intervention,
    }
    assert list(segments) == list(SEGMENT_DESCRIPTIONS)
    return segments


WEATHER_SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "very_hot_day": "Local days whose weighted ERA5 maximum temperature is at least {hot:g} °C "
    "(ex-post)",
    "very_cloudy_day": "Local days whose weighted ERA5 cloud cover averages at least {cloud:g}% "
    "over {day} local (ex-post)",
}


def describe_weather_segments(cfg: EvaluationConfig) -> dict[str, str]:
    lo, hi = cfg.daytime_hours_local
    return {
        k: v.format(
            hot=cfg.hot_day_max_temp_c,
            cloud=cfg.cloudy_day_mean_cloud_pct,
            day=f"{lo:02d}:00-{hi:02d}:00",
        )
        for k, v in WEATHER_SEGMENT_DESCRIPTIONS.items()
    }


def weather_segments(
    local_day: np.ndarray,
    local_hour: np.ndarray,
    temperature: np.ndarray,
    cloud_cover: np.ndarray,
    cfg: EvaluationConfig,
) -> dict[str, np.ndarray]:
    """Very hot / very cloudy local days, as masks aligned with the context grid.

    ``temperature`` and ``cloud_cover`` are observed (ERA5) population-weighted values on
    the 5-minute grid. Missing values are ignored; a day with no values is in neither
    segment. Daytime is ``[start, end)`` local hour of the interval start.
    """
    lo, hi = cfg.daytime_hours_local
    frame = pd.DataFrame(
        {
            "day": local_day,
            "temp": temperature,
            "cloud": np.where((local_hour >= lo) & (local_hour < hi), cloud_cover, np.nan),
        }
    )
    daily = frame.groupby("day").agg(max_temp=("temp", "max"), day_cloud=("cloud", "mean"))
    max_temp = daily["max_temp"].reindex(local_day).to_numpy(dtype=np.float64)
    day_cloud = daily["day_cloud"].reindex(local_day).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        segments = {
            "very_hot_day": max_temp >= cfg.hot_day_max_temp_c,
            "very_cloudy_day": day_cloud >= cfg.cloudy_day_mean_cloud_pct,
        }
    assert list(segments) == list(WEATHER_SEGMENT_DESCRIPTIONS)
    return segments

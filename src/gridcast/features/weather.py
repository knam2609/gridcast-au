"""Weather at the target time, as known at the issue time (variants C, D, C⁺, D⁺).

Pipeline (m2_plan.md §2.1, docs/m2_data_spike.md §8):

1. Hourly rows → population-weighted means per forecast offset, plus the single site's
   temperature (:func:`gridcast.weather.align.site_weighted`).
2. Hourly → 5-minute values on the market grid, per offset
   (:func:`gridcast.weather.align.to_five_minute`). An interpolated value exists only if
   both bracketing hours exist, and is available from the later hour's ``available_at``.
3. For each (issue time *o*, target *t*): take the **freshest offset** (smallest
   ``offset_days``) whose every required value at *t* exists and has ``available_at <= o``.
   If none qualifies, the pair has no forecast weather (NaN; excluded downstream).

Observed weather (ERA5, variants D/D⁺) skips step 3: it is a perfect-foresight oracle
by construction and is read at *t* regardless of *o*.

All weather of one pair comes from one offset, so the features of a pair are one
coherent forecast (instead of mixing, say, a day-1 temperature with a day-2 humidity).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.weather.align import site_weighted, to_five_minute

# Source series: name -> (hourly variable, aggregation). "weighted" = population-weighted
# mean over the region's sites; "single" = the configured single site on its own.
SOURCES: dict[str, tuple[str, str]] = {
    "temp_weighted": ("temperature_2m", "weighted"),
    "temp_single": ("temperature_2m", "single"),
    "apparent_temp_weighted": ("apparent_temperature", "weighted"),
    "humidity_weighted": ("relative_humidity_2m", "weighted"),
    "cloud_weighted": ("cloud_cover", "weighted"),
    "radiation_weighted": ("shortwave_radiation", "weighted"),
}
NEVER = np.iinfo(np.int64).max  # available_at of a value that is missing (never usable)


def to_ns(values: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds since the epoch; NaT becomes :data:`NEVER`.

    pandas may store datetimes in ns or µs; converting the unit explicitly keeps every
    comparison in one unit (never mix raw ``asi8`` values of different units).
    """
    idx = pd.DatetimeIndex(values)
    if idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError("expected tz-aware UTC timestamps")
    out = idx.as_unit("ns").asi8.copy()
    out[np.asarray(idx.isna())] = NEVER
    return out


@dataclass(frozen=True, slots=True)
class AlignedWeather:
    """5-minute weather on a market grid: arrays aligned by position with ``times``."""

    series: dict[str, np.ndarray]  # source -> float64 (NaN = missing)
    available_at: dict[str, np.ndarray] | None  # source -> int64 ns; None for observations


@dataclass(frozen=True, slots=True)
class WeatherInputs:
    """Everything the feature builder needs about weather, aligned to one market grid."""

    times_ns: np.ndarray  # int64 ns of the grid's interval_end values
    forecasts: dict[int, AlignedWeather]  # offset_days -> aligned forecast
    observed: AlignedWeather | None

    @classmethod
    def from_hourly(
        cls,
        times: pd.DatetimeIndex,
        weights: Mapping[str, float],
        kinds: Mapping[str, str],
        single_site: str,
        forecasts: pd.DataFrame | None = None,
        observations: pd.DataFrame | None = None,
        offsets: Sequence[int] = (1, 2),
        publication_delay: pd.Timedelta | None = None,
    ) -> WeatherInputs:
        """Align hourly rows (as returned by the ``load_weather_*`` queries).

        ``publication_delay`` recomputes ``available_at = run_init + delay`` (sensitivity
        checks); by default the stored ``available_at`` is used.
        """
        if single_site not in weights:
            raise ValueError(f"single site {single_site!r} has no weight")
        aligned: dict[int, AlignedWeather] = {}
        if forecasts is not None:
            if publication_delay is not None:
                if publication_delay <= pd.Timedelta(0):
                    raise ValueError("publication_delay must be positive")
                forecasts = forecasts.assign(available_at=forecasts["run_init"] + publication_delay)
            unknown = set(forecasts["offset_days"].unique()) - set(offsets)
            if unknown:
                raise ValueError(f"forecast offsets {sorted(unknown)} are not configured")
            for k in offsets:
                rows = forecasts[forecasts["offset_days"] == k].drop(columns="offset_days")
                aligned[k] = _align(
                    rows, times, weights, kinds, single_site, with_availability=True
                )
        observed = None
        if observations is not None:
            observed = _align(
                observations, times, weights, kinds, single_site, with_availability=False
            )
        return cls(times_ns=to_ns(times), forecasts=aligned, observed=observed)


def _align(
    hourly: pd.DataFrame,
    times: pd.DatetimeIndex,
    weights: Mapping[str, float],
    kinds: Mapping[str, str],
    single_site: str,
    with_availability: bool,
) -> AlignedWeather:
    n = len(times)
    empty = np.full(n, np.nan)
    values: dict[str, np.ndarray] = {name: empty.copy() for name in SOURCES}
    available: dict[str, np.ndarray] = {name: np.full(n, NEVER) for name in SOURCES}
    if hourly.empty:
        return AlignedWeather(values, available if with_availability else None)

    weighted_vars = sorted({var for var, how in SOURCES.values() if how == "weighted"})
    single_vars = sorted({var for var, how in SOURCES.values() if how == "single"})
    keep = ["valid_time", "available_at"] if with_availability else ["valid_time"]
    frames = {
        # site_weighted keeps the latest available_at of the sites; a missing site makes the
        # weighted value NaN, which the selection rule treats as unusable anyway.
        "weighted": site_weighted(hourly, weights, weighted_vars),
        "single": hourly.loc[hourly["site_id"] == single_site, [*keep, *single_vars]],
    }
    for how, frame in frames.items():
        variables = weighted_vars if how == "weighted" else single_vars
        five = to_five_minute(frame, times, {v: kinds[v] for v in variables})
        five = five.set_index("interval_end").reindex(times)
        for name, (var, agg) in SOURCES.items():
            if agg != how:
                continue
            values[name] = five[var].to_numpy(dtype="float64", na_value=np.nan)
            if with_availability:
                available[name] = to_ns(five["available_at"])
    return AlignedWeather(values, available if with_availability else None)


def select_forecast(
    inputs: WeatherInputs,
    required: Sequence[str],
    origin_pos: np.ndarray,
    target_pos: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Forecast values of ``required`` sources for each (origin, target) row.

    Returns the values (NaN where no offset qualifies) and the offset used per row
    (``0`` = none). Only values with ``available_at <= origin time`` are ever read.
    """
    if not inputs.forecasts:
        raise ValueError("no forecast weather loaded")
    origin_ns = inputs.times_ns[origin_pos]
    used = np.zeros(len(target_pos), dtype=np.int8)
    for k in sorted(inputs.forecasts):  # freshest first
        fc = inputs.forecasts[k]
        assert fc.available_at is not None
        ok = used == 0
        for name in required:
            ok &= fc.available_at[name][target_pos] <= origin_ns
            ok &= np.isfinite(fc.series[name][target_pos])
        used[ok] = k
    out: dict[str, np.ndarray] = {}
    for name in required:
        col = np.full(len(target_pos), np.nan)
        for k, fc in inputs.forecasts.items():
            rows = used == k
            col[rows] = fc.series[name][target_pos[rows]]
        out[name] = col
    return out, used


def select_observed(
    inputs: WeatherInputs, required: Sequence[str], target_pos: np.ndarray
) -> dict[str, np.ndarray]:
    """Observed (reanalysis) values at each target: the perfect-foresight oracle."""
    if inputs.observed is None:
        raise ValueError("no observed weather loaded")
    return {name: inputs.observed.series[name][target_pos] for name in required}

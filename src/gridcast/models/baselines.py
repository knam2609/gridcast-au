"""Baseline forecasters. Every ML model must beat these to justify its complexity.

All baselines are *vectorised over origins and horizons* and use only the information
set at each origin (``interval_end <= origin - latency``) for observed values, plus
calendar attributes of the target. ``assert_known`` guards the index arithmetic, and
``tests/unit/test_leakage.py`` checks the property end to end by perturbing the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from gridcast.forecasting.origins import ForecastSetup, SeriesContext, target_positions

DAY_STEPS = 288
WEEK_STEPS = 7 * DAY_STEPS


def assert_known(source: np.ndarray, origins: np.ndarray, setup: ForecastSetup) -> None:
    """Fail if any source position lies after the information cutoff of its origin."""
    cutoff = (origins - setup.latency_steps).reshape(-1, *([1] * (source.ndim - 1)))
    if np.any(source > cutoff):
        raise AssertionError("forecast uses data after the information cutoff (leakage)")


def _take(y: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """``y[positions]`` with NaN for positions before the start of the series."""
    out = np.full(positions.shape, np.nan)
    valid = positions >= 0
    out[valid] = y[positions[valid]]
    return out


class Forecaster(ABC):
    """Interface shared by baselines and (later) ML models."""

    name: str
    kind: str = "baseline"

    def fit(self, ctx: SeriesContext, cutoff: int, setup: ForecastSetup) -> None:  # noqa: B027
        """Fit on data with position <= ``cutoff``. Baselines need no fitting."""

    @abstractmethod
    def predict(self, ctx: SeriesContext, origins: np.ndarray, setup: ForecastSetup) -> np.ndarray:
        """Return an ``(n_origins, H)`` array of point forecasts."""

    def params(self) -> dict[str, object]:
        return {}


class Persistence(Forecaster):
    """ŷ(o + h) = y(o − latency): the last known value, for every horizon."""

    name = "persistence"

    def predict(self, ctx: SeriesContext, origins: np.ndarray, setup: ForecastSetup) -> np.ndarray:
        source = (origins - setup.latency_steps)[:, None]
        assert_known(source, origins, setup)
        return np.broadcast_to(_take(ctx.y, source), (len(origins), setup.horizon_steps)).copy()


class SeasonalNaive(Forecaster):
    """ŷ(t) = y(t − k·P), with the smallest k such that ``t − k·P`` is already known.

    With latency 0 and H ≤ P this is simply "same interval one period ago".
    """

    def __init__(self, period_steps: int, name: str) -> None:
        self.period_steps = period_steps
        self.name = name

    def predict(self, ctx: SeriesContext, origins: np.ndarray, setup: ForecastSetup) -> np.ndarray:
        h = setup.horizons[None, :]
        k = np.ceil((h + setup.latency_steps) / self.period_steps).astype(np.int64)
        source = target_positions(origins, setup) - k * self.period_steps
        assert_known(source, origins, setup)
        return _take(ctx.y, source)

    def params(self) -> dict[str, object]:
        return {"period_steps": self.period_steps}


class MedianProfile(Forecaster):
    """Trailing median by (day type, local 5-min slot), refreshed once per local day.

    For an origin whose information cutoff falls on local date D, the profile is the
    median of the ``lookback_days`` local days strictly before D, grouped by day type
    (working vs weekend/holiday) and slot. The target's day type and slot (calendar
    attributes, known in advance) select the profile value. If a (day type, slot) cell is
    empty, the slot median across both day types is used.
    """

    name = "median_profile"

    def __init__(self, lookback_days: int = 28) -> None:
        self.lookback_days = lookback_days

    def params(self) -> dict[str, object]:
        return {"lookback_days": self.lookback_days}

    def _profiles(self, ctx: SeriesContext, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Profiles for each local day in ``days``: ``(n_days, 2, 288)`` and ``(n_days, 288)``."""
        frame = pd.DataFrame(
            {"day": ctx.local_day, "dtype": ctx.day_type, "slot": ctx.slot, "y": ctx.y}
        )
        by_type = np.full((len(days), 2, DAY_STEPS), np.nan)
        by_slot = np.full((len(days), DAY_STEPS), np.nan)
        # local_day is non-decreasing along the grid, so windows are contiguous slices.
        lo = np.searchsorted(ctx.local_day, days - self.lookback_days, side="left")
        hi = np.searchsorted(ctx.local_day, days, side="left")
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            window = frame.iloc[a:b]
            if window.empty:
                continue
            med = window.groupby(["dtype", "slot"])["y"].median()
            types, slots = med.index.get_level_values(0), med.index.get_level_values(1)
            by_type[i, types, slots] = med.to_numpy()
            med_slot = window.groupby("slot")["y"].median()
            by_slot[i, med_slot.index.to_numpy()] = med_slot.to_numpy()
        return by_type, by_slot

    def predict(self, ctx: SeriesContext, origins: np.ndarray, setup: ForecastSetup) -> np.ndarray:
        cutoff = origins - setup.latency_steps
        # Local date of the cutoff instant (interval_end at `cutoff`). Every interval that
        # *starts* before that date's local midnight ends at or before the cutoff.
        cutoff_local = ctx.times[cutoff].tz_convert(ctx.tz)
        origin_day = np.array([d.toordinal() for d in cutoff_local.date], dtype=np.int64)
        # The profile window must end at or before the cutoff: assert on positions.
        last_used = np.searchsorted(ctx.local_day, origin_day, side="left") - 1
        assert_known(last_used, origins, setup)

        days, day_idx = np.unique(origin_day, return_inverse=True)
        by_type, by_slot = self._profiles(ctx, days)
        targets = target_positions(origins, setup)
        di = day_idx[:, None]
        pred = by_type[di, ctx.day_type[targets], ctx.slot[targets]]
        fallback = by_slot[di, ctx.slot[targets]]
        return np.where(np.isnan(pred), fallback, pred)


def default_baselines(target: str, lookback_days: int = 28) -> list[Forecaster]:
    """Baseline set per target (demand: naive family; price: naive family + median profile)."""
    common: list[Forecaster] = [
        Persistence(),
        SeasonalNaive(DAY_STEPS, "daily_naive"),
        SeasonalNaive(WEEK_STEPS, "weekly_naive"),
    ]
    if target == "price":
        return [*common, MedianProfile(lookback_days)]
    if target == "demand":
        return common
    raise ValueError(f"unknown target {target!r}")

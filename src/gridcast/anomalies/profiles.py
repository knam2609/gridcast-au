"""Trailing same-time-of-day profiles and robust scales (m4_plan.md §2.3).

A *profile* of a series at interval ``t`` is the median and MAD of its values in ``t``'s
local half-hour slot over the previous ``days`` local days, ending before ``t``'s own local
day (so a profile changes only at local midnight and never sees ``t``'s day). D1 splits it
by day type: a business day's profile uses only the business days among those days, and a
weekend or public holiday's only the other days.

A profile needs its full look-back: it is missing until ``days`` local days of data precede
the day (this only affects the warm-up). Robust scale = max(``consistency`` × MAD, floor).

:func:`lookback_days` names the days a profile uses. The causality tests' leaky control
replaces it with a centred window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from gridcast.anomalies.series import Layout


def lookback_days(day: int, days: int) -> np.ndarray:
    """The local days whose values form day ``day``'s profile: the ``days`` before it."""
    return np.arange(day - days, day)


@dataclass(frozen=True, slots=True)
class Profile:
    """Per-interval profile median and MAD (NaN where the profile is missing)."""

    median: np.ndarray
    mad: np.ndarray

    def z(self, values: np.ndarray, consistency: float, floor: float) -> np.ndarray:
        """Signed robust z: (value − median) / max(consistency × MAD, floor)."""
        return (values - self.median) / np.maximum(consistency * self.mad, floor)


def _cube(
    values: np.ndarray, first: np.ndarray, second: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """``values`` laid out as ``(shape[0], shape[1], cap)``, NaN-padded, keyed by two labels."""
    key = first * shape[1] + second
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
    sizes = np.diff(np.r_[group_start, len(key)])
    pos = np.empty(len(key), dtype=np.int64)
    pos[order] = np.arange(len(key)) - np.repeat(group_start, sizes)
    cube = np.full((*shape, int(pos.max()) + 1), np.nan)
    cube[first, second, pos] = values
    return cube


def _median_mad(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise nan-median and MAD of a 2-D block (all-NaN rows give NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(block, axis=1)
        mad = np.nanmedian(np.abs(block - med[:, None]), axis=1)
    return med, mad


def _days_for(day: int, lay: Layout, days: int, same_type: bool) -> np.ndarray | None:
    rows = lookback_days(day, days)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= lay.n_days:
        return None  # incomplete look-back
    if same_type:
        rows = rows[lay.day_business[rows] == lay.day_business[day]]
    return rows


def slot_profile(values: np.ndarray, lay: Layout, days: int, by_day_type: bool) -> Profile:
    """Same-slot median and MAD over the previous ``days`` local days (§2.3)."""
    cube = _cube(
        np.asarray(values, dtype=np.float64), lay.local_day, lay.slot, (lay.n_days, lay.n_slots)
    )
    med = np.full((lay.n_days, lay.n_slots), np.nan)
    mad = np.full((lay.n_days, lay.n_slots), np.nan)
    for day in range(lay.n_days):
        rows = _days_for(day, lay, days, by_day_type)
        if rows is None or rows.size == 0:
            continue
        block = cube[rows].transpose(1, 0, 2).reshape(lay.n_slots, -1)
        med[day], mad[day] = _median_mad(block)
    return Profile(med[lay.local_day, lay.slot], mad[lay.local_day, lay.slot])


def daily_robust_scale(
    values: np.ndarray, lay: Layout, days: int, consistency: float, floor: float
) -> np.ndarray:
    """Per interval: max(consistency × MAD, floor) of every value in the previous ``days``
    local days (all times of day). D6 scales the 5-minute demand change by it."""
    cube = _cube(
        np.asarray(values, dtype=np.float64),
        lay.local_day,
        np.zeros(len(values), dtype=np.int64),
        (lay.n_days, 1),
    )
    scale = np.full(lay.n_days, np.nan)
    for day in range(lay.n_days):
        rows = _days_for(day, lay, days, same_type=False)
        if rows is None:
            continue
        _, mad = _median_mad(cube[rows].reshape(1, -1))
        scale[day] = max(consistency * mad[0], floor) if np.isfinite(mad[0]) else np.nan
    return scale[lay.local_day]


class SlotValues:
    """The values behind a :func:`slot_profile`, for the evidence (§2.5): the same local
    half-hour over the same look-back days (and day type, if split), so the evidence's
    median is the detector's profile median."""

    def __init__(self, values: np.ndarray, lay: Layout, days: int, by_day_type: bool) -> None:
        self.lay = lay
        self.days = days
        self.by_day_type = by_day_type
        self.cube = _cube(
            np.asarray(values, dtype=np.float64), lay.local_day, lay.slot, (lay.n_days, lay.n_slots)
        )

    def at(self, index: int) -> np.ndarray | None:
        """Finite values of interval ``index``'s profile (None: the profile is missing)."""
        day = int(self.lay.local_day[index])
        rows = _days_for(day, self.lay, self.days, self.by_day_type)
        if rows is None or rows.size == 0:
            return None
        block = self.cube[rows, int(self.lay.slot[index])].ravel()
        block = block[np.isfinite(block)]
        return block if block.size else None

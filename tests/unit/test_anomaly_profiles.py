"""Trailing same-half-hour profiles and robust scales (m4_plan.md §2.3), against pandas."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies.profiles import daily_robust_scale, slot_profile
from gridcast.anomalies.series import layout
from gridcast.timeutils import INTERVAL
from tests.anomaly_fixtures import TZ, synthetic_series

DAYS = 5


def _frame(seed: int = 0) -> tuple[pd.DataFrame, object]:
    series = synthetic_series(seed, start=dt.date(2024, 3, 30), n_days=16)
    local = (series.ends - INTERVAL).tz_convert(TZ)
    frame = pd.DataFrame(
        {
            "value": series.demand,
            "date": local.date,
            "slot": local.hour * 2 + local.minute // 30,
            "business": series.business,
        }
    )
    return frame, series


def _brute(frame: pd.DataFrame, i: int, by_type: bool) -> tuple[float, float]:
    row = frame.iloc[i]
    lo = row["date"] - dt.timedelta(days=DAYS)
    if lo < frame["date"].iloc[0]:
        return np.nan, np.nan
    sel = (frame["date"] >= lo) & (frame["date"] < row["date"]) & (frame["slot"] == row["slot"])
    if by_type:
        sel &= frame["business"] == row["business"]
    vals = frame.loc[sel, "value"].to_numpy()
    if vals.size == 0:
        return np.nan, np.nan
    med = np.median(vals)
    return med, np.median(np.abs(vals - med))


@pytest.mark.parametrize("by_type", [False, True])
def test_slot_profile_matches_brute_force(by_type: bool) -> None:
    frame, series = _frame()
    lay = layout(series, 30)
    prof = slot_profile(series.demand, lay, DAYS, by_type)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(frame), 300, replace=False):
        med, mad = _brute(frame, int(i), by_type)
        np.testing.assert_allclose([prof.median[i], prof.mad[i]], [med, mad], equal_nan=True)


def test_dst_fall_back_slot_holds_twelve_values() -> None:
    frame, series = _frame()
    lay = layout(series, 30)
    # Melbourne fell back at 03:00 AEDT on 7 Apr 2024: 02:00-03:00 local happened twice.
    day = frame["date"] == dt.date(2024, 4, 7)
    assert (frame.loc[day, "slot"] == 4).sum() == 12
    assert lay.n_slots == 48
    # The next days' profiles of slot 4 include all twelve.
    after = np.flatnonzero((frame["date"] == dt.date(2024, 4, 8)).to_numpy() & (lay.slot == 4))
    prof = slot_profile(series.demand, lay, DAYS, by_day_type=False)
    med, mad = _brute(frame, int(after[0]), by_type=False)
    np.testing.assert_allclose([prof.median[after[0]], prof.mad[after[0]]], [med, mad])


def test_profile_needs_its_full_look_back_and_ignores_its_own_day() -> None:
    _, series = _frame()
    lay = layout(series, 30)
    prof = slot_profile(series.demand, lay, DAYS, by_day_type=False)
    assert np.isnan(prof.median[lay.local_day < DAYS]).all()
    assert np.isfinite(prof.median[lay.local_day >= DAYS]).all()
    changed = series.demand.copy()
    changed[lay.local_day == 8] += 1e6
    again = slot_profile(changed, lay, DAYS, by_day_type=False)
    same = lay.local_day <= 8
    np.testing.assert_array_equal(prof.median[same], again.median[same])
    assert not np.array_equal(prof.median[lay.local_day == 9], again.median[lay.local_day == 9])


def test_robust_z_uses_the_floor() -> None:
    _, series = _frame()
    lay = layout(series, 30)
    prof = slot_profile(np.full(len(series), 7.0), lay, DAYS, by_day_type=False)
    z = prof.z(np.full(len(series), 107.0), 1.4826, 50.0)
    assert np.allclose(z[lay.local_day >= DAYS], 2.0)


def test_daily_robust_scale_matches_brute_force() -> None:
    frame, series = _frame()
    lay = layout(series, 30)
    values = np.diff(series.demand, prepend=np.nan)
    scale = daily_robust_scale(values, lay, DAYS, 1.4826, 10.0)
    for d in (DAYS, 9, 15):
        lo = frame["date"].iloc[0] + dt.timedelta(days=d - DAYS)
        hi = frame["date"].iloc[0] + dt.timedelta(days=d)
        sel = ((frame["date"] >= lo) & (frame["date"] < hi)).to_numpy()
        v = values[sel]
        v = v[np.isfinite(v)]
        expected = max(1.4826 * np.median(np.abs(v - np.median(v))), 10.0)
        np.testing.assert_allclose(scale[lay.local_day == d], expected)
    assert np.isnan(scale[lay.local_day < DAYS]).all()

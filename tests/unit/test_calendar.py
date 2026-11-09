import datetime as dt

import pandas as pd
import pytest

from gridcast.features.calendar import build_calendar_days, interval_calendar_features
from gridcast.timeutils import INTERVAL

MEL = "Australia/Melbourne"
START, END = dt.date(2021, 10, 1), dt.date(2026, 9, 30)


@pytest.fixture(scope="module")
def calendar() -> pd.DataFrame:
    return build_calendar_days("VIC", MEL, START, END)


def test_calendar_dst_days(calendar: pd.DataFrame) -> None:
    dst = calendar.loc[calendar["is_dst_change_day"], ["local_date", "n_intervals"]]
    as_dict = dict(zip(dst["local_date"], dst["n_intervals"], strict=True))
    # Melbourne: clocks forward first Sunday of October, back first Sunday of April.
    assert as_dict[dt.date(2021, 10, 3)] == 276
    assert as_dict[dt.date(2022, 4, 3)] == 300
    assert as_dict[dt.date(2024, 10, 6)] == 276
    assert as_dict[dt.date(2025, 4, 6)] == 300
    assert len(as_dict) == 10  # 5 forward + 5 back between Oct 2021 and Sep 2026
    assert all(d.weekday() == 6 for d in as_dict)  # always Sundays


def test_calendar_holidays_include_victorian_days(calendar: pd.DataFrame) -> None:
    hol = calendar.set_index("local_date")
    assert hol.loc[dt.date(2024, 11, 5), "holiday_name"] == "Melbourne Cup Day"
    assert "AFL Grand Final" in hol.loc[dt.date(2024, 9, 27), "holiday_name"]
    assert bool(hol.loc[dt.date(2022, 9, 22), "is_public_holiday"])  # National Day of Mourning
    assert not bool(hol.loc[dt.date(2024, 11, 6), "is_public_holiday"])


def test_utc_grid_matches_calendar_day_lengths(calendar: pd.DataFrame) -> None:
    """A contiguous UTC 5-min grid, bucketed by local date, reproduces n_intervals exactly."""
    starts = pd.Series(
        pd.date_range("2021-10-01 00:00", "2026-09-30 23:55", freq=INTERVAL, tz=MEL).tz_convert(
            "UTC"
        )
    )
    assert starts.diff().dropna().eq(INTERVAL).all()  # UTC series has no gaps/repeats
    per_day = starts.dt.tz_convert(MEL).dt.date.value_counts()
    expected = calendar.set_index("local_date")["n_intervals"]
    assert (per_day.sort_index() == expected.sort_index()).all()


def test_interval_features_across_autumn_transition(calendar: pd.DataFrame) -> None:
    # 2022-04-03: 03:00 AEDT -> 02:00 AEST, so local 02:00-02:55 happens twice.
    starts = pd.Series(
        pd.date_range("2022-04-02 14:00", "2022-04-02 17:55", freq=INTERVAL, tz="UTC")
    )
    feats = interval_calendar_features(starts, MEL, calendar)
    assert feats["is_dst_change_day"].all()
    assert feats["local_start"].dt.hour.value_counts()[2] == 24  # two hours' worth of 02:xx
    offsets = {ts.utcoffset() for ts in feats["local_start"]}
    assert offsets == {dt.timedelta(hours=11), dt.timedelta(hours=10)}


def test_interval_features_values(calendar: pd.DataFrame) -> None:
    starts = pd.Series(
        [
            pd.Timestamp("2024-11-05 07:00", tz="UTC"),  # Melbourne Cup, 18:00 AEDT
            pd.Timestamp("2024-07-06 02:30", tz="UTC"),  # Saturday 12:30 AEST, winter
        ]
    )
    feats = interval_calendar_features(starts, MEL, calendar)
    assert feats["local_hour"].tolist() == [18, 12]
    assert feats["is_public_holiday"].tolist() == [True, False]
    assert feats["is_weekend"].tolist() == [False, True]
    assert feats["day_type"].tolist() == ["non_working", "non_working"]
    assert feats["season"].tolist() == ["spring", "winter"]
    assert feats["slot_of_day"].tolist() == [216, 150]


def test_missing_calendar_dates_raise(calendar: pd.DataFrame) -> None:
    starts = pd.Series([pd.Timestamp("2030-01-01", tz="UTC")])
    with pytest.raises(KeyError, match="calendar_days missing"):
        interval_calendar_features(starts, MEL, calendar)

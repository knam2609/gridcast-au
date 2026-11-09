import datetime as dt

import pandas as pd
import pytest

from gridcast.timeutils import (
    local_day_n_intervals,
    month_bounds_utc,
    nem_naive_to_utc,
    parse_nem_timestamps,
)

MEL = "Australia/Melbourne"


def test_parse_nem_timestamps_fixed_offset() -> None:
    s = pd.Series(["2021/10/01 00:05:00", "2022/01/15 12:00:00"], dtype="string")
    out = parse_nem_timestamps(s)
    assert out.tolist() == [
        pd.Timestamp("2021-09-30 14:05", tz="UTC"),
        pd.Timestamp("2022-01-15 02:00", tz="UTC"),
    ]


def test_parse_nem_timestamps_rejects_garbage_and_missing() -> None:
    with pytest.raises(ValueError):
        parse_nem_timestamps(pd.Series(["2021-10-01T00:05"], dtype="string"))
    with pytest.raises(ValueError, match="missing"):
        parse_nem_timestamps(pd.Series([None], dtype="string"))


def test_nem_naive_to_utc() -> None:
    assert nem_naive_to_utc(dt.datetime(2024, 1, 1, 10)) == pd.Timestamp(
        "2024-01-01 00:00", tz="UTC"
    )
    with pytest.raises(ValueError):
        nem_naive_to_utc(dt.datetime(2024, 1, 1, tzinfo=dt.UTC))


def test_month_bounds_utc() -> None:
    first, last = month_bounds_utc(2021, 10)
    assert first == pd.Timestamp("2021-09-30 14:05", tz="UTC")
    assert last == pd.Timestamp("2021-10-31 14:00", tz="UTC")
    assert (last - first) / pd.Timedelta(minutes=5) + 1 == 31 * 288


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (dt.date(2021, 10, 3), 276),  # AEST -> AEDT (clocks forward)
        (dt.date(2022, 4, 3), 300),  # AEDT -> AEST (clocks back)
        (dt.date(2024, 10, 6), 276),
        (dt.date(2025, 4, 6), 300),
        (dt.date(2023, 7, 1), 288),
    ],
)
def test_local_day_length_across_dst(day: dt.date, expected: int) -> None:
    assert local_day_n_intervals(day, MEL) == expected

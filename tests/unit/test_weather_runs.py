import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gridcast.weather.runs import available_at_for, run_init_for

SIX_HOURS = pd.Timedelta(hours=6)


def _utc(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values), utc=True))


def test_run_init_examples_from_step0() -> None:
    # previous_day1 at 2026-09-28 05:00 comes from the 2026-09-27 00:00 run; at 06:00 from 06:00.
    valid = _utc("2026-09-28 05:00", "2026-09-28 06:00", "2026-09-28 23:00")
    assert run_init_for(valid, 1, SIX_HOURS).tolist() == list(
        _utc("2026-09-27 00:00", "2026-09-27 06:00", "2026-09-27 18:00")
    )
    assert run_init_for(valid, 2, SIX_HOURS).tolist() == list(
        _utc("2026-09-26 00:00", "2026-09-26 06:00", "2026-09-26 18:00")
    )


def test_offsets_use_fixed_24_hour_days_across_dst() -> None:
    # Melbourne clocks went forward on 2024-10-06; UTC arithmetic must ignore that.
    valid = _utc("2024-10-07 03:00")
    assert run_init_for(valid, 1, SIX_HOURS).iloc[0] == pd.Timestamp("2024-10-06 00:00", tz="UTC")


def test_available_at_adds_delay() -> None:
    run = _utc("2026-09-27 06:00")
    assert available_at_for(run, pd.Timedelta(hours=6)).iloc[0] == pd.Timestamp(
        "2026-09-27 12:00", tz="UTC"
    )


def test_rejects_naive_times_and_day_zero() -> None:
    with pytest.raises(ValueError, match="UTC"):
        run_init_for(pd.Series(pd.to_datetime(["2026-01-01"])), 1, SIX_HOURS)
    with pytest.raises(ValueError, match=">= 1"):
        run_init_for(_utc("2026-01-01"), 0, SIX_HOURS)


@given(
    hours=st.integers(min_value=0, max_value=24 * 365 * 6),
    offset=st.integers(min_value=1, max_value=7),
)
def test_run_init_property(hours: int, offset: int) -> None:
    valid = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=hours)
    run = run_init_for(pd.Series([valid]), offset, SIX_HOURS).iloc[0]
    lead_start = valid - pd.Timedelta(hours=24 * offset)
    assert run.hour % 6 == 0 and run.minute == 0
    assert run <= lead_start < run + SIX_HOURS

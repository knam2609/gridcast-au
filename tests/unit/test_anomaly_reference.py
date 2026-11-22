"""AEMO-flagged reference events for criterion A2 (M4 step 1a, m4_plan.md §3.1)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gridcast.anomalies.reference import (
    CLASS_O,
    CLASS_P,
    classify_intervals,
    event_period,
    reference_events,
    render_reference_events,
)
from gridcast.config import ReferenceEventsConfig, load_config
from gridcast.timeutils import NEM_TZ

P_BITS = [1, 2, 4]
O_BITS = [8, 16]


def _nem(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NEM_TZ).tz_convert("UTC")


def _market(flags: dict[str, tuple[int, int, float]], first: str, last: str) -> pd.DataFrame:
    """A 5-minute grid from ``first`` to ``last`` (interval ends, NEM time), unflagged except
    ``flags``: interval end → (APC bits, suspended, price)."""
    ends = pd.date_range(_nem(first), _nem(last), freq="5min")
    frame = pd.DataFrame(
        {
            "interval_end": ends,
            "total_demand_mw": 5000.0,
            "rrp_aud_mwh": 80.0,
            "apc_flag_bits": np.zeros(len(ends), dtype=np.int16),
            "market_suspended_flag": np.zeros(len(ends), dtype=np.int16),
        }
    )
    for when, (bits, suspended, price) in flags.items():
        row = frame.index[frame["interval_end"] == _nem(when)][0]
        frame.loc[row, ["apc_flag_bits", "market_suspended_flag", "rrp_aud_mwh"]] = (
            bits,
            suspended,
            price,
        )
    return frame


def test_class_p_wins_and_class_o_covers_suspension() -> None:
    apc = np.array([0, 1, 2, 4, 5, 8, 16, 17, 20, 24, 0])
    suspended = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
    got = classify_intervals(apc, suspended, P_BITS, O_BITS)
    assert list(got) == ["", "P", "P", "P", "P", "O", "O", "P", "P", "O", "O"]


def test_events_merge_under_the_gap_and_split_at_it() -> None:
    cfg = load_config()
    flags = {
        # P: a gap of 23 h 50 min between two intervals (under 24 h) -> one event ...
        "2024-02-13 13:20": (4, 0, 16600.0),
        "2024-02-14 13:15": (2, 0, 16600.0),
        # ... and one exactly 24 h after the previous end -> a new event.
        "2024-02-15 13:20": (4, 0, -1000.0),
        # O: a suspension inside the first P event is still its own class.
        "2024-02-13 13:25": (0, 1, 50.0),
        "2024-02-13 13:30": (16, 1, 60.0),
    }
    events = reference_events(_market(flags, "2024-02-13 00:05", "2024-02-16 00:00"), cfg)
    p = [e for e in events if e.cls == CLASS_P]
    o = [e for e in events if e.cls == CLASS_O]
    assert [e.event_id for e in p] == ["P01", "P02"]
    assert p[0].start == _nem("2024-02-13 13:15")
    assert p[0].end == _nem("2024-02-14 13:15")
    assert p[0].n_intervals == 2
    assert p[0].bits == (2, 4)
    assert (p[0].price_min, p[0].price_max) == (16600.0, 16600.0)
    assert p[1].start == _nem("2024-02-15 13:15")
    assert p[1].price_min == -1000.0
    assert [e.event_id for e in o] == ["O01"]
    assert (o[0].n_intervals, o[0].n_suspended, o[0].bits) == (2, 2, (16,))
    assert all(e.period == "development" for e in events)
    assert all(e.judged_by_a2 for e in p)
    assert not o[0].judged_by_a2


def test_periods_follow_the_nem_day_of_the_start() -> None:
    cfg = load_config()
    first = cfg.anomalies.market_first_day
    assert event_period(_nem(f"{first - dt.timedelta(days=1)} 23:55"), cfg) == "warm-up"
    assert event_period(_nem(f"{first} 00:00"), cfg) == "development"
    holdout = cfg.backtest.holdout_start
    assert event_period(_nem(f"{holdout - dt.timedelta(days=1)} 23:55"), cfg) == "development"
    assert event_period(_nem(f"{holdout} 00:00"), cfg) == "holdout"
    assert cfg.backtest.holdout_end is not None
    assert event_period(_nem(f"{cfg.backtest.holdout_end} 00:00"), cfg) == "after the holdout"


def test_holdout_class_p_events_are_not_judged() -> None:
    cfg = load_config()
    day = cfg.backtest.holdout_start + dt.timedelta(days=10)
    flags = {f"{day} 12:00": (4, 0, 17500.0)}
    events = reference_events(_market(flags, f"{day} 00:05", f"{day} 23:55"), cfg)
    assert [(e.event_id, e.period, e.judged_by_a2) for e in events] == [("P01", "holdout", False)]


def test_document_is_deterministic_and_lists_every_event() -> None:
    cfg = load_config()
    flags = {
        "2024-08-05 08:05": (4, 0, 17500.0),
        "2024-09-05 14:00": (0, 1, 40.0),
    }
    events = reference_events(_market(flags, "2024-08-05 00:05", "2024-09-06 00:00"), cfg)
    text = render_reference_events(events, cfg, "VIC1")
    assert text == render_reference_events(events, cfg, "VIC1")
    assert "| P01 | development | 2024-08-05 08:00 | 2024-08-05 08:05 |" in text
    assert "| O01 | development | 2024-09-05 13:55 |" in text
    assert "**A2 judges 1 class-P events:** P01." in text
    assert "APCFLAG bit 1, 2 or 4 set" in text


def test_config_rejects_overlapping_or_unknown_bits() -> None:
    with pytest.raises(ValidationError):
        ReferenceEventsConfig(class_p_apc_bits=[1, 8], class_o_apc_bits=[8])
    with pytest.raises(ValidationError):
        ReferenceEventsConfig(class_p_apc_bits=[3], class_o_apc_bits=[8])

"""The frozen recipe's own rules (m4_plan.md step 4): hash, case studies, cuts, band joining."""

import datetime as dt
from functools import cache

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies import profiles
from gridcast.anomalies.events import Event
from gridcast.anomalies.inputs import concat_bands
from gridcast.anomalies.recipe import (
    anomaly_config_hash,
    case_order,
    case_studies,
    cut_indices,
    dev_split,
    differences,
    forest_leaks,
    holdout_split,
    truncated_detection,
)
from gridcast.anomalies.run import Detection, detect
from gridcast.anomalies.series import QuantileBand
from gridcast.config import load_config
from gridcast.timeutils import INTERVAL, NEM_TZ
from tests.anomaly_fixtures import base_inputs, small_config, split

CFG = small_config()
SPLIT = split()


def _event(first: int, min_p: float | None, detectors: tuple[str, ...]) -> Event:
    return Event(
        first=first,
        last=first + 2,
        peak=first,
        detectors=detectors,
        detector_min_p=tuple(None if d.startswith("price_line") else min_p for d in detectors),
        tier="notable",
        min_p=min_p,
        categories=("unusual demand",),
    )


# --------------------------------------------------------------------------- the hash


def test_hash_ignores_paths_and_the_freeze_record() -> None:
    cfg = load_config()
    base = anomaly_config_hash(cfg)
    a = cfg.anomalies
    cfg.anomalies = a.model_copy(
        update={
            "output_dir": "elsewhere",
            "frozen_config_hash": "x",
            "frozen_commit": "y",
            "explain": a.explain.model_copy(update={"output_dir": "other"}),
        }
    )
    assert anomaly_config_hash(cfg) == base


def test_hash_sees_settings_and_split_dates() -> None:
    cfg = load_config()
    base = anomaly_config_hash(cfg)
    a = cfg.anomalies
    changed = cfg.model_copy(deep=True)
    changed.anomalies = a.model_copy(update={"cases": a.cases.model_copy(update={"n": 6})})
    assert anomaly_config_hash(changed) != base
    moved = cfg.model_copy(deep=True)
    moved.backtest = cfg.backtest.model_copy(update={"holdout_start": dt.date(2025, 11, 1)})
    assert anomaly_config_hash(moved) != base


def test_splits_follow_the_plan() -> None:
    cfg = load_config()
    dev, hold = dev_split(cfg), holdout_split(cfg)
    assert (dev.market_first_day, dev.residual_first_day) == (
        dt.date(2022, 5, 1),
        dt.date(2023, 6, 1),
    )
    assert dev.end_day == hold.market_first_day == hold.residual_first_day == dt.date(2025, 10, 1)
    assert hold.end_day == dt.date(2026, 10, 1)


# --------------------------------------------------------------------------- case studies


def test_case_order_ranks_by_p_then_corroboration_then_a_seeded_draw() -> None:
    events = [
        _event(0, 1e-4, ("demand_level",)),
        _event(10, 2e-5, ("demand_level",)),
        _event(20, 2e-5, ("demand_level", "price_level", "price_line_300")),
        _event(30, None, ("price_line_300",)),  # lines only: never a case study
        _event(40, 2e-5, ("price_level",)),
    ]
    order = case_order(events, seed=7)
    assert order[0] == 2  # two calibrated detectors beat one at the same p
    assert set(order[1:3]) == {1, 4} and order[3] == 0
    assert 3 not in order
    assert case_order(events, seed=7) == order  # seeded
    flipped = {tuple(case_order(events, seed=s)[1:3]) for s in range(20)}
    assert flipped == {(1, 4), (4, 1)}  # the draw decides the remaining tie


def test_named_days_are_added_once() -> None:
    series, _ = base_inputs()
    ends = series.ends

    def day_of(i: int) -> dt.date:
        return (ends[i] - INTERVAL).tz_convert(NEM_TZ).date()

    events = [
        _event(1000, 1e-3, ("demand_level",)),
        _event(5000, 2e-5, ("demand_level",)),  # rank 1
        _event(9000, 5e-4, ("price_level",)),
    ]
    cfg = CFG.model_copy(
        update={
            "cases": CFG.cases.model_copy(
                update={"n": 1, "named_days": [day_of(5000), day_of(9000), dt.date(1999, 1, 1)]}
            )
        }
    )
    picks = case_studies(events, ends, cfg, named=True)
    assert [(p.event, p.reason.split(" (")[0]) for p in picks] == [
        (1, "lowest p"),
        (2, f"named day {day_of(9000).isoformat()}"),
        (None, "named day 1999-01-01"),
    ]
    assert picks[0].also_named == (day_of(5000),)
    assert [p.event for p in case_studies(events, ends, cfg, named=False)] == [1]


# --------------------------------------------------------------------------- real-data cuts


@cache
def _full() -> Detection:
    return detect(*base_inputs(), SPLIT, CFG)


def test_cuts_are_seeded_inside_the_scored_period() -> None:
    series, _ = base_inputs()
    cuts = cut_indices(series.ends, SPLIT, CFG)
    start, stop = SPLIT.index_range(series.ends)
    assert len(cuts) == CFG.causality.real_cuts == len(set(cuts.tolist()))
    assert np.all(np.diff(cuts) > 0) and cuts[0] >= start and cuts[-1] < stop - 1
    assert np.array_equal(cuts, cut_indices(series.ends, SPLIT, CFG))


def test_the_cut_check_passes_on_causal_code() -> None:
    series, band = base_inputs()
    for c in cut_indices(series.ends, SPLIT, CFG):
        assert differences(_full(), truncated_detection(series, band, SPLIT, CFG, int(c)), c) == []
    assert forest_leaks(_full()) == []


def test_the_cut_check_catches_a_leaky_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    series, band = base_inputs()
    c = int(cut_indices(series.ends, SPLIT, CFG)[0])
    monkeypatch.setattr(
        profiles, "lookback_days", lambda day, days: np.arange(day - days // 2, day + days // 2)
    )
    leaky = detect(series, band, SPLIT, CFG)
    assert differences(leaky, truncated_detection(series, band, SPLIT, CFG, c), c) != []


# --------------------------------------------------------------------------- band joining


def _band(start: str, n: int, leads: int = 4) -> tuple[QuantileBand, np.ndarray]:
    times = pd.date_range(start, periods=n, freq="30min", tz="UTC")
    lower = np.arange(n * leads, dtype=float).reshape(n, leads)
    return QuantileBand(times, lower, lower + 100, np.ones((n, leads), bool)), lower + 50


def test_concat_bands_joins_consecutive_ranges() -> None:
    a, b = _band("2025-09-30 12:00", 4), _band("2025-09-30 14:00", 3)
    band, actual = concat_bands([a, b])
    assert len(band.issue_times) == 7 and str(band.issue_times.tz) == "UTC"
    assert np.array_equal(band.lower, np.vstack([a[0].lower, b[0].lower]))
    assert np.array_equal(actual, np.vstack([a[1], b[1]]))


def test_concat_bands_refuses_overlaps_and_lead_mismatches() -> None:
    with pytest.raises(ValueError, match="overlap"):
        concat_bands([_band("2025-09-30 12:00", 4), _band("2025-09-30 13:30", 3)])
    with pytest.raises(ValueError, match="leads"):
        concat_bands([_band("2025-09-30 12:00", 4), _band("2025-09-30 14:00", 3, leads=5)])

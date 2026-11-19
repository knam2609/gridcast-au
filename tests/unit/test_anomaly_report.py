"""The M4 report's own logic (step 4): the recomputation guard and the criteria's readings."""

import datetime as dt
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies.detectors import DETECTORS, PRICE_LEVEL
from gridcast.anomalies.dev import interval_frame
from gridcast.anomalies.reference import CLASS_P, ReferenceEvent
from gridcast.anomalies.report import (
    A3_CHECKS,
    Recomputed,
    ReportMismatch,
    StoredRun,
    a1_passed,
    a3_rows,
    forest_leaks,
    pval,
    recompute,
    reference_checks,
    smallest_reaching,
)
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.config import AppConfig, load_config
from gridcast.timeutils import INTERVAL, NEM_TZ
from tests.anomaly_fixtures import base_inputs, small_config, split


def _split() -> Split:
    series, _ = base_inputs()
    last = (series.ends[-1] - INTERVAL).tz_convert(NEM_TZ).date()
    return replace(split(), end_day=last + dt.timedelta(days=1))


def _cfg() -> AppConfig:
    cfg = load_config()
    cfg.anomalies = small_config()
    return cfg


@cache
def _detection() -> Detection:
    series, band = base_inputs()
    return detect(series, band, _split(), small_config())


def _stored(det: Detection) -> StoredRun:
    ends = det.run.ends
    events = pd.DataFrame(
        {
            "first_interval_end": [ends[e.first] for e in det.events],
            "last_interval_end": [ends[e.last] for e in det.events],
            "peak_interval_end": [ends[e.peak] for e in det.events],
            "tier": [e.tier for e in det.events],
            "detectors": [list(e.detectors) for e in det.events],
            "min_p": [e.min_p for e in det.events],
        }
    )
    counts = {
        "alerts": {d: int((t > 0).sum()) for d, t in det.tiers.items()},
        "events": len(det.events),
    }
    return StoredRun(
        1, "b", None, {"counts": counts}, Path(), interval_frame(det, _split()),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), events,
    )  # fmt: skip


def test_recompute_reproduces_a_stored_run() -> None:
    det = _detection()
    series, _ = base_inputs()
    rc = recompute(_stored(det), series.price, _cfg(), _split())
    assert len(rc.events) == len(det.events) > 0
    for d in DETECTORS:
        assert np.array_equal(rc.tiers[d], det.tiers[d])


def test_recompute_stops_on_a_changed_tier_event_or_count() -> None:
    det = _detection()
    series, _ = base_inputs()
    run = _stored(det)
    col = f"{PRICE_LEVEL}__tier"
    i = int(np.flatnonzero(run.intervals[col].to_numpy() > 0)[0])
    run.intervals.loc[i, col] = 0
    with pytest.raises(ReportMismatch, match="price_level tiers"):
        recompute(run, series.price, _cfg(), _split())
    run = _stored(det)
    run.events = run.events.iloc[1:]
    with pytest.raises(ReportMismatch, match="events"):
        recompute(run, series.price, _cfg(), _split())
    run = _stored(det)
    price = series.price.copy()
    period = run.intervals["price_line_5000__in_period"].to_numpy()
    j = int(np.flatnonzero(period & (price < 300))[0])
    price[j] = 10_000.0  # a new $5,000 line alert that the run never stored
    with pytest.raises(ReportMismatch, match="price_line_5000"):
        recompute(run, price, _cfg(), _split())


def test_a1_band_edges_are_inclusive() -> None:
    row = pd.Series({"p01": 0.005, "p001": 0.002})
    assert a1_passed(row) == (True, True)
    assert a1_passed(pd.Series({"p01": 0.0201, "p001": 0.00049})) == (False, False)


def _rc(n: int = 40) -> Recomputed:
    ends = pd.date_range("2024-01-01 00:05", periods=n, freq="5min", tz="UTC")
    p = {d: np.full(n, 0.5) for d in DETECTORS}
    t = {d: np.zeros(n, dtype=np.int8) for d in DETECTORS}
    on = {d: np.ones(n, dtype=bool) for d in DETECTORS}
    return Recomputed(
        DETECTORS, ends, np.zeros(n), t, p, {d: np.zeros(n) for d in DETECTORS}, on, []
    )


def _ref(start: str, end: str) -> ReferenceEvent:
    return ReferenceEvent(
        "P99", CLASS_P, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
        1, (4,), 0, 100.0, 100.0, "development",
    )  # fmt: skip


def test_a2_window_is_thirty_minutes_either_side_by_interval_end() -> None:
    rc = _rc()
    ref = _ref("2024-01-01 01:00", "2024-01-01 01:05")
    # 00:30 is exactly 30 min before the start: inside. 00:25 is outside.
    k = int(rc.ends.get_loc(pd.Timestamp("2024-01-01 00:30", tz="UTC")))
    rc.tiers[PRICE_LEVEL][k], rc.pvalues[PRICE_LEVEL][k] = 1, 0.0008
    assert reference_checks([ref], rc, 30)[0].fired == ((PRICE_LEVEL, 0.0008),)
    rc2 = _rc()
    k2 = int(rc2.ends.get_loc(pd.Timestamp("2024-01-01 00:25", tz="UTC")))
    rc2.tiers[PRICE_LEVEL][k2], rc2.pvalues[PRICE_LEVEL][k2] = 1, 0.0008
    check = reference_checks([ref], rc2, 30)[0]
    assert not check.caught and check.lowest == (DETECTORS[0], 0.5)


def test_a3_lookups_and_thresholds() -> None:
    power = pd.DataFrame(
        {
            "label": PRICE_LEVEL,
            "detector": PRICE_LEVEL,
            "size": [-1000.0, -300.0, 300.0, 15000.0],
            "reported_only": False,
            "n": 1000,
            "power": [0.95, 0.6, 0.1, 0.92],
        }
    )
    rows = {(d, s): p for d, s, _, p in a3_rows(power)}
    assert rows[(PRICE_LEVEL, 15000.0)] == 0.92 and rows[(PRICE_LEVEL, -1000.0)] == 0.95
    assert len(rows) == len(A3_CHECKS) and np.isnan(rows[(DETECTORS[0], 2000.0)])
    assert smallest_reaching(power, PRICE_LEVEL, -1, 0.5) == -300.0
    assert smallest_reaching(power, PRICE_LEVEL, -1, 0.9) == -1000.0
    assert smallest_reaching(power, PRICE_LEVEL, 1, 0.95) is None


def test_forest_leaks_and_pvalue_format() -> None:
    forests = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02"],
            "month_start": pd.to_datetime(["2023-12-31 14:00", "2024-01-31 14:00"], utc=True),
            "train_last_end": pd.to_datetime(["2023-12-31 14:00", "2024-01-31 14:05"], utc=True),
        }
    )
    assert forest_leaks(forests) == ["2024-02"]
    assert pval(1.9077e-5) == "1.91 × 10⁻⁵"
    assert pval(None) == "–"

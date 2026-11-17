"""The event merge rule (m4_plan.md §2.4)."""

import numpy as np

from gridcast.anomalies.detectors import (
    DEMAND_LEVEL,
    LINE_NOTABLE,
    LINE_SEVERE,
    PRICE_LEVEL,
    RESIDUAL_SHORT,
)
from gridcast.anomalies.events import merge_events

N = 100


def _inputs(alerts: dict[str, dict[int, tuple[int, float, float]]]):
    """alerts: detector -> {index: (tier, p, score)} (p, score ignored for lines)."""
    tiers, pvalues, scores = {}, {}, {}
    for d, rows in alerts.items():
        t, p, s = np.zeros(N, np.int8), np.full(N, 0.5), np.zeros(N)
        for i, (tier, pp, ss) in rows.items():
            t[i], p[i], s[i] = tier, pp, ss
        tiers[d], pvalues[d], scores[d] = t, p, s
    return tiers, pvalues, scores


def _events(alerts, price=None, start=0, stop=N, gap=6):
    tiers, pvalues, scores = _inputs(alerts)
    price = np.zeros(N) if price is None else price
    return merge_events(tiers, pvalues, scores, price, start, stop, gap)


def test_alerts_30_minutes_apart_merge_35_do_not() -> None:
    ev = _events({DEMAND_LEVEL: {10: (1, 1e-3, 5), 16: (1, 1e-3, 5), 23: (1, 1e-3, 5)}})
    assert [(e.first, e.last) for e in ev] == [(10, 16), (23, 23)]


def test_alerts_from_different_detectors_merge() -> None:
    ev = _events({DEMAND_LEVEL: {10: (1, 8e-4, 5)}, PRICE_LEVEL: {14: (2, 5e-5, 9)}})
    assert len(ev) == 1
    e = ev[0]
    assert e.detectors == (DEMAND_LEVEL, PRICE_LEVEL)
    assert e.detector_min_p == (8e-4, 5e-5)
    assert e.min_p == 5e-5 and e.peak == 14 and e.tier == "severe"
    assert e.categories == ("unusual demand", "unusual price")


def test_peak_ties_by_largest_score_then_earliest() -> None:
    ev = _events({DEMAND_LEVEL: {10: (1, 1e-4, 3), 12: (1, 1e-4, 7), 14: (1, 1e-4, 7)}})
    assert ev[0].peak == 12


def test_lines_only_event() -> None:
    price = np.zeros(N)
    price[30:33] = [400, 6000, 900]
    ev = _events(
        {LINE_NOTABLE: {30: (1, 0, 0), 31: (1, 0, 0), 32: (1, 0, 0)}, LINE_SEVERE: {31: (2, 0, 0)}},
        price,
    )
    e = ev[0]
    assert e.lines_only and e.min_p is None and e.peak == 31 and e.tier == "severe"
    assert e.detector_min_p == (None, None) and e.categories == ("high price",)


def test_a_line_with_a_calibrated_detector_is_not_lines_only() -> None:
    ev = _events({RESIDUAL_SHORT: {50: (1, 9e-4, 2)}, LINE_NOTABLE: {52: (1, 0, 0)}})
    e = ev[0]
    assert not e.lines_only and e.peak == 50 and e.tier == "notable"
    assert e.categories == ("unexpected demand (short-term)", "high price")


def test_events_stay_inside_their_range() -> None:
    alerts = {DEMAND_LEVEL: {48: (1, 1e-3, 5), 50: (1, 1e-3, 5), 52: (1, 1e-3, 5)}}
    assert [(e.first, e.last) for e in _events(alerts, stop=50)] == [(48, 48)]
    assert [(e.first, e.last) for e in _events(alerts, start=50)] == [(50, 52)]


def test_no_alerts_no_events() -> None:
    assert _events({DEMAND_LEVEL: {}}) == []

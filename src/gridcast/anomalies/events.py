"""Alerts merged into events (m4_plan.md §2.4).

Notable-or-severe alerts from any detector, price lines included, whose interval ends are at
most ``merge_gap_minutes`` (30 min, 6 intervals) apart form one event. The rule works live:
an event opens at its first alert and closes after 30 quiet minutes, so an event's opening
never depends on later data.

Per event:

* first and last alert interval, and the **peak**: the alert of a calibrated detector with
  the lowest p, ties by the largest score, then the earliest interval. Scores of different
  detectors are on different scales, so a tie across detectors is settled by an arbitrary
  but fixed rule. A lines-only event's peak is its highest price (earliest on ties);
* every detector that fired, with its lowest p (None for the lines);
* the tier: severe if any severe alert (L2 included), else notable;
* the categories of the detectors that fired, in §2.4's order.

Events are merged within one index range (a split's scored period), so an event never
crosses the development/holdout boundary and the holdout run's development part can
reproduce the development batch exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gridcast.anomalies.detectors import CALIBRATED, CATEGORIES, CATEGORY, LINES
from gridcast.anomalies.pvalues import NOTABLE, SEVERE, TIER_NAMES

# Canonical order: every version's calibrated detectors (a run has one D6), then the lines.
ORDER = (*CALIBRATED, *LINES)


@dataclass(frozen=True, slots=True)
class Event:
    first: int  # interval index of the first alert
    last: int  # interval index of the last alert
    peak: int
    detectors: tuple[str, ...]  # in §2.3's order, lines last
    detector_min_p: tuple[float | None, ...]  # lowest p per detector (None: a line)
    tier: str  # "notable" or "severe"
    min_p: float | None  # lowest p of a calibrated detector; None if lines only
    categories: tuple[str, ...]

    @property
    def lines_only(self) -> bool:
        return all(d in LINES for d in self.detectors)


def alert_matrix(tiers: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
    """Detectors present (in canonical order) and their ``(k, n)`` tier rows."""
    names = tuple(d for d in ORDER if d in tiers)
    unknown = set(tiers) - set(ORDER)
    if unknown:
        raise ValueError(f"unknown detectors {sorted(unknown)}")
    return names, np.vstack([np.asarray(tiers[d], dtype=np.int8) for d in names])


def merge_events(
    tiers: Mapping[str, np.ndarray],
    pvalues: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    price: np.ndarray,
    start: int,
    stop: int,
    gap_intervals: int,
) -> list[Event]:
    """Events from the alerts of intervals ``[start, stop)``, in time order."""
    names, tier_rows = alert_matrix(tiers)
    window = tier_rows[:, start:stop]
    alert_idx = np.flatnonzero((window >= NOTABLE).any(axis=0)) + start
    if alert_idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(alert_idx) > gap_intervals) + 1
    events = []
    for group in np.split(alert_idx, breaks):
        first, last = int(group[0]), int(group[-1])
        span = tier_rows[:, first : last + 1]
        fired = [k for k in range(len(names)) if (span[k] >= NOTABLE).any()]
        detectors = tuple(names[k] for k in fired)
        min_ps: list[float | None] = []
        best: tuple[float, float, int] | None = None
        for k in fired:
            name = names[k]
            if name in LINES:
                min_ps.append(None)
                continue
            rows = first + np.flatnonzero(span[k] >= NOTABLE)
            p = np.asarray(pvalues[name])[rows]
            s = np.asarray(scores[name])[rows]
            min_ps.append(float(p.min()))
            o = int(np.lexsort((rows, -s, p))[0])
            key = (float(p[o]), -float(s[o]), int(rows[o]))
            if best is None or key < best:
                best = key
        if best is None:
            seg = np.asarray(price)[first : last + 1]
            peak = first + int(np.argmax(seg))
        else:
            peak = best[2]
        severe = bool((span >= SEVERE).any())
        calibrated = [p for p in min_ps if p is not None]
        events.append(
            Event(
                first=first,
                last=last,
                peak=peak,
                detectors=detectors,
                detector_min_p=tuple(min_ps),
                tier=TIER_NAMES[SEVERE if severe else NOTABLE],
                min_p=min(calibrated) if calibrated else None,
                categories=tuple(c for c in CATEGORIES if any(CATEGORY[d] == c for d in detectors)),
            )
        )
    return events

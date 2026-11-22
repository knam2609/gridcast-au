"""Conformal p-values against a trailing window of each detector's own scores (§2.2).

For a detector's score ``s(t)`` at interval ``t``::

    p(t) = (1 + #{s in the window : s ≥ s(t)}) / (1 + n)

where the window holds the detector's ``n`` valid scores of the intervals ending in
``[t − window, t)``: 26 weeks is 52,416 intervals, so ``p`` can go as low as about
1.9 × 10⁻⁵. Ties count against the alert. A score at ``t`` is known at ``t``'s end, so every
score in the window was known before ``t``.

The windows reuse M3's Fenwick structure (:class:`~gridcast.intervals.rolling.
RollingOrderStats`, frozen code that is not modified here): every valid score is ranked
once per detector, each window is a Fenwick tree over those ranks, and the count of window
scores at or above a value is the window size minus a prefix sum. Ranking up front does not
leak: a count only depends on the scores inside the window (the causality tests check it).

:func:`window_pvalues` also answers *extra* queries, ``(interval, detector, value)``, against
the same unmodified windows. The injection harness (§2.9) scores injected values this way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridcast.intervals.rolling import RollingOrderStats


class RollingCounts(RollingOrderStats):
    """One trailing window per column (detector), with counts of scores at or above a value.

    After ``advance(i)``, column ``h``'s window holds its valid scores at rows
    ``[i − window, i − 1]``.
    """

    def __init__(self, scores: np.ndarray, valid: np.ndarray, window: int) -> None:
        n, n_cols = scores.shape
        super().__init__(
            np.where(valid, scores, 0.0),
            valid,
            np.zeros((n, n_cols), dtype=np.int16),
            1,
            np.ones(n_cols, dtype=np.int64),
            [window],
        )
        # Sorted valid scores of each column: rank r sits at position off + r.
        self._sorted = [
            self._vals[self._off[h, 0] + 1 : self._off[h, 0] + self._size[h, 0] + 1]
            for h in range(n_cols)
        ]

    def count_at_least(self, cols: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Window scores ``≥ values[q]`` in column ``cols[q]``, and each window's size."""
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        below_rank = np.empty(len(cols), dtype=np.int64)
        for h in np.unique(cols):
            m = cols == h
            below_rank[m] = np.searchsorted(self._sorted[h], values[m], side="left")
        # Window scores ranked 1..below_rank are exactly those < value.
        base = self._off[cols, 0]
        below = np.zeros(len(cols), dtype=np.int64)
        pos = below_rank
        live = pos > 0
        while live.any():
            below[live] += self._tree_flat[base[live] + pos[live]]
            pos = np.where(live, pos - (pos & -pos), 0)
            live = pos > 0
        n_window = self._count[0, cols, 0]
        return n_window - below, n_window


@dataclass(frozen=True, slots=True)
class Queries:
    """Extra values to score against the unmodified windows: one per (row, column)."""

    rows: np.ndarray  # (Q,) interval index
    cols: np.ndarray  # (Q,) detector column
    values: np.ndarray  # (Q,) score

    def __post_init__(self) -> None:
        if not (len(self.rows) == len(self.cols) == len(self.values)):
            raise ValueError("queries need rows, cols and values of one length")
        if len(self.values) and not np.all(np.isfinite(self.values)):
            raise ValueError("query values must be finite")


@dataclass(frozen=True, slots=True)
class WindowPValues:
    """p-values of every valid score, plus the extra queries' (in their given order)."""

    count: np.ndarray  # (n, H) window scores ≥ s(t); -1 where s(t) is not valid
    n_window: np.ndarray  # (n, H) valid scores in the window; -1 where s(t) is not valid
    extra_count: np.ndarray  # (Q,)
    extra_n_window: np.ndarray  # (Q,)

    @property
    def p(self) -> np.ndarray:
        return conformal_p(self.count, self.n_window)

    @property
    def extra_p(self) -> np.ndarray:
        return conformal_p(self.extra_count, self.extra_n_window)


def conformal_p(count: np.ndarray, n_window: np.ndarray) -> np.ndarray:
    """``(1 + count) / (1 + n)``; NaN where ``n`` is negative (no score)."""
    count = np.asarray(count)
    n_window = np.asarray(n_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (1.0 + count) / (1.0 + n_window)
    return np.where(n_window >= 0, p, np.nan)


def window_pvalues(
    scores: np.ndarray, valid: np.ndarray, window: int, extra: Queries | None = None
) -> WindowPValues:
    """Sweep the intervals once and score every valid ``scores[t, h]`` (and ``extra``)."""
    n, n_cols = scores.shape
    if valid.shape != scores.shape:
        raise ValueError("scores and valid must have the same shape")
    stats = RollingCounts(scores, valid, window)
    count = np.full((n, n_cols), -1, dtype=np.int64)
    n_window = np.full((n, n_cols), -1, dtype=np.int64)

    if extra is None:
        extra = Queries(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    q_rows = np.asarray(extra.rows, dtype=np.int64)
    if len(q_rows) and (q_rows.min() < 0 or q_rows.max() >= n):
        raise ValueError("query rows outside the series")
    order = np.argsort(q_rows, kind="stable")
    bounds = np.searchsorted(q_rows[order], np.arange(n + 1), side="left")
    extra_count = np.zeros(len(q_rows), dtype=np.int64)
    extra_n = np.zeros(len(q_rows), dtype=np.int64)
    cols_all = np.arange(n_cols)

    for i in range(n):
        stats.advance(i)
        cols = cols_all[valid[i]]
        if cols.size:
            c, m = stats.count_at_least(cols, scores[i, cols])
            count[i, cols] = c
            n_window[i, cols] = m
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            idx = order[lo:hi]
            c, m = stats.count_at_least(extra.cols[idx], extra.values[idx])
            extra_count[idx] = c
            extra_n[idx] = m
    return WindowPValues(count, n_window, extra_count, extra_n)


def static_pvalues(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """p-values of ``values`` against one fixed set of ``reference`` scores (D6)."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    count = len(ref) - np.searchsorted(ref, values, side="left")
    p = (1.0 + count) / (1.0 + len(ref))
    return np.where(np.isfinite(values), p, np.nan)


NONE, NOTABLE, SEVERE = 0, 1, 2
TIER_NAMES = {NOTABLE: "notable", SEVERE: "severe"}


def tiers(p: np.ndarray, notable: float, severe: float) -> np.ndarray:
    """0 (none), 1 (notable: p ≤ ``notable``) or 2 (severe: p ≤ ``severe``); NaN → 0."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros(p.shape, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out[p <= notable] = NOTABLE
        out[p <= severe] = SEVERE
    return out

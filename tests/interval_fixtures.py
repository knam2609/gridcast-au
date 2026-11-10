"""Synthetic interval problems and a slow reference of the rolling conformal run (M3).

The reference follows the definitions in ``gridcast.intervals.conformal`` literally, with
loops, time arithmetic and plain sorting (no lags, ranks, trees or negated matrices), so
that the fast implementation can be checked against it exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from gridcast.intervals.conformal import (
    TAILS,
    CandidateGrid,
    ConformalProblem,
    band_label,
    point_problem,
    quantile_problem,
    tod_labels,
)
from gridcast.intervals.scores import LEVELS, alpha

MEL = "Australia/Melbourne"


def synthetic_point(
    n_days: int = 21,
    n_leads: int = 18,
    seed: int = 0,
    nan_share: float = 0.03,
    start: str = "2024-03-30 14:00",  # NEM midnight; DST ends in Melbourne on 7 Apr 2024
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Issue times every 30 min, actuals and a point forecast whose bias drifts."""
    rng = np.random.default_rng(seed)
    n = n_days * 48
    times = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="30min")
    lead = np.arange(1, n_leads + 1)
    tgt = np.arange(n)[:, None] * 6 + lead[None, :]
    level = 5000 + 800 * np.sin(2 * np.pi * tgt / 288)
    y = level + rng.normal(0, 60, (n, n_leads)) * np.sqrt(lead / 6)
    drift = 80 * np.sin(2 * np.pi * np.arange(n) / (n / 1.5))[:, None]
    point = level + drift + rng.normal(0, 40, (n, n_leads))
    y[rng.random(y.shape) < nan_share] = np.nan
    point[rng.random(point.shape) < nan_share / 2] = np.nan
    return times, np.round(y, 2), np.round(point, 2).astype(np.float32).astype(np.float64)


def make_point_problem(
    times: pd.DatetimeIndex,
    y: np.ndarray,
    point: np.ndarray,
    block_starts: Sequence[int] = (0, 8, 16),
    levels: Sequence[int] = LEVELS,
) -> ConformalProblem:
    valid = np.isfinite(y) & np.isfinite(point)
    groups = _groups(times, y.shape, block_starts)
    return point_problem(times, 6, 0, y, point, valid, groups, levels)


def _groups(times: pd.DatetimeIndex, shape: tuple[int, int], block_starts: Sequence[int]) -> dict:
    return {
        "lead": (np.zeros(shape, dtype=np.int8), 1),
        "lead_tod": tod_labels(times, shape[1], 6, MEL, block_starts),
    }


def synthetic_quantiles(
    point: np.ndarray, levels: Sequence[int] = LEVELS, seed: int = 2
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Asymmetric quantile forecasts around ``point``, too narrow (CQR must widen them)."""
    rng = np.random.default_rng(seed)
    out = {}
    for level in levels:
        half = 40 * level / 100 * rng.uniform(0.7, 1.3, point.shape)
        out[level] = (point - 0.6 * half, point + 1.2 * half)
    return out


def make_quantile_problem(
    times: pd.DatetimeIndex,
    y: np.ndarray,
    quantiles: dict[int, tuple[np.ndarray, np.ndarray]],
    block_starts: Sequence[int] = (0, 8, 16),
) -> ConformalProblem:
    valid = np.isfinite(y)
    for lo, hi in quantiles.values():
        valid &= np.isfinite(lo) & np.isfinite(hi)
    return quantile_problem(times, 6, 0, y, quantiles, valid, _groups(times, y.shape, block_starts))


def _minutes(times: pd.DatetimeIndex) -> np.ndarray:
    return ((times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()


def reference_candidate(
    problem: ConformalProblem,
    window_weeks: float,
    grouping: str,
    gamma: float,
    levels: Sequence[int],
) -> dict[tuple[int, str], np.ndarray]:
    """Bounds of one candidate stream at every issue time, by (level, tail)."""
    n, n_leads = problem.shape
    t = _minutes(problem.origin_times)
    issue_minutes = problem.issue_every_steps * 5
    lat = problem.latency_steps * 5
    window = round(window_weeks * 7 * 1440 / issue_minutes)
    labels = problem.groups[grouping][0]
    sign = {"lo": -1.0, "hi": 1.0}
    out = {(lv, tl): np.full((n, n_leads), np.nan) for lv in levels for tl in TAILS}
    beta = {(lv, tl): np.full(n_leads, alpha(lv) / 2) for lv in levels for tl in TAILS}

    def last_known(h: int, i: int) -> int:
        """Latest issue time whose lead-h target has ended by issue time i (-1 if none)."""
        j = i - 1
        while j >= 0 and t[j] + 5 * h + lat > t[i]:
            j -= 1
        return j

    for i in range(n):
        for h in range(1, n_leads + 1):
            col = h - 1
            last = last_known(h, i)
            # ACI: outcomes that become known at issue time i (known now, not before).
            before = last_known(h, i - 1) if i > 0 else -1
            for j in range(before + 1, last + 1):
                if not problem.valid[j, col]:
                    continue
                for lv in levels:
                    lo, hi = out[(lv, "lo")][j, col], out[(lv, "hi")][j, col]
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        continue
                    yj = problem.y[j, col]
                    for tl, miss in (("lo", yj < lo), ("hi", yj > hi)):
                        b = beta[(lv, tl)]
                        b[col] = min(max(b[col] + gamma * (alpha(lv) / 2 - miss), 0.0), 0.5)
            # Calibration window: the latest `window` issue times with a known target.
            recent = range(max(0, last - window + 1), last + 1)
            pool = [j for j in recent if problem.valid[j, col] and labels[j, col] == labels[i, col]]
            for lv in levels:
                for tl in TAILS:
                    spec = problem.tails[(lv, tl)]
                    if not pool:
                        continue
                    scores = sorted(
                        sign[tl] * (problem.y[j, col] - spec.base[j, col]) for j in pool
                    )
                    m = len(scores)
                    k = math.ceil((m + 1) * (1 - beta[(lv, tl)][col]) - 1e-9)
                    k = min(max(k, 1), m)
                    out[(lv, tl)][i, col] = spec.base[i, col] + sign[tl] * scores[k - 1]
    return out


def reference_conformal(
    problem: ConformalProblem,
    grid: CandidateGrid,
    levels: Sequence[int],
    retrains: Sequence[pd.Timestamp],
    bands: Sequence[tuple[int, int]],
    selection_weeks: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[tuple[str, int, str, str]]]:
    """Selected lower/upper bounds from the first retrain on, and the choices made."""
    n, n_leads = problem.shape
    t = _minutes(problem.origin_times)
    streams = {
        c.key: reference_candidate(problem, c.window_weeks, c.grouping, c.gamma, levels)
        for c in grid.candidates()
    }
    span = round(selection_weeks * 7 * 1440)
    r_idx = [int(np.searchsorted(t, _minutes(pd.DatetimeIndex([r]))[0])) for r in retrains]
    i0 = r_idx[0]
    lower = {lv: np.full((n - i0, n_leads), np.nan) for lv in levels}
    upper = {lv: np.full((n - i0, n_leads), np.nan) for lv in levels}
    choices = []
    for q, ri in enumerate(r_idx):
        end = r_idx[q + 1] if q + 1 < len(r_idx) else n
        for lv in levels:
            a = alpha(lv)
            for lo_h, hi_h in bands:
                cols = range(lo_h - 1, hi_h)
                pairs = [
                    (j, c)
                    for j in range(ri)
                    for c in cols
                    if t[j] >= t[ri] - span and t[j] + 5 * (c + 1) <= t[ri] and problem.valid[j, c]
                ]
                best, best_score, best_n = None, math.inf, -1
                counts = {}
                for key, s in streams.items():
                    vals = []
                    for j, c in pairs:
                        lo, hi, yy = s[(lv, "lo")][j, c], s[(lv, "hi")][j, c], problem.y[j, c]
                        if np.isfinite(lo) and np.isfinite(hi):
                            vals.append((hi - lo) + 2 / a * (max(lo - yy, 0) + max(yy - hi, 0)))
                    counts[key] = vals
                    best_n = max(best_n, len(vals))
                for key, vals in counts.items():
                    if len(vals) == best_n and best_n > 0:
                        mean = sum(vals) / len(vals)
                        if mean < best_score:
                            best, best_score = key, mean
                assert best is not None
                choices.append(
                    (problem.origin_times[ri].isoformat(), lv, band_label((lo_h, hi_h)), best)
                )
                for i in range(ri, end):
                    for c in cols:
                        lo = streams[best][(lv, "lo")][i, c]
                        hi = streams[best][(lv, "hi")][i, c]
                        if lo > hi:
                            lo = hi = (lo + hi) / 2
                        lower[lv][i - i0, c] = lo
                        upper[lv][i - i0, c] = hi
    return lower, upper, choices

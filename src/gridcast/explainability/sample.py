"""Which pairs SHAP explains, and their feature rows (m4_plan.md §2.6).

* **Global sample:** per (retrain, lead band), ``explain.sample_pairs`` (20,000) pairs drawn
  without replacement, uniformly over the fit's scored pairs in the period: issue times in
  the fit's fold and the period, leads in the band, and M2's common mask (the pairs every
  model of the batch forecast and that have an actual). The draw is seeded by the anomaly
  seed, the retrain date and the band, so it never depends on another fit's draw.
* **Feature rows** are rebuilt with M2's frozen builder, one lead at a time, and converted
  to float32 exactly as M2's ``Sample.from_matrix`` did.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.explainability.stored import StoredFit
from gridcast.features.builder import FeatureBuilder


@dataclass(frozen=True, slots=True)
class Period:
    name: str  # "dev" or "holdout" (or "smoke")
    start: pd.Timestamp  # first issue time (UTC), inclusive
    end: pd.Timestamp  # last issue time (UTC), exclusive


@dataclass(frozen=True, slots=True)
class Pairs:
    """(issue time, lead) pairs as rows and columns of a stored batch."""

    rows: np.ndarray  # int64: issue-time row of the stored batch
    leads: np.ndarray  # int64: 1…H

    def __len__(self) -> int:
        return len(self.rows)


def fit_seed(seed: int, fit: StoredFit) -> list[int]:
    """Seed sequence of one fit's draw: the anomaly seed, the retrain date and the band."""
    return [seed, dt.date.fromisoformat(fit.retrain).toordinal(), fit.band[0], fit.band[1]]


def eligible_pairs(
    fit: StoredFit, issue_times: pd.DatetimeIndex, mask: np.ndarray, period: Period
) -> Pairs:
    """Every scored pair of ``fit`` inside ``period``, in (row, lead) order."""
    if mask.shape[0] != len(issue_times):
        raise ValueError("the mask must have one row per issue time")
    times_ok = np.asarray((issue_times >= period.start) & (issue_times < period.end))
    times_ok &= np.asarray(issue_times >= fit.fold_start)
    if fit.fold_end is not None:
        times_ok &= np.asarray(issue_times < fit.fold_end)
    lo, hi = fit.band
    sub = np.zeros_like(mask, dtype=bool)
    sub[:, lo - 1 : hi] = mask[:, lo - 1 : hi]
    sub &= times_ok[:, None]
    rows, cols = np.nonzero(sub)
    return Pairs(rows.astype(np.int64), cols.astype(np.int64) + 1)


def sample_pairs(
    fit: StoredFit,
    issue_times: pd.DatetimeIndex,
    mask: np.ndarray,
    period: Period,
    n: int,
    seed: int,
) -> Pairs:
    """Up to ``n`` of the fit's eligible pairs, seeded, sorted by (row, lead)."""
    pool = eligible_pairs(fit, issue_times, mask, period)
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(fit_seed(seed, fit))
    take = np.sort(rng.choice(len(pool), size=n, replace=False))
    return Pairs(pool.rows[take], pool.leads[take])


def origin_positions(builder: FeatureBuilder, issue_times: pd.DatetimeIndex) -> np.ndarray:
    """Positions of issue times on the builder's data grid (each must lie on it)."""
    times = builder.ctx.times
    pos = np.asarray(times.searchsorted(issue_times), dtype=np.int64)
    bad = (pos >= len(times)) | (times[np.minimum(pos, len(times) - 1)] != issue_times)
    if np.any(bad):
        raise LookupError(f"{int(bad.sum())} issue times are not on the data grid")
    return pos


def feature_rows(
    builder: FeatureBuilder, variant: str, origins: np.ndarray, leads: np.ndarray
) -> np.ndarray:
    """``(n, p)`` float32 feature rows of the pairs (origin position, lead), in their order.

    Built one lead at a time (each pair's row depends only on its own origin and target), and
    converted as M2's ``Sample.from_matrix``: ``DataFrame.to_numpy(float32, na_value=nan)``.
    """
    origins = np.asarray(origins, dtype=np.int64)
    leads = np.asarray(leads, dtype=np.int64)
    if origins.shape != leads.shape:
        raise ValueError("one lead per origin")
    out: np.ndarray | None = None
    for h in np.unique(leads):
        sel = np.flatnonzero(leads == h)
        fm = builder.build(variant, origins[sel], np.array([h]))
        X = fm.X.to_numpy(dtype=np.float32, na_value=np.nan)
        if out is None:
            out = np.full((len(origins), X.shape[1]), np.nan, dtype=np.float32)
        out[sel] = X
    if out is None:
        raise ValueError("no pairs")
    return out

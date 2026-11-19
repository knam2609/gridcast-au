"""D6 v2: a robust distance, refitted at each NEM-month start (m4_plan.md §2.10).

The score of an interval with features f is √((f − m)ᵀ S⁻¹ (f − m)) under its NEM month's
model, fitted on the previous ``train_weeks`` of complete rows (if at least
``min_train_rows``):

* m is the coordinate-wise median of the training rows;
* S is the Ledoit–Wolf shrinkage covariance of the training rows about m (scikit-learn's
  ``LedoitWolf(assume_centered=True)`` on f − m), so S is invertible even when a feature
  barely varies.

Unlike an isolation tree, the distance keeps growing beyond the training range, so a
disturbance in one feature is not capped at the training extreme (v1's diagnosis). The
p-value is not taken here: D6 v2 is a rolling detector, scored against the trailing window
of its own out-of-sample scores like D1–D5 (:func:`gridcast.anomalies.run.run_detectors`).

Scoring uses fixed-order element-wise sums, not a matrix product, so a row's score is the
same bit for bit whichever rows are scored with it (the cut checks and the injection harness
rely on that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from gridcast.anomalies.isolation import month_start_utc
from gridcast.anomalies.series import Layout, epoch_ns
from gridcast.config import RobustDistanceConfig
from gridcast.timeutils import INTERVAL


@dataclass(frozen=True, slots=True)
class DistanceModel:
    month: int  # 12 * year + month − 1 (NEM time)
    start: pd.Timestamp  # UTC instant of the month's start
    centre: np.ndarray  # (k,) coordinate-wise median of the training rows
    precision: np.ndarray  # (k, k) inverse of the shrunk covariance about the centre
    shrinkage: float  # Ledoit–Wolf's shrinkage intensity
    train_first_end: pd.Timestamp  # interval end of the first training row
    train_last_end: pd.Timestamp  # interval end of the last training row (≤ start)
    n_train: int

    def score(self, rows: np.ndarray) -> np.ndarray:
        """The distance of each row (``(m, k)``) from the centre, in a fixed summation order."""
        d = np.asarray(rows, dtype=np.float64) - self.centre
        k = len(self.centre)
        y = np.zeros_like(d)
        for j in range(k):
            y += d[:, j : j + 1] * self.precision[j]
        q = np.zeros(len(d))
        for j in range(k):
            q += y[:, j] * d[:, j]
        return np.sqrt(np.maximum(q, 0.0))


def fit_distance(
    rows: np.ndarray, month: int, ends: pd.DatetimeIndex, index: np.ndarray
) -> DistanceModel:
    """One month's model from its training ``rows`` (``index``: their interval positions)."""
    centre = np.median(rows, axis=0)
    lw = LedoitWolf(assume_centered=True).fit(rows - centre)
    return DistanceModel(
        month=int(month),
        start=month_start_utc(month),
        centre=centre,
        precision=np.asarray(lw.precision_, dtype=np.float64),
        shrinkage=float(lw.shrinkage_),
        train_first_end=ends[index[0]],
        train_last_end=ends[index[-1]],
        n_train=len(index),
    )


def fit_distance_models(
    features: np.ndarray,
    ends: pd.DatetimeIndex,
    lay: Layout,
    cfg: RobustDistanceConfig,
) -> dict[int, DistanceModel]:
    """One model per NEM month present in the series, fitted on the ``train_weeks`` of
    complete rows starting before the month's start."""
    complete = np.all(np.isfinite(features), axis=1)
    starts = epoch_ns(ends - INTERVAL)
    window = pd.Timedelta(weeks=cfg.train_weeks)
    models: dict[int, DistanceModel] = {}
    for month in np.unique(lay.nem_month):
        start = month_start_utc(month)
        index = np.flatnonzero(
            complete
            & (starts >= (start - window).as_unit("ns").value)
            & (starts < start.as_unit("ns").value)
        )
        if len(index) < cfg.min_train_rows:
            continue
        models[int(month)] = fit_distance(features[index], int(month), ends, index)
    return models

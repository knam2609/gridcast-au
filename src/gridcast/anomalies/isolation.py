"""D6: one Isolation Forest per NEM month, fitted on the previous 26 weeks (§2.3).

The model for month ``m`` is fitted at ``m``'s start (NEM midnight on the 1st) on the
intervals that *start* in ``[start − train_weeks, start)`` with all nine features present,
so it has seen nothing from ``m`` on. A month is scored only if its window holds at least
``max_samples`` rows (every month after the warm-up has a full window).

The score is ``−score_samples`` (larger = more isolated). Its p-value is against the
training window's own scores: each tree sees 256 of about 52,000 rows, so a training row's
score is close to an out-of-sample score. A1 checks whether that holds in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gridcast.anomalies.pvalues import static_pvalues
from gridcast.anomalies.series import Layout, epoch_ns
from gridcast.config import IsolationForestConfig
from gridcast.timeutils import INTERVAL, NEM_TZ


@dataclass(frozen=True, slots=True)
class MonthModel:
    month: int  # 12 * year + month − 1 (NEM time)
    start: pd.Timestamp  # UTC instant of the month's start
    model: IsolationForest
    train_first_end: pd.Timestamp  # interval end of the first training row
    train_last_end: pd.Timestamp  # interval end of the last training row (≤ start)
    train_scores: np.ndarray  # sorted −score_samples of the training rows

    @property
    def n_train(self) -> int:
        return len(self.train_scores)

    def score(self, rows: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(rows)

    def pvalues(self, scores: np.ndarray) -> np.ndarray:
        return static_pvalues(self.train_scores, scores)


def month_start_utc(month: int) -> pd.Timestamp:
    year, m0 = divmod(int(month), 12)
    return pd.Timestamp(year=year, month=m0 + 1, day=1, tz=NEM_TZ).tz_convert("UTC")


def fit_month_models(
    features: np.ndarray,
    ends: pd.DatetimeIndex,
    lay: Layout,
    cfg: IsolationForestConfig,
    seed: int,
) -> dict[int, MonthModel]:
    """Fit one model per NEM month present in the series."""
    complete = np.all(np.isfinite(features), axis=1)
    starts = epoch_ns(ends - INTERVAL)
    window = pd.Timedelta(weeks=cfg.train_weeks)
    models: dict[int, MonthModel] = {}
    for month in np.unique(lay.nem_month):
        start = month_start_utc(month)
        rows = np.flatnonzero(
            complete
            & (starts >= (start - window).as_unit("ns").value)
            & (starts < start.as_unit("ns").value)
        )
        if len(rows) < cfg.max_samples:
            continue
        model = IsolationForest(
            n_estimators=cfg.n_estimators, max_samples=cfg.max_samples, random_state=seed
        ).fit(features[rows])
        models[int(month)] = MonthModel(
            month=int(month),
            start=start,
            model=model,
            train_first_end=ends[rows[0]],
            train_last_end=ends[rows[-1]],
            train_scores=np.sort(-model.score_samples(features[rows])),
        )
    return models


def score_months(features: np.ndarray, lay: Layout, models: dict[int, MonthModel]) -> np.ndarray:
    """Each interval's score under its month's model (NaN: no model or a missing feature)."""
    out = np.full(len(features), np.nan)
    complete = np.all(np.isfinite(features), axis=1)
    for month, mm in models.items():
        rows = np.flatnonzero(complete & (lay.nem_month == month))
        if rows.size:
            out[rows] = mm.score(features[rows])
    return out


def month_pvalues(
    scores: np.ndarray, lay: Layout, models: dict[int, MonthModel]
) -> tuple[np.ndarray, np.ndarray]:
    """p-values against each month's training scores, and the training-window sizes."""
    p = np.full(len(scores), np.nan)
    n_window = np.full(len(scores), -1, dtype=np.int64)
    for month, mm in models.items():
        rows = np.flatnonzero((lay.nem_month == month) & np.isfinite(scores))
        if rows.size:
            p[rows] = mm.pvalues(scores[rows])
            n_window[rows] = mm.n_train
    return p, n_window

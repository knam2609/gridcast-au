"""One detector pass over a series: scores, p-values, alert tiers and events (§2.1–§2.4).

Scores and p-values are computed for every interval with the data they need, warm-up
included, because the trailing windows need the warm-up's scores. Alerts are raised only
inside each detector's scored period (§2.1): the market detectors and the price lines from
``market_first_day``, D2 and D3 from ``residual_first_day``, both up to the split's end.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import (
    DETECTORS,
    ISOLATION_FOREST,
    LINE_NOTABLE,
    LINE_SEVERE,
    LINES,
    RESIDUAL,
    ROLLING,
    DetectorState,
    compute_scores,
)
from gridcast.anomalies.events import Event, merge_events
from gridcast.anomalies.isolation import month_pvalues
from gridcast.anomalies.pvalues import (
    NOTABLE,
    SEVERE,
    Queries,
    WindowPValues,
    tiers,
    window_pvalues,
)
from gridcast.anomalies.series import (
    INTERVALS_PER_DAY,
    MarketSeries,
    QuantileBand,
    nem_day_ordinals,
)
from gridcast.config import AnomaliesConfig

INTERVALS_PER_WEEK = 7 * INTERVALS_PER_DAY


def window_intervals(cfg: AnomaliesConfig) -> int:
    """The p-value window in intervals (26 weeks: 52,416)."""
    return cfg.pvalues.window_weeks * INTERVALS_PER_WEEK


def gap_intervals(cfg: AnomaliesConfig) -> int:
    return cfg.events.merge_gap_minutes // 5


@dataclass(frozen=True, slots=True)
class Split:
    """A scored period: NEM days ``[market_first_day, end_day)`` for the market detectors
    and lines, ``[residual_first_day, end_day)`` for D2 and D3 (``end_day`` None: open)."""

    name: str  # "dev", "holdout" (or e.g. "smoke")
    market_first_day: dt.date
    residual_first_day: dt.date
    end_day: dt.date | None

    def masks(self, ends: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        days = nem_day_ordinals(ends)
        end = self.end_day.toordinal() if self.end_day else np.iinfo(np.int64).max
        market = (days >= self.market_first_day.toordinal()) & (days < end)
        residual = (days >= self.residual_first_day.toordinal()) & (days < end)
        return {d: residual if d in RESIDUAL else market for d in (*DETECTORS, *LINES)}

    def index_range(self, ends: pd.DatetimeIndex) -> tuple[int, int]:
        """``[start, stop)`` of the intervals from the earlier first day to the end."""
        days = nem_day_ordinals(ends)
        first = min(self.market_first_day, self.residual_first_day).toordinal()
        end = self.end_day.toordinal() if self.end_day else np.iinfo(np.int64).max
        inside = np.flatnonzero((days >= first) & (days < end))
        if inside.size == 0:
            return 0, 0
        return int(inside[0]), int(inside[-1]) + 1


@dataclass(slots=True)
class DetectorRun:
    ends: pd.DatetimeIndex
    scores: dict[str, np.ndarray]  # calibrated detectors, NaN where not valid
    pvalues: dict[str, np.ndarray]  # NaN where the score is not valid (or no D6 model)
    n_window: dict[str, np.ndarray]  # window (D6: training) size; -1 where no p-value
    state: DetectorState
    window: WindowPValues  # the rolling detectors' sweep (columns in ROLLING order)


def run_detectors(
    series: MarketSeries,
    band: QuantileBand | None,
    cfg: AnomaliesConfig,
    extra: Queries | None = None,
) -> DetectorRun:
    """Scores and p-values of every calibrated detector over the whole series.

    ``extra`` queries (rows, ROLLING column, value) are answered against the same windows.
    """
    scores, state = compute_scores(series, band, cfg)
    matrix = np.column_stack([scores[d] for d in ROLLING])
    wp = window_pvalues(matrix, np.isfinite(matrix), window_intervals(cfg), extra)
    pvalues = {d: wp.p[:, k] for k, d in enumerate(ROLLING)}
    n_window = {d: wp.n_window[:, k] for k, d in enumerate(ROLLING)}
    p6, n6 = month_pvalues(scores[ISOLATION_FOREST], state.lay, state.models)
    pvalues[ISOLATION_FOREST] = p6
    n_window[ISOLATION_FOREST] = n6
    return DetectorRun(series.ends, scores, pvalues, n_window, state, wp)


def alert_tiers(
    run: DetectorRun, price: np.ndarray, split: Split, cfg: AnomaliesConfig
) -> dict[str, np.ndarray]:
    """Per detector (lines included): 0, 1 notable, 2 severe, inside the scored periods."""
    masks = split.masks(run.ends)
    out: dict[str, np.ndarray] = {}
    for d in DETECTORS:
        t = tiers(run.pvalues[d], cfg.pvalues.notable, cfg.pvalues.severe)
        t[~masks[d]] = 0
        out[d] = t
    price = np.asarray(price, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out[LINE_NOTABLE] = np.where(
            masks[LINE_NOTABLE] & (price >= cfg.price_lines.notable_aud), NOTABLE, 0
        ).astype(np.int8)
        out[LINE_SEVERE] = np.where(
            masks[LINE_SEVERE] & (price >= cfg.price_lines.severe_aud), SEVERE, 0
        ).astype(np.int8)
    return out


def split_events(
    run: DetectorRun,
    alert: dict[str, np.ndarray],
    price: np.ndarray,
    split: Split,
    cfg: AnomaliesConfig,
) -> list[Event]:
    start, stop = split.index_range(run.ends)
    return merge_events(alert, run.pvalues, run.scores, price, start, stop, gap_intervals(cfg))


@dataclass(slots=True)
class Detection:
    run: DetectorRun
    tiers: dict[str, np.ndarray]
    events: list[Event]


def detect(
    series: MarketSeries, band: QuantileBand | None, split: Split, cfg: AnomaliesConfig
) -> Detection:
    """Scores, p-values, alerts and events of one split."""
    run = run_detectors(series, band, cfg)
    alert = alert_tiers(run, series.price, split, cfg)
    return Detection(run, alert, split_events(run, alert, series.price, split, cfg))

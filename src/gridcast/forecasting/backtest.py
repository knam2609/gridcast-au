"""Walk-forward (rolling-origin) backtesting and evaluation.

* Origins are processed in monthly folds (NEM time). Before predicting a fold, every
  model is ``fit`` on data known at the fold's first origin (expanding window). Baselines
  ignore ``fit``; ML models will train here, so the same harness enforces that training
  data always precedes validation data.
* All models are scored on the **same** (origin, horizon) pairs: those where the actual
  and every model's forecast are finite.
* Splits are assigned by origin: ``dev`` (before ``holdout_start``) and ``holdout``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.config import EvaluationConfig
from gridcast.evaluation.metrics import METRICS_BY_TARGET, masked_metrics
from gridcast.evaluation.segments import target_segments
from gridcast.forecasting.origins import (
    ForecastSetup,
    SeriesContext,
    origin_positions,
    target_positions,
)
from gridcast.logs import get_logger
from gridcast.models.baselines import Forecaster
from gridcast.timeutils import NEM_TZ

log = get_logger(__name__)

SPLITS = ("dev", "holdout", "all")


@dataclass(slots=True)
class BacktestOutput:
    target: str
    setup: ForecastSetup
    origins: np.ndarray  # positions in the context grid
    origin_times: pd.DatetimeIndex
    targets: np.ndarray  # (n_origins, H) positions
    actual: np.ndarray  # (n_origins, H) float64
    predictions: dict[str, np.ndarray]  # model name -> (n_origins, H) float32
    folds: list[str]

    @property
    def model_names(self) -> list[str]:
        return list(self.predictions)

    def common_mask(self) -> np.ndarray:
        mask = np.isfinite(self.actual)
        for pred in self.predictions.values():
            mask &= np.isfinite(pred)
        return mask


def monthly_folds(origin_times: pd.DatetimeIndex) -> list[tuple[str, np.ndarray]]:
    """Group origin indices by NEM-time calendar month, in chronological order."""
    labels = origin_times.tz_convert(NEM_TZ).strftime("%Y-%m")
    folds: list[tuple[str, np.ndarray]] = []
    for label in pd.unique(labels):
        folds.append((str(label), np.flatnonzero(labels == label)))
    return folds


def run_backtest(
    ctx: SeriesContext,
    models: Sequence[Forecaster],
    setup: ForecastSetup,
    target: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> BacktestOutput:
    origins = origin_positions(ctx, setup, start, end)
    if origins.size == 0:
        raise ValueError("no forecast origins in the requested window")
    targets = target_positions(origins, setup)
    origin_times = pd.DatetimeIndex(ctx.times[origins])
    preds = {
        m.name: np.full((len(origins), setup.horizon_steps), np.nan, np.float32) for m in models
    }
    folds = monthly_folds(origin_times)
    for label, idx in folds:
        fold_origins = origins[idx]
        cutoff = int(fold_origins[0]) - setup.latency_steps
        for model in models:
            model.fit(ctx, cutoff, setup)
            preds[model.name][idx] = model.predict(ctx, fold_origins, setup)
        log.debug("backtest.fold", fold=label, origins=len(idx))
    log.info(
        "backtest.done",
        target=target,
        origins=len(origins),
        folds=len(folds),
        first=str(origin_times[0]),
        last=str(origin_times[-1]),
    )
    return BacktestOutput(
        target=target,
        setup=setup,
        origins=origins,
        origin_times=origin_times,
        targets=targets,
        actual=ctx.y[targets],
        predictions=preds,
        folds=[label for label, _ in folds],
    )


def evaluate(
    out: BacktestOutput,
    ctx: SeriesContext,
    eval_cfg: EvaluationConfig,
    holdout_start: pd.Timestamp,
    extra_segments: Mapping[str, np.ndarray] | None = None,
    splits: Sequence[str] = SPLITS,
) -> pd.DataFrame:
    """Long table of metrics: model, split, segment, horizon (0 = pooled), metric, value, n.

    ``extra_segments`` adds masks aligned with the context grid (e.g. the M2 weather
    segments); ``splits`` limits the splits computed (a development-only run has no holdout).
    """
    metrics = METRICS_BY_TARGET[out.target]
    common = out.common_mask()
    is_holdout = np.asarray(out.origin_times >= holdout_start)
    split_rows = {"dev": ~is_holdout, "holdout": is_holdout, "all": np.ones_like(is_holdout)}
    if unknown := set(splits) - set(split_rows):
        raise ValueError(f"unknown splits {sorted(unknown)}")
    segments = target_segments(ctx.frame, eval_cfg)
    if extra_segments:
        if clash := set(extra_segments) & set(segments):
            raise ValueError(f"segments defined twice: {sorted(clash)}")
        segments |= dict(extra_segments)
    reported = np.array([h for h in eval_cfg.horizons_reported if h <= out.setup.horizon_steps])

    records: list[tuple[str, str, str, int, str, float, int]] = []
    for name, pred in out.predictions.items():
        for split in splits:
            base = common & split_rows[split][:, None]
            for segment, seg_mask in segments.items():
                mask = base & seg_mask[out.targets]
                pooled, n = masked_metrics(pred, out.actual, mask, metrics)
                for metric, value in pooled.items():
                    records.append((name, split, segment, 0, metric, float(value), int(n)))
                horizons = out.setup.horizons if segment == "all" else reported
                cols = horizons - 1
                per_h, n_h = masked_metrics(
                    pred[:, cols], out.actual[:, cols], mask[:, cols], metrics, axis=0
                )
                for j, h in enumerate(horizons):
                    for metric, values in per_h.items():
                        records.append(
                            (name, split, segment, int(h), metric, float(values[j]), int(n_h[j]))
                        )
    return pd.DataFrame(
        records, columns=["model", "split", "segment", "horizon", "metric", "value", "n"]
    )

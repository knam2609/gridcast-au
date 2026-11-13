"""Stored point forecasts that the interval methods read (m3_plan.md §2.1).

M2 stored one wide parquet per model and batch (``{target}__{model}__{batch}.parquet``:
``origin_time`` plus ``h001…hHHH``), plus the actuals under the model name ``actual``
(:func:`gridcast.evaluation.store.save_predictions`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACTUAL = "actual"


@dataclass(slots=True)
class StoredBatch:
    batch_id: str
    origin_times: pd.DatetimeIndex
    actual: np.ndarray  # (n, H) float64, NaN where missing
    forecasts: dict[str, np.ndarray]  # model name -> (n, H) float64

    @property
    def n_leads(self) -> int:
        return int(self.actual.shape[1])

    def common_mask(self) -> np.ndarray:
        """Pairs where the actual and every model's forecast exist (M2's scored pairs)."""
        mask = np.isfinite(self.actual)
        for values in self.forecasts.values():
            mask &= np.isfinite(values)
        return mask

    def restrict(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> StoredBatch:
        """Issue times in ``[start, end)``."""
        keep = np.ones(len(self.origin_times), dtype=bool)
        if start is not None:
            keep &= self.origin_times >= start
        if end is not None:
            keep &= self.origin_times < end
        return StoredBatch(
            self.batch_id,
            self.origin_times[keep],
            self.actual[keep],
            {k: v[keep] for k, v in self.forecasts.items()},
        )


def read_wide(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """One wide parquet (``origin_time`` plus one column per lead): UTC issue times, values."""
    frame = pd.read_parquet(path)
    times = pd.DatetimeIndex(frame.pop("origin_time"))
    if times.tz is None or str(times.tz) != "UTC":
        raise ValueError(f"{path.name}: origin_time must be UTC")
    return times, frame.to_numpy(dtype=np.float64, na_value=np.nan)


def read_stored_batch(directory: Path, batch_id: str, target: str = "demand") -> StoredBatch:
    """Every model of one stored backtest batch, checked to share issue times and leads."""
    files = sorted(directory.glob(f"{target}__*__{batch_id}.parquet"))
    if not files:
        raise FileNotFoundError(f"no {target} predictions for batch {batch_id} in {directory}")
    times: pd.DatetimeIndex | None = None
    actual: np.ndarray | None = None
    forecasts: dict[str, np.ndarray] = {}
    for path in files:
        name = path.name.split("__")[1]
        t, values = read_wide(path)
        if times is None:
            times = t
        elif not times.equals(t):
            raise ValueError(f"{path.name}: issue times differ from the batch's other files")
        if name == ACTUAL:
            actual = values
        else:
            forecasts[name] = values
    if actual is None or times is None:
        raise FileNotFoundError(f"batch {batch_id} has no actuals file")
    if any(v.shape != actual.shape for v in forecasts.values()):
        raise ValueError("every model must have the same leads as the actuals")
    return StoredBatch(batch_id, times, actual, forecasts)


def read_stored_model(
    directory: Path, batch_id: str, model: str, target: str = "demand"
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """One model's stored forecasts (or ``actual``) of a batch: issue times and values."""
    path = directory / f"{target}__{model}__{batch_id}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no stored {target} forecasts of {model!r} for batch {batch_id}")
    return read_wide(path)

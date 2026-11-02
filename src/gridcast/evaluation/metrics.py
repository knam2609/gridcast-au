"""Point-forecast error metrics on masked (origin x horizon) error matrices.

Conventions: error = forecast − actual, so positive bias means over-forecasting.
MAPE is reported for demand only: prices are frequently zero or negative, which makes
percentage errors undefined or meaningless. Price gets the median absolute error as a
robust complement to MAE/RMSE, which are dominated by rare spikes.
"""

from __future__ import annotations

import numpy as np

METRICS_BY_TARGET: dict[str, tuple[str, ...]] = {
    "demand": ("mae", "rmse", "bias", "mape"),
    "price": ("mae", "rmse", "bias", "medae"),
}


def masked_metrics(
    forecast: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    metrics: tuple[str, ...],
    axis: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute ``metrics`` over entries where ``mask`` is True.

    ``axis=None`` pools everything; ``axis=0`` gives one value per column (horizon).
    Returns ``(values_by_metric, n)``; values are NaN where ``n == 0``.
    """
    err = np.where(mask, forecast.astype(np.float64) - actual, np.nan)
    n = mask.sum(axis=axis)
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        if "mae" in metrics or "medae" in metrics:
            abs_err = np.abs(err)
        if "mae" in metrics:
            out["mae"] = np.nansum(abs_err, axis=axis) / n
        if "rmse" in metrics:
            out["rmse"] = np.sqrt(np.nansum(err**2, axis=axis) / n)
        if "bias" in metrics:
            out["bias"] = np.nansum(err, axis=axis) / n
        if "mape" in metrics:
            pct = np.where(mask, np.abs(err) / np.abs(actual), np.nan)
            out["mape"] = 100.0 * np.nansum(pct, axis=axis) / n
        if "medae" in metrics:
            if axis is None:
                values = abs_err[mask]
                out["medae"] = np.asarray(np.median(values) if values.size else np.nan)
            else:
                out["medae"] = _nanmedian_columns(abs_err, n)
    return out, np.asarray(n)


def _nanmedian_columns(values: np.ndarray, n: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[1], np.nan)
    has = n > 0
    if has.any():
        out[has] = np.nanmedian(values[:, has], axis=0)
    return out

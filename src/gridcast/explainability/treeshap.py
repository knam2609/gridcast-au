"""TreeSHAP of a stored LightGBM C fit, and criterion E1's gate (m4_plan.md §2.6, §3).

LightGBM's built-in TreeSHAP (``predict(pred_contrib=True)``) returns, per row, one
contribution per feature plus the base value (the expected raw prediction). The raw
prediction is the model's correction to its target form's anchor, so

    forecast = anchor + base value + Σ contributions,

where the anchor is 0 for *direct*, y(o) for *latest*, y(t − 24 h) for *residual* and
y(o) + y(t − 24 h) − y(o − 24 h) for *anchored* (M2's ``MlModel.baseline``).

**E1 gate** (user decision 2026-11-19, §1 "Clarified in step 3"), for every explained pair:

1. the forecast recomputed through M2's code path, cast to float32 as M2 stored it, equals
   the stored forecast bit for bit;
2. base value + Σ contributions equals the raw prediction within ``sum_tolerance``
   (float64).

If either check fails for any pair, the explanations are withheld (:class:`GateError`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridcast.models.ml import LightGBMModel


class GateError(RuntimeError):
    """E1 failed: the explanations must be withheld."""


@dataclass(slots=True)
class Explanation:
    columns: list[str]
    forecast: np.ndarray  # (n,) float64, M2's code path
    anchor: np.ndarray  # (n,) float64, the target form's fixed term
    raw: np.ndarray  # (n,) float64, the booster's prediction
    base: np.ndarray  # (n,) float64, expected raw prediction (one value per model)
    contributions: np.ndarray  # (n, n_columns) float64
    target_form: str

    def __len__(self) -> int:
        return len(self.forecast)

    @property
    def sum_error(self) -> np.ndarray:
        """|base + Σ contributions − raw| per row."""
        return np.abs(self.base + self.contributions.sum(axis=1) - self.raw)


def explain(model: LightGBMModel, X: np.ndarray) -> Explanation:
    """Forecasts and TreeSHAP contributions of complete float32 feature rows ``X``."""
    if model.booster is None:
        raise RuntimeError("model is not fitted")
    if X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != len(model.columns):
        raise ValueError("X must be float32 rows of the model's columns (as M2 built them)")
    if not np.isfinite(X).all():
        raise ValueError("only complete rows have a forecast to explain")
    threads = model.cfg.num_threads
    forecast = np.asarray(model.predict(X), dtype=np.float64)
    raw = np.asarray(model.booster.predict(X, num_threads=threads), dtype=np.float64)
    contrib = np.asarray(
        model.booster.predict(X, pred_contrib=True, num_threads=threads), dtype=np.float64
    )
    if contrib.shape != (len(X), len(model.columns) + 1):
        raise AssertionError(f"unexpected pred_contrib shape {contrib.shape}")
    return Explanation(
        columns=list(model.columns),
        forecast=forecast,
        anchor=np.asarray(model.baseline(X), dtype=np.float64),
        raw=raw,
        base=contrib[:, -1].copy(),
        contributions=contrib[:, :-1].copy(),
        target_form=model.target_form,
    )


@dataclass(frozen=True, slots=True)
class GateResult:
    n: int
    forecast_mismatches: int  # float32(recomputed) != stored
    max_forecast_diff_mw: float  # |recomputed (float64) − stored|, for information
    sum_failures: int  # |base + Σ − raw| > tolerance
    max_sum_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.n > 0 and self.forecast_mismatches == 0 and self.sum_failures == 0

    def merge(self, other: GateResult) -> GateResult:
        if other.tolerance != self.tolerance:
            raise ValueError("gate results with different tolerances")
        return GateResult(
            self.n + other.n,
            self.forecast_mismatches + other.forecast_mismatches,
            max(self.max_forecast_diff_mw, other.max_forecast_diff_mw),
            self.sum_failures + other.sum_failures,
            max(self.max_sum_error, other.max_sum_error),
            self.tolerance,
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n": self.n,
            "forecast_mismatches": self.forecast_mismatches,
            "max_forecast_diff_mw": self.max_forecast_diff_mw,
            "sum_failures": self.sum_failures,
            "max_sum_error": self.max_sum_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def empty_gate(tolerance: float) -> GateResult:
    return GateResult(0, 0, 0.0, 0, 0.0, tolerance)


def gate(explanation: Explanation, stored: np.ndarray, tolerance: float) -> GateResult:
    """E1's two checks against the stored forecasts of the same pairs (any float dtype)."""
    stored = np.asarray(stored)
    if stored.shape != explanation.forecast.shape:
        raise ValueError("one stored forecast per explained pair")
    if not np.isfinite(stored).all():
        raise ValueError("an explained pair has no stored forecast")
    stored32 = stored.astype(np.float32)
    if not np.array_equal(stored32.astype(stored.dtype), stored):
        raise ValueError("stored forecasts must be float32 values (as M2 stored them)")
    mismatch = explanation.forecast.astype(np.float32) != stored32
    diff = np.abs(explanation.forecast - stored32.astype(np.float64))
    err = explanation.sum_error
    n = len(explanation)
    return GateResult(
        n=n,
        forecast_mismatches=int(mismatch.sum()),
        max_forecast_diff_mw=float(diff.max()) if n else 0.0,
        sum_failures=int((err > tolerance).sum()),
        max_sum_error=float(err.max()) if n else 0.0,
        tolerance=tolerance,
    )


def require(result: GateResult, what: str) -> None:
    """Raise :class:`GateError` unless the gate passed (explanations are then withheld)."""
    if not result.passed:
        raise GateError(
            f"E1 failed for {what}: {result.forecast_mismatches} of {result.n} forecasts differ "
            f"from the stored ones (max {result.max_forecast_diff_mw:.3g} MW), "
            f"{result.sum_failures} SHAP sums off by more than {result.tolerance:g} "
            f"(max {result.max_sum_error:.3g}); explanations withheld"
        )

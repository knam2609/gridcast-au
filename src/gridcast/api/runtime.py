"""Recompute and explain a stored LightGBM C forecast (m5_plan.md §2.5, step 3).

* **Loaded at startup** (step 3 question round, 2026-11-25): M2's feature builder (context and
  weather, from Postgres), the stored fits of both splits (run 2's and ``holdout_main``'s
  records, :func:`~gridcast.explainability.stored.stored_fits`) and every booster (from the
  data root, usually the SSD). Each booster must match its record (trees, feature names).
  What failed to load is retried by a later request, at most every :data:`RETRY_SECONDS`.
* **The batches must be the published ones:** ``serving.sources.demand_point`` names the
  same batch and model as ``anomalies.explain`` for each split.
* **Boosters are cached by model file.** The ``2025-08-01`` fits are in both batches (run 2's
  last retrain, ``holdout_main``'s first; byte-identical files), so a cache keyed by retrain
  and band, as M4's :class:`~gridcast.explainability.pairs.ModelCache`, would name one
  batch's file for the other's forecasts.
* **Recompute** goes through M2's code path (:meth:`LightGBMModel.predict` on the float32
  rows M2's builder gives) for every lead whose feature row is complete, and casts to float32
  as M2 stored it. M2 forecast only complete rows, so an incomplete row is ``None``, and
  ``matches_stored`` requires those to fall exactly where nothing was stored.
* **Explain** composes the frozen functions :func:`~gridcast.explainability.pairs.explain_pairs`
  is made of (``locate``, ``feature_rows``, ``explain``, ``gate``), so it also keeps the feature
  rows it explains. E1's gate covers every explained lead; on failure the explanations are
  withheld (:class:`~gridcast.explainability.treeshap.GateError`).
* **One lock** serialises loading and every model request: they are CPU-bound, and LightGBM
  already uses ``ml.lightgbm.num_threads`` per call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from gridcast.api.inputs import Unavailable
from gridcast.config import SERVING_SPLITS, AppConfig
from gridcast.explainability.families import Families, families
from gridcast.explainability.global_run import ANCHOR_TERMS
from gridcast.explainability.pairs import ModelCache
from gridcast.explainability.sample import feature_rows, origin_positions
from gridcast.explainability.stored import (
    DEV_SET,
    HOLDOUT_SET,
    StoredFit,
    batch_config,
    fits_from_records,
    load_model,
    locate,
    split_period,
)
from gridcast.explainability.treeshap import Explanation, GateResult, empty_gate, explain, gate
from gridcast.features.builder import FeatureBuilder
from gridcast.logs import get_logger
from gridcast.models.ml import LightGBMModel

log = get_logger(__name__)

RETRY_SECONDS = 10.0


class PathModelCache(ModelCache):
    """M4's model cache keyed by model file instead of retrain and band."""

    def get(self, fit: StoredFit) -> LightGBMModel:
        key = str(fit.path)
        if key not in self._models:
            self._models[key] = load_model(
                fit, self.columns, self.cfg.ml.lightgbm, self.cfg.ml.seed
            )
        return self._models[key]

    def __len__(self) -> int:
        return len(self._models)


def check_batches(cfg: AppConfig) -> None:
    """The fits explained are those of the published demand forecasts, for each split."""
    ex = cfg.anomalies.explain
    want = {"dev": ex.dev_batch, "holdout": ex.holdout_batch}
    problems = []
    for split in SERVING_SPLITS:
        src = cfg.serving.sources["demand_point"][split]
        if src.batch_id != want[split] or src.model_name != ex.model:
            problems.append(
                f"{split}: serving publishes {src.model_name} of batch {src.batch_id}, the "
                f"stored fits are {ex.model} of batch {want[split]}"
            )
    if problems:
        raise ValueError("; ".join(problems))


def relative_file(cfg: AppConfig, path: Path) -> str:
    """A model file as served: relative to the data root (no host paths in responses)."""
    root = cfg.resolve(cfg.paths.data_root)
    return str(path.relative_to(root)) if path.is_relative_to(root) else path.name


# --------------------------------------------------------------------------- loading


@dataclass(slots=True)
class Features:
    builder: FeatureBuilder
    fits: dict[str, list[StoredFit]]
    seconds: float


def split_fits(engine: Engine, cfg: AppConfig, split: str) -> list[StoredFit]:
    """:func:`~gridcast.explainability.stored.stored_fits` without its check that the data
    root exists, so the fits are known while the drive is unmounted (the boosters are not)."""
    ex = cfg.anomalies.explain
    batch, evaluation_set = (
        (ex.dev_batch, DEV_SET) if split == "dev" else (ex.holdout_batch, HOLDOUT_SET)
    )
    config = batch_config(engine, cfg, batch, evaluation_set)
    return fits_from_records(
        config["model_params"]["retrains"],
        cfg.resolve(cfg.paths.data_root),
        cfg.ml.lead_bands,
        split_period(cfg, split),
    )


def load_features(engine: Engine, cfg: AppConfig) -> Features:
    """M2's builder on everything stored (as ``explain-dev``/``-holdout`` used) and the fits."""
    from gridcast.features.inputs import load_context, load_feature_builder

    check_batches(cfg)
    t0 = time.perf_counter()
    fits = {s: split_fits(engine, cfg, s) for s in SERVING_SPLITS}
    region = cfg.serving.region
    builder = load_feature_builder(engine, cfg, region, ctx=load_context(engine, cfg, region))
    return Features(builder, fits, time.perf_counter() - t0)


def load_boosters(cfg: AppConfig, fits: dict[str, list[StoredFit]]) -> PathModelCache:
    """Every fit's booster, each checked against its record; refuses a missing data root."""
    root = cfg.resolve(cfg.paths.data_root)
    if not root.is_dir():
        raise FileNotFoundError("the data root is not found (is the drive mounted?)")
    cache = PathModelCache(cfg)
    for split_fits in fits.values():
        for fit in split_fits:
            if not fit.path.is_file():
                raise FileNotFoundError(f"model file not found: {relative_file(cfg, fit.path)}")
            cache.get(fit)
    return cache


@dataclass(slots=True)
class ModelState:
    """Everything recompute and explain read; immutable once loaded."""

    builder: FeatureBuilder
    fits: dict[str, list[StoredFit]]
    models: PathModelCache
    families: Families
    variant: str


def _reason(exc: BaseException) -> str:
    if isinstance(exc, OperationalError):
        return "database unavailable"
    return f"{type(exc).__name__}: {exc}"


class ModelRuntime:
    """Loads :class:`ModelState` in two parts (features, boosters) and retries what failed.

    ``features`` may be given (tests); otherwise it is loaded from ``engine()``.
    """

    def __init__(
        self,
        cfg: AppConfig,
        engine: Callable[[], Engine] | None,
        features: Features | None = None,
    ) -> None:
        self.cfg = cfg
        self._engine = engine
        self._features = features
        self._models: PathModelCache | None = None
        self._state: ModelState | None = None
        self.lock = threading.Lock()  # loading and every model request
        self.problems: dict[str, str] = {}  # "feature_context" / "model_files" -> reason
        self._failed_at: float | None = None
        self.seconds: dict[str, float] = {}

    @property
    def state(self) -> ModelState | None:
        return self._state

    def load(self) -> ModelState | None:
        """Load what is missing (unless a failure is under :data:`RETRY_SECONDS` old)."""
        if self._state is not None:
            return self._state
        with self.lock:
            if self._state is not None:
                return self._state
            recent = self._failed_at is not None and (
                time.monotonic() - self._failed_at < RETRY_SECONDS
            )
            if recent:
                return None
            self._load_locked()
            return self._state

    def _load_locked(self) -> None:
        if self._features is None:
            try:
                if self._engine is None:
                    raise RuntimeError("no database engine")
                self._features = load_features(self._engine(), self.cfg)
                self.seconds["features"] = self._features.seconds
                self.problems.pop("feature_context", None)
            except Exception as exc:  # reported by /health and as 503
                self.problems["feature_context"] = _reason(exc)
                self.problems["model_files"] = "not loaded: the stored fits are not known yet"
                self._failed_at = time.monotonic()
                log.warning("api.features_unavailable", error=self.problems["feature_context"])
                return
        try:
            t0 = time.perf_counter()
            self._models = load_boosters(self.cfg, self._features.fits)
            self.seconds["boosters"] = time.perf_counter() - t0
            self.problems.pop("model_files", None)
        except Exception as exc:  # reported by /health and as 503
            self.problems["model_files"] = _reason(exc)
            self._failed_at = time.monotonic()
            log.warning("api.boosters_unavailable", error=self.problems["model_files"])
            return
        variant = self.cfg.anomalies.explain.variant
        self._state = ModelState(
            self._features.builder, self._features.fits, self._models, families(variant), variant
        )
        self._failed_at = None
        log.info(
            "api.models_loaded",
            fits={s: len(f) for s, f in self._features.fits.items()},
            boosters=len(self._models),
            seconds={k: round(v, 2) for k, v in self.seconds.items()},
        )

    def require(self) -> ModelState:
        """The loaded state, or :class:`Unavailable` (HTTP 503) with the reason."""
        state = self.load()
        if state is None:
            reasons = "; ".join(f"{k}: {v}" for k, v in sorted(self.problems.items()))
            raise Unavailable(f"recompute and explain are unavailable ({reasons or 'loading'})")
        return state

    def checks(self) -> dict[str, tuple[bool, str]]:
        """``/health``'s two model checks (it retries a load first, see :meth:`load`)."""
        self.load()
        out: dict[str, tuple[bool, str]] = {}
        f = self._features
        if f is not None:
            t = f.builder.ctx.times
            out["feature_context"] = (
                True,
                f"{self.cfg.serving.region}, interval ends {t[0].isoformat()} → "
                f"{t[-1].isoformat()}",
            )
        else:
            out["feature_context"] = (False, self.problems.get("feature_context", "not loaded"))
        if self._state is not None:
            n = {s: len(v) for s, v in self._state.fits.items()}
            out["model_files"] = (
                True,
                f"{len(self._state.models)} boosters for {sum(n.values())} fits "
                f"(dev {n['dev']}, holdout {n['holdout']}), each matching its record",
            )
        else:
            out["model_files"] = (False, self.problems.get("model_files", "not loaded"))
        return out


# --------------------------------------------------------------------------- one issue time


@dataclass(slots=True)
class Rows:
    """The feature rows of every lead of one issue time, and the fit of each lead."""

    issue_time: pd.Timestamp
    leads: np.ndarray  # 1…H
    X: np.ndarray  # (H, p) float32, M2's builder
    complete: np.ndarray  # (H,) bool: every feature present
    where: np.ndarray  # (H,) position in ``fits``
    fits: list[StoredFit]


def issue_rows(state: ModelState, split: str, issue_time: pd.Timestamp, horizon: int) -> Rows:
    leads = np.arange(1, horizon + 1, dtype=np.int64)
    times = pd.DatetimeIndex([issue_time]).repeat(horizon)
    fits = state.fits[split]
    where = locate(fits, times, leads)
    if np.any(where < 0):
        raise LookupError(
            f"{int((where < 0).sum())} leads of {issue_time.isoformat()} have no stored {split} fit"
        )
    X = feature_rows(state.builder, state.variant, origin_positions(state.builder, times), leads)
    return Rows(issue_time, leads, X, np.isfinite(X).all(axis=1), where, fits)


@dataclass(slots=True)
class Recomputed:
    rows: Rows
    forecast: np.ndarray  # (H,) float64, M2's code path (NaN: incomplete row)
    recomputed: np.ndarray  # (H,) float32, as M2 stored it
    stored: np.ndarray  # (H,) float32 (NaN: nothing stored)

    @property
    def differs(self) -> np.ndarray:
        a, b = self.recomputed, self.stored
        return (np.isnan(a) != np.isnan(b)) | (np.isfinite(a) & np.isfinite(b) & (a != b))

    @property
    def matches(self) -> bool:
        return not bool(self.differs.any())

    def _both(self) -> np.ndarray:
        return np.isfinite(self.recomputed) & np.isfinite(self.stored)

    @property
    def max_abs_diff(self) -> float:
        both = self._both()
        diff = np.abs(self.recomputed[both].astype(np.float64) - self.stored[both])
        return float(diff.max()) if both.any() else 0.0

    @property
    def max_float64_diff(self) -> float:
        both = self._both()
        diff = np.abs(self.forecast[both] - self.stored[both].astype(np.float64))
        return float(diff.max()) if both.any() else 0.0


def stored_array(stored: dict[int, float], horizon: int) -> np.ndarray:
    """Stored forecasts by lead as float32 (NaN where none); each must be a float32 value."""
    out = np.full(horizon, np.nan, dtype=np.float32)
    for h, v in stored.items():
        if not 1 <= h <= horizon:
            raise ValueError(f"stored lead {h} is outside 1…{horizon}")
        out[h - 1] = v
        if float(out[h - 1]) != v:
            raise ValueError(f"stored lead {h}: {v!r} is not a float32 value (as M2 stored it)")
    return out


def recompute(
    state: ModelState, split: str, issue_time: pd.Timestamp, stored: dict[int, float], horizon: int
) -> Recomputed:
    rows = issue_rows(state, split, issue_time, horizon)
    forecast = np.full(horizon, np.nan, dtype=np.float64)
    for k in np.unique(rows.where[rows.complete]):
        idx = np.flatnonzero(rows.complete & (rows.where == k))
        model = state.models.get(rows.fits[int(k)])
        forecast[idx] = np.asarray(model.predict(rows.X[idx]), dtype=np.float64)
    return Recomputed(rows, forecast, forecast.astype(np.float32), stored_array(stored, horizon))


@dataclass(slots=True)
class Explained:
    rows: Rows
    idx: np.ndarray  # positions (lead − 1) explained
    explanation: Explanation  # row i explains lead idx[i] + 1
    gate: GateResult
    incomplete_stored: int = 0  # stored leads whose feature row is incomplete (a mismatch)


def explain_leads(
    state: ModelState,
    split: str,
    issue_time: pd.Timestamp,
    stored: dict[int, float],
    leads: Sequence[int],
    horizon: int,
    tolerance: float,
) -> Explained:
    """TreeSHAP of the stored ``leads`` (each must have a stored forecast) and E1's gate.

    A stored lead whose feature row is incomplete cannot be recomputed: it counts as a
    forecast mismatch, so the gate fails and the explanations are withheld.
    """
    rows = issue_rows(state, split, issue_time, horizon)
    values = stored_array(stored, horizon)
    idx = np.asarray(sorted({int(h) - 1 for h in leads}), dtype=np.int64)
    if len(idx) == 0 or np.isnan(values[idx]).any():
        raise ValueError("every explained lead needs a stored forecast")
    incomplete = int((~rows.complete[idx]).sum())
    use = idx[rows.complete[idx]]
    total = empty_gate(tolerance)
    parts: list[tuple[np.ndarray, Explanation]] = []
    for k in np.unique(rows.where[use]):
        sel = use[rows.where[use] == k]
        fit = rows.fits[int(k)]
        ex = explain(state.models.get(fit), rows.X[sel])
        total = total.merge(gate(ex, values[sel], tolerance))
        parts.append((sel, ex))
    if incomplete:
        total = GateResult(
            total.n + incomplete,
            total.forecast_mismatches + incomplete,
            total.max_forecast_diff_mw,
            total.sum_failures,
            total.max_sum_error,
            tolerance,
        )
    return Explained(rows, idx, _stack(parts, idx, rows), total, incomplete)


def _stack(parts: list[tuple[np.ndarray, Explanation]], idx: np.ndarray, rows: Rows) -> Explanation:
    """One explanation whose row ``i`` is lead ``idx[i] + 1`` (NaN where not explained)."""
    n = len(idx)
    p = rows.X.shape[1]
    pos = {int(j): i for i, j in enumerate(idx)}
    forecast, anchor, raw, base = (np.full(n, np.nan) for _ in range(4))
    contributions = np.full((n, p), np.nan)
    columns: list[str] = []
    forms = set()
    for sel, ex in parts:
        at = np.array([pos[int(j)] for j in sel], dtype=np.int64)
        forecast[at], anchor[at], raw[at], base[at] = ex.forecast, ex.anchor, ex.raw, ex.base
        contributions[at] = ex.contributions
        columns = ex.columns
        forms.add(ex.target_form)
    form = forms.pop() if len(forms) == 1 else "mixed"
    return Explanation(columns, forecast, anchor, raw, base, contributions, form)


def anchor_terms(target_form: str) -> list[str]:
    """The features M2's anchor of ``target_form`` adds up (empty for *direct*)."""
    return list(ANCHOR_TERMS[target_form])


class ModelFault(RuntimeError):
    """A published issue time the stored fits cannot recompute (a server fault): HTTP 500."""


class ExplanationWithheld(RuntimeError):
    """E1's gate failed for the requested leads: HTTP 500 with the gate's result."""

    def __init__(self, message: str, result: GateResult) -> None:
        super().__init__(message)
        self.result = result

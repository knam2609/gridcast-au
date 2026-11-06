"""Machine-learning regressors for the M2 walk-forward: LightGBM and ridge.

Both models learn one of two *target forms* of demand (m2_plan.md §2.4):

* ``residual``: r = y(t) − y(t − 24 h), the correction to "same time yesterday"; the
  forecast is ``dem_same_time_yesterday + r̂``;
* ``direct``: y(t) itself (the ablation);
* ``anchored``: the correction to y(o) + [y(t − 24 h) − y(o − 24 h)], i.e. persistence plus
  yesterday's change over the same lead (``dem_same_time_yesterday + dem_now_vs_yesterday``).
  Added in step 3 for short leads, where the other forms cannot reproduce y(o) precisely.

The forecast error is the same in both forms (ŷ − y = r̂ − r), so MAE compares them
directly, and the target form is selected like any other setting.

Each model has two steps, used by the walk-forward's per-retrain selection:

* :meth:`MlModel.score` fits on the early-stopping rows ``es_train`` and returns the MAE on
  ``es_valid`` (the last weeks of the training window, issued after every ``es_train``
  target). LightGBM early-stops there and remembers the best tree count.
* :meth:`MlModel.refit` fits the final model on **every** training row (LightGBM with the
  early-stopped tree count). Ridge always fits a seeded subsample of its rows.
"""

from __future__ import annotations

import gzip
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gridcast.config import LightGBMConfig, MlConfig, RidgeConfig

BASE_FEATURE = "dem_same_time_yesterday"
TODAY_VS_YESTERDAY = "dem_now_vs_yesterday"
TARGET_FORMS = ("residual", "direct", "anchored")
FAMILIES = ("lightgbm", "ridge")
FAMILY_PREFIX = {"lightgbm": "lgbm", "ridge": "ridge"}
PREDICT_CHUNK = 250_000  # rows per ridge prediction block (bounds the dense design matrix)


@dataclass(slots=True)
class Score:
    """Early-stopping result of one setting: MAE (MW) on ``es_valid``."""

    mae: float
    best_iteration: int | None  # LightGBM only
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - actual)))


class MlModel(ABC):
    """A regressor on float32 feature arrays whose columns are ``columns``, in order."""

    family: str

    def __init__(self, columns: Sequence[str], target_form: str, seed: int) -> None:
        if target_form not in TARGET_FORMS:
            raise ValueError(f"unknown target form {target_form!r}")
        if BASE_FEATURE not in columns or TODAY_VS_YESTERDAY not in columns:
            raise ValueError(f"features must include {BASE_FEATURE} and {TODAY_VS_YESTERDAY}")
        self.columns = list(columns)
        self.target_form = target_form
        self.seed = seed
        self._base = self.columns.index(BASE_FEATURE)
        self._shift = self.columns.index(TODAY_VS_YESTERDAY)

    # --------------------------------------------------------------- target form

    def baseline(self, X: np.ndarray) -> np.ndarray:
        """The value the target form is a correction to (zero for ``direct``)."""
        if self.target_form == "direct":
            return np.zeros(len(X))
        base = X[:, self._base].astype(np.float64)
        if self.target_form == "anchored":
            base = base + X[:, self._shift].astype(np.float64)
        return base

    def target(self, X: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return actual.astype(np.float64) - self.baseline(X)

    def to_forecast(self, X: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return self.baseline(X) + pred

    # --------------------------------------------------------------- interface

    def _check(self, X: np.ndarray, actual: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != len(self.columns) or len(X) != len(actual):
            raise ValueError("feature array does not match the model's columns or the target")
        if not (np.isfinite(X).all() and np.isfinite(actual).all()):
            raise ValueError("training rows must be complete (no missing feature or target)")

    def score(
        self, X: np.ndarray, actual: np.ndarray, es_train: np.ndarray, es_valid: np.ndarray
    ) -> Score:
        """Fit on ``es_train`` rows and measure MAE (MW) on ``es_valid`` rows."""
        self._check(X, actual)
        if not es_train.any() or not es_valid.any():
            raise ValueError("early stopping needs both training and validation rows")
        t0 = time.perf_counter()
        mae, best = self._score(X, self.target(X, actual), actual, es_train, es_valid)
        return Score(mae, best, round(time.perf_counter() - t0, 2))

    def refit(self, X: np.ndarray, actual: np.ndarray, n_trees: int | None = None) -> float:
        """Fit the final model on every row; returns the seconds taken."""
        self._check(X, actual)
        t0 = time.perf_counter()
        self._refit(X, self.target(X, actual), n_trees)
        return round(time.perf_counter() - t0, 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Demand forecast (MW) for complete rows of ``X``."""
        if X.ndim != 2 or X.shape[1] != len(self.columns):
            raise ValueError("feature array does not match the model's columns")
        return self.to_forecast(X, self._predict(X))

    @abstractmethod
    def _score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        actual: np.ndarray,
        es_train: np.ndarray,
        es_valid: np.ndarray,
    ) -> tuple[float, int | None]: ...

    @abstractmethod
    def _refit(self, X: np.ndarray, y: np.ndarray, n_trees: int | None) -> None: ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, stem: Path) -> Path:
        """Write the fitted model next to ``stem`` (suffix added) and return the path."""

    @abstractmethod
    def params(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- LightGBM


class LightGBMModel(MlModel):
    family = "lightgbm"

    def __init__(
        self,
        columns: Sequence[str],
        target_form: str,
        objective: str,
        num_leaves: int,
        cfg: LightGBMConfig,
        seed: int,
    ) -> None:
        super().__init__(columns, target_form, seed)
        self.objective = objective
        self.num_leaves = num_leaves
        self.cfg = cfg
        self.booster: Any = None
        self.n_trees: int | None = None

    def _lgb_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "metric": "l1",  # early stopping always on MAE, the evaluation metric
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.cfg.min_data_in_leaf,
            "feature_fraction": self.cfg.feature_fraction,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.cfg.num_threads,
            "verbosity": -1,
        }

    def _score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        actual: np.ndarray,
        es_train: np.ndarray,
        es_valid: np.ndarray,
    ) -> tuple[float, int | None]:
        import lightgbm as lgb

        dtrain = lgb.Dataset(X[es_train], y[es_train], feature_name=self.columns)
        dvalid = lgb.Dataset(X[es_valid], y[es_valid], reference=dtrain)
        stopped = lgb.train(
            self._lgb_params(),
            dtrain,
            num_boost_round=self.cfg.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )
        self.n_trees = max(int(stopped.best_iteration), 1)
        # LightGBM's l1 on the target equals the forecast MAE (both forms share errors).
        return float(stopped.best_score["valid_0"]["l1"]), self.n_trees

    def _refit(self, X: np.ndarray, y: np.ndarray, n_trees: int | None) -> None:
        import lightgbm as lgb

        trees = n_trees if n_trees is not None else self.n_trees
        if trees is None:
            raise ValueError("refit needs a tree count (score first, or pass n_trees)")
        self.n_trees = trees
        full = lgb.Dataset(X, y, feature_name=self.columns)
        self.booster = lgb.train(self._lgb_params(), full, num_boost_round=trees)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return np.asarray(self.booster.predict(X, num_threads=self.cfg.num_threads))

    def save(self, stem: Path) -> Path:
        path = stem.with_suffix(".txt.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(self.booster.model_to_string())
        return path

    def params(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "target_form": self.target_form,
            "objective": self.objective,
            "num_leaves": self.num_leaves,
            "n_trees": self.n_trees,
            "learning_rate": self.cfg.learning_rate,
            "min_data_in_leaf": self.cfg.min_data_in_leaf,
            "feature_fraction": self.cfg.feature_fraction,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------- ridge

# Ridge design (m2_plan.md §2.4): standardised numeric features, one-hot half-hour × day
# type, temperature spline terms, plus one-hot lead-time hour and its interactions with
# the two "how today differs" features (a linear model cannot otherwise learn that their
# effect fades with lead time).
BINARY = (
    "cal_non_working",
    "cal_public_holiday",
    "cal_dst_change_day",
    "cal_yesterday_non_working",
    "cal_yesterday_public_holiday",
)
CATEGORICAL = ("cal_slot", "cal_day_of_week", "cal_month")
LEAD_INTERACTIONS = ("dem_now_vs_yesterday", "dem_change_1h")
SPLINE_FEATURE = "wx_temp"
LEAD_BANDS = 24  # hourly lead-time bands over 24 h


def _one_hot(index: np.ndarray, n: int, name: str) -> np.ndarray:
    idx = index.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"{name} outside 0..{n - 1}")
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out


class RidgeDesign:
    """Builds the dense ridge design matrix from a variant's feature array."""

    def __init__(self, columns: Sequence[str], spline_knots: int) -> None:
        self.columns = list(columns)
        col = {name: i for i, name in enumerate(self.columns)}
        missing = {"lead_minutes", *CATEGORICAL, *LEAD_INTERACTIONS} - set(col)
        if missing:
            raise ValueError(f"ridge design needs {sorted(missing)}")
        self._col = col
        self.numeric = [c for c in self.columns if c not in CATEGORICAL and c not in BINARY]
        self.binary = [c for c in self.columns if c in BINARY]
        self.spline_knots = spline_knots
        self.spline: Any = None

    def fit(self, X: np.ndarray) -> RidgeDesign:
        if SPLINE_FEATURE in self._col:
            from sklearn.preprocessing import SplineTransformer

            self.spline = SplineTransformer(
                n_knots=self.spline_knots, degree=3, extrapolation="constant", include_bias=False
            ).fit(X[:, [self._col[SPLINE_FEATURE]]].astype(np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        c = self._col

        def get(name: str) -> np.ndarray:
            return X[:, c[name]].astype(np.float64)

        lead_band = (get("lead_minutes").astype(np.int64) - 5) // 60
        lead = _one_hot(lead_band, LEAD_BANDS, "lead band")
        half_hour = get("cal_slot").astype(np.int64) // 6
        non_working = get("cal_non_working").astype(np.int64)
        blocks = [
            np.column_stack([get(n) for n in self.numeric]),
            np.column_stack([get(n) for n in self.binary]),
            _one_hot(half_hour * 2 + non_working, 96, "half hour × day type"),
            _one_hot(get("cal_day_of_week"), 7, "day of week"),
            _one_hot(get("cal_month") - 1, 12, "month"),
            lead,
            *(lead * get(name)[:, None] for name in LEAD_INTERACTIONS),
        ]
        if self.spline is not None:
            blocks.append(self.spline.transform(X[:, [c[SPLINE_FEATURE]]].astype(np.float64)))
        return np.hstack(blocks)

    @property
    def n_columns(self) -> int:
        spline = 0 if self.spline is None else int(self.spline.n_features_out_)
        return len(self.numeric) + len(self.binary) + 96 + 7 + 12 + LEAD_BANDS * 3 + spline


class RidgeModel(MlModel):
    family = "ridge"

    def __init__(
        self,
        columns: Sequence[str],
        target_form: str,
        alpha: float,
        cfg: RidgeConfig,
        seed: int,
    ) -> None:
        super().__init__(columns, target_form, seed)
        self.alpha = alpha
        self.cfg = cfg
        self.design: RidgeDesign | None = None
        self.scaler: Any = None
        self.ridge: Any = None
        self.rows_fitted = 0

    def _fit_rows(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(self.seed)
        n = len(X)
        take = np.sort(rng.choice(n, size=min(n, self.cfg.max_rows), replace=False))
        self.rows_fitted = len(take)
        self.design = RidgeDesign(self.columns, self.cfg.spline_knots).fit(X[take])
        D = self.design.transform(X[take])
        self.scaler = StandardScaler().fit(D)
        self.ridge = Ridge(alpha=self.alpha).fit(self.scaler.transform(D), y[take])

    def _score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        actual: np.ndarray,
        es_train: np.ndarray,
        es_valid: np.ndarray,
    ) -> tuple[float, int | None]:
        self._fit_rows(X[es_train], y[es_train])
        Xv = X[es_valid]
        return _mae(self.to_forecast(Xv, self._predict(Xv)), actual[es_valid]), None

    def _refit(self, X: np.ndarray, y: np.ndarray, n_trees: int | None) -> None:
        self._fit_rows(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.design is None:
            raise RuntimeError("model is not fitted")
        out = np.empty(len(X))
        for lo in range(0, len(X), PREDICT_CHUNK):
            block = self.design.transform(X[lo : lo + PREDICT_CHUNK])
            out[lo : lo + PREDICT_CHUNK] = self.ridge.predict(self.scaler.transform(block))
        return out

    def save(self, stem: Path) -> Path:
        import joblib

        path = stem.with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"design": self.design, "scaler": self.scaler, "ridge": self.ridge}, path, 3)
        return path

    def params(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "target_form": self.target_form,
            "alpha": self.alpha,
            "rows_fitted": self.rows_fitted,
            "max_rows": self.cfg.max_rows,
            "spline_knots": self.cfg.spline_knots,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------- factory


def grid_choices(family: str, cfg: MlConfig) -> list[dict[str, Any]]:
    """Settings searched at every retrain, in a fixed order (ties go to the earliest)."""
    g = cfg.grid
    if family == "lightgbm":
        return [
            {"target_form": form, "objective": obj, "num_leaves": leaves}
            for form in g.target_forms
            for obj in g.lightgbm_objective
            for leaves in g.lightgbm_num_leaves
        ]
    if family == "ridge":
        return [
            {"target_form": form, "alpha": alpha}
            for form in g.target_forms
            for alpha in g.ridge_alpha
        ]
    raise ValueError(f"unknown model family {family!r}")


def make_model(
    family: str, columns: Sequence[str], choice: Mapping[str, Any], cfg: MlConfig
) -> MlModel:
    """Model of ``family`` with ``choice`` (target form plus family settings)."""
    form = str(choice["target_form"])
    if family == "lightgbm":
        return LightGBMModel(
            columns,
            form,
            str(choice["objective"]),
            int(choice["num_leaves"]),
            cfg.lightgbm,
            cfg.seed,
        )
    if family == "ridge":
        return RidgeModel(columns, form, float(choice["alpha"]), cfg.ridge, cfg.seed)
    raise ValueError(f"unknown model family {family!r}")

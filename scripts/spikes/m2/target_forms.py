"""M2 step 3: target forms by lead time, on pre-scoring data only.

usage: uv run python scripts/spikes/m2/target_forms.py

At the first walk-forward retrain (2022-11-01), fits LightGBM (L2, 31 leaves, the configured
fixed settings, early-stopped) for variant C with three target forms and prints the MAE by
lead band on the early-stopping slice (issued 6 Sep → 31 Oct 2022, before the first scored
issue time), next to persistence and daily naive on the same pairs:

* residual: y(t) − y(t − 24 h)                           (the planned target)
* direct:   y(t)                                          (the planned ablation)
* anchored: y(t) − [y(o) + y(t − 24 h) − y(o − 24 h)]     (persistence plus yesterday's shape)

Output feeds m2_plan.md §2.4 (target forms).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from gridcast.config import load_config
from gridcast.database.engine import get_engine
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.models.baselines import DAY_STEPS

BANDS = ((1, 1), (2, 6), (7, 12), (13, 24), (25, 72), (73, 288), (1, 288))


def label(lo: int, hi: int) -> str:
    def m(h: int) -> str:
        return f"{5 * h} min" if h < 12 else f"{5 * h / 60:g} h"

    return m(lo) if lo == hi else f"{m(lo)}–{m(hi)}"


def main() -> None:
    cfg = load_config()
    builder = load_feature_builder(get_engine(), cfg, "VIC1")
    cutoff = int(builder.ctx.times.searchsorted(nem_midnight_utc(str(cfg.ml.first_issue))))
    es_steps = cfg.ml.early_stopping_weeks * 7 * DAY_STEPS
    sample = wf.training_sample(builder, "C", wf.train_origins(builder, cutoff), 3)
    rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, es_steps)
    X, actual = sample.X[rows], sample.actual[rows].astype(np.float64)
    col = {n: i for i, n in enumerate(sample.columns)}

    def f(name: str) -> np.ndarray:
        return X[:, col[name]].astype(np.float64)

    bases = {
        "residual": f("dem_same_time_yesterday"),
        "direct": np.zeros(len(X)),
        "anchored": f("dem_same_time_yesterday") + f("dem_now_vs_yesterday"),
    }
    h = sample.horizons[rows][es_valid]
    forecasts = {
        "persistence": f("dem_now")[es_valid],
        "daily naive": f("dem_same_time_yesterday")[es_valid],
    }
    params = {
        "objective": "l2",
        "metric": "l1",
        "learning_rate": cfg.ml.lightgbm.learning_rate,
        "num_leaves": 31,
        "min_data_in_leaf": cfg.ml.lightgbm.min_data_in_leaf,
        "feature_fraction": cfg.ml.lightgbm.feature_fraction,
        "seed": cfg.ml.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    for form, base in bases.items():
        y = actual - base
        dtrain = lgb.Dataset(X[es_train], y[es_train], feature_name=sample.columns)
        dvalid = lgb.Dataset(X[es_valid], y[es_valid], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            cfg.ml.lightgbm.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(cfg.ml.lightgbm.early_stopping_rounds, verbose=False)],
        )
        pred = base[es_valid] + booster.predict(X[es_valid], num_iteration=booster.best_iteration)
        forecasts[f"lgbm {form} ({booster.best_iteration} trees)"] = pred

    truth = actual[es_valid]
    print(f"early-stopping slice of retrain {cfg.ml.first_issue}: {int(es_valid.sum()):,} pairs")
    print("MAE (MW)".ljust(32) + "".join(label(lo, hi).rjust(12) for lo, hi in BANDS))
    for name, pred in forecasts.items():
        cells = []
        for lo, hi in BANDS:
            sel = (h >= lo) & (h <= hi)
            cells.append(f"{np.mean(np.abs(pred[sel] - truth[sel])):12.1f}")
        print(name.ljust(32) + "".join(cells))


if __name__ == "__main__":
    main()

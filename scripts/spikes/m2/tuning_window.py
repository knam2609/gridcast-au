"""M2 step 3: why tuning on the pre-scoring window (retrain 2022-07-01) was abandoned.

usage: uv run python scripts/spikes/m2/tuning_window.py

For two retrains of variant C (the agreed tuning retrain 2022-07-01, and the first
walk-forward retrain 2022-11-01; both early-stopping slices precede the first scored issue
time), builds the walk-forward training rows, fits LightGBM (residual target, L2 objective,
the configured fixed settings, one model for all leads as first planned) on the
early-stopping training rows, and prints the MAE on the early-stopping slice (the last
8 weeks before the retrain) after various tree counts, next to "same time yesterday" (daily
naive) on the same rows. Only data before each retrain is read. Output feeds m2_plan.md §2.4.
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

RETRAINS = ("2022-07-01", "2022-11-01")  # both early-stopping slices precede scoring
LEAVES = (31, 127)
CHECKPOINTS = (10, 50, 100, 200, 400, 600)


def main() -> None:
    cfg = load_config()
    builder = load_feature_builder(get_engine(), cfg, "VIC1")
    es_steps = cfg.ml.early_stopping_weeks * 7 * DAY_STEPS
    for day in RETRAINS:
        cutoff = int(builder.ctx.times.searchsorted(nem_midnight_utc(day)))
        origins = wf.train_origins(builder, cutoff)
        sample = wf.training_sample(builder, "C", origins, cfg.ml.sample_every_steps)
        rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, es_steps)
        X, actual = sample.X[rows], sample.actual[rows]
        r = actual - X[:, sample.columns.index("dem_same_time_yesterday")]
        first = builder.ctx.times[sample.origins[rows][es_train].min()]
        split = builder.ctx.times[sample.origins[rows][es_valid].min()]
        print(
            f"== retrain {day}: fit on issue times {first:%Y-%m-%d} → {split:%Y-%m-%d}, "
            f"monitor {split:%Y-%m-%d} → {day}; rows {len(rows):,}; "
            f"daily naive MAE {np.abs(r[es_valid]).mean():.1f} MW"
        )
        for leaves in LEAVES:
            params = {
                "objective": "l2",
                "metric": "l1",
                "learning_rate": cfg.ml.lightgbm.learning_rate,
                "num_leaves": leaves,
                "min_data_in_leaf": cfg.ml.lightgbm.min_data_in_leaf,
                "feature_fraction": cfg.ml.lightgbm.feature_fraction,
                "seed": cfg.ml.seed,
                "deterministic": True,
                "force_row_wise": True,
                "verbosity": -1,
            }
            dtrain = lgb.Dataset(X[es_train], r[es_train], feature_name=sample.columns)
            dvalid = lgb.Dataset(X[es_valid], r[es_valid], reference=dtrain)
            history: dict[str, dict[str, list[float]]] = {}
            lgb.train(
                params,
                dtrain,
                max(CHECKPOINTS),
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.record_evaluation(history)],
            )
            curve = np.array(history["valid"]["l1"])
            points = "  ".join(f"{k}: {curve[k - 1]:.0f}" for k in CHECKPOINTS)
            print(
                f"   {leaves:3d} leaves: best {curve.min():.1f} MW after {curve.argmin() + 1} trees"
                f" | MAE after n trees  {points}"
            )


if __name__ == "__main__":
    main()

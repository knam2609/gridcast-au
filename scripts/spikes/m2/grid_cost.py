"""M2 step 3: cost and accuracy of grid settings, on pre-scoring data only.

usage: uv run python scripts/spikes/m2/grid_cost.py

At the first walk-forward retrain (2022-11-01), variant C, scores four LightGBM settings per lead
band at learning rates 0.05 and 0.1 on the early-stopping slice (issued 6 Sep → 31 Oct 2022,
before the first scored issue time): MAE, early-stopped tree count and seconds. Evidence for
dropping 255 leaves from the grid (slowest, least accurate) and keeping learning rate 0.05.
"""

from __future__ import annotations

import time

from gridcast.config import load_config
from gridcast.database.engine import get_engine
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.models.ml import LightGBMModel

cfg = load_config()
b = load_feature_builder(get_engine(), cfg, "VIC1")
cut = int(b.ctx.times.searchsorted(nem_midnight_utc(str(cfg.ml.first_issue))))
s = wf.training_sample(b, "C", wf.train_origins(b, cut), 3)
rows, tr, va = wf.fit_rows(s, s.usable, cut, 8 * 7 * 288)
h = s.horizons[rows]
for lo, hi in cfg.ml.lead_bands:
    m = (h >= lo) & (h <= hi)
    X, a = s.X[rows[m]], s.actual[rows[m]]
    for form, obj, leaves in [
        ("direct", "l1", 31),
        ("direct", "l2", 31),
        ("anchored", "l2", 127),
        ("residual", "l1", 255),
    ]:
        out = []
        for lr in (0.05, 0.1):
            c = cfg.ml.lightgbm.model_copy(update={"learning_rate": lr})
            t0 = time.perf_counter()
            sc = LightGBMModel(s.columns, form, obj, leaves, c, cfg.ml.seed).score(
                X, a, tr[m], va[m]
            )
            secs = time.perf_counter() - t0
            out.append(f"lr {lr}: {sc.mae:6.1f} MW {sc.best_iteration:5d} trees {secs:5.1f}s")
        print(f"band {lo:3d}-{hi:3d} {form:8s} {obj} {leaves:3d} | " + " | ".join(out), flush=True)

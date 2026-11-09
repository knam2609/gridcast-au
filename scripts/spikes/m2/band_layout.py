"""M2 second development run: lead-band layout and the ``latest`` target form, pre-scoring data.

usage: uv run python scripts/spikes/m2/band_layout.py

The first development run had two weaknesses (reports/m2_demand.md): at 5 minutes LightGBM
lost to persistence, and MAE jumped by ~77 MW where the short-lead fit (≤ 2 h) hands over to
the long-lead fit (> 2 h). This spike picks the fix on data the scored period never sees: the
early-stopping slice of the first retrain (2022-11-01; issued 6 Sep → 31 Oct 2022).

For LightGBM C (L1, 31 leaves: the most frequent winner in the first run) it fits every
target form on every candidate band, mimics the walk-forward's selection (lowest pooled
early-stopping MAE per band), and prints, per layout and form set, the MAE in lead windows
either side of each boundary. ``latest`` is the new form: y(t) − y(o).
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
from gridcast.models.ml import make_model

FORMS_RUN1 = ("residual", "direct", "anchored")
FORMS_RUN2 = (*FORMS_RUN1, "latest")
LAYOUTS = {
    "2 bands (run 1)": ((1, 24), (25, 288)),
    "3 bands": ((1, 24), (25, 72), (73, 288)),
    "4 bands": ((1, 6), (7, 24), (25, 72), (73, 288)),
}
WINDOWS = {
    "5 min": (1, 1),
    "5m-1h": (1, 12),
    "1h50-2h": (22, 24),
    "2h05-15": (25, 27),
    "5h50-6h": (70, 72),
    "6h05-15": (73, 75),
    "≤2 h": (1, 24),
    "2-24 h": (25, 288),
    "all": (1, 288),
}


def main() -> None:
    cfg = load_config()
    builder = load_feature_builder(get_engine(), cfg, "VIC1")
    cutoff = int(builder.ctx.times.searchsorted(nem_midnight_utc(str(cfg.ml.first_issue))))
    es_steps = cfg.ml.early_stopping_weeks * 7 * DAY_STEPS
    sample = wf.training_sample(
        builder, "C", wf.train_origins(builder, cutoff), cfg.ml.sample_every_steps
    )
    rows, es_train, es_valid = wf.fit_rows(sample, sample.usable, cutoff, es_steps)
    X, actual = sample.X[rows], sample.actual[rows].astype(np.float64)
    h_all = sample.horizons[rows].astype(int)
    hv = h_all[es_valid]
    truth = actual[es_valid]
    dem_now = X[es_valid, sample.columns.index("dem_now")].astype(np.float64)
    print(f"early-stopping slice of retrain {cfg.ml.first_issue}: {int(es_valid.sum()):,} pairs")

    bands = sorted({b for layout in LAYOUTS.values() for b in layout})
    # (band, form) -> (pooled ES MAE, forecasts for es_valid rows in the band, trees)
    fits: dict[tuple[tuple[int, int], str], tuple[float, np.ndarray, int]] = {}
    for lo, hi in bands:
        in_band = (h_all >= lo) & (h_all <= hi)
        sel = np.flatnonzero(in_band)
        Xb, ab = X[sel], actual[sel]
        tr, va = es_train[sel], es_valid[sel]
        for form in FORMS_RUN2:
            choice = {"target_form": form, "objective": "l1", "num_leaves": 31}
            model = make_model("lightgbm", sample.columns, choice, cfg.ml)
            y = model.target(Xb, ab)
            dtrain = lgb.Dataset(Xb[tr], y[tr], feature_name=sample.columns)
            dvalid = lgb.Dataset(Xb[va], y[va], reference=dtrain)
            booster = lgb.train(
                model._lgb_params(),  # type: ignore[attr-defined]
                dtrain,
                cfg.ml.lightgbm.max_rounds,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(cfg.ml.lightgbm.early_stopping_rounds, verbose=False)
                ],
            )
            pred = model.baseline(Xb[va]) + booster.predict(
                Xb[va], num_iteration=booster.best_iteration
            )
            mae = float(np.mean(np.abs(pred - ab[va])))
            fits[((lo, hi), form)] = (mae, pred, int(booster.best_iteration))
            print(f"  band {lo:3d}-{hi:3d} {form:9s} ES MAE {mae:7.1f}  "
                  f"{booster.best_iteration:5d} trees", flush=True)  # fmt: skip

    def forecast(layout: tuple[tuple[int, int], ...], forms: tuple[str, ...]) -> tuple:
        out = np.full(len(hv), np.nan)
        chosen = []
        for lo, hi in layout:
            form = min(forms, key=lambda f: fits[((lo, hi), f)][0])
            chosen.append(f"{lo}-{hi}:{form}")
            out[(hv >= lo) & (hv <= hi)] = fits[((lo, hi), form)][1]
        return out, chosen

    print("\nMAE (MW) on the early-stopping slice, by lead window")
    print("".ljust(34) + "".join(w.rjust(9) for w in WINDOWS))

    def row(name: str, pred: np.ndarray) -> None:
        cells = []
        for lo, hi in WINDOWS.values():
            s = (hv >= lo) & (hv <= hi)
            cells.append(f"{np.mean(np.abs(pred[s] - truth[s])):9.1f}")
        print(name.ljust(34) + "".join(cells))

    row("persistence", dem_now)
    for name, layout in LAYOUTS.items():
        for label, forms in (("3 forms", FORMS_RUN1), ("+latest", FORMS_RUN2)):
            pred, chosen = forecast(layout, forms)
            row(f"{name}, {label}", pred)
            print("    chosen: " + ", ".join(chosen))


if __name__ == "__main__":
    main()

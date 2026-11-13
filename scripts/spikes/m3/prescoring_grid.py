"""M3 step 1: pre-scoring spike that fixes the conformal candidate grid (m3_plan.md §2.2).

usage: uv run python scripts/spikes/m3/prescoring_grid.py docs/m3_prescoring_tables.md [--no-weather]

Reads run 2's stored development forecasts (batch ``main``) and keeps **only issue times
before 2023-08-01** (the pre-scoring data; the interval development period starts there).
Every stream runs from the first ML issue time (2022-11-01) on past data only; candidates
are compared on issue days 2023-02-03 → 2023-07-31, when every window of up to 13 weeks
is full (26-week windows cannot be evaluated on pre-scoring data).

Sections written to OUT.md:
  1. LightGBM C residual spread by local hour of the target, per lead band;
  2. time-of-day block layouts: per lead × block streams against per-lead streams (γ = 0);
  3. ACI steps γ for per-lead streams: interval score, coverage, coverage by band and month,
     and how often a bound sits at the window's extreme; LightGBM C and the baselines;
  4. cold days: the day-ahead GFS forecast daily maximum (population-weighted) per local
     day, with counts below candidate thresholds and the hot-day (≥ 35 °C) count
     (needs the database: `make db-up`; skip with --no-weather).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.config import load_config
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.conformal import (
    CandidateGrid,
    ConformalResult,
    band_label,
    point_problem,
    run_conformal,
    tod_labels,
)
from gridcast.intervals.inputs import StoredBatch, read_stored_batch
from gridcast.intervals.scores import LEVELS
from gridcast.timeutils import NEM_TZ

BATCH = "6d783932-5834-4e18-a9e8-0e3567db46e6"  # run 2, evaluation set "main"
REGION = "VIC1"
TZ = "Australia/Melbourne"
PRESCORING_END = "2023-08-01"  # first issue day of the interval development period
EVAL_START = pd.Timestamp("2023-02-03").date()
EVAL_END = pd.Timestamp("2023-07-31").date()
WINDOWS = (4.0, 8.0, 13.0)
GAMMAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
SELECTION_WEEKS = 8.0
RETRAIN = "2023-05-01"  # a retrain the run needs; the spike reads every candidate's daily stats
MODELS = ("lgbm_C", "persistence", "daily_naive", "weekly_naive")
LAYOUTS: dict[str, tuple[int, ...]] = {
    "2 (07-22, 22-07)": (7, 22),
    "3 x 8 h": (0, 8, 16),
    "4 x 6 h": (0, 6, 12, 18),
    "6 x 4 h": (0, 4, 8, 12, 16, 20),
    "8 x 3 h": tuple(range(0, 24, 3)),
    "12 x 2 h": tuple(range(0, 24, 2)),
    "24 x 1 h": tuple(range(24)),
}


def md_table(frame: pd.DataFrame, floatfmt: str = ".1f") -> str:
    return frame.to_markdown(floatfmt=floatfmt)


def evaluate(res: ConformalResult, by: list[str]) -> pd.DataFrame:
    """Pooled stats over the evaluation days, grouped by ``by`` (from the daily table)."""
    d = res.daily
    d = d.loc[(d["nem_day"] >= EVAL_START) & (d["nem_day"] <= EVAL_END)]
    g = d.groupby(by)[["n", "covered", "below", "above", "width", "interval_score"]].sum()
    out = pd.DataFrame(index=g.index)
    out["coverage_%"] = 100 * g["covered"] / g["n"]
    out["below_%"] = 100 * g["below"] / g["n"]
    out["above_%"] = 100 * g["above"] / g["n"]
    out["width"] = g["width"] / g["n"]
    out["IS"] = g["interval_score"] / g["n"]
    out["n"] = g["n"]
    return out


def problem_for(batch: StoredBatch, model: str, layout: tuple[int, ...], setup_steps: int):  # type: ignore[no-untyped-def]
    mask = batch.common_mask()
    groups = {
        "lead": (np.zeros(batch.actual.shape, dtype=np.int8), 1),
        "lead_tod": tod_labels(batch.origin_times, batch.n_leads, setup_steps, TZ, layout),
    }
    return point_problem(
        batch.origin_times,
        setup_steps,
        0,
        batch.actual,
        batch.forecasts[model],
        mask,
        groups,
        LEVELS,
    )


def section_spread(batch: StoredBatch, bands: list[tuple[int, int]]) -> str:
    mask = batch.common_mask()
    resid = batch.actual - batch.forecasts["lgbm_C"]
    hours, _ = tod_labels(batch.origin_times, batch.n_leads, 6, TZ, tuple(range(24)))
    rows = {}
    for lo, hi in bands:
        cols = slice(lo - 1, hi)
        r, h, m = resid[:, cols], hours[:, cols], mask[:, cols]
        spread = {}
        for hour in range(24):
            v = r[m & (h == hour)]
            spread[hour] = np.quantile(v, 0.9) - np.quantile(v, 0.1)
        rows[band_label((lo, hi))] = spread
    table = pd.DataFrame(rows)
    table.index.name = "local hour"
    return (
        "## 1. Residual spread by local hour of the target (LightGBM C)\n\n"
        "Width of the central 80% of residuals (q90 − q10, MW), all pre-scoring issue "
        "times, by lead band.\n\n" + md_table(table, ".0f") + "\n"
    )


def section_layouts(batch: StoredBatch, bands: list[tuple[int, int]], retrain: pd.Timestamp) -> str:
    grid = CandidateGrid(WINDOWS, ("lead", "lead_tod"), (0.0,))
    parts = []
    lead_ref = None
    for name, layout in LAYOUTS.items():
        t0 = time.perf_counter()
        res = run_conformal(
            problem_for(batch, "lgbm_C", layout, 6), grid, LEVELS, [retrain], bands, SELECTION_WEEKS
        )
        stats = evaluate(res, ["candidate", "level"])
        is_tab = stats["IS"].unstack("level")
        cov = stats["coverage_%"].unstack("level")
        if lead_ref is None:
            lead_ref = is_tab.loc[[c for c in is_tab.index if "_lead_g" in c]]
            lead_cov = cov.loc[lead_ref.index]
        tod = is_tab.loc[[c for c in is_tab.index if "lead_tod" in c]]
        for c in tod.index:
            w = c.split("_")[0]
            ref = lead_ref.loc[f"{w}_lead_g0"]
            row = {"layout": name, "window": w}
            for level in LEVELS:
                row[f"IS{level}"] = tod.loc[c, level]
                row[f"vs lead {level} %"] = 100 * (tod.loc[c, level] / ref[level] - 1)
                row[f"cov{level}"] = cov.loc[c, level]
            parts.append(row)
        print(f"layout {name}: {time.perf_counter() - t0:.0f}s", flush=True)
    assert lead_ref is not None
    ref_rows = []
    for c in lead_ref.index:
        row = {"layout": "per lead (no blocks)", "window": c.split("_")[0]}
        for level in LEVELS:
            row[f"IS{level}"] = lead_ref.loc[c, level]
            row[f"vs lead {level} %"] = 0.0
            row[f"cov{level}"] = lead_cov.loc[c, level]
        ref_rows.append(row)
    table = pd.DataFrame(ref_rows + parts).set_index(["layout", "window"])
    return (
        "## 2. Time-of-day block layouts (LightGBM C, γ = 0)\n\n"
        "Mean interval score (MW) and coverage (%) on issue days "
        f"{EVAL_START} → {EVAL_END}; `vs lead` is the change against per-lead streams with "
        "the same window (negative = better). Blocks are local hours of the target's "
        "start; the first block starts at the first listed hour.\n\n"
        + md_table(table, ".1f")
        + "\n"
    )


def section_aci(batch: StoredBatch, bands: list[tuple[int, int]], retrain: pd.Timestamp) -> str:
    grid = CandidateGrid(WINDOWS, ("lead",), GAMMAS)
    out = ["## 3. ACI steps (per-lead streams)\n"]
    for model in MODELS:
        t0 = time.perf_counter()
        res = run_conformal(
            problem_for(batch, model, (0,), 6), grid, LEVELS, [retrain], bands, SELECTION_WEEKS
        )
        print(f"aci {model}: {time.perf_counter() - t0:.0f}s", flush=True)
        stats = evaluate(res, ["candidate", "level"])
        wide = pd.DataFrame(
            {
                f"{stat}{level}": stats[col].xs(level, level="level")
                for level in LEVELS
                for stat, col in (("IS", "IS"), ("cov", "coverage_%"))
            }
        )
        wide.index = pd.MultiIndex.from_tuples(
            [(c.split("_")[0], float(c.split("_g")[1])) for c in wide.index],
            names=["window", "gamma"],
        )
        wide = wide.sort_index()
        out.append(f"### 3.{MODELS.index(model) + 1} {model}\n")
        out.append(
            f"Mean interval score (MW) and coverage (%), issue days {EVAL_START} → {EVAL_END}.\n"
        )
        out.append(md_table(wide, ".1f") + "\n")
        if model == "lgbm_C":
            band_stats = evaluate(res, ["candidate", "level", "band"])
            cov80 = band_stats["coverage_%"].xs(80, level="level").unstack("band")
            out.append("80% coverage by lead band (LightGBM C):\n")
            out.append(md_table(cov80, ".1f") + "\n")
            d = res.daily.copy()
            d = d.loc[(d["nem_day"] >= EVAL_START) & (d["nem_day"] <= EVAL_END)]
            d["month"] = pd.to_datetime(d["nem_day"]).dt.strftime("%Y-%m")
            m = d.loc[d["level"] == 80].groupby(["candidate", "month"])[["n", "covered"]].sum()
            month = (100 * m["covered"] / m["n"]).unstack("month")
            out.append("80% coverage by month of issue (LightGBM C):\n")
            out.append(md_table(month, ".1f") + "\n")
            is_m = (
                d.loc[d["level"] == 80]
                .groupby(["candidate", "month"])[["n", "interval_score"]]
                .sum()
            )
            out.append("80% interval score by month of issue (LightGBM C):\n")
            out.append(
                md_table((is_m["interval_score"] / is_m["n"]).unstack("month"), ".1f") + "\n"
            )
            clip = res.clipped.pivot_table(
                index="candidate", columns=["level", "tail"], values="clipped_share"
            )
            out.append(
                "Share of bounds at the calibration window's extreme (rank clipped), output "
                f"issue times from {retrain.tz_convert(NEM_TZ).date()} (NEM), LightGBM C, %:\n"
            )
            out.append(md_table(100 * clip, ".2f") + "\n")
    return "\n".join(out)


def section_cold(cfg) -> str:  # type: ignore[no-untyped-def]
    from gridcast.database.engine import get_engine
    from gridcast.features.inputs import load_weather_inputs

    first = nem_midnight_utc("2022-11-01")
    last = nem_midnight_utc("2023-08-02")
    times = pd.date_range(first + pd.Timedelta(minutes=5), last, freq="5min")
    weather = load_weather_inputs(get_engine(), cfg, REGION, times)
    local_day = (times - pd.Timedelta(minutes=5)).tz_convert(TZ).date
    fc = pd.Series(weather.forecasts[1].series["temp_weighted"], index=times)
    obs = pd.Series(weather.observed.series["temp_weighted"], index=times)  # type: ignore[union-attr]
    daily = pd.DataFrame(
        {
            "forecast_max": fc.groupby(local_day).max(),
            "forecast_n": fc.groupby(local_day).count(),
            "era5_max": obs.groupby(local_day).max(),
        }
    )
    daily = daily.loc[
        (daily.index >= pd.Timestamp("2022-11-01").date()) & (daily["forecast_n"] == 288)
    ]
    rows = []
    for thr in range(8, 17):
        rows.append(
            {
                "threshold °C": thr,
                "days forecast max ≤ thr": int((daily["forecast_max"] <= thr).sum()),
                "days ERA5 max ≤ thr": int((daily["era5_max"] <= thr).sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("threshold °C")
    hot = int((daily["forecast_max"] >= 35).sum())
    hot_obs = int((daily["era5_max"] >= 35).sum())
    winter = daily.loc[pd.to_datetime(daily.index).month.isin([6, 7])]
    q = daily["forecast_max"].quantile([0.01, 0.02, 0.05, 0.10])
    return (
        "## 4. Cold days (GFS day-ahead forecast, population-weighted daily maximum)\n\n"
        f"Local days with a complete day-ahead forecast: {len(daily)} "
        f"({daily.index.min()} → {daily.index.max()}). Hot days (forecast max ≥ 35 °C): "
        f"{hot} (ERA5: {hot_obs}). June–July days: {len(winter)}, forecast daily max "
        f"min {winter['forecast_max'].min():.1f}, median {winter['forecast_max'].median():.1f} °C.\n\n"
        "Quantiles of the forecast daily maximum over all days: "
        + ", ".join(f"q{int(100 * k):02d} = {v:.1f} °C" for k, v in q.items())
        + ".\n\n"
        + md_table(table, ".0f")
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--no-weather", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    bands = [tuple(b) for b in cfg.ml.lead_bands]
    batch = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), BATCH)
    batch = batch.restrict(end=nem_midnight_utc(PRESCORING_END))
    assert batch.origin_times[-1] < nem_midnight_utc(PRESCORING_END)
    retrain = nem_midnight_utc(RETRAIN)
    header = (
        "# M3 step 1 pre-scoring spike: tables\n\n"
        f"Generated by `scripts/spikes/m3/prescoring_grid.py` from batch `{BATCH}`, issue "
        f"times {batch.origin_times[0]} → {batch.origin_times[-1]} (UTC) only.\n\n"
    )
    sections = [header, section_spread(batch, bands)]
    sections.append(section_layouts(batch, bands, retrain))
    sections.append(section_aci(batch, bands, retrain))
    if not args.no_weather:
        sections.append(section_cold(cfg))
    args.out.write_text("\n".join(sections))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

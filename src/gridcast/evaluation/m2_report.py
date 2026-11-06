"""Generate ``reports/m2_demand.md`` (+ figures) from the stored M2 walk-forward run.

Nothing in the report is typed by hand. Every number is read from the stored batches
(``model_runs``, ``forecast_metrics``, the prediction parquets and model files), from the
weather drop/add test (``gridcast weather-ablation``), or measured when the report is
built (the leakage property tests are run by this command). S1–S4 are computed exactly as
pre-registered in m2_plan.md §3; nothing here changes a model.
"""

from __future__ import annotations

import datetime as dt
import gzip
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sqlalchemy import Engine

from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.m2_results import (
    actual_path,
    artifact_path,
    load_evaluation_set,
    retrain_records,
    run_config,
)
from gridcast.evaluation.report import (
    AXIS,
    GRID,
    INK,
    INK_2,
    MODEL_COLORS,
    MUTED,
    SURFACE,
    TargetResults,
    _fmt,
    _md_table,
    horizon_label,
    horizon_table,
    model_label,
)
from gridcast.evaluation.segments import describe_segments, describe_weather_segments
from gridcast.features.builder import FeatureBuilder
from gridcast.features.calendar import load_calendar_days
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting.ablation import (
    fits_frame,
    flag_table,
    read_ablation,
    reproduction_ok,
)
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

HEADLINE, WEATHER_FREE, ORACLE = "lgbm_C", "lgbm_A", "lgbm_D"
S1_FIRST_STEP = 24  # S1 covers every lead time from 2 h (h = 24) to 24 h (m2_plan.md §3)
LEAD_GROUPS = ((1, 12), (13, 24), (25, 72), (73, 144), (145, 288))
LEAKAGE_TESTS = (
    "tests/unit/test_feature_leakage.py",
    "tests/unit/test_leakage.py",
    "tests/unit/test_walkforward.py::test_forecasts_ignore_demand_after_their_issue_time",
)
MAIN_PLOT = ("daily_naive", "persistence", "lgbm_A", "ridge_C", "lgbm_C")  # validated order
SHORT_PLOT = ("persistence", "lgbm_A", "lgbm_C")
MAIN_TABLE = (
    "lgbm_A", "lgbm_C", "lgbm_D", "ridge_A", "ridge_C", "ridge_D",
    "persistence", "daily_naive", "weekly_naive",
)  # fmt: skip
SEGMENT_MODELS = ("lgbm_A", "lgbm_C", "lgbm_D", "ridge_C", "persistence", "daily_naive")
FAILURE_MODELS = ("lgbm_C", "lgbm_A", "daily_naive")
FULL_DAY_SHARE = 0.9  # failure days need this share of the median day's scored pairs
FIGURE_LEAD = "m2_demand_mae_by_lead.png"
FIGURE_SHORT = "m2_demand_short_leads.png"


# --------------------------------------------------------------------------- leakage tests


@dataclass(frozen=True, slots=True)
class LeakageOutcome:
    """Result of running the leakage property tests (part of S4)."""

    passed: int
    failed: int
    returncode: int
    tests: tuple[str, ...] = LEAKAGE_TESTS

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.failed == 0 and self.passed > 0


def parse_pytest_summary(output: str) -> tuple[int, int]:
    """``(passed, failed + errors)`` from pytest's final summary line."""
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", last))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)\b", last))
    return passed, failed


def run_leakage_tests() -> LeakageOutcome:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *LEAKAGE_TESTS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed, failed = parse_pytest_summary(proc.stdout)
    return LeakageOutcome(passed, failed, proc.returncode)


# --------------------------------------------------------------------------- criteria


@dataclass(frozen=True, slots=True)
class Criterion:
    key: str
    rule: str
    measured: str
    passed: bool


def baselines_of(res: TargetResults) -> list[str]:
    return [str(m) for m in res.runs.loc[res.runs["model_kind"] == "baseline", "model_name"]]


def pooled(res: TargetResults, split: str = "dev") -> dict[str, float]:
    """Pooled MAE (all lead times) per model."""
    return {m: res.value(m, split, "all", 0, "mae") for m in res.models}


def success_criteria(
    table: pd.DataFrame,
    pooled_mae: Mapping[str, float],
    baselines: Sequence[str],
    leakage: LeakageOutcome,
    s1_first: int = S1_FIRST_STEP,
) -> list[Criterion]:
    """S1–S4 of m2_plan.md §3 from the MAE-by-lead ``table`` (horizon x model)."""
    envelope = table[list(baselines)].min(axis=1)
    margin = (table[HEADLINE] - envelope).loc[s1_first:]
    closest = int(margin.index[int(np.argmax(margin.to_numpy()))])
    pos = int(np.flatnonzero(table.index.to_numpy() == closest)[0])
    c, a, d = pooled_mae[HEADLINE], pooled_mae[WEATHER_FREE], pooled_mae[ORACLE]
    mean_envelope = float(envelope.mean())
    tests = (
        f"leakage tests {leakage.passed} passed, {leakage.failed} failed"
        if leakage.returncode >= 0
        else "leakage tests not run"
    )
    return [
        Criterion(
            "S1",
            "MAE_C(h) < min over baselines of MAE(h), for every h from 2 h to 24 h",
            f"{int((margin < 0).sum())} of {len(margin)} lead times; closest at "
            f"{horizon_label(closest)}: {float(table[HEADLINE].to_numpy()[pos]):,.1f} vs "
            f"{float(envelope.to_numpy()[pos]):,.1f} MW",
            bool((margin < 0).all()),
        ),
        Criterion(
            "S2",
            "pooled MAE_C < mean over h of (min over baselines of MAE(h))",
            f"{c:,.1f} vs {mean_envelope:,.1f} MW",
            bool(c < mean_envelope),
        ),
        Criterion(
            "S3",
            "pooled MAE_C < pooled MAE_A",
            f"{c:,.1f} vs {a:,.1f} MW",
            bool(c < a),
        ),
        Criterion(
            "S4",
            "pooled MAE_D ≤ pooled MAE_C, and every leakage property test passes",
            f"{d:,.1f} vs {c:,.1f} MW; {tests}",
            bool(d <= c and leakage.ok),
        ),
    ]


def criteria_table(criteria: Sequence[Criterion]) -> str:
    rows = [[c.key, c.rule, c.measured, "**PASS**" if c.passed else "**FAIL**"] for c in criteria]
    return _md_table(["#", "Pass if", "Measured", "Result"], rows, "lllc")


# --------------------------------------------------------------------------- lead-time facts


def lead_group_label(lo: int, hi: int) -> str:
    return f"{horizon_label(lo)}–{horizon_label(hi)}"


def group_means(
    table: pd.DataFrame, groups: Sequence[tuple[int, int]] = LEAD_GROUPS
) -> pd.DataFrame:
    """Mean per-horizon MAE of each model within each lead group."""
    rows = {lead_group_label(lo, hi): table.loc[lo:hi].mean() for lo, hi in groups}
    return pd.DataFrame(rows).T


def first_better(table: pd.DataFrame, model: str, reference: str) -> int | None:
    better = table.index[table[model] < table[reference]]
    return int(better.min()) if len(better) else None


def band_jump(table: pd.DataFrame, model: str, last_short: int) -> tuple[float, float]:
    """MAE step across the band boundary, and the median absolute step elsewhere."""
    series = table[model]
    steps = series.diff().abs().drop(index=[series.index[0], last_short + 1])
    return float(series.loc[last_short + 1] - series.loc[last_short]), float(steps.median())


# --------------------------------------------------------------------------- figures


def _line_chart(
    table: pd.DataFrame,
    models: Sequence[str],
    path: Path,
    title: str,
    xmax_hours: float,
    boundary_hours: float | None,
    ymax: float | None = None,
) -> None:
    """MAE vs lead time, one 2px line per model, legend above and direct labels at the end."""
    sel = table.loc[table.index * 5 / 60 <= xmax_hours, list(models)]
    hours = sel.index.to_numpy() * 5 / 60
    fig = Figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    for model in models:
        ax.plot(
            hours,
            sel[model],
            color=MODEL_COLORS.get(model, INK),
            linewidth=2,
            label=model_label(model),
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    top = ymax if ymax is not None else float(sel.to_numpy().max()) * 1.08
    ax.set_ylim(0, top)
    ax.set_xlim(0, xmax_hours)
    if boundary_hours is not None:
        ax.axvline(boundary_hours, color=AXIS, linewidth=1)
        ax.annotate(f"lead bands meet ({boundary_hours:g} h)", xy=(boundary_hours, top * 0.97),
                    xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8,
                    va="top")  # fmt: skip

    ends = sorted((min(float(sel[m].iloc[-1]), top * 0.98), m) for m in models)
    gap = top * 0.05
    placed: list[float] = []
    for y, model in ends:
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.annotate(model_label(model), xy=(xmax_hours, y),
                    xytext=(xmax_hours * 1.02, y_label), color=INK_2, fontsize=8.5,
                    va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6})  # fmt: skip

    ax.set_xlabel("Forecast lead time (hours after issue time)", color=INK_2, fontsize=9)
    ax.set_ylabel("MAE (MW)", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=30)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                       fontsize=8.5, ncol=len(models), handlelength=1.6)  # fmt: skip
    for t in legend.get_texts():
        t.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.84, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


# --------------------------------------------------------------------------- stored models


def selection_table(records: Sequence[Mapping[str, Any]]) -> str:
    """Per retrain and band: the chosen setting and the best early-stopping MAE per form."""
    forms = list(
        dict.fromkeys(c["setting"]["target_form"] for r in records for c in r["candidates"])
    )
    headers = ["Retrain", "Lead band", "Chosen", "Trees", *[f"Best {f} (MW)" for f in forms]]
    rows = []
    for r in records:
        ch = r["chosen"]
        best = {
            f: min(c["mae"] for c in r["candidates"] if c["setting"]["target_form"] == f)
            for f in forms
        }
        winner = min(best.values())
        chosen = f"{ch['target_form']}, {ch['objective'].upper()}, {ch['num_leaves']} leaves"
        rows.append(
            [
                r["retrain"],
                band_label(r["band"]),
                chosen,
                f"{r['params'].get('n_trees') or '–'}",
                *[f"**{_fmt(best[f])}**" if best[f] == winner else _fmt(best[f]) for f in forms],
            ]
        )
    return _md_table(headers, rows, "llll" + "r" * len(forms))


def band_label(key: str) -> str:
    lo, hi = (int(x) for x in key.removeprefix("h").split("-"))
    return lead_group_label(lo, hi)


def chosen_counts(records_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> pd.DataFrame:
    """How often each target form won, by family and lead band (grid-selected fits only)."""
    rows = []
    for name, records in records_by_model.items():
        family = "LightGBM" if name.startswith("lgbm") else "Ridge"
        for r in records:
            rows.append({"family": family, "band": band_label(r["band"]),
                         "form": r["chosen"]["target_form"]})  # fmt: skip
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["family", "band"], columns="form", aggfunc="size", fill_value=0)


def importance(records: Sequence[Mapping[str, Any]], data_root: Path) -> dict[str, pd.DataFrame]:
    """Per lead band: gain and split shares of each feature across retrains (LightGBM)."""
    import lightgbm as lgb

    per_band: dict[str, list[pd.DataFrame]] = {}
    for r in records:
        with gzip.open(data_root / r["path"], "rt", encoding="utf-8") as fh:
            booster = lgb.Booster(model_str=fh.read())
        gain = booster.feature_importance("gain")
        split = booster.feature_importance("split")
        frame = pd.DataFrame(
            {"gain": gain / gain.sum(), "split": split / split.sum()},
            index=booster.feature_name(),
        )
        per_band.setdefault(r["band"], []).append(frame)
    out = {}
    for band, frames in per_band.items():
        gains = pd.concat([f["gain"] for f in frames], axis=1)
        splits = pd.concat([f["split"] for f in frames], axis=1)
        out[band] = pd.DataFrame(
            {
                "gain_mean": gains.mean(axis=1),
                "gain_min": gains.min(axis=1),
                "gain_max": gains.max(axis=1),
                "split_mean": splits.mean(axis=1),
            }
        ).sort_values("gain_mean", ascending=False)
    return out


def importance_table(frame: pd.DataFrame, top: int = 12) -> str:
    rows = [
        [
            f"`{name}`",
            f"{100 * r['gain_mean']:.1f}",
            f"{100 * r['gain_min']:.1f}–{100 * r['gain_max']:.1f}",
            f"{100 * r['split_mean']:.1f}",
        ]
        for name, r in frame.head(top).iterrows()
    ]
    return _md_table(
        ["Feature", "Gain share, mean (%)", "Range over retrains (%)", "Split share, mean (%)"],
        rows,
    )


# --------------------------------------------------------------------------- pair-level data


@dataclass(slots=True)
class Pairs:
    """Stored forecasts of one batch on the common (issue time, horizon) pairs."""

    origin_times: pd.DatetimeIndex
    origins: np.ndarray  # positions on the context grid
    predictions: dict[str, np.ndarray]  # (n_origins, H) float32
    actual: np.ndarray  # (n_origins, H)
    mask: np.ndarray  # (n_origins, H) bool: scored by every model of the batch


def load_pairs(res: TargetResults, data_root: Path, times: pd.DatetimeIndex) -> Pairs:
    frames = {m: pd.read_parquet(artifact_path(res, m, data_root)) for m in res.models}
    actual = pd.read_parquet(actual_path(res, data_root))
    origin_times = pd.DatetimeIndex(actual["origin_time"])
    cols = [c for c in actual.columns if c != "origin_time"]
    predictions = {}
    for m, f in frames.items():
        if not pd.DatetimeIndex(f["origin_time"]).equals(origin_times):
            raise ValueError(f"{m}: issue times differ from the actuals")
        predictions[m] = f[cols].to_numpy(dtype=np.float32)
    y = actual[cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(y)
    for p in predictions.values():
        mask &= np.isfinite(p)
    origins = times.get_indexer(origin_times)
    if (origins < 0).any():
        raise ValueError("stored issue times are not on the data grid")
    return Pairs(origin_times, origins, predictions, y, mask)


def daily_errors(
    pairs: Pairs,
    local_day: np.ndarray,
    models: Sequence[str],
    lo: int,
    hi: int,
) -> pd.DataFrame:
    """Per local target day: pairs, MAE of each model and the first model's bias (h lo…hi)."""
    horizons = np.arange(lo, hi + 1)
    targets = pairs.origins[:, None] + horizons[None, :]
    mask = pairs.mask[:, lo - 1 : hi]
    day = local_day[targets][mask]
    y = pairs.actual[:, lo - 1 : hi][mask]
    data: dict[str, np.ndarray] = {"day": day}
    for m in models:
        err = pairs.predictions[m][:, lo - 1 : hi][mask].astype(np.float64) - y
        data[f"ae_{m}"] = np.abs(err)
        if m == models[0]:
            data["err"] = err
    g = pd.DataFrame(data).groupby("day")
    out = g[[f"ae_{m}" for m in models]].mean()
    out.columns = list(models)
    out["bias"] = g["err"].mean()
    out["n"] = g.size()
    return out


def yearly_errors(pairs: Pairs, models: Sequence[str]) -> pd.DataFrame:
    """Pooled MAE per NEM year of the issue time (all lead times)."""
    year = pairs.origin_times.tz_convert(NEM_TZ).year.to_numpy()
    month = pairs.origin_times.tz_convert(NEM_TZ).month.to_numpy()
    rows = {}
    for yr in np.unique(year):
        sel = year == yr
        mask = pairs.mask[sel]
        y = pairs.actual[sel][mask]
        months = np.unique(month[sel])
        if len(months) == 12:
            label = str(yr)
        elif len(months) == 1:
            label = f"{yr} ({_month(months[0])})"
        else:
            label = f"{yr} ({_month(months[0])}–{_month(months[-1])})"
        row = {m: float(np.mean(np.abs(pairs.predictions[m][sel][mask] - y))) for m in models}
        row["n"] = int(mask.sum())
        rows[label] = row
    return pd.DataFrame(rows).T


def _month(m: int) -> str:
    return dt.date(2000, int(m), 1).strftime("%b")


# --------------------------------------------------------------------------- report


@dataclass(slots=True)
class DelayCheck:
    """The publication-delay sensitivity batch, as the report uses it."""

    model: str  # e.g. lgbm_C_delay8h
    hours: float  # assumed delay in the check
    headline_hours: float  # configured delay of the headline run
    table: pd.DataFrame  # MAE by horizon
    pooled: dict[str, float]

    @classmethod
    def of(cls, res: TargetResults, headline_hours: float) -> DelayCheck:
        model = next(m for m in res.models if m.startswith(f"{HEADLINE}_delay"))
        return cls(
            model,
            float(run_config(res, model)["delay_hours"]),
            headline_hours,
            res.by_horizon("dev", "mae"),
            pooled(res),
        )


@dataclass(slots=True)
class Loaded:
    main: TargetResults
    plus: TargetResults | None
    delay: TargetResults | None


def _optional(engine: Engine, region_id: str, key: str) -> TargetResults | None:
    try:
        return load_evaluation_set(engine, region_id, key)
    except LookupError:
        return None


def _provenance(cfg: AppConfig, sets: Loaded, leakage: LeakageOutcome, region_id: str) -> str:
    main = sets.main
    rc = run_config(main, HEADLINE)
    all_runs = pd.concat([s.runs for s in (sets.main, sets.plus, sets.delay) if s is not None])
    shas = sorted({str(s) for s in all_runs["git_sha"]})
    hashes = sorted({str(c["config_hash"]) for c in all_runs["config"]})
    current = ml_config_hash(cfg)
    excluded = rc["total_pairs"] - rc["scored_pairs"]
    batches = ", ".join(
        f"{k} `{s.batch_id}`"
        for k, s in (("main", sets.main), ("plus", sets.plus), ("delay", sets.delay))
        if s is not None
    )
    first = pd.Timestamp(rc["first_issue"]).tz_convert(NEM_TZ)
    last = pd.Timestamp(rc["last_issue"]).tz_convert(NEM_TZ)
    rows = [
        ["Region", region_id],
        ["Batches", batches],
        ["Git commit(s) of the runs", ", ".join(f"`{s}`" for s in shas)],
        [
            "ML config hash",
            ", ".join(f"`{h}`" for h in hashes)
            + (
                " (matches the current config)"
                if hashes == [current]
                else f" (**differs** from the current config `{current}`)"
            ),
        ],
        [
            "Development issue times (NEM time)",
            f"{first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}, {rc['n_origins']:,} "
            f"(every {5 * rc['setup']['issue_every_steps']} min)",
        ],
        [
            "Retrains",
            f"{len(rc['retrains'])}, every {rc['ml']['retrain_every_months']} months, "
            f"{rc['retrains'][0]} → {rc['retrains'][-1]}",
        ],
        [
            "Lead bands (steps of 5 min)",
            ", ".join(f"{lo}–{hi}" for lo, hi in rc["lead_bands"]),
        ],
        [
            "Scored (issue time, lead) pairs, every model",
            f"{rc['scored_pairs']:,} of {rc['total_pairs']:,} "
            f"({excluded:,} excluded, {100 * excluded / rc['total_pairs']:.2f}%)",
        ],
        [
            "Training rows at the last retrain (shared by A, C, D)",
            f"{rc['shared_train_rows']:,}",
        ],
        [
            "Leakage property tests (run by this command)",
            f"{leakage.passed} passed, {leakage.failed} failed",
        ],
        ["Report generated from commit", f"`{git_sha()}`"],
    ]
    return _md_table(["Item", "Value"], rows, "ll")


def _weather_value_table(main_table: pd.DataFrame, delay: DelayCheck | None) -> str:
    g = group_means(main_table)
    headers = ["Lead time", "LightGBM A", "LightGBM C", "LightGBM D", "C vs A (%)", "C − D (MW)"]
    dg = None
    if delay is not None:
        dg = group_means(delay.table)
        headers.append(f"{delay.hours:g} h delay − C (MW)")
    rows = []
    for label, r in g.iterrows():
        cells = [
            str(label),
            _fmt(r[WEATHER_FREE]),
            _fmt(r[HEADLINE]),
            _fmt(r[ORACLE]),
            f"{100 * (r[HEADLINE] / r[WEATHER_FREE] - 1):+.1f}",
            f"{r[HEADLINE] - r[ORACLE]:+.1f}",
        ]
        if dg is not None and delay is not None:
            d_row = dg.to_dict(orient="index")[label]
            cells.append(f"{d_row[delay.model] - d_row[HEADLINE]:+.1f}")
        rows.append(cells)
    return _md_table(headers, rows)


def _ablation_section(
    ablation: dict[str, Any] | None,
    main_pooled: Mapping[str, float],
    delay: DelayCheck | None,
    main_groups: pd.DataFrame,
    leakage: LeakageOutcome,
    flag_share: float,
    es_weeks: int,
) -> list[str]:
    out = ["## Weather value: the “investigate before believing” check\n"]
    gain = 1 - main_pooled[HEADLINE] / main_pooled[WEATHER_FREE]
    out.append(
        f"The plan (§2.5) says any feature that improves pooled MAE by more than "
        f"{100 * flag_share:g}% on its own is checked for leakage before it is believed. Over "
        f"the development period, weather (C vs A) cuts the pooled MAE by **{100 * gain:.1f}%**"
        + (
            ", above that threshold. Weather is a group of features, so each one is also "
            "tested alone below.\n"
            if gain > flag_share
            else ", within that threshold.\n"
        )
    )
    if ablation is None:
        out.append(
            "**Not run yet:** `make weather-ablation` writes the drop/add test this section "
            "reports.\n"
        )
    else:
        fits = fits_frame(ablation)
        flags = flag_table(fits, float(ablation["flag_share"]))
        share = 100 * float(ablation["flag_share"])
        out += [
            "**Drop/add test.** At each listed retrain, LightGBM C's chosen setting was refitted "
            f"on that retrain's early-stopping rows (the last {es_weeks} weeks before it are "
            "monitored) "
            "with each weather feature added alone to the weather-free set, or dropped from "
            "the full set. MAE is pooled over both lead bands. *Keep gain* is what the "
            "feature adds on top of the other weather features. A feature is flagged above "
            f"{share:g}%. The all-features fit "
            + (
                "reproduced the stored run's early-stopping MAE exactly at every retrain and band."
                if reproduction_ok(ablation)
                else "**did not** reproduce the stored run's early-stopping MAE; the "
                "test is not comparable to the stored run."
            )
            + f" Run from commit `{ablation['git_sha']}`.\n",
            _md_table(
                [
                    "Retrain",
                    "Feature",
                    "MAE no weather",
                    "MAE + feature",
                    "Add gain (%)",
                    "Keep gain (%)",
                    "All weather gain (%)",
                    "Flagged",
                ],
                [
                    [
                        str(r["retrain"]),
                        f"`{r['feature']}`",
                        _fmt(r["mae_none"]),
                        _fmt(r["mae_add"]),
                        f"{100 * r['add_gain']:.1f}",
                        f"{100 * r['keep_gain']:.1f}",
                        f"{100 * r['group_gain']:.1f}",
                        "**yes**" if r["flagged"] else "no",
                    ]
                    for r in flags.to_dict(orient="records")
                ],
            ),
            "",
        ]
        flagged = sorted(set(flags.loc[flags["flagged"], "feature"]))
        out.append(
            (
                "**Flagged:** " + ", ".join(f"`{f}`" for f in flagged) + ". "
                "The evidence below is the leakage check the plan asks for."
                if flagged
                else "**No single weather feature is flagged.** The evidence below "
                "covers the group."
            )
            + "\n"
        )
    g = main_groups.to_dict(orient="index")
    labels = list(g)
    first, last = labels[0], labels[-1]
    gain_first = 100 * (g[first][HEADLINE] / g[first][WEATHER_FREE] - 1)
    gain_last = 100 * (g[last][HEADLINE] / g[last][WEATHER_FREE] - 1)
    oracle_everywhere = all(r[HEADLINE] >= r[ORACLE] for r in g.values())
    evidence = [
        f"- **Leakage property tests:** {leakage.passed} passed, {leakage.failed} failed "
        "(future demand and unpublished forecast rows perturbed or deleted; C's features "
        "must not change).",
        f"- **Where the gain comes from:** C vs A is {gain_first:+.1f}% at {first} and "
        f"{gain_last:+.1f}% at {last}"
        + (
            ", so weather helps most at long leads, where recent demand says least."
            if gain_last < gain_first
            else "."
        ),
        f"- **Perfect foresight bounds it:** D's pooled MAE is "
        f"{main_pooled[HEADLINE] - main_pooled[ORACLE]:,.1f} MW below C's"
        + (
            ", and C is no better than D in any lead-time range."
            if oracle_everywhere
            else ", but C beats D in some lead-time range (see the table above)."
        ),
    ]
    if delay is not None:
        cost = delay.pooled[delay.model] - delay.pooled[HEADLINE]
        weather_gain = main_pooled[WEATHER_FREE] - main_pooled[HEADLINE]
        small = abs(cost) < 0.1 * weather_gain
        evidence.insert(
            1,
            f"- **Publication timing:** assuming GFS runs are published {delay.hours:g} h after "
            f"their run time instead of {delay.headline_hours:g} h changes the pooled MAE by "
            f"{cost:+.1f} MW, against a C-vs-A gain of {weather_gain:,.1f} MW"
            + (
                ". The gain does not hinge on the last hours before publication, where a "
                "timing leak would sit."
                if small
                else ", a sizeable share of the gain: check the timing rule."
            ),
        )
    out += ["**Leakage evidence:**\n", "\n".join(evidence), ""]
    return out


def _failure_section(
    cfg: AppConfig,
    engine: Engine,
    region_id: str,
    pairs: Pairs,
    builder: FeatureBuilder,
) -> list[str]:
    ctx = builder.ctx
    assert builder.weather is not None and builder.weather.observed is not None
    lo, hi = S1_FIRST_STEP, builder.setup.horizon_steps
    daily = daily_errors(pairs, ctx.local_day, FAILURE_MODELS, lo, hi)
    full = daily[daily["n"] >= FULL_DAY_SHARE * daily["n"].median()]
    worst = full.sort_values(HEADLINE, ascending=False).head(cfg.evaluation.failure_days)
    ranks = full[HEADLINE].rank(ascending=False, method="min").astype(int)
    events = {e.day.toordinal(): e.label for e in cfg.evaluation.known_events}
    days = list(worst.index) + [d for d in events if d not in worst.index and d in daily.index]

    observed = builder.weather.observed.series["temp_weighted"]
    max_temp = pd.Series(observed).groupby(ctx.local_day).max()
    temp_error = _temperature_error(builder, pairs, days, lo, hi)
    with engine.connect() as conn:
        cal = load_calendar_days(conn, cfg.region(region_id).holiday_subdiv)
    holidays = {
        d.toordinal(): str(n)
        for d, n, h in zip(
            cal["local_date"], cal["holiday_name"], cal["is_public_holiday"], strict=True
        )
        if h
    }

    by_day = daily.to_dict(orient="index")

    def row(day: int) -> list[str]:
        r = by_day[day]
        date = dt.date.fromordinal(int(day))
        kind = holidays.get(day, date.strftime("%A"))
        rank = f"{ranks[day]}" if day in ranks.index else "–"
        return [
            rank,
            f"{date:%Y-%m-%d}",
            kind,
            *[_fmt(r[m]) for m in FAILURE_MODELS],
            f"{r['bias']:+,.0f}",
            _fmt(max_temp.get(day, np.nan)),
            _fmt(temp_error.get(day, np.nan)),
        ]

    headers = ["Rank", "Local date", "Day", *[f"MAE {model_label(m)}" for m in FAILURE_MODELS],
               "Bias C (MW)", "ERA5 max temp (°C)", "C temp forecast error (°C)"]  # fmt: skip
    out = [
        "## Failure cases\n",
        f"The {cfg.evaluation.failure_days} local days with the highest LightGBM C MAE at lead "
        f"times {horizon_label(lo)}–{horizon_label(hi)} (S1's range), among days with at least "
        f"{100 * FULL_DAY_SHARE:g}% of the usual number of scored pairs "
        f"({len(full):,} of {len(daily):,} days). "
        "Bias is forecast − actual. *C temp forecast error* is the mean absolute difference "
        "between C's forecast temperature and ERA5 on that day's pairs.\n",
        _md_table(headers, [row(d) for d in worst.index]),
        "",
    ]
    known = [d for d in events if d in daily.index]
    if known:
        out += [
            "**Known events** (rank among all complete days; listed whatever their rank):\n",
            _md_table(["Event", *headers], [[events[d], *row(d)] for d in known]),
            "",
        ]
    return out


def _temperature_error(
    builder: FeatureBuilder, pairs: Pairs, days: Sequence[int], lo: int, hi: int
) -> dict[int, float]:
    """Mean |C forecast temperature − ERA5| over each day's scored pairs (h lo…hi)."""
    ctx = builder.ctx
    horizons = np.arange(lo, hi + 1)
    targets = pairs.origins[:, None] + horizons[None, :]
    target_day = ctx.local_day[targets]
    out: dict[int, float] = {}
    for day in days:
        hit = (target_day == day) & pairs.mask[:, lo - 1 : hi]
        rows = np.flatnonzero(hit.any(axis=1))
        if rows.size == 0:
            continue
        origins = pairs.origins[rows]
        c = builder.build("C", origins, horizons)
        d = builder.build("D", origins, horizons)
        if not (np.array_equal(c.origins, d.origins) and np.array_equal(c.horizons, d.horizons)):
            raise ValueError("C and D feature rows differ")
        # Map each feature row back to its (issue time, lead) cell of ``hit``.
        row_of = {int(o): i for i, o in enumerate(origins)}
        r_idx = np.array([row_of[int(o)] for o in c.origins])
        keep = hit[rows][r_idx, c.horizons.astype(np.int64) - lo]
        diff = (c.X["wx_temp"].to_numpy() - d.X["wx_temp"].to_numpy())[keep]
        out[day] = float(np.nanmean(np.abs(diff)))
    return out


def build_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    leakage: LeakageOutcome | None = None,
) -> str:
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    data_root = cfg.data_root()
    sets = Loaded(
        load_evaluation_set(engine, region_id, "main"),
        _optional(engine, region_id, "plus"),
        _optional(engine, region_id, "delay"),
    )
    leakage = leakage or run_leakage_tests()
    main = sets.main
    table = main.by_horizon("dev", "mae")
    main_pooled = pooled(main)
    baselines = baselines_of(main)
    criteria = success_criteria(table, main_pooled, baselines, leakage)
    groups = group_means(table)
    rc = run_config(main, HEADLINE)
    delay = (
        DelayCheck.of(sets.delay, float(rc["publication_delay_hours"]))
        if sets.delay is not None
        else None
    )
    ablation = read_ablation(cfg, main.batch_id)
    last_short = int(rc["lead_bands"][0][1])
    issue_minutes = 5 * int(rc["setup"]["issue_every_steps"])
    es_weeks = int(rc["ml"]["early_stopping_weeks"])
    retrain_months = int(rc["ml"]["retrain_every_months"])
    plus_start = str(rc["ml"]["plus_train_start"])
    s1_label = horizon_label(S1_FIRST_STEP)

    figures = reports_dir / "figures"
    boundary = last_short / 12  # hours
    title = "Victoria demand: MAE by lead time (development period)"
    _line_chart(table, MAIN_PLOT, figures / FIGURE_LEAD, title, 24, boundary)
    title = "Short lead times: ML against persistence (development period)"
    _line_chart(table, SHORT_PLOT, figures / FIGURE_SHORT, title, 3.0, boundary)

    builder = load_feature_builder(engine, cfg, region_id)
    pairs = load_pairs(main, data_root, builder.ctx.times)
    if int(pairs.mask.sum()) != int(run_config(main, HEADLINE)["scored_pairs"]):
        raise ValueError("stored predictions do not reproduce the stored scored-pair count")

    horizons = sorted({*cfg.evaluation.horizons_reported, last_short, last_short + 1})
    cross = first_better(table, HEADLINE, "persistence")
    jump, typical = band_jump(table, HEADLINE, last_short)
    jump_a, _ = band_jump(table, WEATHER_FREE, last_short)
    passed = sum(c.passed for c in criteria)
    at_1 = {m: float(v) for m, v in table.iloc[0].items()}  # lead time 5 min

    c_records = retrain_records(main, HEADLINE)
    grid_models = {
        m: retrain_records(main, m) for m in main.models if m.startswith(("lgbm", "ridge"))
    }
    if sets.plus is not None:
        grid_models |= {
            m: retrain_records(sets.plus, m) for m in sets.plus.models if m.endswith("plus")
        }
    counts = chosen_counts(grid_models)
    imp = importance(c_records, data_root)
    yearly = yearly_errors(pairs, ["lgbm_A", "lgbm_C", "lgbm_D", "ridge_C", "daily_naive"])

    out: list[str] = [
        "# Milestone 2: weather-aware demand forecasts (development period)\n",
        "> **Generated** by `gridcast report-m2` on "
        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M UTC} from the stored walk-forward run. "
        "Do not edit by hand: re-run the pipeline instead. The holdout period is not scored "
        "here.\n",
        "## Provenance\n",
        _provenance(cfg, sets, leakage, region_id),
        "",
        "## Success criteria (pre-registered)\n",
        "Set in `docs/plans/m2_plan.md` §3 before any ML result existed. They apply to "
        "LightGBM, variant C (GFS temperature forecasts as available at the issue time), "
        "scored on exactly the same pairs as the rescored baselines. *Best baseline* means "
        f"the lowest MAE among {', '.join(model_label(b).lower() for b in baselines)} at each "
        "lead time. S5 (the holdout, scored once) belongs to step 6.\n",
        criteria_table(criteria),
        f"\n**{passed} of {len(criteria)} criteria pass.**\n",
        "## How to read the results\n",
        f"- Each issue time (every {issue_minutes} min) gets a 24-hour forecast in 5-minute "
        "steps; errors "
        "are grouped by **lead time** and pooled over every issue time.\n"
        "- **A** has no weather, **C** uses GFS temperature forecasts published before the "
        "issue time, **D** uses ERA5 reanalysis for the target time (perfect foresight, an "
        "upper bound for C's inputs). C and D have identical features, so C − D is the cost "
        "of weather-forecast error.\n"
        f"- Every ML model is two fits: one for leads up to {horizon_label(last_short)}, one "
        "beyond. Its settings (target form, objective, size or penalty) are re-selected at "
        f"every retrain (every {retrain_months} months) on the last {es_weeks} weeks before "
        "it.\n"
        "- **Bold** marks the lowest error in each column. Error = forecast − actual.\n",
        "## MAE by lead time\n",
        f"![MAE by lead time](figures/{FIGURE_LEAD})\n",
        "### MAE (MW), development period\n",
        horizon_table(main, "dev", horizons, "mae", [m for m in MAIN_TABLE if m in main.models]),
        "\n### RMSE (MW), development period\n",
        horizon_table(main, "dev", horizons, "rmse", [m for m in MAIN_TABLE if m in main.models]),
        "\n### Bias (MW, forecast − actual), development period\n",
        horizon_table(main, "dev", horizons, "bias", [m for m in MAIN_TABLE if m in main.models]),
        "",
        "## Short lead times and the band boundary\n",
        f"![Short lead times](figures/{FIGURE_SHORT})\n",
        f"- At {horizon_label(1)}, LightGBM C's MAE is {at_1[HEADLINE]:,.1f} MW "
        f"against {at_1['persistence']:,.1f} MW for persistence; "
        + (
            f"LightGBM C is better from {horizon_label(cross)} on."
            if cross is not None
            else "LightGBM C never beats persistence."
        )
        + f" Lead times under {s1_label} are reported, not pass/fail.",
        f"- Between {horizon_label(last_short)} (the short-lead fit's last step) and "
        f"{horizon_label(last_short + 1)} (the long-lead fit's first), LightGBM C's MAE "
        f"changes by {jump:+,.1f} MW (A: {jump_a:+,.1f} MW), against a median step of "
        f"{typical:,.1f} MW between neighbouring lead times elsewhere. The long-lead fit is "
        "weakest at its shortest leads.",
        "- Both are known weaknesses of this run. They are reported as they are; a fix would "
        "be a separate, disclosed development run (decision of 2026-11-06).\n",
        "## Weather value by lead time\n",
        "Mean of the per-lead MAE within each lead-time range (MW). The delay column comes "
        "from the delay batch, on its own common pairs.\n",
        _weather_value_table(table, delay),
        "",
        *_ablation_section(
            ablation,
            main_pooled,
            delay,
            groups,
            leakage,
            cfg.evaluation.ablation_flag_share,
            es_weeks,
        ),
    ]
    if sets.plus is not None:
        plus = sets.plus
        prc = run_config(plus, "lgbm_Cplus")
        plus_models: list[str] = [m for m in ("lgbm_Cplus", "lgbm_Dplus", "lgbm_C", "ridge_Cplus",
                                   "ridge_Dplus", "ridge_C", "persistence", "daily_naive",
                                   "weekly_naive") if m in plus.models]  # fmt: skip
        out += [
            "## C⁺ and D⁺: all forecast weather variables (reported, not pass/fail)\n",
            "C⁺ adds apparent temperature, humidity, cloud cover and solar radiation "
            f"forecasts, archived only from {plus_start}, so it trains from then and is scored "
            f"from {pd.Timestamp(prc['first_issue']).tz_convert(NEM_TZ):%Y-%m-%d} "
            f"({prc['n_origins']:,} issue times, {prc['scored_pairs']:,} pairs), next to "
            "headline C on the same pairs.\n",
            horizon_table(plus, "dev", horizons, "mae", plus_models),
            "",
        ]
    if sets.delay is not None and delay is not None:
        out += [
            "## Publication-delay sensitivity (reported, never the headline)\n",
            "C rebuilt as if every GFS run were published "
            f"{delay.hours:g} h after its run time "
            f"(headline: {delay.headline_hours:g} h), with "
            "headline C's chosen settings at every retrain, so only the input timing "
            "differs.\n",
            horizon_table(sets.delay, "dev", horizons, "mae"),
            "",
        ]
    out += [
        "## Errors by year\n",
        "Pooled MAE (MW) by NEM year of the issue time. Weather-model upgrades can change "
        "C's forecast error over time; C − D tracks that cost.\n",
        _md_table(
            ["Year", "Pairs", *[model_label(m) for m in yearly.columns if m != "n"], "C − D"],
            [
                [
                    str(label),
                    f"{int(r['n']):,}",
                    *[_fmt(r[m]) for m in yearly.columns if m != "n"],
                    f"{r[HEADLINE] - r[ORACLE]:+.1f}",
                ]
                for label, r in yearly.iterrows()
            ],
        ),
        "",
        "## Model selection at each retrain\n",
        "LightGBM C, per retrain and lead band: the winning setting and the best "
        "early-stopping MAE each target form reached (residual: correction to the same time "
        "yesterday; direct: demand itself; anchored: correction to the latest demand plus "
        "yesterday's change). This is the planned residual-vs-direct ablation.\n",
        selection_table(c_records),
        "\nWinning target form across every grid-selected fit (A, C, D, C⁺, D⁺):\n",
        _md_table(
            ["Family", "Lead band", *[str(c) for c in counts.columns]],
            [
                [str(fam), str(band), *[str(int(v)) for v in r]]
                for (fam, band), r in zip(counts.index, counts.to_numpy(), strict=True)
            ],
        ),
        "",
        "## Feature importance (LightGBM C)\n",
        "Built-in importance of the saved models, as a share of each model's total, "
        "averaged over the retrains. Gain depends on the target form, which changes "
        "between retrains, so read it as a ranking, not an effect size.\n",
    ]
    for band, frame in imp.items():
        out += [f"### Lead band {band_label(band)}\n", importance_table(frame), ""]
    seg_desc = {**describe_segments(cfg.evaluation), **describe_weather_segments(cfg.evaluation)}
    seg_models = [m for m in SEGMENT_MODELS if m in main.models]
    seg_rows = []
    for seg, text_ in seg_desc.items():
        values = [main.value(m, "dev", seg, 0, "mae") for m in seg_models]
        finite = [v for v in values if pd.notna(v)]
        best = min(finite) if finite else None
        seg_rows.append(
            [
                text_,
                f"{main.n('dev', seg, 0):,}",
                *[f"**{_fmt(v)}**" if v == best else _fmt(v) for v in values],
            ]
        )
    out += [
        "## MAE (MW) by segment, all lead times pooled\n",
        "Weather and market-state segments are defined by what actually happened "
        "(*ex-post*); they show where models fail, not a way to choose one.\n",
        _md_table(["Segment", "Pairs", *[model_label(m) for m in seg_models]], seg_rows),
        "",
        *_failure_section(cfg, engine, region_id, pairs, builder),
        "## Notes and limitations\n",
        "- S1–S4 judge LightGBM C only. Ridge, C⁺/D⁺, the delay check and lead times under "
        f"{s1_label} are reported, not pass/fail.\n"
        "- All of this is the development period, inside which the settings were selected "
        "(each retrain sees only data before it). The holdout is scored once, in step 6.\n"
        "- Headline C uses forecast temperature only: other forecast variables are archived "
        f"only from {plus_start} (see C⁺). Pairs without C weather (the GFS archive gap) are "
        "excluded for every model.\n"
        "- The drop/add test runs at a few retrains on their early-stopping slices, not as "
        "a full walk-forward per feature.\n"
        "- `TOTALDEMAND` excludes rooftop PV, so cloudy and sunny days shift midday demand "
        "in ways temperature alone cannot explain.\n",
    ]
    return "\n".join(out)


def write_report(
    engine: Engine, cfg: AppConfig, region_id: str, leakage: LeakageOutcome | None = None
) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m2_demand.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id, leakage))
    return path

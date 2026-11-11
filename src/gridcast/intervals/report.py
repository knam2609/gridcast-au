"""Generate ``reports/m3_intervals.md`` (+ figures) from the stored interval batches (M3 step 4).

Nothing in the report is typed by hand. Every number comes from the stored bounds (parquet
under the data root), the stored run records (``model_runs``), the market and weather data
in the database, or the leakage property tests, which this command runs (U4). The pooled
numbers it recomputes from the bounds must equal the metrics stored with each batch, or the
report stops. U1–U4 are computed exactly as pre-registered in m3_plan.md §3.

Methods (all on run 2's common mask, development issue times from ``intervals.first_issue``):

* ``cqr_lgbm_C``: CQR on quantile LightGBM C, the pre-registered headline (step 3);
* ``cqr_lgbm_C_rearranged``: the same bounds sorted per pair across levels, reported only
  (user decision 2026-11-12), computed here;
* ``conformal_lgbm_C`` and the conformal baselines around persistence, daily naive and
  weekly naive (step 1);
* ``qr_lgbm_C``: the raw (rearranged) quantile LightGBM C bounds (step 2).
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sqlalchemy import Engine, text

from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.m2_report import (
    LEAD_GROUPS,
    Criterion,
    LeakageOutcome,
    criteria_table,
    lead_group_label,
    parse_pytest_summary,
)
from gridcast.evaluation.report import (
    AXIS,
    GRID,
    INK,
    INK_2,
    MODEL_COLORS,
    MUTED,
    SURFACE,
    _fmt,
    _md_table,
    horizon_label,
    horizon_ranges,
)
from gridcast.features.inputs import load_context, load_weather_inputs
from gridcast.features.weather import to_ns
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import (
    EVALUATION_SET,
    TARGET,
    interval_config_hash,
    selection_retrains,
)
from gridcast.intervals.evaluation import (
    BOOTSTRAP_DRAWS,
    U2_RANGE,
    U3_FIRST_STEP,
    Aggregator,
    Estimate,
    StatSums,
    bootstrap_ratio,
    forecast_day_max,
    issue_day_codes,
    pair_stats,
    per_lead,
    pooled,
    rearrange_bounds,
    success_criteria,
    week_blocks,
)
from gridcast.intervals.inputs import read_stored_batch, read_wide
from gridcast.intervals.quantile import interval_quantiles, quantile_key, quantile_levels
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

HEADLINE = "cqr_lgbm_C"
REARRANGED = "cqr_lgbm_C_rearranged"
CONFORMAL_C = "conformal_lgbm_C"
RAW_QR = "qr_lgbm_C"
BASELINES = ("conformal_persistence", "conformal_daily_naive", "conformal_weekly_naive")
METHODS = (HEADLINE, REARRANGED, CONFORMAL_C, RAW_QR, *BASELINES)
POINT_OF = {
    HEADLINE: "lgbm_C",
    REARRANGED: "lgbm_C",
    CONFORMAL_C: "lgbm_C",
    RAW_QR: "lgbm_C",
    "conformal_persistence": "persistence",
    "conformal_daily_naive": "daily_naive",
    "conformal_weekly_naive": "weekly_naive",
}
LABELS = {
    HEADLINE: "CQR, LightGBM C (headline)",
    REARRANGED: "CQR, rearranged (reported only)",
    CONFORMAL_C: "Conformal around LightGBM C",
    RAW_QR: "Quantile LightGBM C, raw",
    "conformal_persistence": "Conformal around persistence",
    "conformal_daily_naive": "Conformal around daily naive",
    "conformal_weekly_naive": "Conformal around weekly naive",
}
SHORT = {
    HEADLINE: "CQR",
    REARRANGED: "CQR sorted",
    CONFORMAL_C: "Conformal C",
    RAW_QR: "Raw QR",
    "conformal_persistence": "Persistence",
    "conformal_daily_naive": "Daily naive",
    "conformal_weekly_naive": "Weekly naive",
}
# Validated (dataviz validator, light surface): the baselines keep their M1/M2 hues; CQR takes
# LightGBM C's green, conformal C violet, raw QR magenta, the rearranged row yellow.
COLORS = {
    HEADLINE: MODEL_COLORS["lgbm_C"],
    REARRANGED: "#eda100",
    CONFORMAL_C: "#6250d6",
    RAW_QR: "#e87ba4",
    "conformal_persistence": MODEL_COLORS["persistence"],
    "conformal_daily_naive": MODEL_COLORS["daily_naive"],
    "conformal_weekly_naive": MODEL_COLORS["weekly_naive"],
}
BAND_FILL = "#f0efec"
LEAKAGE_TESTS = (
    "tests/unit/test_interval_leakage.py",
    "tests/unit/test_quantile.py::test_quantiles_ignore_demand_after_their_issue_time",
)
WEEK_DAYS = 7
MIN_MONTH_DAYS = 7  # months with fewer target days are tabled but left out of the figure
MIN_BOOTSTRAP_DAYS = 10  # groups whose pairs come from fewer issue days get no interval
# Per-pair statistics also summed by label and group (the conditional tables).
CONDITIONAL = re.compile(r"^(cov|below|above|width|is)_\d+$")
# Stored metric of each recomputed statistic (horizon 0), checked before anything is written.
STORED_NAME = {
    "cov": "coverage",
    "below": "below",
    "above": "above",
    "width": "width",
    "is": "interval_score",
    "le": "share_le",
    "pin": "pinball",
}


# --------------------------------------------------------------------------- runs


@dataclass(frozen=True, slots=True)
class IntervalRun:
    run_id: int
    batch_id: str
    name: str
    git_sha: str | None
    directory: Path
    config: dict[str, Any]


def _interval_runs(engine: Engine, region_id: str) -> list[IntervalRun]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, batch_id, model_name, git_sha, artifact_path, config
                FROM model_runs
                WHERE model_kind = 'interval' AND target = :target AND region_id = :region
                  AND status = 'success' AND config->>'evaluation_set' = :set
                ORDER BY run_id
                """
            ),
            {"target": TARGET, "region": region_id, "set": EVALUATION_SET},
        ).all()
    return [
        IntervalRun(
            int(r.run_id),
            str(r.batch_id),
            r.model_name,
            r.git_sha,
            Path(r.artifact_path),
            dict(r.config),
        )
        for r in rows
    ]


def find_runs(engine: Engine, cfg: AppConfig, region_id: str) -> dict[str, IntervalRun]:
    """The stored runs the report scores, keyed by method, checked against the current recipe.

    The latest CQR run built on ``intervals.qr_batch``, that QR run, and the latest
    conformal runs, which must come from one batch. Every run must carry the current
    interval config hash and ``intervals.source_batch``.
    """
    current = interval_config_hash(cfg)
    runs = _interval_runs(engine, region_id)
    latest: dict[str, IntervalRun] = {}
    for run in runs:
        if run.name == RAW_QR and run.batch_id != cfg.intervals.qr_batch:
            continue
        if run.name == HEADLINE and run.config.get("qr_batch") != cfg.intervals.qr_batch:
            continue
        latest[run.name] = run
    wanted = [HEADLINE, CONFORMAL_C, RAW_QR, *BASELINES]
    if missing := [m for m in wanted if m not in latest]:
        raise LookupError(f"no stored development interval run for {missing}")
    out = {m: latest[m] for m in wanted}
    conformal = {out[m].batch_id for m in (CONFORMAL_C, *BASELINES)}
    if len(conformal) != 1:
        raise ValueError(f"the latest conformal runs come from several batches: {conformal}")
    for m, run in out.items():
        if run.config.get("interval_config_hash") != current:
            raise ValueError(
                f"{m} (run {run.run_id}) has interval config hash "
                f"{run.config.get('interval_config_hash')}, the current recipe {current}"
            )
        if run.config.get("source_batch") != cfg.intervals.source_batch:
            raise ValueError(f"{m} (run {run.run_id}) was built on another source batch")
    return out


def stored_metrics(engine: Engine, run_id: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT metric, horizon, value, n FROM forecast_metrics
                WHERE run_id = :r AND split = 'dev' AND segment = 'all'
                """
            ),
            conn,
            params={"r": run_id},
        )


# --------------------------------------------------------------------------- leakage


def run_leakage_tests() -> LeakageOutcome:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *LEAKAGE_TESTS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed, failed = parse_pytest_summary(proc.stdout)
    return LeakageOutcome(passed, failed, proc.returncode, LEAKAGE_TESTS)


# --------------------------------------------------------------------------- pairs


@dataclass(slots=True)
class Pairs:
    """The scored pairs, their labels and groups."""

    times: pd.DatetimeIndex
    actual: np.ndarray
    mask: np.ndarray
    points: dict[str, np.ndarray]
    day_code: np.ndarray
    day_ordinals: np.ndarray  # per day code: NEM date ordinal of issue
    months: list[tuple[int, int]]  # label → (year, month) of the target's local date
    month_days: list[int]  # distinct target days per month label
    labels: dict[str, tuple[np.ndarray, int]]
    groups: dict[str, np.ndarray]
    group_text: dict[str, str]
    group_days: dict[str, int]
    known_groups: list[str]
    outcome_groups: list[str]
    top_demand_mw: float


def _month_labels(target_dates: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Per-pair month label from target local date ordinals, and the (year, month) of each."""
    unique = np.unique(target_dates)
    ym = np.array([(d.year * 12 + d.month - 1) for d in map(dt.date.fromordinal, unique)])
    first = int(ym.min())
    code_of_day = ym - first
    lookup = np.searchsorted(unique, target_dates)
    labels = code_of_day[lookup].astype(np.int16)
    months = [divmod(first + k, 12) for k in range(int(ym.max()) - first + 1)]
    return labels, [(y, m + 1) for y, m in months]


def load_pairs(engine: Engine, cfg: AppConfig, region_id: str, times: pd.DatetimeIndex) -> Pairs:
    """Run 2's actuals, mask and point forecasts on ``times`` plus every label and group."""
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    source = read_stored_batch(
        cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch, TARGET
    ).restrict(end=end)
    rows = np.asarray(source.origin_times >= times[0])
    if not source.origin_times[rows].equals(times):
        raise ValueError("the interval batches' issue times differ from run 2's")
    actual, mask = source.actual[rows], source.common_mask()[rows]
    points = {m: source.forecasts[m][rows] for m in sorted(set(POINT_OF.values()))}
    n_leads = actual.shape[1]

    nem_dates = times.tz_convert(NEM_TZ).date
    day_code, day_ordinals = issue_day_codes(np.array([d.toordinal() for d in nem_dates]))

    ctx = load_context(engine, cfg, region_id)
    opos = ctx.times.get_indexer(times)
    if np.any(opos < 0):
        raise ValueError("an issue time is not on the market grid")
    tpos = opos[:, None] + np.arange(1, n_leads + 1)[None, :]
    if not np.allclose(ctx.y[tpos][mask], actual[mask]):
        raise ValueError("the stored actuals differ from the market data")
    target_day = ctx.local_day[tpos]
    month_label, months = _month_labels(target_day)
    month_days = [
        int(np.unique(target_day[mask & (month_label == k)]).size) for k in range(len(months))
    ]
    hour = ctx.frame["local_hour"].to_numpy(dtype=np.int16)[tpos]

    # Groups known at the issue time: GFS forecast daily maximum, holidays, DST changes.
    lo = int(np.searchsorted(ctx.local_day, target_day.min(), "left"))
    hi = int(np.searchsorted(ctx.local_day, target_day.max(), "right"))
    weather = load_weather_inputs(engine, cfg, region_id, ctx.times[lo:hi])
    forecasts = []
    for k in sorted(weather.forecasts):  # freshest first, as C's features
        fc = weather.forecasts[k]
        if fc.available_at is None:
            raise ValueError("forecast weather has no publication times")
        forecasts.append((fc.series["temp_weighted"], fc.available_at["temp_weighted"]))
    day_max = forecast_day_max(to_ns(times), tpos - lo, ctx.local_day[lo:hi], forecasts)
    if np.any(mask & ~np.isfinite(day_max)):
        raise ValueError("a scored pair has no forecast daily maximum known at its issue time")
    hot, cold = cfg.intervals.hot_day_max_temp_c, cfg.intervals.cold_day_max_temp_c
    frame = ctx.frame
    groups = {
        "hot": day_max >= hot,
        "cold": day_max <= cold,
        "public_holiday": frame["is_public_holiday"].to_numpy(dtype=bool)[tpos],
        "dst_change_day": frame["is_dst_change_day"].to_numpy(dtype=bool)[tpos],
    }
    text_of = {
        "hot": f"Forecast daily maximum ≥ {hot:g} °C",
        "cold": f"Forecast daily maximum ≤ {cold:g} °C",
        "public_holiday": "Victorian public holidays",
        "dst_change_day": "Days when clocks change (23 h / 25 h)",
    }
    known = list(groups)

    # Groups defined by the outcome.
    targets = np.unique(tpos[mask])
    top = float(np.quantile(ctx.y[targets], 0.99))
    price = frame["rrp_aud_mwh"].to_numpy(dtype=np.float64, na_value=np.nan)
    spike = cfg.evaluation.price_spike_threshold
    with np.errstate(invalid="ignore"):
        groups["top_demand"] = ctx.y[tpos] >= top
        groups["price_spike"] = price[tpos] > spike
    text_of["top_demand"] = f"Top 1% of actual demand intervals (≥ {top:,.0f} MW)"
    text_of["price_spike"] = f"Realised price > ${spike:,.0f}/MWh"
    for event in cfg.evaluation.known_events:
        key = f"event_{event.day}"
        groups[key] = target_day == pd.Timestamp(event.day).date().toordinal()
        text_of[key] = f"{event.day}: {event.label}"
    outcome = [g for g in groups if g not in known]
    group_days = {g: int(np.unique(target_day[mask & m]).size) for g, m in groups.items()}
    return Pairs(
        times,
        actual,
        mask,
        points,
        day_code,
        day_ordinals,
        months,
        month_days,
        {"hour": (hour, 24), "month": (month_label, len(months))},
        groups,
        text_of,
        group_days,
        known,
        outcome,
        top,
    )


# --------------------------------------------------------------------------- bounds


def _wide_rows(path: Path, times: pd.DatetimeIndex) -> np.ndarray:
    t, values = read_wide(path)
    keep = np.asarray(t >= times[0])
    if not t[keep].equals(times):
        raise ValueError(f"{path.name}: issue times differ from the other batches")
    return values[keep]


def load_bounds(
    run: IntervalRun, method: str, levels: Sequence[int], data_root: Path, times: pd.DatetimeIndex
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Stored bounds of one method on ``times`` (raw QR: its rearranged quantiles)."""
    directory = data_root / run.directory
    lower, upper = {}, {}
    for level in levels:
        if method == RAW_QR:
            lo_key, hi_key = interval_quantiles(level)
            lo_path = directory / f"{RAW_QR}__{lo_key}.parquet"
            hi_path = directory / f"{RAW_QR}__{hi_key}.parquet"
        else:
            lo_path = directory / f"{method}__lo{level}.parquet"
            hi_path = directory / f"{method}__hi{level}.parquet"
        lower[level] = _wide_rows(lo_path, times)
        upper[level] = _wide_rows(hi_path, times)
    return lower, upper


def score_method(
    agg: Aggregator,
    lower: Mapping[int, np.ndarray],
    upper: Mapping[int, np.ndarray],
    pairs: Pairs,
    point: np.ndarray,
) -> dict[str, StatSums]:
    if any(np.any(pairs.mask & ~np.isfinite(b)) for b in (*lower.values(), *upper.values())):
        raise ValueError("a scored pair has a missing bound")
    return {
        name: agg.add(values, conditional=bool(CONDITIONAL.match(name)))
        for name, values in pair_stats(lower, upper, pairs.actual, point)
    }


def check_against_stored(
    method: str, stats: Mapping[str, StatSums], counts: StatSums, stored: pd.DataFrame
) -> int:
    """Every pooled statistic with a stored counterpart must match it; returns how many."""
    values = stored.loc[stored["horizon"] == 0].set_index("metric")["value"]
    checked = 0
    for name, sums in stats.items():
        prefix, _, rest = name.partition("_")
        if name in ("wis", "non_nested"):
            metric = name
        elif prefix in STORED_NAME and rest:
            metric = f"{STORED_NAME[prefix]}_{rest}"
        else:
            continue
        if metric not in values.index:
            continue
        mine, theirs = pooled(sums, counts), float(values[metric])
        if not np.isclose(mine, theirs, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{method}: recomputed {metric} {mine!r} ≠ stored {theirs!r}")
        checked += 1
    if checked == 0:
        raise ValueError(f"{method}: no stored metric to check against")
    return checked


# --------------------------------------------------------------------------- results


@dataclass(slots=True)
class Results:
    pairs: Pairs
    counts: StatSums
    stats: dict[str, dict[str, StatSums]]
    levels: list[int]
    seed: int
    checked: dict[str, int]

    def pooled(self, method: str, stat: str) -> float:
        return pooled(self.stats[method][stat], self.counts)

    def per_lead(self, method: str, stat: str) -> np.ndarray:
        return per_lead(self.stats[method][stat], self.counts)

    def estimate(self, method: str, stat: str, blocks: np.ndarray | None = None) -> Estimate:
        return bootstrap_ratio(
            self.stats[method][stat].by_day, self.counts.by_day, self.seed, blocks=blocks
        )

    def difference(self, a: str, b: str, stat: str, blocks: np.ndarray | None = None) -> Estimate:
        sums = self.stats[a][stat].by_day - self.stats[b][stat].by_day
        return bootstrap_ratio(sums, self.counts.by_day, self.seed, blocks=blocks)

    @property
    def weeks(self) -> np.ndarray:
        return week_blocks(self.pairs.day_ordinals, WEEK_DAYS)

    def best_baseline(self, level: int = 80) -> str:
        return min(BASELINES, key=lambda m: self.pooled(m, f"is_{level}"))

    def lead_group(self, method: str, stat: str, lo: int, hi: int) -> float:
        s = self.stats[method][stat].by_lead[:, lo - 1 : hi].sum()
        return float(s / self.counts.by_lead[:, lo - 1 : hi].sum())

    def by_label(self, method: str, stat: str, label: str) -> np.ndarray:
        s = self.stats[method][stat].by_label[label].sum(axis=0)
        n = self.counts.by_label[label].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(s / n)

    def group(self, method: str, stat: str, group: str) -> Estimate:
        return bootstrap_ratio(
            self.stats[method][stat].by_group[group],
            self.counts.by_group[group],
            self.seed,
        )


def score_all(
    engine: Engine, cfg: AppConfig, region_id: str, runs: Mapping[str, IntervalRun]
) -> Results:
    levels = sorted(cfg.intervals.levels)
    data_root = cfg.data_root()
    first = data_root / runs[HEADLINE].directory / f"{HEADLINE}__lo{levels[0]}.parquet"
    times, _ = read_wide(first)
    pairs = load_pairs(engine, cfg, region_id, times)
    agg = Aggregator(pairs.mask, pairs.day_code, pairs.labels, pairs.groups)
    stats: dict[str, dict[str, StatSums]] = {}
    checked: dict[str, int] = {}
    for method in (HEADLINE, CONFORMAL_C, RAW_QR, *BASELINES):
        run = runs[method]
        lower, upper = load_bounds(run, method, levels, data_root, times)
        point = pairs.points[POINT_OF[method]]
        stats[method] = score_method(agg, lower, upper, pairs, point)
        checked[method] = check_against_stored(
            method, stats[method], agg.counts, stored_metrics(engine, run.run_id)
        )
        if method == HEADLINE:
            lo_r, hi_r = rearrange_bounds(lower, upper)
            stats[REARRANGED] = score_method(agg, lo_r, hi_r, pairs, point)
            del lo_r, hi_r
        del lower, upper
    return Results(pairs, agg.counts, stats, levels, cfg.ml.seed, checked)


# --------------------------------------------------------------------------- formatting


def pct(v: float, digits: int = 1) -> str:
    return _fmt(100 * v, digits)


def ci(e: Estimate, scale: float = 1.0, digits: int = 1) -> str:
    if e.n_pairs == 0:
        return "–"
    return (
        f"{_fmt(scale * e.value, digits)} ({_fmt(scale * e.lo, digits)} to "
        f"{_fmt(scale * e.hi, digits)})"
    )


def signed(v: float, digits: int = 1) -> str:
    return f"{'−' if v < 0 else '+'}{abs(v):,.{digits}f}"


def ci_signed(e: Estimate, digits: int = 1) -> str:
    return f"{signed(e.value, digits)} ({signed(e.lo, digits)} to {signed(e.hi, digits)})"


def _nem_date(value: str) -> str:
    return f"{pd.Timestamp(value).tz_convert(NEM_TZ):%Y-%m-%d}"


def month_name(year: int, month: int) -> str:
    return dt.date(year, month, 1).strftime("%b %Y")


# --------------------------------------------------------------------------- figures


def _style(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=30)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)


def _legend(ax: Axes, n: int) -> None:
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=8.5,
        ncol=min(n, 6),
        handlelength=1.6,
    )
    for t in legend.get_texts():
        t.set_color(INK_2)


def _end_labels(
    ax: Axes, x: float, ends: Sequence[tuple[float, str]], span: tuple[float, float]
) -> None:
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    lo, hi = span
    gap = (hi - lo) * 0.055
    placed: list[float] = []
    x_text = x + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
    for y, name in sorted(ends):
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(x_text, y_label),
            color=INK_2,
            fontsize=8.5,
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6},
        )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def _hours(n_leads: int) -> np.ndarray:
    return np.arange(1, n_leads + 1) * 5 / 60


def _new_figure(right: float = 0.78) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.08, right=right, top=0.82, bottom=0.12)
    return fig, ax


def plot_coverage_by_lead(res: Results, path: Path) -> None:
    order = [*BASELINES, RAW_QR, CONFORMAL_C, HEADLINE]  # headline drawn last, on top
    series = {m: 100 * res.per_lead(m, "cov_80") for m in order}
    hours = _hours(len(series[HEADLINE]))
    fig, ax = _new_figure()
    lo2, hi2 = U2_RANGE
    ax.axhspan(lo2, hi2, color=BAND_FILL, linewidth=0, zorder=0)
    ax.axhline(80, color=AXIS, linewidth=1, zorder=1)
    for m in order:
        ax.plot(
            hours,
            series[m],
            color=COLORS[m],
            linewidth=2,
            label=SHORT[m],
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
    low = min(float(np.nanmin(v)) for v in series.values())
    high = max(float(np.nanmax(v)) for v in series.values())
    span = (max(0.0, np.floor(low / 5) * 5 - 2), min(100.0, max(hi2, high) + 3))
    ax.set_ylim(*span)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.annotate(
        f"U2 range {lo2:g}–{hi2:g}%",
        xy=(0.3, hi2),
        xytext=(0, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    _end_labels(ax, 24, [(float(series[m][-1]), SHORT[m]) for m in order], span)
    _style(
        ax,
        "Forecast lead time (hours after issue time)",
        "Share of actuals inside (%)",
        "80% intervals: coverage by lead time (development, Aug 2023 → Sep 2025)",
    )
    _legend(ax, len(order))
    _save(fig, path)


def plot_is_vs_envelope(res: Results, path: Path) -> None:
    envelope = np.min(np.stack([res.per_lead(m, "is_80") for m in BASELINES]), axis=0)
    series = {
        CONFORMAL_C: res.per_lead(CONFORMAL_C, "is_80"),
        HEADLINE: res.per_lead(HEADLINE, "is_80"),
    }
    hours = _hours(len(envelope))
    fig, ax = _new_figure()
    ax.plot(
        hours,
        envelope,
        color=INK_2,
        linewidth=2,
        linestyle=(0, (4, 2)),
        label="Best baseline at each lead",
    )
    for m, v in series.items():
        ax.plot(
            hours,
            v,
            color=COLORS[m],
            linewidth=2,
            label=SHORT[m],
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    top = float(max(envelope.max(), *(v.max() for v in series.values()))) * 1.08
    ax.set_ylim(0, top)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    start = U3_FIRST_STEP * 5 / 60
    ax.axvline(start, color=AXIS, linewidth=1)
    ax.annotate(
        f"U3 judged from {horizon_label(U3_FIRST_STEP)}",
        xy=(start, top * 0.97),
        xytext=(4, 0),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
        va="top",
    )
    ends = [(float(envelope[-1]), "Best baseline")]
    ends += [(float(v[-1]), SHORT[m]) for m, v in series.items()]
    _end_labels(ax, 24, ends, (0, top))
    _style(
        ax,
        "Forecast lead time (hours after issue time)",
        "Mean interval score IS₈₀ (MW)",
        "80% intervals: interval score by lead time, CQR against the baseline envelope",
    )
    _legend(ax, 3)
    _save(fig, path)


def plot_reliability(res: Results, path: Path) -> None:
    taus = quantile_levels(res.levels)
    order = [RAW_QR, CONFORMAL_C, REARRANGED, HEADLINE]
    series = {
        m: np.array([100 * (res.pooled(m, f"le_{quantile_key(t)}") - t) for t in taus])
        for m in order
    }
    fig, ax = _new_figure()
    ax.axhline(0, color=AXIS, linewidth=1)
    for m in order:
        ax.plot(
            taus,
            series[m],
            color=COLORS[m],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=SHORT[m],
        )
    low = min(float(v.min()) for v in series.values())
    high = max(float(v.max()) for v in series.values())
    pad = max(1.0, (high - low) * 0.1)
    span = (min(low, 0.0) - pad, max(high, 0.0) + pad)
    ax.set_ylim(*span)
    ax.set_xlim(0, 1)
    ax.set_xticks(list(taus))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    _end_labels(ax, float(taus[-1]), [(float(series[m][-1]), SHORT[m]) for m in order], span)
    _style(
        ax,
        "Nominal quantile level τ of the bound",
        "Observed share at or below, minus τ (pp)",
        "Reliability: share of actuals at or below each bound, minus its nominal level",
    )
    _legend(ax, len(order))
    _save(fig, path)


def plot_coverage_by_month(res: Results, path: Path) -> list[str]:
    """80% coverage by target month; returns the months left out (too few target days)."""
    keep = [k for k, d in enumerate(res.pairs.month_days) if d >= MIN_MONTH_DAYS]
    left_out = [
        month_name(*res.pairs.months[k]) for k in range(len(res.pairs.months)) if k not in keep
    ]
    order = [RAW_QR, CONFORMAL_C, HEADLINE]
    x = np.arange(len(keep))
    series = {m: 100 * res.by_label(m, "cov_80", "month")[keep] for m in order}
    fig, ax = _new_figure()
    ax.axhline(80, color=AXIS, linewidth=1)
    for m in order:
        ax.plot(
            x,
            series[m],
            color=COLORS[m],
            linewidth=2,
            marker="o",
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label=SHORT[m],
        )
    low = min(float(np.nanmin(v)) for v in series.values())
    high = max(float(np.nanmax(v)) for v in series.values())
    span = (np.floor(low / 5) * 5 - 2, min(100.0, high + 3))
    ax.set_ylim(*span)
    ax.set_xlim(-0.5, len(keep) - 0.5)
    ticks = [i for i in range(len(keep)) if res.pairs.months[keep[i]][1] in (1, 4, 7, 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([month_name(*res.pairs.months[keep[i]]) for i in ticks])
    _end_labels(ax, float(x[-1]), [(float(series[m][-1]), SHORT[m]) for m in order], span)
    _style(
        ax,
        "Month of the target (local time)",
        "Share of actuals inside (%)",
        "80% intervals: coverage by month of the target",
    )
    _legend(ax, len(order))
    _save(fig, path)
    return left_out


# --------------------------------------------------------------------------- sections


def _provenance(
    cfg: AppConfig,
    runs: Mapping[str, IntervalRun],
    res: Results,
    leakage: LeakageOutcome,
    region_id: str,
    report_sha: str | None,
) -> str:
    cqr = runs[HEADLINE].config
    first = res.pairs.times[0].tz_convert(NEM_TZ)
    last = res.pairs.times[-1].tz_convert(NEM_TZ)
    scored = int(res.pairs.mask.sum())
    total = int(res.pairs.mask.size)
    batch_rows = [
        f"conformal `{runs[CONFORMAL_C].batch_id}` (runs "
        + ", ".join(str(runs[m].run_id) for m in (CONFORMAL_C, *BASELINES))
        + f"; commit `{(runs[CONFORMAL_C].git_sha or '')[:7]}`)",
        f"quantile LightGBM C `{runs[RAW_QR].batch_id}` (run {runs[RAW_QR].run_id}; commit "
        f"`{(runs[RAW_QR].git_sha or '')[:7]}`)",
        f"CQR `{runs[HEADLINE].batch_id}` (run {runs[HEADLINE].run_id}; commit "
        f"`{(runs[HEADLINE].git_sha or '')[:7]}`)",
    ]
    rows = [
        ["Region", region_id],
        ["Interval batches", "; ".join(batch_rows)],
        ["Point forecasts (M2 run 2)", f"`{cfg.intervals.source_batch}`"],
        ["Interval config hash", f"`{cqr['interval_config_hash']}` (matches the current config)"],
        ["QR config hash", f"`{cqr['qr_config_hash']}`"],
        [
            "Scored issue times (NEM time)",
            f"{first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}, {len(res.pairs.times):,} "
            "(every 30 min)",
        ],
        [
            "Scored (issue time, lead) pairs, every method",
            f"{scored:,} of {total:,} (run 2's common mask)",
        ],
        [
            "Selection retrains",
            f"{len(cqr['retrains'])}, quarterly: {_nem_date(cqr['retrains'][0])} → "
            f"{_nem_date(cqr['retrains'][-1])}",
        ],
        [
            "Pooled metrics checked against the stored ones",
            ", ".join(f"{SHORT[m]} {n}" for m, n in res.checked.items()) + " (all equal)",
        ],
        [
            "Bootstrap",
            f"{BOOTSTRAP_DRAWS:,} draws, whole NEM issue days ({len(res.pairs.day_ordinals)} "
            f"days), 95% percentile intervals, seed {res.seed}; sensitivity: "
            f"{WEEK_DAYS}-day blocks ({int(res.weeks.max()) + 1} blocks)",
        ],
        [
            "Leakage property tests (run by this command)",
            f"{leakage.passed} passed, {leakage.failed} failed",
        ],
        ["Report generated from commit", f"`{report_sha}`"],
    ]
    return _md_table(["Item", "Value"], rows, "ll")


def pooled_table(res: Results) -> str:
    rows = []
    for m in METHODS:
        rows.append(
            [LABELS[m]]
            + [pct(res.pooled(m, f"cov_{lv}")) for lv in res.levels]
            + [
                _fmt(res.pooled(m, "width_80"), 0),
                _fmt(res.pooled(m, "is_80")),
                _fmt(res.pooled(m, "wis")),
            ]
        )
    headers = [
        "Method",
        *[f"Coverage at {lv}% (%)" for lv in res.levels],
        "Width 80% (MW)",
        "IS₈₀ (MW)",
        "WIS (MW)",
    ]
    return _md_table(headers, rows)


def level_table(res: Results) -> str:
    rows = []
    for m in METHODS:
        for lv in res.levels:
            rows.append(
                [
                    LABELS[m] if lv == res.levels[0] else "",
                    f"{lv}%",
                    ci(res.estimate(m, f"cov_{lv}"), 100),
                    pct(res.pooled(m, f"below_{lv}")),
                    pct(res.pooled(m, f"above_{lv}")),
                    _fmt(res.pooled(m, f"width_{lv}"), 0),
                    _fmt(res.pooled(m, f"is_{lv}")),
                ]
            )
    headers = [
        "Method",
        "Level",
        "Coverage % (95% CI)",
        "Below %",
        "Above %",
        "Width (MW)",
        "Interval score (MW)",
    ]
    return _md_table(headers, rows, "llrrrrr")


def headline_sensitivity(res: Results) -> tuple[str, list[tuple[str, str, Estimate, Estimate]]]:
    """CQR's coverage, then its score differences, with day and 7-day block intervals."""
    weeks = res.weeks
    rows = []
    for lv in res.levels:
        day, week = res.estimate(HEADLINE, f"cov_{lv}"), res.estimate(HEADLINE, f"cov_{lv}", weeks)
        rows.append(
            [
                f"CQR coverage at {lv}% (%)",
                pct(day.value, 2),
                f"{pct(day.lo, 2)} to {pct(day.hi, 2)}",
                f"{pct(week.lo, 2)} to {pct(week.hi, 2)}",
            ]
        )
    best = res.best_baseline()
    diffs: list[tuple[str, str, Estimate, Estimate]] = []
    for other in (CONFORMAL_C, best):
        for stat, name in (("is_80", "IS₈₀"), ("wis", "WIS")):
            d = res.difference(HEADLINE, other, stat)
            w = res.difference(HEADLINE, other, stat, weeks)
            diffs.append((other, name, d, w))
            rows.append(
                [
                    f"CQR − {SHORT[other]}: {name} (MW)",
                    signed(d.value),
                    f"{signed(d.lo)} to {signed(d.hi)}",
                    f"{signed(w.lo)} to {signed(w.hi)}",
                ]
            )
    headers = ["Quantity", "Estimate", "95% CI, day blocks", f"95% CI, {WEEK_DAYS}-day blocks"]
    return _md_table(headers, rows, "lrrr"), diffs


def other_differences(res: Results) -> str:
    rows = []
    pairs = [
        (HEADLINE, RAW_QR, "is_80", "IS₈₀"),
        (HEADLINE, RAW_QR, "wis", "WIS"),
        (REARRANGED, HEADLINE, "wis", "WIS"),
        (REARRANGED, HEADLINE, "is_80", "IS₈₀"),
        (CONFORMAL_C, res.best_baseline(), "is_80", "IS₈₀"),
    ]
    for a, b, stat, name in pairs:
        d = res.difference(a, b, stat)
        rows.append([f"{SHORT[a]} − {SHORT[b]}", name, ci_signed(d)])
    return _md_table(["Comparison", "Score", "Difference, MW (95% CI, day blocks)"], rows, "llr")


def lead_group_tables(res: Results) -> tuple[str, str]:
    groups = list(LEAD_GROUPS)
    cov_rows, is_rows = [], []
    for lo, hi in groups:
        label = lead_group_label(lo, hi)
        cov_rows.append([label] + [pct(res.lead_group(m, "cov_80", lo, hi)) for m in METHODS])
        is_rows.append([label] + [_fmt(res.lead_group(m, "is_80", lo, hi)) for m in METHODS])
    headers = ["Lead time", *[SHORT[m] for m in METHODS]]
    return _md_table(headers, cov_rows), _md_table(headers, is_rows)


def envelope_facts(res: Results) -> list[str]:
    stacked = np.stack([res.per_lead(m, "is_80") for m in BASELINES])
    best = stacked.argmin(axis=0)
    out = []
    for i, m in enumerate(BASELINES):
        leads = [h + 1 for h in np.flatnonzero(best == i)]
        if leads:
            out.append(f"{SHORT[m]} at {horizon_ranges([int(h) for h in leads])}")
    return out


def reliability_tables(res: Results) -> tuple[str, str]:
    taus = quantile_levels(res.levels)
    methods = [HEADLINE, REARRANGED, CONFORMAL_C, RAW_QR, *BASELINES]
    share_rows, pin_rows = [], []
    for m in methods:
        share_rows.append([LABELS[m]] + [pct(res.pooled(m, f"le_{quantile_key(t)}")) for t in taus])
        pins = [res.pooled(m, f"pin_{quantile_key(t)}") for t in taus]
        pin_rows.append([LABELS[m]] + [_fmt(p, 2) for p in pins] + [_fmt(float(np.mean(pins)), 2)])
    headers = ["Method", *[f"τ = {t:g}" for t in taus]]
    return (
        _md_table(headers, share_rows),
        _md_table([*headers, "Mean"], pin_rows),
    )


def nesting_table(res: Results) -> str:
    lv = res.levels
    pairs = list(pairwise(lv))
    rows = []
    for m in (HEADLINE, REARRANGED, CONFORMAL_C, RAW_QR, *BASELINES):
        cells = [pct(res.pooled(m, "non_nested"), 2)]
        for narrow, wider in pairs:
            cells += [
                pct(res.pooled(m, f"nest_lo_{narrow}_{wider}"), 2),
                pct(res.pooled(m, f"nest_hi_{narrow}_{wider}"), 2),
            ]
        rows.append([LABELS[m], *cells])
    headers = ["Method", "Not nested, any level (%)"]
    for narrow, wider in pairs:
        headers += [f"lo{wider} > lo{narrow} (%)", f"hi{wider} < hi{narrow} (%)"]
    return _md_table(headers, rows)


def conditional_methods(res: Results) -> list[str]:
    return [HEADLINE, CONFORMAL_C, RAW_QR, res.best_baseline()]


def month_table(res: Results) -> str:
    methods = conditional_methods(res)
    cov = {m: res.by_label(m, "cov_80", "month") for m in methods}
    below = res.by_label(HEADLINE, "below_80", "month")
    above = res.by_label(HEADLINE, "above_80", "month")
    rows = []
    for k, (y, mo) in enumerate(res.pairs.months):
        rows.append(
            [month_name(y, mo), str(res.pairs.month_days[k])]
            + [pct(cov[m][k]) for m in methods]
            + [pct(below[k]), pct(above[k])]
        )
    headers = [
        "Target month",
        "Days",
        *[f"{SHORT[m]} 80%" for m in methods],
        "CQR below %",
        "CQR above %",
    ]
    return _md_table(headers, rows)


def hour_table(res: Results) -> str:
    methods = conditional_methods(res)
    cov = {m: res.by_label(m, "cov_80", "hour") for m in methods}
    is80 = res.by_label(HEADLINE, "is_80", "hour")
    rows = [
        [f"{h:02d}:00"] + [pct(cov[m][h]) for m in methods] + [_fmt(is80[h])] for h in range(24)
    ]
    headers = ["Target hour (local)", *[f"{SHORT[m]} 80%" for m in methods], "CQR IS₈₀"]
    return _md_table(headers, rows)


def year_table(res: Results) -> str:
    methods = conditional_methods(res)
    years = sorted({y for y, _ in res.pairs.months})
    rows = []
    for year in years:
        cols = [k for k, (y, _) in enumerate(res.pairs.months) if y == year]
        n = res.counts.by_label["month"][:, cols].sum()

        def share(m: str, stat: str, cols: list[int] = cols, n: float = n) -> float:
            return float(res.stats[m][stat].by_label["month"][:, cols].sum() / n)

        rows.append(
            [str(year), str(sum(res.pairs.month_days[k] for k in cols))]
            + [pct(share(HEADLINE, f"cov_{lv}")) for lv in res.levels]
            + [pct(share(m, "cov_80")) for m in methods[1:]]
        )
    headers = [
        "Target year",
        "Days",
        *[f"CQR {lv}%" for lv in res.levels],
        *[f"{SHORT[m]} 80%" for m in methods[1:]],
    ]
    return _md_table(headers, rows)


def quarter_table(res: Results, retrains: Sequence[pd.Timestamp]) -> str:
    starts = np.array([r.tz_convert(NEM_TZ).date().toordinal() for r in retrains])
    quarter = np.searchsorted(starts, res.pairs.day_ordinals, "right") - 1
    methods = conditional_methods(res)
    rows = []
    n_day = res.counts.by_day
    for q, start in enumerate(retrains):
        days = quarter == q
        n = n_day[days].sum()
        if n == 0:
            continue

        def share(m: str, stat: str, days: np.ndarray = days, n: float = n) -> float:
            return float(res.stats[m][stat].by_day[days].sum() / n)

        rows.append(
            [f"{start.tz_convert(NEM_TZ):%Y-%m-%d}", str(int(days.sum()))]
            + [pct(share(HEADLINE, f"cov_{lv}")) for lv in res.levels]
            + [_fmt(share(HEADLINE, "is_80"))]
            + [pct(share(m, "cov_80")) for m in methods[1:]]
        )
    headers = [
        "Retrain (issue times from)",
        "Issue days",
        *[f"CQR {lv}%" for lv in res.levels],
        "CQR IS₈₀",
        *[f"{SHORT[m]} 80%" for m in methods[1:]],
    ]
    return _md_table(headers, rows)


def group_summary(res: Results, g: str) -> str:
    """CQR's 80% coverage in one group, with its interval and the side it misses on most."""
    e = res.group(HEADLINE, "cov_80", g)
    if e.n_pairs == 0:
        return f"{res.pairs.group_text[g]}: no scored pairs"
    below = res.group(HEADLINE, "below_80", g).value
    above = res.group(HEADLINE, "above_80", g).value
    side = "above" if above > below else "below"
    interval = f" (95% CI {pct(e.lo)}–{pct(e.hi)}%)" if e.n_blocks >= MIN_BOOTSTRAP_DAYS else ""
    return (
        f"{res.pairs.group_text[g]}: {pct(e.value)}%{interval}, missing {side} on "
        f"{pct(max(above, below))}% of pairs"
    )


def group_table(res: Results, groups: Sequence[str]) -> str:
    methods = conditional_methods(res)
    rows = []
    for g in groups:
        n = int(res.counts.by_group[g].sum())
        if n == 0:
            rows.append([res.pairs.group_text[g], "0", "0", "0", *["–"] * (len(methods) + 2)])
            continue
        below = res.group(HEADLINE, "below_80", g).value
        above = res.group(HEADLINE, "above_80", g).value
        is80 = res.group(HEADLINE, "is_80", g).value
        cells = []
        for m in methods:
            e = res.group(m, "cov_80", g)
            cells.append(ci(e, 100) if e.n_blocks >= MIN_BOOTSTRAP_DAYS else pct(e.value))
        n_days = res.group(HEADLINE, "cov_80", g).n_blocks
        rows.append(
            [
                res.pairs.group_text[g],
                str(res.pairs.group_days[g]),
                str(n_days),
                f"{n:,}",
                *cells,
                f"{pct(below)} / {pct(above)}",
                _fmt(is80),
            ]
        )
    headers = [
        "Group",
        "Target days",
        "Issue days",
        "Pairs",
        *[f"{SHORT[m]} 80% (95% CI)" for m in methods],
        "CQR below / above %",
        "CQR IS₈₀",
    ]
    return _md_table(headers, rows, "l" + "r" * (len(headers) - 1))


_KEY = re.compile(r"^w(?P<w>[\d.]+)_(?P<g>lead_tod|lead)_g(?P<gamma>[\d.e-]+)$")


def selection_counts(runs: Mapping[str, IntervalRun], levels: Sequence[int]) -> str:
    rows = []
    windows: list[str] = []
    gammas: list[str] = []
    parsed: dict[tuple[str, int], list[dict[str, str]]] = {}
    for m in (HEADLINE, CONFORMAL_C):
        for s in runs[m].config["selections"]:
            match = _KEY.match(s["chosen"])
            if match is None:
                raise ValueError(f"unexpected candidate key {s['chosen']!r}")
            parsed.setdefault((m, int(s["level"])), []).append(match.groupdict())
    for c in runs[HEADLINE].config["candidates"]:
        match = _KEY.match(c)
        if match is None:
            raise ValueError(f"unexpected candidate key {c!r}")
        if match["w"] not in windows:
            windows.append(match["w"])
        if match["gamma"] not in gammas:
            gammas.append(match["gamma"])
    for (m, lv), picks in parsed.items():
        rows.append(
            [SHORT[m], f"{lv}%", str(len(picks))]
            + [str(sum(p["w"] == w for p in picks)) for w in windows]
            + [str(sum(p["g"] == g for p in picks)) for g in ("lead", "lead_tod")]
            + [str(sum(p["gamma"] == g for p in picks)) for g in gammas]
        )
    headers = [
        "Method",
        "Level",
        "Selections",
        *[f"{w} weeks" for w in windows],
        "per lead",
        "lead × time of day",
        *[f"γ = {g}" for g in gammas],
    ]
    return _md_table(headers, rows)


def selection_grid(run: IntervalRun, level: int) -> str:
    picks = [s for s in run.config["selections"] if int(s["level"]) == level]
    bands = sorted({s["band"] for s in picks})
    by = {(s["retrain"], s["band"]): s["chosen"] for s in picks}
    retrains = sorted({s["retrain"] for s in picks})
    rows = [
        [f"{pd.Timestamp(r).tz_convert(NEM_TZ):%Y-%m-%d}"] + [f"`{by[(r, b)]}`" for b in bands]
        for r in retrains
    ]
    return _md_table(["Retrain", *[_band_text(b) for b in bands]], rows)


def _band_text(band: str) -> str:
    lo, hi = (int(x) for x in band.removeprefix("h").split("-"))
    return f"{horizon_label(lo)}–{horizon_label(hi)}"


def crossing_facts(engine: Engine, run: IntervalRun) -> list[str]:
    stored = stored_metrics(engine, run.run_id)
    cross = stored.loc[stored["metric"] == "crossing"].set_index("horizon")["value"]
    fits = run.config["fits"]
    by_band: dict[str, list[float]] = {}
    for f in fits:
        by_band.setdefault(f["band"], []).append(float(f["crossing_share"]))
    parts = [
        f"{_band_text(b)} {100 * min(v):.1f}–{100 * max(v):.1f}%"
        for b, v in sorted(by_band.items())
    ]
    return [
        f"Quantile pairs that crossed before sorting: {100 * float(cross.loc[0]):.1f}% of scored "
        f"pairs ({100 * float(run.config['crossing_share_all_pairs']):.1f}% of all pairs "
        "forecast from 2022-11-01).",
        "Per (retrain, band), the range of the crossing share over the retrains: "
        + "; ".join(parts)
        + ".",
    ]


def _criteria(res: Results, leakage: LeakageOutcome) -> list[Criterion]:
    return success_criteria(
        100 * res.per_lead(HEADLINE, "cov_80"),
        100 * res.pooled(HEADLINE, "cov_80"),
        res.per_lead(HEADLINE, "is_80"),
        res.pooled(HEADLINE, "is_80"),
        {m: res.per_lead(m, "is_80") for m in BASELINES},
        leakage,
        SHORT,
    )


# --------------------------------------------------------------------------- report


def build_report(
    engine: Engine, cfg: AppConfig, region_id: str, leakage: LeakageOutcome | None = None
) -> str:
    report_sha = git_sha()  # before any figure is written, which would make the tree dirty
    runs = find_runs(engine, cfg, region_id)
    leakage = leakage or run_leakage_tests()
    res = score_all(engine, cfg, region_id, runs)
    figures = cfg.resolve(cfg.paths.reports_dir) / "figures"
    plot_coverage_by_lead(res, figures / "m3_coverage_by_lead.png")
    plot_is_vs_envelope(res, figures / "m3_is80_vs_baselines.png")
    plot_reliability(res, figures / "m3_reliability.png")
    left_out = plot_coverage_by_month(res, figures / "m3_coverage_by_month.png")

    criteria = _criteria(res, leakage)
    best = res.best_baseline()
    sens_table, diffs = headline_sensitivity(res)
    cov_lead, is_lead = lead_group_tables(res)
    share_table, pin_table = reliability_tables(res)
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    retrains = selection_retrains(cfg, end)
    cov80 = 100 * res.per_lead(HEADLINE, "cov_80")
    lo2, hi2 = U2_RANGE
    outside = [int(h) + 1 for h in np.flatnonzero((cov80 < lo2) | (cov80 > hi2))]
    month_cov = res.by_label(HEADLINE, "cov_80", "month")
    full = [k for k, d in enumerate(res.pairs.month_days) if d >= MIN_MONTH_DAYS]
    worst_m = min(full, key=lambda k: month_cov[k])
    best_m = max(full, key=lambda k: month_cov[k])
    cqr_cov = res.estimate(HEADLINE, "cov_80")
    cqr_cov_w = res.estimate(HEADLINE, "cov_80", res.weeks)
    d_best = next(d for o, n, d, _ in diffs if o == best and n == "IS₈₀")
    d_conf = next(d for o, n, d, _ in diffs if o == CONFORMAL_C and n == "IS₈₀")
    w_conf = next(d for o, n, d, _ in diffs if o == CONFORMAL_C and n == "WIS")
    d_conf_w = next(w for o, n, _, w in diffs if o == CONFORMAL_C and n == "IS₈₀")
    rearr = res.difference(REARRANGED, HEADLINE, "wis")
    passed = sum(c.passed for c in criteria)
    verdicts = ", ".join(f"{c.key} {'pass' if c.passed else 'FAIL'}" for c in criteria)
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    is_best = res.pooled(best, "is_80")
    is_conf = res.pooled(CONFORMAL_C, "is_80")
    cqr_cfg = runs[HEADLINE].config

    out = [
        "# Milestone 3: prediction intervals for Victorian demand (development period)\n",
        f"> **Generated** by `gridcast report-m3` on {generated} from the stored interval "
        "batches. Do not edit by hand: re-run the pipeline instead. The holdout year is not "
        "scored here (step 6).\n",
        "## Summary\n",
        f"- **{passed} of {len(criteria)} pre-registered development criteria pass** "
        f"({verdicts}). "
        "U5, the single holdout run, belongs to step 6.",
        f"- **Calibration:** the headline CQR intervals cover {pct(cqr_cov.value, 2)}% of actual "
        f"demand at 80% (95% CI {pct(cqr_cov.lo, 2)}–{pct(cqr_cov.hi, 2)}% with day blocks, "
        f"{pct(cqr_cov_w.lo, 2)}–{pct(cqr_cov_w.hi, 2)}% with {WEEK_DAYS}-day blocks), "
        f"{pct(res.pooled(HEADLINE, 'cov_50'))}% at 50% and "
        f"{pct(res.pooled(HEADLINE, 'cov_95'))}% at 95%. The raw quantile LightGBM C bounds "
        f"it corrects cover {pct(res.pooled(RAW_QR, 'cov_80'))}% at 80%.",
        f"- **Sharpness:** CQR's pooled IS₈₀ is {_fmt(res.pooled(HEADLINE, 'is_80'))} MW, "
        f"against {_fmt(is_best)} MW for the best baseline interval ({LABELS[best].lower()}; "
        f"{signed(100 * d_best.value / is_best)}%) and {_fmt(is_conf)} MW for conformal "
        f"intervals around the same point forecast ({signed(100 * d_conf.value / is_conf)}%; "
        f"difference {ci_signed(d_conf)} MW with day blocks, "
        f"{signed(d_conf_w.lo)} to {signed(d_conf_w.hi)} with {WEEK_DAYS}-day blocks). Over all "
        f"three levels, the WIS difference to conformal C is {ci_signed(w_conf)} MW"
        + ("; its 95% interval includes zero." if w_conf.lo < 0 < w_conf.hi else ".")
        + "",
        f"- **By lead time:** 80% coverage stays within {lo2:g}–{hi2:g}% at "
        f"{len(cov80) - len(outside)} of {len(cov80)} leads"
        + (f" (outside at {horizon_ranges(outside)})" if outside else "")
        + f", from {_fmt(float(cov80.min()), 2)}% to {_fmt(float(cov80.max()), 2)}%. "
        f"**By month** it ranges from {pct(month_cov[worst_m])}% "
        f"({month_name(*res.pairs.months[worst_m])}) to {pct(month_cov[best_m])}% "
        f"({month_name(*res.pairs.months[best_m])}).",
        "- **Extreme events known at the issue time** (reported only): "
        + "; ".join(group_summary(res, g) for g in res.pairs.known_groups)
        + ".",
        f"- **Disclosed:** on {pct(res.pooled(HEADLINE, 'non_nested'))}% of scored pairs the "
        "CQR levels are not nested (each level is corrected separately). Sorting the bounds "
        f"(a reported-only row) changes the WIS by {ci_signed(rearr, 2)} MW.\n",
        "## Provenance\n",
        _provenance(cfg, runs, res, leakage, region_id, report_sha) + "\n",
        "## Success criteria (pre-registered)\n",
        "Set in `docs/plans/m3_plan.md` §3 before any interval result existed. They apply to "
        "the headline, **CQR on quantile LightGBM C at 80%**, on the development issue times "
        "and the same pairs as every other method. *Baseline intervals* are conformal "
        "intervals around persistence, daily naive and weekly naive, built with the same "
        "procedure and per-retrain selection; *the minimum over baselines* is taken at each "
        "lead time. U3's pooled part compares with the mean over every lead time of that "
        "minimum, as M2's S2 did. U5 belongs to step 6.\n",
        criteria_table(criteria) + "\n",
        "## Methods\n",
        "- **CQR (headline):** quantile LightGBM C (step 2: six quantiles, each selecting its "
        "own target form and leaf count at every retrain and lead band, then sorted per pair) "
        "corrected by rolling conformal calibration with one score per side "
        "(q̂_lo − y, y − q̂_hi).",
        "- **Conformal around a point forecast:** the same rolling machinery on signed "
        "residuals of a stored point forecast (LightGBM C, or a baseline for the baseline "
        "intervals).",
        "- **Shared machinery (step 1):** 24 candidate streams (calibration windows of "
        f"{', '.join(str(w) for w in cfg.intervals.grid.window_weeks)} weeks × per lead or per "
        "lead × 3-hour block of the target × ACI step γ ∈ "
        f"{{{', '.join(f'{g:g}' for g in cfg.intervals.grid.aci_gammas)}}}), re-selected at "
        "every quarterly retrain, per level and lead band, by mean interval score over the "
        f"previous {cfg.intervals.selection_weeks} weeks. Only outcomes known at the issue "
        "time are ever used.",
        "- **Raw QR** is reported to show what the conformal step adds. **CQR, rearranged** "
        "sorts each pair's six CQR bounds as quantiles (reported only, added at step 4).",
        "- **Scores:** coverage, interval score IS_α = (u − l) + (2/α)(l − y)⁺ + (2/α)(y − u)⁺, "
        "and the weighted interval score (WIS) of the three intervals plus the method's point "
        "forecast as the median. Lower scores are better. All in MW.\n",
        "## Pooled results\n",
        pooled_table(res) + "\n",
        "Per level, with 95% intervals for coverage (block bootstrap over whole issue days) "
        "and the share of actuals below and above the interval:\n",
        level_table(res) + "\n",
        "### Headline differences and the block-length sensitivity\n",
        "Errors are correlated in time: weather spells and the winter bias last for weeks, and "
        "24-hour leads overlap the next issue day. The same quantities are therefore also "
        f"bootstrapped with {WEEK_DAYS}-day blocks (user decision 2026-11-12). Negative "
        "differences mean CQR scores better.\n",
        sens_table + "\n",
        "Other differences (reported only):\n",
        other_differences(res) + "\n",
        "## By lead time\n",
        "![80% coverage by lead](figures/m3_coverage_by_lead.png)\n",
        f"80% coverage by lead time, with U2's {lo2:g}–{hi2:g}% range shaded.\n",
        "80% coverage (%) by lead-time group:\n",
        cov_lead + "\n",
        "![IS80 against the baseline envelope](figures/m3_is80_vs_baselines.png)\n",
        "The dashed line is the lowest baseline IS₈₀ at each lead time ("
        + "; ".join(envelope_facts(res))
        + ").\n",
        "IS₈₀ (MW) by lead-time group:\n",
        is_lead + "\n",
        "## Reliability and quantile loss\n",
        "![Reliability diagram](figures/m3_reliability.png)\n",
        "Each bound of a central interval at level L is a quantile at τ = ½ ∓ L/200. The figure "
        "shows how far the observed share of actuals at or below each bound is from τ "
        "(0 = perfectly calibrated). Share at or below each bound (%):\n",
        share_table + "\n",
        "Mean pinball loss per quantile level (MW):\n",
        pin_table + "\n",
        "## Nesting of the levels (disclosure)\n",
        "Each level of CQR is corrected separately (its own scores, ACI state and selection), so "
        "a wider level's bound can fall inside a narrower level's. Each level's coverage and "
        "interval score are unaffected, but the WIS and the reliability diagram combine "
        "levels. The raw QR quantiles were sorted and are always nested. Share of scored pairs "
        "(%):\n",
        nesting_table(res) + "\n",
        f"Sorting CQR's bounds per pair changes its WIS by {ci_signed(rearr, 2)} MW and its "
        f"80% coverage from {pct(res.pooled(HEADLINE, 'cov_80'), 2)}% to "
        f"{pct(res.pooled(REARRANGED, 'cov_80'), 2)}%. The headline is not changed: U1–U3 judge "
        "the CQR bounds as stored.\n",
        "## Conditional coverage\n",
        f"Shown for CQR, conformal around LightGBM C, raw QR and the best baseline interval "
        f"({LABELS[best].lower()}, lowest pooled IS₈₀). Hours, months and years are those of "
        "the target in local time; retrain quarters group issue times.\n",
        "![80% coverage by month](figures/m3_coverage_by_month.png)\n",
        "80% coverage by month of the target"
        + (
            f" ({', '.join(left_out)} left out of the figure: fewer than {MIN_MONTH_DAYS} "
            "target days)."
            if left_out
            else "."
        )
        + "\n",
        "### By month\n",
        month_table(res) + "\n",
        "### By target hour\n",
        hour_table(res) + "\n",
        "### By year\n",
        year_table(res) + "\n",
        "### By retrain quarter\n",
        quarter_table(res, retrains) + "\n",
        "## Extreme events (reported only)\n",
        "Coverage intervals resample whole issue days. They are shown only for groups whose "
        f"pairs come from at least {MIN_BOOTSTRAP_DAYS} issue days; with fewer, resampling "
        "days says little. A target day is reached from the issue days before and on it.\n",
        "### Groups known at the issue time\n",
        "These groups are honest conditional coverage: a forecaster could have known them when "
        "the interval was issued. The forecast daily maximum is the population-weighted GFS "
        "temperature of the target's local day, using for each 5-minute interval the freshest "
        "forecast published by the issue time (the rule C's features use).\n",
        group_table(res, res.pairs.known_groups) + "\n",
        "### Groups defined by the outcome\n",
        "These groups are chosen by what happened, so their coverage is biased downwards by "
        "construction: an interval that missed is more likely to sit in a group defined by an "
        "extreme outcome. They show where the intervals fail, not how well calibrated they "
        "are. The top-1% threshold is the 99th percentile of actual demand over the scored "
        "target intervals.\n",
        group_table(res, res.pairs.outcome_groups) + "\n",
        "## Per-retrain selections\n",
        "How often each setting was chosen (selections per level: retrains × lead bands):\n",
        selection_counts(runs, res.levels) + "\n",
        "CQR's 80% selection at each retrain and lead band (`w<weeks>_<grouping>_g<γ>`):\n",
        selection_grid(runs[HEADLINE], 80) + "\n",
        "Crossed CQR intervals, output as zero width at the midpoint: "
        + ", ".join(f"{int(v):,} at {k}%" for k, v in cqr_cfg["crossed"].items())
        + " (all issue times with an output).\n",
        "### Quantile crossing (raw QR)\n",
        "\n".join(f"- {line}" for line in crossing_facts(engine, runs[RAW_QR])) + "\n",
        "## Leakage tests (U4)\n",
        "Run by this command: " + ", ".join(f"`{t}`" for t in LEAKAGE_TESTS) + ". They check "
        "that perturbing actual demand after an issue time, or point and QR forecasts issued "
        "after it, leaves every interval issued at that time unchanged, for conformal around a "
        "point forecast and for CQR. Control tests show the checks fail when the information "
        f"rule is broken by one step. **{leakage.passed} passed, {leakage.failed} failed.**\n",
        "## Disclosures\n",
        "- The candidate grid (including ACI) was designed after M2's development and holdout "
        "point errors had been seen, including the winter under-forecast. Settings are "
        "re-selected from past data only, but the design itself was not blind.",
        "- QR tuning was switched to per-quantile selection after a smoke run on the first "
        "retrain (pre-scoring data) whose raw-QR results were seen (m3_plan.md §1).",
        "- The *CQR, rearranged* row and the 7-day-block sensitivity were added at step 4, "
        "after step 3's results (including the non-nesting share) were seen. Both are "
        "reported only.",
        "- The outcome-defined groups are biased downwards by construction (see above).",
        f"- {res.pairs.mask.size - int(res.pairs.mask.sum()):,} pairs without C's weather "
        "inputs are excluded for every method (run 2's common mask).",
    ]
    return "\n".join(out) + "\n"


def write_report(
    engine: Engine, cfg: AppConfig, region_id: str, leakage: LeakageOutcome | None = None
) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m3_intervals.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id, leakage))
    return path

"""M3 step 5: our 80% intervals against AEMO's POE90–POE10 band (development part).

``gridcast report-m3-aemo`` writes ``reports/m3_aemo.md`` and two figures. Nothing is typed by
hand and nothing is pass/fail (m3_plan.md §3): AEMO's band is a benchmark, never a model input.

* **Our intervals** are the half-hour batch (:mod:`gridcast.intervals.halfhour`): the frozen
  recipe applied to 30-minute period means. The command refuses to run without a freeze
  record, and refuses runs whose recipe hashes or commit differ from it.
* **Issue times** *o* run from ``aemo.first_issue`` to ``backtest.holdout_start``, as in M2's
  AEMO comparison; holdout issue times are never read. Period k (k = 1 … 48) ends at
  o + 30k min.
* **AEMO's band** is PDPASA's ``[DEMAND90, DEMAND10]`` (``aemo_demand_poe``, LOR runs).
  Alignment as M2's (m3_plan.md §2.6): *matched* (headline), the run whose nominal run time
  is *o*; *strict*, the latest run published by *o*. *Re-centred*: the same half-width
  around the pre-dispatch ``TOTALDEMAND`` of the run with the same nominal run time
  (published before the PDPASA run), as a sensitivity check.
* **Pairs:** every interval is scored on the same (issue time, period) pairs: our scored
  periods (all six 5-minute pairs in run 2's common mask) where all four AEMO bands exist.
* **Statistics** (80% only, the level AEMO publishes): coverage, share below and above, mean
  width and interval score; pooled differences get M2's day-block bootstrap and a 7-day
  block sensitivity (m3_plan.md step 4).
* **Integrity:** before anything is written, every pooled 80% statistic of each half-hour
  run, recomputed from its stored bounds over its whole development period, must equal the
  stored metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.aemo_benchmark import (
    AEMO_COLOR,
    N_PERIODS,
    TABLE_PERIODS,
    aligned,
    load_aemo,
    period_label,
    strict_runs,
    window,
)
from gridcast.evaluation.report import _fmt, _md_table
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import TARGET, interval_config_hash
from gridcast.intervals.evaluation import (
    BOOTSTRAP_DRAWS,
    Aggregator,
    Estimate,
    StatSums,
    bootstrap_ratio,
    issue_day_codes,
    week_blocks,
)
from gridcast.intervals.halfhour import CONFORMAL, CQR_NAME, PERIOD_MINUTES, period_mask
from gridcast.intervals.inputs import read_stored_batch, read_wide
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.report import (
    COLORS,
    IntervalRun,
    _end_labels,
    _interval_runs,
    _legend,
    _new_figure,
    _save,
    _style,
    stored_metrics,
)
from gridcast.intervals.scores import alpha, covered, interval_score
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

LEVEL = 80
HEADLINE = CQR_NAME  # cqr30_lgbm_C
CONFORMAL_C = f"{CONFORMAL}_lgbm_C"
BASELINES = tuple(f"{CONFORMAL}_{m}" for m in ("persistence", "daily_naive", "weekly_naive"))
OURS = (HEADLINE, CONFORMAL_C, *BASELINES)
PM, PS, RM, RS = (
    "aemo_published_matched",
    "aemo_published_strict",
    "aemo_recentred_matched",
    "aemo_recentred_strict",
)
AEMO = (PM, PS, RM, RS)
BEST = "best_baseline"  # per lead: the baseline with the lowest IS80 there
LABELS = {
    HEADLINE: "CQR, LightGBM C (headline)",
    CONFORMAL_C: "Conformal around LightGBM C",
    BASELINES[0]: "Conformal around persistence",
    BASELINES[1]: "Conformal around daily naive",
    BASELINES[2]: "Conformal around weekly naive",
    PM: "AEMO band (matched)",
    PS: "AEMO band (strict)",
    RM: "AEMO band re-centred (matched)",
    RS: "AEMO band re-centred (strict)",
}
SHORT = {
    HEADLINE: "CQR",
    CONFORMAL_C: "Conformal C",
    BASELINES[0]: "Persistence",
    BASELINES[1]: "Daily naive",
    BASELINES[2]: "Weekly naive",
    PM: "AEMO (matched)",
    PS: "AEMO (strict)",
    RM: "AEMO re-centred (matched)",
    RS: "AEMO re-centred (strict)",
}
STATS = ("cov", "below", "above", "width", "is")
STORED = {"cov": "coverage", "below": "below", "above": "above", "width": "width",
          "is": "interval_score"}  # fmt: skip
FIGURES = ("m3_aemo_coverage_by_lead.png", "m3_aemo_is80_by_lead.png")
TITLE_PERIOD = "development, Oct 2024 → Sep 2025"
RELATIVE_TOLERANCE = 1e-9
SYMMETRY_MW = 1.0  # |(POE10 − POE50) − (POE50 − POE90)| within this counts as symmetric


# --------------------------------------------------------------------------- runs


def find_halfhour_runs(engine: Engine, cfg: AppConfig, region_id: str) -> dict[str, IntervalRun]:
    """The latest half-hour runs, from one batch, built by the frozen recipe and commit."""
    iv = cfg.intervals
    if not (iv.frozen_config_hash and iv.frozen_qr_config_hash and iv.frozen_commit):
        raise LookupError("the AEMO comparison runs only after the freeze (intervals.frozen_*)")
    current = interval_config_hash(cfg)
    if current != iv.frozen_config_hash or qr_config_hash(cfg) != iv.frozen_qr_config_hash:
        raise ValueError("the current interval or QR recipe differs from the frozen one")
    latest: dict[str, IntervalRun] = {}
    for run in _interval_runs(engine, region_id):
        if run.config.get("resolution_minutes") == PERIOD_MINUTES:
            latest[run.name] = run
    if missing := [m for m in OURS if m not in latest]:
        raise LookupError(f"no stored half-hour interval run for {missing}")
    out = {m: latest[m] for m in OURS}
    if len({r.batch_id for r in out.values()}) != 1:
        raise ValueError("the latest half-hour runs come from several batches")
    for name, run in out.items():
        cfg_run = run.config
        if cfg_run.get("interval_config_hash") != iv.frozen_config_hash:
            raise ValueError(f"{name} (run {run.run_id}) has another interval config hash")
        if cfg_run.get("qr_config_hash") != iv.frozen_qr_config_hash:
            raise ValueError(f"{name} (run {run.run_id}) has another QR config hash")
        if not (run.git_sha or "").startswith(iv.frozen_commit):
            raise ValueError(
                f"{name} (run {run.run_id}) was built at {run.git_sha}, not the frozen commit "
                f"{iv.frozen_commit}"
            )
        if cfg_run.get("source_batch") != iv.source_batch:
            raise ValueError(f"{name} (run {run.run_id}) was built on another source batch")
    return out


# --------------------------------------------------------------------------- statistics


def pair_stats(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pair 80% statistics (NaN propagates; callers mask)."""
    with np.errstate(invalid="ignore"):
        return {
            "cov": covered(lo, hi, y).astype(np.float64),
            "below": (y < lo).astype(np.float64),
            "above": (y > hi).astype(np.float64),
            "width": hi - lo,
            "is": interval_score(lo, hi, y, alpha(LEVEL)),
        }


def check_against_stored(
    engine: Engine,
    runs: dict[str, IntervalRun],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    actual: np.ndarray,
    mask: np.ndarray,
) -> list[str]:
    """Recompute each run's pooled 80% statistics over its whole period; they must match."""
    checked = []
    for name, run in runs.items():
        stored = stored_metrics(engine, run.run_id)
        pooled_stored = stored[stored["horizon"] == 0].set_index("metric")["value"]
        stats = pair_stats(*bounds[name], actual)
        for stat, values in stats.items():
            mine = float(np.where(mask, values, 0.0).sum() / mask.sum())
            theirs = float(pooled_stored[f"{STORED[stat]}_{LEVEL}"])
            if not np.isclose(mine, theirs, rtol=RELATIVE_TOLERANCE, atol=0.0):
                raise ValueError(
                    f"{name}: recomputed {STORED[stat]}_{LEVEL} {mine!r} differs from the "
                    f"stored {theirs!r}"
                )
            checked.append(f"{name}:{stat}")
    return checked


# --------------------------------------------------------------------------- loading


@dataclass(slots=True)
class Comparison:
    """Everything the report needs, on (issue time × period) grids."""

    issue_times: pd.DatetimeIndex
    runs: dict[str, IntervalRun]
    actual: np.ndarray  # (n, 48) period means
    ours_mask: np.ndarray  # our scored periods
    bounds: dict[str, tuple[np.ndarray, np.ndarray]]  # method -> (lower, upper)
    median: dict[str, np.ndarray]  # PM / PS: DEMAND50; RM / RS: pre-dispatch TOTALDEMAND
    available: dict[str, np.ndarray]  # PM, PS: (n, 48) whether the band exists
    mask: np.ndarray  # the compared pairs
    poe_runs: int  # PDPASA LOR runs in the window
    expected_runs: int
    strict_lag: pd.Series  # o − strict run time
    symmetric_share: float  # PDPASA rows in the window whose band is symmetric within 1 MW
    recentring_after: int  # chosen runs whose pre-dispatch run was published after the PDPASA one
    checked: list[str]  # recomputed statistics that matched the stored metrics


def load_poe(
    engine: Engine, region_id: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT run_time, target_end, available_at, demand10_mw, demand50_mw, demand90_mw
                FROM aemo_demand_poe
                WHERE region_id = :r AND run_time BETWEEN :a AND :b
                ORDER BY run_time, target_end
                """
            ),
            conn,
            params={"r": region_id, "a": first.to_pydatetime(), "b": last.to_pydatetime()},
        )


def _grid(
    frame: pd.DataFrame, column: str, issue_times: pd.DatetimeIndex, chosen: pd.DatetimeIndex
) -> np.ndarray:
    renamed = frame[["run_time", "target_end", column]].rename(columns={column: "total_demand_mw"})
    return aligned(renamed, issue_times, chosen, PERIOD_MINUTES, N_PERIODS)


def _read_level(
    run: IntervalRun, data_root: Path, side: str
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    return read_wide(data_root / run.directory / f"{run.name}__{side}{LEVEL}.parquet")


def load_comparison(engine: Engine, cfg: AppConfig, region_id: str) -> Comparison:
    """Our stored half-hour intervals and AEMO's bands, lined up on the development window."""
    first, end = window(cfg, "development")
    runs = find_halfhour_runs(engine, cfg, region_id)
    data_root = cfg.data_root()
    directory = data_root / runs[HEADLINE].directory
    times_all, y_all = read_wide(directory / "actual.parquet")

    # Our scored periods over the batch's whole period, from run 2's common mask.
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    source = read_stored_batch(
        cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch, TARGET
    ).restrict(start=times_all[0], end=holdout)
    if not source.origin_times.equals(times_all):
        raise ValueError("the half-hour batch's issue times differ from run 2's")
    mask_all = period_mask(source.common_mask())
    bounds_all: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, run in runs.items():
        t_lo, lo = _read_level(run, data_root, "lo")
        t_hi, hi = _read_level(run, data_root, "hi")
        if not (t_lo.equals(times_all) and t_hi.equals(times_all)):
            raise ValueError(f"{name}: issue times differ from the batch's actuals")
        bounds_all[name] = (lo, hi)
    checked = check_against_stored(engine, runs, bounds_all, y_all, mask_all)

    keep = np.asarray((times_all >= first) & (times_all < end))
    issue_times = times_all[keep]
    if len(issue_times) == 0:
        raise LookupError("no stored half-hour issue times in the AEMO window")
    if issue_times.max() >= end:
        raise AssertionError("issue times after the window must never be read")
    actual = y_all[keep]
    ours_mask = mask_all[keep]
    bounds = {m: (lo[keep], hi[keep]) for m, (lo, hi) in bounds_all.items()}

    # AEMO: PDPASA runs up to a day before the first issue time cover the strict alignment.
    poe = load_poe(engine, region_id, first - pd.Timedelta(days=1), end)
    if poe.empty:
        raise LookupError("no PDPASA POE forecasts in the AEMO window (gridcast ingest-aemo-poe)")
    pre = load_aemo(engine, region_id, "predispatch", first - pd.Timedelta(days=1), end)
    poe_runs = poe.groupby("run_time")["available_at"].max().reset_index()
    pre_runs = pre.groupby("run_time")["available_at"].max()
    strict = strict_runs(poe_runs, issue_times)
    choices = {"matched": pd.DatetimeIndex(issue_times), "strict": strict}
    median: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}
    after = 0
    published = poe_runs.set_index("run_time")["available_at"]
    for alignment, chosen in choices.items():
        d10 = _grid(poe, "demand10_mw", issue_times, chosen)
        d50 = _grid(poe, "demand50_mw", issue_times, chosen)
        d90 = _grid(poe, "demand90_mw", issue_times, chosen)
        centre = aligned(pre, issue_times, chosen, PERIOD_MINUTES, N_PERIODS)
        half = (d10 - d90) / 2
        pub, rec = (PM, RM) if alignment == "matched" else (PS, RS)
        bounds[pub] = (d90, d10)
        bounds[rec] = (centre - half, centre + half)
        median[pub], median[rec] = d50, centre
        available[pub] = np.isfinite(d50)
        runs_used = pd.DatetimeIndex(chosen.dropna().unique())
        pdpasa_at = published.reindex(runs_used)
        pre_at = pre_runs.reindex(runs_used)
        after += int((pre_at > pdpasa_at).sum())

    mask = ours_mask & np.isfinite(actual)
    for lo, hi in bounds.values():
        mask &= np.isfinite(lo) & np.isfinite(hi)

    in_window = poe[(poe["run_time"] >= first) & (poe["run_time"] < end)]
    asym = (in_window["demand10_mw"] - in_window["demand50_mw"]) - (
        in_window["demand50_mw"] - in_window["demand90_mw"]
    )
    n_runs = int(in_window["run_time"].nunique())
    expected = int((end - first) / pd.Timedelta(minutes=PERIOD_MINUTES))
    lag = pd.Series(issue_times - strict, index=issue_times).dropna()
    return Comparison(
        issue_times=pd.DatetimeIndex(issue_times),
        runs=runs,
        actual=actual,
        ours_mask=ours_mask,
        bounds=bounds,
        median=median,
        available=available,
        mask=mask,
        poe_runs=n_runs,
        expected_runs=expected,
        strict_lag=lag,
        symmetric_share=float((asym.abs() <= SYMMETRY_MW).mean()),
        recentring_after=after,
        checked=checked,
    )


# --------------------------------------------------------------------------- results


@dataclass(slots=True)
class Results:
    c: Comparison
    agg: Aggregator
    sums: dict[str, dict[str, StatSums]]  # method -> stat -> sums
    day_ordinals: np.ndarray
    quarters: list[str]
    seed: int

    def per_lead(self, method: str, stat: str) -> np.ndarray:
        n = self.agg.counts.by_lead.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(self.sums[method][stat].by_lead.sum(axis=0) / n)

    def pooled(self, method: str, stat: str) -> float:
        return float(self.sums[method][stat].by_lead.sum() / self.agg.counts.by_lead.sum())

    def estimate(self, method: str, stat: str, blocks: np.ndarray | None = None) -> Estimate:
        s = self.sums[method][stat].by_day
        return bootstrap_ratio(s, self.agg.counts.by_day, self.seed, blocks=blocks)

    def difference(self, a: str, b: str, stat: str, blocks: np.ndarray | None = None) -> Estimate:
        s = self.sums[a][stat].by_day - self.sums[b][stat].by_day
        return bootstrap_ratio(s, self.agg.counts.by_day, self.seed, blocks=blocks)

    def weeks(self) -> np.ndarray:
        return week_blocks(self.day_ordinals, 7)

    def best_baseline(self) -> str:
        """The baseline with the lowest pooled IS80 on the compared pairs."""
        return min(BASELINES, key=lambda m: self.pooled(m, "is"))

    def best_per_lead(self) -> list[str]:
        """Per period: the baseline with the lowest IS80 there."""
        stacked = np.stack([self.per_lead(m, "is") for m in BASELINES])
        return [BASELINES[i] for i in np.argmin(stacked, axis=0)]

    def by_quarter(self, method: str, stat: str) -> np.ndarray:
        s = self.sums[method][stat].by_label["quarter"].sum(axis=0)
        n = self.agg.counts.by_label["quarter"].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(s / n)

    def quarter_pairs(self) -> np.ndarray:
        return np.asarray(self.agg.counts.by_label["quarter"].sum(axis=0))


def score(c: Comparison, seed: int) -> Results:
    nem_dates = c.issue_times.tz_convert(NEM_TZ)
    day_code, day_ordinals = issue_day_codes(np.array([d.toordinal() for d in nem_dates.date]))
    quarter_of = [f"{t.year} Q{(t.month - 1) // 3 + 1}" for t in nem_dates]
    quarters = sorted(set(quarter_of))
    q_idx = np.array([quarters.index(q) for q in quarter_of])
    labels = {"quarter": (np.repeat(q_idx[:, None], N_PERIODS, axis=1), len(quarters))}
    agg = Aggregator(c.mask, day_code, labels=labels)
    sums: dict[str, dict[str, StatSums]] = {}
    for method, (lo, hi) in c.bounds.items():
        stats = pair_stats(lo, hi, c.actual)
        sums[method] = {s: agg.add(np.nan_to_num(v, nan=0.0)) for s, v in stats.items()}
    return Results(c, agg, sums, day_ordinals, quarters, seed)


# --------------------------------------------------------------------------- figures


def _hours() -> np.ndarray:
    return np.arange(1, N_PERIODS + 1) * PERIOD_MINUTES / 60


PLOTTED = (HEADLINE, CONFORMAL_C, PM, PS)
# The interval report's hues for our methods; AEMO red as in M2's AEMO figure (strict dashed).
COLOR = {
    HEADLINE: COLORS["cqr_lgbm_C"],
    CONFORMAL_C: COLORS["conformal_lgbm_C"],
    **{b: COLORS[b.replace(f"{CONFORMAL}_", "conformal_")] for b in BASELINES},
    PM: AEMO_COLOR,
    PS: AEMO_COLOR,
}


def _plot(res: Results, stat: str, path: Path, ylabel: str, title: str) -> None:
    best = res.best_baseline()
    order = [best, *PLOTTED]
    scale = 100.0 if stat == "cov" else 1.0
    series = {m: scale * res.per_lead(m, stat) for m in order}
    hours = _hours()
    fig, ax = _new_figure()
    if stat == "cov":
        ax.axhline(80, color="#b5b3ad", linewidth=1, zorder=1)
    for m in order:
        ax.plot(
            hours,
            series[m],
            color=COLOR[m],
            linewidth=2,
            linestyle="--" if m == PS else "-",
            label=SHORT[m],
            solid_capstyle="round",
            zorder=3,
        )
    low = min(float(np.nanmin(v)) for v in series.values())
    high = max(float(np.nanmax(v)) for v in series.values())
    pad = (high - low) * 0.08 or 1.0
    span = (low - pad, high + pad)
    if stat == "cov":
        span = (min(span[0], 78.0), max(span[1], 82.0))
    else:
        span = (max(0.0, span[0]), span[1])
    ax.set_ylim(*span)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ends = []
    for m in order:
        finite = np.flatnonzero(np.isfinite(series[m]))
        if finite.size:
            ends.append((float(series[m][finite[-1]]), SHORT[m]))
    _end_labels(ax, 24, ends, span)
    _style(ax, "Forecast lead time (hours from issue time to period end)", ylabel, title)
    _legend(ax, len(order))
    _save(fig, path)


# --------------------------------------------------------------------------- text


def pct(v: float, digits: int = 1) -> str:
    return _fmt(100 * v, digits)


def signed(v: float, digits: int = 1) -> str:
    return "–" if not np.isfinite(v) else f"{v:+,.{digits}f}"


def ci_text(e: Estimate, scale: float = 1.0, digits: int = 1) -> str:
    return f"{scale * e.lo:+,.{digits}f} to {scale * e.hi:+,.{digits}f}"


def _lead_rows(
    res: Results, stat: str, methods: Sequence[str], scale: float, digits: int = 1
) -> list[list[str]]:
    best = res.best_per_lead()
    pairs = res.agg.counts.by_lead.sum(axis=0)
    rows = []
    for k in TABLE_PERIODS:
        row = [period_label(k)]
        for m in methods:
            if m == BEST:
                b = best[k - 1]
                row.append(f"{_fmt(scale * res.per_lead(b, stat)[k - 1], digits)} ({SHORT[b]})")
            else:
                row.append(_fmt(scale * res.per_lead(m, stat)[k - 1], digits))
        row.append(f"{int(pairs[k - 1]):,}")
        rows.append(row)
    pooled_best = res.best_baseline()
    last = ["**All periods (pooled)**"]
    for m in methods:
        name = pooled_best if m == BEST else m
        value = _fmt(scale * res.pooled(name, stat), digits)
        last.append(f"{value} ({SHORT[name]})" if m == BEST else f"**{value}**")
    last.append(f"{int(pairs.sum()):,}")
    rows.append(last)
    return rows


def _header(methods: Sequence[str]) -> list[str]:
    return ["Period ends", *("Best baseline" if m == BEST else SHORT[m] for m in methods), "Pairs"]


MAIN = (PM, PS, HEADLINE, CONFORMAL_C, BEST)


def coverage_table(res: Results) -> str:
    return _md_table(_header(MAIN), _lead_rows(res, "cov", MAIN, 100.0))


def miss_table(res: Results) -> str:
    methods = (PM, HEADLINE, CONFORMAL_C)
    headers = ["Period ends"]
    for m in methods:
        headers += [f"{SHORT[m]} below", f"{SHORT[m]} above"]
    rows = []
    for k in TABLE_PERIODS:
        row = [period_label(k)]
        for m in methods:
            row += [pct(res.per_lead(m, "below")[k - 1]), pct(res.per_lead(m, "above")[k - 1])]
        rows.append(row)
    last = ["**All periods (pooled)**"]
    for m in methods:
        last += [pct(res.pooled(m, "below")), pct(res.pooled(m, "above"))]
    rows.append(last)
    return _md_table(headers, rows)


def width_table(res: Results) -> str:
    return _md_table(_header(MAIN), _lead_rows(res, "width", MAIN, 1.0))


def is_table(res: Results) -> str:
    return _md_table(_header(MAIN), _lead_rows(res, "is", MAIN, 1.0))


def sensitivity_table(res: Results) -> str:
    methods = (*AEMO, HEADLINE, CONFORMAL_C, res.best_baseline())
    rows = [
        [
            LABELS[m],
            pct(res.pooled(m, "cov")),
            pct(res.pooled(m, "below")),
            pct(res.pooled(m, "above")),
            _fmt(res.pooled(m, "width")),
            _fmt(res.pooled(m, "is")),
        ]
        for m in methods
    ]
    headers = ["Interval", "Coverage %", "Below %", "Above %", "Mean width (MW)", "IS80 (MW)"]
    return _md_table(headers, rows)


DIFFERENCES = (
    (HEADLINE, PM, "is"),
    (HEADLINE, PS, "is"),
    (HEADLINE, RM, "is"),
    (CONFORMAL_C, PM, "is"),
    (HEADLINE, PM, "width"),
    (HEADLINE, PM, "cov"),
)


def difference_rows(res: Results) -> list[list[str]]:
    rows = []
    for a, b, stat in DIFFERENCES:
        day = res.difference(a, b, stat)
        week = res.difference(a, b, stat, blocks=res.weeks())
        scale, digits, unit = (100.0, 2, " pp") if stat == "cov" else (1.0, 1, " MW")
        name = {"is": "IS80", "width": "Mean width", "cov": "Coverage"}[stat]
        rows.append(
            [
                name,
                SHORT[a],
                SHORT[b],
                f"{signed(scale * day.value, digits)}{unit}",
                ci_text(day, scale, digits),
                ci_text(week, scale, digits),
                f"{day.n_pairs:,}",
                f"{day.n_blocks}",
            ]
        )
    return rows


def quarter_table(res: Results) -> str:
    methods = (PM, HEADLINE, CONFORMAL_C)
    headers = ["Issue quarter (NEM time)"]
    headers += [f"{SHORT[m]} coverage %" for m in methods]
    headers += [f"{SHORT[m]} IS80" for m in methods]
    headers.append("Pairs")
    cov = {m: res.by_quarter(m, "cov") for m in methods}
    isv = {m: res.by_quarter(m, "is") for m in methods}
    pairs = res.quarter_pairs()
    rows = []
    for q, name in enumerate(res.quarters):
        row = [name, *(pct(cov[m][q]) for m in methods), *(_fmt(isv[m][q]) for m in methods)]
        rows.append([*row, f"{int(pairs[q]):,}"])
    return _md_table(headers, rows)


def availability_table(c: Comparison) -> str:
    n = len(c.issue_times)
    rows = []
    for k in TABLE_PERIODS:
        rows.append(
            [
                period_label(k),
                f"{pct(float(c.available[PM][:, k - 1].sum()) / n)}%",
                f"{pct(float(c.available[PS][:, k - 1].sum()) / n)}%",
                f"{pct(float(c.mask[:, k - 1].sum()) / n)}%",
            ]
        )
    headers = ["Period ends", "AEMO (matched) available", "AEMO (strict) available", "Scored"]
    return _md_table(headers, rows)


def _provenance(res: Results, cfg: AppConfig, region_id: str, report_sha: str | None) -> str:
    c = res.c
    run = c.runs[HEADLINE]
    first = c.issue_times[0].tz_convert(NEM_TZ)
    last = c.issue_times[-1].tz_convert(NEM_TZ)
    minutes = np.round(c.strict_lag.dt.total_seconds().to_numpy() / 60).astype(int)
    values, counts = np.unique(minutes, return_counts=True)
    lag_text = ", ".join(f"{int(n):,} × {int(v)} min" for v, n in zip(values, counts, strict=True))
    run_ids = ", ".join(f"{SHORT[m]} {c.runs[m].run_id}" for m in OURS)
    rows = [
        ["Region", region_id],
        ["Our intervals", f"half-hour batch `{run.batch_id}` (runs {run_ids})"],
        ["Built from commit", f"`{run.git_sha}`"],
        [
            "Interval recipe",
            f"`{cfg.intervals.frozen_config_hash}` = frozen (`intervals.frozen_config_hash`); "
            f"QR recipe `{cfg.intervals.frozen_qr_config_hash}` = frozen; frozen code at "
            f"`{cfg.intervals.frozen_commit}`",
        ],
        [
            "Issue times compared (NEM time)",
            f"{first:%Y-%m-%d %H:%M} → {last:%Y-%m-%d %H:%M}, {len(c.issue_times):,}, every 30 min",
        ],
        ["AEMO PDPASA `LOR` runs in the window", f"{c.poe_runs:,} of {c.expected_runs:,}"],
        ["Strict alignment: run used", f"{lag_text} before the issue time"],
        [
            "Stored metrics recomputed",
            f"{len(c.checked)} pooled 80% statistics of {len(c.runs)} runs (all equal)",
        ],
        ["Report generated from commit", f"`{report_sha}`"],
    ]
    return _md_table(["Item", "Value"], rows, align="ll")


def _offset_text(c: Comparison) -> str:
    m = c.mask
    offset = (c.median[PM] - c.median[RM])[m]
    half = ((c.bounds[PM][1] - c.bounds[PM][0]) / 2)[m]
    centre = c.median[PM][m]
    return (
        f"- **Symmetric:** {pct(c.symmetric_share)}% of PDPASA `LOR` rows in the window have "
        f"POE10 − POE50 equal to POE50 − POE90 within {SYMMETRY_MW:g} MW.\n"
        f"- **Half-width:** {_fmt(float(half.mean()))} MW on the compared pairs on average, "
        f"{pct(float((half / centre).mean()), 2)}% of `DEMAND50`.\n"
        f"- **Offset:** `DEMAND50` sits {signed(float(offset.mean()))} MW above the matched "
        f"pre-dispatch `TOTALDEMAND` on average (standard deviation "
        f"{_fmt(float(offset.std()))} MW), as step 0 found. Neither the documentation nor "
        "the data say why, so the data cannot tell a definition difference from a different "
        "forecast (docs/m3_data_spike.md §4).\n"
        f"- **Re-centring uses earlier information:** the pre-dispatch run with the same "
        f"nominal run time was published after the PDPASA run for {c.recentring_after} of the "
        "runs used."
    )


def _summary(res: Results) -> str:
    cov_pm, cov_cqr = res.pooled(PM, "cov"), res.pooled(HEADLINE, "cov")
    is_pm, is_cqr = res.pooled(PM, "is"), res.pooled(HEADLINE, "is")
    w_pm, w_cqr = res.pooled(PM, "width"), res.pooled(HEADLINE, "width")
    d = res.difference(HEADLINE, PM, "is")
    better = res.per_lead(HEADLINE, "is") < res.per_lead(PM, "is")
    n_better = int(better.sum())
    verdict = (
        "lower than" if d.hi < 0 else "higher than" if d.lo > 0 else "not clearly different from"
    )
    return (
        f"- **Coverage:** AEMO's band (matched) contains {pct(cov_pm)}% of actual 30-minute "
        f"demand; our CQR 80% interval contains {pct(cov_cqr)}%. An 80% interval should "
        "contain 80%.\n"
        f"- **Width:** AEMO's band is {_fmt(w_pm)} MW wide on average, CQR's {_fmt(w_cqr)} MW.\n"
        f"- **Interval score** (lower is better; width plus a penalty for each miss): AEMO "
        f"{_fmt(is_pm)} MW, CQR {_fmt(is_cqr)} MW, so CQR's is {verdict} AEMO's "
        f"({signed(d.value)} MW, 95% interval {ci_text(d)} MW). CQR scores better at "
        f"{n_better} of {N_PERIODS} lead times."
    )


INTRO = (
    "AEMO publishes probabilistic demand forecasts in pre-dispatch PASA (`PDPASA_REGIONSOLUTION`): "
    "for every 30-minute period up to the end of the next trading day, demand with a 10%, 50% "
    "and 90% probability of exceedance (POE). Its POE90–POE10 range is an 80% band. This "
    "report compares that band with GridCast's 80% prediction intervals over the development "
    "part of the benchmark window. AEMO's band is a benchmark only, never a model input, and "
    "the comparison is not pass/fail (m3_plan.md §3). The interval recipe was frozen before "
    "AEMO's band was loaded for this comparison, so nothing here can change it. The holdout "
    "part (Oct 2025 → Sep 2026) is compared once, in step 6."
)

ALIGNMENT = (
    "- **Our intervals** are issued every 30 minutes. AEMO's band is for 30-minute means, and "
    "the mean of six 5-minute quantiles is not a quantile of the mean, so the comparison uses "
    "our intervals recalibrated on half-hour periods: the frozen recipe (calibration windows, "
    "time-of-day blocks, ACI, per-retrain selection) applied to the period means of the "
    "actuals, the point forecasts and the QR quantiles. A period counts only if all six of its "
    "5-minute pairs are scored. Lead time runs from our issue time to the end of the period.\n"
    "- **AEMO (matched)**, the headline: the PDPASA run whose nominal run time is our issue "
    "time, so both use information up to the same moment; AEMO publishes it 4–7 minutes "
    "later. A run's nominal time is 30 minutes before its `RUN_DATETIME` (step 0).\n"
    "- **AEMO (strict)**: the latest run already published at our issue time, normally the "
    "one from 30 minutes earlier.\n"
    "- **Re-centred** (sensitivity): the same half-width around the `TOTALDEMAND` of the "
    "pre-dispatch run with the same nominal run time, instead of `DEMAND50`.\n"
    "- Every interval is scored on the **same (issue time, period) pairs**: our scored periods "
    "where all four AEMO bands exist. Runs end at the end of the next trading day, so long "
    "leads cover fewer issue times:"
)

NOTES = (
    "- **Different products.** Pre-dispatch PASA assesses system adequacy (the `LOR` run "
    "checks for lack of reserve). The data model describes its demand columns only as 10%, "
    "50% and 90% POE forecasts (step 0); reading POE90–POE10 as an 80% band is this "
    "project's choice (m3_plan.md §2.6). Its coverage and interval score are facts about the "
    "band as published, not a judgement of what AEMO built it for.\n"
    '- **What is "actual".** The target is the 30-minute mean of '
    "`DISPATCHREGIONSUM.TOTALDEMAND`, as in M2's AEMO comparison.\n"
    "- **Choices made before any result.** The alignments, the re-centred sensitivity and the "
    "half-hour recipe were fixed before this comparison existed (m3_plan.md §2.6 and step 5). "
    "Step 0 looked at four sampled months of the band (Oct 2024, Jan 2025, Jul 2025, Sep "
    "2026) to decide whether to include it.\n"
    "- **Our development numbers** come from settings re-selected at every retrain from past "
    "data only, but the recipe was designed after M2's point errors had been seen "
    "(m3_plan.md §3). The holdout, scored once, is the unbiased comparison."
)


def build_report(engine: Engine, cfg: AppConfig, region_id: str) -> str:
    c = load_comparison(engine, cfg, region_id)
    res = score(c, cfg.ml.seed)
    figures = cfg.resolve(cfg.paths.reports_dir) / "figures"
    title = f"VIC1 demand, 30-minute periods, same pairs ({TITLE_PERIOD})"
    _plot(
        res, "cov", figures / FIGURES[0], "Share of actuals inside (%)", f"80%: coverage. {title}"
    )
    _plot(res, "is", figures / FIGURES[1], "Mean interval score (MW)", f"IS80. {title}")
    diff_headers = [
        "Statistic", "A", "B", "A − B", "95% interval (day blocks)",
        "95% interval (7-day blocks)", "Pairs", "Days",
    ]  # fmt: skip
    parts = [
        "# Milestone 3: our intervals against AEMO's POE band (development part)",
        "*Generated by `gridcast report-m3-aemo` from the stored half-hour interval batch and "
        "AEMO's archived PDPASA runs. Do not edit by hand.*",
        INTRO,
        "## Summary",
        _summary(res),
        "## Provenance",
        _provenance(res, cfg, region_id, git_sha()),
        "## How the intervals are lined up",
        ALIGNMENT,
        availability_table(c),
        "## Coverage",
        f"![80% coverage by lead time]({'figures/' + FIGURES[0]})",
        "### Share of actual demand inside the 80% interval (%)",
        coverage_table(res),
        "*Best baseline:* the conformal baseline with the lowest IS80 at that lead (pooled row: "
        "lowest pooled IS80).",
        "### Misses below and above (%)",
        "A calibrated 80% interval misses 10% of the time on each side.",
        miss_table(res),
        "## Width and interval score",
        f"![IS80 by lead time]({'figures/' + FIGURES[1]})",
        "### Mean width (MW)",
        width_table(res),
        "### Interval score IS80 (MW)",
        "IS80 = width + 10 × the distance by which the actual falls outside the interval "
        "(Gneiting & Raftery 2007); lower is better.",
        is_table(res),
        "## Pooled differences, with uncertainty",
        f"Block bootstrap ({BOOTSTRAP_DRAWS:,} draws) over whole NEM issue days, and over "
        "consecutive 7-day blocks as a sensitivity check (misses cluster in weather spells).",
        _md_table(diff_headers, difference_rows(res)),
        "## Sensitivity: the re-centred band and the strict alignment",
        "All periods pooled, same pairs.",
        sensitivity_table(res),
        "## By quarter of the issue time",
        quarter_table(res),
        "## What AEMO's band is",
        _offset_text(c),
        "## Notes and limitations",
        NOTES,
    ]
    return "\n\n".join(parts) + "\n"


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m3_aemo.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id))
    return path

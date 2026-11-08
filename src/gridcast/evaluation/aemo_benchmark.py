"""AEMO benchmark (M2 step 5): the frozen model's development forecasts against AEMO's own.

Generates ``reports/m2_aemo.md`` and one figure. Nothing is typed by hand and nothing is
pass/fail (m2_plan.md §3): AEMO's forecasts are a benchmark, never a model input.

* **Our forecasts** are the stored development predictions of the frozen config's ``main``
  batch (``backtest.frozen_ml_config_hash``); the command refuses any other batch.
* **Issue times** *o* run from ``aemo.first_issue`` up to ``backtest.holdout_start`` (NEM-time
  days), split by issue time like the M2 report. Holdout issue times are never read.
* **30-minute pre-dispatch** is compared on 30-minute means: our 5-minute forecasts for
  h = 6k − 5 … 6k and the actual demand averaged over the period ending o + 30k min,
  k = 1 … 48. Alignment (m2_plan.md §2.6, decided before any result):

  - *matched* (headline): AEMO's run whose nominal run time is *o*, i.e. the same
    information cut-off as ours; it is published one to fourteen minutes after *o*;
  - *strict*: the latest run published (``LASTCHANGED``) by *o*, normally the run from
    *o* − 30 min.

  Every forecast is scored on the same pairs: those where both AEMO alignments exist.
* **5-minute pre-dispatch** is compared at h = 1 … 11 with the run at *o*, which is
  published before *o*; a run published later is excluded and counted.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.m2_results import (
    actual_path,
    artifact_path,
    load_evaluation_set,
    run_config,
)
from gridcast.evaluation.report import TargetResults, _fmt, _md_table, model_label
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

HEADLINE = "lgbm_C"
COMPARED = ("lgbm_C", "lgbm_A", "lgbm_D", "ridge_C", "persistence", "daily_naive", "weekly_naive")
BASELINES = ("persistence", "daily_naive", "weekly_naive")
MATCHED, STRICT, P5 = "aemo_matched", "aemo_strict", "aemo_p5min"
AEMO_LABELS = {
    MATCHED: "AEMO (matched)",
    STRICT: "AEMO (strict)",
    P5: "AEMO 5-min pre-dispatch",
}
PERIOD_STEPS = 6  # 5-minute steps in a 30-minute period
N_PERIODS = 48  # 24 h
P5_STEPS = 11  # 5-minute pre-dispatch covers h = 1 ... 11
TABLE_PERIODS = (1, 2, 4, 8, 12, 24, 36, 48)  # 30 min, 1, 2, 4, 6, 12, 18, 24 h
BOOTSTRAP_DRAWS = 2000
FIGURE = "m2_aemo_mae_by_lead.png"
# Baselines stay in the tables: persistence peaks near 1,300 MW and would flatten the
# comparison the figure is for. Green/red pass the palette validator (CVD warn band, so the
# lines are direct-labelled and strict is dashed).
PLOT = (HEADLINE, MATCHED, STRICT)
AEMO_COLOR = "#e34948"  # categorical slot 8 (unused by the project); strict is dashed


def _at(frame: pd.DataFrame, row: int, col: str) -> float:
    return float(frame[col].to_numpy(dtype=np.float64)[frame.index.get_loc(row)])


def label(name: str) -> str:
    return AEMO_LABELS.get(name) or model_label(name)


# --------------------------------------------------------------------------- alignment


def _naive_ns(values: Any) -> np.ndarray:
    """UTC timestamps (Series or index, tz-aware) as naive datetime64[ns]; NaT stays NaT."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").to_numpy()


def thirty_minute_means(wide: np.ndarray) -> np.ndarray:
    """(n, 288) 5-minute values -> (n, 48) means of h = 6k-5 ... 6k (NaN if any is NaN)."""
    n, h = wide.shape
    if h != PERIOD_STEPS * N_PERIODS:
        raise ValueError(f"expected {PERIOD_STEPS * N_PERIODS} horizons, got {h}")
    return wide.reshape(n, N_PERIODS, PERIOD_STEPS).mean(axis=2)


def strict_runs(runs: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each issue time, the latest run (by run time) published at or before it.

    ``runs`` has one row per run: ``run_time``, ``available_at``. NaT where none is.
    """
    by_pub = runs.sort_values("available_at").reset_index(drop=True)
    by_pub["latest_run"] = by_pub["run_time"].cummax()
    merged = pd.merge_asof(
        pd.DataFrame({"o": issue_times}),
        by_pub[["available_at", "latest_run"]],
        left_on="o",
        right_on="available_at",
        direction="backward",
    )
    return pd.DatetimeIndex(merged["latest_run"])


def aligned(
    forecasts: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    chosen_runs: pd.DatetimeIndex,
    step_minutes: int,
    n_leads: int,
) -> np.ndarray:
    """(n_issue, n_leads): the chosen run's forecast for the target ending o + k·step.

    ``forecasts`` has ``run_time``, ``target_end``, ``total_demand_mw``. Missing runs
    (NaT) or targets beyond a run's horizon give NaN.
    """
    lookup = pd.Series(
        forecasts["total_demand_mw"].to_numpy(dtype=np.float64),
        index=pd.MultiIndex.from_arrays(
            [_naive_ns(forecasts["run_time"]), _naive_ns(forecasts["target_end"])]
        ),
    )
    if not lookup.index.is_unique:
        raise ValueError("duplicate (run_time, target_end) in AEMO forecasts")
    never = np.datetime64("1900-01-01T00:00", "ns")
    runs = np.where(pd.isna(chosen_runs), never, _naive_ns(chosen_runs))
    leads = (np.arange(1, n_leads + 1) * step_minutes).astype("timedelta64[m]")
    targets = _naive_ns(issue_times)[:, None] + leads[None, :]
    keys = pd.MultiIndex.from_arrays([np.repeat(runs, n_leads), targets.ravel()])
    pos = lookup.index.get_indexer(keys)
    values = np.where(pos >= 0, lookup.to_numpy()[np.maximum(pos, 0)], np.nan)
    return values.reshape(len(issue_times), n_leads)


# --------------------------------------------------------------------------- loading


@dataclass(slots=True)
class Benchmark:
    """Everything the report needs, on numpy grids indexed by issue time."""

    issue_times: pd.DatetimeIndex
    batch: TargetResults
    # 30-minute comparison: (n_issue, 48)
    ours30: dict[str, np.ndarray]
    actual30: np.ndarray
    aemo30: dict[str, np.ndarray]  # MATCHED, STRICT
    # 5-minute comparison: (n_issue, 11)
    ours5: dict[str, np.ndarray]
    actual5: np.ndarray
    p5: np.ndarray
    p5_late: int  # issue times whose 5-minute run was published after o (excluded)
    p5_missing: int  # issue times with no 5-minute run at o
    runs: pd.DataFrame  # source, run_time, available_at (within the window)
    strict_lag: pd.Series  # o - strict run time, per issue time with a strict run

    @property
    def common30(self) -> np.ndarray:
        mask = np.isfinite(self.actual30)
        for a in (*self.ours30.values(), *self.aemo30.values()):
            mask &= np.isfinite(a)
        return mask

    @property
    def common5(self) -> np.ndarray:
        mask = np.isfinite(self.actual5) & np.isfinite(self.p5)
        for a in self.ours5.values():
            mask &= np.isfinite(a)
        return mask


def window(cfg: AppConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[first, end) issue times in UTC: aemo.first_issue up to the holdout start."""
    first = nem_midnight_utc(str(cfg.aemo.first_issue))
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    if not first < end:
        raise ValueError("aemo.first_issue must be before backtest.holdout_start")
    return first, end


def frozen_main_batch(engine: Engine, cfg: AppConfig, region_id: str) -> TargetResults:
    """The latest complete ``main`` batch; it must be the frozen config's."""
    frozen = cfg.backtest.frozen_ml_config_hash
    if frozen is None:
        raise LookupError("no frozen ML config (backtest.frozen_ml_config_hash)")
    res = load_evaluation_set(engine, region_id, "main")
    stored = str(run_config(res, HEADLINE)["config_hash"])
    if stored != frozen:
        raise LookupError(
            f"latest main batch {res.batch_id} has config {stored}, not the frozen {frozen}"
        )
    return res


def _read_wide(path: Path, first: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    times = pd.DatetimeIndex(frame["origin_time"])
    return frame.loc[(times >= first) & (times < end)].reset_index(drop=True)


def load_aemo(
    engine: Engine, region_id: str, source: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT run_time, target_end, available_at, total_demand_mw
                FROM aemo_demand_forecasts
                WHERE source = :s AND region_id = :r AND run_time BETWEEN :a AND :b
                ORDER BY run_time, target_end
                """
            ),
            conn,
            params={
                "s": source,
                "r": region_id,
                "a": first.to_pydatetime(),
                "b": last.to_pydatetime(),
            },
        )


def _one_per_run(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("run_time")["available_at"].max().reset_index()


def load_benchmark(engine: Engine, cfg: AppConfig, region_id: str) -> Benchmark:
    first, end = window(cfg)
    batch = frozen_main_batch(engine, cfg, region_id)
    data_root = cfg.data_root()
    actual = _read_wide(actual_path(batch, data_root), first, end)
    issue_times = pd.DatetimeIndex(actual["origin_time"])
    if len(issue_times) == 0:
        raise LookupError("no stored development issue times in the AEMO window")
    if issue_times.max() >= end:
        raise AssertionError("holdout issue times must never be read")
    cols = [c for c in actual.columns if c != "origin_time"]
    y = actual[cols].to_numpy(dtype=np.float64)
    ours: dict[str, np.ndarray] = {}
    for model in COMPARED:
        frame = _read_wide(artifact_path(batch, model, data_root), first, end)
        if not pd.DatetimeIndex(frame["origin_time"]).equals(issue_times):
            raise ValueError(f"{model}: issue times differ from the actuals")
        ours[model] = frame[cols].to_numpy(dtype=np.float64)

    # Pre-dispatch runs up to 24 h before the first issue time cover its strict alignment.
    pd_fc = load_aemo(engine, region_id, "predispatch", first - pd.Timedelta(days=1), end)
    p5_fc = load_aemo(engine, region_id, "p5min", first - pd.Timedelta(hours=1), end)
    pd_runs, p5_runs = _one_per_run(pd_fc), _one_per_run(p5_fc)

    strict = strict_runs(pd_runs, issue_times)
    matched30 = aligned(pd_fc, issue_times, issue_times, 30, N_PERIODS)
    strict30 = aligned(pd_fc, issue_times, strict, 30, N_PERIODS)

    published = p5_runs.set_index("run_time")["available_at"].reindex(issue_times)
    late = (published.notna() & (published > published.index.to_series())).to_numpy()
    p5_choice = pd.DatetimeIndex(issue_times.where(~late))
    p5 = aligned(p5_fc, issue_times, p5_choice, 5, P5_STEPS)

    runs = pd.concat(
        [pd_runs.assign(source="predispatch"), p5_runs.assign(source="p5min")], ignore_index=True
    )
    runs = runs[(runs["run_time"] >= first) & (runs["run_time"] < end)]
    lag = pd.Series(issue_times - strict, index=issue_times).dropna()
    return Benchmark(
        issue_times=issue_times,
        batch=batch,
        ours30={m: thirty_minute_means(v) for m, v in ours.items()},
        actual30=thirty_minute_means(y),
        aemo30={MATCHED: matched30, STRICT: strict30},
        ours5={m: v[:, :P5_STEPS] for m, v in ours.items()},
        actual5=y[:, :P5_STEPS],
        p5=p5,
        p5_late=int(late.sum()),
        p5_missing=int(published.isna().sum()),
        runs=runs.reset_index(drop=True),
        strict_lag=lag,
    )


# --------------------------------------------------------------------------- metrics


def by_lead(
    forecasts: dict[str, np.ndarray], actual: np.ndarray, mask: np.ndarray, metric: str
) -> pd.DataFrame:
    """Per lead (column of the grids) over ``mask``: MAE or bias (forecast − actual)."""
    out = {}
    for name, f in forecasts.items():
        err = np.where(mask, f - actual, np.nan)
        with warnings.catch_warnings():  # a lead without pairs is NaN, not a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = np.nanmean(np.abs(err) if metric == "mae" else err, axis=0)
    frame = pd.DataFrame(out, index=pd.RangeIndex(1, actual.shape[1] + 1, name="lead"))
    frame["n"] = mask.sum(axis=0)
    return frame


def pooled_mae(
    forecasts: dict[str, np.ndarray], actual: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    return {n: float(np.abs(f - actual)[mask].mean()) for n, f in forecasts.items()}


@dataclass(frozen=True, slots=True)
class Difference:
    """Pooled MAE(a) − MAE(b) on the same pairs, with a day-block bootstrap 95% interval."""

    a: str
    b: str
    mae_a: float
    mae_b: float
    diff: float
    lo: float
    hi: float
    n_pairs: int
    n_days: int


def mae_difference(
    a: str,
    b: str,
    fa: np.ndarray,
    fb: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    days: np.ndarray,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> Difference:
    """Resampling whole issue days (with replacement) keeps within-day error correlation."""
    ea = np.where(mask, np.abs(fa - actual), 0.0).sum(axis=1)
    eb = np.where(mask, np.abs(fb - actual), 0.0).sum(axis=1)
    n = mask.sum(axis=1)
    frame = pd.DataFrame({"day": days, "ea": ea, "eb": eb, "n": n}).groupby("day").sum()
    frame = frame[frame["n"] > 0]
    sa, sb, sn = (frame[c].to_numpy(dtype=np.float64) for c in ("ea", "eb", "n"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frame), size=(draws, len(frame)))
    boot = (sa[idx].sum(axis=1) - sb[idx].sum(axis=1)) / sn[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    total = sn.sum()
    return Difference(
        a,
        b,
        sa.sum() / total,
        sb.sum() / total,
        (sa.sum() - sb.sum()) / total,
        float(lo),
        float(hi),
        int(total),
        len(frame),
    )


def lead_runs(better: Sequence[bool], leads: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous ranges of ``leads`` where ``better`` is true."""
    out: list[tuple[int, int]] = []
    for lead, ok in zip(leads, better, strict=True):
        if ok and out and out[-1][1] == lead - 1:
            out[-1] = (out[-1][0], lead)
        elif ok:
            out.append((lead, lead))
    return out


def period_label(k: int) -> str:
    minutes = 30 * k
    return f"{minutes} min" if minutes < 60 else f"{minutes / 60:g} h"


def step_label(h: int) -> str:
    return f"{5 * h} min"


def _ranges_text(ranges: Sequence[tuple[int, int]], fmt: Any) -> str:
    if not ranges:
        return "none"
    return ", ".join(fmt(lo) if lo == hi else f"{fmt(lo)}–{fmt(hi)}" for lo, hi in ranges)


# --------------------------------------------------------------------------- figure


def _figure(table: pd.DataFrame, path: Path) -> None:
    """MAE by lead (30-minute periods): AEMO solid (matched) and dashed (strict)."""
    from matplotlib.figure import Figure

    from gridcast.evaluation.report import AXIS, GRID, INK, INK_2, MODEL_COLORS, MUTED, SURFACE

    colors = {**MODEL_COLORS, MATCHED: AEMO_COLOR, STRICT: AEMO_COLOR}
    hours = table.index.to_numpy() * 0.5
    fig = Figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    for name in PLOT:
        ax.plot(
            hours,
            table[name],
            color=colors.get(name, INK),
            linewidth=2,
            linestyle=(0, (4, 2)) if name == STRICT else "-",
            label=label(name),
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    top = float(np.nanmax(table[list(PLOT)].to_numpy())) * 1.08
    ax.set_ylim(0, top)
    ax.set_xlim(0, 24)
    last = int(table[list(PLOT)].dropna().index.max())  # leads without pairs are left out
    ends = sorted((_at(table, last, n), n) for n in PLOT)
    placed: list[float] = []
    for y, name in ends:
        y_label = max(y, placed[-1] + top * 0.05) if placed else y
        placed.append(y_label)
        ax.annotate(
            label(name),
            xy=(last * 0.5, y),
            xytext=(24 * 1.02, y_label),
            color=INK_2,
            fontsize=8.5,
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6},
        )
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel(
        "Lead time to the end of the 30-minute period (hours after issue time)",
        color=INK_2,
        fontsize=9,
    )
    ax.set_ylabel("MAE of 30-minute mean demand (MW)", color=INK_2, fontsize=9)
    ax.set_title(
        "VIC1 demand, Oct 2024 → Sep 2025 (development): 30-minute periods, same pairs",
        color=INK,
        fontsize=11,
        loc="left",
        pad=30,
    )
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=8.5,
        ncol=len(PLOT),
        handlelength=2.4,
    )
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


# --------------------------------------------------------------------------- report


def _mae_table(table: pd.DataFrame, names: Sequence[str], pooled: dict[str, float]) -> str:
    headers = ["Lead", *[label(n) for n in names], "Best baseline", "Pairs"]
    baselines = table[list(BASELINES)]
    rows = []
    for k in TABLE_PERIODS:
        values = {m: _at(baselines, k, m) for m in BASELINES}
        best = min(BASELINES, key=lambda m: np.inf if np.isnan(values[m]) else values[m])
        rows.append(
            [
                period_label(k),
                *[_fmt(_at(table, k, n)) for n in names],
                f"{_fmt(values[best])} ({label(best)})" if np.isfinite(values[best]) else "–",
                f"{int(_at(table, k, 'n')):,}",
            ]
        )
    best_pooled = min(BASELINES, key=lambda m: pooled[m])
    rows.append(
        [
            "**All leads (pooled)**",
            *[f"**{_fmt(pooled[n])}**" for n in names],
            f"{_fmt(pooled[best_pooled])} ({label(best_pooled)})",
            "",
        ]
    )
    return _md_table(headers, rows)


def _difference_rows(diffs: Sequence[Difference]) -> str:
    rows = [
        [
            label(d.a),
            label(d.b),
            _fmt(d.mae_a),
            _fmt(d.mae_b),
            f"{d.diff:+,.1f}",
            f"{d.lo:+,.1f} to {d.hi:+,.1f}",
            f"{d.n_pairs:,}",
            f"{d.n_days:,}",
        ]
        for d in diffs
    ]
    headers = [
        "Forecast A",
        "Forecast B",
        "MAE A",
        "MAE B",
        "A − B",
        "95% interval",
        "Pairs",
        "Days",
    ]
    return _md_table(headers, rows, "llrrrrrr")


def _verdict(d: Difference) -> str:
    if d.hi < 0:
        return f"{label(d.a)} is more accurate (the interval excludes zero)"
    if d.lo > 0:
        return f"{label(d.b)} is more accurate (the interval excludes zero)"
    return "the difference is not distinguishable from zero (the interval includes it)"


def _quarter_table(b: Benchmark, mask: np.ndarray) -> str:
    nem = b.issue_times.tz_convert(NEM_TZ)
    quarter = np.asarray([f"{t.year} Q{(t.month - 1) // 3 + 1}" for t in nem])
    forecasts = {HEADLINE: b.ours30[HEADLINE], **b.aemo30}
    rows = []
    for q in sorted(set(quarter)):
        sel = mask & (quarter == q)[:, None]
        mae = pooled_mae(forecasts, b.actual30, sel)
        rows.append([q, *[_fmt(mae[n]) for n in forecasts], f"{int(sel.sum()):,}"])
    return _md_table(["Issue quarter (NEM time)", *[label(n) for n in forecasts], "Pairs"], rows)


def _provenance(b: Benchmark, cfg: AppConfig, region_id: str, report_sha: str | None) -> str:
    rc = run_config(b.batch, HEADLINE)
    run_sha = ", ".join(sorted({str(v) for v in b.batch.runs["git_sha"]}))
    nem = b.issue_times.tz_convert(NEM_TZ)
    runs = b.runs.groupby("source").size()
    first, end = window(cfg)
    days = (end - first) / pd.Timedelta(days=1)
    rows = [
        ["Region", region_id],
        ["Our forecasts", f"main batch `{b.batch.batch_id}` (commit `{run_sha}`)"],
        ["ML config", f"`{rc['config_hash']}` = frozen config (`backtest.frozen_ml_config_hash`)"],
        [
            "Issue times compared (NEM time)",
            f"{nem.min():%Y-%m-%d %H:%M} → {nem.max():%Y-%m-%d %H:%M}, {len(nem):,}, every 30 min",
        ],
        [
            "AEMO pre-dispatch runs in the window",
            f"{runs.get('predispatch', 0):,} of {int(days * 48):,} (48 a day)",
        ],
        [
            "AEMO 5-minute pre-dispatch runs in the window",
            f"{runs.get('p5min', 0):,} of {int(days * 288):,} (288 a day)",
        ],
        [
            "Strict alignment: run used",
            ", ".join(
                f"{int(n):,} × {int(lag.total_seconds() // 60)} min before the issue time"
                for lag, n in zip(
                    pd.to_timedelta(b.strict_lag.value_counts().sort_index().index),
                    b.strict_lag.value_counts().sort_index().to_numpy(),
                    strict=True,
                )
            ),
        ],
        [
            "5-minute run at the issue time missing / published after it",
            f"{b.p5_missing:,} / {b.p5_late:,} issue times (excluded)",
        ],
        ["Report generated from commit", f"`{report_sha}`"],
    ]
    return _md_table(["Item", "Value"], rows, "ll")


def build_report(engine: Engine, cfg: AppConfig, region_id: str) -> str:
    report_sha = git_sha()  # before figures are written, so the tree is still clean
    b = load_benchmark(engine, cfg, region_id)
    seed = cfg.ml.seed
    days = np.asarray(b.issue_times.tz_convert(NEM_TZ).strftime("%Y-%m-%d"))

    mask30 = b.common30
    f30 = {**b.ours30, **b.aemo30}
    mae30 = by_lead(f30, b.actual30, mask30, "mae")
    bias30 = by_lead(f30, b.actual30, mask30, "bias")
    pooled30 = pooled_mae(f30, b.actual30, mask30)
    diffs30 = [
        mae_difference(
            HEADLINE, aemo, b.ours30[HEADLINE], b.aemo30[aemo], b.actual30, mask30, days, seed
        )
        for aemo in (MATCHED, STRICT)
    ]

    mask5 = b.common5
    f5 = {**b.ours5, P5: b.p5}
    mae5 = by_lead(f5, b.actual5, mask5, "mae")
    pooled5 = pooled_mae(f5, b.actual5, mask5)
    diff5 = mae_difference(HEADLINE, P5, b.ours5[HEADLINE], b.p5, b.actual5, mask5, days, seed)

    figure_path = cfg.resolve(cfg.paths.reports_dir) / "figures" / FIGURE
    _figure(mae30, figure_path)

    leads30 = list(mae30.index)
    loses_m = lead_runs(list(mae30[HEADLINE] >= mae30[MATCHED]), leads30)
    loses_s = lead_runs(list(mae30[HEADLINE] >= mae30[STRICT]), leads30)
    n_win_m = int((mae30[HEADLINE] < mae30[MATCHED]).sum())
    n_win_s = int((mae30[HEADLINE] < mae30[STRICT]).sum())
    leads5 = list(mae5.index)
    loses5 = lead_runs(list(mae5[HEADLINE] >= mae5[P5]), leads5)
    n_win5 = int((mae5[HEADLINE] < mae5[P5]).sum())

    coverage = pd.DataFrame(
        {
            name: np.isfinite(grid).mean(axis=0)
            for name, grid in (
                (MATCHED, b.aemo30[MATCHED]),
                (STRICT, b.aemo30[STRICT]),
            )
        },
        index=pd.RangeIndex(1, N_PERIODS + 1),
    )
    cover_rows = [
        [
            period_label(k),
            f"{100 * _at(coverage, k, MATCHED):.1f}%",
            f"{100 * _at(coverage, k, STRICT):.1f}%",
            f"{100 * mask30[:, k - 1].mean():.1f}%",
        ]
        for k in TABLE_PERIODS
    ]
    covered = coverage.index[coverage[MATCHED] >= 0.999]
    full_m = int(covered.max()) if len(covered) else 0

    ours_cols = [HEADLINE, "lgbm_A", "lgbm_D", "ridge_C"]
    names30 = [MATCHED, STRICT, *ours_cols]
    bias_rows = [
        [period_label(k), *[f"{_at(bias30, k, n):+,.1f}" for n in (MATCHED, STRICT, HEADLINE)]]
        for k in TABLE_PERIODS
    ]
    p5_rows = [
        [
            step_label(h),
            _fmt(_at(mae5, h, P5)),
            _fmt(_at(mae5, h, HEADLINE)),
            _fmt(_at(mae5, h, "persistence")),
            f"{int(_at(mae5, h, 'n')):,}",
        ]
        for h in leads5
    ]
    p5_rows.append(
        [
            "**All (pooled)**",
            f"**{_fmt(pooled5[P5])}**",
            f"**{_fmt(pooled5[HEADLINE])}**",
            f"**{_fmt(pooled5['persistence'])}**",
            "",
        ]
    )

    hours = lambda k: period_label(k)  # noqa: E731
    sections = [
        "# Milestone 2: AEMO benchmark (development part)",
        "",
        "*Generated by `gridcast report-aemo` from the stored development forecasts of the "
        "frozen model and AEMO's archived forecasts. Do not edit by hand.*",
        "",
        "AEMO publishes its own demand forecasts: **30-minute pre-dispatch** (every half hour, "
        "up to the end of the next trading day) and **5-minute pre-dispatch** (every 5 minutes, "
        "the next hour). This report compares them with GridCast's frozen headline model, "
        f"{label(HEADLINE)} (archived GFS temperature forecasts), over the development part "
        "of the benchmark window. AEMO's forecasts are a benchmark only, never a model input, "
        "and the comparison is not pass/fail (m2_plan.md §3). The model config was frozen "
        "before this comparison existed, so nothing here can change it. The holdout part "
        "(Oct 2025 → Sep 2026) is compared once, in step 6.",
        "",
        "## Provenance",
        "",
        _provenance(b, cfg, region_id, report_sha),
        "",
        "## How the forecasts are lined up",
        "",
        "- **Our forecasts** are issued every 30 minutes for every 5 minutes of the next 24 h. "
        "For the 30-minute comparison they are averaged over each period, because AEMO's "
        "pre-dispatch `TOTALDEMAND` is the period mean (step 0). The actual demand is averaged "
        "the same way. Lead time is measured from our issue time to the end of the period.",
        "- **AEMO (matched)**, the headline: AEMO's run whose nominal run time is our issue "
        "time. Both forecasts then use information up to the same moment; AEMO publishes its "
        "run one to fourteen minutes later.",
        "- **AEMO (strict)**: the latest run already published at our issue time, normally the "
        "run from 30 minutes earlier. This is what an operator could have read at that moment, "
        "but AEMO's information is half an hour older than ours, so it flatters our model.",
        "- **AEMO 5-minute pre-dispatch**: the run at our issue time, published before it; "
        "its intervals 1–11 are our leads of 5–55 minutes.",
        "- Every forecast in a table is scored on the **same (issue time, lead) pairs**: "
        "those where both AEMO alignments have a value. Pre-dispatch does not always reach "
        "24 h (runs issued from 04:00 to 12:00 NEM time end at the next 04:00), so long leads "
        "cover fewer issue times:",
        "",
        _md_table(
            ["Lead", "AEMO (matched) available", "AEMO (strict) available", "Pairs scored"],
            cover_rows,
        ),
        "",
        f"AEMO (matched) covers every issue time up to {hours(full_m)} ahead. Beyond that the "
        "scored pairs come from fewer issue times (those whose AEMO run reaches that far), "
        "so the shape of every curve there partly reflects which issue times remain.",
        "",
        "## 30-minute pre-dispatch vs our forecasts",
        "",
        f"![MAE by lead time, 30-minute periods](figures/{FIGURE})",
        "",
        "### MAE (MW) of 30-minute mean demand",
        "",
        _mae_table(mae30, names30, pooled30),
        "",
        f"- {label(HEADLINE)} has a lower MAE than AEMO (matched) at {n_win_m} of "
        f"{len(leads30)} leads; AEMO (matched) is as good or better at: "
        f"{_ranges_text(loses_m, hours)}.",
        f"- Against AEMO (strict), {label(HEADLINE)} is lower at {n_win_s} of {len(leads30)} "
        f"leads; AEMO (strict) is as good or better at: {_ranges_text(loses_s, hours)}.",
        "",
        "### Pooled difference, with uncertainty",
        "",
        "Pooled over every lead on the same pairs. The 95% interval comes from resampling "
        f"whole issue days ({BOOTSTRAP_DRAWS:,} bootstrap draws), which keeps the strong "
        "correlation between errors within a day.",
        "",
        _difference_rows([*diffs30, diff5]),
        "",
        *[f"- {label(d.a)} vs {label(d.b)}: {_verdict(d)}." for d in (*diffs30, diff5)],
        "",
        "### Bias (MW, forecast − actual)",
        "",
        _md_table(["Lead", label(MATCHED), label(STRICT), label(HEADLINE)], bias_rows),
        "",
        "### By quarter of the issue time (all leads pooled, MAE in MW)",
        "",
        _quarter_table(b, mask30),
        "",
        "## 5-minute pre-dispatch vs our forecasts (leads 5–55 minutes)",
        "",
        _md_table(["Lead", label(P5), label(HEADLINE), label("persistence"), "Pairs"], p5_rows),
        "",
        f"- {label(HEADLINE)} has a lower MAE than AEMO 5-minute pre-dispatch at {n_win5} of "
        f"{len(leads5)} leads; AEMO is as good or better at: {_ranges_text(loses5, step_label)}.",
        "",
        "## Notes and limitations",
        "",
        "- **Different purposes.** Pre-dispatch is an operational market forecast that AEMO "
        "re-runs every half hour with the latest data and bids; it is not built to minimise "
        "the error of a 24-hour-ahead demand forecast. GridCast's model is.",
        '- **What is "actual".** The target is `DISPATCHREGIONSUM.TOTALDEMAND`, AEMO\'s '
        "operational dispatch demand (step 0), not metered consumption. Interval 0 of every "
        "5-minute pre-dispatch run equals it, apart from a few small differences "
        "(`aemo_p5min_alignment`, docs/data_sources.md §6).",
        "- **Alignment was chosen before any result.** The plan first named only the strict "
        "alignment; matched became the headline on 2026-11-08, before this comparison was "
        "run, because strict gives AEMO older information than our model. Both are shown.",
        "- **The development period was looked at twice** when choosing between two "
        "development runs (`reports/m2_demand.md`, *Development runs*), so our development "
        "numbers are slightly optimistic. AEMO's are not. The holdout, scored once, is the "
        "unbiased comparison.",
        "- **Missing runs.** AEMO runs missing from the archive (data-quality check "
        "`aemo_missing_runs`) leave their issue times out of the matched comparison; the "
        "strict alignment falls back to the latest published run.",
        "",
    ]
    return "\n".join(sections)


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m2_aemo.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id))
    return path

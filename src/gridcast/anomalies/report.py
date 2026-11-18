"""Generate ``reports/m4_anomalies.md`` (+ figures) from the stored development run (M4 step 4).

Every number comes from the stored run: the per-interval parquet, the injections, the forest
ranges, the cut checks and the events in ``anomalies``. Before anything is written, the report
recomputes what the run recorded and stops on any difference (:class:`ReportMismatch`):

* every calibrated detector's tier from its stored p-value and scored-period flag;
* the price lines from the stored prices;
* the events, merged again from those tiers by the frozen rule, against the stored events;
* the alert and event counts against the run's record.

Criteria A1–A5 are computed as pre-registered (m4_plan.md §3), with the readings fixed in §1
("How `report-m4` reads the criteria"). A5 runs the synthetic causality suite itself. E1 is
reported for the demand-peak explanations; the global SHAP sample is step 5.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sqlalchemy import Engine, text

from gridcast.anomalies.detectors import (
    CATEGORIES,
    DEMAND_LEVEL,
    DETECTORS,
    ISOLATION_FOREST,
    LINE_NOTABLE,
    LINE_SEVERE,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
)
from gridcast.anomalies.events import Event, merge_events
from gridcast.anomalies.evidence import banned_in, missing_fields
from gridcast.anomalies.freeze import freeze_state
from gridcast.anomalies.pvalues import tiers as tiers_from_p
from gridcast.anomalies.recipe import case_studies, dev_split
from gridcast.anomalies.reference import CLASS_O, CLASS_P, ReferenceEvent, reference_events
from gridcast.anomalies.run import Split, gap_intervals
from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.database.queries import load_market_frame
from gridcast.evaluation.m2_report import Criterion, criteria_table, parse_pytest_summary
from gridcast.evaluation.report import AXIS, GRID, INK, INK_2, MUTED, SURFACE, _md_table, money
from gridcast.provenance import git_sha
from gridcast.timeutils import INTERVAL, NEM_TZ

REPORT_NAME = "m4_anomalies.md"
CAUSALITY_TESTS = ("tests/unit/test_anomaly_causality.py",)

NAMES = {
    DEMAND_LEVEL: "D1 demand level",
    RESIDUAL_SHORT: "D2 demand residual, 5–30 min",
    RESIDUAL_DAYAHEAD: "D3 demand residual, 23.5–24 h",
    PRICE_LEVEL: "D4 price level",
    PRICE_VOLATILITY: "D5 price volatility",
    ISOLATION_FOREST: "D6 Isolation Forest",
    LINE_NOTABLE: "L1 price ≥ $300",
    LINE_SEVERE: "L2 price ≥ $5,000",
}
SHORT = {d: NAMES[d].split(" ", 1)[0] for d in NAMES}
# Categorical slots 1-6 of the validated palette (fixed order; colour follows the detector).
COLORS = dict(
    zip(DETECTORS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"), strict=True)
)
A1_BANDS = ((0.01, 0.005, 0.02), (0.001, 0.0005, 0.002))  # (level, low, high)
A3_POWER = 0.9
# A3's checks: (detector, size) at the largest pre-registered size of each sign (§3).
A3_CHECKS = (
    (DEMAND_LEVEL, -2000.0),
    (DEMAND_LEVEL, 2000.0),
    (RESIDUAL_SHORT, -1000.0),
    (RESIDUAL_SHORT, 1000.0),
    (RESIDUAL_DAYAHEAD, -2000.0),
    (RESIDUAL_DAYAHEAD, 2000.0),
    (PRICE_LEVEL, 15000.0),
    (PRICE_LEVEL, -1000.0),
    (PRICE_VOLATILITY, 5000.0),
    (ISOLATION_FOREST, -2000.0),
    (ISOLATION_FOREST, 2000.0),
)
UNITS = {
    DEMAND_LEVEL: "MW added",
    RESIDUAL_SHORT: "MW added",
    RESIDUAL_DAYAHEAD: "MW added",
    PRICE_LEVEL: "price set to, $/MWh",
    PRICE_VOLATILITY: "± amplitude, $/MWh",
    ISOLATION_FOREST: "MW added",
}


class ReportMismatch(RuntimeError):
    """A recomputed value differs from the stored run; nothing is written."""


# --------------------------------------------------------------------------- the stored run


@dataclass(slots=True)
class StoredRun:
    run_id: int
    batch_id: str
    git_sha: str | None
    config: dict[str, Any]
    directory: Path
    intervals: pd.DataFrame
    injections: pd.DataFrame
    forests: pd.DataFrame
    cuts: pd.DataFrame
    events: pd.DataFrame  # one row per stored event, in time order


RUN_SQL = """
SELECT run_id, batch_id::text AS batch_id, git_sha, config, artifact_path
FROM model_runs
WHERE model_kind = 'anomaly' AND region_id = :region AND status = 'success'
  AND config->>'split' = 'dev' AND config->>'anomaly_config_hash' = :hash
"""
EVENTS_SQL = """
SELECT first_interval_end, last_interval_end, peak_interval_end, tier, detectors, min_p,
       categories, lines_only, evidence, summary
FROM anomalies WHERE run_id = :run ORDER BY first_interval_end
"""


def load_run(engine: Engine, cfg: AppConfig, region_id: str) -> StoredRun:
    frozen = cfg.anomalies.frozen_config_hash
    if not frozen:
        raise ReportMismatch("no freeze recorded (anomalies.frozen_config_hash)")
    with engine.connect() as conn:
        rows = conn.execute(text(RUN_SQL), {"region": region_id, "hash": frozen}).all()
        if len(rows) != 1:
            raise ReportMismatch(
                f"expected one development run for hash {frozen}, found {len(rows)}"
            )
        row = rows[0]
        events = pd.read_sql(text(EVENTS_SQL), conn, params={"run": row.run_id})
    directory = Path(row.artifact_path)
    for c in ("first_interval_end", "last_interval_end", "peak_interval_end"):
        events[c] = pd.to_datetime(events[c], utc=True)
    return StoredRun(
        run_id=int(row.run_id),
        batch_id=str(row.batch_id),
        git_sha=row.git_sha,
        config=dict(row.config),
        directory=directory,
        intervals=pd.read_parquet(directory / "intervals.parquet"),
        injections=pd.read_parquet(directory / "injections.parquet"),
        forests=pd.read_parquet(directory / "forests.parquet"),
        cuts=pd.read_parquet(directory / "cuts.parquet"),
        events=events,
    )


def load_prices(engine: Engine, region_id: str, ends: pd.DatetimeIndex) -> np.ndarray:
    """Stored prices on the run's grid."""
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(
                "SELECT interval_end, rrp_aud_mwh FROM market_intervals "
                "WHERE region_id = :r AND interval_end >= :a AND interval_end <= :b "
                "ORDER BY interval_end"
            ),
            conn,
            params={"r": region_id, "a": ends[0], "b": ends[-1]},
        )
    got = pd.DatetimeIndex(pd.to_datetime(frame["interval_end"], utc=True))
    if not got.equals(ends):
        raise ReportMismatch("market_intervals differ from the run's interval grid")
    return frame["rrp_aud_mwh"].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- recomputation


@dataclass(slots=True)
class Recomputed:
    ends: pd.DatetimeIndex
    price: np.ndarray
    tiers: dict[str, np.ndarray]
    pvalues: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    in_period: dict[str, np.ndarray]
    events: list[Event]


def recompute(
    run: StoredRun, price: np.ndarray, cfg: AppConfig, split: Split | None = None
) -> Recomputed:
    """Tiers, lines and events from the stored p-values and prices; must equal the run.

    ``split`` defaults to the development split (tests pass their own)."""
    a = cfg.anomalies
    f = run.intervals
    ends = pd.DatetimeIndex(pd.to_datetime(f["interval_end"], utc=True))
    problems = []
    tiers: dict[str, np.ndarray] = {}
    pvalues: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    in_period: dict[str, np.ndarray] = {}
    for d in DETECTORS:
        p = f[f"{d}__p"].to_numpy(dtype=np.float64)
        period = f[f"{d}__in_period"].to_numpy(dtype=bool)
        t = tiers_from_p(p, a.pvalues.notable, a.pvalues.severe)
        t[~period] = 0
        if not np.array_equal(t, f[f"{d}__tier"].to_numpy(dtype=np.int8)):
            problems.append(f"{d} tiers")
        tiers[d], pvalues[d], in_period[d] = t, p, period
        scores[d] = f[f"{d}__score"].to_numpy(dtype=np.float64)
    for d, line, level in (
        (LINE_NOTABLE, a.price_lines.notable_aud, 1),
        (LINE_SEVERE, a.price_lines.severe_aud, 2),
    ):
        period = f[f"{d}__in_period"].to_numpy(dtype=bool)
        with np.errstate(invalid="ignore"):
            t = np.where(period & (price >= line), level, 0).astype(np.int8)
        if not np.array_equal(t, f[f"{d}__tier"].to_numpy(dtype=np.int8)):
            problems.append(f"{d} tiers")
        tiers[d], in_period[d] = t, period
    split = split or dev_split(cfg)
    start, stop = split.index_range(ends)
    events = merge_events(tiers, pvalues, scores, price, start, stop, gap_intervals(a))
    stored = run.events
    if len(events) != len(stored):
        problems.append(f"{len(events)} events recomputed, {len(stored)} stored")
    else:
        for e, (_, row) in zip(events, stored.iterrows(), strict=True):
            same = (
                ends[e.first] == row.first_interval_end
                and ends[e.last] == row.last_interval_end
                and ends[e.peak] == row.peak_interval_end
                and tuple(row.detectors) == e.detectors
                and e.tier == row.tier
                and ((e.min_p is None and pd.isna(row.min_p)) or e.min_p == row.min_p)
            )
            if not same:
                problems.append(f"event at {row.first_interval_end} differs")
                break
    counts = run.config["counts"]
    for d, t in tiers.items():
        if int((t > 0).sum()) != int(counts["alerts"][d]):
            problems.append(
                f"{d}: {int((t > 0).sum())} alerts, the run recorded {counts['alerts'][d]}"
            )
    if counts["events"] != len(stored):
        problems.append(f"the run recorded {counts['events']} events, {len(stored)} are stored")
    if problems:
        raise ReportMismatch("; ".join(problems))
    return Recomputed(ends, price, tiers, pvalues, scores, in_period, events)


# --------------------------------------------------------------------------- A1


def a1_rows(rc: Recomputed) -> pd.DataFrame:
    """Per detector: scored intervals and the shares at p ≤ 0.01, 0.001 and 0.0001."""
    rows = []
    for d in DETECTORS:
        p = rc.pvalues[d]
        scored = rc.in_period[d] & np.isfinite(p)
        ps = p[scored]
        rows.append(
            {
                "detector": d,
                "scored": int(scored.sum()),
                "p01": float(np.mean(ps <= 0.01)) if ps.size else np.nan,
                "p001": float(np.mean(ps <= 0.001)) if ps.size else np.nan,
                "p0001": float(np.mean(ps <= 0.0001)) if ps.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def a1_passed(row: pd.Series) -> tuple[bool, bool]:
    (_, lo1, hi1), (_, lo2, hi2) = A1_BANDS
    return bool(lo1 <= row["p01"] <= hi1), bool(lo2 <= row["p001"] <= hi2)


def rates_by(rc: Recomputed, freq: str) -> pd.DataFrame:
    """Share of scored intervals with p ≤ 0.001 per NEM-time period (``"Y"`` or ``"M"``)."""
    nem = (rc.ends - INTERVAL).tz_convert(NEM_TZ).tz_localize(None)
    key = nem.to_period(freq)
    out = {}
    for d in DETECTORS:
        p = rc.pvalues[d]
        scored = rc.in_period[d] & np.isfinite(p)
        s = pd.Series((p <= 0.001)[scored], index=key[scored])
        out[d] = s.groupby(level=0).mean()
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- A2


@dataclass(frozen=True, slots=True)
class ReferenceCheck:
    event: ReferenceEvent
    fired: tuple[tuple[str, float], ...]  # calibrated detectors with a notable alert, lowest p
    lowest: tuple[str, float] | None  # reported only: lowest p of any calibrated detector

    @property
    def caught(self) -> bool:
        return bool(self.fired)


def reference_checks(
    refs: Sequence[ReferenceEvent], rc: Recomputed, margin_minutes: int
) -> list[ReferenceCheck]:
    margin = pd.Timedelta(minutes=margin_minutes)
    out = []
    for ref in refs:
        inside = np.asarray((rc.ends >= ref.start - margin) & (rc.ends <= ref.end + margin))
        fired = []
        lowest: tuple[str, float] | None = None
        for d in DETECTORS:
            hit = inside & (rc.tiers[d] >= 1)
            if hit.any():
                fired.append((d, float(np.min(rc.pvalues[d][hit]))))
            near = inside & rc.in_period[d] & np.isfinite(rc.pvalues[d])
            if near.any():
                p = float(np.min(rc.pvalues[d][near]))
                if lowest is None or p < lowest[1]:
                    lowest = (d, p)
        out.append(ReferenceCheck(ref, tuple(fired), lowest))
    return out


# --------------------------------------------------------------------------- A3


def power_table(inj: pd.DataFrame) -> pd.DataFrame:
    """Detected share and number of injections per (label, size)."""
    g = inj.groupby(["label", "detector", "size", "reported_only"], sort=False)["detected"]
    return g.agg(n="size", power="mean").reset_index()


def a3_rows(power: pd.DataFrame) -> list[tuple[str, float, int, float]]:
    rows = []
    for d, size in A3_CHECKS:
        sel = power[(power["label"] == d) & (power["size"] == size)]
        if sel.empty:
            rows.append((d, size, 0, np.nan))
        else:
            rows.append((d, size, int(sel["n"].iloc[0]), float(sel["power"].iloc[0])))
    return rows


def smallest_reaching(power: pd.DataFrame, label: str, sign: int, level: float) -> float | None:
    sel = power[(power["label"] == label) & (np.sign(power["size"]) == sign)]
    sel = sel.assign(mag=sel["size"].abs()).sort_values("mag")
    hit = sel[sel["power"] >= level]
    return None if hit.empty else float(hit["size"].iloc[0])


# --------------------------------------------------------------------------- A4, A5


def a4_counts(events: pd.DataFrame, phrases: Sequence[str]) -> tuple[int, int, list[str]]:
    incomplete = [missing_fields(ev) for ev in events["evidence"]]
    banned = [banned_in(s, phrases) for s in events["summary"]]
    found = sorted({p for b in banned for p in b})
    return sum(bool(m) for m in incomplete), sum(bool(b) for b in banned), found


@dataclass(frozen=True, slots=True)
class CausalityOutcome:
    passed: int
    failed: int
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.failed == 0 and self.passed > 0


def run_causality_tests() -> CausalityOutcome:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *CAUSALITY_TESTS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed, failed = parse_pytest_summary(proc.stdout)
    return CausalityOutcome(passed, failed, proc.returncode)


def forest_leaks(forests: pd.DataFrame) -> list[str]:
    late = pd.to_datetime(forests["train_last_end"], utc=True) > pd.to_datetime(
        forests["month_start"], utc=True
    )
    return [str(m) for m in forests.loc[late, "month"]]


# --------------------------------------------------------------------------- formatting


def pct(v: float, digits: int = 2) -> str:
    return "–" if pd.isna(v) else f"{100 * v:.{digits}f}%"


SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pval(v: float | None) -> str:
    """e.g. ``1.91 × 10⁻⁵``."""
    if v is None or pd.isna(v):
        return "–"
    mantissa, exponent = f"{v:.2e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


def size_label(detector: str, size: float) -> str:
    if detector in (PRICE_LEVEL, PRICE_VOLATILITY):
        return money(size)
    return f"{size:+,.0f} MW".replace("+", "+").replace("-", "−")


def local(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M")


def duration_minutes(events: pd.DataFrame) -> pd.Series:
    span = events["last_interval_end"] - events["first_interval_end"] + INTERVAL
    return pd.Series(pd.to_timedelta(span).to_numpy() / pd.Timedelta(minutes=1), index=events.index)


# --------------------------------------------------------------------------- figures


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def plot_power(power: pd.DataFrame, path: Path) -> None:
    """Small multiples: one detector per panel (sizes are in different units)."""
    fig = Figure(figsize=(10, 6.2), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1, hspace=0.55, wspace=0.28)
    fig.suptitle(
        "Detection power by injection size (development, 1,000 shared starts per detector)",
        color=INK,
        fontsize=11,
        x=0.07,
        ha="left",
    )
    for ax, d in zip(axes.ravel(), DETECTORS, strict=True):
        sel = power[(power["label"] == d) & ~power["reported_only"]].sort_values("size")
        x = np.arange(len(sel))
        ax.axhline(100 * A3_POWER, color=AXIS, linewidth=1, zorder=1)
        ax.plot(
            x,
            100 * sel["power"],
            color=COLORS[d],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.set_xticks(x, [size_label(d, s) for s in sel["size"]], rotation=45, ha="right")
        ax.set_ylim(-3, 103)
        _style(ax, NAMES[d], UNITS[d], "detected, %")
    axes[0, 0].annotate(
        "90% (A3)", xy=(0, 90), xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=7
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def plot_monthly_rates(monthly: pd.DataFrame, path: Path) -> None:
    """Small multiples: each detector's monthly share at p ≤ 0.001, with A1's band."""
    fig = Figure(figsize=(10, 6.2), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(2, 3, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.1, hspace=0.4, wspace=0.28)
    fig.suptitle(
        "Share of scored intervals with p ≤ 0.001, by NEM month (A1's band 0.05–0.2% shaded)",
        color=INK,
        fontsize=11,
        x=0.07,
        ha="left",
    )
    x = pd.PeriodIndex(monthly.index).to_timestamp()
    for ax, d in zip(axes.ravel(), DETECTORS, strict=True):
        y = 100 * monthly[d]
        ax.axhspan(0.05, 0.2, color="#f0efec", linewidth=0, zorder=0)
        ax.axhline(0.1, color=AXIS, linewidth=1, zorder=1)
        ax.plot(x, y, color=COLORS[d], linewidth=2, zorder=3)
        top = float(np.nanmax(y)) if np.isfinite(y).any() else 0.3
        ax.set_ylim(0, max(0.3, 1.08 * top))
        _style(ax, NAMES[d], "", "% of intervals")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


# --------------------------------------------------------------------------- the report


@dataclass(slots=True)
class Results:
    run: StoredRun
    rc: Recomputed
    a1: pd.DataFrame
    yearly: pd.DataFrame
    monthly: pd.DataFrame
    refs: list[ReferenceCheck]
    refs_o: list[ReferenceCheck]
    power: pd.DataFrame
    a3: list[tuple[str, float, int, float]]
    a4: tuple[int, int, list[str]]
    causality: CausalityOutcome
    cut_differences: int
    forest_leaks: list[str]


def compute(
    engine: Engine, cfg: AppConfig, region_id: str, causality: CausalityOutcome | None = None
) -> Results:
    run = load_run(engine, cfg, region_id)
    ends = pd.DatetimeIndex(pd.to_datetime(run.intervals["interval_end"], utc=True))
    rc = recompute(run, load_prices(engine, region_id, ends), cfg)
    market = load_market_frame(engine, region_id)
    market = market[market["interval_end"] <= ends[-1]]
    refs = [r for r in reference_events(market, cfg) if r.period == "development"]
    margin = cfg.anomalies.reference_events.alert_margin_minutes
    power = power_table(run.injections)
    leaks = forest_leaks(run.forests)
    stored_leaks = run.config["a5_real"]["forest_leaks"]
    if sorted(leaks) != sorted(stored_leaks):
        raise ReportMismatch(f"forest leaks {leaks} recomputed, {stored_leaks} recorded")
    return Results(
        run=run,
        rc=rc,
        a1=a1_rows(rc),
        yearly=rates_by(rc, "Y"),
        monthly=rates_by(rc, "M"),
        refs=reference_checks([r for r in refs if r.cls == CLASS_P], rc, margin),
        refs_o=reference_checks([r for r in refs if r.cls == CLASS_O], rc, margin),
        power=power,
        a3=a3_rows(power),
        a4=a4_counts(run.events, cfg.anomalies.evidence.banned_phrases),
        causality=causality or run_causality_tests(),
        cut_differences=int(run.cuts["n_differences"].sum()),
        forest_leaks=leaks,
    )


def criteria(res: Results) -> list[Criterion]:
    a1_checks = [a1_passed(r) for _, r in res.a1.iterrows()]
    n_a1 = sum(x for pair in a1_checks for x in pair)
    worst = [
        f"{SHORT[r['detector']]} {pct(r['p01'])} / {pct(r['p001'], 3)}"
        for (_, r), (ok1, ok2) in zip(res.a1.iterrows(), a1_checks, strict=True)
        if not (ok1 and ok2)
    ]
    judged = res.refs
    caught = sum(c.caught for c in judged)
    missed = [c.event.event_id for c in judged if not c.caught]
    a3_ok = [not pd.isna(p) and p >= A3_POWER for *_, p in res.a3]
    a3_failed = [
        f"{SHORT[d]} {size_label(d, s)} {pct(p, 1)}"
        for (d, s, _, p), ok in zip(res.a3, a3_ok, strict=True)
        if not ok
    ]
    incomplete, banned, phrases = res.a4
    n_events = len(res.run.events)
    cuts = res.run.cuts
    gate = res.run.config["shap_gate"]
    return [
        Criterion(
            "A1",
            "every calibrated detector: share at p ≤ 0.01 in [0.5%, 2%] and at p ≤ 0.001 in "
            "[0.05%, 0.2%] (12 checks)",
            f"{n_a1} of 12 checks inside"
            + (f"; outside: {', '.join(worst)} (p ≤ 0.01 / p ≤ 0.001)" if worst else ""),
            n_a1 == 12,
        ),
        Criterion(
            "A2",
            "every scorable class-P reference event has a notable calibrated alert within 30 min "
            "of its span",
            f"{caught} of {len(judged)} caught"
            + (f"; missed {', '.join(missed)}" if missed else ""),
            bool(judged) and caught == len(judged),
        ),
        Criterion(
            "A3",
            "≥ 90% of injections detected at each detector's largest size, per sign (11 checks)",
            f"{sum(a3_ok)} of 11 at ≥ 90%"
            + (f"; below: {', '.join(a3_failed)}" if a3_failed else ""),
            all(a3_ok),
        ),
        Criterion(
            "A4",
            "every development event has every evidence field filled or marked unavailable with "
            "a reason, and no summary has a banned phrase",
            f"{n_events - incomplete} of {n_events} complete; {banned} with a banned phrase"
            + (f" ({', '.join(phrases)})" if phrases else ""),
            incomplete == 0 and banned == 0,
        ),
        Criterion(
            "A5",
            "the synthetic causality suite passes (controls included), every real cut is "
            "identical, and no forest was fitted on its own month",
            f"suite {res.causality.passed} passed, {res.causality.failed} failed; "
            f"{len(cuts)} real cuts, {res.cut_differences} differences; "
            f"{len(res.forest_leaks)} forest leaks",
            res.causality.ok
            and res.cut_differences == 0
            and not res.forest_leaks
            and len(cuts) > 0,
        ),
        Criterion(
            "E1 (demand peaks)",
            "the recomputed forecast equals the stored float32 value and base + Σ SHAP equals "
            "the raw prediction within 10⁻⁶, for every explained pair",
            f"{gate['n']:,} pairs; {gate['forecast_mismatches']} forecast mismatches, "
            f"{gate['sum_failures']} sum failures (largest error {gate['max_sum_error']:.1e})",
            bool(gate["passed"]),
        ),
    ]


def _a1_table(res: Results) -> str:
    rows = []
    for _, r in res.a1.iterrows():
        ok1, ok2 = a1_passed(r)
        rows.append(
            [
                NAMES[r["detector"]],
                f"{r['scored']:,}",
                pct(r["p01"]),
                "in" if ok1 else "**out**",
                pct(r["p001"], 3),
                "in" if ok2 else "**out**",
                pct(r["p0001"], 4),
            ]
        )
    return _md_table(
        [
            "Detector",
            "Scored intervals",
            "p ≤ 0.01",
            "[0.5%, 2%]",
            "p ≤ 0.001",
            "[0.05%, 0.2%]",
            "p ≤ 0.0001 (reported)",
        ],
        rows,
        "lrrcrcr",
    )


def _yearly_table(res: Results) -> str:
    y = res.yearly
    rows = [[str(p)] + [pct(y.loc[p, d], 3) for d in DETECTORS] for p in y.index]
    return _md_table(["NEM year", *(SHORT[d] for d in DETECTORS)], rows)


def _reference_table(checks: Sequence[ReferenceCheck], tz: str) -> str:
    rows = []
    for c in checks:
        e = c.event
        fired = ", ".join(f"{SHORT[d]} ({pval(p)})" for d, p in c.fired) or "none"
        rows.append(
            [
                e.event_id,
                f"{local(e.start, tz)} → {local(e.end, tz)}",
                f"{e.n_intervals:,}",
                f"{money(e.price_min)} to {money(e.price_max)}",
                "yes" if c.caught else "**no**",
                fired,
                "–" if c.lowest is None else f"{SHORT[c.lowest[0]]} {pval(c.lowest[1])}",
            ]
        )
    return _md_table(
        [
            "Event",
            "Span (local)",
            "Flagged intervals",
            "Price",
            "Caught",
            "Notable calibrated detectors (lowest p)",
            "Lowest p of any calibrated detector (reported)",
        ],
        rows,
        "llrlcll",
    )


def _power_table(res: Results) -> str:
    rows = []
    for d in DETECTORS:
        sel = res.power[(res.power["label"] == d)].sort_values("size")
        cells = ", ".join(
            f"{size_label(d, s)}: {pct(p, 1)}"
            for s, p in zip(sel["size"], sel["power"], strict=True)
        )
        n = int(sel["n"].iloc[0]) if len(sel) else 0
        rows.append([NAMES[d], f"{n:,}", cells])
    return _md_table(["Detector", "Injections per size", "Detected share by size"], rows, "lrl")


def _a3_table(res: Results) -> str:
    rows = [
        [
            NAMES[d],
            size_label(d, s),
            f"{n:,}",
            pct(p, 1),
            "pass" if (not pd.isna(p) and p >= A3_POWER) else "**fail**",
        ]
        for d, s, n, p in res.a3
    ]
    return _md_table(["Detector", "Largest size", "Injections", "Detected", "≥ 90%"], rows, "llrrc")


def _thresholds_table(res: Results) -> str:
    rows = []
    for d in DETECTORS:
        cells = []
        for sign in (-1, 1):
            if not (res.power["label"].eq(d) & (np.sign(res.power["size"]) == sign)).any():
                cells += ["–", "–"]
                continue
            for level in (0.5, 0.9):
                s = smallest_reaching(res.power, d, sign, level)
                cells.append("not reached" if s is None else size_label(d, s))
        rows.append([NAMES[d], *cells])
    return _md_table(
        ["Detector", "50%, negative", "90%, negative", "50%, positive", "90%, positive"],
        rows,
        "lllll",
    )


def _power_by_year(res: Results) -> str:
    inj = res.run.injections
    inj = inj.assign(year=pd.to_datetime(inj["start"], utc=True).dt.tz_convert(NEM_TZ).dt.year)
    rows = []
    for d, s, _, _ in res.a3:
        sel = inj[(inj["label"] == d) & (inj["size"] == s)]
        by = sel.groupby("year")["detected"].agg(["mean", "size"])
        rows.append(
            [
                NAMES[d],
                size_label(d, s),
                "; ".join(f"{y}: {pct(m, 0)} (n {int(n)})" for y, (m, n) in by.iterrows()),
            ]
        )
    return _md_table(["Detector", "Size", "Detected by start year (NEM time)"], rows, "lll")


def _catalogue(res: Results) -> tuple[str, str]:
    ev = res.run.events
    calibrated = ev[~ev["lines_only"]]
    lines = ev[ev["lines_only"]]
    dur = duration_minutes(ev)

    def row(name: str, sel: pd.DataFrame) -> list[str]:
        d = dur[sel.index]
        return [
            name,
            f"{len(sel):,}",
            f"{int((sel['tier'] == 'severe').sum()):,}",
            f"{d.median():.0f}" if len(sel) else "–",
            f"{d.quantile(0.9):.0f}" if len(sel) else "–",
            f"{d.max():,.0f}" if len(sel) else "–",
        ]

    table = _md_table(
        [
            "Events",
            "Count",
            "Severe",
            "Median duration, min",
            "90th percentile, min",
            "Longest, min",
        ],
        [
            row("from calibrated detectors", calibrated),
            row("price lines only", lines),
            row("all", ev),
        ],
    )
    cat_rows = []
    for c in CATEGORIES:
        has = calibrated["categories"].apply(lambda cs, c=c: c in cs)
        cat_rows.append([c, f"{int(has.sum()):,}"])
    years = (
        calibrated["first_interval_end"].dt.tz_convert(NEM_TZ).dt.year.value_counts().sort_index()
    )
    cats = _md_table(
        ["Category (calibrated events; an event can hold several)", "Events"], cat_rows
    )
    cats += (
        "\n\nCalibrated events by NEM year: "
        + ", ".join(f"{y} {n:,}" for y, n in years.items())
        + "."
    )
    return table, cats


def _overlap(res: Results) -> str:
    ev = res.run.events[~res.run.events["lines_only"]]
    sets = [set(d) for d in ev["detectors"]]
    rows = []
    for a in DETECTORS:
        cells = [f"{sum(a in s and b in s for s in sets):,}" for b in DETECTORS]
        rows.append([SHORT[a], *cells])
    return _md_table(["", *(SHORT[d] for d in DETECTORS)], rows)


def _case_studies(res: Results, cfg: AppConfig, tz: str) -> str:
    ev = res.run.events
    picks = case_studies(res.rc.events, res.rc.ends, cfg.anomalies, named=True)
    parts = []
    for k, pick in enumerate(picks, start=1):
        if pick.event is None:
            parts.append(f"**{k}. {pick.reason}:** no calibrated event touched this day.")
            continue
        row = ev.iloc[pick.event]
        e = res.rc.events[pick.event]
        named = (
            f" (also the named day {', '.join(d.isoformat() for d in pick.also_named)})"
            if pick.also_named
            else ""
        )
        dets = ", ".join(
            f"{SHORT[d]}" + (f" (p {pval(p)})" if p is not None else "")
            for d, p in zip(e.detectors, e.detector_min_p, strict=True)
        )
        parts.append(
            f"**{k}. {pick.reason}{named}:** {local(row.first_interval_end - INTERVAL, tz)} → "
            f"{local(row.last_interval_end, tz)} local, peak interval ending "
            f"{local(row.peak_interval_end, tz)}; "
            f"{row.tier}; {', '.join(row.categories)}; detectors {dets}.\n\n> {row.summary}"
        )
    return "\n\n".join(parts)


def build_report(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    causality: CausalityOutcome | None = None,
    figures: Path | None = None,
) -> str:
    res = compute(engine, cfg, region_id, causality)
    figures = figures or cfg.resolve(cfg.paths.reports_dir) / "figures"
    plot_power(res.power, figures / "m4_power_by_size.png")
    plot_monthly_rates(res.monthly, figures / "m4_alert_rate_by_month.png")
    tz = cfg.region(region_id).local_tz
    c = res.run.config
    crit = criteria(res)
    state = freeze_state(cfg)
    flat = res.power[res.power["reported_only"]]
    flat_line = (
        f"{pct(float(flat['power'].iloc[0]), 1)} of {int(flat['n'].iloc[0]):,}"
        if len(flat)
        else "not run"
    )
    o_caught = sum(x.caught for x in res.refs_o)
    lines_only = res.run.events[res.run.events["lines_only"]]
    l2_events = int(lines_only["detectors"].apply(lambda d: LINE_SEVERE in d).sum())
    catalogue, categories = _catalogue(res)
    report_sha = git_sha()
    sections = [
        "# M4: market anomalies in development (Victoria)\n",
        "Generated by `gridcast report-m4` from the stored development run. Every number below "
        "is recomputed from the stored files and events; the report stops if a recomputed tier, "
        "line, event or count differs from the run's record. The criteria were pre-registered in "
        "[`docs/plans/m4_plan.md`](../docs/plans/m4_plan.md) §3, and every detector setting was "
        "fixed in §2 before any detector code or score existed. The recipe was frozen before "
        "this run, so these verdicts are blind to tuning.\n",
        "## Verdicts (development)\n",
        criteria_table(crit) + "\n",
        f"**{sum(x.passed for x in crit)} of {len(crit)} pass.** Failures are published as "
        "measured; nothing in §2 was changed after this run. A disclosed, re-hashed v2 may "
        "follow (plan §6).\n",
        "## Run\n",
        _md_table(
            ["", ""],
            [
                ["Run", f"{res.run.run_id} (batch `{res.run.batch_id}`)"],
                [
                    "Anomaly config hash",
                    f"`{c['anomaly_config_hash']}` (frozen `{cfg.anomalies.frozen_config_hash}`)",
                ],
                [
                    "Frozen recipe code",
                    f"`{c['frozen_commit']}`; run from `{(res.run.git_sha or '')[:7]}`",
                ],
                [
                    "Series",
                    f"{c['series']['intervals']:,} intervals, {c['series']['first_end']} → "
                    f"{c['series']['last_end']} (UTC interval ends)",
                ],
                [
                    "Scored",
                    f"D1, D4–D6 and lines from {c['periods']['market_first_day']}; D2–D3 "
                    f"from {c['periods']['residual_first_day']}; to "
                    f"{c['periods']['end_day']} (exclusive, NEM days)",
                ],
                [
                    "D2/D3 band",
                    f"QR batch `{c['inputs']['qr_batch'][:8]}` with run 2's mask: "
                    f"{c['inputs']['band_issue_times']:,} issue times; actuals equal market "
                    f"demand on {c['inputs']['checked_actuals']:,} pairs",
                ],
                [
                    "Run time",
                    ", ".join(
                        f"{k} {c['seconds'][k]:,.0f} s"
                        for k in ("load", "detect", "inject", "cuts", "evidence")
                        if k in c["seconds"]
                    ),
                ],
                ["Freeze now", "holds" if not state.problems else "; ".join(state.problems)],
                ["Report code", f"`{report_sha or 'unknown'}`"],
            ],
            "ll",
        )
        + "\n",
        "## A1: alert budget\n",
        "Each detector's p-value is a conformal rank against its own trailing 26 weeks, so under "
        "exchangeability p ≤ 0.001 should flag about 0.1% of scored intervals. A1 allows a factor "
        "of 2 either way. *Scored* means inside the detector's scored period with a p-value "
        "(D2/D3: pairs in M2's common mask; D6: a month model).\n",
        _a1_table(res) + "\n",
        "Share at p ≤ 0.001 by NEM year (reported only):\n",
        _yearly_table(res) + "\n",
        "![Monthly alert rates](figures/m4_alert_rate_by_month.png)\n",
        "## A2: AEMO-flagged reference events\n",
        "Class-P events (price at a cap, floor or administered limit by AEMO's own flags, which "
        "are never detector inputs), from "
        "[`docs/m4_reference_events.md`](../docs/m4_reference_events.md). "
        "An event is caught if any calibrated detector (price lines excluded) raised a notable "
        "alert at an interval ending within 30 min of its span.\n",
        _reference_table(res.refs, tz) + "\n",
        f"Class O (reported only; such flags may leave no visible trace): {o_caught} of "
        f"{len(res.refs_o)} development events had a calibrated alert in the same window.\n",
        _reference_table(res.refs_o, tz) + "\n",
        "## A3: detection power\n",
        "Each injection is scored alone against the unmodified profiles, baselines, bands, forests "
        "and windows, at starts where the unmodified detector raised no notable alert within the "
        "span ± 1 h. Detected = at least one notable alert from that detector inside the span.\n",
        _a3_table(res) + "\n",
        "All sizes (reported):\n",
        _power_table(res) + "\n",
        "Smallest size reaching 50% and 90% (reported):\n",
        _thresholds_table(res) + "\n",
        "Power at the largest size by start year (reported):\n",
        _power_by_year(res) + "\n",
        "Reported only: holding the price flat for 3 h (D5's calm side) was detected in "
        f"{flat_line}.\n",
        "![Power by size](figures/m4_power_by_size.png)\n",
        "## A4: evidence without causal claims\n",
        f"{len(res.run.events):,} development events; {res.a4[0]} incomplete, {res.a4[1]} with a "
        "banned phrase. Each evidence field is a fact with its source, labelled *known at the "
        "time* or *retrospective*, or marked unavailable with a reason.\n",
        "## A5: causality\n",
        f"- Synthetic suite (`{CAUSALITY_TESTS[0]}`, run by this report): "
        f"{res.causality.passed} passed, {res.causality.failed} failed. It includes a leaky "
        "control, which must be caught.\n"
        f"- Real data: the run repeated detection on the series truncated at "
        f"{len(res.run.cuts)} seeded cuts ("
        + ", ".join(str(t) for t in pd.to_datetime(res.run.cuts["cut_end"], utc=True))
        + f"); {res.cut_differences} differences at or before the cuts.\n"
        f"- Forests: {len(res.run.forests)} monthly models; {len(res.forest_leaks)} saw an "
        "interval ending after their month's start.\n",
        "## Event catalogue (reported)\n",
        catalogue + "\n",
        categories + "\n",
        f"Price-line-only events: {len(lines_only):,}, of which {l2_events:,} reached $5,000/MWh. "
        "They are counted separately (§2.4).\n",
        "Calibrated events in which both detectors fired (diagonal: events with that detector):\n",
        _overlap(res) + "\n",
        "## Case studies\n",
        f"Chosen by the fixed rule (§3.2 and the step-4 tie rule): the {cfg.anomalies.cases.n} "
        "events with the lowest p from a calibrated detector, ties by more calibrated detectors "
        "firing, then a seeded draw; then the named days. Summaries are generated from the "
        "evidence by fixed templates and state facts only.\n",
        _case_studies(res, cfg, tz) + "\n",
        "## Disclosures\n",
        "- The detectors were designed after M2's and M3's development and holdout results had "
        "been seen. The planning saw some development facts (the flagged days, the largest "
        "5-minute demand changes) and some holdout facts (plan §3.3).\n"
        "- The warm-up smoke run (before any development score) previewed likely failures at "
        "the largest sizes for D1, D3, D4 and D6 and a low D1 alert rate; the user chose to keep "
        "the pre-registered recipe (plan §1).\n"
        "- How this report reads each criterion was fixed in the plan while the development run "
        "was in progress, before any of its output was read (plan §1, step 4).\n"
        "- After the first generation of this report, only its presentation changed (p-value "
        "formatting, local start times in the case studies, the order of the run times, and the "
        "reported-only column of each reference event's lowest p). No criterion's computation "
        "changed.\n"
        "- Price associations, SHAP summaries and the holdout come in steps 5 and 6.\n",
    ]
    return "\n".join(sections)


def write_report(
    engine: Engine, cfg: AppConfig, region_id: str, causality: CausalityOutcome | None = None
) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id, causality), encoding="utf-8")
    return path
